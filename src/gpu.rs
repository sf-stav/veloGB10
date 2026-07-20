//! Full-GPU forward for Qwen3.5-0.8B (f32). cuBLAS gemv for linears + custom CUDA kernels
//! for elementwise/recurrent ops. Validated against the same reference as qwen.rs.

use std::sync::Arc;
use std::collections::HashMap;

use cudarc::cublas::{sys::cublasOperation_t as OP, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaFunction, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

use crate::qwen::{Config, LayerType};

pub type S = cudarc::driver::CudaSlice<f32>;
pub type B = cudarc::driver::CudaSlice<half::bf16>; // big matmul weights in bf16

pub struct GpuModel {
    dev: Arc<CudaDevice>,
    blas: CudaBlas,
    stream: cudarc::driver::CudaStream,
    cfg: Config,
    embed: W,
    lm_head: Option<W>,  // None = tied (use embed)
    final_norm: S,
    layers: Vec<GpuLayer>,
    mtp: Option<GpuMtpLayer>,
    k: KernelTable,
    bk: HashMap<String, CudaFunction>,
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
    sv_p:  cudarc::driver::CudaSlice<f32>,  // [MAX_VERIFY] out: p_of_draft
    sv_r:  cudarc::driver::CudaSlice<i32>,  // [MAX_VERIFY] out: resid_tok (bonus at depth-1)
    mr_tok: cudarc::driver::CudaSlice<i32>, // [MAX_VERIFY] batched re-prime tokens
    mr_pos: cudarc::driver::CudaSlice<i32>, // [MAX_VERIFY] batched re-prime positions
    /// bf16 scratch for dequantize-then-cuBLAS on the PREFILL path. Grown to the largest tensor.
    deq_scratch: std::sync::Mutex<Option<(B, usize)>>,
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
    /// Proof D per-head visit counters (GB10_TP_HEAD_PROOF only; None otherwise).
    head_visits: Option<cudarc::driver::CudaSlice<u64>>,
}

/// Widest verify the persistent stochastic scratch is sized for (matches GEMM_BINV_NMAX).
pub const MAX_VERIFY: usize = 16;
/// Query positions per gqa_attn_prefill block (== warps per block). The 8 warps sweep the SAME keys,
/// so one K/V fetch from L2 serves 8 queries. MUST match `QT` in the kernel.
pub const GQA_PF_QT: usize = 8;
/// Query / key tile for the tiled (cuBLAS tensor-core) prefill attention. Sized so S stays a few MB
/// per kv head and the GEMMs stay big enough that launch overhead is noise.
/// Force the scalar `gqa_attn_prefill` instead of the tiled tensor-core path.
///
/// The scalar kernel is the ORACLE. It is slow but it is the reference that the tiled path — and any
/// future mma kernel — must agree with, and it stays reachable on purpose: this kernel has already
/// produced two confident wrong diagnoses, so "run it the simple way and diff" must always be one env
/// var away. `RUST_INFER_PREFILL_SCALAR=1`.
fn prefill_scalar() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUST_INFER_PREFILL_SCALAR").is_ok())
}

/// RUST_INFER_ZERO_KV=1 — restore the (default-off) full-KV-cache memset on cold admits. Off is the
/// production behavior: attention never reads KV beyond pos, so zeroing it was dead TTFT work.
fn zero_kv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUST_INFER_ZERO_KV").is_ok())
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

/// How many of the most-frequent token ids the DRAFT head keeps (FR-Spec). Measured on prose+code,
/// the top 65536 ids of this BPE vocabulary (26%) cover 97.5% of emitted tokens -- and the tokens they
/// miss are rare AND hard to predict, i.e. exactly the ones the drafter was already getting wrong.
/// `RUST_INFER_DRAFT_VOCAB=0` turns it off.
pub const DRAFT_VOCAB_TOP: usize = 65536;

/// Depths the auto-policy calibrates and may choose between. Capped at PROFILE_MAX_N because the
/// calibration probe runs a real verify at each of these widths against a `2 + PROFILE_MAX_N`-slot
/// state. Acceptance decays geometrically while a flat verify makes deeper drafts nearly free, so the
/// optimum sits in the middle of this range — it is not a corner.
pub const AUTO_DEPTHS: &[usize] = &[2, 3, 4, 5, 6, 8];
/// Deepest depth the scheduler must be able to serve; sizes the per-lane checkpoint slots.
pub const MAX_AUTO_DEPTH: usize = 8;

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
            W::Nvfp4 { m, k, .. } => m * k / 2 + m * k / 16 + m / 4,
            W::Nvfp4Raw { m, k, .. } => m * k / 2 + m * k / 16,
        }
    }
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
}

pub struct GpuMlp { pub gate: W, pub up: W, pub down: W }

/// MoE FFN (qwen3_5_moe). Experts are STACKED, gate+up FUSED, exactly as the checkpoint stores them:
///   `gate_up` = `experts.gate_up_proj` [num_experts, 2*moe_inter, hidden]  (flattened to one W)
///   `down`    = `experts.down_proj`    [num_experts, hidden, moe_inter]    (flattened to one W)
///   `router`  = `mlp.gate.weight`      [num_experts, hidden]
///   `shared`  = a standard MLP (shared_expert), `shared_gate` = [1, hidden] sigmoid gate.
/// Per-expert dims come from `cfg` (num_experts / num_experts_per_tok / moe_intermediate_size) at
/// forward time. Forward: softmax→top-k→renorm the k weights, sum the k experts, + sigmoid(shared_gate·h)·shared(h).
pub struct GpuMoe {
    pub router: W,
    pub gate_up: W,
    pub down: W,
    pub shared: GpuMlp,
    pub shared_gate: W,
    /// TP=2 expert-parallel: set by `tp_shard_weights` when THIS layer's `gate_up`/`down` were halved
    /// along the stacked-expert dim (this rank owns experts [rank·ne/2, rank·ne/2 + ne/2)). Read by
    /// `moe_batch` to (a) remap the router's global expert ids to local ones and zero remote slots,
    /// and (b) all-reduce the expert-combined output before the (replicated) shared-expert add.
    /// A property of the TENSOR, not of the process (see `ffn_batch`): the MTP head's MoE keeps
    /// `false` even under TP=2 — the draft head must stay barrier-free.
    pub experts_sharded: bool,
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

pub struct GpuFullAttn { pub qkv: AttnIn, pub o_proj: W, pub q_norm: S, pub k_norm: S }
pub struct GpuLinearAttn {
    pub in_proj: GdnIn, pub conv1d: S,
    pub a_log: S, pub dt_bias: S, pub norm: S, pub out_proj: W,
}

impl AttnIn {
    /// Rows of the fused attention weight: [q|gate ; k ; v].
    pub fn fused_m(cfg: &Config) -> usize { cfg.num_heads * cfg.head_dim * 2 + 2 * cfg.num_kv_heads * cfg.head_dim }
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
}
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

/// Create the engine's compute stream as a BLOCKING stream (CU_STREAM_DEFAULT) rather than a
/// NonBlocking fork. This is the crux of MTP correctness on GB10.
///
/// cudarc 0.9.15 runs ALL of its memory operations on the device's legacy NULL stream (handle 0):
/// `alloc_zeros`/`memset_zeros` (cuMemAllocAsync/cuMemsetAsync), `htod_*`/`dtoh_*`, and
/// `CudaSlice::drop` → `cuMemFreeAsync`. Our kernels + cuBLAS + dtod copies run on this compute
/// stream. `fork_default_stream()` makes a NonBlocking stream, which by definition performs NO
/// implicit synchronization with the NULL stream — so a buffer produced by `cuMemAllocAsync` on the
/// NULL stream can be touched by a kernel on the compute stream *before the allocation is ordered*
/// (compute-sanitizer: "accessed before it is allocated"), and a `cuMemFreeAsync` on drop can recycle
/// memory a still-in-flight compute kernel is using. That cross-stream, stream-ordered-allocator race
/// is the source of the nondeterministic MTP corruption (it only bites the MTP loop because that path
/// churns fresh allocations — embed/verify/logits — between GPU-in-flight kernels; the pooled decode
/// path masks it because `Pool::get` force-syncs after each fresh alloc).
///
/// A BLOCKING stream implicitly synchronizes with the NULL stream in BOTH directions: a kernel on it
/// waits for prior NULL-stream work (allocs/memsets/htod), and NULL-stream work (frees/dtoh) waits for
/// prior kernels on it. That establishes the exact stream-order edges the async allocator requires,
/// eliminating the race with zero hot-path changes. There is no concurrency loss: the single-lane
/// decode/MTP loop never overlaps the two streams anyway.
fn fork_blocking_stream(dev: &std::sync::Arc<CudaDevice>) -> cudarc::driver::CudaStream {
    use cudarc::driver::result::stream::{create, destroy, StreamKind};
    let mut s = dev.fork_default_stream().expect("fork stream");
    unsafe {
        destroy(s.stream).expect("destroy nonblocking stream");
        s.stream = create(StreamKind::Default).expect("create blocking stream");
    }
    s
}

fn d<T>(s: &cudarc::driver::CudaSlice<T>) -> u64 { *s.device_ptr() }
fn grid(n: usize) -> (u32, u32, u32) { (((n + 255) / 256) as u32, 1, 1) }
fn fbits(x: f32) -> u64 { x.to_bits() as u64 }
fn u(x: usize) -> u64 { x as u64 }

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

impl GpuModel {
    /// Synchronize the inference stream (blocks until all GPU work on our stream completes).
    pub fn sync_stream(&self) {
        unsafe { cudarc::driver::result::stream::synchronize(self.stream.stream) }.unwrap();
    }
    pub fn new(host: &crate::qwen::Model) -> anyhow::Result<Self> {
        let dev = CudaDevice::new(0)?;
        let stream = fork_blocking_stream(&dev);
        let blas = CudaBlas::new(dev.clone())?;
        unsafe { blas.set_stream(Some(&stream))?; }
        Self::init_ptx(&dev)?;
        let (k, bk) = Self::load_batch_kernels(&dev)?;
        let (cos_table, sin_table) = Self::build_rope_tables(&dev, &host.config)?;
        let sc_pos = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_rope = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_slot = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_winsrc = dev.alloc_zeros::<i32>(MAX_VERIFY * 8).unwrap();
        let sc_parent = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_path = dev.alloc_zeros::<u8>(MAX_VERIFY * MAX_VERIFY).unwrap();
        let sc_tok = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_pstart = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let moe_ids = dev.alloc_zeros::<i32>(16 * MAX_VERIFY).unwrap();
        let moe_wts = dev.alloc_zeros::<f32>(16 * MAX_VERIFY).unwrap();
        let sc_i1a = dev.alloc_zeros::<i32>(1).unwrap();
        let sc_i1b = dev.alloc_zeros::<i32>(1).unwrap();
        let sv_pf = dev.alloc_zeros::<f32>(3 * MAX_VERIFY).unwrap();
        let sv_ki = dev.alloc_zeros::<i32>(2 * MAX_VERIFY).unwrap();
        let sv_sd = dev.alloc_zeros::<u32>(MAX_VERIFY).unwrap();
        let sv_p  = dev.alloc_zeros::<f32>(MAX_VERIFY).unwrap();
        let sv_r  = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let mr_tok = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let mr_pos = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        // alloc_zeros is cuMemAllocAsync and does NOT zero. This buffer MUST be all zeros.
        let mut mtp_sids = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        dev.memset_zeros(&mut mtp_sids).unwrap();
        let deq_scratch = std::sync::Mutex::new(None);
        dev.synchronize().unwrap();
        let cfg = host.config.clone();
        let _h = cfg.hidden_size;
        let _pref = "model.language_model";

        // Upload weights directly from the Model's f32 fields (small models only)
        let up_f = |v: &[f32]| -> S { dev.htod_sync_copy(v).unwrap() };
        let up_b = |v: &[f32]| -> B {
            let bv: Vec<half::bf16> = v.iter().map(|&x| half::bf16::from_f32(x)).collect();
            dev.htod_sync_copy(&bv).unwrap()
        };
        let embed = W::Bf16(up_b(&host.embed_tokens));
        let final_norm = up_f(&host.norm);
        let lm_head = host.lm_head.as_ref().map(|lh| W::Bf16(up_b(lh)));

        let layers: Vec<GpuLayer> = host.layers.iter().map(|l| {
            let la = l.linear_attn.as_ref().map(|la| GpuLinearAttn {
                in_proj: GdnIn::Split {
                    qkv: W::Bf16(up_b(&la.in_proj_qkv)), z: W::Bf16(up_b(&la.in_proj_z)),
                    b: W::Bf16(up_b(&la.in_proj_b)), a: W::Bf16(up_b(&la.in_proj_a)) },
                conv1d: up_f(&la.conv1d), a_log: up_f(&la.a_log), dt_bias: up_f(&la.dt_bias),
                norm: up_f(&la.norm), out_proj: W::Bf16(up_b(&la.out_proj)),
            });
            let fa = l.full_attn.as_ref().map(|fa| GpuFullAttn {
                qkv: AttnIn::Split { q: W::Bf16(up_b(&fa.q_proj)), k: W::Bf16(up_b(&fa.k_proj)),
                                     v: W::Bf16(up_b(&fa.v_proj)) },
                o_proj: W::Bf16(up_b(&fa.o_proj)), q_norm: up_f(&fa.q_norm), k_norm: up_f(&fa.k_norm),
            });
            GpuLayer {
                layer_type: l.layer_type, la, fa,
                mlp: Ffn::Dense(GpuMlp { gate: W::Bf16(up_b(&l.mlp.gate_proj)), up: W::Bf16(up_b(&l.mlp.up_proj)), down: W::Bf16(up_b(&l.mlp.down_proj)) }),
                input_ln: up_f(&l.input_layernorm), post_ln: up_f(&l.post_attention_layernorm),
            }
        }).collect();

        // Load MTP head if present
        let mtp = Self::load_mtp_gpu(&host, &dev)?;
        dev.synchronize()?;
        Ok(Self { dev, blas, stream, cfg, embed, lm_head, final_norm, layers, mtp, k, bk, cos_table, sin_table, sc_pos, sc_rope, sc_slot, sc_winsrc, sc_parent, sc_path, sc_tok, sc_pstart, moe_ids, moe_wts, sc_i1a, sc_i1b, sv_pf, sv_ki, sv_sd, sv_p, sv_r, mr_tok, mr_pos, deq_scratch, mtp_sids, draft_head: None, draft_ids: Vec::new(), tp_rank: 0, tp_world: 1, tp_ctx_dptr: 0, head_visits: None })
    }

    /// Stream-load from safetensors directly as bf16 — no f32 intermediate.
    /// Required for 27B+ models where the f32 intermediate would exceed 128 GB.
    pub fn load_from_dir(model_dir: &str) -> anyhow::Result<(Self, crate::qwen::Config)> {
        use safetensors::{SafeTensors, Dtype};
        let config_path = format!("{}/config.json", model_dir.trim_end_matches('/'));
        let cfg = crate::qwen::Config::from_config_json(&config_path)?;
        let _h = cfg.hidden_size;
        let _pref = "model.language_manager".rsplit_once('.').map(|(p, _)| format!("{}.{}", p, "language_model")).unwrap_or("model.language_model".to_string());

        // Find and load all safetensors shards
        let dir = std::path::Path::new(model_dir);
        let index_path = dir.join("model.safetensors.index.json");
        let safetensors_files: Vec<std::path::PathBuf> = if index_path.exists() {
            let index_raw = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_raw)?;
            index["weight_map"].as_object().unwrap().values()
                .filter_map(|v| v.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .map(|s| dir.join(s))
                .collect()
        } else {
            std::fs::read_dir(dir)?.filter_map(|e| {
                let e = e.ok()?;
                let n = e.file_name().to_string_lossy().to_string();
                if n.ends_with(".safetensors") { Some(e.path()) } else { None }
            }).collect()
        };

        // Process shards ONE AT A TIME — load, upload to GPU, drop raw bytes.
        // Peak memory: one shard (~5 GB) + accumulated GPU weights (~55.6 GB) ≈ 61 GB.
        let pref = "model.language_model";
        let dev = CudaDevice::new(0)?;
        let stream = fork_blocking_stream(&dev);
        let blas = CudaBlas::new(dev.clone())?;
        unsafe { blas.set_stream(Some(&stream))?; }
        Self::init_ptx(&dev)?;
        let (k, bk) = Self::load_batch_kernels(&dev)?;
        let (cos_table, sin_table) = Self::build_rope_tables(&dev, &cfg)?;
        let sc_pos = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_rope = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_slot = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_winsrc = dev.alloc_zeros::<i32>(MAX_VERIFY * 8).unwrap();
        let sc_parent = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_path = dev.alloc_zeros::<u8>(MAX_VERIFY * MAX_VERIFY).unwrap();
        let sc_tok = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_pstart = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let moe_ids = dev.alloc_zeros::<i32>(16 * MAX_VERIFY).unwrap();
        let moe_wts = dev.alloc_zeros::<f32>(16 * MAX_VERIFY).unwrap();
        let sc_i1a = dev.alloc_zeros::<i32>(1).unwrap();
        let sc_i1b = dev.alloc_zeros::<i32>(1).unwrap();
        let sv_pf = dev.alloc_zeros::<f32>(3 * MAX_VERIFY).unwrap();
        let sv_ki = dev.alloc_zeros::<i32>(2 * MAX_VERIFY).unwrap();
        let sv_sd = dev.alloc_zeros::<u32>(MAX_VERIFY).unwrap();
        let sv_p  = dev.alloc_zeros::<f32>(MAX_VERIFY).unwrap();
        let sv_r  = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let mr_tok = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let mr_pos = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        // alloc_zeros is cuMemAllocAsync and does NOT zero. This buffer MUST be all zeros.
        let mut mtp_sids = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        dev.memset_zeros(&mut mtp_sids).unwrap();
        let deq_scratch = std::sync::Mutex::new(None);
        dev.synchronize().unwrap();

        let mut gpu_bf16: std::collections::HashMap<String, B> = std::collections::HashMap::new();
        let mut gpu_f32: std::collections::HashMap<String, S> = std::collections::HashMap::new();

        // Simulated NVFP4 (RUST_INFER_FAKE_QUANT). Round-trips a weight through the 4-bit codec and
        // back to bf16: the bytes stay bf16 so the engine is unmodified, but the VALUES carry exactly
        // the error the real kernel would produce. This is how we test — in the real engine, before
        // any kernel exists — whether the LM head and the GDN projections actually need high
        // precision, or whether that is just something everyone repeats.
        let fq = crate::quant::fake_quant_spec();
        let mut fq_stats: Vec<(String, f32, f32)> = Vec::new();
        if let Some(map) = &fq {
            let names: Vec<String> = map.iter()
                .map(|&(g, f)| format!("{}:{}", crate::quant::group_name(g), crate::quant::fmt_name(f)))
                .collect();
            println!("SIMULATED QUANTIZATION (fake-quant): {}", names.join(" "));
        }

        fn needs_f32(name: &str) -> bool {
            name.ends_with("A_log") || name.ends_with("dt_bias") ||
            (name.contains("norm") && name.ends_with(".weight")) ||
            name.ends_with("conv1d.weight")
        }

        // A quantized artifact stores each quantized tensor as a GROUP of entries rather than one
        // `.weight`. Reassemble them here and dequantize to bf16 on the way to the GPU.
        //
        // This is deliberately the slow path: it proves the artifact end-to-end (encoder -> file ->
        // loader -> dequant) against the simulated-quantization perplexity, with zero kernel work.
        // It does NOT make decode faster — the weights land in VRAM as bf16, so the same bytes stream
        // per token. The speed comes from `gemm_binv_fp4_b` keeping them packed; this is the
        // correctness rung below it.
        //   NVFP4: {stem}.weight_packed [M,K/2] u8 + {stem}.weight_scale [M,K/16] e4m3
        //          + {stem}.weight_global_scale [1] f32
        //   FP8:   {stem}.weight (F8_E4M3) [M,K]  + {stem}.weight_scale [M] f32
        let mut n_dq4 = 0usize;
        let mut n_dq8 = 0usize;
        // Escape hatch: dequantize to bf16 at load. Same numbers, no speedup — useful to prove the
        // fused kernels against, since the two paths must agree exactly.
        let dequant_at_load = std::env::var("RUST_INFER_DEQUANT_AT_LOAD").is_ok();
        // Quantized tensors are held HOST-side until every shard is read, then fused, repacked into
        // mma-fragment order, and uploaded. Deferring is what makes fusion possible: the four GDN
        // input projections must be concatenated along M *before* the mma permutation, and they are
        // not guaranteed to arrive in the same shard.
        type Q4H = (Vec<u8>, Vec<u8>, f32, usize, usize);   // qweight, scales, 1/global_scale, m, k
        type Q8H = (Vec<u8>, Vec<f32>, usize, usize);       // qweight, row_scale, m, k
        let mut host_q4: std::collections::HashMap<String, Q4H> = std::collections::HashMap::new();
        let mut host_q8: std::collections::HashMap<String, Q8H> = std::collections::HashMap::new();

        for (i, sf_path) in safetensors_files.iter().enumerate() {
            if safetensors_files.len() > 1 {
                println!("  Shard {}/{}: {}", i+1, safetensors_files.len(),
                         sf_path.file_name().unwrap_or_default().to_string_lossy());
            }
            let raw = std::fs::read(sf_path)?;
            let st = SafeTensors::deserialize(&raw)?;

            // --- NVFP4 groups: upload PACKED. Keeping them packed IS the speedup; dequantizing
            // here would stream bf16 bytes per token again and gain nothing. ---
            let packed: Vec<String> = st.names().iter()
                .filter(|n| n.ends_with(".weight_packed"))
                .map(|n| n.to_string()).collect();
            for pname in &packed {
                let stem = pname.trim_end_matches(".weight_packed");
                let pv = st.tensor(pname)?;
                let sv = st.tensor(&format!("{}.weight_scale", stem))?;
                let gv = st.tensor(&format!("{}.weight_global_scale", stem))?;
                let (m, k) = (pv.shape()[0], pv.shape()[1] * 2);
                let gs = f32::from_le_bytes(gv.data()[..4].try_into().unwrap());
                if dequant_at_load {
                    let q = crate::quant::Nvfp4Tensor {
                        qweight: pv.data().to_vec(), scales: sv.data().to_vec(),
                        global_scale: gs, m, k };
                    let bv = crate::quant::dequantize_nvfp4(&q);
                    gpu_bf16.insert(format!("{}.weight", stem), dev.htod_sync_copy(&bv).unwrap());
                } else {
                    host_q4.insert(format!("{}.weight", stem), (
                        pv.data().to_vec(), sv.data().to_vec(),
                        1.0f32 / gs,           // kernel multiplies, so pre-invert once here
                        m, k));
                }
                n_dq4 += 1;
            }

            // --- FP8 tensors (dtype F8_E4M3 + an f32 row-scale sibling) ---
            let fp8: Vec<String> = st.names().iter()
                .filter(|n| n.ends_with(".weight"))
                .filter(|n| st.tensor(n).map(|t| t.dtype() == Dtype::F8_E4M3).unwrap_or(false))
                .map(|n| n.to_string()).collect();
            for wname in &fp8 {
                let stem = wname.trim_end_matches(".weight");
                let wv = st.tensor(wname)?;
                let sv = st.tensor(&format!("{}.weight_scale", stem))?;
                let (m, k) = (wv.shape()[0], wv.shape()[1]);
                let row_scale: Vec<f32> = bytemuck::cast_slice::<u8, f32>(sv.data()).to_vec();
                if dequant_at_load {
                    let q = crate::quant::Fp8Tensor {
                        qweight: wv.data().to_vec(), row_scale, m, k };
                    let bv = crate::quant::dequantize_fp8(&q);
                    gpu_bf16.insert(wname.clone(), dev.htod_sync_copy(&bv).unwrap());
                } else {
                    host_q8.insert(wname.clone(), (wv.data().to_vec(), row_scale, m, k));
                }
                n_dq8 += 1;
            }

            for (name, view) in st.tensors() {
                // Skip anything already handled as part of a quantized group.
                if name.ends_with(".weight_packed") || name.ends_with(".weight_scale")
                    || name.ends_with(".weight_global_scale") { continue; }
                if view.dtype() == Dtype::F8_E4M3 { continue; }
                let dt = match view.dtype() { Dtype::BF16 => "BF16", Dtype::F16 => "F16", Dtype::F32 => "F32", _ => "OTHER" };
                let data = view.data();
                if needs_f32(&name) {
                    let fv: Vec<f32> = if dt == "F32" {
                        bytemuck::cast_slice(data).to_vec()
                    } else {
                        bytemuck::cast_slice::<u8, half::bf16>(data).iter().map(|x| x.to_f32()).collect()
                    };
                    gpu_f32.insert(name, dev.htod_sync_copy(&fv).unwrap());
                } else {
                    // Upload bf16 directly — no Vec copy needed for bf16 source data
                    let owned: Option<Vec<half::bf16>> = if dt == "BF16" || dt == "F16" {
                        None
                    } else {
                        let fv: &[f32] = bytemuck::cast_slice(data);
                        Some(fv.iter().map(|&x| half::bf16::from_f32(x)).collect())
                    };
                    let bv: &[half::bf16] = match &owned {
                        Some(v) => v,
                        None => bytemuck::cast_slice(data),
                    };

                    // Apply simulated NVFP4 if this tensor is in scope. Only 2-D weights with K%16==0
                    // (every reduction dim in this family qualifies); shape is [M, K] row-major with
                    // K = in_features = the reduction dim, which is the axis blocks run along.
                    let shape = view.shape();
                    let fmt = fq.as_ref().map_or(crate::quant::Fmt::Bf16,
                                                 |m| crate::quant::fmt_for(m, &name));
                    let in_scope = fmt != crate::quant::Fmt::Bf16;
                    if in_scope && shape.len() == 2 && shape[1] % crate::quant::BLOCK == 0 {
                        let (m, k) = (shape[0], shape[1]);
                        let mut q = bv.to_vec();
                        crate::quant::fake_quant(&mut q, m, k, fmt);
                        let (rel, mx) = crate::quant::roundtrip_error(bv, &q);
                        fq_stats.push((name.clone(), rel, mx));
                        gpu_bf16.insert(name, dev.htod_sync_copy(&q).unwrap());
                    } else {
                        gpu_bf16.insert(name, dev.htod_sync_copy(bv).unwrap());
                    }
                }
            }
            // raw dropped here
        }
        if n_dq4 + n_dq8 > 0 {
            println!("  QUANTIZED artifact: {} NVFP4 + {} FP8 tensors ({})", n_dq4, n_dq8,
                     if dequant_at_load { "dequantized to bf16 at load — NO speedup, reference only" }
                     else { "kept packed; dequantized in-register by the fused GEMV" });
        }

        // Per-group round-trip error — the first thing to look at if quality moves.
        if !fq_stats.is_empty() {
            use std::collections::BTreeMap;
            let mut agg: BTreeMap<&str, (usize, f32, f32)> = BTreeMap::new();
            for (n, rel, mx) in &fq_stats {
                let e = agg.entry(crate::quant::group_name(crate::quant::group_of(n)))
                           .or_insert((0, 0.0, 0.0));
                e.0 += 1;
                e.1 += *rel;
                e.2 = e.2.max(*mx);
            }
            println!("  fake-quant round-trip error (relative L2 per tensor, averaged per group):");
            for (g, (n, sr, mx)) in agg {
                println!("    {:<8} {:3} tensors   rel={:.4}   max|Δw|={:.4}", g, n, sr / n as f32, mx);
            }
        }

        // Helper to get weights from the HashMaps
        // The MTP head is a weight like any other and gets quantized too — so this must look in ALL
        // the format maps. Checking only the bf16 map silently drops MTP on every quantized model,
        // costing the entire 1.4-1.6x speculative-decoding lever.
        // The MTP head's MoE experts must be in the FUSED `experts.gate_up_proj` layout the loader
        // ingests. Some checkpoints (122B) store the MTP experts UN-fused (per-expert
        // `experts.<N>.gate_proj/up_proj/down_proj`), which the fused-GEMM MoE path can't consume.
        // MTP is an optional speculative-decode accelerator, so SKIP it (the model decodes correctly
        // without it) rather than panic on a missing fused tensor.
        let mtp_moe_ok = !cfg.is_moe || {
            let k = "mtp.layers.0.mlp.experts.gate_up_proj";
            gpu_bf16.keys().any(|n| n.starts_with(k))
                || host_q4.keys().any(|n| n.starts_with(k))
                || host_q8.keys().any(|n| n.starts_with(k))
        };
        let has_mtp = (gpu_bf16.contains_key("mtp.fc.weight")
            || host_q4.contains_key("mtp.fc.weight")
            || host_q8.contains_key("mtp.fc.weight")) && mtp_moe_ok;
        if !mtp_moe_ok {
            println!("  MTP head present but its MoE experts are stored un-fused (per-expert) — \
                      skipping MTP (optional speculative-decode head; model decodes fine without it)");
        }
        let mut gb = |n: &str| -> B { gpu_bf16.remove(n).unwrap_or_else(|| panic!("missing bf16 tensor: {}", n)) };

        // Build ONE weight from one-or-more source tensors, concatenated along M.
        //
        // Concatenating is the point: several projections in this architecture read the SAME
        // activation and are separate tensors only because the checkpoint stores them that way. Fused,
        // they cost ONE GEMM — and the M=32 GDN beta/alpha projections stop being their own
        // catastrophically under-parallel launch (grid=2 on a 48-SM GPU; 26 us to move 74 KB; 4.7% of
        // all GEMM time for 0.03% of the bytes).
        //
        // Fusion happens BEFORE the mma permutation, and the NVFP4 tensor scale becomes a per-16-row
        // -tile lookup (every segment boundary is 16-aligned, so a tile never straddles two segments).
        // bf16 is NOT fused: it is not the serving path, and the legacy single-stream forward that the
        // PyTorch reference tests anchor indexes the projections individually.
        // ---- FR-Spec: a smaller LM head, for picking DRAFT tokens only ----
        //
        // Built by subsetting the rows of the real head, which is EXACT (rows do not interact in either
        // codec). Done here, before the mma repack consumes the host bytes.
        //
        // `RUST_INFER_DRAFT_VOCAB=0` disables it; otherwise it is the number of most-frequent token ids
        // to keep (plus the whole tail of the id range, where every special token lives -- a drafter
        // that cannot propose <|im_end|> cannot draft the end of a chat turn).
        let draft_top: usize = std::env::var("RUST_INFER_DRAFT_VOCAB").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(DRAFT_VOCAB_TOP);
        let head_name = if cfg.tie_word_embeddings { format!("{}.embed_tokens.weight", pref) }
                        else { "lm_head.weight".to_string() };
        let (draft_head, draft_ids) = if draft_top == 0 {
            (None, Vec::new())
        } else {
            let rows = crate::quant::draft_vocab_rows(draft_top, cfg.vocab_size);
            let w = if let Some((qw, sc, inv_gs, _m, k)) = host_q4.get(&head_name) {
                let (sq, ss) = crate::quant::subset_rows_nvfp4(qw, sc, *k, &rows);
                let (wt, st) = crate::quant::repack_nvfp4_mma(&sq, &ss, rows.len(), *k);
                let gsv = vec![*inv_gs; rows.len() / 16];
                Some(W::Nvfp4 { qweight: dev.htod_sync_copy(&wt).unwrap(),
                                scales:  dev.htod_sync_copy(&st).unwrap(),
                                gs:      dev.htod_sync_copy(&gsv).unwrap(),
                                m: rows.len(), k: *k })
            } else if let Some((qw, rs, _m, k)) = host_q8.get(&head_name) {
                let (sq, srs) = crate::quant::subset_rows_fp8(qw, rs, *k, &rows);
                let wt = crate::quant::repack_fp8_mma(&sq, rows.len(), *k);
                Some(W::Fp8 { data: dev.htod_sync_copy(&wt).unwrap(),
                              row_scale: dev.htod_sync_copy(&srs).unwrap(),
                              m: rows.len(), k: *k })
            } else {
                None    // bf16 model: not the serving path, skip FR-Spec rather than duplicate 15 GB
            };
            match w {
                Some(w) => {
                    println!("  draft head: {} of {} tokens ({:.0}% of the vocabulary) -- the draft \
                              chain reads {:.0}% of the LM head's bytes",
                             rows.len(), cfg.vocab_size,
                             100.0 * rows.len() as f32 / cfg.vocab_size as f32,
                             100.0 * rows.len() as f32 / cfg.vocab_size as f32);
                    (Some(w), rows)
                }
                None => (None, Vec::new()),
            }
        };


        // A bf16 MTP head (recipe `-mtp`, to fix speculative acceptance) inside an otherwise-quantized
        // model needs the LOADER to treat the MTP head per-weight, not globally: its attention can't be
        // FUSED (a quantized-only trick), and its stacked experts are named WITHOUT the `.weight` suffix
        // the packed path adds. `mtp_quant` says whether the MTP head is quantized (checked before `gwn`
        // borrows the maps).
        let mtp_quant = host_q4.contains_key("mtp.fc.weight") || host_q8.contains_key("mtp.fc.weight");

        let mut gwn = |names: &[String]| -> W {
            if host_q4.contains_key(&names[0]) {
                let parts: Vec<Q4H> = names.iter()
                    .map(|n| host_q4.remove(n).unwrap_or_else(|| panic!("missing nvfp4 tensor: {}", n)))
                    .collect();
                let k = parts[0].4;
                let m: usize = parts.iter().map(|p| p.3).sum();
                let refs: Vec<(&[u8], &[u8], f32, usize)> =
                    parts.iter().map(|p| (&p.0[..], &p.1[..], p.2, p.3)).collect();
                let (qw, sc, gsv) = crate::quant::fuse_nvfp4(&refs, k);
                let (wt, st) = crate::quant::repack_nvfp4_mma(&qw, &sc, m, k);
                return W::Nvfp4 {
                    qweight: dev.htod_sync_copy(&wt).unwrap(),
                    scales:  dev.htod_sync_copy(&st).unwrap(),
                    gs:      dev.htod_sync_copy(&gsv).unwrap(),
                    m, k };
            }
            if host_q8.contains_key(&names[0]) {
                let parts: Vec<Q8H> = names.iter()
                    .map(|n| host_q8.remove(n).unwrap_or_else(|| panic!("missing fp8 tensor: {}", n)))
                    .collect();
                let k = parts[0].3;
                let m: usize = parts.iter().map(|p| p.2).sum();
                let refs: Vec<(&[u8], &[f32], usize)> =
                    parts.iter().map(|p| (&p.0[..], &p.1[..], p.2)).collect();
                let (qw, rs) = crate::quant::fuse_fp8(&refs, k);
                let wt = crate::quant::repack_fp8_mma(&qw, m, k);
                return W::Fp8 {
                    data:      dev.htod_sync_copy(&wt).unwrap(),
                    row_scale: dev.htod_sync_copy(&rs).unwrap(),
                    m, k };
            }
            assert_eq!(names.len(), 1, "bf16 weights are not fused (see GdnIn/AttnIn)");
            W::Bf16(gpu_bf16.remove(&names[0]).unwrap_or_else(|| panic!("missing tensor: {}", names[0])))
        };
        let mut gf = |n: &str| -> S { gpu_f32.remove(n).unwrap_or_else(|| panic!("missing f32 tensor: {}", n)) };

        // Fuse if the artifact is quantized (`gwn` concatenates along M); leave bf16 split.
        let quantized = n_dq4 + n_dq8 > 0 && !dequant_at_load;
        let attn_in = |gwn: &mut dyn FnMut(&[String]) -> W, lp: &str, fuse: bool| -> AttnIn {
            let n = |s: &str| format!("{}.self_attn.{}.weight", lp, s);
            if fuse { AttnIn::Fused(gwn(&[n("q_proj"), n("k_proj"), n("v_proj")])) }
            else { AttnIn::Split { q: gwn(&[n("q_proj")]), k: gwn(&[n("k_proj")]), v: gwn(&[n("v_proj")]) } }
        };
        let gdn_in = |gwn: &mut dyn FnMut(&[String]) -> W, lp: &str| -> GdnIn {
            let n = |s: &str| format!("{}.linear_attn.{}.weight", lp, s);
            if quantized { GdnIn::Fused(gwn(&[n("in_proj_qkv"), n("in_proj_z"), n("in_proj_b"), n("in_proj_a")])) }
            else { GdnIn::Split { qkv: gwn(&[n("in_proj_qkv")]), z: gwn(&[n("in_proj_z")]),
                                  b: gwn(&[n("in_proj_b")]), a: gwn(&[n("in_proj_a")]) } }
        };
        // FFN loader: dense MLP (qwen3_5) or the MoE block (qwen3_5_moe). Expert tensors are STACKED and
        // gate+up FUSED, stored WITHOUT a `.weight` suffix; router/shared/shared_gate carry `.weight`.
        let load_ffn = |gwn: &mut dyn FnMut(&[String]) -> W, lp: &str, is_moe: bool, q: bool| -> Ffn {
            // The bf16 checkpoint names the stacked experts WITHOUT a `.weight` suffix; the quantizer
            // packs them as `<name>.weight_packed`, which the packed-ingestion keys as `<name>.weight`.
            // `q` = are THESE experts quantized (per-head, so a bf16 MTP head in a quantized model works).
            let esuf = if q { ".weight" } else { "" };
            if is_moe {
                Ffn::Moe(GpuMoe {
                    router:      gwn(&[format!("{}.mlp.gate.weight", lp)]),
                    gate_up:     gwn(&[format!("{}.mlp.experts.gate_up_proj{}", lp, esuf)]),
                    down:        gwn(&[format!("{}.mlp.experts.down_proj{}", lp, esuf)]),
                    shared: GpuMlp {
                        gate: gwn(&[format!("{}.mlp.shared_expert.gate_proj.weight", lp)]),
                        up:   gwn(&[format!("{}.mlp.shared_expert.up_proj.weight", lp)]),
                        down: gwn(&[format!("{}.mlp.shared_expert.down_proj.weight", lp)]),
                    },
                    shared_gate: gwn(&[format!("{}.mlp.shared_expert_gate.weight", lp)]),
                    experts_sharded: false,
                })
            } else {
                Ffn::Dense(GpuMlp {
                    gate: gwn(&[format!("{}.mlp.gate_proj.weight", lp)]),
                    up:   gwn(&[format!("{}.mlp.up_proj.weight", lp)]),
                    down: gwn(&[format!("{}.mlp.down_proj.weight", lp)]),
                })
            }
        };

        let embed = gwn(&[format!("{}.embed_tokens.weight", pref)]);
        let final_norm = gf(&format!("{}.norm.weight", pref));
        let lm_head = if cfg.tie_word_embeddings { None } else { Some(gwn(&["lm_head.weight".to_string()])) };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lpref = format!("{}.layers.{}", pref, i);
            let lt = cfg.layer_types[i];
            let (la, fa) = match lt {
                crate::qwen::LayerType::LinearAttention => {
                    let la = GpuLinearAttn {
                        in_proj: gdn_in(&mut gwn, &lpref),
                        conv1d: gf(&format!("{}.linear_attn.conv1d.weight", lpref)),
                        a_log: gf(&format!("{}.linear_attn.A_log", lpref)),
                        dt_bias: gf(&format!("{}.linear_attn.dt_bias", lpref)),
                        norm: gf(&format!("{}.linear_attn.norm.weight", lpref)),
                        out_proj: gwn(&[format!("{}.linear_attn.out_proj.weight", lpref)]),
                    };
                    (Some(la), None)
                }
                crate::qwen::LayerType::FullAttention => {
                    let fa = GpuFullAttn {
                        qkv: attn_in(&mut gwn, &lpref, quantized),
                        o_proj: gwn(&[format!("{}.self_attn.o_proj.weight", lpref)]),
                        q_norm: gf(&format!("{}.self_attn.q_norm.weight", lpref)),
                        k_norm: gf(&format!("{}.self_attn.k_norm.weight", lpref)),
                    };
                    (None, Some(fa))
                }
            };
            layers.push(GpuLayer {
                layer_type: lt, la, fa,
                mlp: load_ffn(&mut gwn, &lpref, cfg.is_moe_layer(i), quantized),
                input_ln: gf(&format!("{}.input_layernorm.weight", lpref)),
                post_ln: gf(&format!("{}.post_attention_layernorm.weight", lpref)),
            });
        }

        // Load MTP if present
        let mtp = if has_mtp {
            println!("Loading MTP head...");
            Some(GpuMtpLayer {
                fc: gwn(&["mtp.fc.weight".to_string()]),
                pre_fc_norm_hidden: gf("mtp.pre_fc_norm_hidden.weight"),
                pre_fc_norm_embedding: gf("mtp.pre_fc_norm_embedding.weight"),
                input_ln: gf("mtp.layers.0.input_layernorm.weight"),
                post_ln: gf("mtp.layers.0.post_attention_layernorm.weight"),
                fa: GpuFullAttn {
                    qkv: attn_in(&mut gwn, "mtp.layers.0", mtp_quant),
                    o_proj: gwn(&["mtp.layers.0.self_attn.o_proj.weight".to_string()]),
                    q_norm: gf("mtp.layers.0.self_attn.q_norm.weight"),
                    k_norm: gf("mtp.layers.0.self_attn.k_norm.weight"),
                },
                mlp: load_ffn(&mut gwn, "mtp.layers.0", cfg.is_moe, mtp_quant),
                final_norm: gf("mtp.norm.weight"),
            })
        } else { None };

        // HashMaps now empty (all weights consumed). Drop explicitly.
        drop(gpu_bf16);
        drop(gpu_f32);
        dev.synchronize()?;

        Ok((Self { dev, blas, stream, cfg: cfg.clone(), embed, lm_head, final_norm, layers, mtp, k, bk, cos_table, sin_table, sc_pos, sc_rope, sc_slot, sc_winsrc, sc_parent, sc_path, sc_tok, sc_pstart, moe_ids, moe_wts, sc_i1a, sc_i1b, sv_pf, sv_ki, sv_sd, sv_p, sv_r, mr_tok, mr_pos, deq_scratch, mtp_sids, draft_head, draft_ids, tp_rank: 0, tp_world: 1, tp_ctx_dptr: 0, head_visits: None }, cfg))
    }

    fn init_ptx(dev: &Arc<CudaDevice>) -> anyhow::Result<()> {
        let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_kernels.ptx")?);
        let fnames = ["rmsnorm_qwen","rmsnorm_gated","rmsnorm_perhead","silu_mul","add_residual",
            "sigmoid_gate","rope_rot_half","gqa_attention","write_kv","split_qgate","conv1d_step","delta_step",
            "f32tobf16","bf16tof32","fused_residual_rmsnorm","argmax_pass1","argmax_pass2"];
        dev.load_ptx(ptx, "gpu_kernels", &fnames)?;
        Ok(())
    }

    /// Refuse to run a binary whose PTX came from different kernel sources.
    ///
    /// The PTX is loaded from disk at runtime, so a deploy that copies the binary but not the kernels
    /// leaves a fresh binary driving OLD kernels. When launch parameters have changed (shared-memory
    /// size, grid shape, an argument list) that is not a slowdown — it is CUDA_ERROR_ILLEGAL_ADDRESS
    /// out of code that is perfectly correct. It shipped once. `scripts/deploy.sh` checks the three
    /// files agree, but a procedural check only guards the paths it is actually run on.
    ///
    /// So build.rs hashes the .cu bytes and bakes the hash into BOTH the PTX (`-DKERNEL_BUILD_ID`) and
    /// the binary (`env!("KERNEL_BUILD_ID")`). If they disagree the pair is mismatched, and we say so
    /// at startup — loudly, on any box, however it was assembled.
    fn assert_kernel_build_id(dev: &Arc<CudaDevice>, module: &str) -> anyhow::Result<()> {
        let expect = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap_or(0);
        let Some(f) = dev.get_func(module, "kernel_build_id") else {
            anyhow::bail!("src/ptx/{module}.ptx has no kernel_build_id — it predates the build-ID stamp \
                           and cannot be verified against this binary. Run `cargo build --release`.");
        };
        let out = dev.alloc_zeros::<u64>(1)?;
        unsafe {
            use cudarc::driver::LaunchAsync;
            f.launch(cudarc::driver::LaunchConfig {
                grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0
            }, (&out,))?;
        }
        dev.synchronize()?;
        let got = dev.dtoh_sync_copy(&out)?[0];
        if got != expect {
            anyhow::bail!(
                "STALE KERNELS: src/ptx/{module}.ptx was built from different kernel sources than this \
                 binary (ptx={got:016x}, binary={expect:016x}). A deploy is THREE files — the binary \
                 AND both src/ptx/*.ptx. Run `cargo build --release && ./scripts/deploy.sh`.");
        }
        Ok(())
    }

    fn load_batch_kernels(dev: &Arc<CudaDevice>) -> anyhow::Result<(KernelTable, HashMap<String, CudaFunction>)> {
        let bptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx")?);
        let bfnames = ["rmsnorm_b","fused_res_rmsnorm_b","add_residual_b","silu_mul_b",
            "rmsnorm_perhead_b","rmsnorm_gated_b","rope_b","conv1d_b","delta_step_b",
            "write_kv_b","compact_kv_b","sigmoid_gate_b","split_qgate_b",
            "gather_rope_b","embed_gather_b","argmax_b",
            "conv1d_prefill","conv1d_prefill_state","delta_step_prefill","attn_softmax_tile","attn_finalize","attn_tile_init","write_kv_prefill","gqa_attn_prefill",
            "rep_penalty_b","gqa_attn_splitk","gqa_attn_reduce","sample_b","concat_b",
            "gemm_binv_b","sample_prob_b","spec_verify_b","nll_b",
            "gemm_mma_fp4_b","gemm_mma_fp8_b","dequant_fp4_tiled_b","dequant_fp8_tiled_b",
            "embed_gather_fp4_tiled_b","embed_gather_fp8_tiled_b","bw_read_b",
            "split_gdn_b","split_qkv_b",
            "moe_router_topk_b","moe_experts_b","moe_experts_fp4_b","moe_shared_combine_b",
            "moe_gate_up_fp4_warp","moe_silu_b","moe_down_fp4_warp",
            "gemm_moe_mma_fp4","moe_combine_experts_b","moe_silu_bf16_b",
            "moe_count_b","moe_offsets_b","moe_scatter_b","moe_tilemap_b","moe_gather_x_b",
            "gemm_moe_grouped_mma_fp4","moe_combine_grouped_b",
            "tp_mask_rows","tp_gate_copy_signal","tp_wait_add",
            "tp_bench_fill","tp_bench_validate","tp_bench_stall",
            "kernel_build_id"];
        dev.load_ptx(bptx, "gpu_batch", &bfnames)?;
        Self::assert_kernel_build_id(dev, "gpu_batch")?;
        let mut bk = HashMap::new();
        for n in &bfnames {
            if let Some(f) = dev.get_func("gpu_batch", n) { bk.insert(n.to_string(), f); }
        }
        let gf = |n: &str| dev.get_func("gpu_kernels", n);
        let k = KernelTable {
            rmsnorm_qwen: gf("rmsnorm_qwen"), rmsnorm_gated: gf("rmsnorm_gated"),
            rmsnorm_perhead: gf("rmsnorm_perhead"), silu_mul: gf("silu_mul"),
            add_residual: gf("add_residual"), sigmoid_gate: gf("sigmoid_gate"),
            rope_rot_half: gf("rope_rot_half"), gqa_attention: gf("gqa_attention"),
            write_kv: gf("write_kv"), split_qgate: gf("split_qgate"),
            conv1d_step: gf("conv1d_step"), delta_step: gf("delta_step"),
            f32tobf16: gf("f32tobf16"), bf16tof32: gf("bf16tof32"),
            fused_residual_rmsnorm: gf("fused_residual_rmsnorm"),
            argmax_pass1: gf("argmax_pass1"), argmax_pass2: gf("argmax_pass2"),
        };
        Ok((k, bk))
    }

    fn build_rope_tables(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config) -> anyhow::Result<(S, S)> {
        let rdim = cfg.rotary_dim;
        let half = rdim / 2;
        let theta = cfg.rope_theta;
        let max_pos = cfg.max_position_embeddings;
        let mut cos_t = vec![0.0f32; max_pos * rdim];
        let mut sin_t = vec![0.0f32; max_pos * rdim];
        for p in 0..max_pos {
            let pf = p as f32;
            for i in 0..half {
                let f = pf * theta.powf(-(2.0 * i as f32) / rdim as f32);
                let (c, s) = (f.cos(), f.sin());
                cos_t[p * rdim + i] = c; sin_t[p * rdim + i] = s;
                cos_t[p * rdim + i + half] = c; sin_t[p * rdim + i + half] = s;
            }
        }
        let cos_table = dev.htod_sync_copy(&cos_t)?;
        let sin_table = dev.htod_sync_copy(&sin_t)?;
        Ok((cos_table, sin_table))
    }

    fn load_mtp_gpu(host: &crate::qwen::Model, dev: &Arc<CudaDevice>) -> anyhow::Result<Option<GpuMtpLayer>> {
        let mtp = match &host.mtp {
            Some(m) => {
                let up_f = |v: &[f32]| -> S { dev.htod_sync_copy(v).unwrap() };
                let up_b = |v: &[f32]| -> B {
                    let bv: Vec<half::bf16> = v.iter().map(|&x| half::bf16::from_f32(x)).collect();
                    dev.htod_sync_copy(&bv).unwrap()
                };
                println!("Loading MTP head...");
                Some(GpuMtpLayer {
                    fc: W::Bf16(up_b(&m.fc)),
                    pre_fc_norm_hidden: up_f(&m.pre_fc_norm_hidden),
                    pre_fc_norm_embedding: up_f(&m.pre_fc_norm_embedding),
                    input_ln: up_f(&m.input_ln), post_ln: up_f(&m.post_ln),
                    fa: GpuFullAttn {
                        qkv: AttnIn::Split { q: W::Bf16(up_b(&m.q_proj)), k: W::Bf16(up_b(&m.k_proj)),
                                             v: W::Bf16(up_b(&m.v_proj)) },
                        o_proj: W::Bf16(up_b(&m.o_proj)), q_norm: up_f(&m.q_norm), k_norm: up_f(&m.k_norm),
                    },
                    mlp: Ffn::Dense(GpuMlp { gate: W::Bf16(up_b(&m.gate_proj)), up: W::Bf16(up_b(&m.up_proj)), down: W::Bf16(up_b(&m.down_proj)) }),
                    final_norm: up_f(&m.final_norm),
                })
            }
            None => { println!("No MTP head."); None }
        };
        Ok(mtp)
    }

    pub fn dev(&self) -> &Arc<CudaDevice> { &self.dev }
    pub fn cfg(&self) -> &Config { &self.cfg }
    pub fn layer(&self, li: usize) -> &GpuLayer { &self.layers[li] }

    /// bf16 GEMM: out[outn] = W_bf16[outn,inn] @ x_f32[inn], compute in f32, output f32.
    fn gemm_bf16(&self, pool: &mut Pool, w: &B, x: &S, out: &mut S, inn: usize, outn: usize) {
        let xb = pool.get_bf16(inn);
        let mut yb = pool.get_bf16(outn);
        unsafe { self.k.f32tobf16.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(inn),block_dim:(256,1,1),shared_mem_bytes:0},
            (*xb.device_ptr() as u64, d(x), inn as i32)).unwrap(); }
        let cfg = cudarc::cublas::GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: outn as i32, n: 1, k: inn as i32,
            alpha: half::bf16::from_f32(1.0), lda: inn as i32,
            ldb: inn as i32, beta: half::bf16::from_f32(0.0), ldc: outn as i32,
        };
        unsafe { self.blas.gemm(cfg, w, &xb, &mut yb).expect("gemm bf16"); }
        unsafe { self.k.bf16tof32.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(outn),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(out), *yb.device_ptr() as u64, outn as i32)).unwrap(); }
        pool.release_bf16(xb, inn);
        pool.release_bf16(yb, outn);
    }

    pub fn rmsnorm_qwen(&self, out: &S, x: &S, w: &S, n: usize, eps: f32) {
        unsafe { self.k.rmsnorm_qwen.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(1,1,1),block_dim:(n as u32,1,1),shared_mem_bytes:(n*4) as u32},
            (d(out), d(x), d(w), n as i32, fbits(eps))).unwrap(); }
    }
    pub fn rmsnorm_gated(&self, out: &S, x: &S, z: &S, w: &S, m: usize, eps: f32) {
        unsafe { self.k.rmsnorm_gated.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(1,1,1),block_dim:(m as u32,1,1),shared_mem_bytes:(m*4) as u32},
            (d(out), d(x), d(z), d(w), m as i32, fbits(eps))).unwrap(); }
    }
    pub fn test_conv1d(&self, x: &S, state: &mut S, w: &S, conv_dim: usize, k: usize) {
        unsafe { self.k.conv1d_step.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(conv_dim),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(x), d(state), d(w), conv_dim as i32, k as i32)).unwrap(); }
    }
    pub fn test_delta_step(&self, out: &mut S, q: &S, k: &S, v: &S, b: &S, a: &S,
                           state: &mut S, nh: usize, kd: usize, vd: usize, a_log: &S, dt_bias: &S) {
        let smem = ((kd + 4*vd + kd) as u32)*4;
        unsafe { self.k.delta_step.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nh as u32,1,1),block_dim:(kd as u32,1,1),shared_mem_bytes:smem},
            (d(out), d(q), d(k), d(v), d(b), d(a), d(state),
             nh as i32, kd as i32, vd as i32, d(a_log), d(dt_bias))).unwrap(); }
    }

    /// forward one token; `hidden` becomes the residual (consumed). Returns final-normed hidden.
    pub fn forward_token(&self, pool: &mut Pool, hidden: S, pos: usize, state: &mut GpuState, cos: &S, sin: &S) -> S {
        self.forward_token_inner(pool, hidden, pos, state, cos, sin, None)
    }
    pub fn forward_token_captured(&self, pool: &mut Pool, hidden: S, pos: usize, state: &mut GpuState, cos: &S, sin: &S,
                                  caps: &mut Vec<Vec<f32>>) -> S {
        self.forward_token_inner(pool, hidden, pos, state, cos, sin, Some(caps))
    }
    fn forward_token_inner(&self, pool: &mut Pool, hidden: S, pos: usize, state: &mut GpuState, cos: &S, sin: &S,
                           mut caps: Option<&mut Vec<Vec<f32>>>) -> S {
        let h = self.cfg.hidden_size;
        let residual = hidden;
        let normed = pool.get(h);
        for (li, layer) in self.layers.iter().enumerate() {
            self.rmsnorm_qwen(&normed, &residual, &layer.input_ln, h, self.cfg.rms_eps);
            let mixer = match layer.layer_type {
                LayerType::LinearAttention => self.linear_attn(pool, &normed, layer.la.as_ref().unwrap(),
                    state.conv_state[li].as_mut().unwrap(), state.s_state[li].as_mut().unwrap()),
                LayerType::FullAttention => self.full_attn(pool, &normed, layer.fa.as_ref().unwrap(), pos,
                    state.k_cache[li].as_mut().unwrap(), state.v_cache[li].as_mut().unwrap(), cos, sin),
            };
            // fused: residual += mixer; then normed = rmsnorm(residual, post_ln)
            unsafe { self.k.fused_residual_rmsnorm.clone().unwrap().launch_on_stream(&self.stream,
                LaunchConfig{grid_dim:(1,1,1),block_dim:(h as u32,1,1),shared_mem_bytes:(h*4) as u32},
                (d(&normed), d(&residual), d(&mixer), d(&layer.post_ln), h as i32, fbits(self.cfg.rms_eps))).unwrap(); }
            let mlp_out = match &layer.mlp {
                Ffn::Dense(m) => self.mlp(pool, &normed, m),
                Ffn::Moe(_) => panic!("legacy single-token forward does not support MoE (use the batched path)"),
            };
            unsafe { self.k.add_residual.clone().unwrap().launch_on_stream(&self.stream,
                LaunchConfig{grid_dim:grid(h),block_dim:(256,1,1),shared_mem_bytes:0},
                (d(&residual), d(&residual), d(&mlp_out), h as i32)).unwrap(); }
            if let Some(c) = caps.as_mut() {
                // TRUNCATE to h. Pool buffers are power-of-two BUCKETS now, so a dtoh of the whole
                // slice can hand back more elements than were asked for (h=5120 on 27B lands in an
                // 8192 bucket). Anything comparing these captures by length would silently disagree.
                c.push({
                    self.sync_stream();
                    let mut v = self.dev.dtoh_sync_copy(&residual).unwrap();
                    v.truncate(h);
                    v
                });
            }
            pool.release(mixer, h); pool.release(mlp_out, h);
        }
        let out = pool.get(h);
        self.rmsnorm_qwen(&out, &residual, &self.final_norm, h, self.cfg.rms_eps);
        pool.release(residual, h); pool.release(normed, h);
        out
    }

    pub fn mlp(&self, pool: &mut Pool, x: &S, mlp: &GpuMlp) -> S {
        let (im, h) = (self.cfg.intermediate_size, self.cfg.hidden_size);
        let mut gate = pool.get(im);
        let mut upb = pool.get(im);
        self.gemm_bf16(pool, mlp.gate.bf16(), x, &mut gate, h, im);
        self.gemm_bf16(pool, mlp.up.bf16(), x, &mut upb, h, im);
        unsafe { self.k.silu_mul.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(im),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&gate), d(&gate), d(&upb), im as i32)).unwrap(); }
        let mut out = pool.get(h);
        self.gemm_bf16(pool, mlp.down.bf16(), &gate, &mut out, im, h);
        pool.release(gate, im); pool.release(upb, im);
        out
    }

    pub fn full_attn(&self, pool: &mut Pool, hidden: &S, fa: &GpuFullAttn, pos: usize,
                 kc: &mut S, vc: &mut S, cos: &S, sin: &S) -> S {
        let cfg = &self.cfg;
        let (h, nh, nkv, hd, rdim) = (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);
        let stride = state_stride(cfg);
        let mut qg = pool.get(nh*hd*2);
        let mut k = pool.get(nkv*hd);
        let mut v = pool.get(nkv*hd);
        let q = pool.get(nh*hd);
        let gate = pool.get(nh*hd);
        let (qp, kp, vp) = match &fa.qkv {
            AttnIn::Split { q, k, v } => (q, k, v),
            AttnIn::Fused(_) => panic!("legacy single-stream path cannot use a fused (quantized) model"),
        };
        self.gemm_bf16(pool, qp.bf16(), hidden, &mut qg, h, nh*hd*2);
        self.gemm_bf16(pool, kp.bf16(), hidden, &mut k, h, nkv*hd);
        self.gemm_bf16(pool, vp.bf16(), hidden, &mut v, h, nkv*hd);
        unsafe { self.k.split_qgate.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(nh*hd),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&q), d(&gate), d(&qg), nh as i32, hd as i32)).unwrap(); }
        unsafe { self.k.rmsnorm_perhead.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nh as u32,1,1),block_dim:(hd as u32,1,1),shared_mem_bytes:(hd*4) as u32},
            (d(&q), d(&q), d(&fa.q_norm), nh as i32, hd as i32, fbits(cfg.rms_eps))).unwrap(); }
        unsafe { self.k.rmsnorm_perhead.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nkv as u32,1,1),block_dim:(hd as u32,1,1),shared_mem_bytes:(hd*4) as u32},
            (d(&k), d(&k), d(&fa.k_norm), nkv as i32, hd as i32, fbits(cfg.rms_eps))).unwrap(); }
        unsafe { self.k.rope_rot_half.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(nh*(rdim/2)),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&q), d(cos), d(sin), nh as i32, hd as i32, rdim as i32)).unwrap(); }
        unsafe { self.k.rope_rot_half.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(nkv*(rdim/2)),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&k), d(cos), d(sin), nkv as i32, hd as i32, rdim as i32)).unwrap(); }
        unsafe { self.k.write_kv.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(nkv*hd),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(kc), d(vc), d(&k), d(&v), pos as i32, stride as i32, nkv as i32, hd as i32)).unwrap(); }
        let attn = pool.get(nh*hd);
        let scale = 1.0f32/(hd as f32).sqrt();
        let smem = ((pos+1) as u32 + hd as u32)*4;
        unsafe { self.k.gqa_attention.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nh as u32,1,1),block_dim:(hd as u32,1,1),shared_mem_bytes:smem},
            (d(&attn), d(&q), d(kc), d(vc), (pos+1) as i32, stride as i32, nh as i32, nkv as i32, hd as i32, fbits(scale))).unwrap(); }
        unsafe { self.k.sigmoid_gate.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(nh*hd),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&attn), d(&gate), (nh*hd) as i32)).unwrap(); }
        let mut out = pool.get(h);
        self.gemm_bf16(pool, fa.o_proj.bf16(), &attn, &mut out, nh*hd, h);
        pool.release(qg, nh*hd*2); pool.release(k, nkv*hd); pool.release(v, nkv*hd);
        pool.release(q, nh*hd); pool.release(gate, nh*hd); pool.release(attn, nh*hd);
        out
    }

    pub fn linear_attn(&self, pool: &mut Pool, hidden: &S, la: &GpuLinearAttn, conv_state: &mut S, s_state: &mut S) -> S {
        let cfg = &self.cfg;
        let (h, vd, nh) = (cfg.hidden_size, cfg.lin_v_dim, cfg.lin_num_v_heads);
        let value_dim = cfg.value_dim();
        let core = self.linear_attn_core(pool, hidden, la, conv_state, s_state);
        let mut z = pool.get(value_dim);
        let pz = match &la.in_proj {
            GdnIn::Split { z, .. } => z,
            GdnIn::Fused(_) => panic!("legacy single-stream path cannot use a fused (quantized) model"),
        };
        self.gemm_bf16(pool, pz.bf16(), hidden, &mut z, h, value_dim);
        let normed = pool.get(value_dim);
        unsafe { self.k.rmsnorm_gated.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nh as u32,1,1),block_dim:(vd as u32,1,1),shared_mem_bytes:(vd*4) as u32},
            (d(&normed), d(&core), d(&z), d(&la.norm), vd as i32, fbits(cfg.rms_eps))).unwrap(); }
        let mut out = pool.get(h);
        self.gemm_bf16(pool, la.out_proj.bf16(), &normed, &mut out, value_dim, h);
        pool.release(core, nh*vd); pool.release(z, value_dim); pool.release(normed, value_dim);
        out
    }
    pub fn linear_attn_core(&self, pool: &mut Pool, hidden: &S, la: &GpuLinearAttn, conv_state: &mut S, s_state: &mut S) -> S {
        let cfg = &self.cfg;
        let (h, nh, kd, vd) = (cfg.hidden_size, cfg.lin_num_v_heads, cfg.lin_k_dim, cfg.lin_v_dim);
        let key_dim = cfg.key_dim(); let value_dim = cfg.value_dim();
        let conv_dim = key_dim*2 + value_dim; let ck = cfg.conv_kernel;
        let (pqkv, pb, pa) = match &la.in_proj {
            GdnIn::Split { qkv, b, a, .. } => (qkv, b, a),
            GdnIn::Fused(_) => panic!("legacy single-stream path cannot use a fused (quantized) model"),
        };
        let mut qkv = pool.get(conv_dim);
        self.gemm_bf16(pool, pqkv.bf16(), hidden, &mut qkv, h, conv_dim);
        unsafe { self.k.conv1d_step.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(conv_dim),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&qkv), d(conv_state), d(&la.conv1d), conv_dim as i32, ck as i32)).unwrap(); }
        let qkv_base = d(&qkv);
        let k_ptr = qkv_base + u(key_dim)*4;
        let v_ptr = qkv_base + u(2*key_dim)*4;
        let mut b = pool.get(nh);
        let mut a = pool.get(nh);
        self.gemm_bf16(pool, pb.bf16(), hidden, &mut b, h, nh);
        self.gemm_bf16(pool, pa.bf16(), hidden, &mut a, h, nh);
        let core = pool.get(nh*vd);
        let smem = ((kd + 4*vd + kd) as u32)*4;
        unsafe { self.k.delta_step.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nh as u32,1,1),block_dim:(kd as u32,1,1),shared_mem_bytes:smem},
            (d(&core), qkv_base, k_ptr, v_ptr, d(&b), d(&a), d(s_state),
             nh as i32, kd as i32, vd as i32, d(&la.a_log), d(&la.dt_bias))).unwrap(); }
        pool.release(qkv, conv_dim); pool.release(b, nh); pool.release(a, nh);
        core
    }

    pub fn logits(&self, pool: &mut Pool, hidden: &S) -> S {
        let (v, h) = (self.cfg.vocab_size, self.cfg.hidden_size);
        let mut logits = pool.get(v);
        self.gemm_bf16(pool, self.embed.bf16(), hidden, &mut logits, h, v);
        logits
    }

    /// Greedy argmax of logits, computed on device (two-pass). Returns the token id.
    pub fn argmax_gpu(&self, pool: &mut Pool, logits: &S) -> u32 {
        let v = self.cfg.vocab_size;
        let block = 1024usize;
        let nblocks = (v + block - 1) / block;
        let idxs = pool.get(nblocks);   // i32 reinterpreted
        let vals = pool.get(nblocks);   // f32
        unsafe { self.k.argmax_pass1.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nblocks as u32,1,1),block_dim:(block as u32,1,1),shared_mem_bytes:(block*4*2) as u32},
            (d(&idxs), d(&vals), d(logits), v as i32)).unwrap(); }
        let token_dev = self.dev.alloc_zeros::<i32>(1).unwrap();
        unsafe { self.k.argmax_pass2.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(1,1,1),block_dim:(nblocks as u32,1,1),shared_mem_bytes:(nblocks*4*2) as u32},
            (*token_dev.device_ptr() as u64, d(&idxs), d(&vals), nblocks as i32)).unwrap(); }
        self.sync_stream();
        let token_host = self.dev.dtoh_sync_copy(&token_dev).unwrap();
        pool.release(idxs, nblocks);
        pool.release(vals, nblocks);
        token_host[0] as u32
    }

    pub fn embed_row(&self, _token: u32) -> S {
        // Deprecated — use embed_gather_b via embed_batch instead
        self.dev.alloc_zeros::<f32>(self.cfg.hidden_size).unwrap()
    }

    /// Allocate per-layer inference state (KV caches for full-attn, conv+recurrent for linear-attn).
    pub fn new_state(&self) -> GpuState { self.new_state_stride(self.cfg.max_position_embeddings) }
    pub fn new_state_stride(&self, stride: usize) -> GpuState {
        let cfg = &self.cfg;
        let mut k_cache = vec![];
        let mut v_cache = vec![];
        let mut conv_state = vec![];
        let mut s_state = vec![];
        for lt in &cfg.layer_types {
            match lt {
                LayerType::FullAttention => {
                    let bytes = cfg.num_kv_heads * cfg.head_dim * stride;
                    k_cache.push(Some(self.dev.alloc_zeros::<f32>(bytes).unwrap()));
                    v_cache.push(Some(self.dev.alloc_zeros::<f32>(bytes).unwrap()));
                    conv_state.push(None);
                    s_state.push(None);
                }
                LayerType::LinearAttention => {
                    let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
                    conv_state.push(Some(self.dev.alloc_zeros::<f32>(conv_dim * cfg.conv_kernel).unwrap()));
                    s_state.push(Some(self.dev.alloc_zeros::<f32>(cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim).unwrap()));
                    k_cache.push(None);
                    v_cache.push(None);
                }
            }
        }
        GpuState { k_cache, v_cache, conv_state, s_state, pos: 0, max_seq_len: stride }
    }

    // ===================== BATCHED (continuous batching) =====================

    /// batched bf16 GEMM: out[outn,B] = W_bf16[outn,inn] @ X_f32[inn,B]
    /// Pure bf16→bf16 GEMM for the activation path: zero conversion kernels.
    /// out[outn,B] = W_bf16[outn,inn] @ X_bf16[inn,B]
    /// Batch-invariant custom kernel for batch <= BINV_BATCH (the MTP verify runs at N=2, per-lane
    /// decode at N=1) so the verify's column 0 is bit-identical to N=1 → no cuBLAS N=1-vs-N=2 drift
    /// → 9B/27B MTP is coherent. cuBLAS for larger batch (multi-lane decode) where tensor cores win.
    /// The custom kernel is also slightly faster than cuBLAS at batch 1 (more coalesced, less overhead).
    /// Dequantize a weight into persistent bf16 scratch and immediately GEMM with it (PREFILL only).
    ///
    /// The scratch is sized once to the largest tensor and reused, so there is no hot-path alloc. The
    /// dequant and the GEMM happen together while the buffer is held, so the scratch is never aliased.
    ///
    /// This is emphatically NOT the decode path: dequantizing to bf16 there would stream bf16 bytes
    /// again and throw away the whole point. It exists because prefill reads each weight once for many
    /// tokens, so bandwidth stops being the constraint and cuBLAS's tensor cores win.
    fn gemm_quant_prefill(&self, w: &W, x: &B, out: &mut B, inn: usize, outn: usize, batch: usize) {
        let n = inn * outn;
        let mut guard = self.deq_scratch.lock().unwrap();
        if guard.as_ref().map_or(true, |(_, cap)| *cap < n) {
            let buf = self.dev.alloc_zeros::<half::bf16>(n).unwrap();
            self.dev.synchronize().unwrap();
            *guard = Some((buf, n));
        }
        let (buf, _) = guard.as_mut().unwrap();
        let ptr = *buf.device_ptr() as u64;
        let grid = ((n + 255) / 256) as u32;
        match w {
            W::Nvfp4 { qweight, scales, gs, .. } => {
                blaunch!(self, "dequant_fp4_tiled_b", (grid,1,1), (256,1,1), 0,
                    (ptr, *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                     d(gs), outn as i32, inn as i32));
            }
            W::Fp8 { data, row_scale, .. } => {
                blaunch!(self, "dequant_fp8_tiled_b", (grid,1,1), (256,1,1), 0,
                    (ptr, *data.device_ptr() as u64, d(row_scale), outn as i32, inn as i32));
            }
            W::Nvfp4Raw { .. } => unreachable!("Nvfp4Raw is MoE-experts-only"),
            W::Bf16(_) => unreachable!(),
        }
        let cfg = GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: outn as i32, n: batch as i32, k: inn as i32,
            alpha: half::bf16::from_f32(1.0), lda: inn as i32,
            ldb: inn as i32, beta: half::bf16::from_f32(0.0), ldc: outn as i32,
        };
        unsafe { self.blas.gemm(cfg, &*buf, x, out).expect("gemm dequant prefill"); }
    }

    /// The single dispatcher every big matmul funnels through.
    ///
    /// For QUANTIZED weights every decode width (1..=MAX_VERIFY) runs the SAME fixed-shape tensor-core
    /// kernel with N padded to 16: it is bound by the packed-weight bytes, which do not change with N,
    /// so verify costs what decode costs and speculation finally pays. Above 16 — i.e. prefill —
    /// weights are read once per many tokens, bandwidth stops being the constraint, and it is cheaper
    /// to dequantize once into scratch and let cuBLAS's tensor cores do the work.
    fn gemm_act(&self, w: &W, x: &B, out: &mut B, inn: usize, outn: usize, batch: usize) {
        const BINV_BATCH: usize = 2;

        match w {
            W::Nvfp4 { qweight, scales, gs, .. } if batch <= MAX_VERIFY => {
                blaunch!(self, "gemm_mma_fp4_b", ((outn / 16) as u32,1,1), (256,1,1), 0,
                    (d(out), *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                     d(gs), d(x), outn as i32, inn as i32, batch as i32, 0u64));
                return;
            }
            W::Fp8 { data, row_scale, .. } if batch <= MAX_VERIFY => {
                blaunch!(self, "gemm_mma_fp8_b", ((outn / 16) as u32,1,1), (256,1,1), 0,
                    (d(out), *data.device_ptr() as u64, d(row_scale),
                     d(x), outn as i32, inn as i32, batch as i32, 0u64));
                return;
            }
            W::Nvfp4 { .. } | W::Fp8 { .. } => {
                self.gemm_quant_prefill(w, x, out, inn, outn, batch);
                return;
            }
            W::Nvfp4Raw { .. } => unreachable!("Nvfp4Raw is MoE-experts-only, not for gemm_act"),
            W::Bf16(_) => {}
        }
        let w = match w { W::Bf16(b) => b, _ => unreachable!() };

        if batch <= BINV_BATCH {
            let smem = (batch * 256 * 4) as u32;
            blaunch!(self, "gemm_binv_b", (outn as u32,1,1), (256,1,1), smem,
                (d(out), d(w), d(x), outn as i32, inn as i32, batch as i32));
            return;
        }
        let cfg = GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: outn as i32, n: batch as i32, k: inn as i32,
            alpha: half::bf16::from_f32(1.0), lda: inn as i32,
            ldb: inn as i32, beta: half::bf16::from_f32(0.0), ldc: outn as i32,
        };
        unsafe { self.blas.gemm(cfg, w, x, out).expect("gemm bf16 act"); }
    }

    /// Batch-invariant bf16 matmul at EVERY serving width: forces `gemm_binv_b` (strided-k +
    /// fixed tree-reduce, N-independent column arithmetic — the same kernel decode uses) instead
    /// of `gemm_act`'s batch>2 cuBLAS arm. cuBLAS picks a different kernel per shape, so a
    /// depth≥3 verify computed ulp-different logits than a decode — which flipped top-8 expert
    /// selections on near-ties and silently broke MoE-MTP losslessness (found 2026-07-20 on the
    /// 122B: single-node d4 MISMATCH at token 25). Quantized weights already route to the
    /// batch-invariant mma kernels via `gemm_act`, so delegate there; only bf16 needs the force.
    /// Used by the MoE router and shared_gate — the bf16 tensors on the decode/verify path.
    fn gemm_act_binv(&self, w: &W, x: &B, out: &mut B, inn: usize, outn: usize, batch: usize) {
        let w = match w {
            W::Bf16(b) if batch <= MAX_VERIFY => b,
            // prefill (batch > MAX_VERIFY) is outside the invariance contract — cuBLAS is fine
            // there; quantized weights are already batch-invariant via the mma path.
            _ => { self.gemm_act(w, x, out, inn, outn, batch); return; }
        };
        let smem = (batch * 256 * 4) as u32;
        blaunch!(self, "gemm_binv_b", (outn as u32,1,1), (256,1,1), smem,
            (d(out), d(w), d(x), outn as i32, inn as i32, batch as i32));
    }

    /// Dispatch a layer's FFN: dense MLP or MoE block. Keeps the four batched forward sites uniform.
    /// `sharded` says whether THIS ffn's weights were split by `tp_shard_weights`. It must not be
    /// inferred from `tp_world`: the MTP head is replicated even under TP=2, so a global flag would
    /// halve `im` on a full-width weight and then all-reduce two identical halves — wrong output, and
    /// wrong timing too. Sharding is a property of the tensor, not of the process.
    fn ffn_batch(&self, pool: &mut Pool, x: &B, ffn: &Ffn, batch: usize, sharded: bool) -> B {
        match ffn {
            Ffn::Dense(m)   => self.mlp_batch(pool, x, m, batch, sharded),
            Ffn::Moe(moe)   => self.moe_batch(pool, x, moe, batch),
        }
    }

    /// MoE FFN forward (qwen3_5_moe), correctness-first bf16 path.
    ///   router → softmax→top-k→renorm → Σ_j w_j·expert_{e_j}(x) + sigmoid(shared_gate·x)·shared(x).
    /// Experts are the STACKED fused weights read directly by `moe_experts_b`. Not yet fused-dequant or
    /// tuned — this is the oracle-correct reference the NVFP4 grouped kernel is validated against.
    fn moe_batch(&self, pool: &mut Pool, x: &B, moe: &GpuMoe, batch: usize) -> B {
        let cfg = &self.cfg;
        let (h, ne, k) = (cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok);
        let (mi, si) = (cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size);

        // TP=2 expert-parallel (only when THIS layer's experts were halved at attach — a property of
        // the tensor, NOT of the process; the MTP MoE stays replicated and barrier-free). The router
        // is replicated, so `moe_router_topk_b` still emits GLOBAL ids; the expert GEMMs take this
        // rank's band [e_base, e_base+e_span) and contribute EXACT ZEROS for remote experts, and one
        // all-reduce on the expert-combined output (below, before the shared add) restores the sum.
        let (e_base, e_span) = if moe.experts_sharded {
            let span = (ne / 2) as i32;
            (self.tp_rank.max(0) as i32 * span, span)
        } else {
            (0, ne as i32)
        };

        // 1. Router logits [ne, batch] → top-k expert ids + renormalized weights.
        let mut logits = pool.get_bf16(ne * batch);
        self.gemm_act_binv(&moe.router, x, &mut logits, h, ne, batch);
        // Router scratch: reuse the preallocated buffers for graph-CAPTURED small batches (decode/verify,
        // batch<=MAX_VERIFY) — a fresh alloc here is illegal during CUDA-graph capture. Prefill
        // (batch>MAX_VERIFY) is not captured, so a fresh alloc is fine and keeps the scratch bounded.
        let owned = (batch > MAX_VERIFY).then(||
            (self.dev.alloc_zeros::<i32>(k*batch).unwrap(), self.dev.alloc_zeros::<f32>(k*batch).unwrap()));
        let (ids_ptr, wts_ptr) = match &owned {
            Some((i, w)) => (*i.device_ptr() as u64, *w.device_ptr() as u64),
            None => (*self.moe_ids.device_ptr() as u64, *self.moe_wts.device_ptr() as u64),
        };
        blaunch!(self, "moe_router_topk_b", (batch as u32,1,1), (256,1,1), (ne*4) as u32,
            (ids_ptr, wts_ptr, d(&logits), ne as i32, k as i32, batch as i32));
        pool.release_bf16(logits, ne * batch);

        // 2. Grouped expert MLP over the stacked fused weights → out [h, batch].
        let mut out = pool.get_bf16(h * batch);
        let smem = ((2*mi + mi + h) * 4) as u32;    // gu[2I] + hh[I] + acc[H]
        match (&moe.gate_up, &moe.down) {
            (W::Bf16(gu), W::Bf16(dn)) => {
                blaunch!(self, "moe_experts_b", (batch as u32,1,1), (256,1,1), smem,
                    (d(&out), d(x), ids_ptr, wts_ptr, d(gu), d(dn),
                     h as i32, mi as i32, k as i32, batch as i32));
            }
            (W::Nvfp4 { qweight: guq, scales: gus, gs: gugs, .. },
             W::Nvfp4 { qweight: dnq, scales: dns, gs: dngs, .. }) => {
              // Token-gather wins only when tokens/expert > 1 (batch·k ≫ num_experts) — i.e. prefill.
              // Decode (1) and verify (≤ MAX_VERIFY) route through the lean per-(token,slot) path, which
              // avoids the counting-sort + per-layer sync that would otherwise dominate a tiny batch.
              if batch < 128 {
                // DECODE/VERIFY (N=1 per pair): no token-sharing at small batch, so no gather needed.
                let bk = batch * k;
                let gu_s = pool.get_bf16(bk * 2 * mi);
                let h_s  = pool.get_bf16(bk * mi);
                let dn_s = pool.get_bf16(bk * h);
                blaunch!(self, "gemm_moe_mma_fp4", (((2*mi/16) as u32), bk as u32, 1), (256,1,1), 0,
                    (d(&gu_s), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(x), ids_ptr,
                     (2*mi) as i32, h as i32, k as i32, 0i32, e_base, e_span));
                blaunch!(self, "moe_silu_bf16_b", grid(bk * mi), (256,1,1), 0,
                    (d(&h_s), d(&gu_s), mi as i32, bk as i32));
                blaunch!(self, "gemm_moe_mma_fp4", (((h/16) as u32), bk as u32, 1), (256,1,1), 0,
                    (d(&dn_s), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(&h_s), ids_ptr,
                     h as i32, mi as i32, k as i32, 1i32, e_base, e_span));
                blaunch!(self, "moe_combine_experts_b", grid(batch * h), (256,1,1), 0,
                    (d(&out), d(&dn_s), wts_ptr, h as i32, k as i32, batch as i32));
                pool.release_bf16(gu_s, bk*2*mi); pool.release_bf16(h_s, bk*mi); pool.release_bf16(dn_s, bk*h);
              } else {
                // TOKEN-GATHER (prefill/verify): counting-sort pairs by expert → ONE weight read/expert.
                let ne = cfg.num_experts;
                let p = batch * k;
                let ppad_max = p + ne * 16;   // experts padded to mult-of-16 (grouped-GEMM weight-reuse fix)
                let count = self.dev.alloc_zeros::<i32>(ne).unwrap();
                blaunch!(self, "moe_count_b", grid(p), (256,1,1), 0,
                    (*count.device_ptr() as u64, ids_ptr, p as i32, e_base, e_base + e_span));
                let poff = self.dev.alloc_zeros::<i32>(ne + 1).unwrap();
                let cursor = self.dev.alloc_zeros::<i32>(ne).unwrap();
                blaunch!(self, "moe_offsets_b", (1u32,1,1), (32,1,1), 0,
                    (*poff.device_ptr() as u64, *cursor.device_ptr() as u64, *count.device_ptr() as u64, ne as i32));
                self.sync_stream();
                let poff_h = self.dev.dtoh_sync_copy(&poff).unwrap();
                let ppad = poff_h[ne] as usize;
                let ngroups = ppad / 16;   // each block covers a 16-token group (weight read once per group)
                let perm_tok = self.dev.htod_sync_copy(&vec![-1i32; ppad_max]).unwrap();
                let perm_wt = self.dev.alloc_zeros::<f32>(ppad_max).unwrap();
                let inv_pos = self.dev.alloc_zeros::<i32>(p).unwrap();
                let tile_e = self.dev.alloc_zeros::<i32>(ppad_max / 8 + 1).unwrap();
                blaunch!(self, "moe_scatter_b", grid(p), (256,1,1), 0,
                    (*perm_tok.device_ptr() as u64, *perm_wt.device_ptr() as u64, *inv_pos.device_ptr() as u64,
                     *cursor.device_ptr() as u64, ids_ptr, wts_ptr, p as i32, k as i32,
                     e_base, e_base + e_span));
                blaunch!(self, "moe_tilemap_b", grid(ne), (256,1,1), 0,
                    (*tile_e.device_ptr() as u64, *poff.device_ptr() as u64, ne as i32));
                let x_perm = pool.get_bf16(ppad * h);
                blaunch!(self, "moe_gather_x_b", grid(ppad * h), (256,1,1), 0,
                    (d(&x_perm), d(x), *perm_tok.device_ptr() as u64, h as i32, ppad as i32));
                let gu_p = pool.get_bf16(ppad * 2 * mi);
                blaunch!(self, "gemm_moe_grouped_mma_fp4", (((2*mi/16) as u32), ngroups as u32, 1), (256,1,1), 0,
                    (d(&gu_p), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&x_perm),
                     *tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base));
                let h_p = pool.get_bf16(ppad * mi);
                blaunch!(self, "moe_silu_bf16_b", grid(ppad * mi), (256,1,1), 0,
                    (d(&h_p), d(&gu_p), mi as i32, ppad as i32));
                let dn_p = pool.get_bf16(ppad * h);
                blaunch!(self, "gemm_moe_grouped_mma_fp4", (((h/16) as u32), ngroups as u32, 1), (256,1,1), 0,
                    (d(&dn_p), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(&h_p),
                     *tile_e.device_ptr() as u64, h as i32, mi as i32, e_base));
                blaunch!(self, "moe_combine_grouped_b", grid(batch * h), (256,1,1), 0,
                    (d(&out), d(&dn_p), *perm_wt.device_ptr() as u64, *inv_pos.device_ptr() as u64,
                     h as i32, k as i32, batch as i32));
                pool.release_bf16(x_perm, ppad*h); pool.release_bf16(gu_p, ppad*2*mi);
                pool.release_bf16(h_p, ppad*mi); pool.release_bf16(dn_p, ppad*h);
              }
            }
            _ => panic!("MoE experts: gate_up and down must both be bf16 or both Nvfp4Raw"),
        }

        // TP=2 expert-parallel: `out` holds only THIS rank's experts (remote slots are exact zeros on
        // both paths — the decode kernel zeroes remote (token,slot) rows; the prefill combine skips
        // inv_pos == -1 pairs), so it is a PARTIAL. ONE all-reduce restores the full Σ. It MUST run
        // here — after the expert combine, BEFORE the shared-expert add — because the shared expert is
        // replicated: reducing after the shared add would double it. Same call shape as mlp_batch's
        // row-parallel epilogue: elementwise, chunked to the ring payload (both ranks compute the same
        // chunk count from the same n), canonical rank0+rank1 order → bit-identical across ranks.
        // bf16 partials only: the combine kernels emit bf16, so there is no unrounded FP32 accumulator
        // to preserve — the exact analog of the dense path's wide-batch bf16 fallback.
        if moe.experts_sharded {
            self.tp_all_reduce_bf16(&mut out, h * batch);
        }

        // 3. Shared expert: standard MLP (intermediate = si), gated by sigmoid(shared_gate·x).
        let mut sg = pool.get_bf16(si * batch);
        let mut su = pool.get_bf16(si * batch);
        self.gemm_act(&moe.shared.gate, x, &mut sg, h, si, batch);
        self.gemm_act(&moe.shared.up,   x, &mut su, h, si, batch);
        blaunch!(self, "silu_mul_b", grid(si*batch), (256,1,1), 0, (d(&sg), d(&sg), d(&su), (si*batch) as i32));
        let mut shared_out = pool.get_bf16(h * batch);
        self.gemm_act(&moe.shared.down, &sg, &mut shared_out, si, h, batch);
        let mut sgate = pool.get_bf16(batch);
        self.gemm_act_binv(&moe.shared_gate, x, &mut sgate, h, 1, batch);
        blaunch!(self, "moe_shared_combine_b", grid(h*batch), (256,1,1), 0,
            (d(&out), d(&shared_out), d(&sgate), h as i32, batch as i32));

        pool.release_bf16(sg, si*batch); pool.release_bf16(su, si*batch);
        pool.release_bf16(shared_out, h*batch); pool.release_bf16(sgate, batch);
        out
    }

    /// Attach the TP=2 data-plane link + this process's rank/world after the cluster sync + RDMA
    /// bring-up, then SHARD the weights so each box streams ~half the bytes/token (the decode speedup).
    /// Decode is weight-bandwidth-bound, so halving bytes-read is the win; the per-reduction all-reduce
    /// (`tp_all_reduce_bf16`) stitches the row-parallel partials back together.
    pub fn attach_tp(&mut self, rank: i32, world: i32, mut link: crate::net::TpLink) {
        self.tp_rank = rank;
        self.tp_world = world;
        if world == 2 {
            let t0 = std::time::Instant::now();
            self.tp_shard_weights(rank as usize);
            eprintln!("[tp] rank {rank}/{world} — weights sharded (FFN col/row-parallel) in {:.1}s", t0.elapsed().as_secs_f32());
            // All reductions are the batch=1 hidden vector, so the payload is one fixed size. Set it
            // BEFORE the proxy starts — both the proxy and K1/K2 read it, and I8 forbids mutating
            // protocol state underneath a running system.
            let fp32 = self.tp_fp32_partials();
            // GB10_TP_BATCH_PROBE=N widens the payload so a batch-N forward can all-reduce
            // hidden*N. Set once, before the proxy starts (I8).
            let nbytes = self.cfg.hidden_size * self.tp_probe_batch() * if fp32 { 4 } else { 2 };
            // Serving mode sets batch_probe = max_batch, so this is also the batched-decode reduce
            // payload. The slot size is baked into captured CUDA graphs — an oversize payload cannot
            // be patched later, so refuse LOUDLY at attach rather than corrupt a ring at runtime.
            // (bf16 hidden=5120 fits max_batch <= 12; fp32 partials halve that.)
            assert!(nbytes <= crate::tp::TP_SLOT_BYTES,
                    "FATAL: TP all-reduce payload {nbytes} B exceeds TP_SLOT_BYTES ({} B) — lower \
                     --max-batch{} (hidden {}, probe batch {})",
                    crate::tp::TP_SLOT_BYTES,
                    if fp32 { " or disable FP32 partials" } else { "" },
                    self.cfg.hidden_size, self.tp_probe_batch());
            link.set_payload(nbytes, fp32).expect("net_set_payload");
            if fp32 { eprintln!("[tp] FP32-PRESERVING partials ({nbytes} B/reduction, one bf16 round in K2)"); }
            self.tp_ctx_dptr = link.ctx_device_ptr();
            // Hand the RDMA ctx to the persistent proxy thread. The proxy OWNS the transport from here,
            // so we `mem::forget` the TpLink (its Drop would net_shutdown the ctx out from under the
            // proxy); the OS reclaims at exit. Core 19 is a big X925 and pairs with the launch thread on
            // core 9 — pinning was worth 9.0→15.1 tok/s on 27B, so a failure to pin is loud.
            // GB10_TP_TRACE=1 turns on the same per-barrier timestamping the microbench uses, so a slow
            // model run can be decomposed against the bench's floor instead of guessed at.
            if std::env::var("GB10_TP_TRACE").is_ok()
                || crate::tp::tp_config().map(|c| c.trace).unwrap_or(false) {
                crate::net::trace_enable(&mut link);
                eprintln!("[tp] per-barrier tracing ON (GB10_TP_TRACE)");
            }
            if self.tp_head_proof() {
                self.head_visits = Some(self.dev.alloc_zeros::<u64>(self.cfg.lin_num_v_heads).unwrap());
            }
            let ctx_addr = link.ctx_addr();
            std::mem::forget(link);
            crate::net::spawn_proxy(ctx_addr, 19);
            eprintln!("[tp] rank {rank}/{world} — RDMA proxy thread up (doorbell ring R={}, {nbytes} B/reduction)", 8);
        }
    }

    /// Split the FFN weights for TP=2 (called once, at attach). gate/up are COLUMN-parallel (each rank
    /// owns half the output rows → a contiguous tile slice of the packed NVFP4); down is ROW-parallel
    /// (each rank owns half of K → a per-output-tile repack of the k-blocks; its output is a partial
    /// summed by the all-reduce). No kernel change — the sharded weight is just smaller, so the proven
    /// `gemm_mma_fp4_b` reads fewer bytes. Also halves FFN weight MEMORY. (attn/GDN sharding lands next.)
    fn tp_shard_weights(&mut self, rank: usize) {
        let shard_mixers = self.tp_shard_mixers();
        let (nh, nkv, hd) = (self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim);
        // Fused attention qkv output segments (per split_qkv_b): [qg | k | v]. Each is head-split.
        let qkv_segs = [(0, nh * hd * 2, true), (nh * hd * 2, nkv * hd, true), (nh * hd * 2 + nkv * hd, nkv * hd, true)];
        // Fused GDN in_proj output segments (per split_gdn_b / GdnIn::fused_m):
        //   [ qkv (= q key_dim | k key_dim | v value_dim) | z value_dim | b n_v | a n_v ]
        // Six independently head-split segments. q/k split on KEY heads, v/z/b/a on VALUE heads — the
        // 48/16 = 3 value-heads-per-key-head grouping survives because both splits are contiguous.
        let (lin_nk, lin_nv) = (self.cfg.lin_num_k_heads, self.cfg.lin_num_v_heads);
        let (kdim, vdim) = (self.cfg.key_dim(), self.cfg.value_dim());
        let (cdim, ck) = (kdim * 2 + vdim, self.cfg.conv_kernel);
        // b/a (one row per value head) are carried WHOLE (split=false): 48 rows will not halve at
        // NVFP4's 16-row tile granularity, so both ranks keep all of them and split_gdn_b slices them
        // to the local head range. They must still be PRESENT in the weight or the fused GEMM's M and
        // the tensor disagree — which is exactly the bug this flag was added to fix.
        let gdn_segs = [(0, kdim, true), (kdim, kdim, true), (2 * kdim, vdim, true), (cdim, vdim, true),
                        (cdim + vdim, lin_nv, false), (cdim + vdim + lin_nv, lin_nv, false)];
        // conv1d is [conv_dim][conv_kernel] — slice the same q/k/v channel segments.
        let conv_segs = [(0, kdim), (kdim, kdim), (2 * kdim, vdim)];
        let _ = lin_nk;
        let mut layers = std::mem::take(&mut self.layers);
        for layer in &mut layers {
            // FFN: gate/up column-parallel, down row-parallel.
            if let Ffn::Dense(mlp) = &mut layer.mlp {
                mlp.gate = self.shard_nvfp4_col(&mlp.gate, rank);
                mlp.up   = self.shard_nvfp4_col(&mlp.up, rank);
                mlp.down = self.shard_nvfp4_row(&mlp.down, rank);
            }
            // MoE EXPERT sharding (122B TP=2), under the SAME flag as the mixers (audit option (b):
            // mixers AND experts together; default-off). The stacked experts are expert-major along
            // M — expert e's 16-row NVFP4 tiles sit at e*(M/16) (gpu_batch.cu gemm_moe_mma_fp4) — and
            // each expert's M (2·mi gate_up rows, h down rows) is a multiple of 16, so a plain
            // COLUMN-parallel shard of the flattened [ne·M_e, K] weight IS the contiguous expert band
            // [rank·ne/2, rank·ne/2 + ne/2): one byte-slice of qweight/scales/gs, no repack, one
            // uniform gs per (expert,tile) preserved. router/shared/shared_gate stay replicated.
            // Semantics differ from the dense col-shard, though: each rank now computes DIFFERENT
            // experts (not half of every output), so `moe_batch` remotes-zero the other half and ONE
            // all-reduce on the expert-combined output restores the full sum.
            // The MTP layer's MoE is deliberately NOT sharded (tp_shard_weights touches only
            // self.layers): the draft head must stay barrier-free — an all-reduce inside drafting
            // would serialize the draft/verify pipeline — and ~0.7 GB/rank is noise.
            if shard_mixers {
                if let Ffn::Moe(moe) = &mut layer.mlp {
                    moe.gate_up = self.shard_nvfp4_col(&moe.gate_up, rank);
                    moe.down    = self.shard_nvfp4_col(&moe.down, rank);
                    moe.experts_sharded = true;
                }
            }
            // Attention (mixer-sharding, flag-gated): fused qkv column-parallel (per-head, 3 segments),
            // o_proj row-parallel. q_norm/k_norm are shared [head_dim] (rmsnorm_perhead reads w[tid]) → NOT sharded.
            if shard_mixers {
                if let Some(fa) = &mut layer.fa {
                    if let AttnIn::Fused(w) = &fa.qkv {
                        fa.qkv = AttnIn::Fused(self.shard_col_segs(w, &qkv_segs, rank));
                    }
                    fa.o_proj = self.shard_row(&fa.o_proj, rank);
                }
                // GDN (48 of 64 layers, 21 % of decode weight bytes — the largest block still replicated).
                // Head-parallel: in_proj column-parallel per segment, conv/a_log/dt_bias sliced to the
                // local heads, out_proj row-parallel → partial → all-reduce. `norm` is [v_head_dim],
                // shared across heads (rmsnorm_gated_b indexes it by lane), so it is NOT sharded.
                if let Some(la) = &mut layer.la {
                    if let GdnIn::Fused(w) = &la.in_proj {
                        la.in_proj = GdnIn::Fused(self.shard_col_segs(w, &gdn_segs, rank));
                    } else {
                        panic!("tp GDN shard: expected a fused (quantized) in_proj; the Split path is unsharded");
                    }
                    la.conv1d  = self.shard_f32_segs(&la.conv1d, &conv_segs, ck, rank);
                    la.a_log   = self.shard_f32_segs(&la.a_log, &[(0, lin_nv)], 1, rank);
                    la.dt_bias = self.shard_f32_segs(&la.dt_bias, &[(0, lin_nv)], 1, rank);
                    if self.tp_head_proof() {
                        // Proof A: our convention is slice+local, so a GLOBAL index (head + h0) would
                        // read past the local range. Pad to full width with NaN so that read is loud.
                        la.a_log   = self.pad_nan_redzone(&la.a_log, lin_nv);
                        la.dt_bias = self.pad_nan_redzone(&la.dt_bias, lin_nv);
                    }
                    la.out_proj = self.shard_row(&la.out_proj, rank);
                }
            }
        }
        self.layers = layers;
    }

    /// Column-parallel split of a packed NVFP4 weight [M,K] → rank-local [M/2, K]. The kernel packs
    /// output tiles of 16 rows contiguously (`Wt32[(mt*nblk+kb)*32+lane]`, `gs[mt]`), so rank r owns the
    /// contiguous tile band [r·M/2/16, …) — a straight byte-slice of qweight/scales/gs. Byte-exact.
    fn shard_nvfp4_col(&self, w: &W, rank: usize) -> W {
        let m = match w { W::Nvfp4 { m, .. } => *m, _ => panic!("tp col-shard: expected NVFP4") };
        self.shard_nvfp4_col_segs(w, &[(0, m, true)], rank)   // whole-M single segment (FFN gate/up)
    }

    /// Column-parallel split over one or more output-row SEGMENTS. Each `(row_off, row_count)` segment is
    /// independently halved by rank and the node's halves are concatenated → rank-local weight. One
    /// segment = a plain col-shard (FFN gate/up). Three segments = the fused attention qkv, whose output
    /// is `[qg (nh·hd·2) | k (nkv·hd) | v (nkv·hd)]` and must be split per head WITHIN each segment (so
    /// the node gets its q/gate heads AND its kv heads). Every segment offset is 16-tile-aligned and each
    /// half is 16-tile-aligned (row_count % 32 == 0), so this is an exact byte gather. gs is per output
    /// tile → sliced with the tiles.
    /// Each segment is `(row_off, row_count, split)`. `split=false` copies the segment WHOLE to both
    /// ranks — needed for tensors whose rows are one-per-head but too few to halve at NVFP4's 16-row
    /// tile granularity (GDN `b`/`a`: 48 rows, 3 tiles, and 24 is not a multiple of 16). Such a segment
    /// only has to be tile-aligned, not halvable.
    fn shard_nvfp4_col_segs(&self, w: &W, segs: &[(usize, usize, bool)], rank: usize) -> W {
        let (qw, sc, gs, _m, k) = match w {
            W::Nvfp4 { qweight, scales, gs, m, k } => (qweight, scales, gs, *m, *k),
            _ => panic!("tp col-shard: expected NVFP4, got another format"),
        };
        let nblk = k / 16;
        let qpt = nblk * 128;            // qweight bytes per output tile (nblk k-blocks × 32 u32)
        let spt = nblk * 16;             // scale bytes per output tile
        let qh = self.dev.dtoh_sync_copy(qw).unwrap();
        let sh = self.dev.dtoh_sync_copy(sc).unwrap();
        let gh = self.dev.dtoh_sync_copy(gs).unwrap();
        let (mut q_new, mut s_new, mut g_new) = (Vec::new(), Vec::new(), Vec::new());
        let mut m_local = 0usize;
        for &(off, cnt, split) in segs {
            assert!(off % 16 == 0, "tp col-shard seg (off {off}) not tile-aligned");
            let take = if split { cnt / 2 } else { cnt };
            if split { assert!(cnt % 32 == 0, "tp col-shard seg (off {off}, cnt {cnt}) won't halve at 16-row tiles"); }
            else     { assert!(cnt % 16 == 0, "tp col-shard unsplit seg (off {off}, cnt {cnt}) not tile-aligned"); }
            let tile_lo = (off + if split { rank * take } else { 0 }) / 16;
            let ntl = take / 16;
            q_new.extend_from_slice(&qh[tile_lo * qpt..(tile_lo + ntl) * qpt]);
            s_new.extend_from_slice(&sh[tile_lo * spt..(tile_lo + ntl) * spt]);
            g_new.extend_from_slice(&gh[tile_lo..tile_lo + ntl]);
            m_local += take;
        }
        W::Nvfp4 {
            qweight: self.dev.htod_sync_copy(&q_new).unwrap(),
            scales: self.dev.htod_sync_copy(&s_new).unwrap(),
            gs: self.dev.htod_sync_copy(&g_new).unwrap(),
            m: m_local, k,
        }
    }

    /// Row-parallel split of a packed NVFP4 weight [M,K] → rank-local [M, K/2]. K is chunked into
    /// 16-wide k-blocks per output tile; rank r owns k-blocks [r·nblk/2, …) of EVERY output tile, so the
    /// packed bytes are gathered per tile (strided, one contiguous chunk per tile). gs is per output
    /// tile (K-independent) → copied unchanged: with the same gs on both ranks, (v0·gs)+(v1·gs) =
    /// (v0+v1)·gs, so applying gs to each partial then summing is exact. Byte-exact per tile.
    /// Column-parallel split of an FP32 device tensor laid out as `[rows][row_stride]`, over one or more
    /// row SEGMENTS. Same contiguous per-segment halving as `shard_nvfp4_col_segs`, for the GDN tensors
    /// that are not quantized: `conv1d` ([conv_dim][k], sliced into its q/k/v channel segments) and
    /// `a_log` / `dt_bias` ([n_v_heads], row_stride 1).
    fn shard_f32_segs(&self, s: &S, segs: &[(usize, usize)], row_stride: usize, rank: usize) -> S {
        let host = self.dev.dtoh_sync_copy(s).expect("dtoh gdn f32");
        let mut out: Vec<f32> = Vec::new();
        for &(off, rows) in segs {
            assert!(rows % 2 == 0, "gdn f32 shard: segment of {rows} rows won't halve");
            let half = rows / 2;
            let base = (off + rank * half) * row_stride;
            out.extend_from_slice(&host[base..base + half * row_stride]);
        }
        self.dev.htod_sync_copy(&out).expect("htod gdn f32")
    }

    /// Grow an FP32 vector to `full` entries, filling everything above its current length with NaN.
    /// Used only by the head-execution proof (see `tp_head_proof`).
    fn pad_nan_redzone(&self, s: &S, full: usize) -> S {
        let mut h = self.dev.dtoh_sync_copy(s).expect("dtoh redzone");
        assert!(h.len() <= full);
        h.resize(full, f32::NAN);
        self.dev.htod_sync_copy(&h).expect("htod redzone")
    }

    /// Proof B verification: every GDN state red zone must still be entirely NaN. Any real number in
    /// there is a kernel that ran at the FULL head count and wrote outside this rank's heads — the
    /// gate-invisible, timing-invisible failure mode. Returns (checked, violations).
    pub fn verify_head_redzones(&self, state: &BatchGpuState) -> (usize, usize) {
        if !self.tp_head_proof() { return (0, 0); }
        let cfg = &self.cfg;
        let (mut checked, mut bad) = (0usize, 0usize);
        // Proof D: every LOCAL head must have been visited, and no head above the local range at all.
        // A `blk % nh` aliasing fold shows up as unequal counts across local heads.
        if let Some(v) = &self.head_visits {
            let h = self.dev.dtoh_sync_copy(v).expect("dtoh visits");
            let nloc = self.eff_lin_v_heads();
            let local: Vec<u64> = h[..nloc].to_vec();
            let beyond: u64 = h[nloc..].iter().sum();
            let uniform = local.iter().all(|c| *c == local[0]) && local[0] > 0;
            checked += 1;
            if !uniform || beyond > 0 {
                bad += 1;
                eprintln!("[head-proof] VISIT VIOLATION: local head counts {:?} (want all equal, >0), \
                           visits to heads >= {nloc}: {beyond}", &local[..local.len().min(8)]);
            } else {
                eprintln!("[head-proof] visits: all {nloc} local heads visited {} times, 0 beyond — \
                           no aliasing fold", local[0]);
            }
        }
        for (li, lt) in cfg.layer_types.iter().enumerate() {
            if !matches!(lt, crate::qwen::LayerType::LinearAttention) { continue; }
            let conv_n = self.eff_conv_dim() * cfg.conv_kernel;
            let s_n = self.eff_lin_v_heads() * cfg.lin_k_dim * cfg.lin_v_dim;
            for (buf, local_n, what) in [(&state.conv_state[li], conv_n, "conv"),
                                         (&state.s_state[li], s_n, "recurrent")] {
                let Some(b) = buf else { continue };
                let h = self.dev.dtoh_sync_copy(b).expect("dtoh redzone check");
                if h.len() < local_n * 2 { continue; }
                let violations = h[local_n..].iter().filter(|x| x.to_bits() != TP_REDZONE_SENTINEL).count();
                checked += 1;
                if violations > 0 {
                    bad += 1;
                    eprintln!("[head-proof] VIOLATION layer {li} {what} state: {violations} of {} \
                               red-zone floats were WRITTEN — a kernel ran at full head count",
                              h.len() - local_n);
                }
            }
        }
        (checked, bad)
    }

    fn shard_nvfp4_row(&self, w: &W, rank: usize) -> W {
        let (qw, sc, gs, m, k) = match w {
            W::Nvfp4 { qweight, scales, gs, m, k } => (qweight, scales, gs, *m, *k),
            _ => panic!("tp row-shard: expected NVFP4 (dense FFN down), got another format"),
        };
        assert!((k / 16) % 2 == 0 && (k / 2) % 32 == 0, "tp row-shard: k={k} won't split into 32-aligned halves");
        let k_local = k / 2;
        let nblk_full = k / 16;
        let nblk_local = k_local / 16;
        let tiles = m / 16;
        let kb_lo = rank * nblk_local;
        let qh = self.dev.dtoh_sync_copy(qw).unwrap();
        let sh = self.dev.dtoh_sync_copy(sc).unwrap();
        let gh = self.dev.dtoh_sync_copy(gs).unwrap();
        let mut q_new = Vec::with_capacity(tiles * nblk_local * 128);
        let mut s_new = Vec::with_capacity(tiles * nblk_local * 16);
        for mt in 0..tiles {
            let q_src = (mt * nblk_full + kb_lo) * 128;
            q_new.extend_from_slice(&qh[q_src..q_src + nblk_local * 128]);
            let s_src = (mt * nblk_full + kb_lo) * 16;
            s_new.extend_from_slice(&sh[s_src..s_src + nblk_local * 16]);
        }
        W::Nvfp4 {
            qweight: self.dev.htod_sync_copy(&q_new).unwrap(),
            scales: self.dev.htod_sync_copy(&s_new).unwrap(),
            gs: self.dev.htod_sync_copy(&gh).unwrap(),
            m, k: k_local,
        }
    }

    /// Mixer-shard dispatch by weight format. The -mixed recipes (0.8b/2b/4b/122B-mixed) keep the GDN
    /// (and sometimes attn) mixer tensors in FP8 while the FFN is NVFP4, so the four mixer shard sites
    /// must accept both. The geometry (segments, rank, 16-row tiles) is format-independent.
    fn shard_col_segs(&self, w: &W, segs: &[(usize, usize, bool)], rank: usize) -> W {
        match w {
            W::Nvfp4 { .. } => self.shard_nvfp4_col_segs(w, segs, rank),
            W::Fp8 { .. }   => self.shard_fp8_col_segs(w, segs, rank),
            _ => panic!("tp col-shard: expected NVFP4 or FP8, got another format"),
        }
    }
    fn shard_row(&self, w: &W, rank: usize) -> W {
        match w {
            W::Nvfp4 { .. } => self.shard_nvfp4_row(w, rank),
            W::Fp8 { .. }   => self.shard_fp8_row(w, rank),
            _ => panic!("tp row-shard: expected NVFP4 or FP8, got another format"),
        }
    }

    /// Column-parallel split over output-row SEGMENTS of a packed FP8 weight — the FP8 twin of
    /// `shard_nvfp4_col_segs`. The data is MMA-repacked (`repack_fp8_mma`: [m/16 tiles][nblk
    /// k-blocks][256 B]), so the same contiguous 16-row tile-band halving applies — the 16-row
    /// alignment is a PACKED-layout requirement, hence the same segment asserts as NVFP4 even though
    /// FP8 itself needs no row alignment. `row_scale` is one f32 per OUTPUT ROW (constant over K,
    /// folded into the f32 accumulator at the epilogue) → sliced with the rows, where NVFP4's gs is
    /// sliced per tile.
    fn shard_fp8_col_segs(&self, w: &W, segs: &[(usize, usize, bool)], rank: usize) -> W {
        let (qw, rs, _m, k) = match w {
            W::Fp8 { data, row_scale, m, k } => (data, row_scale, *m, *k),
            _ => panic!("tp col-shard: expected FP8, got another format"),
        };
        let nblk = k / 16;
        let qpt = nblk * 256;            // fp8 bytes per output tile (nblk k-blocks × 16×16 B)
        let qh = self.dev.dtoh_sync_copy(qw).unwrap();
        let rh = self.dev.dtoh_sync_copy(rs).unwrap();
        let (mut q_new, mut r_new) = (Vec::new(), Vec::new());
        let mut m_local = 0usize;
        for &(off, cnt, split) in segs {
            assert!(off % 16 == 0, "tp fp8 col-shard seg (off {off}) not tile-aligned");
            let take = if split { cnt / 2 } else { cnt };
            if split { assert!(cnt % 32 == 0, "tp fp8 col-shard seg (off {off}, cnt {cnt}) won't halve at 16-row tiles"); }
            else     { assert!(cnt % 16 == 0, "tp fp8 col-shard unsplit seg (off {off}, cnt {cnt}) not tile-aligned"); }
            let row_lo = off + if split { rank * take } else { 0 };
            let tile_lo = row_lo / 16;
            let ntl = take / 16;
            q_new.extend_from_slice(&qh[tile_lo * qpt..(tile_lo + ntl) * qpt]);
            r_new.extend_from_slice(&rh[row_lo..row_lo + take]);
            m_local += take;
        }
        W::Fp8 {
            data: self.dev.htod_sync_copy(&q_new).unwrap(),
            row_scale: self.dev.htod_sync_copy(&r_new).unwrap(),
            m: m_local, k,
        }
    }

    /// Row-parallel split of a packed FP8 weight [M,K] → rank-local [M, K/2]. Same per-output-tile
    /// k-block gather as `shard_nvfp4_row`, with 256 B k-blocks. `gemm_mma_fp8_b` reads k-blocks
    /// SINGLY (no adjacent-pair loop like fp4's), so halves only need 16-alignment, not 32.
    /// `row_scale` is per output row (K-independent) → copied WHOLE on both ranks: with the same rs
    /// on both, (v0·rs)+(v1·rs) = (v0+v1)·rs, so applying rs to each partial then all-reducing is
    /// exact under the accepted fp-reassociation class — the NVFP4 row shard's gs argument verbatim.
    fn shard_fp8_row(&self, w: &W, rank: usize) -> W {
        let (qw, rs, m, k) = match w {
            W::Fp8 { data, row_scale, m, k } => (data, row_scale, *m, *k),
            _ => panic!("tp row-shard: expected FP8, got another format"),
        };
        assert!((k / 16) % 2 == 0, "tp fp8 row-shard: k={k} won't split into 16-aligned halves");
        let k_local = k / 2;
        let nblk_full = k / 16;
        let nblk_local = k_local / 16;
        let tiles = m / 16;
        let kb_lo = rank * nblk_local;
        let qh = self.dev.dtoh_sync_copy(qw).unwrap();
        let rh = self.dev.dtoh_sync_copy(rs).unwrap();
        let mut q_new = Vec::with_capacity(tiles * nblk_local * 256);
        for mt in 0..tiles {
            let q_src = (mt * nblk_full + kb_lo) * 256;
            q_new.extend_from_slice(&qh[q_src..q_src + nblk_local * 256]);
        }
        W::Fp8 {
            data: self.dev.htod_sync_copy(&q_new).unwrap(),
            row_scale: self.dev.htod_sync_copy(&rh).unwrap(),   // whole: per-row, K-independent
            m, k: k_local,
        }
    }

    pub fn tp_world(&self) -> i32 { self.tp_world }
    pub fn tp_rank(&self) -> i32 { self.tp_rank }

    /// Whether to shard the attn/GDN MIXERS (not just the FFN). Correct but MEASURED NET-NEGATIVE on 27B
    /// (attn-sharded 15.1→4.4 tok/s): the extra per-layer all-reduce barriers hit a comm-throughput wall
    /// in the eager single-proxy design — at 64 FFN barriers comm is free, but denser barriers (2/attn
    /// layer) stall the GPU on the pinned proxy. So it's OFF by default (the fast 1.29× FFN-only path);
    /// the code is kept, flag-gated, for the graph-captured-barrier rework that would make it pay.
    /// `GB10_TP_SHARD_MIXERS=1` to enable. See TP_M4_NOTES.md.
    fn tp_shard_mixers(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        self.tp_world == 2 && *ON.get_or_init(|| std::env::var("GB10_TP_SHARD_MIXERS").is_ok()
            || crate::tp::tp_config().map(|c| c.shard_mixers).unwrap_or(false))
    }

    /// Effective (TP-local) head counts: each box owns half the heads when mixer-sharding is on. Used by
    /// the sharded attn mixer + KV-cache sizing so weights, state, and kernel launches all agree.
    fn eff_num_heads(&self) -> usize { self.cfg.num_heads / if self.tp_shard_mixers() { 2 } else { 1 } }
    fn eff_num_kv_heads(&self) -> usize { self.cfg.num_kv_heads / if self.tp_shard_mixers() { 2 } else { 1 } }

    /// TP-local GDN geometry. The 48 GDN layers are 21 % of decode weight bytes — the single largest
    /// block that was still replicated on both boxes — and head-parallel is the correct split for a
    /// linear-attention layer: the delta rule is independent per value head, conv is per channel, and the
    /// recurrent state is per head, so the state halves with the heads and never crosses the wire. Only
    /// `out_proj` becomes a partial and needs the all-reduce.
    ///
    /// The value→key head map inside `delta_step_b` is `head * n_k_heads / nh`, so a CONTIGUOUS split
    /// keeps each key head's group of value heads intact (48/16 = 3 per group → 24/8 = 3 per group).
    ///
    /// DANGER, and it is why these accessors exist rather than open-coded halving: if any per-head kernel
    /// is left launched with the FULL head count, the output is still CORRECT — each rank's `out_proj`
    /// only consumes its own heads — so the token-identical gate cannot see it. It just silently does 2×
    /// the work. Every GDN dimension on a sharded path must come from here.
    fn gdn_split(&self) -> usize { if self.tp_shard_mixers() { 2 } else { 1 } }
    fn eff_lin_v_heads(&self) -> usize { self.cfg.lin_num_v_heads / self.gdn_split() }
    fn eff_lin_k_heads(&self) -> usize { self.cfg.lin_num_k_heads / self.gdn_split() }
    fn eff_key_dim(&self) -> usize { self.cfg.lin_k_dim * self.eff_lin_k_heads() }
    fn eff_value_dim(&self) -> usize { self.cfg.lin_v_dim * self.eff_lin_v_heads() }
    fn eff_conv_dim(&self) -> usize { self.eff_key_dim() * 2 + self.eff_value_dim() }
    /// M of the (possibly sharded) fused GDN in_proj. NOTE the b/a segments keep the FULL head count:
    /// they are one row per value head and NVFP4 tiles output rows in 16s, so 48 rows will not halve at
    /// tile granularity. They are 0.6 % of in_proj, so they stay replicated and `split_gdn_b` slices
    /// them to the local head range instead.
    fn eff_gdn_fused_m(&self) -> usize {
        self.eff_conv_dim() + self.eff_value_dim() + 2 * self.cfg.lin_num_v_heads
    }
    /// First value head owned by this rank (contiguous split).
    fn gdn_head0(&self) -> usize {
        if self.tp_shard_mixers() { self.tp_rank.max(0) as usize * self.eff_lin_v_heads() } else { 0 }
    }
    /// The GDN paths that are NOT wired for sharding (MTP draft/verify, checkpoint probes). Reaching one
    /// while mixer-sharded would read full-width weights with local dims — wrong results, not slow ones.
    /// Fail loudly instead.
    /// `GB10_TP_HEAD_PROOF=1` — positive proof that the GDN kernels execute at LOCAL head count.
    ///
    /// This exists because the failure mode is invisible to everything else we have. A per-head kernel
    /// left launched at the full head count stays CORRECT (each rank's `out_proj` only consumes its own
    /// heads), so the token-identical gate cannot see it — it just silently does 2× the work. And it
    /// cannot be caught by TIMING either: on a ~48-SM part a 24-head and a 48-head state kernel both run
    /// in a single wave, so the wasted work does not even show up as latency.
    ///
    /// So we prove it positively, by making the wrong answer LOUD:
    ///   A. parameter red-zone — `a_log`/`dt_bias` are padded to FULL width with NaN above the local
    ///      range. Our convention is slice+local, so a global index (`head + h0`) lands in the NaN.
    ///   B. state red-zone — conv and recurrent state are allocated at FULL head shape with NaN above
    ///      the local range. A full-head-count launch writes into it; we check it is pristine afterwards.
    ///   C. launch telemetry — grid geometry of the per-head kernels asserted once against local dims.
    fn tp_head_proof(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        self.tp_shard_mixers() && *ON.get_or_init(|| std::env::var("GB10_TP_HEAD_PROOF").is_ok())
    }

    fn assert_gdn_unsharded(&self, who: &str) {
        assert!(!self.tp_shard_mixers(),
                "{who} uses full-width GDN geometry but the mixers are sharded (GB10_TP_SHARD_MIXERS). \
                 This path is not TP-aware; wire it through the eff_* accessors before using it.");
    }

    /// TP=2 sum-all-reduce of a bf16 device buffer across the two boxes — SPIN-WAIT comm (no main-thread
    /// sync). TWO kernels queued on the decode stream, then the main thread races on:
    ///   K1 `tp_gate_copy_signal`: reuse gate → copy local partial into send[e%R] → publish gpu_ready=e
    ///   K2 `tp_wait_add`        : wait cpu_done ≥ e → buf = rank0_partial + rank1_partial
    /// The persistent proxy thread watches gpu_ready, ships payload+inline epoch to the peer, and bounces
    /// the peer's arrival into cpu_done to release K2. So the GPU only stalls for the actual (µs) comm,
    /// not the (ms) per-layer launch/sync bubble — the CPU stays arbitrarily far ahead.
    ///
    /// The epoch is NOT a kernel argument: it lives in the device-side ctx counter and K1 increments it.
    /// That is the round-3 capture-hygiene rule, and it is the whole reason graph capture is a no-op
    /// wrap rather than a protocol rewrite. Both ranks call this symmetrically at every reduction →
    /// residual streams identical → greedy SPMD, no token broadcast.
    /// Elements per barrier for the active payload — the chunk size for a reduction too large for one
    /// ring slot.
    fn tp_chunk_elems(&self, elem_bytes: usize) -> usize {
        (self.cfg.hidden_size * self.tp_probe_batch() * if self.tp_fp32_partials() { 4 } else { 2 })
            / elem_bytes
    }

    fn tp_all_reduce_bf16(&self, buf: &mut B, n: usize) {
        let buf_ptr = *buf.device_ptr();
        // One block on purpose: the flag spin must have a single spinner, and the copy/add is
        // bandwidth-trivial — keeping the wait kernel low-occupancy reserves ~no SM capacity (R3d).
        // CHUNKED: prefill reduces hidden * prompt_len, which is unbounded and can exceed a ring slot by
        // orders of magnitude. Split into slot-sized barriers. Both ranks compute the same n, so the
        // barrier COUNT matches by construction — which is what the protocol requires.
        let chunk = self.tp_chunk_elems(2).max(1);
        let mut off = 0usize;
        while off < n {
            let c = (n - off).min(chunk);
            let p = buf_ptr + (off * 2) as u64;
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (512,1,1), 0, (self.tp_ctx_dptr, p, (c * 2) as u32));
            blaunch!(self, "tp_wait_add", (1u32,1,1), (512,1,1), 0, (self.tp_ctx_dptr, p, p, c as i32, 0i32));
            off += c;
        }
    }

    /// Row-parallel GEMM that leaves its output as the UNROUNDED FP32 accumulator. Only the NVFP4 MMA
    /// path supports it — that is the production 27B weight format for every row-parallel site — and any
    /// other format is refused rather than silently falling back to a bf16 round, which would defeat the
    /// entire point of the FP32-preserving reduction.
    fn gemm_act_f32(&self, w: &W, x: &B, out: &mut S, inn: usize, outn: usize, batch: usize) {
        match w {
            W::Nvfp4 { qweight, scales, gs, .. } if batch <= MAX_VERIFY => {
                blaunch!(self, "gemm_mma_fp4_b", ((outn / 16) as u32,1,1), (256,1,1), 0,
                    (0u64, *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                     d(gs), d(x), outn as i32, inn as i32, batch as i32, d(out)));
            }
            // The -mixed recipes hold the GDN out_proj in FP8: the row-parallel partial must be
            // FP32-preserving there too, or GB10_TP_FP32_PARTIALS is silently a no-op on exactly
            // the layers that dominate a hybrid model. Same epilogue Cf path as fp4.
            W::Fp8 { data, row_scale, .. } if batch <= MAX_VERIFY => {
                blaunch!(self, "gemm_mma_fp8_b", ((outn / 16) as u32,1,1), (256,1,1), 0,
                    (0u64, *data.device_ptr() as u64, d(row_scale),
                     d(x), outn as i32, inn as i32, batch as i32, d(out)));
            }
            _ => panic!("FP32-preserving TP partials require NVFP4 or FP8 row-parallel weights at batch<={};                        got another format. Disable GB10_TP_FP32_PARTIALS for this model.", MAX_VERIFY),
        }
    }

    /// FP32-preserving all-reduce: ship the UNROUNDED partial, sum in FP32 on both ranks, round to bf16
    /// exactly once. `partial` is the GEMM's FP32 accumulator; `out` is the bf16 buffer the residual
    /// consumes. Same two kernels and the same ring as the bf16 path — only the payload width changes,
    /// which is why the slots were sized for FP32 from day one.
    fn tp_all_reduce_fp32(&self, out: &mut B, partial: &S, n: usize) {
        let (o, p) = (*out.device_ptr(), *partial.device_ptr());
        let chunk = self.tp_chunk_elems(4).max(1);
        let mut off = 0usize;
        while off < n {
            let c = (n - off).min(chunk);
            let (oc, pc) = (o + (off * 2) as u64, p + (off * 4) as u64);
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (512,1,1), 0, (self.tp_ctx_dptr, pc, (c * 4) as u32));
            blaunch!(self, "tp_wait_add", (1u32,1,1), (512,1,1), 0, (self.tp_ctx_dptr, oc, pc, c as i32, 1i32));
            off += c;
        }
    }

    /// Batch the TP link is sized for. 1 in production; `GB10_TP_BATCH_PROBE=N` widens it so we can
    /// measure how a batch-N forward behaves under sharding (the MTP verify cost shape).
    fn tp_probe_batch(&self) -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| std::env::var("GB10_TP_BATCH_PROBE").ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| crate::tp::tp_config().and_then(|c| c.batch_probe))
            .filter(|n| *n >= 1 && *n <= MAX_VERIFY).unwrap_or(1))
    }

    /// Q6 PROBE — does "a batch-N forward costs ≈ a batch-1 forward" survive TP?
    ///
    /// This is the whole mechanism of speculative decoding: a batch-N verify reads the weights ONCE and
    /// produces N drafted positions. Single-node it holds tightly (N=8 costs +4.4 % over N=1). If it does
    /// NOT hold under TP — because the all-reduce payload widens with N, or the sharded GEMM shapes
    /// degrade at N columns — then MTP-under-TP cannot pay and Stage 3A stops.
    ///
    /// It measures the COST SHAPE only: identical kernel shapes and a hidden*N all-reduce per site. It is
    /// not verify semantics, and the tokens it produces are meaningless.
    pub fn tp_batch_probe(&self, batch: usize, iters: usize, max_seq_len: usize) {
        let mut pool = Pool::new(self.dev.clone());
        let mut state = self.new_batch_state(batch.max(1), batch.max(1), max_seq_len);
        let mut bufs = self.new_decode_buffers(batch);
        let toks: Vec<i32> = (0..batch).map(|i| (1000 + i) as i32).collect();
        let pos: Vec<i32> = (0..batch).map(|_| 8i32).collect();
        let slots: Vec<i32> = (0..batch).map(|i| i as i32).collect();
        self.dev.htod_sync_copy_into(&toks, &mut bufs.tokens_dev).unwrap();
        self.dev.htod_sync_copy_into(&pos, &mut bufs.pos_dev).unwrap();
        self.dev.htod_sync_copy_into(&slots, &mut bufs.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();

        for _ in 0..8 {   // warm the pool, cuBLAS, and both proxies
            self.forward_decode_gpu(&mut pool, &mut bufs, &mut state, max_seq_len, max_seq_len, batch);
            self.sync_stream();
        }
        let mut ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            self.forward_decode_gpu(&mut pool, &mut bufs, &mut state, max_seq_len, max_seq_len, batch);
            self.sync_stream();
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = ms[ms.len() / 2];
        eprintln!("[tp{}] BATCH PROBE N={batch}: p50 {p50:.2} ms/forward  ({:.2} ms per drafted position)",
                  self.tp_rank, p50 / batch as f64);
    }

    /// PROBE 2 — measure a synthetic MTP step under TP directly, instead of assembling it from five
    /// separate estimates. Runs the real sequence with scripted (ignored) acceptance:
    ///   (depth-1) × draft  →  verify forward at batch=depth  →  GDN rollback
    /// Timing only; the tokens it produces are meaningless. `mtp_reprime` is excluded and added as its
    /// measured constant, because it needs a real accepted-prefix to reprime from.
    pub fn tp_synthetic_step_probe(&self, depth: usize, iters: usize, max_seq_len: usize) {
        let h = self.cfg.hidden_size;
        let mut pool = Pool::new(self.dev.clone());
        let nslots = depth.max(2);
        let mut state = self.new_batch_state(nslots, nslots, max_seq_len);
        let mut bufs = self.new_decode_buffers(depth);
        let (nkv, hd) = (self.cfg.num_kv_heads, self.cfg.head_dim);
        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(nkv * max_seq_len * hd).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(nkv * max_seq_len * hd).unwrap();
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        let hidden = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let toks: Vec<i32> = (0..depth).map(|i| (1000 + i) as i32).collect();
        let pos: Vec<i32> = (0..depth).map(|_| 16i32).collect();
        let slots: Vec<i32> = (0..depth).map(|i| i as i32).collect();
        self.dev.htod_sync_copy_into(&toks, &mut bufs.tokens_dev).unwrap();
        self.dev.htod_sync_copy_into(&pos, &mut bufs.pos_dev).unwrap();
        self.dev.htod_sync_copy_into(&slots, &mut bufs.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();
        let (kc, vc) = (*mtp_kc.device_ptr(), *mtp_vc.device_ptr());

        let mut step = |p: &mut Pool, st: &mut BatchGpuState, b: &mut DecodeBuffers| {
            for d in 0..depth.saturating_sub(1) {
                let o = self.mtp_draft_step(p, &hidden, 1000 + d as i32, 16 + d, kc, vc, max_seq_len);
                p.release_bf16(o, h);
            }
            self.forward_decode_gpu(p, b, st, max_seq_len, max_seq_len, depth);
            self.copy_gdn_slot(st, 0, 1);
            self.sync_stream();
        };
        for _ in 0..4 { step(&mut pool, &mut state, &mut bufs); }
        let mut ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            step(&mut pool, &mut state, &mut bufs);
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = ms[ms.len() / 2];
        eprintln!("[probe2] rank {} depth {depth}: synthetic step p50 {p50:.2} ms \
                   (+1.46 reprime = {:.2} ms) — drafts {} + verify N={depth} + rollback",
                  self.tp_rank, p50 + 1.46, depth.saturating_sub(1));
    }

    /// Is the FP32-preserving reduction active? bf16 partials round EACH rank's partial before summing,
    /// which is a real (if small) precision hole the single-node path does not have; FP32 partials close
    /// it, leaving only reassociation. Opt-in until it has run the long divergence gate.
    fn tp_fp32_partials(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        self.tp_world == 2 && *ON.get_or_init(|| std::env::var("GB10_TP_FP32_PARTIALS").is_ok()
            || crate::tp::tp_config().map(|c| c.fp32_partials).unwrap_or(false))
    }

    /// One row-parallel reduction site: FP32-preserving when enabled, bf16 otherwise.
    fn tp_reduce_site(&self, pool: &mut Pool, w: &W, x: &B, out: &mut B,
                      inn: usize, outn: usize, batch: usize) {
        // FP32-preserving partials where the width allows them (decode/verify, batch <= MAX_VERIFY):
        // one rounding boundary for the whole reduction. bf16 partials round each rank's partial
        // BEFORE the cross-rank sum (measured 1868/5120 mismatched elements @2 ULP vs 7/5120 @1 ULP
        // for fp32), and that error compounds in the residual stream across tokens — the MTP draft
        // head is scale-sensitive, so bf16 partials cost measurable acceptance at long context.
        // Wide prefill reductions stay on the chunked bf16 path: gemm_act_f32 only supports
        // batch <= MAX_VERIFY, and prefill reassociation is accepted anyway. (Previously the flag
        // panicked on ANY batch > 16, which made FP32 partials unusable end-to-end.)
        if self.tp_fp32_partials() && batch <= MAX_VERIFY {
            let mut p = pool.get(outn * batch);
            self.gemm_act_f32(w, x, &mut p, inn, outn, batch);
            self.tp_all_reduce_fp32(out, &p, outn * batch);
            pool.release(p, outn * batch);
        } else {
            self.gemm_act(w, x, out, inn, outn, batch);
            self.tp_all_reduce_bf16(out, outn * batch);
        }
    }

    fn mlp_batch(&self, pool: &mut Pool, x: &B, mlp: &GpuMlp, batch: usize, sharded: bool) -> B {
        let h = self.cfg.hidden_size;
        // TP=2 shards the FFN (weights split at attach_tp): gate/up column-parallel (each rank owns
        // im/2 output rows → each streams half the gate/up bytes), down row-parallel (each rank owns
        // im/2 of K → its down output is a PARTIAL). So the effective intermediate here is im/world,
        // and the row-parallel partials are summed by the all-reduce. world==1 = full, unsharded.
        let im = self.cfg.intermediate_size / if sharded { 2 } else { 1 };
        let mut gate = pool.get_bf16(im*batch);
        let mut upb = pool.get_bf16(im*batch);
        self.gemm_act(&mlp.gate, x, &mut gate, h, im, batch);
        self.gemm_act(&mlp.up, x, &mut upb, h, im, batch);
        blaunch!(self, "silu_mul_b", grid(im*batch), (256,1,1), 0, (d(&gate), d(&gate), d(&upb), (im*batch) as i32));
        let mut out = pool.get_bf16(h*batch);
        if sharded {
            self.tp_reduce_site(pool, &mlp.down, &gate, &mut out, im, h, batch);   // row-parallel partial → all-reduce
        } else {
            self.gemm_act(&mlp.down, &gate, &mut out, im, h, batch);
        }
        pool.release_bf16(gate, im*batch); pool.release_bf16(upb, im*batch);
        out
    }

    /// Tiled prefill attention: `S = QᵀK` and `O += P·Vᵀ` through cuBLAS tensor-core GEMMs, with an
    /// online-softmax kernel between them.
    ///
    /// The scalar `gqa_attn_prefill` is bound by instruction issue, not memory — query tiling cut its
    /// L1/L2 traffic 8× for 11%, and 4 independent loads in flight per warp bought 0%. A *perfect*
    /// scalar kernel bottoms out near 25 ms/layer; a *mediocre* tensor-core one starts at ~9. So the
    /// fix is which hardware unit runs the two GEMMs, not how the scalar loop is tuned.
    ///
    /// GQA: a group's 4 query heads share one K/V, and cuBLAS cannot express "stride 0 for four, then
    /// jump" — so we issue one strided-batched GEMM per index-within-group `r`, batched over kv heads.
    ///
    /// Prefill sits OUTSIDE the batch-invariance contract (it feeds decode and verify identically), so
    /// tiling and reordering are free here. Still deterministic: fixed tiles, fixed order.
    #[allow(clippy::too_many_arguments)]
    fn attn_prefill_tiled(&self, pool: &mut Pool, q: &B, kc_ptr: u64, vc_ptr: u64,
                          stride: usize, nh: usize, nkv: usize, n: usize, pos_start: usize, attn: &mut B) {
        use cudarc::cublas::sys::{cudaDataType, cublasComputeType_t, cublasGemmAlgo_t};
        let cfg = &self.cfg;
        // Head counts come from the CALLER, never cfg: under TP mixer sharding the q/k/v tensors and
        // the KV cache hold nh/2 and nkv/2 (local) heads — reading cfg's full counts here over-reads
        // every input (masked by pool slack at small n, an ILLEGAL ADDRESS at large n) and writes
        // `attn` with the wrong layout. Same root cause as the attn_dispatch half-heads bug:
        // sharding is a property of the tensors passed in, never of the process.
        let hd = cfg.head_dim;
        let gqa = nh / nkv;
        let pc = pos_start + n;                       // total keys any query here may attend to
        let scale = 1.0f32 / (hd as f32).sqrt();

        // Tile sizes. Note this function is ALSO the verify path (`verify_forward` calls it with
        // n = depth, not a prompt length), so it must stay bit-identical across n — which it is: the
        // per-query accumulation order does not depend on br. Bounding the Pool is `bucket_up`'s job,
        // not this line's.
        let br = PF_BR.min(n);
        let bc = PF_BC;

        let s_buf = pool.get(br * bc * nkv);          // S: [Br x Bc] f32, one slab per kv head
        let p_buf = pool.get_bf16(br * bc * nkv);     // P: same, bf16, fed to the PV GEMM
        let o_buf = pool.get(br * hd * nh);           // O: [Br x hd] f32 per query head, carried
        let m_buf = pool.get(br * nh);
        let l_buf = pool.get(br * nh);

        let handle = *self.blas.handle();
        let one: f32 = 1.0;
        let zero: f32 = 0.0;
        let bf = cudaDataType::CUDA_R_16BF;
        let f32t = cudaDataType::CUDA_R_32F;
        let comp = cublasComputeType_t::CUBLAS_COMPUTE_32F;
        let algo = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT;

        let mut t0 = 0usize;
        while t0 < n {
            let rows = (n - t0).min(br);              // live rows in this query tile
            blaunch!(self, "attn_tile_init", grid(nh * br * hd), (256,1,1), 0,
                (d(&o_buf), d(&m_buf), d(&l_buf), br as i32, hd as i32, nh as i32));

            // The last key any query in this tile may see. Key tiles beyond it are fully masked.
            let kmax = pos_start + t0 + rows - 1;
            let mut s0 = 0usize;
            while s0 <= kmax {
                let kcols = bc.min(pc - s0);
                for r in 0..gqa {
                    // ---- Sᵀ = Kᵀ · Q_tile, batched over kv heads ----
                    // TRANSPOSED on purpose: the softmax walks one query's keys, and it must read them
                    // contiguously. C is [kcols x rows], ld = bc, so row i of S lives at i*bc.
                    unsafe {
                        cudarc::cublas::result::gemm_strided_batched_ex(
                            handle,
                            OP::CUBLAS_OP_T, OP::CUBLAS_OP_N,
                            kcols as i32, rows as i32, hd as i32,
                            &scale as *const f32 as *const _,      // fold 1/sqrt(hd) into alpha
                            (kc_ptr + (s0 * hd * 2) as u64) as *const _,
                            bf, hd as i32, (stride * hd) as i64,
                            (*q.device_ptr() as u64 + ((r * hd + t0 * nh * hd) * 2) as u64) as *const _,
                            bf, (nh * hd) as i32, (gqa * hd) as i64,
                            &zero as *const f32 as *const _,
                            *s_buf.device_ptr() as *mut _,
                            f32t, bc as i32, (br * bc) as i64,
                            nkv as i32, comp, algo,
                        ).expect("prefill K^T Q");
                    }

                    // Packed args: cudarc's launch tuple caps at 12. scale rides in the GEMM alpha.
                    let nh_gqa_r = (nh * 10000 + gqa * 100 + r) as i32;
                    let s0_pc = (((s0 as i64) << 32) | (pc as i64)) as i64;
                    let smem = 256u32 * 4;
                    blaunch!(self, "attn_softmax_tile", (br as u32, nkv as u32, 1), (256,1,1), smem,
                        (d(&s_buf), d(&p_buf), d(&o_buf), d(&m_buf), d(&l_buf),
                         // Bc is the SLAB STRIDE (the allocated tile width), not the valid column
                         // count: the GEMM strides S/P slabs by br*bc, so indexing them by kcols
                         // sends every kv head after the first to the wrong slab. Columns past the
                         // key range are masked out by the causal test anyway.
                         br as i32, bc as i32, hd as i32, nh_gqa_r, rows as i32,
                         (pos_start + t0) as i32, s0_pc));

                    // ---- O += P · V_tileᵀ (beta=1 accumulates across key tiles) ----
                    // P is stored transposed too ([kcols x rows], ld = bc), so A is transposed here.
                    unsafe {
                        cudarc::cublas::result::gemm_strided_batched_ex(
                            handle,
                            OP::CUBLAS_OP_T, OP::CUBLAS_OP_T,
                            rows as i32, hd as i32, kcols as i32,
                            &one as *const f32 as *const _,
                            *p_buf.device_ptr() as *const _,
                            bf, bc as i32, (br * bc) as i64,
                            (vc_ptr + (s0 * hd * 2) as u64) as *const _,
                            bf, hd as i32, (stride * hd) as i64,
                            &one as *const f32 as *const _,
                            (*o_buf.device_ptr() as u64 + (r * br * hd * 4) as u64) as *mut _,
                            f32t, br as i32, (gqa * br * hd) as i64,
                            nkv as i32, comp, algo,
                        ).expect("prefill PV");
                    }
                }
                s0 += bc;
            }

            assert!(hd <= 256, "attn_finalize smem tile is sized for hd <= 256");
            blaunch!(self, "attn_finalize", ((br.div_ceil(32)) as u32, nh as u32, 1), (32u32,8u32,1), 0,
                (d(attn), d(&o_buf), d(&l_buf), br as i32, hd as i32, nh as i32, t0 as i32, n as i32));
            t0 += br;
        }

        pool.release(s_buf, br * bc * nkv);
        pool.release_bf16(p_buf, br * bc * nkv);
        pool.release(o_buf, br * hd * nh);
        pool.release(m_buf, br * nh);
        pool.release(l_buf, br * nh);
    }

    /// Canonical attention KV-write + softmax dispatch, shared by decode (`full_attn_batch`) and
    /// MTP verify (`verify_forward`). Writes new K/V via `write_kv_b` (slot-indexed) then runs
    /// `gqa_attn_splitk`+`gqa_attn_reduce` — the ONLY decode-attention path there is.
    /// Both call sites MUST use this so the verify forward (batch N) column 0 is
    /// bit-identical to a single-token decode (N=1) at the same position — the lossless-MTP contract.
    /// kc_ptr/vc_ptr are BASE cache pointers (slot offset applied inside the kernels via slot_ids);
    /// `max_pc` is the largest `pos[b]+1` across the batch.
    fn attn_dispatch(&self, pool: &mut Pool, q: &B, k: &B, v: &B, pos_dev: &cudarc::driver::CudaSlice<i32>,
                     kc_ptr: u64, vc_ptr: u64, slot_ids_ptr: u64, max_pc: usize, kv_stride: usize, batch: usize,
                     prefill_pos_start: Option<usize>, path_ptr: u64, rope_ptr: u64, col_pos_start_ptr: u64,
                     sharded: bool) -> B {
        let cfg = &self.cfg;
        // Head counts must match the CALLER's tensors, not a global flag. The MTP head is replicated
        // under TP, so it hands us full-width q/k/v and a full-width KV cache; deriving the counts from
        // eff_num_heads() here ran its attention at HALF the heads and produced garbage drafts
        // (acceptance collapsed 58% -> 2.7%). Third instance of the same root cause: sharding is a
        // property of the tensors passed in, never of the process.
        let (nh, nkv, hd) = if sharded { (self.eff_num_heads(), self.eff_num_kv_heads(), cfg.head_dim) }
                            else { (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) };
        let stride = kv_stride;
        let scale = 1.0f32/(hd as f32).sqrt();

        // PREFILL mode: a wide causal-append into ONE slot's cache, positions pos_start..pos_start+N-1.
        // Used by the prompt prefill and by the MTP head's prompt prime. The slot-indexed decode
        // kernels are wrong here for two reasons: they need a slot_ids entry per column (the MTP head's
        // is a single permanently-zero slot, sized for MAX_VERIFY, so a wide batch reads past it), and
        // their split-K fan-out would launch batch*nh*n_splits blocks -- millions at an 8K prompt.
        let nw = (hd / 32) as u32;                      // warps per block (blockDim = hd = 256)
        let smem = (nw * hd as u32 + 2 * nw) * 4;
        if let Some(ps) = prefill_pos_start {
            blaunch!(self, "write_kv_prefill", grid(batch*nkv*hd), (256,1,1), 0,
                (kc_ptr, vc_ptr, d(k), d(v), stride as i32, nkv as i32, hd as i32, batch as i32, ps as i32));
            let mut attn = pool.get_bf16(nh*hd*batch);
            if batch >= PF_MIN && !prefill_scalar() {
                // Tensor-core path: the two GEMMs go through cuBLAS. See attn_prefill_tiled.
                self.attn_prefill_tiled(pool, q, kc_ptr, vc_ptr, stride, nh, nkv, batch, ps, &mut attn);
            } else {
                // Short chunk: the scalar kernel's fixed costs are lower than the tiled path's.
                let ntile = (batch + GQA_PF_QT - 1) / GQA_PF_QT;
                blaunch!(self, "gqa_attn_prefill", ((ntile*nh) as u32,1,1), (hd as u32,1,1), 0,
                    (d(&attn), d(q), kc_ptr, vc_ptr, stride as i32, nh as i32, nkv as i32, hd as i32,
                     fbits(scale), batch as i32, ps as i32));
            }
            return attn;
        }

        let pos_ptr = *pos_dev.device_ptr() as u64;
        let logical_ptr = if rope_ptr != 0 { rope_ptr } else { pos_ptr };  // splitk/reduce use LOGICAL pos (rope); write_kv uses the KV SLOT (pos)
        blaunch!(self, "write_kv_b", grid(batch*nkv*hd), (256,1,1), 0,
            (kc_ptr, vc_ptr, d(k), d(v), pos_ptr, stride as i32, nkv as i32, hd as i32, batch as i32, slot_ids_ptr));
        let attn = pool.get_bf16(nh*hd*batch);
        let nh_packed = (nh * 1000 + nkv) as i32;

        // ===================== THE LOSSLESS-MTP CONTRACT =====================
        //
        // Attention for a query at position p must be a pure function of (q, KV[0..p], pc). It may NOT
        // depend on `batch`, on `max_pc`, or on anything else derived from the other columns -- because
        // column k of an N-wide verify has to be BIT-IDENTICAL to a 1-token decode at that position.
        //
        // Two versions of this got it wrong, and the second is the instructive one:
        //   * n_splits from batch*nh -> a decode split the keys 6 ways, a 4-wide verify 2 ways.
        //   * n_splits from max_pc   -> STILL WRONG. A decode has max_pc = pos+1, a verify has
        //     max_pc = pos_start+N. They straddle a 256-boundary regularly, and when one landed on
        //     n_splits==1 it ran a DIFFERENT KERNEL (gqa_attn_flash) with different rounding. MTP was
        //     silently non-lossless and STILL PASSED THE GATE most runs, because a 1-ulp difference
        //     rarely flips an argmax. --bench-mtp only caught it once an unrelated change perturbed
        //     the trajectory. A gate that fails as a coin toss is not a gate.
        //
        // So the split count is now computed INSIDE the kernel from each column's OWN pc (sk_nsplits),
        // there is only ONE kernel (gqa_attn_flash is deleted), and the partials are fp32. `ns_grid` is
        // purely a launch bound: pc_b <= max_pc implies ns_b <= ns_grid, so every block a column needs
        // exists, and the surplus ones return immediately.
        let ns_grid = (max_pc / 256).clamp(1, 32);
        let n_partial = batch * nh * ns_grid;
        let pm = pool.get(n_partial);
        let pl = pool.get(n_partial);
        let pa = pool.get(n_partial * hd);          // fp32: the bf16 round-trip was lossy for nothing
        // bs_packed = stride(bits 0-18) | ns_grid(19-24) | batch(25-30). stride packed to fit the
        // 12-arg launch cap; ranges: stride<=262144<2^19, ns_grid<=32<2^6, batch<2^6.
        debug_assert!(stride < (1<<19) && ns_grid < (1<<6) && batch < (1<<6), "bs_packed field overflow");
        let bs_packed = ((batch as i32) << 25) | ((ns_grid as i32) << 19) | (stride as i32);
        blaunch!(self, "gqa_attn_splitk", ((batch * nh * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
            (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
             logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr, col_pos_start_ptr));
        blaunch!(self, "gqa_attn_reduce", ((batch*nh) as u32,1,1), (hd as u32,1,1), 0,
            (d(&attn), d(&pm), d(&pl), d(&pa), logical_ptr, ns_grid as i32, batch as i32, nh_packed));
        pool.release(pm, n_partial);
        pool.release(pl, n_partial);
        pool.release(pa, n_partial * hd);
        attn
    }

    fn full_attn_batch(&self, pool: &mut Pool, hidden: &B, fa: &GpuFullAttn, pos_dev: &cudarc::driver::CudaSlice<i32>,
                       max_pc: usize, kv_stride: usize, kc_ptr: u64, vc_ptr: u64, cos: &S, sin: &S, slot_ids_ptr: u64, batch: usize,
                       prefill_pos_start: Option<usize>, sharded: bool) -> B {
        let cfg = &self.cfg;
        let (h, hd, rdim) = (cfg.hidden_size, cfg.head_dim, cfg.rotary_dim);
        // TP=2: each box owns half the heads (12Q/2KV). Weights + KV cache are sharded to match, so the
        // whole mixer runs at local head counts and a single all-reduce on the o_proj partial stitches
        // the boxes back together. world==1 → full counts, no all-reduce.
        let (nh, nkv) = if sharded { (self.eff_num_heads(), self.eff_num_kv_heads()) }
                        else { (self.cfg.num_heads, self.cfg.num_kv_heads) };
        let mut qg = pool.get_bf16(nh*hd*2*batch);
        let mut k = pool.get_bf16(nkv*hd*batch);
        let mut v = pool.get_bf16(nkv*hd*batch);
        let q = pool.get_bf16(nh*hd*batch);
        let gate = pool.get_bf16(nh*hd*batch);
        match &fa.qkv {
            AttnIn::Fused(w) => {
                // ONE GEMM for q|gate, k and v: they all read the same activation, and k/v alone are
                // M=1024 -- only 84 GB/s against a 234 GB/s machine, because grid=64 barely fills it.
                let mtot = nh*hd*2 + 2*nkv*hd;   // local fused width (matches the sharded qkv weight)
                let mut fused = pool.get_bf16(mtot * batch);
                self.gemm_act(w, hidden, &mut fused, h, mtot, batch);
                blaunch!(self, "split_qkv_b", grid(mtot*batch), (256,1,1), 0,
                    (d(&qg), d(&k), d(&v), d(&fused), (nh*hd*2) as i32, (nkv*hd) as i32, batch as i32));
                pool.release_bf16(fused, mtot * batch);
            }
            AttnIn::Split { q: qp, k: kp, v: vp } => {
                self.gemm_act(qp, hidden, &mut qg, h, nh*hd*2, batch);
                self.gemm_act(kp, hidden, &mut k, h, nkv*hd, batch);
                self.gemm_act(vp, hidden, &mut v, h, nkv*hd, batch);
            }
        }
        blaunch!(self, "split_qgate_b", grid(nh*hd*batch), (256,1,1), 0, (d(&q), d(&gate), d(&qg), nh as i32, hd as i32, batch as i32));
        blaunch!(self, "rmsnorm_perhead_b", ((batch*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(&q), d(&q), d(&fa.q_norm), nh as i32, hd as i32, batch as i32, fbits(cfg.rms_eps)));
        blaunch!(self, "rmsnorm_perhead_b", ((batch*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(&k), d(&k), d(&fa.k_norm), nkv as i32, hd as i32, batch as i32, fbits(cfg.rms_eps)));
        blaunch!(self, "rope_b", grid(batch*nh*(rdim/2)), (256,1,1), 0, (d(&q), d(cos), d(sin), nh as i32, hd as i32, rdim as i32, batch as i32));
        blaunch!(self, "rope_b", grid(batch*nkv*(rdim/2)), (256,1,1), 0, (d(&k), d(cos), d(sin), nkv as i32, hd as i32, rdim as i32, batch as i32));
        let attn = self.attn_dispatch(pool, &q, &k, &v, pos_dev, kc_ptr, vc_ptr, slot_ids_ptr, max_pc, kv_stride, batch, prefill_pos_start, 0u64, 0u64, 0u64, sharded);
        blaunch!(self, "sigmoid_gate_b", grid(nh*hd*batch), (256,1,1), 0, (d(&attn), d(&gate), (nh*hd*batch) as i32));
        let mut out = pool.get_bf16(h*batch);
        if sharded {
            self.tp_reduce_site(pool, &fa.o_proj, &attn, &mut out, nh*hd, h, batch);
        } else {
            self.gemm_act(&fa.o_proj, &attn, &mut out, nh*hd, h, batch);
        }
        pool.release_bf16(qg, nh*hd*2*batch); pool.release_bf16(k, nkv*hd*batch); pool.release_bf16(v, nkv*hd*batch);
        pool.release_bf16(q, nh*hd*batch); pool.release_bf16(gate, nh*hd*batch); pool.release_bf16(attn, nh*hd*batch);
        out
    }

    fn linear_attn_batch(&self, pool: &mut Pool, hidden: &B, la: &GpuLinearAttn,
                         conv_ptr: u64, s_ptr: u64, slot_ids_ptr: u64, batch: usize) -> B {
        let cfg = &self.cfg;
        // TP-local GDN geometry (see eff_lin_v_heads). Every dim below is rank-local when the mixers are
        // sharded, so the conv/state/gate/norm kernels touch only this box's heads.
        let (h, kd, vd) = (cfg.hidden_size, cfg.lin_k_dim, cfg.lin_v_dim);
        let nh = self.eff_lin_v_heads();
        let key_dim = self.eff_key_dim(); let value_dim = self.eff_value_dim();
        let conv_dim = self.eff_conv_dim(); let ck = cfg.conv_kernel;
        let mut qkv = pool.get_bf16(conv_dim*batch);
        let mut z = pool.get_bf16(value_dim*batch);
        let mut b = pool.get_bf16(nh*batch);
        let mut a = pool.get_bf16(nh*batch);
        match &la.in_proj {
            GdnIn::Fused(w) => {
                // ONE GEMM instead of four. in_proj_b/in_proj_a are M=nh (32 on 9B), i.e. grid=2 on a
                // 48-SM GPU: 26 us to move 74 KB, and across 24 GDN layers that was 4.7% of ALL GEMM
                // time for 0.03% of the bytes. Fused, they are just rows of a big efficient kernel.
                let mtot = self.eff_gdn_fused_m();
                let mut fused = pool.get_bf16(mtot * batch);
                self.gemm_act(w, hidden, &mut fused, h, mtot, batch);
                blaunch!(self, "split_gdn_b", grid(mtot*batch), (256,1,1), 0,
                    (d(&qkv), d(&z), d(&b), d(&a), d(&fused),
                     conv_dim as i32, value_dim as i32, nh as i32, batch as i32,
                     self.cfg.lin_num_v_heads as i32, self.gdn_head0() as i32));
                pool.release_bf16(fused, mtot * batch);
            }
            GdnIn::Split { qkv: pq, z: pz, b: pb, a: pa } => {
                self.gemm_act(pq, hidden, &mut qkv, h, conv_dim, batch);
                self.gemm_act(pz, hidden, &mut z, h, value_dim, batch);
                self.gemm_act(pb, hidden, &mut b, h, nh, batch);
                self.gemm_act(pa, hidden, &mut a, h, nh, batch);
            }
        }
        blaunch!(self, "conv1d_b", grid(batch*conv_dim), (256,1,1), 0, (d(&qkv), conv_ptr, d(&la.conv1d), conv_dim as i32, ck as i32, batch as i32, slot_ids_ptr));
        // NEGATIVE CONTROL (GB10_TP_HEAD_PROOF_FAULT=1): deliberately launch the state kernel at the
        // FULL head count — precisely the bug the expert warned about. The output stays CORRECT (out_proj
        // reads only the local value_dim), so the token gate still passes and timing barely moves; only
        // the state red zone can catch it. If this fault does NOT trip the detector, the detector is
        // worthless and the PASS above means nothing.
        // Proof D: per-head visit counter (see delta_step_b). Null outside the proof build.
        let visits_ptr = if self.tp_head_proof() {
            self.head_visits.as_ref().map(|v| *v.device_ptr() as u64).unwrap_or(0)
        } else { 0u64 };
        let fault = self.tp_head_proof() && std::env::var("GB10_TP_HEAD_PROOF_FAULT").is_ok();
        let nh_launch = if fault { self.cfg.lin_num_v_heads } else { nh };
        let nk_launch = if fault { self.cfg.lin_num_k_heads } else { self.eff_lin_k_heads() };
        let core = pool.get_bf16(nh_launch*vd*batch);
        let (nchunk, smem) = gdn_launch(kd, vd);
        if self.tp_head_proof() {
            // Proof C. NOTE what this asserts on: `nh_launch`/`nk_launch`, the values ACTUALLY handed to
            // the kernel — not the eff_* accessors. The first version of this check read the accessors,
            // and the injected-fault control sailed straight past it while the kernel launched at full
            // width. A geometry check that does not read the launch argument is theatre.
            static ONCE: std::sync::Once = std::sync::Once::new();
            let (gnh, gnk) = (self.cfg.lin_num_v_heads, self.cfg.lin_num_k_heads);
            ONCE.call_once(|| {
                eprintln!("[head-proof] launch geometry: v_heads {nh_launch} (global {gnh}), k_heads \
                           {nk_launch} (global {gnk}), conv_dim {conv_dim}, value_dim {value_dim}");
            });
            if !fault {
                assert_eq!(nh_launch, gnh / 2, "delta_step_b launched {nh_launch} value heads, expected local {}", gnh / 2);
                assert_eq!(nk_launch, gnk / 2, "delta_step_b launched {nk_launch} key heads, expected local {}", gnk / 2);
            }
            assert_eq!(conv_dim, self.eff_conv_dim(), "conv1d_b grid not local");
            assert_eq!(value_dim, self.eff_value_dim(), "rmsnorm_gated_b extent not local");
        }
        blaunch!(self, "delta_step_b", ((batch*nh_launch*nchunk) as u32,1,1), (kd as u32,1,1), smem, (d(&core), d(&qkv), s_ptr, d(&b), d(&a), (nh_launch as i32) | ((nk_launch as i32) << 16), kd as i32, vd as i32, d(&la.a_log), d(&la.dt_bias), slot_ids_ptr, visits_ptr));
        let normed = pool.get_bf16(value_dim*batch);
        blaunch!(self, "rmsnorm_gated_b", ((batch*nh) as u32,1,1), (vd as u32,1,1), (vd*4) as u32, (d(&normed), d(&core), d(&z), d(&la.norm), vd as i32, nh as i32, batch as i32, fbits(cfg.rms_eps)));
        let mut out = pool.get_bf16(h*batch);
        if self.tp_shard_mixers() {
            self.tp_reduce_site(pool, &la.out_proj, &normed, &mut out, value_dim, h, batch);
        } else {
            self.gemm_act(&la.out_proj, &normed, &mut out, value_dim, h, batch);
        }
        pool.release_bf16(qkv, conv_dim*batch); pool.release_bf16(z, value_dim*batch); pool.release_bf16(b, nh*batch); pool.release_bf16(a, nh*batch);
        pool.release_bf16(core, nh_launch*vd*batch); pool.release_bf16(normed, value_dim*batch);
        out
    }

    /// One batched decode step. hidden[hidden,B] bf16. pos_host: per-seq positions.
    pub fn forward_batch(&self, pool: &mut Pool, hidden: B, pos_host: &[usize],
                         state: &mut BatchGpuState, kv_stride: usize, batch: usize) -> B {
        self.forward_batch_core(pool, hidden, pos_host, state, 0, kv_stride, batch)
    }
    /// Prefill/decode a single token into a specific slot (B=1, state offset by `slot`).
    pub fn forward_into_slot(&self, pool: &mut Pool, hidden: B, pos: usize,
                             state: &mut BatchGpuState, slot: usize, kv_stride: usize) -> B {
        self.forward_batch_core(pool, hidden, &[pos], state, slot, kv_stride, 1)
    }
    fn forward_batch_core(&self, pool: &mut Pool, hidden: B, pos_host: &[usize],
                          state: &mut BatchGpuState, slot: usize, kv_stride: usize, batch: usize) -> B {
        let cfg = &self.cfg;
        let rdim = cfg.rotary_dim;
        let max_pc = pos_host.iter().map(|&p| p + 1).max().unwrap_or(1);

        let pos_i32: Vec<i32> = pos_host.iter().map(|&p| p as i32).collect();
        let pos_dev = self.dev.htod_sync_copy(&pos_i32).expect("htod pos");
        let slot_ids: Vec<i32> = (0..batch).map(|i| slot as i32 + i as i32).collect();
        let slot_ids_dev = self.dev.htod_sync_copy(&slot_ids).expect("htod slot_ids");
        let cos = pool.get(batch * rdim);
        let sin = pool.get(batch * rdim);
        self.dev.synchronize().unwrap(); // htod + allocs on NULL stream must complete before non-blocking stream reads
        blaunch!(self, "gather_rope_b", grid(batch*rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *pos_dev.device_ptr() as u64, rdim as i32, batch as i32));

        let out = self.forward_batch_dev(pool, hidden, &pos_dev, &cos, &sin, max_pc,
                               state, *slot_ids_dev.device_ptr(), kv_stride, batch);
        self.sync_stream(); // ensure compute stream done before pos_dev (non-pool CudaSlice) drops
        pool.release(cos, batch * rdim);
        pool.release(sin, batch * rdim);
        out
    }

    /// Core forward pass using device-side pos/cos/sin (no host syncs). All activations bf16.
    fn forward_batch_dev(&self, pool: &mut Pool, hidden: B, pos_dev: &cudarc::driver::CudaSlice<i32>,
                         cos: &S, sin: &S, max_pc: usize,
                         state: &mut BatchGpuState, slot_ids_ptr: u64, kv_stride: usize, batch: usize) -> B {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;

        let residual = hidden;
        let normed = pool.get_bf16(h*batch);
        // FFN epilogue fusion (GB10_FUSE_RESIDUAL=1). The MIXER's residual add is already fused into
        // `fused_res_rmsnorm_b`; the FFN's is not — `add_residual_b` then the next layer's `rmsnorm_b`
        // are two kernels doing exactly what that one fused kernel does. Folding them removes ONE kernel
        // per layer (64/token) and hoists the very first input norm out of the loop.
        //
        // It is NOT numerics-neutral, which is why it is a flag: the two-kernel path computes sum_sq from
        // the bf16-ROUNDED residual, while the fused kernel uses the unrounded FP32 sum. That is one
        // fewer rounding — strictly better, and the same reassociation class as the FP32 partials — but
        // it does change output bytes, so the operator makes the call.
        let fuse = self.fuse_residual_norm();
        let nlayers = self.layers.len();
        if fuse {
            blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32, (d(&normed), d(&residual), d(&self.layers[0].input_ln), h as i32, batch as i32, fbits(cfg.rms_eps)));
        }
        for (li, layer) in self.layers.iter().enumerate() {
            if !fuse {
                blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32, (d(&normed), d(&residual), d(&layer.input_ln), h as i32, batch as i32, fbits(cfg.rms_eps)));
            }
            let mixer = match layer.layer_type {
                LayerType::LinearAttention => {
                    let conv_ptr = *state.conv_state[li].as_ref().unwrap().device_ptr();
                    let s_ptr = *state.s_state[li].as_ref().unwrap().device_ptr();
                    self.linear_attn_batch(pool, &normed, layer.la.as_ref().unwrap(), conv_ptr, s_ptr, slot_ids_ptr, batch)
                }
                LayerType::FullAttention => {
                    let kc_ptr = *state.k_cache[li].as_ref().unwrap().device_ptr();
                    let vc_ptr = *state.v_cache[li].as_ref().unwrap().device_ptr();
                    self.full_attn_batch(pool, &normed, layer.fa.as_ref().unwrap(),
                        &pos_dev, max_pc, kv_stride, kc_ptr, vc_ptr, &cos, &sin, slot_ids_ptr, batch, None,
                        self.tp_shard_mixers())
                }
            };
            blaunch!(self, "fused_res_rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32, (d(&normed), d(&residual), d(&mixer), d(&layer.post_ln), h as i32, batch as i32, fbits(cfg.rms_eps)));
            let mlp_out = self.ffn_batch(pool, &normed, &layer.mlp, batch, self.tp_world == 2);
            if fuse {
                // residual += mlp_out AND produce the NEXT layer's input norm (or, on the last layer,
                // the final norm) in a single kernel.
                let next_w = if li + 1 < nlayers { &self.layers[li + 1].input_ln } else { &self.final_norm };
                blaunch!(self, "fused_res_rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32, (d(&normed), d(&residual), d(&mlp_out), d(next_w), h as i32, batch as i32, fbits(cfg.rms_eps)));
            } else {
                let tot = h*batch;
                blaunch!(self, "add_residual_b", grid(tot), (256,1,1), 0, (d(&residual), d(&residual), d(&mlp_out), tot as i32));
            }
            pool.release_bf16(mixer, h*batch); pool.release_bf16(mlp_out, h*batch);
        }
        let out = if fuse {
            normed          // the last iteration already wrote rmsnorm(residual, final_norm) here
        } else {
            let o = pool.get_bf16(h*batch);
            blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32, (d(&o), d(&residual), d(&self.final_norm), h as i32, batch as i32, fbits(cfg.rms_eps)));
            pool.release_bf16(normed, h*batch);
            o
        };
        pool.release_bf16(residual, h*batch);
        out
    }

    /// FFN residual+norm fusion (see forward_batch_dev). Opt-in: it removes 64 kernels/token but moves a
    /// rounding boundary (sum_sq from the unrounded FP32 sum rather than the bf16-rounded residual).
    fn fuse_residual_norm(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("GB10_FUSE_RESIDUAL").is_ok())
    }

    pub fn logits_batch(&self, pool: &mut Pool, hidden: &B, batch: usize) -> B {
        let (v, h) = (self.cfg.vocab_size, self.cfg.hidden_size);
        let mut logits = pool.get_bf16(v*batch);
        let w = self.lm_head.as_ref().unwrap_or(&self.embed);
        self.gemm_act(w, hidden, &mut logits, h, v, batch);
        logits
    }

    // ===================== OPTIMIZED DECODE (pre-computed everything) =====================

    pub fn new_decode_buffers(&self, max_batch: usize) -> DecodeBuffers {
        let rdim = self.cfg.rotary_dim;
        DecodeBuffers {
            pos_dev: self.dev.htod_sync_copy(&vec![0i32; max_batch]).unwrap(),
            tokens_dev: self.dev.htod_sync_copy(&vec![0i32; max_batch]).unwrap(),
            slot_ids_dev: self.dev.htod_sync_copy(&(0..max_batch as i32).collect::<Vec<_>>()).unwrap(),
            cos_dev: self.dev.alloc_zeros::<f32>(max_batch * rdim).unwrap(),
            sin_dev: self.dev.alloc_zeros::<f32>(max_batch * rdim).unwrap(),
            token_ids_dev: self.dev.alloc_zeros::<i32>(max_batch).unwrap(),
            penalty_tokens_dev: self.dev.htod_sync_copy(&vec![-1i32; MAX_PEN_TOKENS * max_batch]).unwrap(),
            penalty_counts_dev: self.dev.htod_sync_copy(&vec![0i16; MAX_PEN_TOKENS * max_batch]).unwrap(),
            temps_dev: self.dev.htod_sync_copy(&vec![1.0f32; max_batch]).unwrap(),
            topk_dev: self.dev.htod_sync_copy(&vec![1i32; max_batch]).unwrap(),
            topp_dev: self.dev.htod_sync_copy(&vec![1.0f32; max_batch]).unwrap(),
            seeds_dev: self.dev.htod_sync_copy(&vec![0u32; max_batch]).unwrap(),
            rep_pen_dev: self.dev.htod_sync_copy(&vec![1.0f32; max_batch]).unwrap(),
            presence_dev: self.dev.htod_sync_copy(&vec![0.0f32; max_batch]).unwrap(),
            frequency_dev: self.dev.htod_sync_copy(&vec![0.0f32; max_batch]).unwrap(),
        }
    }

    /// One fully device-side decode step using pre-computed RoPE tables and persistent buffers.
    /// Caller must write `tokens` and `pos` to bufs before calling.
    /// GPU-only decode step — writes to bufs.token_ids_dev. Fully graph-capturable (no host sync).
    pub fn forward_decode_gpu(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                          state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize) {
        // Caller must sync NULL stream before calling (after htod copies into bufs, incl. per-lane
        // rep/presence/frequency in bufs). Fully graph-capturable (no host sync).
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let rdim = cfg.rotary_dim;
        let v = cfg.vocab_size;

        let hidden = pool.get_bf16(h * batch);
        self.embed_gather(*hidden.device_ptr() as u64,
                          *bufs.tokens_dev.device_ptr() as u64, h, batch);
        blaunch!(self, "gather_rope_b", grid(batch*rdim), (256,1,1), 0,
            (d(&bufs.cos_dev), d(&bufs.sin_dev),
             d(&self.cos_table), d(&self.sin_table),
             *bufs.pos_dev.device_ptr() as u64, rdim as i32, batch as i32));
        let out = self.forward_batch_dev(pool, hidden, &bufs.pos_dev, &bufs.cos_dev, &bufs.sin_dev,
                                         max_pc, state, *bufs.slot_ids_dev.device_ptr(), kv_stride, batch);
        let logits = self.logits_batch(pool, &out, batch);
        pool.release_bf16(out, h * batch);
        blaunch!(self, "rep_penalty_b", (batch as u32,1,1), (256,1,1), 0,
            (d(&logits), *bufs.penalty_tokens_dev.device_ptr() as u64,
             *bufs.penalty_counts_dev.device_ptr() as u64, MAX_PEN_TOKENS as i32,
             *bufs.rep_pen_dev.device_ptr() as u64,
             *bufs.presence_dev.device_ptr() as u64,
             *bufs.frequency_dev.device_ptr() as u64,
             v as i32, batch as i32));
        let block = 1024u32;
        blaunch!(self, "argmax_b", (batch as u32, 1, 1), (block, 1, 1), (block as u32 * 8),
            (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, batch as i32));
        pool.release_bf16(logits, v * batch);
    }

    /// Full decode step: GPU forward + readback. Non-graph path.
    pub fn forward_decode(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                          state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize) -> Vec<u32> {
        self.forward_decode_gpu(pool, bufs, state, kv_stride, max_pc, batch);
        self.sync_stream();
        self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
            .into_iter().take(batch).map(|x| x as u32).collect()
    }

    /// Decode with CPU sampling (for non-greedy requests). Reads logits to host,
    /// applies temperature + top_k + top_p, samples per-lane.
    pub fn forward_decode_sample(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                                 state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize,
                                 temps: &[f32], top_ks: &[usize], top_ps: &[f32]) -> Vec<u32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let rdim = cfg.rotary_dim;
        let v = cfg.vocab_size;

        // Forward pass (same as forward_decode_gpu but without argmax)
        let hidden = pool.get_bf16(h * batch);
        self.embed_gather(*hidden.device_ptr() as u64,
                          *bufs.tokens_dev.device_ptr() as u64, h, batch);
        blaunch!(self, "gather_rope_b", grid(batch*rdim), (256,1,1), 0,
            (d(&bufs.cos_dev), d(&bufs.sin_dev),
             d(&self.cos_table), d(&self.sin_table),
             *bufs.pos_dev.device_ptr() as u64, rdim as i32, batch as i32));
        let out = self.forward_batch_dev(pool, hidden, &bufs.pos_dev, &bufs.cos_dev, &bufs.sin_dev,
                                         max_pc, state, *bufs.slot_ids_dev.device_ptr(), kv_stride, batch);
        let logits = self.logits_batch(pool, &out, batch);
        pool.release_bf16(out, h * batch);
        blaunch!(self, "rep_penalty_b", (batch as u32,1,1), (256,1,1), 0,
            (d(&logits), *bufs.penalty_tokens_dev.device_ptr() as u64,
             *bufs.penalty_counts_dev.device_ptr() as u64, MAX_PEN_TOKENS as i32,
             *bufs.rep_pen_dev.device_ptr() as u64,
             *bufs.presence_dev.device_ptr() as u64,
             *bufs.frequency_dev.device_ptr() as u64,
             v as i32, batch as i32));

        // Read logits to host and sample per-lane
        self.sync_stream();
        let lh: Vec<half::bf16> = self.dev.dtoh_sync_copy(&logits).unwrap();
        pool.release_bf16(logits, v * batch);

        (0..batch).map(|i| {
            let col: Vec<f32> = lh[i*v..(i+1)*v].iter().map(|x| half::bf16::to_f32(*x)).collect();
            crate::sampler::sample(&col, temps[i], top_ks[i], top_ps[i])
        }).collect()
    }

    /// Decode + on-GPU sampling. Assumes the caller has already htod'd tokens/pos/penalty
    /// (rep_pen/presence/frequency) AND the sampling params (temps/topk/topp/seeds) into `bufs`
    /// and synced the NULL stream. Non-graph path: runs the core, syncs, reads back token ids.
    pub fn forward_decode_sample_gpu(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                                  state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize) -> Vec<u32> {
        self.decode_sample_gpu_core(pool, bufs, state, kv_stride, max_pc, batch);
        self.sync_stream();
        let ids: Vec<i32> = self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap();
        ids.into_iter().take(batch).map(|x| x as u32).collect()
    }

    /// Pure-GPU decode + sample kernel sequence (embed → rope → forward → logits → rep_penalty →
    /// sample_b). Reads every input from persistent `bufs` and writes `bufs.token_ids_dev`. No host
    /// sync, so it is CUDA-graph capturable. Caller must htod all inputs first.
    fn decode_sample_gpu_core(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                              state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let rdim = cfg.rotary_dim;
        let v = cfg.vocab_size;

        let hidden = pool.get_bf16(h * batch);
        self.embed_gather(*hidden.device_ptr() as u64,
                          *bufs.tokens_dev.device_ptr() as u64, h, batch);
        blaunch!(self, "gather_rope_b", grid(batch*rdim), (256,1,1), 0,
            (d(&bufs.cos_dev), d(&bufs.sin_dev),
             d(&self.cos_table), d(&self.sin_table),
             *bufs.pos_dev.device_ptr() as u64, rdim as i32, batch as i32));
        let out = self.forward_batch_dev(pool, hidden, &bufs.pos_dev, &bufs.cos_dev, &bufs.sin_dev,
                                         max_pc, state, *bufs.slot_ids_dev.device_ptr(), kv_stride, batch);
        let logits = self.logits_batch(pool, &out, batch);
        pool.release_bf16(out, h * batch);
        blaunch!(self, "rep_penalty_b", (batch as u32,1,1), (256,1,1), 0,
            (d(&logits), *bufs.penalty_tokens_dev.device_ptr() as u64,
             *bufs.penalty_counts_dev.device_ptr() as u64, MAX_PEN_TOKENS as i32,
             *bufs.rep_pen_dev.device_ptr() as u64,
             *bufs.presence_dev.device_ptr() as u64,
             *bufs.frequency_dev.device_ptr() as u64,
             v as i32, batch as i32));
        // sample_b: one block per lane, 256 threads, smem = topv[K] + topi[K] + red[2*nthr].
        let kmax = 64usize;
        let nthr = 256u32;
        let smem = ((2 * kmax + 2 * nthr as usize) * 4) as u32;
        blaunch!(self, "sample_b", (batch as u32,1,1), (nthr,1,1), smem,
            (*bufs.token_ids_dev.device_ptr() as u64, d(&logits),
             *bufs.temps_dev.device_ptr() as u64, *bufs.topk_dev.device_ptr() as u64,
             *bufs.topp_dev.device_ptr() as u64, *bufs.seeds_dev.device_ptr() as u64,
             v as i32, batch as i32));
        pool.release_bf16(logits, v * batch);
    }

    /// Capture a decode+sample graph for a given batch size. Caller must have valid params in bufs.
    pub fn capture_decode_sample_graph(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                                       state: &mut BatchGpuState, kv_stride: usize, max_pc: usize,
                                       batch: usize) -> Option<CudaGraph> {
        use cudarc::driver::sys;
        let stream = self.stream.stream;
        self.decode_sample_gpu_core(pool, bufs, state, kv_stride, max_pc, batch);
        self.sync_stream();
        unsafe {
            if sys::cuStreamBeginCapture_v2(
                stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                != sys::CUresult::CUDA_SUCCESS {
                return None;
            }
        }
        self.decode_sample_gpu_core(pool, bufs, state, kv_stride, max_pc, batch);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        unsafe {
            if sys::cuStreamEndCapture(stream, &mut graph) != sys::CUresult::CUDA_SUCCESS { return None; }
        }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        unsafe {
            let r = sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0);
            sys::cuGraphDestroy(graph);
            if r != sys::CUresult::CUDA_SUCCESS { return None; }
        }
        Some(CudaGraph { exec, stream })
    }

    /// Replay a captured decode+sample graph and read back token ids. Caller must htod fresh
    /// seeds (and any changed params) into bufs before calling.
    pub fn replay_decode_sample(&self, bufs: &DecodeBuffers, graph: &CudaGraph, batch: usize) -> Vec<u32> {
        graph.launch();
        self.sync_stream();
        self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
            .into_iter().take(batch).map(|x| x as u32).collect()
    }

    /// Capture a CUDA graph for the decode forward pass at a specific batch size and max_pc.
    /// After capture, replay via `graph.launch()` + `dev.synchronize()` + dtoh readback.
    /// The pool's buffer addresses are frozen at capture time — do not change forward path without re-capturing.
    pub fn capture_decode_graph(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                                state: &mut BatchGpuState, kv_stride: usize, max_pc: usize,
                                batch: usize) -> Option<CudaGraph> {
        use cudarc::driver::sys;
        let stream = self.stream.stream;  // forked non-blocking stream (NOT the NULL stream)

        // Warmup: run once to populate pool + warm cuBLAS algorithm cache
        self.forward_decode_gpu(pool, bufs, state, kv_stride, max_pc, batch);
        self.sync_stream();

        // I8/Q4 tripwire (TP=2 only): the barrier protocol must be QUIESCED at capture time. Capture
        // records launches without executing them, so the device epoch must not move here; if it and the
        // published watermark disagree, a barrier is in flight and the graph would inherit a torn epoch.
        // Cheap, permanent, and the only thing standing between us and a very hard bug.
        if self.tp_world == 2 {
            let (epoch, ready) = (crate::net::traced_device_epoch(), crate::net::traced_gpu_ready());
            assert_eq!(epoch, ready,
                "graph capture with the TP barrier protocol in flight (device epoch {epoch} != gpu_ready {ready})");
        }

        // Begin capture. cudarc 0.9.15 launches on the legacy NULL stream, which is NOT capturable
        // (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED). Detect and fall back to non-graph decode.
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(
                stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            if r != sys::CUresult::CUDA_SUCCESS {
                return None;
            }
        }

        self.forward_decode_gpu(pool, bufs, state, kv_stride, max_pc, batch);

        let mut graph: sys::CUgraph = std::ptr::null_mut();
        unsafe {
            let r = sys::cuStreamEndCapture(stream, &mut graph);
            if r != sys::CUresult::CUDA_SUCCESS { return None; }
        }

        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        unsafe {
            let r = sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0);
            sys::cuGraphDestroy(graph);
            if r != sys::CUresult::CUDA_SUCCESS { return None; }
        }

        Some(CudaGraph { exec, stream })
    }

    /// Replay a captured graph and read back token IDs.
    pub fn replay_decode(&self, bufs: &DecodeBuffers, graph: &CudaGraph, batch: usize) -> Vec<u32> {
        graph.launch();
        self.sync_stream();
        self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
            .into_iter().take(batch).map(|x| x as u32).collect()
    }

    /// Gather `batch` embedding rows into `out_ptr`, dequantizing if the table is quantized.
    /// The embedding is a weight like any other — and on a tied-embedding model it IS the LM head, so
    /// leaving it bf16 would keep ~23% of the model's bytes unquantized.
    ///
    /// When `mma` is set the table is stored permuted into mma-fragment order, so the gather has to
    /// invert that permutation per element. This is the ONLY place a weight is read element-wise on
    /// the hot path; it costs a few index ops on one row per token.
    fn embed_gather(&self, out_ptr: u64, toks_ptr: u64, h: usize, batch: usize) {
        match &self.embed {
            W::Bf16(e) => {
                blaunch!(self, "embed_gather_b", grid(h*batch), (256,1,1), 0,
                    (out_ptr, *e.device_ptr() as u64, toks_ptr, h as i32, batch as i32));
            }
            W::Nvfp4 { qweight, scales, gs, .. } => {
                blaunch!(self, "embed_gather_fp4_tiled_b", grid(h*batch), (256,1,1), 0,
                    (out_ptr, *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                     d(gs), toks_ptr, h as i32, batch as i32));
            }
            W::Fp8 { data, row_scale, .. } => {
                blaunch!(self, "embed_gather_fp8_tiled_b", grid(h*batch), (256,1,1), 0,
                    (out_ptr, *data.device_ptr() as u64, d(row_scale),
                     toks_ptr, h as i32, batch as i32));
            }
            W::Nvfp4Raw { .. } => unreachable!("Nvfp4Raw is MoE-experts-only"),
        }
    }

    pub fn embed_batch(&self, tokens: &[u32]) -> B {
        let h = self.cfg.hidden_size;
        let b = tokens.len();
        let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        // htod_sync_copy is already host-blocking, and the blocking compute stream (I1) orders the
        // gather after the NULL-stream copy — the old dev.synchronize() here was a full pipeline
        // drain per verify, and the trailing sync_stream had no host consumer (callers stay on the
        // same stream). alloc_zeros is fine without a memset: the gather overwrites all h*b elems.
        let toks_dev = self.dev.htod_sync_copy(&toks_i32).expect("htod tokens");
        let hidden = self.dev.alloc_zeros::<half::bf16>(h * b).unwrap();
        self.embed_gather(*hidden.device_ptr() as u64, *toks_dev.device_ptr() as u64, h, b);
        hidden
    }

    /// `kv_slots` = slots that can hold a SEQUENCE (one per concurrent lane).
    /// `state_slots` = slots that can hold GDN recurrent state — always >= kv_slots, because the extra
    /// ones are pure state: the MTP GDN-rollback snapshots and the prompt checkpoints.
    ///
    /// These used to be the same number, so every snapshot slot also got a full KV cache it can never
    /// use — nothing indexes k_cache above a real lane. At 27B/depth-8 that was SEVEN unused KV slots,
    /// ~3.8 GB of VRAM. KV is position-addressable and the snapshots never roll it back (verify simply
    /// overwrites stale positions), which is exactly why they need none.
    pub fn new_batch_state(&self, kv_slots: usize, state_slots: usize, max_seq_len: usize) -> BatchGpuState {
        let cfg = &self.cfg;
        let stride = max_seq_len;
        let mut k_cache = vec![];
        let mut v_cache = vec![];
        let mut conv_state = vec![];
        let mut s_state = vec![];
        for lt in &cfg.layer_types {
            match lt {
                LayerType::FullAttention => {
                    // TP=2: each box caches only its half of the KV heads (2 of 4).
                    let bytes = kv_slots * self.eff_num_kv_heads() * stride * cfg.head_dim;
                    k_cache.push(Some(self.dev.alloc_zeros::<half::bf16>(bytes).unwrap()));
                    v_cache.push(Some(self.dev.alloc_zeros::<half::bf16>(bytes).unwrap()));
                    conv_state.push(None); s_state.push(None);
                }
                LayerType::LinearAttention => {
                    // TP=2 mixer-sharded: each box holds only its half of the GDN heads, so the conv and
                    // recurrent state halve too and never cross the wire.
                    let conv_dim = self.eff_conv_dim();
                    let conv_n = state_slots * conv_dim * cfg.conv_kernel;
                    let s_n = state_slots * self.eff_lin_v_heads() * cfg.lin_k_dim * cfg.lin_v_dim;
                    if self.tp_head_proof() {
                        // Proof B: over-allocate to the FULL head shape and poison everything above the
                        // local range with NaN. The kernels index by local head/channel, so correct code
                        // never touches the red zone; a full-head-count launch writes straight into it.
                        assert_eq!(state_slots, 1, "head-proof red zone assumes a single state slot");
                        // A distinctive BIT PATTERN, not plain NaN: the injected-fault control writes
                        // NaN into these heads (their a_log is NaN-padded by proof A), and NaN-over-NaN
                        // would be undetectable. Comparing bits catches any write at all.
                        let (cf, sf) = (conv_n * 2, s_n * 2);
                        let sent = f32::from_bits(TP_REDZONE_SENTINEL);
                        let mut cv = vec![0f32; cf]; for x in &mut cv[conv_n..] { *x = sent; }
                        let mut sv = vec![0f32; sf]; for x in &mut sv[s_n..] { *x = sent; }
                        conv_state.push(Some(self.dev.htod_sync_copy(&cv).unwrap()));
                        s_state.push(Some(self.dev.htod_sync_copy(&sv).unwrap()));
                    } else {
                    conv_state.push(Some(self.dev.alloc_zeros::<f32>(conv_n).unwrap()));
                    s_state.push(Some(self.dev.alloc_zeros::<f32>(s_n).unwrap()));
                    }
                    k_cache.push(None); v_cache.push(None);
                }
            }
        }
        BatchGpuState { k_cache, v_cache, conv_state, s_state }
    }

    /// Zero all GDN recurrent + conv state and KV cache. Needed after graph capture
    /// (capture advances stateful GDN state during warmup+capture passes).
    pub fn zero_state(&self, state: &mut BatchGpuState) {
        for opt in state.conv_state.iter_mut() {
            if let Some(s) = opt { self.dev.memset_zeros(s).unwrap(); }
        }
        for opt in state.s_state.iter_mut() {
            if let Some(s) = opt { self.dev.memset_zeros(s).unwrap(); }
        }
        for opt in state.k_cache.iter_mut() {
            if let Some(s) = opt { self.dev.memset_zeros(s).unwrap(); }
        }
        for opt in state.v_cache.iter_mut() {
            if let Some(s) = opt { self.dev.memset_zeros(s).unwrap(); }
        }
    }

    /// Zero the GDN recurrent + conv state for a specific slot (when reusing a slot for a new request).
    /// Uses the compute stream (ordered with subsequent prefill kernels).
    pub fn zero_slot_state(&self, state: &mut BatchGpuState, slot: usize, kv_stride: usize) {
        let cfg = &self.cfg;
        // must match new_batch_state's (possibly TP-local) allocation, or this zeroes the wrong extent
        let conv_dim = self.eff_conv_dim();
        let ck = cfg.conv_kernel;
        let lin_nh = self.eff_lin_v_heads();
        let kd = cfg.lin_k_dim;
        let vd = cfg.lin_v_dim;
        let nkv = self.eff_num_kv_heads();
        let hd = cfg.head_dim;
        let stream = self.stream.stream;
        for (li, lt) in cfg.layer_types.iter().enumerate() {
            match lt {
                LayerType::LinearAttention => {
                    let conv_bytes = conv_dim * ck * 4;
                    let s_bytes = lin_nh * kd * vd * 4;
                    unsafe {
                        if let Some(s) = state.conv_state[li].as_ref() {
                            let ptr = *s.device_ptr() as u64 + (slot * conv_bytes) as u64;
                            cudarc::driver::sys::cuMemsetD8Async(
                                ptr as cudarc::driver::sys::CUdeviceptr, 0, conv_bytes, stream);
                        }
                        if let Some(s) = state.s_state[li].as_ref() {
                            let ptr = *s.device_ptr() as u64 + (slot * s_bytes) as u64;
                            cudarc::driver::sys::cuMemsetD8Async(
                                ptr as cudarc::driver::sys::CUdeviceptr, 0, s_bytes, stream);
                        }
                    }
                }
                LayerType::FullAttention => {
                    // KV does NOT need zeroing on a cold admit: every attention consumer is
                    // read-bounded at pos/pc — gqa_attn_splitk caps keys at min(split, pc),
                    // gqa_attn_prefill uses pc = pos_start + t + 1, attn_prefill_tiled is
                    // kmax-bounded, write_kv only writes — and prefill writes 0..plen before
                    // anything reads. This memset was ~64 KB x kv_stride per layer of dead work on
                    // EVERY cold admit (~2.4 ms at 8K, ~9.4 ms at 32K of pure TTFT on 9B; ~2x on
                    // 27B), and `new_batch_state` uses alloc_zeros so this was the only zeroing.
                    // RUST_INFER_ZERO_KV=1 restores it (A/B escape hatch). The GDN conv/recurrent
                    // state above DOES need zeroing — the recurrence reads it every step.
                    if zero_kv_enabled() {
                        let kv_bytes = nkv * kv_stride * hd * 2;
                        unsafe {
                            if let Some(s) = state.k_cache[li].as_ref() {
                                let ptr = *s.device_ptr() as u64 + (slot * kv_bytes) as u64;
                                cudarc::driver::sys::cuMemsetD8Async(
                                    ptr as cudarc::driver::sys::CUdeviceptr, 0, kv_bytes, stream);
                            }
                            if let Some(s) = state.v_cache[li].as_ref() {
                                let ptr = *s.device_ptr() as u64 + (slot * kv_bytes) as u64;
                                cudarc::driver::sys::cuMemsetD8Async(
                                    ptr as cudarc::driver::sys::CUdeviceptr, 0, kv_bytes, stream);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Batched prefill: process all N prompt tokens in one pass.
    /// GDN layers use sequential prefill kernels; full-attn uses batched causal attention.
    /// Returns the first generated token.
    /// Prefill `prompt` into `slot`, starting at absolute position `pos_start`.
    ///
    /// `pos_start > 0` is PREFIX REUSE: the slot's KV cache already holds positions [0, pos_start), and
    /// its GDN recurrent + conv1d states are already the states AFTER token pos_start-1. The caller has
    /// verified the cached tokens are an exact prefix of this prompt, and must NOT have zeroed the slot.
    ///
    /// The recurrent state is what makes this different from a pure-attention engine. KV is
    /// position-addressable — any prefix of it is meaningful on its own. The GDN state is NOT: it exists
    /// only at the single point in the sequence where it was left. So a prefix can only be reused from a
    /// slot that was carried to exactly that token and no further, which is why the cache is per-slot and
    /// keyed on the whole previous sequence rather than a shared radix tree.
    pub fn prefill_batch(&self, pool: &mut Pool, prompt: &[u32],
                         state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                         pos_start: usize) -> (u32, B) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        // TP-local attention heads when the mixers are sharded (matches the sharded qkv/o_proj and the
        // head-sharded KV cache these paths write into).
        let nh = self.eff_num_heads(); let nkv = self.eff_num_kv_heads(); let hd = cfg.head_dim; let rdim = cfg.rotary_dim;
        // TP-local GDN geometry (see eff_lin_v_heads). Same rule as the decode path: every dim on a
        // sharded path comes from an eff_* accessor, never from cfg directly.
        let lin_nh = self.eff_lin_v_heads(); let kd = cfg.lin_k_dim; let vd = cfg.lin_v_dim;
        let conv_dim = self.eff_conv_dim(); let ck = cfg.conv_kernel;
        let value_dim = self.eff_value_dim();
        let v = cfg.vocab_size;
        let n = prompt.len();
        let _max_pc = n;

        let pos_i32: Vec<i32> = (0..n).map(|i| (pos_start + i) as i32).collect();
        let pos_dev = self.dev.htod_sync_copy(&pos_i32).expect("htod pos");
        let cos = pool.get(n * rdim);
        let sin = pool.get(n * rdim);
        self.dev.synchronize().unwrap(); // NULL stream htod must complete before non-blocking stream reads
        blaunch!(self, "gather_rope_b", grid(n*rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *pos_dev.device_ptr() as u64, rdim as i32, n as i32));

        let residual = self.embed_batch(prompt);
        let normed = pool.get_bf16(h * n);
        let ckpt_slot: Option<usize> = None; // prefill never needs the MTP ping-pong checkpoint
        for (li, layer) in self.layers.iter().enumerate() {
            blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&normed), d(&residual), d(&layer.input_ln), h as i32, n as i32, fbits(cfg.rms_eps)));
            let mixer = match layer.layer_type {
                LayerType::LinearAttention => {
                    let la = layer.la.as_ref().unwrap();
                    let conv_off = (slot * conv_dim * ck * 4) as u64;
                    let s_off = (slot * lin_nh * kd * vd * 4) as u64;
                    let conv_ptr = *state.conv_state[li].as_ref().unwrap().device_ptr() + conv_off;
                    let s_ptr = *state.s_state[li].as_ref().unwrap().device_ptr() + s_off;
                    // MTP depth-2 ping-pong: if a checkpoint slot is given, the GDN kernels snapshot
                    // the recurrent state after the always-accepted committed token (S1) into that
                    // slot, so a rejected draft restores S1 with no second model forward.
                    let mid_conv_ptr: u64 = match ckpt_slot {
                        Some(ss) => *state.conv_state[li].as_ref().unwrap().device_ptr()
                            + (ss as u64 * conv_dim as u64 * ck as u64 * 4),
                        None => 0,
                    };
                    let mid_s_ptr: u64 = match ckpt_slot {
                        Some(ss) => *state.s_state[li].as_ref().unwrap().device_ptr()
                            + (ss as u64 * lin_nh as u64 * kd as u64 * vd as u64 * 4),
                        None => 0,
                    };
                    let mut qkv = pool.get_bf16(conv_dim * n);
                    let mut z = pool.get_bf16(value_dim * n);
                    let mut b = pool.get_bf16(lin_nh * n);
                    let mut a = pool.get_bf16(lin_nh * n);
                    match &la.in_proj {
                        GdnIn::Fused(w) => {
                            let mtot = self.eff_gdn_fused_m();
                            let mut fused = pool.get_bf16(mtot * n);
                            self.gemm_act(w, &normed, &mut fused, h, mtot, n);
                            blaunch!(self, "split_gdn_b", grid(mtot*n), (256,1,1), 0,
                                (d(&qkv), d(&z), d(&b), d(&a), d(&fused),
                                 conv_dim as i32, value_dim as i32, lin_nh as i32, n as i32,
                                 self.cfg.lin_num_v_heads as i32, self.gdn_head0() as i32));
                            pool.release_bf16(fused, mtot * n);
                        }
                        GdnIn::Split { qkv: pq, z: pz, b: pb, a: pa } => {
                            self.gemm_act(pq, &normed, &mut qkv, h, conv_dim, n);
                            self.gemm_act(pz, &normed, &mut z, h, value_dim, n);
                            self.gemm_act(pb, &normed, &mut b, h, lin_nh, n);
                            self.gemm_act(pa, &normed, &mut a, h, lin_nh, n);
                        }
                    }
                    // conv1d is a causal depthwise STENCIL, not a recurrence: out[t] reads in[t-3..t].
                    // It is now fully parallel over (channel, position), which forced it out of place --
                    // a thread reads inputs its neighbours are writing. So: convolve into a fresh buffer,
                    // swap, and carry the final window back into `state` in a separate launch (the main
                    // kernel reads `state` from every thread; writing it there would race).
                    let mut convo = pool.get_bf16(conv_dim * n);
                    blaunch!(self, "conv1d_prefill", grid(conv_dim * n), (256,1,1), 0,
                        (d(&convo), d(&qkv), conv_ptr, d(&la.conv1d),
                         conv_dim as i32, ck as i32, n as i32, mid_conv_ptr, 0u64, 0u64));
                    blaunch!(self, "conv1d_prefill_state", grid(conv_dim), (256,1,1), 0,
                        (conv_ptr, d(&qkv), conv_dim as i32, ck as i32, (n - 1) as i32, n as i32, 0u64));
                    std::mem::swap(&mut qkv, &mut convo);
                    pool.release_bf16(convo, conv_dim * n);
                    let core = pool.get_bf16(lin_nh * vd * n);
                    let (nchunk, smem) = gdn_launch(kd, vd);
                    blaunch!(self, "delta_step_prefill", ((lin_nh*nchunk) as u32,1,1), (kd as u32,1,1), smem,
                        (d(&core), d(&qkv), s_ptr, d(&b), d(&a), lin_nh as i32, (kd as i32) | ((vd as i32) << 16),
                         d(&la.a_log), d(&la.dt_bias), (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24),
                         mid_s_ptr, 0u64));
                    let gnormed = pool.get_bf16(value_dim * n);
                    blaunch!(self, "rmsnorm_gated_b", ((n*lin_nh) as u32,1,1), (vd as u32,1,1), (vd*4) as u32,
                        (d(&gnormed), d(&core), d(&z), d(&la.norm), vd as i32, lin_nh as i32, n as i32, fbits(cfg.rms_eps)));
                    let mut out = pool.get_bf16(h * n);
                    if self.tp_shard_mixers() {
                        self.tp_reduce_site(pool, &la.out_proj, &gnormed, &mut out, value_dim, h, n);
                    } else {
                        self.gemm_act(&la.out_proj, &gnormed, &mut out, value_dim, h, n);
                    }
                    pool.release_bf16(qkv, conv_dim*n); pool.release_bf16(z, value_dim*n);
                    pool.release_bf16(b, lin_nh*n); pool.release_bf16(a, lin_nh*n);
                    pool.release_bf16(core, lin_nh*vd*n); pool.release_bf16(gnormed, value_dim*n);
                    out
                }
                LayerType::FullAttention => {
                    let fa = layer.fa.as_ref().unwrap();
                    let kv_elems = nkv * kv_stride * hd;
                    let off = (slot * kv_elems * 2) as u64;
                    let kc_ptr = *state.k_cache[li].as_ref().unwrap().device_ptr() + off;
                    let vc_ptr = *state.v_cache[li].as_ref().unwrap().device_ptr() + off;
                    let mut qg = pool.get_bf16(nh*hd*2*n);
                    let mut k = pool.get_bf16(nkv*hd*n);
                    let mut v = pool.get_bf16(nkv*hd*n);
                    let q = pool.get_bf16(nh*hd*n);
                    let gate = pool.get_bf16(nh*hd*n);
                    match &fa.qkv {
                        AttnIn::Fused(w) => {
                            let mtot = if self.tp_shard_mixers() { nh*hd*2 + 2*nkv*hd } else { AttnIn::fused_m(cfg) };
                            let mut fused = pool.get_bf16(mtot * n);
                            self.gemm_act(w, &normed, &mut fused, h, mtot, n);
                            blaunch!(self, "split_qkv_b", grid(mtot*n), (256,1,1), 0,
                                (d(&qg), d(&k), d(&v), d(&fused),
                                 (nh*hd*2) as i32, (nkv*hd) as i32, n as i32));
                            pool.release_bf16(fused, mtot * n);
                        }
                        AttnIn::Split { q: qp, k: kp, v: vp } => {
                            self.gemm_act(qp, &normed, &mut qg, h, nh*hd*2, n);
                            self.gemm_act(kp, &normed, &mut k, h, nkv*hd, n);
                            self.gemm_act(vp, &normed, &mut v, h, nkv*hd, n);
                        }
                    }
                    blaunch!(self, "split_qgate_b", grid(nh*hd*n), (256,1,1), 0, (d(&q), d(&gate), d(&qg), nh as i32, hd as i32, n as i32));
                    blaunch!(self, "rmsnorm_perhead_b", ((n*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(&q), d(&q), d(&fa.q_norm), nh as i32, hd as i32, n as i32, fbits(cfg.rms_eps)));
                    blaunch!(self, "rmsnorm_perhead_b", ((n*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(&k), d(&k), d(&fa.k_norm), nkv as i32, hd as i32, n as i32, fbits(cfg.rms_eps)));
                    blaunch!(self, "rope_b", grid(n*nh*(rdim/2)), (256,1,1), 0, (d(&q), d(&cos), d(&sin), nh as i32, hd as i32, rdim as i32, n as i32));
                    blaunch!(self, "rope_b", grid(n*nkv*(rdim/2)), (256,1,1), 0, (d(&k), d(&cos), d(&sin), nkv as i32, hd as i32, rdim as i32, n as i32));
                    blaunch!(self, "write_kv_prefill", grid(n*nkv*hd), (256,1,1), 0, (kc_ptr, vc_ptr, d(&k), d(&v), kv_stride as i32, nkv as i32, hd as i32, n as i32, pos_start as i32));
                    let mut attn = pool.get_bf16(nh*hd*n);
                    let scale = 1.0f32/(hd as f32).sqrt();
                    // [NW][hd] partial accumulators + [NW] m + [NW] l, for the final softmax merge.
                    // The INNER loop needs no shared memory at all — it is warp shuffles and registers.
                    let nw = (hd / 32) as u32;                 // warps per block (blockDim = hd = 256)
                    let smem = (nw * hd as u32 + 2 * nw) * 4;
                    if n >= PF_MIN && !prefill_scalar() {
                        self.attn_prefill_tiled(pool, &q, kc_ptr, vc_ptr, kv_stride, nh, nkv, n, pos_start, &mut attn);
                    } else {
                        let ntile = (n + GQA_PF_QT - 1) / GQA_PF_QT;
                        blaunch!(self, "gqa_attn_prefill", ((ntile*nh) as u32,1,1), (hd as u32,1,1), 0, (d(&attn), d(&q), kc_ptr, vc_ptr, kv_stride as i32, nh as i32, nkv as i32, hd as i32, fbits(scale), n as i32, pos_start as i32));
                    }
                    blaunch!(self, "sigmoid_gate_b", grid(nh*hd*n), (256,1,1), 0, (d(&attn), d(&gate), (nh*hd*n) as i32));
                    let mut out = pool.get_bf16(h*n);
                    if self.tp_shard_mixers() {
                        self.tp_reduce_site(pool, &fa.o_proj, &attn, &mut out, nh*hd, h, n);
                    } else {
                        self.gemm_act(&fa.o_proj, &attn, &mut out, nh*hd, h, n);
                    }
                    pool.release_bf16(qg, nh*hd*2*n); pool.release_bf16(k, nkv*hd*n); pool.release_bf16(v, nkv*hd*n);
                    pool.release_bf16(q, nh*hd*n); pool.release_bf16(gate, nh*hd*n); pool.release_bf16(attn, nh*hd*n);
                    out
                }
            };
            blaunch!(self, "fused_res_rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&normed), d(&residual), d(&mixer), d(&layer.post_ln), h as i32, n as i32, fbits(cfg.rms_eps)));
            let mlp_out = self.ffn_batch(pool, &normed, &layer.mlp, n, self.tp_world == 2);
            let tot = h * n;
            blaunch!(self, "add_residual_b", grid(tot), (256,1,1), 0, (d(&residual), d(&residual), d(&mlp_out), tot as i32));
            pool.release_bf16(mixer, h*n); pool.release_bf16(mlp_out, h*n);
        }
        let out = pool.get_bf16(h * n);
        blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&out), d(&residual), d(&self.final_norm), h as i32, n as i32, fbits(cfg.rms_eps)));
        pool.release_bf16(normed, h*n);
        // NOTE: `residual` (the pre-final-RMSNorm backbone hidden) is kept and returned — the MTP
        // head consumes the PRE-norm hidden (it applies its own pre_fc_norm_hidden). Returning the
        // post-norm `out` would double-normalize and materially cut MTP acceptance.

        // Argmax at last position only — so run the LM head (the model's single largest GEMM) on
        // the LAST hidden column alone, not on all n positions. Previously this computed full-vocab
        // logits for every prompt position and discarded columns 0..n-2 unread: at an 8K window that
        // is a 248320×8192 GEMM of waste plus a multi-GB logits buffer and its first-touch memset,
        // paid PER prefill window. Column n-1 of `out` ([h, n] column-major) is contiguous.
        let mut last = pool.get_bf16(h);
        self.copy_hidden_col(d(&last), &out, n - 1);
        let logits = self.logits_batch(pool, &last, 1);
        let block = 1024u32;
        let token_id_dev = self.dev.alloc_zeros::<i32>(1).unwrap();
        blaunch!(self, "argmax_b", (1,1,1), (block,1,1), (block*8),
            (*token_id_dev.device_ptr() as u64, d(&logits), v as i32, 1));
        pool.release_bf16(logits, v);
        pool.release_bf16(last, h);
        pool.release_bf16(out, h*n);
        self.sync_stream();
        let tid = self.dev.dtoh_sync_copy(&token_id_dev).unwrap();
        // Return first token AND the pre-final-norm [h, n] backbone hidden (MTP consumes the
        // PRE-norm hidden; caller extracts h at the last position to seed drafting).
        (tid[0] as u32, residual)
    }

    /// K-token causal-append forward — the MTP verify primitive. Processes `tokens` (K of them) at
    /// absolute positions pos_start..pos_start+K-1, APPENDING to the slot's existing state (KV cache
    /// + GDN recurrent state already populated for 0..pos_start-1). Does NOT zero state. The GDN
    /// prefill kernels are reused as-is: they advance conv/s_state from wherever it currently is.
    /// The full-attention kernels use pos_start so token i writes KV at pos_start+i and attends to
    /// the existing prefix [0..pos_start-1] plus the new causal block [0..i].
    /// Returns K per-position greedy predictions: pred[i] = argmax(logits at pos_start+i), i.e. the
    /// token each verify position predicts for the NEXT position (the MTP acceptance/rollback signal).
    pub fn verify_forward(&self, pool: &mut Pool, tokens: &[u32],
                          state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                          pos_start: usize, ckpt_slot: Option<usize>,
                          penalty: Option<VerifyPenalty>) -> (Vec<u32>, B) {
        self.verify_forward_topo(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, penalty, None)
    }

    /// `verify_forward` with an optional planted tree topology (fork-then-chain in serving).
    pub fn verify_forward_topo(&self, pool: &mut Pool, tokens: &[u32],
                          state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                          pos_start: usize, ckpt_slot: Option<usize>,
                          penalty: Option<VerifyPenalty>, topo: Option<&TreeTopo>) -> (Vec<u32>, B) {
        let cfg = &self.cfg;
        let vocab = cfg.vocab_size;
        let _h = cfg.hidden_size;
        let n = tokens.len();

        let (logits, residual) = self.verify_forward_core_topo(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, topo);

        // Optional repetition/presence/frequency penalty (so MTP greedy lanes keep their penalty →
        // no repetition). All n verify positions share the lane's committed-history penalty.
        if let Some(p) = penalty {
            blaunch!(self, "rep_penalty_b", (n as u32,1,1), (256,1,1), 0,
                (d(&logits), p.tokens_ptr, p.counts_ptr, MAX_PEN_TOKENS as i32,
                 p.rep_pen_ptr, p.presence_ptr, p.freq_ptr, vocab as i32, n as i32));
        }
        let block = 1024u32;
        blaunch!(self, "argmax_b", (n as u32,1,1), (block,1,1), (block*8),
            (*self.sc_tok.device_ptr() as u64, d(&logits), vocab as i32, n as i32));
        pool.release_bf16(logits, vocab*n);
        self.sync_stream();
        let tids = self.dev.dtoh_sync_copy(&self.sc_tok).unwrap();
        // Return predictions AND the pre-final-norm [h, n] backbone hidden (MTP consumes PRE-norm).
        (tids.into_iter().take(n).map(|x| x as u32).collect(), residual)
    }

    /// Shared backbone + logits for both greedy and stochastic verify paths. Returns
    /// (unpenalized logits [vocab, n], pre-final-norm residual [h, n]).
    fn verify_forward_core(&self, pool: &mut Pool, tokens: &[u32],
                           state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                           pos_start: usize, ckpt_slot: Option<usize>) -> (B, B) {
        self.verify_forward_core_topo(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, None)
    }

    /// Like `verify_forward_core`, but an optional PLANTED tree topology (rope/kv/parent/path/winsrc).
    /// `None` builds the chain-identity topology (the normal path). A planted tree is how the Step-2.9
    /// byte gates drive the verify with arbitrary tree shapes before any fork drafter exists.
    fn verify_forward_core_topo(&self, pool: &mut Pool, tokens: &[u32],
                           state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                           pos_start: usize, ckpt_slot: Option<usize>, topo: Option<&TreeTopo>) -> (B, B) {
        // A verify wider than MAX_VERIFY does not fail -- it QUIETLY FALLS OFF A CLIFF. `gemm_act`
        // dispatches batch > MAX_VERIFY to the PREFILL path, which (a) dequantizes every weight into
        // scratch on every single step, and (b) finishes on cuBLAS, which is NOT batch-invariant. So a
        // too-wide verify would be catastrophically slow AND would silently stop being lossless, with
        // every correctness gate still green because the gates never run that wide.
        //
        // This matters now: tree drafting makes wide verifies routine, and the width is a function of
        // depth x branching. Assert, do not hope.
        assert!(tokens.len() <= MAX_VERIFY,
                "verify width {} exceeds MAX_VERIFY {} -- it would fall to the prefill dequant path \
                 and silently stop being batch-invariant", tokens.len(), MAX_VERIFY);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        // TP-local attention heads when the mixers are sharded (matches the sharded qkv/o_proj and the
        // head-sharded KV cache these paths write into).
        let nh = self.eff_num_heads(); let nkv = self.eff_num_kv_heads(); let hd = cfg.head_dim; let rdim = cfg.rotary_dim;
        // TP-local GDN geometry (see eff_lin_v_heads). Same rule as the decode path: every dim on a
        // sharded path comes from an eff_* accessor, never from cfg directly.
        let lin_nh = self.eff_lin_v_heads(); let kd = cfg.lin_k_dim; let vd = cfg.lin_v_dim;
        let conv_dim = self.eff_conv_dim(); let ck = cfg.conv_kernel;
        let value_dim = self.eff_value_dim();
        let n = tokens.len();
        // A chain: KV position and RoPE position coincide (both pos_start+j). A tree splits them —
        // siblings share a RoPE position (their tree depth) but occupy distinct KV positions. Kept as
        // two arrays now, populated identically, so the split is proven byte-identical before the tree
        // ever makes them diverge. `sc_pos` is the KV position (write offset + causal bound); `sc_rope`
        // is the RoPE angle.
        // Chain-identity defaults; a planted `topo` overrides pos/rope/parent/path/winsrc.
        let chain_pos: Vec<i32> = (pos_start as i32..(pos_start + n) as i32).collect();
        let pos_i32: Vec<i32> = topo.map(|t| t.kv_pos.clone()).unwrap_or_else(|| chain_pos.clone());
        let rope_i32: Vec<i32> = topo.map(|t| t.rope.clone()).unwrap_or_else(|| chain_pos.clone());
        // FOREST: per-column lane slot from the topo; a single tree/chain uses the uniform `slot`.
        let slot_ids: Vec<i32> = topo.and_then(|t| t.slot.clone()).unwrap_or_else(|| vec![slot as i32; n]);
        // FOREST: per-column prefix boundary (lane committed length). None => attention uses pos[0]
        // (single lane / tree), which is byte-identical to the pre-forest rank-space attention.
        let col_pos_start: Option<Vec<i32>> = topo.and_then(|t| t.col_pos_start.clone());
        unsafe {
            cudarc::driver::result::memcpy_htod_async(*self.sc_pos.device_ptr() as cudarc::driver::sys::CUdeviceptr, &pos_i32[..n], self.stream.stream).expect("htod pos");
            cudarc::driver::result::memcpy_htod_async(*self.sc_rope.device_ptr() as cudarc::driver::sys::CUdeviceptr, &rope_i32[..n], self.stream.stream).expect("htod rope");
            cudarc::driver::result::memcpy_htod_async(*self.sc_slot.device_ptr() as cudarc::driver::sys::CUdeviceptr, &slot_ids[..n], self.stream.stream).expect("htod slot");
            if let Some(cps) = col_pos_start.as_ref() {
                cudarc::driver::result::memcpy_htod_async(*self.sc_pstart.device_ptr() as cudarc::driver::sys::CUdeviceptr, &cps[..n], self.stream.stream).expect("htod pstart");
            }
        }
        let col_pos_start_ptr: u64 = if col_pos_start.is_some() { *self.sc_pstart.device_ptr() as u64 } else { 0 };
        // FOREST lane segments (start, len): a new lane begins at each scan ROOT (raw parent == -1). The
        // per-lane conv-state carry (conv1d_prefill_state) must run ONCE PER LANE — committing each lane's
        // final conv window to its own slot — else a non-last lane's conv state goes stale after a full
        // accept (the s-state is handled per-lane by delta_step_prefill's commit; conv was the gap). A
        // single tree/chain is one segment (0, n): byte-identical to the old single call.
        let lane_segs: Vec<(usize, usize)> = {
            let raw_par: Vec<i32> = topo.map(|t| t.parent.clone())
                .unwrap_or_else(|| (0..n as i32).map(|t| t - 1).collect());
            let mut segs = Vec::new();
            let mut c = 0usize;
            while c < n {
                let start = c;
                c += 1;
                while c < n && raw_par[c] != -1 { c += 1; }
                segs.push((start, c - start));
            }
            segs
        };
        self.sync_stream();
        // Conv window sources, CHAIN-identity: node t's width-k window is positions t..t+k-1, i.e.
        // win_src[t*k + j] = t + j - (k-1). Exactly reproduces the pre-tree stencil (a tree overwrites
        // this with the ancestor-path window). Built once per verify, reused across all GDN layers.
        {
            // winsrc: chain node t's window is positions t-(k-1)..t; a tree gives the ancestor-path window.
            let ws: Vec<i32> = topo.map(|t| t.winsrc.clone()).unwrap_or_else(|| {
                let mut ws = vec![0i32; n * ck];
                for t in 0..n { for j in 0..ck { ws[t*ck + j] = (t as i32) + (j as i32) - (ck as i32 - 1); } }
                ws
            });
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    *self.sc_winsrc.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                    &ws[..n*ck], self.stream.stream).expect("htod winsrc");
            }
            // parent: chain parent[t]=t-1 (parent[0]=-1); a tree gives the DFS parent. PACKED per column:
            // low 16 bits = DFS parent, high 16 bits = the column's lane slot (delta_step_prefill reads
            // both). Single lane: slot is uniform, so the low bits are the exact old parent and the high
            // bits just re-encode `slot`, keeping the GDN scan byte-identical.
            let par_raw: Vec<i32> = topo.map(|t| t.parent.clone()).unwrap_or_else(|| (0..n as i32).map(|t| t - 1).collect());
            let par: Vec<i32> = (0..n)
                .map(|t| (((slot_ids[t] as u32 & 0xFFFF) << 16) | (par_raw[t] as u32 & 0xFFFF)) as i32)
                .collect();
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    *self.sc_parent.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                    &par[..n], self.stream.stream).expect("htod parent");
            }
            // path: chain path[b][d]=d; a tree gives the ancestor KV-slot offsets (slot-pos_start).
            let pth: Vec<u8> = topo.map(|t| t.path.clone()).unwrap_or_else(|| {
                let mut pth = vec![0u8; n * MAX_VERIFY];
                for b in 0..n { for dd in 0..MAX_VERIFY { pth[b*MAX_VERIFY + dd] = dd as u8; } }
                pth
            });
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    *self.sc_path.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                    &pth[..n*MAX_VERIFY], self.stream.stream).expect("htod path");
            }
        }
        let pos_dev_ptr = *self.sc_pos.device_ptr() as u64;
        let rope_dev_ptr = *self.sc_rope.device_ptr() as u64;
        let slot_ids_ptr = *self.sc_slot.device_ptr() as u64;
        let winsrc_ptr = *self.sc_winsrc.device_ptr() as u64;
        let parent_ptr = *self.sc_parent.device_ptr() as u64;
        let path_ptr = *self.sc_path.device_ptr() as u64;
        let cos = pool.get(n * rdim);
        let sin = pool.get(n * rdim);
        blaunch!(self, "gather_rope_b", grid(n*rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             rope_dev_ptr, rdim as i32, n as i32));

        let residual = self.embed_batch(tokens);
        let normed = pool.get_bf16(h * n);
        for (li, layer) in self.layers.iter().enumerate() {
            blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&normed), d(&residual), d(&layer.input_ln), h as i32, n as i32, fbits(cfg.rms_eps)));
            let mixer = match layer.layer_type {
                LayerType::LinearAttention => {
                    let la = layer.la.as_ref().unwrap();
                    // Forest verify: pass the layer BASE pointers; the GDN prefill kernels apply the
                    // per-column lane offset from `slot_ids` (sc_slot). Single-lane: sc_slot is uniform,
                    // so `slot_ids[t]*stride` reproduces the old pre-offset `slot*stride` byte-for-byte.
                    let conv_ptr = *state.conv_state[li].as_ref().unwrap().device_ptr();
                    let s_ptr = *state.s_state[li].as_ref().unwrap().device_ptr();
                    let mid_conv_ptr: u64 = match ckpt_slot {
                        Some(ss) => *state.conv_state[li].as_ref().unwrap().device_ptr()
                            + (ss as u64 * conv_dim as u64 * ck as u64 * 4),
                        None => 0,
                    };
                    let mid_s_ptr: u64 = match ckpt_slot {
                        Some(ss) => *state.s_state[li].as_ref().unwrap().device_ptr()
                            + (ss as u64 * lin_nh as u64 * kd as u64 * vd as u64 * 4),
                        None => 0,
                    };
                    let mut qkv = pool.get_bf16(conv_dim * n);
                    let mut z = pool.get_bf16(value_dim * n);
                    let mut b = pool.get_bf16(lin_nh * n);
                    let mut a = pool.get_bf16(lin_nh * n);
                    match &la.in_proj {
                        GdnIn::Fused(w) => {
                            let mtot = self.eff_gdn_fused_m();
                            let mut fused = pool.get_bf16(mtot * n);
                            self.gemm_act(w, &normed, &mut fused, h, mtot, n);
                            blaunch!(self, "split_gdn_b", grid(mtot*n), (256,1,1), 0,
                                (d(&qkv), d(&z), d(&b), d(&a), d(&fused),
                                 conv_dim as i32, value_dim as i32, lin_nh as i32, n as i32,
                                 self.cfg.lin_num_v_heads as i32, self.gdn_head0() as i32));
                            pool.release_bf16(fused, mtot * n);
                        }
                        GdnIn::Split { qkv: pq, z: pz, b: pb, a: pa } => {
                            self.gemm_act(pq, &normed, &mut qkv, h, conv_dim, n);
                            self.gemm_act(pz, &normed, &mut z, h, value_dim, n);
                            self.gemm_act(pb, &normed, &mut b, h, lin_nh, n);
                            self.gemm_act(pa, &normed, &mut a, h, lin_nh, n);
                        }
                    }
                    // conv1d is a causal depthwise STENCIL, not a recurrence: out[t] reads in[t-3..t].
                    // It is now fully parallel over (channel, position), which forced it out of place --
                    // a thread reads inputs its neighbours are writing. So: convolve into a fresh buffer,
                    // swap, and carry the final window back into `state` in a separate launch (the main
                    // kernel reads `state` from every thread; writing it there would race).
                    let mut convo = pool.get_bf16(conv_dim * n);
                    blaunch!(self, "conv1d_prefill", grid(conv_dim * n), (256,1,1), 0,
                        (d(&convo), d(&qkv), conv_ptr, d(&la.conv1d),
                         conv_dim as i32, ck as i32, n as i32, mid_conv_ptr, winsrc_ptr, slot_ids_ptr));
                    // Per-lane conv-state carry: commit each lane's final window to ITS slot (see lane_segs).
                    for &(seg_start, seg_len) in &lane_segs {
                        blaunch!(self, "conv1d_prefill_state", grid(conv_dim), (256,1,1), 0,
                            (conv_ptr, d(&qkv), conv_dim as i32, ck as i32,
                             (seg_start + seg_len - 1) as i32, seg_len as i32, slot_ids_ptr));
                    }
                    std::mem::swap(&mut qkv, &mut convo);
                    pool.release_bf16(convo, conv_dim * n);
                    let core = pool.get_bf16(lin_nh * vd * n);
                    let (nchunk, smem) = gdn_launch(kd, vd);
                    blaunch!(self, "delta_step_prefill", ((lin_nh*nchunk) as u32,1,1), (kd as u32,1,1), smem,
                        (d(&core), d(&qkv), s_ptr, d(&b), d(&a), lin_nh as i32, (kd as i32) | ((vd as i32) << 16),
                         d(&la.a_log), d(&la.dt_bias), (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24),
                         mid_s_ptr, parent_ptr));
                    let gnormed = pool.get_bf16(value_dim * n);
                    blaunch!(self, "rmsnorm_gated_b", ((n*lin_nh) as u32,1,1), (vd as u32,1,1), (vd*4) as u32,
                        (d(&gnormed), d(&core), d(&z), d(&la.norm), vd as i32, lin_nh as i32, n as i32, fbits(cfg.rms_eps)));
                    let mut out = pool.get_bf16(h * n);
                    if self.tp_shard_mixers() {
                        self.tp_reduce_site(pool, &la.out_proj, &gnormed, &mut out, value_dim, h, n);
                    } else {
                        self.gemm_act(&la.out_proj, &gnormed, &mut out, value_dim, h, n);
                    }
                    pool.release_bf16(qkv, conv_dim*n); pool.release_bf16(z, value_dim*n);
                    pool.release_bf16(b, lin_nh*n); pool.release_bf16(a, lin_nh*n);
                    pool.release_bf16(core, lin_nh*vd*n); pool.release_bf16(gnormed, value_dim*n);
                    out
                }
                LayerType::FullAttention => {
                    let fa = layer.fa.as_ref().unwrap();
                    let kc_ptr = *state.k_cache[li].as_ref().unwrap().device_ptr();
                    let vc_ptr = *state.v_cache[li].as_ref().unwrap().device_ptr();
                    let mut qg = pool.get_bf16(nh*hd*2*n);
                    let mut k = pool.get_bf16(nkv*hd*n);
                    let mut vbuf = pool.get_bf16(nkv*hd*n);
                    let q = pool.get_bf16(nh*hd*n);
                    let gate = pool.get_bf16(nh*hd*n);
                    match &fa.qkv {
                        AttnIn::Fused(w) => {
                            let mtot = if self.tp_shard_mixers() { nh*hd*2 + 2*nkv*hd } else { AttnIn::fused_m(cfg) };
                            let mut fused = pool.get_bf16(mtot * n);
                            self.gemm_act(w, &normed, &mut fused, h, mtot, n);
                            blaunch!(self, "split_qkv_b", grid(mtot*n), (256,1,1), 0,
                                (d(&qg), d(&k), d(&vbuf), d(&fused),
                                 (nh*hd*2) as i32, (nkv*hd) as i32, n as i32));
                            pool.release_bf16(fused, mtot * n);
                        }
                        AttnIn::Split { q: qp, k: kp, v: vp } => {
                            self.gemm_act(qp, &normed, &mut qg, h, nh*hd*2, n);
                            self.gemm_act(kp, &normed, &mut k, h, nkv*hd, n);
                            self.gemm_act(vp, &normed, &mut vbuf, h, nkv*hd, n);
                        }
                    }
                    blaunch!(self, "split_qgate_b", grid(nh*hd*n), (256,1,1), 0, (d(&q), d(&gate), d(&qg), nh as i32, hd as i32, n as i32));
                    blaunch!(self, "rmsnorm_perhead_b", ((n*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(&q), d(&q), d(&fa.q_norm), nh as i32, hd as i32, n as i32, fbits(cfg.rms_eps)));
                    blaunch!(self, "rmsnorm_perhead_b", ((n*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(&k), d(&k), d(&fa.k_norm), nkv as i32, hd as i32, n as i32, fbits(cfg.rms_eps)));
                    blaunch!(self, "rope_b", grid(n*nh*(rdim/2)), (256,1,1), 0, (d(&q), d(&cos), d(&sin), nh as i32, hd as i32, rdim as i32, n as i32));
                    blaunch!(self, "rope_b", grid(n*nkv*(rdim/2)), (256,1,1), 0, (d(&k), d(&cos), d(&sin), nkv as i32, hd as i32, rdim as i32, n as i32));
                    let attn = self.attn_dispatch(pool, &q, &k, &vbuf, &self.sc_pos, kc_ptr, vc_ptr,
                        slot_ids_ptr, pos_start + n, kv_stride, n, None, path_ptr, rope_dev_ptr, col_pos_start_ptr, self.tp_shard_mixers());
                    blaunch!(self, "sigmoid_gate_b", grid(nh*hd*n), (256,1,1), 0, (d(&attn), d(&gate), (nh*hd*n) as i32));
                    let mut out = pool.get_bf16(h*n);
                    if self.tp_shard_mixers() {
                        self.tp_reduce_site(pool, &fa.o_proj, &attn, &mut out, nh*hd, h, n);
                    } else {
                        self.gemm_act(&fa.o_proj, &attn, &mut out, nh*hd, h, n);
                    }
                    pool.release_bf16(qg, nh*hd*2*n); pool.release_bf16(k, nkv*hd*n); pool.release_bf16(vbuf, nkv*hd*n);
                    pool.release_bf16(q, nh*hd*n); pool.release_bf16(gate, nh*hd*n); pool.release_bf16(attn, nh*hd*n);
                    out
                }
            };
            blaunch!(self, "fused_res_rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&normed), d(&residual), d(&mixer), d(&layer.post_ln), h as i32, n as i32, fbits(cfg.rms_eps)));
            let mlp_out = self.ffn_batch(pool, &normed, &layer.mlp, n, self.tp_world == 2);
            let tot = h * n;
            blaunch!(self, "add_residual_b", grid(tot), (256,1,1), 0, (d(&residual), d(&residual), d(&mlp_out), tot as i32));
            pool.release_bf16(mixer, h*n); pool.release_bf16(mlp_out, h*n);
        }
        let out = pool.get_bf16(h * n);
        blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&out), d(&residual), d(&self.final_norm), h as i32, n as i32, fbits(cfg.rms_eps)));
        pool.release_bf16(normed, h*n);
        pool.release(cos, n*rdim);
        pool.release(sin, n*rdim);

        let logits = self.logits_batch(pool, &out, n);
        pool.release_bf16(out, h*n);
        (logits, residual)
    }

    /// Stochastic MTP verify forward: like verify_forward but instead of argmax on every column,
    /// computes per-position target probabilities + residual resamples + bonus token for the
    /// speculative rejection-sampling accept loop. Returns (VerifySample, vout [h, n]).
    pub fn verify_forward_sample(&self, pool: &mut Pool, tokens: &[u32],
                                  state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                                  pos_start: usize, ckpt_slot: Option<usize>,
                                  penalty: Option<VerifyPenalty>,
                                  draft_tokens: &[u32], draft_qprobs: &[f32],
                                  temp: f32, top_k: usize, top_p: f32,
                                  seeds: &[u32]) -> (VerifySample, B) {
        let cfg = &self.cfg;
        let _h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let n = tokens.len();
        assert_eq!(draft_tokens.len(), n - 1, "draft_tokens must be length depth-1");
        assert_eq!(draft_qprobs.len(), n - 1, "draft_qprobs must be length depth-1");
        assert_eq!(seeds.len(), n, "seeds must be length depth");

        let (logits, residual) = self.verify_forward_core(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot);

        // Apply penalty (same as verify_forward).
        if let Some(p) = &penalty {
            blaunch!(self, "rep_penalty_b", (n as u32,1,1), (256,1,1), 0,
                (d(&logits), p.tokens_ptr, p.counts_ptr, MAX_PEN_TOKENS as i32,
                 p.rep_pen_ptr, p.presence_ptr, p.freq_ptr, vocab as i32, n as i32));
        }

        // Persistent scratch, packed and uploaded on the COMPUTE stream. Everything here used to be
        // a fresh alloc_zeros + a synchronous NULL-stream htod + a drop, per decode step; against a
        // blocking compute stream each of those serializes the pipeline. Two packed async uploads
        // and one readback replace six of each.
        assert!(n <= MAX_VERIFY, "verify width {} exceeds MAX_VERIFY {}", n, MAX_VERIFY);
        let mut pf = vec![0.0f32; 3 * MAX_VERIFY];   // temps | top_ps | draft_qprobs
        let mut ki = vec![0i32; 2 * MAX_VERIFY];     // draft_tokens | top_ks
        for j in 0..n {
            pf[j] = temp;
            pf[MAX_VERIFY + j] = top_p;
            ki[MAX_VERIFY + j] = top_k as i32;
        }
        for j in 0..n - 1 {
            pf[2 * MAX_VERIFY + j] = draft_qprobs[j];
            ki[j] = draft_tokens[j] as i32;
        }
        let dptr = |s: &cudarc::driver::CudaSlice<f32>| *s.device_ptr() as u64;
        unsafe {
            use cudarc::driver::sys::CUdeviceptr;
            cudarc::driver::result::memcpy_htod_async(
                *self.sv_pf.device_ptr() as CUdeviceptr, &pf[..], self.stream.stream).expect("htod pf");
            cudarc::driver::result::memcpy_htod_async(
                *self.sv_ki.device_ptr() as CUdeviceptr, &ki[..], self.stream.stream).expect("htod ki");
            cudarc::driver::result::memcpy_htod_async(
                *self.sv_sd.device_ptr() as CUdeviceptr, &seeds[..n], self.stream.stream).expect("htod sd");
        }
        let temps_ptr  = dptr(&self.sv_pf);
        let topp_ptr   = temps_ptr + (MAX_VERIFY * 4) as u64;
        let qprob_ptr  = temps_ptr + (2 * MAX_VERIFY * 4) as u64;
        let dtok_ptr   = *self.sv_ki.device_ptr() as u64;
        let topk_ptr   = dtok_ptr + (MAX_VERIFY * 4) as u64;

        let smem = ((2 * 64usize + 2 * 256usize) * 4) as u32;
        blaunch!(self, "spec_verify_b", (n as u32,1,1), (256,1,1), smem,
            (*self.sv_p.device_ptr() as u64, *self.sv_r.device_ptr() as u64,
             d(&logits), dtok_ptr, qprob_ptr,
             temps_ptr, topk_ptr, topp_ptr, *self.sv_sd.device_ptr() as u64,
             vocab as i32, n as i32));
        pool.release_bf16(logits, vocab*n);
        self.sync_stream();

        // One readback each for the f32 and i32 outputs (the bonus rides in resid_tok[n-1]).
        let p_vec = self.dev.dtoh_sync_copy(&self.sv_p).unwrap();
        let r_vec = self.dev.dtoh_sync_copy(&self.sv_r).unwrap();

        let p_of_draft: Vec<f32> = p_vec[..n - 1].to_vec();
        let resid_tok: Vec<u32> = r_vec[..n - 1].iter().map(|&x| x as u32).collect();
        let bonus_tok = r_vec[n - 1] as u32;
        (VerifySample { p_of_draft, resid_tok, bonus_tok }, residual)
    }

    /// Lossless-correctness probe for `verify_forward` (the MTP verify primitive). Prefills the same
    /// prompt into slots 0 and 1 (identical starting state), runs `depth` sequential greedy decodes
    /// on slot 0 (ground truth), then runs `verify_forward` on slot 1 over the first `depth`
    /// ground-truth tokens. The verify predictions must equal the ground-truth tokens shifted by one:
    /// verify token at position plen+i predicts the token at plen+i+1, so preds[i] must equal
    /// seq_tokens[i+1]. This is the non-negotiable lossless gate — the K-token append path (using
    /// delta_step_prefill + gqa_attn_prefill) must match the 1-token decode path (delta_step_b +
    /// gqa_attn_splitk) token-for-token. Returns (seq_tokens, preds, per_position_match).
    pub fn bench_verify(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                        kv_stride: usize, depth: usize) -> (Vec<u32>, Vec<u32>, Vec<bool>) {
        let max_batch = 2usize;
        let mut bufs = self.new_decode_buffers(max_batch);

        // Prefill the same prompt into slots 0 and 1 (identical starting state).
        let h = self.cfg.hidden_size;
        self.zero_slot_state(state, 0, kv_stride);
        let (s0a, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * prompt.len());
        self.zero_slot_state(state, 1, kv_stride);
        let (s0b, h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        pool.release_bf16(h1, h * prompt.len());
        assert_eq!(s0a, s0b, "prefill divergence between slot 0 and slot 1");

        let plen = prompt.len();
        let mut seq_tokens = vec![s0a];   // token at position plen

        // Sequential greedy decode on slot 0 (batch=1): produce ground-truth tokens.
        let mut pos = plen;
        for _ in 0..depth {
            let tok = *seq_tokens.last().unwrap() as i32;
            let toks_i32 = vec![tok, 0];
            let pos_i32 = vec![pos as i32, 0];
            self.dev.htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
            // slot_ids_dev stays identity [0,1] from new_decode_buffers; lane 0 -> physical slot 0.
            self.dev.synchronize().unwrap();
            let max_pc = pos + 1;
            let next = self.forward_decode(pool, &mut bufs, state, kv_stride, max_pc, 1);
            seq_tokens.push(next[0]);
            pos += 1;
        }

        // Verify on slot 1: feed [s_0..s_{depth-1}] at positions plen..plen+depth-1.
        let verify_input: Vec<u32> = seq_tokens[..depth].to_vec();
        let (preds, vout) = self.verify_forward(pool, &verify_input, state, 1, kv_stride, plen, None, None);
        pool.release_bf16(vout, h * depth);

        // preds[i] must equal seq_tokens[i+1] (token predicted at pos plen+i is the one at plen+i+1).
        let matches: Vec<bool> = (0..depth).map(|i| preds[i] == seq_tokens[i + 1]).collect();
        (seq_tokens, preds, matches)
    }

    /// State-divergence probe: prefills both slots identically, advances slot 0 by ONE
    /// `verify_forward(N tokens)` call and slot 1 by N individual `forward_decode` calls, then
    /// returns the max abs f32 diff over all GDN `s_state` (recurrent) entries between the two slots.
    /// If verify_forward (delta_step_prefill) and forward_decode (delta_step_b) produce identical
    /// recurrent state, the diff is 0. A nonzero diff localizes the divergence to the GDN recurrence
    /// path (or the projections feeding it) — distinguishing a real kernel bug from cuBLAS N=1-vs-N
    /// rounding. `n` is the verify batch size (1 = single-token append, 2 = two-token append).
    pub fn verify_state_diff(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                             kv_stride: usize, n: usize) -> f32 {
        let h = self.cfg.hidden_size;
        let mut bufs = self.new_decode_buffers(2);
        self.zero_slot_state(state, 0, kv_stride);
        let (s0a, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * prompt.len());
        self.zero_slot_state(state, 1, kv_stride);
        let (s0b, h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        pool.release_bf16(h1, h * prompt.len());
        assert_eq!(s0a, s0b);

        let plen = prompt.len();
        // Generate n ground-truth tokens by decoding on slot 0 first (to know the inputs), then we
        // will RE-zero slot 0 and use verify there. Decode n tokens on slot 0 to get the token list.
        let mut toks = vec![s0a];
        let mut pos = plen;
        for _ in 0..n {
            let t = *toks.last().unwrap() as i32;
            self.dev.htod_sync_copy_into(&[t, 0], &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&[pos as i32, 0], &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            let nx = self.forward_decode(pool, &mut bufs, state, kv_stride, pos + 1, 1);
            toks.push(nx[0]);
            pos += 1;
        }
        // Now slot 0 has been advanced by decode; RE-zero it and re-prefill so it's clean, then
        // advance it by ONE verify_forward of the same n tokens.
        let inputs: Vec<u32> = toks[..n].to_vec();
        self.zero_slot_state(state, 0, kv_stride);
        let (_s, hh) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(hh, h * prompt.len());
        let (_p, vout) = self.verify_forward(pool, &inputs, state, 0, kv_stride, plen, None, None);
        pool.release_bf16(vout, h * n);

        // Slot 1 was advanced only by the prefill above (the decode loop used lane 0 → slot 0, then
        // we re-zeroed slot 0). Advance slot 1 by n decodes with the same tokens.
        self.dev.htod_sync_copy_into(&[1i32, 0], &mut bufs.slot_ids_dev).unwrap();
        let mut s1pos = plen;
        for i in 0..n {
            let t = inputs[i] as i32;
            self.dev.htod_sync_copy_into(&[t, 0], &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&[s1pos as i32, 0], &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            let _nx = self.forward_decode(pool, &mut bufs, state, kv_stride, s1pos + 1, 1);
            s1pos += 1;
        }
        drop(bufs);

        // Compare slot 0 (verify path) vs slot 1 (decode path): s_state, conv_state, AND k/v cache.
        // All three must be ~0 for verify_forward to be a bit-identical stand-in for decode (the
        // lossless-MTP contract); s_state alone is insufficient.
        let lin_nh = self.cfg.lin_num_v_heads; let kd = self.cfg.lin_k_dim; let vd = self.cfg.lin_v_dim;
        let per_slot = lin_nh * kd * vd;
        let conv_dim = self.cfg.key_dim()*2 + self.cfg.value_dim(); let ck = self.cfg.conv_kernel;
        let per_slot_conv = conv_dim * ck;
        let nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;
        let per_slot_kv = nkv * kv_stride * hd;
        let mut worst_s = 0.0f32; let mut worst_conv = 0.0f32; let mut worst_kv = 0.0f32;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            match lt {
                LayerType::LinearAttention => {
                    let s = self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap();
                    let c = self.dev.dtoh_sync_copy(state.conv_state[li].as_ref().unwrap()).unwrap();
                    for j in 0..per_slot { let dd=(s[per_slot+j]-s[j]).abs(); if dd>worst_s{worst_s=dd;} }
                    for j in 0..per_slot_conv { let dd=(c[per_slot_conv+j]-c[j]).abs(); if dd>worst_conv{worst_conv=dd;} }
                }
                LayerType::FullAttention => {
                    let k = self.dev.dtoh_sync_copy(state.k_cache[li].as_ref().unwrap()).unwrap();
                    let v = self.dev.dtoh_sync_copy(state.v_cache[li].as_ref().unwrap()).unwrap();
                    for j in 0..per_slot_kv {
                        let dk=(half::bf16::to_f32(k[per_slot_kv+j])-half::bf16::to_f32(k[j])).abs(); if dk>worst_kv{worst_kv=dk;}
                        let dv=(half::bf16::to_f32(v[per_slot_kv+j])-half::bf16::to_f32(v[j])).abs(); if dv>worst_kv{worst_kv=dv;}
                    }
                }
            }
        }
        eprintln!("    [probe] N={}  s_state={}, conv_state={}, kv_cache={}", n, worst_s, worst_conv, worst_kv);
        worst_s
    }

    /// Three-way reject-path probe (expert advice §3/§5). Forces a rejection and checks whether the
    /// ping-pong checkpoint/rollback is bit-exact — the decisive reject-path equality test.
    ///   slot A (0): verify([committed, wrong_draft], ckpt=Some(2)) → snapshots S1 into slot 2
    ///   slot B (1): decode committed once → reference S1
    /// Compares (per GDN layer conv+s, and the logical KV region):
    ///   (a) snap(slot 2) vs reference(slot 1)   — must be 0; else the checkpoint WRITE is wrong
    ///   (b) restored(slot 0) vs snap(slot 2)     — must be 0; else the D2D rollback copy is wrong
    ///   (c) restored(slot 0) vs reference(slot 1)— must be 0; the decisive reject-path equality
    ///   (d) KV positions 0..plen (committed region) slot 0 vs slot 1 — must be 0
    /// With f32 state + D2D rollback, every diff must be EXACTLY 0; any nonzero is a discrete bug.
    pub fn probe_reject_path(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                             kv_stride: usize) {
        let h = self.cfg.hidden_size;
        let mut bufs = self.new_decode_buffers(2);
        self.zero_slot_state(state, 0, kv_stride);
        let (a0, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * prompt.len());
        self.zero_slot_state(state, 1, kv_stride);
        let (a1, h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        pool.release_bf16(h1, h * prompt.len());
        assert_eq!(a0, a1, "prefill divergence");
        let plen = prompt.len();
        let committed = a0;
        let wrong_draft = a0;  // deliberately "wrong" → forces nacc=0 reject in general

        // slot B (1): decode committed once → reference S1 (state after committed at pos plen).
        self.dev.htod_sync_copy_into(&[1i32, 0], &mut bufs.slot_ids_dev).unwrap();
        self.dev.htod_sync_copy_into(&[committed as i32, 0], &mut bufs.tokens_dev).unwrap();
        self.dev.htod_sync_copy_into(&[plen as i32, 0], &mut bufs.pos_dev).unwrap();
        self.dev.synchronize().unwrap();
        let _ref_tok = self.forward_decode(pool, &mut bufs, state, kv_stride, plen + 1, 1);

        // slot A (0): verify([committed, wrong_draft], ckpt=Some(2)) → snapshot S1 into slot 2.
        let vi = vec![committed, wrong_draft];
        let (preds, vout) = self.verify_forward(pool, &vi, state, 0, kv_stride, plen, Some(2), None);
        pool.release_bf16(vout, h * 2);
        eprintln!("[reject] committed={} wrong_draft={} verify preds={:?} (forcing nacc=0 reject)", committed, wrong_draft, preds);

        let lin_nh = self.cfg.lin_num_v_heads; let kd = self.cfg.lin_k_dim; let vd = self.cfg.lin_v_dim;
        let per_slot = lin_nh * kd * vd;
        let conv_dim = self.cfg.key_dim()*2 + self.cfg.value_dim(); let ck = self.cfg.conv_kernel;
        let per_slot_conv = conv_dim * ck;
        let nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;

        // (a) snap(slot 2) vs reference(slot 1), pre-rollback.
        let mut a_s = 0.0f32; let mut a_c = 0.0f32;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if matches!(lt, LayerType::LinearAttention) {
                let s = self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap();
                let c = self.dev.dtoh_sync_copy(state.conv_state[li].as_ref().unwrap()).unwrap();
                for j in 0..per_slot { let dd=(s[2*per_slot+j]-s[per_slot+j]).abs(); if dd>a_s{a_s=dd;} }
                for j in 0..per_slot_conv { let dd=(c[2*per_slot_conv+j]-c[per_slot_conv+j]).abs(); if dd>a_c{a_c=dd;} }
            }
        }
        eprintln!("[reject] (a) snap(slot2) vs ref(slot1):   s_state={}, conv={}", a_s, a_c);

        // Rollback: copy_gdn_slot(snap=2 → A=0), then sync before host reads.
        self.copy_gdn_slot(state, 2, 0);
        self.sync_stream();

        // (b) restored(slot 0) vs snap(slot 2); (c) restored(slot 0) vs ref(slot 1).
        let mut b_s = 0.0f32; let mut b_c = 0.0f32; let mut c_s = 0.0f32; let mut c_c = 0.0f32;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if matches!(lt, LayerType::LinearAttention) {
                let s = self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap();
                let c = self.dev.dtoh_sync_copy(state.conv_state[li].as_ref().unwrap()).unwrap();
                for j in 0..per_slot {
                    let (l, sn, rf) = (s[j], s[2*per_slot+j], s[per_slot+j]);
                    let d1=(l-sn).abs(); if d1>b_s{b_s=d1;}
                    let d2=(l-rf).abs(); if d2>c_s{c_s=d2;}
                }
                for j in 0..per_slot_conv {
                    let (l, sn, rf) = (c[j], c[2*per_slot_conv+j], c[per_slot_conv+j]);
                    let d1=(l-sn).abs(); if d1>b_c{b_c=d1;}
                    let d2=(l-rf).abs(); if d2>c_c{c_c=d2;}
                }
            }
        }
        eprintln!("[reject] (b) restored(slot0) vs snap(slot2): s_state={}, conv={}", b_s, b_c);
        eprintln!("[reject] (c) restored(slot0) vs ref(slot1):   s_state={}, conv={}", c_s, c_c);

        // (d) KV positions 0..plen (committed region) slot 0 vs slot 1.
        let mut d_kv = 0.0f32;
        let per_slot_kv = nkv * kv_stride * hd;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if matches!(lt, LayerType::FullAttention) {
                let k = self.dev.dtoh_sync_copy(state.k_cache[li].as_ref().unwrap()).unwrap();
                let v = self.dev.dtoh_sync_copy(state.v_cache[li].as_ref().unwrap()).unwrap();
                for head in 0..nkv {
                    for p in 0..=plen {
                        for dd in 0..hd {
                            let idx = (head*kv_stride + p)*hd + dd;
                            let dk=(half::bf16::to_f32(k[per_slot_kv+idx])-half::bf16::to_f32(k[idx])).abs();
                            if dk>d_kv { d_kv=dk; }
                            let dv=(half::bf16::to_f32(v[per_slot_kv+idx])-half::bf16::to_f32(v[idx])).abs();
                            if dv>d_kv { d_kv=dv; }
                        }
                    }
                }
            }
        }
        eprintln!("[reject] (d) KV[0..plen] slot0 vs slot1 (committed region): {}", d_kv);
        eprintln!("[reject] all four MUST be 0. Nonzero (a)→checkpoint write; (b)→D2D copy; (c)→reject not bit-exact; (d)→KV write/pos.");

        // ---- Second step (expert §5 "next step"): verify→reject→rollback→verify-again, vs decode×2.
        // bonus = preds[0] from the first verify (the greedy next token after committed).
        let bonus = preds[0];
        // slot B (1): decode bonus once more → reference "after [committed, bonus]".
        self.dev.htod_sync_copy_into(&[bonus as i32, 0], &mut bufs.tokens_dev).unwrap();
        self.dev.htod_sync_copy_into(&[(plen + 1) as i32, 0], &mut bufs.pos_dev).unwrap();
        self.dev.synchronize().unwrap();
        let _ref_tok2 = self.forward_decode(pool, &mut bufs, state, kv_stride, plen + 2, 1);
        // slot A (0): verify([bonus, wrong_draft2], ckpt=Some(2)) at pos plen+1.
        let wrong_draft2 = bonus;
        let vi2 = vec![bonus, wrong_draft2];
        let (preds2, vout2) = self.verify_forward(pool, &vi2, state, 0, kv_stride, plen + 1, Some(2), None);
        pool.release_bf16(vout2, h * 2);
        self.copy_gdn_slot(state, 2, 0);
        self.sync_stream();
        let mut e_s = 0.0f32; let mut e_c = 0.0f32; let mut e_kv = 0.0f32;
        let valid_elems = |head: usize, p: usize| (head * kv_stride + p) * hd;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            match lt {
                LayerType::LinearAttention => {
                    let s = self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap();
                    let c = self.dev.dtoh_sync_copy(state.conv_state[li].as_ref().unwrap()).unwrap();
                    for j in 0..per_slot { let dd=(s[j]-s[per_slot+j]).abs(); if dd>e_s{e_s=dd;} }
                    for j in 0..per_slot_conv { let dd=(c[j]-c[per_slot_conv+j]).abs(); if dd>e_c{e_c=dd;} }
                }
                LayerType::FullAttention => {
                    let k = self.dev.dtoh_sync_copy(state.k_cache[li].as_ref().unwrap()).unwrap();
                    let v = self.dev.dtoh_sync_copy(state.v_cache[li].as_ref().unwrap()).unwrap();
                    for head in 0..nkv {
                        for p in 0..=plen+1 {
                            for dd in 0..hd {
                                let idx = valid_elems(head, p) + dd;
                                let dk=(half::bf16::to_f32(k[per_slot_kv+idx])-half::bf16::to_f32(k[idx])).abs(); if dk>e_kv{e_kv=dk;}
                                let dv=(half::bf16::to_f32(v[per_slot_kv+idx])-half::bf16::to_f32(v[idx])).abs(); if dv>e_kv{e_kv=dv;}
                            }
                        }
                    }
                }
            }
        }
        eprintln!("[reject] (e) AFTER 2nd verify+rollback vs decode×2: s_state={}, conv={}, kv[0..plen+1]={}", e_s, e_c, e_kv);
        eprintln!("[reject] 2nd verify preds={:?} (bonus={}, greedy-after-committed={})", preds2, bonus, preds[0]);
        drop(bufs);
    }
    /// `offset` sequential greedy decodes (so both reach the same state at plen+offset), then decodes
    /// `depth` more on slot 0 (ground truth) while verify_forward-ing `depth` on slot 1 at
    /// plen+offset. Isolates whether the causal-append verify path stays lossless after many appends.
    pub fn bench_verify_at_offset(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                                  kv_stride: usize, offset: usize, depth: usize)
                                 -> (Vec<u32>, Vec<u32>, Vec<bool>) {
        let h = self.cfg.hidden_size;
        let mut bufs = self.new_decode_buffers(2);
        self.zero_slot_state(state, 0, kv_stride);
        let (s0a, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * prompt.len());
        self.zero_slot_state(state, 1, kv_stride);
        let (s0b, h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        pool.release_bf16(h1, h * prompt.len());
        assert_eq!(s0a, s0b, "prefill divergence slot0 vs slot1");

        let plen = prompt.len();
        let mut cur = s0a;
        let mut pos = plen;
        // Advance BOTH slots identically by `offset` tokens.
        for _ in 0..offset {
            let toks_i32 = vec![cur as i32, cur as i32];
            let pos_i32 = vec![pos as i32, pos as i32];
            self.dev.htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            let next = self.forward_decode(pool, &mut bufs, state, kv_stride, pos + 1, 2);
            cur = next[0]; // both lanes identical → next[0]==next[1]
            pos += 1;
        }

        // Ground truth: decode `depth` more on slot 0 (lane 0 → physical slot 0).
        let mut seq_tokens = vec![cur];
        for _ in 0..depth {
            let tok = *seq_tokens.last().unwrap() as i32;
            let toks_i32 = vec![tok, 0];
            let pos_i32 = vec![pos as i32, 0];
            self.dev.htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            let next = self.forward_decode(pool, &mut bufs, state, kv_stride, pos + 1, 1);
            seq_tokens.push(next[0]);
            pos += 1;
        }

        // Verify `depth` tokens on slot 1 at position plen+offset.
        let verify_input: Vec<u32> = seq_tokens[..depth].to_vec();
        let (preds, vout) = self.verify_forward(pool, &verify_input, state, 1, kv_stride, plen + offset, None, None);
        pool.release_bf16(vout, h * depth);
        drop(bufs);

        let matches: Vec<bool> = (0..depth).map(|i| preds[i] == seq_tokens[i + 1]).collect();
        (seq_tokens, preds, matches)
    }

    /// Device-copy all GDN-layer recurrent state (conv_state + s_state) for one slot from `src` to
    /// `dst` within the BatchGpuState buffers. Used for MTP GDN-rollback snapshots (the snapshot
    /// lives in a dedicated slot). KV cache is NOT copied: verify overwrites stale KV positions, so
    /// no KV rollback is needed.
    pub fn copy_gdn_slot(&self, state: &BatchGpuState, src: usize, dst: usize) {
        // Route the dtod copy through the COMPUTE stream (async) so the snapshot/restore is stream-
        // ordered with the verify/reverify GDN writes (all on self.stream) — no cross-stream race
        // with the device's default stream.
        let cfg = &self.cfg;
        // TP-aware: the state slots were allocated at the LOCAL head count, so the per-slot stride must
        // match or this copies the wrong extent. Under sharding it also halves — a free win for MTP,
        // since rollback is per-rank-local (each rank owns whole heads, nothing crosses the wire).
        let lin_nh = self.eff_lin_v_heads(); let kd = cfg.lin_k_dim; let vd = cfg.lin_v_dim;
        let conv_dim = self.eff_conv_dim(); let ck = cfg.conv_kernel;
        let cb = (conv_dim * ck) as u64 * 4;      // bytes per slot: conv_state
        let sb = (lin_nh * kd * vd) as u64 * 4;   // bytes per slot: s_state
        let stream = self.stream.stream;
        for (li, lt) in cfg.layer_types.iter().enumerate() {
            if matches!(lt, LayerType::LinearAttention) {
                unsafe {
                    let cs = *state.conv_state[li].as_ref().unwrap().device_ptr();
                    let ss = *state.s_state[li].as_ref().unwrap().device_ptr();
                    cudarc::driver::result::memcpy_dtod_async(cs + dst as u64 * cb, cs + src as u64 * cb, cb as usize, stream).unwrap();
                    cudarc::driver::result::memcpy_dtod_async(ss + dst as u64 * sb, ss + src as u64 * sb, sb as usize, stream).unwrap();
                }
            }
        }
    }

    /// Compact an accepted tree path's KV into contiguous positions: for k in 0..len, move the K/V at
    /// KV position `pos_start + src_pos[k]` down to `pos_start + k`, for every full-attention layer.
    /// Two-pass gather/scatter through pooled scratch (no in-place aliasing). `src_pos` is host-side
    /// (the accepted DFS indices). A no-op when the accepted path is already contiguous (src_pos[k]==k).
    pub fn compact_kv(&self, pool: &mut Pool, state: &BatchGpuState, slot: usize, pos_start: usize,
                      src_pos: &[i32], kv_stride: usize) {
        let len = src_pos.len();
        if len == 0 || src_pos.iter().enumerate().all(|(k, &s)| s as usize == k) { return; }  // already contiguous
        // eff, not full: the KV cache is head-sharded per rank under TP (per-slot stride is
        // eff_nkv*kv_stride*hd). Reading cfg.num_kv_heads here doubled every slot's base and every
        // head extent — the same "sharding is a property of the tensor" bug class as the tiled
        // prefill fix. bench_tree_accept's host check below matches.
        let (nkv, hd) = (self.eff_num_kv_heads(), self.cfg.head_dim);
        // upload src_pos to sc_pos scratch (reused; sized MAX_VERIFY)
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                *self.sc_pos.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                &src_pos[..len], self.stream.stream).expect("htod compact src");
        }
        let sp_ptr = *self.sc_pos.device_ptr() as u64;
        let ks = pool.get_bf16(len * nkv * hd);
        let vs = pool.get_bf16(len * nkv * hd);
        let total = (len * nkv * hd) as u32;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if !matches!(lt, LayerType::FullAttention) { continue; }
            let kc = *state.k_cache[li].as_ref().unwrap().device_ptr();
            let vc = *state.v_cache[li].as_ref().unwrap().device_ptr();
            for dir in 0..2i32 {   // 0 = gather to scratch, 1 = scatter to contiguous
                blaunch!(self, "compact_kv_b", grid(total as usize), (256,1,1), 0,
                    (kc, vc, d(&ks), d(&vs), sp_ptr, len as i32, pos_start as i32,
                     slot as i32, nkv as i32, kv_stride as i32, hd as i32, dir));
            }
        }
        pool.release_bf16(ks, len * nkv * hd);
        pool.release_bf16(vs, len * nkv * hd);
    }

    /// Copy one hidden column (`col`, of `h` bf16 elems) out of an `[h, n]` column-major buffer
    /// `src` into the device pointer `dst`. Routed through the COMPUTE stream so it is stream-ordered
    /// with the kernels that produced/consume these hiddens (no cross-stream race with stream 0).
    pub fn copy_hidden_col(&self, dst_ptr: u64, src: &B, col: usize) {
        let h = self.cfg.hidden_size;
        let stream = self.stream.stream;
        unsafe {
            let src_ptr = *src.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src_ptr, h * 2, stream).unwrap();
        }
    }

    /// Copy `ncols` CONTIGUOUS hidden columns starting at `col` out of an `[h, n]` column-major
    /// buffer into `dst` — ONE dtod for what used to be `ncols` separate `copy_hidden_col` calls
    /// (the re-prime path assembles accepted-prefix hiddens, whose verify columns are contiguous).
    pub fn copy_hidden_cols(&self, dst_ptr: u64, src: &B, col: usize, ncols: usize) {
        let h = self.cfg.hidden_size;
        let stream = self.stream.stream;
        unsafe {
            let src_ptr = *src.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src_ptr, ncols * h * 2, stream).unwrap();
        }
    }

    /// Memset (zero) `bytes` bytes at `ptr` on the COMPUTE stream — ordered with subsequent compute
    /// kernels. Use this (NOT the device-default-stream memset) for any buffer the compute stream
    /// reads, e.g. per-lane MTP KV that must be zeroed before priming.
    pub fn memset_compute_stream(&self, ptr: u64, bytes: usize) {
        unsafe {
            cudarc::driver::sys::cuMemsetD8Async(
                ptr as cudarc::driver::sys::CUdeviceptr, 0, bytes, self.stream.stream);
        }
    }

    /// Greedy argmax of lm_head(hidden) for a single (batch=1) hidden vector. Returns the token id.
    /// Argmax over the DRAFT head (or the full head if no draft vocab is configured). This is only
    /// ever called to pick a draft token -- never to emit one -- so restricting its vocabulary cannot
    /// affect the output, only the acceptance rate.
    pub fn argmax_hidden(&self, pool: &mut Pool, hidden: &B) -> u32 {
        let (w, vocab) = match &self.draft_head {
            Some(dh) => (dh, self.draft_ids.len()),
            None => (self.lm_head.as_ref().unwrap_or(&self.embed), self.cfg.vocab_size),
        };
        let mut logits = pool.get_bf16(vocab);
        self.gemm_act(w, hidden, &mut logits, self.cfg.hidden_size, vocab, 1);
        let block = 1024u32;
        blaunch!(self, "argmax_b", (1,1,1), (block,1,1), (block*8),
            (*self.sc_tok.device_ptr() as u64, d(&logits), vocab as i32, 1));
        pool.release_bf16(logits, vocab);
        self.sync_stream();
        let row = self.dev.dtoh_sync_copy(&self.sc_tok).unwrap()[0] as usize;
        match &self.draft_head {
            Some(_) => self.draft_ids[row.min(self.draft_ids.len() - 1)],
            None => row as u32,
        }
    }

    /// Vocabulary the draft head proposes from (== full vocab when FR-Spec is off).
    pub fn draft_vocab(&self) -> usize {
        if self.draft_head.is_some() { self.draft_ids.len() } else { self.cfg.vocab_size }
    }

    /// The draft head's top-`k` candidate token ids (diagnostic only — pulls the full draft-vocab logit
    /// vector to the host and partial-sorts). Used by `bench_accept` to measure how often the target's
    /// argmax lands in the head's top-2/top-3 — i.e. what a fork at that position could rescue. NOT on
    /// any serving path.
    pub fn topk_hidden(&self, pool: &mut Pool, hidden: &B, k: usize) -> Vec<u32> {
        let (w, vocab) = match &self.draft_head {
            Some(dh) => (dh, self.draft_ids.len()),
            None => (self.lm_head.as_ref().unwrap_or(&self.embed), self.cfg.vocab_size),
        };
        let mut logits = pool.get_bf16(vocab);
        self.gemm_act(w, hidden, &mut logits, self.cfg.hidden_size, vocab, 1);
        self.sync_stream();
        let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&logits).unwrap();
        pool.release_bf16(logits, vocab);
        let mut idx: Vec<usize> = (0..vocab).collect();
        let k = k.min(vocab);
        idx.select_nth_unstable_by(k - 1, |&a, &b| lg[b].to_f32().total_cmp(&lg[a].to_f32()));
        idx.truncate(k);
        idx.sort_by(|&a, &b| lg[b].to_f32().total_cmp(&lg[a].to_f32()));
        idx.into_iter().map(|row| match &self.draft_head {
            Some(_) => self.draft_ids[row.min(self.draft_ids.len() - 1)],
            None => row as u32,
        }).collect()
    }

    /// One MTP draft step: given the previous hidden (main or prior MTP output) and a token, run the
    /// MTP layer forward at the given absolute position, returning the next MTP hidden. The caller
    /// argmaxes lm_head(next_hidden) to get the predicted token. batch=1.
    /// Re-prime the MTP KV over the accepted prefix in ONE batched MTP-layer forward.
    ///
    /// After a verify, the MTP's KV for the accepted positions has to be rewritten using the REAL
    /// main-model hidden (from the verify output) instead of the MTP's own speculative draft hidden,
    /// so the head's KV stays consistent with the sequence the main model actually committed to.
    /// That used to be one `mtp_draft_step` per accepted position, each re-reading the entire MTP
    /// layer's weights — at depth 2, two full weight reads where one would do.
    ///
    /// Batching is safe for exactly the reason the multi-token verify is: `full_attn_batch` writes
    /// ALL N KV entries before the attention kernel runs, and query column k attends to
    /// KV[0..=pos[k]], so column k sees the KV written by columns < k. Causal-append in one pass.
    ///
    /// `hidden_cols` is [h, n] (column k = the real hidden feeding position pos_start+k). The MTP
    /// output is discarded — only the KV writes matter here.
    pub fn mtp_reprime(&self, pool: &mut Pool, hidden_cols: &B, tokens: &[u32], pos_start: usize,
                       mtp_kc_ptr: u64, mtp_vc_ptr: u64, kv_stride: usize) {
        let n = tokens.len();
        if n == 0 { return; }
        assert!(n <= MAX_VERIFY, "re-prime width {} exceeds MAX_VERIFY {}", n, MAX_VERIFY);
        let h = self.cfg.hidden_size;
        let rdim = self.cfg.rotary_dim;
        let tk: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let ps: Vec<i32> = (0..n).map(|k| (pos_start + k) as i32).collect();
        unsafe {
            use cudarc::driver::sys::CUdeviceptr;
            cudarc::driver::result::memcpy_htod_async(
                *self.mr_tok.device_ptr() as CUdeviceptr, &tk[..], self.stream.stream).unwrap();
            cudarc::driver::result::memcpy_htod_async(
                *self.mr_pos.device_ptr() as CUdeviceptr, &ps[..], self.stream.stream).unwrap();
        }
        self.sync_stream();
        let cos = pool.get(rdim * n);
        let sin = pool.get(rdim * n);
        blaunch!(self, "gather_rope_b", grid(rdim * n), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *self.mr_pos.device_ptr() as u64, rdim as i32, n as i32));
        let out = self.mtp_forward(pool, hidden_cols, *self.mr_tok.device_ptr() as u64, &self.mr_pos,
                                   mtp_kc_ptr, mtp_vc_ptr, pos_start + n, &cos, &sin,
                                   kv_stride, n, None);
        // No trailing sync_stream: nothing host-side consumes the output, and pool.release never
        // frees (I5) — the sync was a full pipeline drain serialized into every MTP step. (The
        // sync BEFORE the forward stays: mr_tok/mr_pos are async-copied from stack Vecs whose
        // lifetime must outlive the copy.)
        pool.release(cos, rdim * n);
        pool.release(sin, rdim * n);
        pool.release_bf16(out, h * n);
    }

    /// Prime the MTP head's KV over the WHOLE PROMPT, in batched causal-append passes.
    ///
    /// This used to be one `mtp_draft_step` per prompt token: 8012 sequential single-column forwards
    /// for an 8K prompt, each re-reading the entire MTP layer's weights and launching ~6 kernels.
    /// Measured on an 8K prompt it was 42,240 `gemm_mma_fp4_b` launches and roughly 11 of the 16
    /// seconds it took to produce the first token. The cost is exactly proportional to prompt length,
    /// so every short-prompt benchmark we ran reported it as zero -- it only ever hurt real clients,
    /// which is precisely how it survived this long.
    ///
    /// The head is a single layer, so the whole prompt primes in one causal-append pass. Column k
    /// consumes (hidden[k], embed(prompt[k+1])) and lands at MTP position k; `write_kv_prefill` writes
    /// all N KV entries before the attention kernel runs, and query column k attends to KV[0..=k], so
    /// a column sees the KV of every column before it. Same argument as the multi-token verify.
    ///
    /// Chunked only to bound transient memory (a 32K prompt would otherwise want multi-GB of activation
    /// scratch); the chunks are causally chained via `pos_start`, so the result is independent of CHUNK.
    ///
    /// This CANNOT affect losslessness. The MTP head only ever PROPOSES; every draft is verified by
    /// argmax against the main model. Changing how the head is primed changes the acceptance rate, and
    /// nothing else.
    /// `pos_start` is the absolute MTP position of `tokens[0]`'s slot. Non-zero on a prefix-cache hit:
    /// the head's KV for [0, pos_start) was written by the request that ended there, and only the new
    /// suffix needs priming.
    pub fn mtp_prime_prompt(&self, pool: &mut Pool, hidden_cols: &B, tokens: &[u32],
                            mtp_kc_ptr: u64, mtp_vc_ptr: u64, kv_stride: usize, pos_start: usize) {
        let n = tokens.len();
        if n == 0 { return; }
        let h = self.cfg.hidden_size;
        let rdim = self.cfg.rotary_dim;
        const CHUNK: usize = 2048;

        let mut c0 = 0usize;
        while c0 < n {
            let cn = CHUNK.min(n - c0);
            let tk: Vec<i32> = tokens[c0..c0 + cn].iter().map(|&t| t as i32).collect();
            let ps: Vec<i32> = (c0..c0 + cn).map(|p| (pos_start + p) as i32).collect();
            let tok_dev = self.dev.htod_sync_copy(&tk).expect("htod mtp prime tokens");
            let pos_dev = self.dev.htod_sync_copy(&ps).expect("htod mtp prime pos");
            self.dev.synchronize().unwrap();   // NULL-stream htod must land before the compute stream reads

            // hidden_cols is [h, plen] COLUMN-major, so this chunk's columns are contiguous.
            let hchunk = pool.get_bf16(h * cn);
            unsafe {
                let dst = *hchunk.device_ptr() as u64;
                let src = *hidden_cols.device_ptr() as u64 + (c0 as u64) * (h as u64) * 2;
                cudarc::driver::result::memcpy_dtod_async(dst, src, cn * h * 2, self.stream.stream)
                    .expect("mtp_prime: copy hidden chunk");
            }
            let cos = pool.get(rdim * cn);
            let sin = pool.get(rdim * cn);
            blaunch!(self, "gather_rope_b", grid(rdim * cn), (256,1,1), 0,
                (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
                 *pos_dev.device_ptr() as u64, rdim as i32, cn as i32));
            let out = self.mtp_forward(pool, &hchunk, *tok_dev.device_ptr() as u64, &pos_dev,
                                       mtp_kc_ptr, mtp_vc_ptr, pos_start + c0 + cn, &cos, &sin,
                                       kv_stride, cn, Some(pos_start + c0));
            self.sync_stream();
            pool.release(cos, rdim * cn);
            pool.release(sin, rdim * cn);
            pool.release_bf16(hchunk, h * cn);
            pool.release_bf16(out, h * cn);
            c0 += cn;
        }
    }

    /// Draft a FORK-THEN-CHAIN tree (k=2 at position 1): the head's top-2 for the first token, then a
    /// depth-1 chain from each branch. Returns (parent, tokens) with tokens[0] = committed; each branch
    /// has depth-1 nodes so n = 2*depth-1. Writes MTP KV transiently (branch B reuses/overwrites branch
    /// A's positions — mtp_draft_step writes-then-attends, so each branch's drafts are computed against
    /// its OWN path; the post-accept re-prime fixes the KV over the accepted path). If `ngram>0`, the
    /// second branch's FIRST token is an n-gram continuation of `work` instead of the head's #2 (the
    /// tool-text win, review §5.4). Degenerate forks (both branches' first token equal) fall back to a
    /// single chain. Drafting is speculative: a bad fork only costs acceptance, never correctness.
    pub fn mtp_fork_draft(&self, pool: &mut Pool, h_prev: &B, committed: i32, mtp_pos: usize, depth: usize,
                          mtp_kc_ptr: u64, mtp_vc_ptr: u64, kv_stride: usize,
                          work: &[u32], ngram: usize) -> (Vec<i32>, Vec<u32>) {
        let hs = self.cfg.hidden_size;
        // BRANCH A: the head's greedy chain (exactly the chain drafter). depth-1 tokens, writing MTP KV
        // at mtp_pos..mtp_pos+depth-2. This is the unchanged 1x-cost draft.
        let mut a = Vec::with_capacity(depth - 1);
        let mut m = self.mtp_draft_step(pool, h_prev, committed, mtp_pos, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        for k in 0..depth - 1 {
            let t = self.argmax_hidden(pool, &m) as i32;
            a.push(t as u32);
            if k + 1 < depth - 1 {
                let nx = self.mtp_draft_step(pool, &m, t, mtp_pos + 1 + k, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                pool.release_bf16(m, hs); m = nx;
            }
        }
        pool.release_bf16(m, hs);

        // BRANCH B: a PURE n-gram chain — every token copied by extending an n-gram match in `work`.
        // ZERO GPU (host-side lookup), so the fork is nearly free; it fires only on copyable text (tool
        // output), where it rescues long copies the 1-layer head can't (review §5.4). No MTP KV is
        // written for B (it uses no head forward); the post-accept re-prime sets it over the accepted path.
        let mut b = Vec::new();
        if ngram > 0 && work.len() >= ngram {
            let mut ctx: Vec<u32> = work.to_vec();
            for _ in 0..depth - 1 {
                let ts = ctx.len() - ngram;
                let mut found = None;
                for j in (0..ts).rev() {
                    if ctx[j..j + ngram] == ctx[ts..] { if j + ngram < ctx.len() { found = Some(ctx[j + ngram]); } break; }
                }
                match found { Some(t) => { b.push(t); ctx.push(t); } None => break }
            }
        }

        // Fork only if B exists and DIVERGES from A at position 1 (else the sibling is redundant).
        if b.is_empty() || b[0] == a[0] {
            let n = a.len() + 1;
            let parent: Vec<i32> = (0..n as i32).map(|c| c - 1).collect();
            let mut tokens = vec![committed as u32]; tokens.extend_from_slice(&a);
            return (parent, tokens);
        }
        // Assemble: root, a-branch (cols 1..=|a|), b-branch (cols |a|+1..).
        let (na, nb) = (a.len(), b.len());
        let n = 1 + na + nb;
        let mut parent = vec![-1i32; n];
        for i in 1..=na { parent[i] = (i - 1) as i32; }
        parent[1 + na] = 0;
        for j in 1..nb { parent[1 + na + j] = (na + j) as i32; }
        let mut tokens = vec![committed as u32];
        tokens.extend_from_slice(&a);
        tokens.extend_from_slice(&b);
        (parent, tokens)
    }

    pub fn mtp_draft_step(&self, pool: &mut Pool, hidden: &B, token: i32, pos: usize,
                      mtp_kc_ptr: u64, mtp_vc_ptr: u64, kv_stride: usize) -> B {
        let rdim = self.cfg.rotary_dim;
        // Htod token/pos into persistent scratch on the COMPUTE stream (ordered with the kernels).
        let tk = [token]; let ps = [pos as i32];
        unsafe {
            cudarc::driver::result::memcpy_htod_async(*self.sc_i1a.device_ptr() as cudarc::driver::sys::CUdeviceptr, &tk[..], self.stream.stream).unwrap();
            cudarc::driver::result::memcpy_htod_async(*self.sc_i1b.device_ptr() as cudarc::driver::sys::CUdeviceptr, &ps[..], self.stream.stream).unwrap();
        }
        self.sync_stream(); // ensure htod consumed (SoC source-lifetime; expert §5.5)
        let token_ptr = *self.sc_i1a.device_ptr() as u64;
        let cos = pool.get(rdim);
        let sin = pool.get(rdim);
        blaunch!(self, "gather_rope_b", grid(rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *self.sc_i1b.device_ptr() as u64, rdim as i32, 1));
        let out = self.mtp_forward(pool, hidden, token_ptr, &self.sc_i1b,
                                   mtp_kc_ptr, mtp_vc_ptr, pos + 1, &cos, &sin, kv_stride, 1, None);
        self.sync_stream();
        pool.release(cos, rdim);
        pool.release(sin, rdim);
        out
    }

    /// Stochastic MTP draft step: like mtp_draft_step but samples from the MTP head (instead of
    /// argmax) and returns the chosen token's normalized probability q(x) under the
    /// temperature→topk→softmax→topp nucleus. The caller uses q(x) in the speculative
    /// rejection-sampling accept loop: accept draft with prob min(1, p_target(x)/q_draft(x)).
    pub fn mtp_draft_step_sample(&self, pool: &mut Pool, hidden: &B, token: i32, pos: usize,
                                 mtp_kc_ptr: u64, mtp_vc_ptr: u64, kv_stride: usize,
                                 temp: f32, top_k: usize, top_p: f32, seed: u32,
                                 draft_penalty: Option<&DraftPenalty>) -> (B, u32, f32) {
        let rdim = self.cfg.rotary_dim;
        let _h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let tk = [token]; let ps = [pos as i32];
        unsafe {
            cudarc::driver::result::memcpy_htod_async(*self.sc_i1a.device_ptr() as cudarc::driver::sys::CUdeviceptr, &tk[..], self.stream.stream).unwrap();
            cudarc::driver::result::memcpy_htod_async(*self.sc_i1b.device_ptr() as cudarc::driver::sys::CUdeviceptr, &ps[..], self.stream.stream).unwrap();
        }
        self.sync_stream();
        let token_ptr = *self.sc_i1a.device_ptr() as u64;
        let cos = pool.get(rdim);
        let sin = pool.get(rdim);
        blaunch!(self, "gather_rope_b", grid(rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *self.sc_i1b.device_ptr() as u64, rdim as i32, 1));
        let mhidden = self.mtp_forward(pool, hidden, token_ptr, &self.sc_i1b,
                                       mtp_kc_ptr, mtp_vc_ptr, pos + 1, &cos, &sin, kv_stride, 1, None);
        pool.release(cos, rdim);
        pool.release(sin, rdim);

        // Logits for the DRAFT, from the draft head if FR-Spec is on.
        //
        // Distribution-exactness survives the restriction, and this is the subtle part. `sample_prob_b`
        // samples from the softmax of whatever logits it is given and reports THAT probability as `q`.
        // Feed it the restricted logits and it samples from -- and reports -- the RENORMALIZED
        // restricted distribution. That restricted distribution *is* the proposal `q`, so the verify's
        // `min(1, p/q)` is the standard Leviathan/Chen scheme with a perfectly valid proposal. Tokens
        // outside the subset have q = 0 and can only enter via the residual `(p-q)+` resample, which
        // runs over the FULL vocabulary on the verify side. No approximation anywhere.
        //
        // What would be WRONG is to restrict the draft but report `q` from the full softmax: then the
        // accept ratio uses a `q` the drafter never sampled from, and the output distribution is
        // quietly skewed. The two must be the same distribution.
        let (dw, vocab) = match &self.draft_head {
            Some(dh) => (dh, self.draft_ids.len()),
            None => (self.lm_head.as_ref().unwrap_or(&self.embed), self.cfg.vocab_size),
        };
        let mut logits = pool.get_bf16(vocab);
        self.gemm_act(dw, &mhidden, &mut logits, self.cfg.hidden_size, vocab, 1);
        if let Some(p) = draft_penalty {
            blaunch!(self, "rep_penalty_b", (1,1,1), (256,1,1), 0,
                (d(&logits), p.tokens_ptr, p.counts_ptr, MAX_PEN_TOKENS as i32,
                 p.rep_pen_ptr, p.presence_ptr, p.freq_ptr, vocab as i32, 1));
        }
        let tk_dev = self.dev.alloc_zeros::<i32>(1).unwrap();  // [1] i32
        let qp_dev = pool.get(1);    // [1] f32
        let mut t_dev = pool.get(1);     // [1] f32
        let mut k_dev = self.dev.alloc_zeros::<i32>(1).unwrap();   // [1] i32
        let mut p_dev = pool.get(1);     // [1] f32
        let mut sd_dev = self.dev.alloc_zeros::<u32>(1).unwrap();  // [1] u32
        let t = [temp]; let k = [top_k as i32]; let p = [top_p]; let sd = [seed];
        self.dev.htod_sync_copy_into(&t, &mut t_dev).unwrap();
        self.dev.htod_sync_copy_into(&k, &mut k_dev).unwrap();
        self.dev.htod_sync_copy_into(&p, &mut p_dev).unwrap();
        self.dev.htod_sync_copy_into(&sd, &mut sd_dev).unwrap();
        self.dev.synchronize().unwrap();
        let smem = ((2 * 64usize + 2 * 256usize) * 4) as u32;
        blaunch!(self, "sample_prob_b", (1,1,1), (256,1,1), smem,
            (d(&tk_dev), d(&qp_dev), d(&logits), d(&t_dev), d(&k_dev), d(&p_dev), d(&sd_dev),
             vocab as i32, 1));
        pool.release_bf16(logits, vocab);
        drop(k_dev); drop(sd_dev);
        pool.release(t_dev, 1); pool.release(p_dev, 1);
        self.sync_stream();
        let row = self.dev.dtoh_sync_copy(&tk_dev).unwrap()[0] as usize;
        let token = match &self.draft_head {
            Some(_) => self.draft_ids[row.min(self.draft_ids.len() - 1)],
            None => row as u32,
        };
        let qprob = self.dev.dtoh_sync_copy(&qp_dev).unwrap()[0];
        drop(tk_dev);
        pool.release(qp_dev, 1);
        (mhidden, token, qprob)
    }

    /// End-to-end MTP speculative-decoding probe (single lane, slot 0). Full loop: prefill →
    /// MTP prompt-prime → repeat { draft K-1 → verify K → accept longest prefix → GDN rollback →
    /// re-prime accepted prefix with real hiddens } → compare token-for-token against sequential
    /// greedy on slot 1 (lossless gate). Returns (mtp_tokens, seq_tokens, mtp_tok_s, seq_tok_s,
    /// accept_rate). MTP KV position and RoPE position both track the main model's position.
    /// Mean negative log-likelihood over a token window — the basis of perplexity.
    ///
    /// This is the quality gate for quantization. "The output still looks coherent" is not a gate:
    /// 4-bit round-to-nearest damage shows up as rare-token degradation and long-context drift long
    /// before fluency breaks. Perplexity on held-out text is the standard, sensitive measure, and it
    /// is cheap here — one prefill gives logits for every position at once, and the softmax stays on
    /// device (a [248320, N] logits block would be ~0.5 GB to move per window).
    ///
    /// Position i's logits predict token i+1, so a window of N tokens yields N-1 scored positions.
    pub fn window_nll(&self, pool: &mut Pool, state: &mut BatchGpuState, tokens: &[u32],
                      kv_stride: usize) -> (f64, usize) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let n = tokens.len();
        assert!(n >= 2, "need at least 2 tokens to score one position");

        self.zero_slot_state(state, 0, kv_stride);
        let (_first, residual) = self.prefill_batch(pool, tokens, state, 0, kv_stride, 0);

        // prefill returns the PRE-final-norm hidden; logits_batch expects post-norm.
        let out = pool.get_bf16(h * n);
        blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&out), d(&residual), d(&self.final_norm), h as i32, n as i32, fbits(cfg.rms_eps)));
        pool.release_bf16(residual, h * n);

        let logits = self.logits_batch(pool, &out, n);
        pool.release_bf16(out, h * n);

        // Score positions 0..n-2 against targets tokens[1..n-1].
        let ns = n - 1;
        let tgt: Vec<i32> = tokens[1..].iter().map(|&t| t as i32).collect();
        let mut tgt_dev = self.dev.alloc_zeros::<i32>(ns).unwrap();
        self.dev.memset_zeros(&mut tgt_dev).unwrap();
        self.dev.htod_sync_copy_into(&tgt, &mut tgt_dev).unwrap();
        let nll = pool.get(ns);
        self.dev.synchronize().unwrap();

        let nthr = 1024u32;
        blaunch!(self, "nll_b", (ns as u32,1,1), (nthr,1,1), (nthr * 4),
            (d(&nll), d(&logits), d(&tgt_dev), vocab as i32, ns as i32));
        pool.release_bf16(logits, vocab * n);
        self.sync_stream();

        let v = self.dev.dtoh_sync_copy(&nll).unwrap();
        pool.release(nll, ns);
        let sum: f64 = v.iter().take(ns).map(|&x| x as f64).sum();
        (sum, ns)
    }

    /// DEBUG PROBE (MoE correctness oracle). Run `moe_batch` for the FIRST MoE layer on a caller-given
    /// input `x_host` [hidden*batch] (col-major, bf16) and return `(layer_index, output [hidden*batch])`.
    /// A numpy reference over that layer's checkpoint weights + the same input validates the block.
    pub fn probe_moe(&self, x_host: &[half::bf16], batch: usize) -> (usize, Vec<half::bf16>) {
        let (li, moe) = self.layers.iter().enumerate()
            .find_map(|(i, l)| match &l.mlp { Ffn::Moe(m) => Some((i, m)), _ => None })
            .expect("no MoE layer in this model");
        let mut pool = Pool::new(self.dev.clone());
        let x = self.dev.htod_sync_copy(x_host).unwrap();
        let out = self.moe_batch(&mut pool, &x, moe, batch);
        let ov = self.dev.dtoh_sync_copy(&out).unwrap();
        self.sync_stream();
        (li, ov)
    }

    /// DEBUG PROBE (cross-model spec-decode acceptance study). Teacher-force `tokens` and return the
    /// per-position greedy argmax: `pred[i] = argmax(logits at position i)` = the model's greedy
    /// next-token given `tokens[..=i]`, for i in 0..n. Mirrors `window_nll` exactly (same prefill +
    /// logits) but reads out argmax via the existing `argmax_b` kernel instead of scoring NLL. Purely
    /// observational — no serving path, no behaviour change. Used by `--dump-argmax`.
    pub fn window_argmax(&self, pool: &mut Pool, state: &mut BatchGpuState, tokens: &[u32],
                         kv_stride: usize) -> Vec<u32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let n = tokens.len();
        assert!(n >= 1, "need at least 1 token");

        self.zero_slot_state(state, 0, kv_stride);
        let (_first, residual) = self.prefill_batch(pool, tokens, state, 0, kv_stride, 0);

        let out = pool.get_bf16(h * n);
        blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&out), d(&residual), d(&self.final_norm), h as i32, n as i32, fbits(cfg.rms_eps)));
        pool.release_bf16(residual, h * n);

        let logits = self.logits_batch(pool, &out, n);
        pool.release_bf16(out, h * n);

        let mut tok_dev = self.dev.alloc_zeros::<i32>(n).unwrap();
        self.dev.memset_zeros(&mut tok_dev).unwrap();
        let block = 1024u32;
        blaunch!(self, "argmax_b", (n as u32,1,1), (block,1,1), (block*8),
            (*tok_dev.device_ptr() as u64, d(&logits), vocab as i32, n as i32));
        pool.release_bf16(logits, vocab * n);
        self.sync_stream();

        let ids = self.dev.dtoh_sync_copy(&tok_dev).unwrap();
        ids.into_iter().take(n).map(|x| x as u32).collect()
    }

    /// Measure `r` = (cost of one MTP step) / (cost of one plain decode step).
    ///
    /// This is the whole basis of the auto-policy. A depth-2 MTP step emits `1 + acceptance` tokens
    /// (exactly: the accepted draft plus the bonus), and costs `r` decode-steps, so
    ///
    /// ```text
    /// speedup = (1 + acceptance) / r      =>      MTP pays iff  acceptance > r - 1
    /// ```
    ///
    /// `r` is a pure cost ratio: it depends on the model's shape (above all on what fraction of the
    /// weights the LM head is, since drafting reads it a second time) and not on the prompt. So it can
    /// be measured once at load and trusted. Acceptance, by contrast, is workload-dependent and has to
    /// be tracked live — hence the split: `r` here, acceptance in the scheduler's EMA.
    ///
    /// Measured values (r): 2B 1.54, 4B 1.35, 9B 1.29, 27B 1.12 — i.e. break-even acceptance of
    /// 54% / 35% / 29% / 12%. That is exactly why 2B (30% acceptance) loses and 27B (80%) wins big.
    /// r(d) = (cost of one MTP step at depth d) / (cost of a plain decode step), MEASURED per depth.
    ///
    /// The MTP identity is `speedup = tokens_per_step / r`, so r(d) and the acceptance curve together
    /// decide the optimal depth. Both halves have to be real: this measures r(d) directly rather than
    /// measuring one verify at N=2 and charging it to every depth.
    pub fn calibrate_mtp_r(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                           kv_stride: usize) -> Vec<(usize, f32)> {
        let rows = self.profile_mtp(pool, state, prompt, kv_stride, 3);
        let get = |k: &str| rows.iter().find(|r| r.0.trim().starts_with(k)).map(|r| r.1).unwrap_or(0.0);
        let decode = get("decode step");
        if decode <= 0.0 { return vec![]; }
        let per_draft = get("mtp_draft_step") + get("argmax_hidden");
        let tail = get("copy_gdn_slot") + get("mtp_reprime");
        AUTO_DEPTHS.iter().map(|&d| {
            // One MTP step: the draft chain (d-1 MTP forwards, each + an LM-head argmax), the verify
            // at N=d, the GDN rollback, and the re-prime over the accepted prefix.
            let step = (d - 1) as f64 * per_draft
                + get(&format!("verify_forward_sample N={}", d))
                + tail;
            (d, (step / decode) as f32)
        }).collect()
    }

    /// Phase timings for one stochastic-MTP step (`--profile-mtp`).
    ///
    /// A step accepts ~1.76 tokens but costs ~1.6x a plain decode step, so most of the acceptance
    /// win is being eaten somewhere. This times each phase against a plain decode step (the baseline
    /// a token would otherwise cost) so the overhead can be attributed instead of guessed at.
    pub fn profile_mtp(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                       kv_stride: usize, iters: usize) -> Vec<(String, f64)> {
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        let nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;
        let mut bufs = self.new_decode_buffers(3);

        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        self.dev.synchronize().unwrap();
        let mtp_kc_ptr = *mtp_kc.device_ptr();
        let mtp_vc_ptr = *mtp_vc.device_ptr();
        let h_scratch = self.dev.alloc_zeros::<half::bf16>(h).unwrap();

        self.zero_slot_state(state, 0, kv_stride);
        let (a0, hout0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        let copy_stream = self.stream.stream;
        let copy_col = |dst_ptr: u64, src_buf: &B, col: usize| unsafe {
            let src = *src_buf.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src, h * 2, copy_stream).unwrap();
        };
        copy_col(*h_scratch.device_ptr(), &hout0, plen - 1);
        pool.release_bf16(hout0, h * plen);
        self.sync_stream();

        // Penalty buffers, sized as the scheduler sizes them.
        let depth = 2usize;
        let mp = MAX_PEN_TOKENS;
        let mut pen_tokens = self.dev.alloc_zeros::<i32>(depth * mp).unwrap();
        let mut pen_counts = self.dev.alloc_zeros::<i16>(depth * mp).unwrap();
        let mut pen_rep = self.dev.alloc_zeros::<f32>(depth).unwrap();
        let mut pen_pres = self.dev.alloc_zeros::<f32>(depth).unwrap();
        let mut pen_freq = self.dev.alloc_zeros::<f32>(depth).unwrap();
        self.dev.synchronize().unwrap();

        let mut out: Vec<(String, f64)> = Vec::new();
        let mut time_it = |name: &str, n: usize, f: &mut dyn FnMut()| {
            f(); // warm
            let t0 = std::time::Instant::now();
            for _ in 0..n { f(); }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            out.push((name.to_string(), ms));
        };

        // --- baseline: one plain decode step (what a token costs without MTP) ---
        self.dev.htod_sync_copy_into(&[a0 as i32, 0, 0], &mut bufs.token_ids_dev).unwrap();
        self.dev.htod_sync_copy_into(&[plen as i32, 0, 0], &mut bufs.pos_dev).unwrap();
        self.dev.htod_sync_copy_into(&[0i32, 1, 2], &mut bufs.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();
        {
            let (p, s, b) = (&mut *pool, &mut *state, &mut bufs);
            let mut f = || { self.forward_decode_gpu(p, b, s, kv_stride, plen + 1, 1); self.sync_stream(); };
            time_it("decode step (batch=1, baseline)", iters, &mut f);
        }

        // --- MTP head forward (the draft), and the LM-head argmax that turns it into a token ---
        {
            let p = &mut *pool;
            let mut f = || {
                let m = self.mtp_draft_step(p, &h_scratch, a0 as i32, plen - 1,
                                            mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                self.sync_stream();
                p.release_bf16(m, h);
            };
            time_it("  mtp_draft_step (MTP layer fwd)", iters, &mut f);
        }
        {
            let p = &mut *pool;
            let mut f = || { let _ = self.argmax_hidden(p, &h_scratch); };
            time_it("  argmax_hidden (LM head + dtoh)", iters, &mut f);
        }

        // --- the penalty upload the scheduler does on EVERY step ---
        {
            let tks = vec![1i32; depth * mp];
            let cts = vec![1i16; depth * mp];
            let rp = vec![1.1f32; depth];
            let mut f = || {
                self.dev.htod_sync_copy_into(&tks, &mut pen_tokens).unwrap();
                self.dev.htod_sync_copy_into(&cts, &mut pen_counts).unwrap();
                self.dev.htod_sync_copy_into(&rp, &mut pen_rep).unwrap();
                self.dev.htod_sync_copy_into(&rp, &mut pen_pres).unwrap();
                self.dev.htod_sync_copy_into(&rp, &mut pen_freq).unwrap();
                self.dev.synchronize().unwrap();
            };
            time_it("  penalty upload (5 htod + sync)", iters, &mut f);
        }

        // --- verify: backbone+logits alone, then the greedy and stochastic tails on top ---
        // The verify reads exactly the same weights as a decode step, so at N=1 it *should* cost the
        // same. Sweep N (and the ping-pong checkpoint) to separate "cost of the extra column" from
        // "cost of the prefill-flavoured kernels".
        // Sweep N with the checkpoint on AND off at each width. With a flat-in-N GEMM the weight bytes
        // (which set decode cost) do not change with N, so any growth here is NOT the GEMM: it is the
        // GDN scan, the attention, or the per-column checkpoint writes. Separating them is the point.
        let vin = [a0, a0];
        for (n, ck, label) in [
            (1usize, None,    "verify_core N=1, no ckpt"),
            (1usize, Some(2), "verify_core N=1, +ckpt"),
            (2usize, None,    "verify_core N=2, no ckpt"),
            (2usize, Some(2), "verify_core N=2, +ckpt"),
            (4usize, None,    "verify_core N=4, no ckpt"),
            (4usize, Some(2), "verify_core N=4, +ckpt"),
            (8usize, None,    "verify_core N=8, no ckpt"),
            (8usize, Some(2), "verify_core N=8, +ckpt"),
        ] {
            let toks = vec![a0; n];
            let (p, s) = (&mut *pool, &mut *state);
            let mut f = || {
                let (lg, res) = self.verify_forward_core(p, &toks, s, 0, kv_stride, plen, ck);
                self.sync_stream();
                p.release_bf16(lg, self.cfg.vocab_size * n);
                p.release_bf16(res, h * n);
            };
            time_it(label, iters, &mut f);
        }
        {
            let (p, s) = (&mut *pool, &mut *state);
            let mut f = || {
                let (lg, res) = self.verify_forward_core(p, &vin, s, 0, kv_stride, plen, Some(2));
                self.sync_stream();
                p.release_bf16(lg, self.cfg.vocab_size * 2);
                p.release_bf16(res, h * 2);
            };
            time_it("verify_forward_core (backbone+logits, N=2)", iters, &mut f);
        }
        {
            let (p, s) = (&mut *pool, &mut *state);
            let mut f = || {
                let (_pr, res) = self.verify_forward(p, &vin, s, 0, kv_stride, plen, Some(2), None);
                p.release_bf16(res, h * 2);
            };
            time_it("verify_forward (greedy: +argmax+dtoh)", iters, &mut f);
        }
        // The FULL stochastic verify at every candidate depth. This is what `calibrate_mtp_r` composes
        // r(d) from. Measuring it once at N=2 and charging that to every depth (what this used to do)
        // made depth 2 look optimal by construction — harmless while the verify was superlinear in N,
        // and exactly wrong now that it is flat and depth is the whole decision.
        for &d in AUTO_DEPTHS {
            let toks = vec![a0; d];
            let drafts = vec![a0; d - 1];
            let qp = vec![1.0f32; d - 1];
            let seeds: Vec<u32> = (0..d as u32).collect();
            let (p, s) = (&mut *pool, &mut *state);
            let mut f = || {
                let (_vs, res) = self.verify_forward_sample(
                    p, &toks, s, 0, kv_stride, plen, Some(2), None,
                    &drafts, &qp, 0.7, 20, 0.8, &seeds);
                p.release_bf16(res, h * d);
            };
            time_it(&format!("verify_forward_sample N={}", d), iters, &mut f);
        }

        // --- GDN rollback ---
        {
            let s = &*state;
            let mut f = || { self.copy_gdn_slot(s, 2, 0); self.sync_stream(); };
            time_it("  copy_gdn_slot (GDN rollback)", iters, &mut f);
        }

        // --- batched re-prime over the accepted prefix (depth columns, one MTP-layer forward) ---
        {
            let rp = pool.get_bf16(h * depth);
            let toks = vec![a0; depth];
            let p = &mut *pool;
            let mut f = || {
                self.mtp_reprime(p, &rp, &toks, plen - 1,
                                 mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            };
            time_it("  mtp_reprime (batched, N=depth)", iters, &mut f);
            pool.release_bf16(rp, h * depth);
        }

        out
    }

    /// Distribution-exactness gate for stochastic MTP (`--bench-mtp-sample`).
    ///
    /// Greedy MTP is *bitwise* lossless, and `--bench-mtp` gates it by direct comparison against
    /// sequential greedy. Stochastic MTP can only ever be *distribution*-exact, so it needs a
    /// statistical gate instead. The backbone is already proven (verify column 0 == decode, see
    /// `--bench-verify` / `--probe-state`), which leaves the accept/residual/RNG math as the only
    /// thing that can break exactness. So this pins the prefix — one prefill, one greedy draft, one
    /// verify, giving a FIXED target distribution `p` and a FIXED draft token `x` — and then draws
    /// `trials` emissions several ways:
    ///
    ///   analytic p  the nucleus distribution the plain sampler is *defined* to emit (ground truth)
    ///   A: sampler  real `sample_b` launches on that same logits column   (the sampling-noise floor)
    ///   B: MTP      real `spec_verify_b` + the production host accept rule (the thing under test)
    ///   C: bonus    `spec_verify_b`'s all-accepted bonus column (also a draw from p)
    ///
    /// With a greedy (point-mass) draft, `q(x) = 1`, so speculative rejection sampling accepts `x`
    /// with probability exactly `p(x)` and otherwise draws from `p` with `x` removed and
    /// renormalized — which composes back to exactly `p` for every token. B must therefore be
    /// statistically indistinguishable from A. The gate is `TVD(B, p) <= TVD(A, p)`-ish: A's distance
    /// is the noise floor at this trial count, so "small" is not enough — B has to reach the floor.
    ///
    /// No penalty is applied here: this isolates the sampling math from the penalty transform.
    pub fn bench_mtp_sample(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                            kv_stride: usize, temp: f32, top_k: usize, top_p: f32, trials: usize,
                            rng_base: u64)
                           -> MtpSampleStats {
        use crate::batch::{rng_u32, rng_uniform, splitmix64, RNG_DOM_VERIFY, RNG_DOM_ACCEPT};
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let plen = prompt.len();
        let nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;

        // alloc_zeros is cuMemAllocAsync — it does NOT zero. Explicitly memset anything a compute
        // kernel will read (HANDOFF invariant 1).
        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        self.dev.synchronize().unwrap();
        let mtp_kc_ptr = *mtp_kc.device_ptr();
        let mtp_vc_ptr = *mtp_vc.device_ptr();
        let h_scratch = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let cur_hidden = self.dev.alloc_zeros::<half::bf16>(h).unwrap();

        self.zero_slot_state(state, 0, kv_stride);
        let (a0, hout0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);

        let copy_stream = self.stream.stream;
        let copy_col = |dst_ptr: u64, src_buf: &B, col: usize| unsafe {
            let src = *src_buf.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src, h * 2, copy_stream).unwrap();
        };

        // Prompt-prime the MTP KV exactly as the serving path does (positions 0..plen-2).
        // Must stay a call to the SAME primitive the server uses -- this used to be an open-coded
        // per-token loop whose comment claimed it mirrored serving, which is exactly the kind of
        // copy that silently stops mirroring it.
        self.mtp_prime_prompt(pool, &hout0, &prompt[1..plen], mtp_kc_ptr, mtp_vc_ptr, kv_stride, 0);
        copy_col(*cur_hidden.device_ptr(), &hout0, plen - 1);
        pool.release_bf16(hout0, h * plen);

        // One greedy draft at depth-2, exactly as the production stochastic path drafts.
        let m = self.mtp_draft_step(pool, &cur_hidden, a0 as i32, plen - 1,
                                    mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        copy_col(*cur_hidden.device_ptr(), &m, 0);
        pool.release_bf16(m, h);
        let x_draft = self.argmax_hidden(pool, &cur_hidden);

        // Verify [committed, draft] → target logits. Column 0 is the distribution the drafted
        // position is judged against; that column is what both A and B sample from.
        let (logits, residual) = self.verify_forward_core(pool, &[a0, x_draft], state, 0, kv_stride, plen, None);
        pool.release_bf16(residual, h * 2);
        let mut pristine = self.dev.alloc_zeros::<half::bf16>(vocab).unwrap();
        self.dev.memset_zeros(&mut pristine).unwrap();
        self.dev.synchronize().unwrap();
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                *pristine.device_ptr() as u64, *logits.device_ptr() as u64, vocab * 2, copy_stream).unwrap();
        }
        self.sync_stream();
        pool.release_bf16(logits, vocab * 2);

        // Ground truth: the nucleus distribution, computed on the host from the same logits.
        let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&pristine).unwrap();
        let lgf: Vec<f32> = lg.iter().map(|x| x.to_f32()).collect();
        let p_analytic = nucleus_dist(&lgf, temp, top_k, top_p);
        let mut pmap = vec![0.0f64; vocab];
        for &(t, pr) in &p_analytic { pmap[t as usize] = pr as f64; }
        let p_draft_analytic = pmap[x_draft as usize] as f32;

        // One launch handles CH columns. `sample_b` samples all CH; `spec_verify_b` reserves its
        // last column for the bonus, so it yields CH-1 draft-column trials and exactly ONE bonus
        // sample. The bonus column is the emission on every all-accepted step (~77% of them in
        // production), so it needs enough samples to be testable — hence a modest CH: the draft-trial
        // work is unchanged, but the launch count (and so the bonus sample count) goes up ~8x.
        const CH: usize = 32;
        let per_launch = CH - 1;
        let kmax = 64usize; let nthr = 256u32;
        let smem = ((2 * kmax + 2 * nthr as usize) * 4) as u32;

        let work = pool.get_bf16(vocab * CH);
        let mut tok_out = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let p_out = pool.get(CH);
        let mut r_out = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let mut b_out = self.dev.alloc_zeros::<i32>(1).unwrap();
        let mut dt_dev = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let mut dq_dev = pool.get(CH);
        let mut t_dev = pool.get(CH);
        let mut k_dev = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let mut p_dev = pool.get(CH);
        let mut sd_dev = self.dev.alloc_zeros::<u32>(CH).unwrap();
        self.dev.memset_zeros(&mut tok_out).unwrap();
        self.dev.memset_zeros(&mut r_out).unwrap();
        self.dev.memset_zeros(&mut b_out).unwrap();
        self.dev.synchronize().unwrap();
        self.dev.htod_sync_copy_into(&vec![x_draft as i32; CH], &mut dt_dev).unwrap();
        self.dev.htod_sync_copy_into(&vec![1.0f32; CH], &mut dq_dev).unwrap();   // greedy draft: q = 1
        self.dev.htod_sync_copy_into(&vec![temp; CH], &mut t_dev).unwrap();
        self.dev.htod_sync_copy_into(&vec![top_k as i32; CH], &mut k_dev).unwrap();
        self.dev.htod_sync_copy_into(&vec![top_p; CH], &mut p_dev).unwrap();
        self.dev.synchronize().unwrap();

        // `sample_b` masks its logits row in place as it selects top-k, so the working copy has to be
        // re-seeded from the pristine column before every launch.
        let refill = |w: &B| unsafe {
            for c in 0..CH {
                cudarc::driver::result::memcpy_dtod_async(
                    *w.device_ptr() as u64 + (c * vocab * 2) as u64,
                    *pristine.device_ptr() as u64, vocab * 2, copy_stream).unwrap();
            }
        };

        let mut hist_a = vec![0u64; vocab];
        let mut hist_b = vec![0u64; vocab];
        let mut hist_c = vec![0u64; vocab];
        let (mut n_a, mut n_b, mut n_c, mut n_accept) = (0u64, 0u64, 0u64, 0u64);
        let mut p_draft_device = 0.0f32;
        let mut done = 0usize;
        let mut li = 0usize;

        while done < trials {
            // Each column is an independent trial, keyed like one production decode step. `rng_base`
            // is varied by the caller across temperatures — otherwise every temperature would reuse
            // the SAME uniform stream and merely apply a different threshold to it, so one unlucky
            // draw would show up as a "systematic" bias in every row.
            let keys: Vec<u64> = (0..CH)
                .map(|c| splitmix64(rng_base ^ ((li * CH + c) as u64)))
                .collect();

            // ---- A: the plain sampler on the same column (control / noise floor) ----
            let seeds_a: Vec<u32> = keys.iter().map(|&k| rng_u32(k, RNG_DOM_VERIFY, 7)).collect();
            refill(&work);
            self.dev.htod_sync_copy_into(&seeds_a, &mut sd_dev).unwrap();
            self.dev.synchronize().unwrap();
            blaunch!(self, "sample_b", (CH as u32,1,1), (nthr,1,1), smem,
                (d(&tok_out), d(&work), d(&t_dev), d(&k_dev), d(&p_dev), d(&sd_dev),
                 vocab as i32, CH as i32));
            self.sync_stream();
            let ta = self.dev.dtoh_sync_copy(&tok_out).unwrap();
            for c in 0..CH { hist_a[ta[c] as usize] += 1; n_a += 1; }

            // ---- B: stochastic MTP — the real kernel + the real host accept rule ----
            let seeds_b: Vec<u32> = keys.iter().map(|&k| rng_u32(k, RNG_DOM_VERIFY, 0)).collect();
            refill(&work);
            self.dev.htod_sync_copy_into(&seeds_b, &mut sd_dev).unwrap();
            self.dev.synchronize().unwrap();
            blaunch!(self, "spec_verify_b", (CH as u32,1,1), (nthr,1,1), smem,
                (d(&p_out), d(&r_out), d(&work), d(&dt_dev), d(&dq_dev),
                 d(&t_dev), d(&k_dev), d(&p_dev), d(&sd_dev), vocab as i32, CH as i32));
            self.sync_stream();
            let pv = self.dev.dtoh_sync_copy(&p_out).unwrap();
            let rv = self.dev.dtoh_sync_copy(&r_out).unwrap();
            let bv = [rv[CH - 1]];   // the bonus column now rides in resid_tok[depth-1]
            p_draft_device = pv[0];

            for c in 0..per_launch {
                if done >= trials { break; }
                // q = 1 for a greedy draft, so the accept ratio is exactly p(x_draft).
                let ru = rng_uniform(keys[c], RNG_DOM_ACCEPT, 0);
                let emit = if ru < pv[c] { n_accept += 1; x_draft } else { rv[c] as u32 };
                hist_b[emit as usize] += 1;
                n_b += 1;
                done += 1;
            }
            hist_c[bv[0] as usize] += 1;
            n_c += 1;
            li += 1;
        }

        // The draft is accepted with probability exactly p(x_draft), so the accept rate is itself a
        // binomial check on the accept rule (independent of what the residual then emits).
        let accept_rate = n_accept as f32 / n_b.max(1) as f32;
        let pd = p_draft_analytic as f64;
        let accept_z = if pd <= 0.0 || pd >= 1.0 {
            0.0
        } else {
            ((accept_rate as f64 - pd) / (pd * (1.0 - pd) / n_b as f64).sqrt()) as f32
        };

        MtpSampleStats {
            x_draft,
            nucleus_size: p_analytic.len(),
            p_draft_analytic,
            p_draft_device,
            accept_rate,
            accept_z,
            trials: n_b as usize,
            bonus_trials: n_c as usize,
            sampler: gof(&hist_a, n_a, &p_analytic),
            mtp:     gof(&hist_b, n_b, &p_analytic),
            bonus:   gof(&hist_c, n_c, &p_analytic),
            mtp_vs_sampler:   gof2(&hist_b, n_b, &hist_a, n_a),
            bonus_vs_sampler: gof2(&hist_c, n_c, &hist_a, n_a),
        }
    }

    /// DIAGNOSTIC: why is MTP acceptance 39.5% on tool-calling traffic and ~80% on prose?
    ///
    /// Two causes look identical from the outside and have OPPOSITE fixes:
    ///   * the DRAFT HEAD IS WEAK  — the target is confident and predictable, the head still misses.
    ///                               Fixable: distil a better head, FR-Spec, more draft capacity.
    ///   * the TEXT IS HARD        — the target itself is near-tied between tokens, so its argmax is
    ///                               close to a coin flip. NO draft head can track that. Irreducible:
    ///                               stop spending columns on draft depth, spend them on lanes.
    ///
    /// The discriminator is the TARGET's own confidence at the positions where the draft was rejected.
    /// So: run the real greedy MTP loop and, at every verify column, record the target's top-1
    /// probability and its top1-top2 margin alongside whether the draft matched. Then bucket
    /// acceptance by target confidence.
    ///
    /// (Note: the `1 - TV` acceptance ceiling from the speculative-decoding literature is for
    /// REJECTION SAMPLING. Greedy MTP accepts on ARGMAX AGREEMENT, which is not bounded by 1 - TV —
    /// two distributions can differ wildly and still share an argmax. Hence measuring directly.)
    /// `ngram`: if > 0, override each draft with a prompt-lookup n-gram proposal when the last `ngram`
    /// tokens of the running context appear earlier in it. The MTP forward still runs on the PROPOSED
    /// token, so the draft chain stays coherent; only WHICH token is proposed changes. Verify checks
    /// every draft, so this can never affect correctness — only acceptance.
    /// Build a full `TreeTopo` from a `parent[]` array (parent[0] = -1). Derives rope (depth), kv_pos
    /// (DFS slot), path (ancestor KV-slot offsets) and winsrc (ancestor-path conv window) generically,
    /// so any tree shape is expressible by its parent array alone.
    pub fn topo_from_parent(&self, parent: &[i32], pos_start: usize) -> TreeTopo {
        let n = parent.len();
        let ck = self.cfg.conv_kernel;
        let mut rope = vec![0i32; n];
        let mut kv_pos = vec![0i32; n];
        let mut path = vec![0u8; n * MAX_VERIFY];
        let mut winsrc = vec![0i32; n * ck];
        for c in 0..n {
            // ancestor DFS indices, root-first: walk[depth] = the column at that depth on c's path.
            let mut walk = vec![c as i32];
            let mut x = parent[c];
            while x >= 0 { walk.push(x); x = parent[x as usize]; }
            walk.reverse();
            let depth = walk.len() - 1;
            rope[c] = (pos_start + depth) as i32;
            kv_pos[c] = (pos_start + c) as i32;
            for (dd, &idx) in walk.iter().enumerate() { path[c * MAX_VERIFY + dd] = idx as u8; }
            for j in 0..ck {
                let wd = depth as i32 - (ck as i32 - 1) + j as i32;   // window depth
                winsrc[c * ck + j] = if wd < 0 { wd } else { walk[wd as usize] };
            }
        }
        TreeTopo { rope, kv_pos, parent: parent.to_vec(), path, winsrc, slot: None, col_pos_start: None }
    }

    /// Step-2.9 TWIN-CHAIN byte gate. Plant a fork whose two branches carry IDENTICAL tokens; the two
    /// branches' columns at equal depth must produce BIT-EQUAL logits (equal logical pc, equal ancestor
    /// tokens => identical rank-space computation). This is the direct detector for the §1 slot-derived-pc
    /// flaw: under that design the twins drift by ulps. Returns (n_columns_compared, n_bit_mismatches).
    pub fn bench_tree_twin(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                           kv_stride: usize, depth: usize) -> (usize, usize) {
        let vocab = self.cfg.vocab_size;
        let plen = prompt.len();
        // Prefill the prompt into slot 0 (sets prefix KV + GDN/conv state).
        self.zero_slot_state(state, 0, kv_stride);
        let (committed, hout) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(hout, self.cfg.hidden_size * plen);

        // Twin fork: col 0 = committed; a-branch cols 1..d; b-branch cols d..2d-1 with IDENTICAL tokens.
        let n = 2 * depth - 1;
        assert!(n <= MAX_VERIFY, "twin tree width {n} exceeds MAX_VERIFY");
        let mut parent = vec![-1i32; n];
        for i in 1..depth { parent[i] = (i - 1) as i32; }         // a-branch chains from root
        parent[depth] = 0;                                        // b1 from root
        for j in 1..depth - 1 { parent[depth + j] = (depth + j - 1) as i32; }

        let mut tokens = vec![committed];
        let draft: Vec<u32> = (0..depth - 1).map(|i| ((i * 131 + 7) % (vocab - 1) + 1) as u32).collect();
        tokens.extend_from_slice(&draft);   // a-branch
        tokens.extend_from_slice(&draft);   // b-branch (identical)

        let topo = self.topo_from_parent(&parent, plen);
        // ckpt_slot=Some(2): the GDN scan writes each column's post-state into checkpoint slot (2+t), and
        // a branch reloads its parent's from there. REQUIRED for a tree (a branch's parent != t-1).
        // The caller must allocate >= 2 + n GDN state slots. Slot 0 is the lane; 1 is the MTP snapshot.
        let (logits, resid) = self.verify_forward_core_topo(pool, &tokens, state, 0, kv_stride, plen, Some(2), Some(&topo));
        self.sync_stream();
        let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&logits).unwrap();
        pool.release_bf16(logits, vocab * n);
        pool.release_bf16(resid, self.cfg.hidden_size * n);

        // Compare a-branch col i vs b-branch col i, bitwise over the whole logit row.
        let mut mism = 0usize;
        let mut cmp = 0usize;
        for i in 1..depth {
            let a = &lg[i * vocab..(i + 1) * vocab];
            let bcol = depth - 1 + i;                              // b-branch col at depth i
            let b = &lg[bcol * vocab..(bcol + 1) * vocab];
            cmp += 1;
            if a.iter().map(|x| x.to_bits()).ne(b.iter().map(|x| x.to_bits())) { mism += 1; }
        }
        (cmp, mism)
    }

    /// Step-2.9 PATH-ORACLE byte gate — the strongest: for a RANDOM tree, every column's logits must be
    /// bit-equal to running that column's root-to-leaf token path as an independent PLAIN CHAIN. This is
    /// absolute ground truth (the twin test is relative — it would miss a bug that shifts both twins the
    /// same way). GDN state is snapshotted after the prefill (slot 1) and restored before every run, so
    /// each chain and the tree see the identical prefix state; the prefix KV [0,plen) is never
    /// overwritten (runs write at plen..). Returns (columns_checked, bit_mismatches).
    pub fn bench_tree_oracle(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                             kv_stride: usize, n: usize, seed: u64) -> (usize, usize) {
        let vocab = self.cfg.vocab_size;
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        self.zero_slot_state(state, 0, kv_stride);
        let (committed, hout) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(hout, h * plen);
        self.copy_gdn_slot(state, 0, 1);   // snapshot post-prefill GDN state into slot 1

        // Random tree: parent[c] is a random earlier column (< c), giving a valid DFS tree. Random tokens.
        let mut rng = seed | 1;
        let mut next = |m: u64| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng % m };
        let mut parent = vec![-1i32; n];
        for c in 1..n { parent[c] = next(c as u64) as i32; }
        let tokens: Vec<u32> = std::iter::once(committed)
            .chain((1..n).map(|_| (next((vocab - 2) as u64) + 1) as u32)).collect();

        // Tree run: restore state, verify the whole tree, capture per-column logits.
        self.copy_gdn_slot(state, 1, 0);
        let topo = self.topo_from_parent(&parent, plen);
        let (tl, tr) = self.verify_forward_core_topo(pool, &tokens, state, 0, kv_stride, plen, Some(2), Some(&topo));
        self.sync_stream();
        let tree_lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&tl).unwrap();
        pool.release_bf16(tl, vocab * n); pool.release_bf16(tr, h * n);

        // For each column, run its root-to-leaf path as a plain chain and compare the leaf's logits.
        let mut mism = 0usize;
        for c in 0..n {
            let mut walk = vec![c]; let mut x = parent[c];
            while x >= 0 { walk.push(x as usize); x = parent[x as usize]; }
            walk.reverse();
            let path_toks: Vec<u32> = walk.iter().map(|&i| tokens[i]).collect();
            self.copy_gdn_slot(state, 1, 0);   // restore prefix state
            let (cl, cr) = self.verify_forward_core(pool, &path_toks, state, 0, kv_stride, plen, Some(2));
            self.sync_stream();
            let chain_lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&cl).unwrap();
            let leaf = path_toks.len() - 1;
            pool.release_bf16(cl, vocab * path_toks.len()); pool.release_bf16(cr, h * path_toks.len());
            let a = &tree_lg[c * vocab..(c + 1) * vocab];
            let b = &chain_lg[leaf * vocab..(leaf + 1) * vocab];
            if a.iter().map(|x| x.to_bits()).ne(b.iter().map(|x| x.to_bits())) { mism += 1; }
        }
        (n, mism)
    }

    /// FOREST oracle (batched verify across lanes — LANES design Step 3a/3b). Pack TWO independent draft
    /// chains — one per lane, each rooted in its OWN committed slot state — into ONE verify, and assert
    /// every lane's per-column logits are BIT-EQUAL to running that lane's chain ALONE. This is the
    /// batch-invariance / GDN-forest-independence gate (`LANES_OK`): a lane's emitted logits must not
    /// depend on which other lanes share the forward. Lanes may have DIFFERENT committed lengths — each
    /// column carries its lane's prefix boundary via the topo's `col_pos_start` (Step 3b), so this also
    /// gates the per-column pos_start in the rank-space attention. GDN state of each lane is snapshotted
    /// (slots 2,3) and restored before every run; committed prefix KV is never overwritten (runs write
    /// at plen_lane..). Returns (columns, bit_mismatches).
    pub fn bench_lanes(&self, pool: &mut Pool, state: &mut BatchGpuState,
                       prompt_a: &[u32], prompt_b: &[u32], kv_stride: usize, depth: usize) -> (usize, usize) {
        let vocab = self.cfg.vocab_size;
        let h = self.cfg.hidden_size;
        let ck = self.cfg.conv_kernel;
        let plen_l = [prompt_a.len(), prompt_b.len()];
        assert!(2 * depth <= MAX_VERIFY, "forest width {} exceeds MAX_VERIFY {}", 2 * depth, MAX_VERIFY);

        // Prefill lane 0 -> slot 0, lane 1 -> slot 1; snapshot each GDN state to scratch slots 2, 3.
        self.zero_slot_state(state, 0, kv_stride);
        let (comm_a, ha) = self.prefill_batch(pool, prompt_a, state, 0, kv_stride, 0);
        pool.release_bf16(ha, h * plen_l[0]);
        self.zero_slot_state(state, 1, kv_stride);
        let (comm_b, hb) = self.prefill_batch(pool, prompt_b, state, 1, kv_stride, 0);
        pool.release_bf16(hb, h * plen_l[1]);
        self.copy_gdn_slot(state, 0, 2);
        self.copy_gdn_slot(state, 1, 3);

        // Distinct draft tokens per lane (verify checks them; the oracle only needs them equal across
        // the packed and alone runs). Each lane's chain = [committed, drafts...], `depth` columns.
        let draft_a: Vec<u32> = (0..depth - 1).map(|i| ((i * 131 + 7) % (vocab - 1) + 1) as u32).collect();
        let draft_b: Vec<u32> = (0..depth - 1).map(|i| ((i * 197 + 23) % (vocab - 1) + 1) as u32).collect();
        let toks_a: Vec<u32> = std::iter::once(comm_a).chain(draft_a.iter().copied()).collect();
        let toks_b: Vec<u32> = std::iter::once(comm_b).chain(draft_b.iter().copied()).collect();

        // FOREST topo: columns [lane0 chain (depth), lane1 chain (depth)]. Each lane's first column is a
        // ROOT (parent -1); positions/path are LANE-LOCAL (rank within the lane relative to its OWN
        // committed length), the per-column `slot` routes KV/GDN/conv to the lane, and `col_pos_start`
        // gives the lane's prefix boundary so the attention splits prefix/ancestor at the right point.
        let n = 2 * depth;
        let mut tokens = toks_a.clone(); tokens.extend_from_slice(&toks_b);
        let mut parent = vec![-1i32; n];
        let mut slotv = vec![0i32; n];
        let mut rope = vec![0i32; n];
        let mut kv_pos = vec![0i32; n];
        let mut cps = vec![0i32; n];
        let mut path = vec![0u8; n * MAX_VERIFY];
        let mut winsrc = vec![0i32; n * ck];
        for lane in 0..2usize {
            let base = lane * depth;
            let plen = plen_l[lane];
            for r in 0..depth {                          // r = rank within the lane
                let c = base + r;
                parent[c] = if r == 0 { -1 } else { (c - 1) as i32 };
                slotv[c] = lane as i32;
                rope[c] = (plen + r) as i32;
                kv_pos[c] = (plen + r) as i32;
                cps[c] = plen as i32;                    // this lane's prefix boundary
                for dd in 0..=r { path[c * MAX_VERIFY + dd] = dd as u8; }  // ancestor at lane-KV offset dd
                for j in 0..ck {
                    // conv window: `in[wd*conv_dim]` indexes the GLOBAL packed qkv buffer, so a positive
                    // window depth maps to the GLOBAL column base+wd; a negative one is the committed tail.
                    let wd = r as i32 - (ck as i32 - 1) + j as i32;
                    winsrc[c * ck + j] = if wd < 0 { wd } else { base as i32 + wd };
                }
            }
        }
        let topo = TreeTopo { rope, kv_pos, parent, path, winsrc, slot: Some(slotv), col_pos_start: Some(cps) };
        // max_pc upper bound for the packed verify grid = max lane length + n.
        let pos_start_max = *plen_l.iter().max().unwrap();

        // Packed forest verify (restore both lanes). Chains never backtrack, so no mid_s (ckpt_slot=None).
        self.copy_gdn_slot(state, 2, 0);
        self.copy_gdn_slot(state, 3, 1);
        let (pl, pr) = self.verify_forward_core_topo(pool, &tokens, state, 0, kv_stride, pos_start_max, None, Some(&topo));
        self.sync_stream();
        let packed: Vec<half::bf16> = self.dev.dtoh_sync_copy(&pl).unwrap();
        pool.release_bf16(pl, vocab * n); pool.release_bf16(pr, h * n);

        // Each lane ALONE (single-lane path), from its own restored committed state and pos_start.
        self.copy_gdn_slot(state, 2, 0);
        let (al, ar) = self.verify_forward_core(pool, &toks_a, state, 0, kv_stride, plen_l[0], None);
        self.sync_stream();
        let alone_a: Vec<half::bf16> = self.dev.dtoh_sync_copy(&al).unwrap();
        pool.release_bf16(al, vocab * depth); pool.release_bf16(ar, h * depth);

        self.copy_gdn_slot(state, 3, 1);
        let (bl, br) = self.verify_forward_core(pool, &toks_b, state, 1, kv_stride, plen_l[1], None);
        self.sync_stream();
        let alone_b: Vec<half::bf16> = self.dev.dtoh_sync_copy(&bl).unwrap();
        pool.release_bf16(bl, vocab * depth); pool.release_bf16(br, h * depth);

        // Bitwise compare each lane's columns: packed vs alone.
        let mut mism = 0usize;
        for r in 0..depth {
            let pa = &packed[r * vocab..(r + 1) * vocab];
            let aa = &alone_a[r * vocab..(r + 1) * vocab];
            if pa.iter().map(|x| x.to_bits()).ne(aa.iter().map(|x| x.to_bits())) { mism += 1; }
            let pb = &packed[(depth + r) * vocab..(depth + r + 1) * vocab];
            let bb = &alone_b[r * vocab..(r + 1) * vocab];
            if pb.iter().map(|x| x.to_bits()).ne(bb.iter().map(|x| x.to_bits())) { mism += 1; }
        }
        (n, mism)
    }

    /// Step-3a end-to-end gate: verify + accept-walk + KV compaction on a planted fork whose SECOND
    /// branch carries the target's own greedy continuation (so it is fully accepted) and whose FIRST
    /// branch is a decoy (rejected at depth 1). Checks: (a) the accepted path is the decoy's sibling
    /// chain, (b) compaction moves each accepted node's KV to its contiguous slot (bitwise), (c) the GDN
    /// leaf checkpoint adoption lands the right state. Returns (emitted_matches_greedy, kv_compacted_ok).
    pub fn bench_tree_accept(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                             kv_stride: usize, depth: usize) -> (bool, bool) {
        let vocab = self.cfg.vocab_size;
        let (nkv, hd) = (self.eff_num_kv_heads(), self.cfg.head_dim);   // eff: matches compact_kv (KV is head-sharded under TP)
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        self.zero_slot_state(state, 0, kv_stride);
        let (committed, hout) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(hout, h * plen);
        self.copy_gdn_slot(state, 0, 1);   // snapshot post-prefill GDN

        // The target's own greedy continuation, built one token at a time via chain verify.
        let mut greedy = vec![committed];
        for _ in 0..depth {
            self.copy_gdn_slot(state, 1, 0);
            let (l, r) = self.verify_forward_core(pool, &greedy, state, 0, kv_stride, plen, Some(2));
            self.sync_stream();
            let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&l).unwrap();
            pool.release_bf16(l, vocab * greedy.len()); pool.release_bf16(r, h * greedy.len());
            let last = greedy.len() - 1;
            let col = &lg[last * vocab..(last + 1) * vocab];
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &x) in col.iter().enumerate() { let v = x.to_f32(); if v > bv { bv = v; bi = i; } }
            greedy.push(bi as u32);
        }
        // greedy = [committed, g1, g2, ..., g_depth]. Accepted branch tokens = g1..g_{depth-1}; bonus g_depth.

        // Plant fork-then-chain: col0=committed; col1=decoy (child of root); cols 2..depth = B chain (g1..).
        let n = depth + 1;   // root + decoy + (depth-1) B nodes
        assert!(n <= MAX_VERIFY);
        let mut parent = vec![-1i32; n];
        parent[1] = 0;                                   // decoy, child of root
        parent[2] = 0;                                   // B1, child of root
        for j in 3..n { parent[j] = (j - 1) as i32; }    // B chain
        let decoy = if greedy[1] == 1 { 2 } else { 1 };  // any token != g1
        let mut tokens = vec![committed, decoy];
        tokens.extend_from_slice(&greedy[1..depth]);     // B branch = g1..g_{depth-1}
        assert_eq!(tokens.len(), n);

        self.copy_gdn_slot(state, 1, 0);
        let topo = self.topo_from_parent(&parent, plen);
        let (tl, tr) = self.verify_forward_core_topo(pool, &tokens, state, 0, kv_stride, plen, Some(2), Some(&topo));
        self.sync_stream();
        let tlg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&tl).unwrap();
        pool.release_bf16(tl, vocab * n); pool.release_bf16(tr, h * n);
        // preds per column
        let preds: Vec<u32> = (0..n).map(|c| {
            let col = &tlg[c*vocab..(c+1)*vocab];
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i,&x) in col.iter().enumerate() { let v=x.to_f32(); if v>bv { bv=v; bi=i; } }
            bi as u32
        }).collect();

        // Accept walk (host).
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for c in 1..n { children[parent[c] as usize].push(c); }
        let mut path = vec![0usize]; let mut emitted = vec![]; let mut cur = 0usize;
        loop {
            let want = preds[cur]; emitted.push(want);
            match children[cur].iter().copied().find(|&c| tokens[c] == want) {
                Some(c) => { path.push(c); cur = c; } None => break,
            }
        }
        // Expected emitted = greedy[1..=depth].
        let emit_ok = emitted == greedy[1..=depth].to_vec();

        // Compaction: snapshot the scattered KV of the accepted path, compact, compare.
        let src_pos: Vec<i32> = path.iter().map(|&p| p as i32).collect();
        // read the pre-compact KV at the accepted slots (layer 0's first full-attn layer) for the check
        let fa0 = self.cfg.layer_types.iter().position(|t| matches!(t, LayerType::FullAttention)).unwrap();
        let before: Vec<half::bf16> = self.dev.dtoh_sync_copy(state.k_cache[fa0].as_ref().unwrap()).unwrap();
        self.compact_kv(pool, state, 0, plen, &src_pos, kv_stride);
        self.sync_stream();
        let after: Vec<half::bf16> = self.dev.dtoh_sync_copy(state.k_cache[fa0].as_ref().unwrap()).unwrap();
        // after compaction, position (plen+k) must equal before's position (plen + src_pos[k]) for every head.
        let mut kv_ok = true;
        for k in 0..path.len() {
            for hh in 0..nkv {
                let dst = (((0*nkv + hh) * kv_stride) + plen + k) * hd;
                let src = (((0*nkv + hh) * kv_stride) + plen + src_pos[k] as usize) * hd;
                if before[src..src+hd].iter().map(|x|x.to_bits()).ne(after[dst..dst+hd].iter().map(|x|x.to_bits())) {
                    kv_ok = false;
                }
            }
        }
        (emit_ok, kv_ok)
    }

    pub fn bench_accept(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                        kv_stride: usize, depth: usize, max_new: usize, ngram: usize)
                        -> (Vec<AcceptSample>, Vec<u32>) {
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let plen = prompt.len();
        let nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;
        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        self.dev.synchronize().unwrap();
        let mtp_kc_ptr = *mtp_kc.device_ptr();
        let mtp_vc_ptr = *mtp_vc.device_ptr();
        let h_prev = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let h_save = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let h_scratch = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let cur_hidden = self.dev.alloc_zeros::<half::bf16>(h).unwrap();

        self.zero_slot_state(state, 0, kv_stride);
        let (a0, hout0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);

        let copy_stream = self.stream.stream;
        let copy_col = |dst_ptr: u64, src_buf: &B, col: usize| unsafe {
            let src = *src_buf.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src, h * 2, copy_stream).unwrap();
        };

        self.mtp_prime_prompt(pool, &hout0, &prompt[1..plen], mtp_kc_ptr, mtp_vc_ptr, kv_stride, 0);
        let mut mtp_pos = plen - 1;
        copy_col(*h_prev.device_ptr(), &hout0, plen - 1);
        pool.release_bf16(hout0, h * plen);

        let mut out = vec![a0];
        let mut committed_tok = a0;
        let mut main_pos = plen;
        let mut samples: Vec<AcceptSample> = Vec::new();

        while out.len() < max_new {
            copy_col(*h_save.device_ptr(), &h_prev, 0);
            let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
            let mut head_top3: Vec<[u32; 3]> = Vec::with_capacity(depth - 1);   // head's top-3 per draft pos
            copy_col(*cur_hidden.device_ptr(), &h_prev, 0);
            let mut cur_tok = committed_tok as i32;
            let mut dpos = mtp_pos;
            // Running context for the n-gram lookup: the full realized sequence, extended with the
            // drafts as we propose them (each draft becomes context for the next lookup).
            let mut work: Vec<u32> = Vec::new();
            if ngram > 0 { work.extend_from_slice(prompt); work.extend_from_slice(&out[1..]); }
            for _ in 0..depth - 1 {
                let m = self.mtp_draft_step(pool, &cur_hidden, cur_tok, dpos,
                                            mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                dpos += 1;
                copy_col(*cur_hidden.device_ptr(), &m, 0);
                pool.release_bf16(m, h);
                let t3 = self.topk_hidden(pool, &cur_hidden, 3);   // top-1 == argmax (the draft)
                head_top3.push([t3[0], *t3.get(1).unwrap_or(&t3[0]), *t3.get(2).unwrap_or(&t3[0])]);
                cur_tok = t3[0] as i32;
                // PROMPT-LOOKUP OVERRIDE: if the last `ngram` tokens of the context recur earlier, the
                // token that followed the most recent earlier occurrence is a free, exact-copy draft.
                // Prefer it over the 1-layer head's guess on copyable (tool/JSON) text. The MTP hidden
                // was computed for the PREVIOUS token, so overriding this draft keeps the chain valid:
                // the next step feeds (this hidden, the overridden token), exactly as MTP chains.
                if ngram > 0 && work.len() >= ngram {
                    let tail = &work[work.len() - ngram..];
                    for i in (0..work.len() - ngram).rev() {
                        if &work[i..i + ngram] == tail {
                            if i + ngram < work.len() { cur_tok = work[i + ngram] as i32; }
                            break;
                        }
                    }
                }
                drafts.push(cur_tok as u32);
                if ngram > 0 { work.push(cur_tok as u32); }
            }

            let mut verify_input = vec![committed_tok];
            verify_input.extend(drafts.iter().copied());
            let n = verify_input.len();

            // Same verify the serving path runs, but keep the LOGITS so we can read the target's
            // confidence rather than only its argmax.
            let (logits, vout) = self.verify_forward_core(
                pool, &verify_input, state, 0, kv_stride, main_pos, Some(2));
            self.sync_stream();
            let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&logits).unwrap();
            pool.release_bf16(logits, vocab * n);

            // Per verify column: target argmax, top-1 probability, and the top1-top2 margin.
            let mut preds = Vec::with_capacity(n);
            let mut conf = Vec::with_capacity(n);
            for c in 0..n {
                let col = &lg[c * vocab..(c + 1) * vocab];
                let (mut b1, mut b2, mut i1) = (f32::NEG_INFINITY, f32::NEG_INFINITY, 0usize);
                for (i, &x) in col.iter().enumerate() {
                    let v = x.to_f32();
                    if v > b1 { b2 = b1; b1 = v; i1 = i; } else if v > b2 { b2 = v; }
                }
                // softmax top-1 probability (subtract max for stability)
                let denom: f32 = col.iter().map(|&x| (x.to_f32() - b1).exp()).sum();
                preds.push(i1 as u32);
                conf.push((1.0 / denom, b1 - b2));      // (p_top1, logit margin)
            }

            let mut nacc = 0usize;
            while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }

            // Record only the draft positions whose chain was STILL ON THE CORRECT PREFIX: 0..=nacc.
            //
            // Past the first rejection the drafts are conditioned on a token the target did not
            // choose, so their failure says nothing about the head — counting them would slander it.
            // Position `nacc` (the first rejection, when there is one) is the single most informative
            // sample we get: correct context, head still missed. That is where "weak head" and "hard
            // text" separate.
            for i in 0..drafts.len().min(nacc + 1) {
                let (p1, margin) = conf[i];
                // Does the target's argmax (preds[i]) land in the head's top-2 / top-3 here? That is
                // exactly the coverage a fork offering the head's top-k at this position would rescue.
                let tgt = preds[i];
                let t3 = head_top3[i];
                samples.push(AcceptSample {
                    depth_idx: i + 1,
                    target_top1_p: p1,
                    target_margin: margin,
                    accepted: i < nacc,          // the prefix rule: accepted iff every earlier one was
                    covered_top2: tgt == t3[0] || tgt == t3[1],
                    covered_top3: tgt == t3[0] || tgt == t3[1] || tgt == t3[2],
                });
            }

            let bonus = preds[nacc];
            for &d in drafts.iter().take(nacc) { out.push(d); }
            if out.len() < max_new { out.push(bonus); }

            if nacc + 1 != depth { self.copy_gdn_slot(state, 2 + nacc, 0); }
            copy_col(*h_prev.device_ptr(), &vout, nacc);

            for k in 0..=nacc {
                if k == 0 {
                    let m = self.mtp_draft_step(pool, &h_save, committed_tok as i32, main_pos - 1,
                                                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                    pool.release_bf16(m, h);
                } else {
                    copy_col(*h_scratch.device_ptr(), &vout, k - 1);
                    let m = self.mtp_draft_step(pool, &h_scratch, drafts[k - 1] as i32, main_pos - 1 + k,
                                                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                    pool.release_bf16(m, h);
                }
            }
            pool.release_bf16(vout, h * depth);

            committed_tok = *out.last().unwrap();
            main_pos += nacc + 1;
            mtp_pos += nacc + 1;
            if main_pos + depth + 2 >= kv_stride { break; }
        }
        (samples, out)
    }

    pub fn bench_mtp(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                     kv_stride: usize, depth: usize, max_new: usize)
                    -> (Vec<u32>, Vec<u32>, f32, f32, f32) {
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        let bufs = self.new_decode_buffers(3);
        let nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;
        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
        // alloc_zeros is cuMemAllocAsync — does NOT zero. The MTP attention must not read garbage
        // at unwritten KV positions, so zero both caches explicitly.
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        self.dev.synchronize().unwrap();
        let mtp_kc_ptr = *mtp_kc.device_ptr();
        let mtp_vc_ptr = *mtp_vc.device_ptr();
        let h_prev = self.dev.alloc_zeros::<half::bf16>(h).unwrap();   // cursor hidden (seeds drafts)
        let h_scratch = self.dev.alloc_zeros::<half::bf16>(h).unwrap();    // hidden-column extract scratch
        let h_save = self.dev.alloc_zeros::<half::bf16>(h).unwrap();       // pre-verify hidden (re-prime k=0)
        let cur_hidden = self.dev.alloc_zeros::<half::bf16>(h).unwrap();   // draft-chain cursor hidden

        // Prefill slot 0 (MTP) and slot 1 (sequential ground truth).
        self.zero_slot_state(state, 0, kv_stride);
        let (a0, hout0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        self.zero_slot_state(state, 1, kv_stride);
        let (a1, _hout1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        assert_eq!(a0, a1, "prefill divergence slot0 vs slot1");

        // Helper: copy hidden column `col` of an [h, n] buffer into a [h] device buffer, on the
        // compute stream (stream-ordered with the kernels that produced/consume these hiddens).
        let copy_stream = self.stream.stream;
        let copy_col = |dst_ptr: u64, src_buf: &B, col: usize| unsafe {
            let src = *src_buf.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src, h * 2, copy_stream).unwrap();
        };

        // Prompt-prime MTP over main positions 0..plen-2: step t uses (h_t, prompt[t+1]).
        // Same primitive the server uses -- see the note above.
        self.mtp_prime_prompt(pool, &hout0, &prompt[1..plen], mtp_kc_ptr, mtp_vc_ptr, kv_stride, 0);
        let mut mtp_pos = plen - 1;   // next main position for an MTP write
        // h_prev = h at plen-1 (last prompt position) — seeds the first draft.
        copy_col(*h_prev.device_ptr(), &hout0, plen - 1);
        pool.release_bf16(hout0, h * plen);

        let mut mtp_tokens = vec![a0];      // a0 (prefill's first token) is already emitted
        let mut committed_tok = a0;         // token for the next position the verify will process
        let mut main_pos = plen;           // position committed_tok will be verified at
        let mtp_start = std::time::Instant::now();
        // The lossless gate decodes a sequential ground-truth token for every token MTP emits, in the
        // same loop. That work must NOT land in the MTP timer — it used to, which is why this probe
        // reported MTP at roughly half its real rate and then papered over it by *assigning* the MTP
        // duration to the sequential one (`speedup` was 1.00x by construction, on every run).
        let mut seq_dur = 0.0f32;
        let mut total_drafts = 0usize;
        let mut total_accepted = 0usize;
        let mut n_steps = 0usize;

        // Lockstep greedy ground-truth on slot 1: advanced in parallel with the MTP loop so the
        // emitted MTP token stream can be compared against clean sequential greedy (the lossless gate).
        let mut bufs2 = self.new_decode_buffers(2);
        self.dev.htod_sync_copy_into(&[1i32, 0], &mut bufs2.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();
        let mut seq_tokens = vec![a0];
        let mut spos = plen;

        while mtp_tokens.len() < max_new {
            n_steps += 1;
            copy_col(*h_save.device_ptr(), &h_prev, 0);

            // ---- Draft chain: d_i from (prev_hidden, prev_token) at MTP positions mtp_pos.. ----
            // Uses persistent buffers + compute-stream copy_col (NOT CudaSlice::clone, which dtod-copies
            // on the device/NULL stream and races with the compute stream).
            let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
            copy_col(*cur_hidden.device_ptr(), &h_prev, 0);
            let mut cur_tok = committed_tok as i32;
            let mut dpos = mtp_pos;
            for _ in 0..depth - 1 {
                let m = self.mtp_draft_step(pool, &cur_hidden, cur_tok, dpos,
                                            mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                dpos += 1;
                copy_col(*cur_hidden.device_ptr(), &m, 0);
                pool.release_bf16(m, h);
                cur_tok = self.argmax_hidden(pool, &cur_hidden) as i32;
                drafts.push(cur_tok as u32);
                total_drafts += 1;
            }

            // ---- Verify [committed_tok, drafts...] with ping-pong GDN checkpoint. ----
            // The GDN kernels snapshot S1 (recurrent state after the always-accepted committed token)
            // into slot 2. On draft rejection we restore S1 with a dtod copy — NO second model
            // forward. vout's column nacc already holds the hidden at the last accepted position in
            // BOTH accept/reject cases (the old reverify's hidden was identical), so it is reused.
            let mut verify_input = vec![committed_tok];
            verify_input.extend(drafts.iter().copied());
            let (preds, vout) = self.verify_forward(pool, &verify_input, state, 0, kv_stride, main_pos, Some(2), None);

            // ---- Accept longest prefix (committed_tok already emitted, don't re-emit). ----
            let mut nacc = 0usize;
            while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
            let bonus = preds[nacc];
            let emitted_count_before = mtp_tokens.len();
            for &d in drafts.iter().take(nacc) { mtp_tokens.push(d); }
            if mtp_tokens.len() < max_new { mtp_tokens.push(bonus); }
            total_accepted += nacc;

            // ---- GDN rollback on partial accept: restore the state as of the LAST ACCEPTED column.
            // Checkpoint slots are contiguous from slot 2: slot (2 + t) holds verify column t's
            // post-state. Restoring slot 2 unconditionally (the old behaviour) is only correct at
            // depth 2 and silently corrupted the recurrent state at any deeper setting.
            if nacc + 1 != depth {
                self.copy_gdn_slot(state, 2 + nacc, 0);
            }
            // h_prev = hidden at the last accepted position (vout column nacc).
            copy_col(*h_prev.device_ptr(), &vout, nacc);

            // ---- Re-prime MTP over the accepted prefix with REAL hiddens (vout) so its KV stays
            // ---- consistent. k=0 uses h_save; k>=1 uses vout col k-1.
            for k in 0..=nacc {
                if k == 0 {
                    let m = self.mtp_draft_step(pool, &h_save, committed_tok as i32, main_pos - 1,
                                                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                    pool.release_bf16(m, h);
                } else {
                    copy_col(*h_scratch.device_ptr(), &vout, k - 1);
                    let m = self.mtp_draft_step(pool, &h_scratch, drafts[k - 1] as i32, main_pos - 1 + k,
                                                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                    pool.release_bf16(m, h);
                }
            }
            pool.release_bf16(vout, h * depth);

            // ---- Lockstep greedy check on slot 1: decode one token per emitted MTP token. ----
            // Timed separately: this IS the sequential baseline, and it is not part of the MTP step.
            let seq_t0 = std::time::Instant::now();
            for &_emt in &mtp_tokens[emitted_count_before..] {
                let tok = *seq_tokens.last().unwrap() as i32;
                self.dev.htod_sync_copy_into(&vec![tok, 0], &mut bufs2.tokens_dev).unwrap();
                self.dev.htod_sync_copy_into(&vec![spos as i32, 0], &mut bufs2.pos_dev).unwrap();
                self.dev.synchronize().unwrap();
                let next = self.forward_decode(pool, &mut bufs2, state, kv_stride, spos + 1, 1);
                seq_tokens.push(next[0]);
                spos += 1;
            }
            seq_dur += seq_t0.elapsed().as_secs_f32();

            mtp_pos = main_pos + nacc;
            main_pos += nacc + 1;
            committed_tok = bonus;
            if mtp_tokens.len() >= max_new { break; }
        }
        // MTP time = wall time minus the interleaved sequential ground-truth decode.
        let mtp_dur = (mtp_start.elapsed().as_secs_f32() - seq_dur).max(1e-6);
        drop(h_prev); drop(h_save); drop(h_scratch); drop(cur_hidden); drop(mtp_kc); drop(mtp_vc); drop(bufs);
        let seq_dur = seq_dur.max(1e-6);
        drop(bufs2);

        let mtp_tok_s = if mtp_dur > 0.0 { mtp_tokens.len() as f32 / mtp_dur } else { 0.0 };
        let seq_tok_s = if seq_dur > 0.0 { seq_tokens.len() as f32 / seq_dur } else { 0.0 };
        let accept_rate = if total_drafts > 0 { total_accepted as f32 / total_drafts as f32 } else { 0.0 };
        // PROBE 0 output: the two numbers the --profile-mtp phase table cannot give us. That table's
        // "modelled step" is the DEPTH-1 model (verify N=2, ONE draft), so the real depth-d step cost
        // and tokens/step have to be counted, not derived.
        if n_steps > 0 {
            eprintln!("[probe0] depth {depth}: {n_steps} steps, {} tokens => {:.3} tokens/step, \
                       {:.2} ms/step  |  drafts {total_drafts}, accepted {total_accepted} ({:.1}%)",
                      mtp_tokens.len(), mtp_tokens.len() as f32 / n_steps as f32,
                      mtp_dur * 1000.0 / n_steps as f32, accept_rate * 100.0);
        }
        let n = mtp_tokens.len().min(seq_tokens.len());
        (mtp_tokens[..n].to_vec(), seq_tokens[..n].to_vec(), mtp_tok_s, seq_tok_s, accept_rate)
    }

    /// Batched benchmark: M identical prompts prefilled + decoded together.
    /// Returns (tokens_of_slot0, aggregate_decode_tok_s). Tokens must match single-stream.
    #[allow(non_snake_case)]
    pub fn bench_batch(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                       M: usize, max_new: usize, max_seq_len: usize) -> (Vec<u32>, f32) {
        let v = self.cfg.vocab_size;
        let plen = prompt.len();
        let mut bufs = self.new_decode_buffers(M);
        // ---- prefill (uniform batched; all M slots identical) ----
        let mut last = vec![0u32; M];
        for (t, &tok) in prompt.iter().enumerate() {
            let hidden = self.embed_batch(&vec![tok; M]);
            let out = self.forward_batch(pool, hidden, &vec![t; M], state, max_seq_len, M);
            if t == plen - 1 {
                let logits = self.logits_batch(pool, &out, M);
                let block = 1024u32;
                blaunch!(self, "argmax_b", (M as u32,1,1), (block,1,1), (block*8),
                    (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, M as i32));
                self.sync_stream();
                last = self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
                    .into_iter().take(M).map(|x| x as u32).collect();
                pool.release_bf16(logits, v * M);
            }
            pool.release_bf16(out, self.cfg.hidden_size * M); // release to pool instead of freeing
        }
        let mut out_tokens = vec![last[0]];
        // ---- decode (optimized path: pre-computed rope, device-side embed/argmax) ----
        let start = std::time::Instant::now();
        for step in 0..max_new {
            let pos_val = plen + step;
            let max_pc = pos_val + 1;
            let toks_i32: Vec<i32> = last.iter().map(|&t| t as i32).collect();
            let pos_i32: Vec<i32> = vec![pos_val as i32; M];
            self.dev.htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap(); // htod on NULL must complete before compute stream
            let next = self.forward_decode(pool, &mut bufs, state, max_seq_len, max_pc, M);
            last.copy_from_slice(&next);
            out_tokens.push(last[0]);
        }
        let dur = start.elapsed().as_secs_f32();
        (out_tokens, (M * max_new) as f32 / dur)
    }

    /// TP=2 masked-replicated SPMD greedy decode — Stage 3 Proof v0. Both boxes run this IDENTICALLY
    /// over the same prompt; the per-layer FFN all-reduce (`mlp_batch`, world==2) keeps their residual
    /// streams bit-identical, so both independently greedy-argmax the SAME token every step with NO
    /// token broadcast. Returns the generated ids (both ranks compute them; the head decodes+prints).
    /// Prefill is token-by-token (batch=1) so every all-reduce is a single ~10 KB exchange. Stops on
    /// EOS or after `max_new` — deterministic on both ranks, so the exchange rendezvous stays in lockstep.
    pub fn tp_generate(&self, prompt: &[u32], max_new: usize, max_seq_len: usize) -> Vec<u32> {
        let v = self.cfg.vocab_size;
        let h = self.cfg.hidden_size;
        let eos = self.cfg.eos_token_id;
        let plen = prompt.len();
        let mut pool = Pool::new(self.dev.clone());
        let mut state = self.new_batch_state(1, 1, max_seq_len);
        let mut bufs = self.new_decode_buffers(1);

        // ---- prefill, token by token (batch=1) ----
        let mut last = 0u32;
        for (t, &tok) in prompt.iter().enumerate() {
            let hidden = self.embed_batch(&[tok]);
            let out = self.forward_batch(&mut pool, hidden, &[t], &mut state, max_seq_len, 1);
            if t == plen - 1 {
                let logits = self.logits_batch(&mut pool, &out, 1);
                let block = 1024u32;
                blaunch!(self, "argmax_b", (1u32,1,1), (block,1,1), (block*8),
                    (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, 1i32));
                self.sync_stream();
                last = self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()[0] as u32;
                pool.release_bf16(logits, v);
            }
            pool.release_bf16(out, h);
        }

        // ---- decode ----
        let mut out_tokens = Vec::with_capacity(max_new);
        if last == eos { return out_tokens; }
        out_tokens.push(last);
        // Per-token latency, not aggregate tok/s: an aggregate number folds any startup transient into
        // the steady state and reports the average as if it were the rate. Percentiles over the tokens
        // AFTER a warmup discard are what the comparison actually needs.
        let mut tok_ns: Vec<u64> = Vec::with_capacity(max_new);
        // CUDA-graph capture (GB10_TP_GRAPH=1). Collapses ~320 eager launches/token into one
        // cuGraphLaunch, which is where the measured 9.76 ms/token of non-GEMV time lives.
        //
        // `max_pc` is PINNED to the run maximum for every step, eager and replayed alike. It only feeds
        // `ns_grid`, which is a launch BOUND rather than a length — `gqa_attn_reduce` recomputes each
        // column's true split count from device-side `pos` and uses `ns_grid` purely as an indexing
        // stride — so over-estimating it is always correct and the surplus split blocks exit immediately.
        // Pinning it is what makes one captured graph valid at every position, and it keeps the eager
        // warm-up allocating exactly the pool buffers the capture will close over.
        let want_graph = self.tp_world == 2 && (std::env::var("GB10_TP_GRAPH").is_ok()
            || crate::tp::tp_config().map(|c| c.graph).unwrap_or(false));
        let max_pc_pinned = max_seq_len;
        let mut graph: Option<CudaGraph> = None;
        let (mut host_pre_ns, mut host_post_ns) = (Vec::new(), Vec::new());
        const GRAPH_CAPTURE_AT: usize = 8;   // let the pool and cuBLAS settle first
        for step in 0..max_new.saturating_sub(1) {
            let t_tok = std::time::Instant::now();
            let mut t_gpu_end = 0u64;
            let pos_val = plen + step;
            let max_pc = if want_graph { max_pc_pinned } else { pos_val + 1 };
            let toks_i32 = vec![last as i32];
            let pos_i32 = vec![pos_val as i32];
            self.dev.htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            // Size the prize before building a device-resident loop: this is the per-token HOST round
            // trip (2 htod + a full-device sync) that a captured graph cannot absorb, plus the dtoh
            // below. It matters more under TP than single-node because both ranks pay it and the skew
            // between them lands on the rendezvous critical path.
            let t_pre = t_tok.elapsed().as_nanos() as u64;
            let next = if want_graph && step == GRAPH_CAPTURE_AT && graph.is_none() {
                // capture_decode_graph runs one REAL decode step as its warm-up, so this iteration
                // produces its token normally; the capture itself executes nothing.
                graph = self.capture_decode_graph(&mut pool, &mut bufs, &mut state, max_seq_len, max_pc, 1);
                eprintln!("[tp{}] decode graph {}", self.tp_rank,
                          if graph.is_some() { "CAPTURED" } else { "capture FAILED — staying eager" });
                self.sync_stream();
                self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
                    .into_iter().take(1).map(|x| x as u32).collect::<Vec<u32>>()
            } else if let Some(g) = &graph {
                g.launch();
                self.sync_stream();
                t_gpu_end = t_tok.elapsed().as_nanos() as u64;
                self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
                    .into_iter().take(1).map(|x| x as u32).collect::<Vec<u32>>()
            } else {
                self.forward_decode(&mut pool, &mut bufs, &mut state, max_seq_len, max_pc, 1)
            };
            last = next[0];
            let t_all = t_tok.elapsed().as_nanos() as u64;
            host_pre_ns.push(t_pre);
            host_post_ns.push(t_all.saturating_sub(t_gpu_end));
            tok_ns.push(t_all);
            if last == eos { break; }
            out_tokens.push(last);
        }
        Self::report_token_latency(&tok_ns, self.tp_rank);
        if host_pre_ns.len() > 20 {
            let med = |v: &mut Vec<u64>| { v.sort_unstable(); v[v.len()/2] as f64 / 1e6 };
            let (mut a, mut b) = (host_pre_ns[16..].to_vec(), host_post_ns[16..].to_vec());
            let (pre, post) = (med(&mut a), med(&mut b));
            eprintln!("[tp{}] host round-trip per token: pre(2×htod+sync) {pre:.3} ms + post(dtoh) \
                       {post:.3} ms = {:.3} ms — the residue a device-resident loop would remove",
                      self.tp_rank, pre + post);
        }
        if self.tp_head_proof() {
            let (checked, bad) = self.verify_head_redzones(&state);
            if bad == 0 {
                eprintln!("[head-proof] rank {} — PASS: {checked} GDN state red zones pristine after {} tokens \
                           (no kernel wrote outside this rank's heads)", self.tp_rank, out_tokens.len());
            } else if std::env::var("GB10_TP_HEAD_PROOF_FAULT").is_ok() {
                eprintln!("[head-proof] rank {} — DETECTED (injected fault): {bad} of {checked} red zones \
                           written. Run continues so the token output can be compared: if it is unchanged, \
                           that is the point — this bug is invisible to the correctness gate.",
                          self.tp_rank);
            } else {
                panic!("[head-proof] rank {} — FAIL: {bad} of {checked} red zones were written", self.tp_rank);
            }
        }
        out_tokens
    }

    /// Per-token latency percentiles with the warmup discarded. The first tokens carry one-time costs
    /// (allocator first-touch, cuBLAS workspace, page faults on the mapped comm rings) that have nothing
    /// to do with steady-state decode, and folding them into an aggregate tok/s silently understates the
    /// rate — which is exactly how a run can look like a regression when the steady state is fine.
    fn report_token_latency(tok_ns: &[u64], rank: i32) {
        const WARMUP: usize = 16;
        if tok_ns.len() <= WARMUP + 4 { return; }
        let warm = &tok_ns[..WARMUP];
        let mut v: Vec<u64> = tok_ns[WARMUP..].to_vec();
        v.sort_unstable();
        let p = |q: f64| v[(((v.len() - 1) as f64) * q).round() as usize] as f64 / 1e6;
        let warm_ms: f64 = warm.iter().sum::<u64>() as f64 / 1e6;
        let steady: f64 = v.iter().sum::<u64>() as f64 / 1e6;
        eprintln!("[tp{rank}] per-token latency over {} tokens after {WARMUP} warmup: \
                   p50 {:.2} ms  p95 {:.2} ms  p99 {:.2} ms  max {:.2} ms  => steady-state {:.2} tok/s",
                  v.len(), p(0.50), p(0.95), p(0.99), p(1.0), 1000.0 * v.len() as f64 / steady);
        eprintln!("[tp{rank}] warmup: first {WARMUP} tokens took {warm_ms:.1} ms total \
                   (first token {:.1} ms)", warm[0] as f64 / 1e6);
    }

    /// Dump the barrier trace plus the per-layer-type cost split (needs cfg, hence a method here).
    pub fn tp_trace_dump(&self, label: &str) {
        crate::tp_bench::trace_dump(label);
        let is_gdn: Vec<bool> = self.cfg.layer_types.iter()
            .map(|lt| matches!(lt, crate::qwen::LayerType::LinearAttention)).collect();
        crate::tp_bench::trace_layer_split(label, &is_gdn, self.tp_shard_mixers());
    }

    pub fn dev_for_state(&self) -> &Arc<CudaDevice> { &self.dev }

    /// GEMM batch-invariance probe: measures whether `gemm(W^T, X)` at N=1 and N=2 produce
    /// bit-identical results for the first column. cuBLAS picks different algos (tiling/Split-K) for
    /// different N → different fp32 accumulation order → nonzero diff. This isolates the GEMM (no
    /// model) so we can test fixes (cuBLASLt algo pinning) against the exact shapes that matter.
    /// Uses random bf16 W [m,k] (OP_T) and X [k,2]; reports max|out_N1[i] - out_N2[i,0]|.
    pub fn probe_gemm(&self, m: usize, k: usize) {
        use cudarc::cublas::{sys::cublasOperation_t as OP, Gemm, GemmConfig};
        let w_host: Vec<half::bf16> = (0..m * k)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let x_host: Vec<half::bf16> = (0..k * 2)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let w = self.dev.htod_sync_copy(&w_host).unwrap();
        let x = self.dev.htod_sync_copy(&x_host).unwrap();
        self.dev.synchronize().unwrap();

        // N=1: out1 = W^T @ X[:,0]  -> [m]
        let mut out1 = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        let cfg1 = GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: m as i32, n: 1, k: k as i32,
            alpha: half::bf16::from_f32(1.0), lda: k as i32,
            ldb: k as i32, beta: half::bf16::from_f32(0.0), ldc: m as i32,
        };
        unsafe { self.blas.gemm(cfg1, &w, &x, &mut out1).expect("gemm n1"); }

        // N=2: out2 = W^T @ X[:,0:2] -> [m,2] column-major
        let mut out2 = unsafe { self.dev.alloc::<half::bf16>(m * 2) }.unwrap();
        let cfg2 = GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: m as i32, n: 2, k: k as i32,
            alpha: half::bf16::from_f32(1.0), lda: k as i32,
            ldb: k as i32, beta: half::bf16::from_f32(0.0), ldc: m as i32,
        };
        unsafe { self.blas.gemm(cfg2, &w, &x, &mut out2).expect("gemm n2"); }
        self.sync_stream();

        let o1: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out1).unwrap();
        let o2: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out2).unwrap();
        let mut mx = 0.0f32;
        let mut nz = 0usize;
        for i in 0..m {
            let d = (half::bf16::to_f32(o1[i]) - half::bf16::to_f32(o2[i])).abs();
            if d > 0.0 { nz += 1; }
            if d > mx { mx = d; }
        }
        println!("GEMM(cuBLAS) M={} K={}: N=1 vs N=2 col-0  max-diff={:.6}  ({}/{} elements differ)",
                 m, k, mx, nz, m);
    }

    /// True if this model has a loaded MTP (multi-token prediction) head.
    pub fn mtp_present(&self) -> bool { self.mtp.is_some() }

    /// cuBLASLt batch-invariance probe: tests whether cuBLASLt (default heuristic, AND with Split-K
    /// forced off) gives a batch-invariant GEMM (N=1 == N=2). If either zeroes the diff, it's the
    /// fix for the 9B/27B MTP cuBLAS divergence. Runs on the compute stream (no cross-stream race).
    pub fn probe_gemm_lt(&self, m: usize, k: usize) {
        use cudarc::cublaslt::{result as lt, sys};
        let handle = lt::create_handle().expect("cublasLtCreate");
        let ws_size: usize = 32 * 1024 * 1024;
        let workspace = unsafe { self.dev.alloc::<u8>(ws_size) }.expect("ws alloc");
        let w_host: Vec<half::bf16> = (0..m * k)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let x_host: Vec<half::bf16> = (0..k * 2)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let w = self.dev.htod_sync_copy(&w_host).unwrap();
        let x = self.dev.htod_sync_copy(&x_host).unwrap();
        let mut out1 = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        let mut out2 = unsafe { self.dev.alloc::<half::bf16>(m * 2) }.unwrap();
        self.dev.synchronize().unwrap();

        let bf = sys::cudaDataType_t::CUDA_R_16BF;
        let desc = lt::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType_t::CUDA_R_32F).unwrap();
        let ta: u32 = 1; let tb: u32 = 0;
        unsafe {
            lt::set_matmul_desc_attribute(desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &ta as *const _ as *const _, 4).unwrap();
            lt::set_matmul_desc_attribute(desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &tb as *const _ as *const _, 4).unwrap();
        }
        let pref = lt::create_matmul_pref().unwrap();
        unsafe {
            lt::set_matmul_pref_attribute(pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const _ as *const _, 8).unwrap();
        }
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        let ws_ptr = *workspace.device_ptr() as *mut core::ffi::c_void;
        let stream = self.stream.stream as sys::cudaStream_t;

        let mut run = |force_splitk: bool| -> Option<f32> {
            for &n in &[1usize, 2] {
                let a_layout = lt::create_matrix_layout(bf, k as u64, m as u64, k as i64).ok()?;
                let b_layout = lt::create_matrix_layout(bf, k as u64, n as u64, k as i64).ok()?;
                let c_layout = lt::create_matrix_layout(bf, m as u64, n as u64, m as i64).ok()?;
                let mut heur = lt::get_matmul_algo_heuristic(handle, desc,
                    a_layout, b_layout, c_layout, c_layout, pref).ok()?;
                if force_splitk {
                    let one: u32 = 1;
                    unsafe {
                        sys::cublasLtMatmulAlgoConfigSetAttribute(&mut heur.algo,
                            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                            &one as *const _ as *const _, 4);
                    }
                }
                let out = if n == 1 { &mut out1 } else { &mut out2 };
                let r = unsafe {
                    lt::matmul(handle, desc,
                        &alpha as *const _ as *const core::ffi::c_void,
                        &beta as *const _ as *const core::ffi::c_void,
                        *w.device_ptr() as *const core::ffi::c_void, a_layout,
                        *x.device_ptr() as *const core::ffi::c_void, b_layout,
                        *out.device_ptr() as *const core::ffi::c_void, c_layout,
                        *out.device_ptr() as *mut core::ffi::c_void, c_layout,
                        &heur.algo, ws_ptr, ws_size, stream)
                };
                if r.is_err() { return None; }
            }
            self.sync_stream();
            let o1: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out1).unwrap();
            let o2: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out2).unwrap();
            let mut mx = 0.0f32;
            for i in 0..m {
                let d = (half::bf16::to_f32(o1[i]) - half::bf16::to_f32(o2[i])).abs();
                if d > mx { mx = d; }
            }
            Some(mx)
        };

        match run(false) {
            Some(mx) => println!("GEMM(cuBLASLt default) M={} K={}: N=1 vs N=2 diff={:.6}{}", m, k, mx, if mx==0.0{" << BATCH-INVARIANT"}else{""}),
            None => println!("GEMM(cuBLASLt default) M={} K={}: FAILED", m, k),
        }
        match run(true) {
            Some(mx) => println!("GEMM(cuBLASLt splitK=OFF) M={} K={}: N=1 vs N=2 diff={:.6}{}", m, k, mx, if mx==0.0{" << BATCH-INVARIANT!"}else{""}),
            None => println!("GEMM(cuBLASLt splitK=OFF) M={} K={}: FAILED (algo invalid w/ splitk=1)", m, k),
        }
    }

    /// Custom batch-invariant GEMM probe: runs the `gemm_binv_b` kernel at N=1 and N=2, checks the
    /// column-0 is BIT-IDENTICAL (batch-invariant), and compares vs cuBLAS N=1 (correctness — should
    /// be within bf16 rounding). If the custom kernel is batch-invariant, it's the fix for 9B/27B.
    pub fn probe_gemm_binv(&self, m: usize, k: usize) {
        use cudarc::cublas::{sys::cublasOperation_t as OP, Gemm, GemmConfig};
        let w_host: Vec<half::bf16> = (0..m * k)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let x_host: Vec<half::bf16> = (0..k * 2)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let w = self.dev.htod_sync_copy(&w_host).unwrap();
        let x = self.dev.htod_sync_copy(&x_host).unwrap();
        self.dev.synchronize().unwrap();
        let t = 256u32;
        let smem = (16 * 256 * 4) as u32; // Nmax(16) * T(256) * f32

        // custom batch-invariant kernel, N=1 and N=2
        let binv1 = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        let binv2 = unsafe { self.dev.alloc::<half::bf16>(m * 2) }.unwrap();
        blaunch!(self, "gemm_binv_b", (m as u32,1,1), (t,1,1), smem,
            (d(&binv1), d(&w), d(&x), m as i32, k as i32, 1i32));
        blaunch!(self, "gemm_binv_b", (m as u32,1,1), (t,1,1), smem,
            (d(&binv2), d(&w), d(&x), m as i32, k as i32, 2i32));
        // cuBLAS reference, N=1
        let mut cublas1 = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        let cfg = GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: m as i32, n: 1, k: k as i32,
            alpha: half::bf16::from_f32(1.0), lda: k as i32, ldb: k as i32,
            beta: half::bf16::from_f32(0.0), ldc: m as i32,
        };
        unsafe { self.blas.gemm(cfg, &w, &x, &mut cublas1).expect("cublas ref"); }
        self.sync_stream();

        let b1: Vec<half::bf16> = self.dev.dtoh_sync_copy(&binv1).unwrap();
        let b2: Vec<half::bf16> = self.dev.dtoh_sync_copy(&binv2).unwrap();
        let c1: Vec<half::bf16> = self.dev.dtoh_sync_copy(&cublas1).unwrap();
        let mut inv_diff = 0.0f32;   // binv N=1 vs N=2 col0 (must be 0)
        let mut ref_diff = 0.0f32;   // binv N=1 vs cuBLAS N=1 (correctness, ~bf16 rounding)
        for i in 0..m {
            let di = (half::bf16::to_f32(b1[i]) - half::bf16::to_f32(b2[i])).abs();
            let dr = (half::bf16::to_f32(b1[i]) - half::bf16::to_f32(c1[i])).abs();
            if di > inv_diff { inv_diff = di; }
            if dr > ref_diff { ref_diff = dr; }
        }
        println!("GEMM(gemm_binv custom) M={} K={}: N=1 vs N=2 col-0 diff={:.6}{}   | vs cuBLAS N=1 diff={:.6}",
                 m, k, inv_diff, if inv_diff == 0.0 { " << BATCH-INVARIANT!" } else { "" }, ref_diff);
    }

    /// THE GATE: bitwise batch-invariance of the serving GEMM, on real weights, across the whole
    /// verify width. Column 0 of an N-wide verify must be BIT-IDENTICAL to a single-token decode —
    /// that is the entire reason greedy MTP is lossless rather than merely "usually the same".
    ///
    /// This drives `gemm_act`, the real dispatcher, so it tests whatever the engine would actually
    /// run. It also compares against the PREFILL path (dequantize-to-bf16 + cuBLAS), which reaches
    /// the same weights through the inverse permutation and a completely different kernel — so a
    /// mis-mapped mma fragment cannot hide: the permutation and its inverse would have to be wrong
    /// in exactly compensating ways, and the inverse is separately unit-tested against the forward.
    /// STREAM-style pure-read bandwidth. This is the roofline every other number is measured against,
    /// and we had been quoting two different values for it (248 GB/s vs 216 GB/s observed) — a 15%
    /// spread that decides whether the GEMMs have 10% left in them or 25%, and whether a competitor's
    /// claimed tok/s is even physically possible on this part. So measure it, don't quote it.
    /// §2.3 audit: hammer memory bandwidth CONTINUOUSLY for `seconds`, reporting achieved GB/s per
    /// ~2s window. If sustained bandwidth sags vs the cold peak, LPDDR5x is thermally derating under
    /// load — which would cap every roofline number (decode, TP2, Hy3). Prints one line per window.
    pub fn probe_bandwidth_sustained(&self, seconds: u64) {
        let gib: usize = 8;
        let n4 = gib * 1024 * 1024 * 1024 / 16;
        let buf = self.dev.alloc_zeros::<u8>(n4 * 16).expect("alloc bw buffer");
        let sink = self.dev.alloc_zeros::<f32>(1).unwrap();
        self.dev.synchronize().unwrap();
        let bytes = (n4 * 16) as f64;
        let run = || {
            blaunch!(self, "bw_read_b", (8192u32,1,1), (256,1,1), 0,
                (d(&sink), *buf.device_ptr() as u64, n4 as i64));
        };
        run(); self.sync_stream(); // warm
        println!("=== SUSTAINED bandwidth ({} s, 8 GiB reads back-to-back) — watch for thermal sag ===", seconds);
        println!("  {:>6}  {:>8}  {:>8}", "t(s)", "GB/s", "min-so-far");
        let start = std::time::Instant::now();
        let (mut peak, mut trough) = (0f64, f64::INFINITY);
        while start.elapsed().as_secs() < seconds {
            let w0 = std::time::Instant::now();
            let mut iters = 0u32;
            // run ~2s worth of back-to-back reads
            while w0.elapsed().as_secs_f64() < 2.0 { run(); iters += 1; }
            self.sync_stream();
            let s = w0.elapsed().as_secs_f64();
            let gbs = bytes * iters as f64 / s / 1e9;
            peak = peak.max(gbs); trough = trough.min(gbs);
            println!("  {:>6.0}  {:>8.1}  {:>8.1}", start.elapsed().as_secs_f64(), gbs, trough);
        }
        println!("  PEAK {:.1} GB/s  TROUGH {:.1} GB/s  SAG {:.1}%  ({})",
                 peak, trough, 100.0*(peak-trough)/peak,
                 if (peak-trough)/peak < 0.03 { "STABLE — no thermal derating" } else { "SAG — sustained < cold" });
    }

    pub fn probe_bandwidth(&self) {
        let gib: usize = 8;   // large enough that no cache can hold it
        let n4 = gib * 1024 * 1024 * 1024 / 16;            // uint4 elements
        let buf = self.dev.alloc_zeros::<u8>(n4 * 16).expect("alloc bw buffer");
        let sink = self.dev.alloc_zeros::<f32>(1).unwrap();
        self.dev.synchronize().unwrap();
        let bytes = (n4 * 16) as f64;

        println!("=== pure-read bandwidth (STREAM-style, {} GiB, 16-byte vectorized loads) ===", gib);
        let mut best = 0.0f64;
        for blocks in [1024u32, 2048, 4096, 8192] {
            let run = || {
                blaunch!(self, "bw_read_b", (blocks,1,1), (256,1,1), 0,
                    (d(&sink), *buf.device_ptr() as u64, n4 as i64));
                self.sync_stream();
            };
            run();                                          // warm
            let t0 = std::time::Instant::now();
            for _ in 0..5 { run(); }
            let s = t0.elapsed().as_secs_f64() / 5.0;
            let gbs = bytes / s / 1e9;
            if gbs > best { best = gbs; }
            println!("  {:5} blocks x 256 thr   {:7.2} ms   {:6.1} GB/s", blocks, s * 1000.0, gbs);
        }
        println!("  PEAK OBSERVED: {:.0} GB/s   (GB10 LPDDR5x theoretical ~273 GB/s => {:.0}% of peak)",
                 best, 100.0 * best / 273.0);
        if best < 245.0 {
            println!("  NOTE: an IDLE GB10 reads at ~255 GB/s. A lower figure means something else was");
            println!("        using memory bandwidth (a leftover server? nsys? a build?). Unified LPDDR5x");
            println!("        is shared with the CPU. Re-run on an idle machine before trusting it --");
            println!("        a contended roofline once read 234 and skewed every efficiency number.");
        }
    }

    /// TP=2 half-width GEMV probe. Runs the REAL decode GEMV kernel `gemm_mma_fp4_b` at N=1 on
    /// synthetic (correctly-SIZED) NVFP4 buffers for a linear of shape (outn=M, inn=K), times it, and
    /// reports achieved weight-bandwidth vs the ~245 GB/s sustained roofline. Contents are irrelevant
    /// (timing only): the kernel does identical memory traffic regardless of the bytes it reads.
    ///
    /// Why: TP=2 shards every 27B linear to HALF — column-parallel (gate/up/qkv → half M) or
    /// row-parallel (down/o → half K, i.e. half the reduction depth). The whole ~1.85x projection
    /// hinges on the per-node half-width GEMV still hitting ~80% of roofline. If the smaller shape
    /// drops efficiency (fewer blocks to fill 48 SMs at half-M; a shorter K reduction hiding less
    /// latency at half-K), the win erodes toward ~1.5x. This measures it with ZERO comm code.
    pub fn probe_tp_gemv(&self, m: usize, k: usize, label: &str) {
        assert!(m % 16 == 0 && k % 32 == 0, "shape ({m},{k}) needs M%16==0, K%32==0");
        let qweight = self.dev.alloc_zeros::<u8>(m * k / 2).unwrap();
        let scales  = self.dev.alloc_zeros::<u8>(m * k / 16).unwrap();
        let gs      = self.dev.alloc_zeros::<f32>(m / 16).unwrap();
        let x       = self.dev.alloc_zeros::<half::bf16>(k + 64).unwrap();
        let out     = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        self.dev.synchronize().unwrap();
        let bytes = (m * k / 2 + m * k / 16) as f64;   // fp4 weights + E4M3 per-tile scales
        let run = || {
            blaunch!(self, "gemm_mma_fp4_b", ((m / 16) as u32, 1, 1), (256, 1, 1), 0,
                (d(&out), *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                 d(&gs), d(&x), m as i32, k as i32, 1i32, 0u64));   // Cf = null: bf16 store
        };
        for _ in 0..5 { run(); }        // warm
        self.sync_stream();
        let iters = 400u32;
        let t0 = std::time::Instant::now();
        for _ in 0..iters { run(); }
        self.sync_stream();
        let s = t0.elapsed().as_secs_f64() / iters as f64;
        let gbs = bytes / s / 1e9;
        println!("  {:<24} M={:>6} K={:>6}   {:>7.1} GB/s   {:>5.1}% roof   ({:>6.1} us, {:>6.2} MB)",
                 label, m, k, gbs, 100.0 * gbs / 245.0, s * 1e6, bytes / 1e6);
    }

    pub fn probe_binv(&self) -> bool {
        let h = self.cfg.hidden_size;
        let l0 = &self.layers[0];
        // On MoE models probe the shared expert (a standard MLP); the stacked grouped-expert GEMM gets
        // its own batch-invariance check when that kernel lands.
        let (mlp_ref, im) = match &l0.mlp {
            Ffn::Dense(m) => (m, self.cfg.intermediate_size),
            Ffn::Moe(moe) => (&moe.shared, self.cfg.shared_expert_intermediate_size),
        };
        let mut cases: Vec<(&str, &W, usize, usize)> = vec![
            ("mlp.gate", &mlp_ref.gate, h, im),
            ("mlp.down", &mlp_ref.down, im, h),
        ];
        if let Some(la) = &l0.la {
            // The FUSED in_proj is the one that most needs this: it stacks four source tensors along M,
            // each with its own NVFP4 tensor scale, resolved by a per-tile lookup. A boundary error
            // there would corrupt one segment's magnitudes and nothing else would notice.
            match &la.in_proj {
                GdnIn::Fused(w) => cases.push(("gdn.in_proj FUSED", w, h, GdnIn::fused_m(&self.cfg))),
                GdnIn::Split { qkv, .. } =>
                    cases.push(("gdn.in_proj_qkv", qkv, h, self.cfg.key_dim() * 2 + self.cfg.value_dim())),
            }
            cases.push(("gdn.out_proj", &la.out_proj, self.cfg.value_dim(), h));
        }
        if let Some(fa) = &l0.fa {
            if let AttnIn::Fused(w) = &fa.qkv { cases.push(("attn.qkv FUSED", w, h, AttnIn::fused_m(&self.cfg))); }
        }
        if let Some(lh) = &self.lm_head { cases.push(("lm_head", lh, h, self.cfg.vocab_size)); }

        println!("=== bitwise batch-invariance: col 0 of an N-wide verify vs a N=1 decode ===");
        let mut all_ok = true;
        for (name, w, inn, outn) in cases {
            // One X, MAX_VERIFY+1 columns. Column 0 is shared by every call, so any difference in the
            // result is the kernel reacting to N — which is exactly what must not happen.
            let nref = MAX_VERIFY + 1;
            let x_host: Vec<half::bf16> = (0..inn * nref)
                .map(|_| half::bf16::from_f32(rand::random::<f32>() * 2.0 - 1.0)).collect();
            let x = self.dev.htod_sync_copy(&x_host).unwrap();
            self.dev.synchronize().unwrap();

            let mut col0: Option<Vec<half::bf16>> = None;
            let mut worst = 0u16;
            for n in 1..=MAX_VERIFY {
                let mut out = unsafe { self.dev.alloc::<half::bf16>(outn * n) }.unwrap();
                self.gemm_act(w, &x, &mut out, inn, outn, n);
                self.sync_stream();
                let got: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out).unwrap();
                match &col0 {
                    None => col0 = Some(got[..outn].to_vec()),
                    Some(c0) => for i in 0..outn {
                        // BITWISE. Not "close": a 1-ulp drift in column 0 re-argmaxes a token
                        // somewhere in a long generation, and MTP silently stops being lossless.
                        let d = c0[i].to_bits() ^ got[i].to_bits();
                        if d != 0 && d > worst { worst = d; }
                    },
                }
            }
            // Independent reference: batch > MAX_VERIFY takes the prefill path (dequant + cuBLAS).
            let mut r = unsafe { self.dev.alloc::<half::bf16>(outn * nref) }.unwrap();
            self.gemm_act(w, &x, &mut r, inn, outn, nref);
            self.sync_stream();
            let rv: Vec<half::bf16> = self.dev.dtoh_sync_copy(&r).unwrap();
            let c0 = col0.unwrap();
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..outn {
                let (a, b) = (c0[i].to_f32() as f64, rv[i].to_f32() as f64);
                num += (a - b) * (a - b);
                den += b * b;
            }
            let rel = (num / den.max(1e-30)).sqrt();
            let inv_ok = worst == 0;
            all_ok &= inv_ok;
            println!("  {:<16} [{}x{}]  N=1..{} col-0 {}   | vs dequant+cuBLAS rel-L2 {:.5}",
                     name, outn, inn, MAX_VERIFY,
                     if inv_ok { "BIT-IDENTICAL".to_string() }
                     else { format!("DIVERGED (worst xor 0x{:04x})", worst) }, rel);
        }
        println!("{}", if all_ok { "PASS — greedy MTP is lossless at every depth up to 16." }
                       else { "FAIL — batch-invariance is broken; greedy MTP is NOT lossless." });
        all_ok
    }

    /// Brute-force sweep of all cuBLAS GEMM algos for shape (M=outn, K=inn), measuring the N=1 vs
    /// N=2 column-0 divergence for each. The goal: find an algo whose result is batch-invariant
    /// (diff == 0) so the MTP verify (N=K) matches the decode (N=1) numerically. Prints every algo's
    /// status + diff; algos with diff 0 are the fix for the 9B/27B cuBLAS divergence.
    pub fn probe_gemm_sweep(&self, m: usize, k: usize) {
        use cudarc::cublas::sys;
        use std::mem::transmute;
        let w_host: Vec<half::bf16> = (0..m * k)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let x_host: Vec<half::bf16> = (0..k * 2)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let w = self.dev.htod_sync_copy(&w_host).unwrap();
        let x = self.dev.htod_sync_copy(&x_host).unwrap();
        let out1 = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        let out2 = unsafe { self.dev.alloc::<half::bf16>(m * 2) }.unwrap();
        self.dev.synchronize().unwrap();

        let handle = *self.blas.handle();
        let bf = sys::cudaDataType_t::CUDA_R_16BF;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // Algo values: -1=DEFAULT, 0..24=ALGO0..23 (non-tensor-op), 100..124=TENSOR_OP ALGO0..23.
        let mut algos: Vec<i32> = vec![-1];
        algos.extend(0..24);
        algos.extend(100..124);
        println!("=== cuBLAS algo sweep  M={} K={} (N=1 vs N=2 col-0 diff) ===", m, k);
        let mut best: Vec<(i32, f32)> = vec![];
        for av in algos {
            let algo: sys::cublasGemmAlgo_t = unsafe { transmute(av) };
            // N=1
            let r1 = unsafe {
                sys::cublasGemmEx(handle, sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N, m as i32, 1, k as i32,
                    (&alpha) as *const _ as *const _, *w.device_ptr() as *const _, bf, k as i32,
                    *x.device_ptr() as *const _, bf, k as i32, (&beta) as *const _ as *const _,
                    *out1.device_ptr() as *mut _, bf, m as i32,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, algo).result()
            };
            // N=2
            let r2 = unsafe {
                sys::cublasGemmEx(handle, sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N, m as i32, 2, k as i32,
                    (&alpha) as *const _ as *const _, *w.device_ptr() as *const _, bf, k as i32,
                    *x.device_ptr() as *const _, bf, k as i32, (&beta) as *const _ as *const _,
                    *out2.device_ptr() as *mut _, bf, m as i32,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, algo).result()
            };
            if r1.is_err() || r2.is_err() {
                continue; // algo not valid for this shape
            }
            self.sync_stream();
            let o1: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out1).unwrap();
            let o2: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out2).unwrap();
            let mut mx = 0.0f32;
            for i in 0..m {
                let d = (half::bf16::to_f32(o1[i]) - half::bf16::to_f32(o2[i])).abs();
                if d > mx { mx = d; }
            }
            let tag = if av < 0 { "DEFAULT".to_string() }
                else if av >= 100 { format!("ALGO{}_TENSOR_OP", av - 100) }
                else { format!("ALGO{}", av) };
            println!("  {:>20} ({}): diff={:.6}", tag, av, mx);
            if mx == 0.0 { best.push((av, mx)); }
        }
        if best.is_empty() {
            println!(">>> NO batch-invariant cuBLAS algo found for M={} K={} — need cuBLASLt split-K disable.", m, k);
        } else {
            println!(">>> BATCH-INVARIANT algos (diff=0): {:?}", best.iter().map(|&(a,_)| a).collect::<Vec<_>>());
        }
    }
    /// MTP forward pass: given the main model's hidden state h_t and the predicted token t+1,
    /// compute the next hidden state h' and predict the next token.
    /// Uses the MTP layer's own KV cache (mtp_k_cache, mtp_v_cache).
    /// Returns the predicted token id.
    pub fn mtp_forward(&self, pool: &mut Pool, hidden: &B, token_ptr: u64,
                       pos_dev: &cudarc::driver::CudaSlice<i32>,
                       mtp_kc_ptr: u64, mtp_vc_ptr: u64, max_pc: usize,
                       cos: &S, sin: &S, kv_stride: usize, batch: usize,
                       prefill_pos_start: Option<usize>) -> B {
        let slot_ids_ptr = *self.mtp_sids.device_ptr() as u64;   // always slot 0 — see `mtp_sids`
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let _nh = cfg.num_heads; let _nkv = cfg.num_kv_heads; let _hd = cfg.head_dim; let _rdim = cfg.rotary_dim;
        let mtp = self.mtp.as_ref().expect("MTP layer not loaded");

        // 1. RMSNorm hidden and embedding, then concat and FC
        let norm_h = pool.get_bf16(h * batch);
        blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&norm_h), d(hidden), d(&mtp.pre_fc_norm_hidden), h as i32, batch as i32, fbits(cfg.rms_eps)));

        // Gather embedding for the predicted token (token_ptr is a device-resident i32).
        let norm_e = pool.get_bf16(h * batch);
        self.embed_gather(*norm_e.device_ptr() as u64, token_ptr, h, batch);
        blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&norm_e), d(&norm_e), d(&mtp.pre_fc_norm_embedding), h as i32, batch as i32, fbits(cfg.rms_eps)));

        // Concat [norm_e, norm_h] -> concat [2h, batch] on-device (no host round-trip),
        // then FC: fc_out = fc @ concat, where fc is [h, 2h].
        let concat = pool.get_bf16(2 * h * batch);
        blaunch!(self, "concat_b", grid(2*h*batch), (256,1,1), 0,
            (d(&concat), d(&norm_e), d(&norm_h), h as i32, batch as i32));
        pool.release_bf16(norm_h, h * batch);
        pool.release_bf16(norm_e, h * batch);

        let mut fc_out = pool.get_bf16(h * batch);
        self.gemm_act(&mtp.fc, &concat, &mut fc_out, 2 * h, h, batch);
        pool.release_bf16(concat, 2 * h * batch);

        // 2. Standard PRE-NORM decoder layer — mirror the main full-attn layer EXACTLY. The MTP layer is
        // a real Qwen3.5 decoder layer (confirmed vs vLLM `qwen3_5_mtp.py`): `input_layernorm` is applied
        // BEFORE attention, and the RAW residual (fc_out + attn) — not its post-norm — feeds the MLP add.
        // (Previously this fed un-normed fc_out to attention AND aliased the fused-rmsnorm out/in, so the
        // MLP residual added onto the POST-normed value. Harmless-ish for the dense MLP but it miscalibrated
        // the MoE top-8 ROUTER — the 122B/35B MTP heads drafted at ~40% instead of the dense heads' ~73%.)
        let residual = fc_out;
        let normed = pool.get_bf16(h * batch);
        blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&normed), d(&residual), d(&mtp.input_ln), h as i32, batch as i32, fbits(cfg.rms_eps)));
        // The MTP head is REPLICATED under TP (tp_shard_weights never touches self.mtp), so it must
        // take the unsharded path regardless of tp_world — topology (a).
        let mixer = self.full_attn_batch(pool, &normed, &mtp.fa, pos_dev, max_pc, kv_stride,
                                         mtp_kc_ptr, mtp_vc_ptr, cos, sin, slot_ids_ptr, batch,
                                         prefill_pos_start, false);   // MTP head replicated: unsharded path
        // residual += mixer (raw kept in `residual`); normed = rmsnorm(residual, post_ln) for the MLP input.
        blaunch!(self, "fused_res_rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&normed), d(&residual), d(&mixer), d(&mtp.post_ln), h as i32, batch as i32, fbits(cfg.rms_eps)));
        let mlp_out = self.ffn_batch(pool, &normed, &mtp.mlp, batch, false);
        let tot = h * batch;
        blaunch!(self, "add_residual_b", grid(tot), (256,1,1), 0, (d(&residual), d(&residual), d(&mlp_out), tot as i32));
        pool.release_bf16(mixer, h*batch); pool.release_bf16(mlp_out, h*batch); pool.release_bf16(normed, h*batch);

        // 4. Final norm
        let out = pool.get_bf16(h * batch);
        blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&out), d(&residual), d(&mtp.final_norm), h as i32, batch as i32, fbits(cfg.rms_eps)));
        pool.release_bf16(residual, h*batch);

        out
    }

    pub fn kv_stride(&self) -> usize { self.cfg.max_position_embeddings }
}

/// Per-layer per-sequence inference state for batched decode.
/// KV caches [B, nkv, stride, hd]; conv state [B, conv_dim, k]; recurrent [B, nh, kd, vd].
/// One draft position from `bench_accept`: what the TARGET thought there, and whether the draft head
/// matched it. Only positions still on the correct prefix are recorded — see `bench_accept`.
#[derive(Clone, Copy, Debug)]
pub struct AcceptSample {
    /// 1-based position within the draft chain (1 = the first draft after the committed token).
    pub depth_idx: usize,
    /// The TARGET model's softmax probability for its own argmax. Low = the target is near-tied
    /// between tokens, i.e. the text is intrinsically unpredictable at this position.
    pub target_top1_p: f32,
    /// The target's top1-top2 LOGIT gap. A tiny margin means the argmax is a coin flip between two
    /// tokens, and no draft head can be expected to call it.
    pub target_margin: f32,
    /// Did the draft head propose the target's argmax? (target argmax ∈ head top-1)
    pub accepted: bool,
    /// Is the target's argmax within the head's top-2 candidates? A fork offering top-2 at this
    /// position would rescue it. (top-2 coverage ⊇ top-1 acceptance.)
    pub covered_top2: bool,
    /// ... within the head's top-3.
    pub covered_top3: bool,
}

/// A planted tree topology for the verify (Step-2.9 byte gates). All vectors are length-n (n columns),
/// except `path` which is n*MAX_VERIFY and `winsrc` which is n*conv_kernel.
pub struct TreeTopo {
    pub rope: Vec<i32>,      // logical position (tree depth) per column — governs pc, pos_start
    pub kv_pos: Vec<i32>,    // KV-cache write slot per column (distinct; DFS index + pos_start)
    pub parent: Vec<i32>,    // GDN scan parent (DFS parent; parent[0] = -1)
    pub path: Vec<u8>,       // rank->slot: path[b*MAX_VERIFY + d] = KV-slot offset of b's d-th ancestor
    pub winsrc: Vec<i32>,    // conv window sources per column (ancestor-path window)
    pub slot: Option<Vec<i32>>, // FOREST: per-column lane slot. None => uniform `slot` arg (a single
                                // tree/chain). Some(..) packs multiple lanes' chains into one verify.
    pub col_pos_start: Option<Vec<i32>>, // FOREST: per-column prefix boundary (lane committed length).
                                // None => attention uses pos[0] (single lane / tree, byte-identical).
}

pub struct BatchGpuState {
    pub k_cache: Vec<Option<B>>,
    pub v_cache: Vec<Option<B>>,
    pub conv_state: Vec<Option<S>>,
    pub s_state: Vec<Option<S>>,
}

/// Persistent device buffers for the optimized decode loop.
/// Allocated once, reused every step — eliminates all per-step allocations and large host↔device copies.
pub struct DecodeBuffers {
    pub pos_dev: cudarc::driver::CudaSlice<i32>,
    pub tokens_dev: cudarc::driver::CudaSlice<i32>,
    pub slot_ids_dev: cudarc::driver::CudaSlice<i32>,
    pub cos_dev: S,
    pub sin_dev: S,
    pub token_ids_dev: cudarc::driver::CudaSlice<i32>,
    pub penalty_tokens_dev: cudarc::driver::CudaSlice<i32>,
    pub penalty_counts_dev: cudarc::driver::CudaSlice<i16>,
    pub temps_dev: cudarc::driver::CudaSlice<f32>,
    pub topk_dev: cudarc::driver::CudaSlice<i32>,
    pub topp_dev: cudarc::driver::CudaSlice<f32>,
    pub seeds_dev: cudarc::driver::CudaSlice<u32>,
    pub rep_pen_dev: cudarc::driver::CudaSlice<f32>,
    pub presence_dev: cudarc::driver::CudaSlice<f32>,
    pub frequency_dev: cudarc::driver::CudaSlice<f32>,
}

pub const MAX_PEN_TOKENS: usize = 64;

/// Penalty device-buffer pointers for the MTP verify path (so greedy MTP lanes keep their
/// repetition/presence/frequency penalty → no repetition). All `n` verify positions share the
/// lane's committed-history penalty. Passed as `Option` to `verify_forward`.
pub struct VerifyPenalty {
    pub tokens_ptr: u64,    // [n * MAX_PEN_TOKENS] i32, -1 sentinel for unused slots
    pub counts_ptr: u64,    // [n * MAX_PEN_TOKENS] i16
    pub rep_pen_ptr: u64,   // [n] f32
    pub presence_ptr: u64,  // [n] f32
    pub freq_ptr: u64,      // [n] f32
}

/// Output of the stochastic MTP verify forward — per-position target probabilities, residual
/// resample tokens, and the bonus token (for the all-accepted case). Host reads these small scalars
/// back and runs the speculative rejection-sampling accept loop.
/// Goodness-of-fit of one empirical histogram against a reference distribution.
///
/// Total-variation distance is a *scale-dependent* statistic: its noise floor grows with the support
/// size and shrinks with the trial count, so a fixed "TVD < 0.01" bar is meaningless on its own — on
/// a 47-token nucleus at 100k trials even a perfect sampler sits around 0.009. A chi-square
/// goodness-of-fit is calibrated: `z` is comparable across temperatures, support sizes and trial
/// counts, so one threshold (|z| < 4) is honest everywhere.
#[derive(Debug, Clone, Copy)]
pub struct GofStat {
    pub tvd: f32,        // reported for intuition only — NOT the pass criterion
    pub chi2_over_df: f32,
    pub z: f32,          // Wilson-Hilferty normal score; |z| < 4 is the bar
    pub bins: usize,
    pub outside: u64,    // emissions outside the nucleus — must be 0 (they have probability 0)
}

/// Result of the stochastic-MTP distribution gate (`bench_mtp_sample`).
#[derive(Debug, Clone)]
pub struct MtpSampleStats {
    pub x_draft: u32,               // the greedy draft token whose acceptance is under test
    pub nucleus_size: usize,        // |support| of the target nucleus (after top-k ∩ top-p)
    pub p_draft_analytic: f32,      // p(x_draft) computed on the host from the same logits
    pub p_draft_device: f32,        // p(x_draft) as reported by spec_verify_b — must agree
    pub accept_rate: f32,           // empirical accept rate; converges to p(x_draft)
    pub accept_z: f32,              // binomial z of the accept rate vs p(x_draft)
    pub trials: usize,
    pub bonus_trials: usize,
    pub sampler: GofStat,           // diagnostic: plain sampler vs the analytic nucleus
    pub mtp: GofStat,               // diagnostic: stochastic MTP vs the analytic nucleus
    pub bonus: GofStat,             // diagnostic: the all-accepted bonus column
    pub mtp_vs_sampler: GofStat,    // THE GATE: two-sample test, MTP against the plain sampler
    pub bonus_vs_sampler: GofStat,  // THE GATE: the bonus column against the plain sampler
}

/// Wilson-Hilferty normal score for a chi-square statistic.
fn wh_z(chi2: f64, df: usize) -> f32 {
    if df == 0 { return 0.0; }
    let dff = df as f64;
    let r = chi2 / dff;
    ((r.cbrt() - (1.0 - 2.0 / (9.0 * dff))) / (2.0 / (9.0 * dff)).sqrt()) as f32
}

/// TWO-SAMPLE chi-square test of homogeneity: are `a` and `b` draws from the same distribution?
///
/// This — not the one-sample fit below — is the real gate. Comparing an empirical histogram against
/// a *host-computed* analytic nucleus requires replicating the kernels' float arithmetic exactly, and
/// the top-p cut is razor-sensitive deep in the tail (`cum >= top_p * sum` on bf16 logits with
/// `__expf`): the host and device disagree on the cutoff by a token or two, so BOTH the MTP path and
/// the plain sampler "fail" against the analytic reference, in lockstep. But the claim we actually
/// need is narrower and reference-free: *stochastic MTP emits from the same law as the plain
/// sampler*. Testing the two empirical histograms directly against each other says exactly that, and
/// is immune to the reference-model problem.
fn gof2(a: &[u64], na: u64, b: &[u64], nb: u64) -> GofStat {
    let (naf, nbf) = (na as f64, nb as f64);
    let tot = naf + nbf;
    // TVD between the two empirical distributions (diagnostic only).
    let tvd = 0.5 * (0..a.len())
        .map(|t| (a[t] as f64 / naf - b[t] as f64 / nbf).abs())
        .sum::<f64>();

    // Pool the two samples per token; merge low-expectation tokens into a tail bin.
    let mut bins: Vec<(f64, f64)> = Vec::new(); // (O_a, O_b) per usable bin
    let (mut tail_a, mut tail_b) = (0.0f64, 0.0f64);
    for t in 0..a.len() {
        let (oa, ob) = (a[t] as f64, b[t] as f64);
        let pooled = oa + ob;
        if pooled == 0.0 { continue; }
        // Expected counts under H0 (same law): n_i * pooled/total.
        if naf * pooled / tot >= 5.0 && nbf * pooled / tot >= 5.0 {
            bins.push((oa, ob));
        } else {
            tail_a += oa;
            tail_b += ob;
        }
    }
    let tail_pooled = tail_a + tail_b;
    if naf * tail_pooled / tot >= 5.0 && nbf * tail_pooled / tot >= 5.0 {
        bins.push((tail_a, tail_b));
    }

    let mut chi2 = 0.0f64;
    for &(oa, ob) in &bins {
        let pooled = oa + ob;
        let ea = naf * pooled / tot;
        let eb = nbf * pooled / tot;
        chi2 += (oa - ea) * (oa - ea) / ea + (ob - eb) * (ob - eb) / eb;
    }
    let df = bins.len().saturating_sub(1);
    GofStat {
        tvd: tvd as f32,
        chi2_over_df: if df == 0 { 0.0 } else { (chi2 / df as f64) as f32 },
        z: wh_z(chi2, df),
        bins: bins.len(),
        outside: 0,
    }
}

/// Chi-square goodness-of-fit of `hist` against `support`, with the Wilson-Hilferty normal score.
/// Bins whose expected count falls below 5 are merged into a tail bin so the chi-square stays valid.
/// Diagnostic only — see `gof2` for why this is not the gate.
fn gof(hist: &[u64], n: u64, support: &[(u32, f32)]) -> GofStat {
    let nf = n as f64;
    let inside: u64 = support.iter().map(|&(t, _)| hist[t as usize]).sum();
    let outside = n - inside;

    // TVD (diagnostic only).
    let mut pmap = vec![0.0f64; hist.len()];
    for &(t, p) in support { pmap[t as usize] = p as f64; }
    let tvd = 0.5 * (0..hist.len())
        .map(|t| (hist[t] as f64 / nf - pmap[t]).abs())
        .sum::<f64>();

    // Chi-square over merged bins (expected >= 5).
    let mut sorted: Vec<(u32, f32)> = support.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (mut chi2, mut bins) = (0.0f64, 0usize);
    let (mut tail_o, mut tail_e) = (0.0f64, 0.0f64);
    for &(t, p) in &sorted {
        let e = nf * p as f64;
        let o = hist[t as usize] as f64;
        if e >= 5.0 {
            chi2 += (o - e) * (o - e) / e;
            bins += 1;
        } else {
            tail_o += o;
            tail_e += e;
        }
    }
    if tail_e >= 5.0 {
        chi2 += (tail_o - tail_e) * (tail_o - tail_e) / tail_e;
        bins += 1;
    }

    let df = bins.saturating_sub(1);
    let chi2_over_df = if df == 0 { 0.0 } else { (chi2 / df as f64) as f32 };
    GofStat { tvd: tvd as f32, chi2_over_df, z: wh_z(chi2, df), bins, outside }
}

/// The target nucleus distribution, replicating `sample_b`/`spec_verify_b` exactly:
/// top-k by logit → scale by 1/temp → softmax (max-subtracted) → smallest prefix whose cumulative
/// mass reaches `top_p * sum` → renormalize over that prefix. Returns (token, prob) over the support.
fn nucleus_dist(lg: &[f32], temp: f32, top_k: usize, top_p: f32) -> Vec<(u32, f32)> {
    let k = top_k.clamp(1, 64).min(lg.len());
    let mut idx: Vec<u32> = (0..lg.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        lg[b as usize].partial_cmp(&lg[a as usize]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<u32> = idx.into_iter().take(k).collect();
    let inv = 1.0f32 / temp;
    let scaled: Vec<f32> = top.iter().map(|&t| lg[t as usize] * inv).collect();
    let mx = scaled[0];
    let probs: Vec<f32> = scaled.iter().map(|&v| (v - mx).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let mut cum = 0.0f32;
    let mut nc = k - 1;
    for j in 0..k {
        cum += probs[j];
        if cum >= top_p * sum { nc = j; break; }
    }
    (0..=nc).map(|j| (top[j], probs[j] / cum)).collect()
}

pub struct VerifySample {
    pub p_of_draft: Vec<f32>, // [depth-1] p_j(drafted_token) — target prob of drafted token
    pub resid_tok:  Vec<u32>, // [depth-1] token ~ normalize(max(0, p_j \ {draft}))
    pub bonus_tok:  u32,      // token ~ p_{depth-1} (all-accept bonus)
}

/// Penalty device-buffer pointers for the MTP DRAFT path (single column, for the MTP head before
/// sampling). Like VerifyPenalty but for batch=1. Host fills and uploads before mtp_draft_step_sample.
pub struct DraftPenalty {
    pub tokens_ptr: u64,    // [MAX_PEN_TOKENS] i32
    pub counts_ptr: u64,    // [MAX_PEN_TOKENS] i16
    pub rep_pen_ptr: u64,   // [1] f32
    pub presence_ptr: u64,  // [1] f32
    pub freq_ptr: u64,      // [1] f32
}

/// A captured CUDA graph that replays the full decode forward pass.
pub struct CudaGraph {
    exec: cudarc::driver::sys::CUgraphExec,
    stream: cudarc::driver::sys::CUstream,
}

// Safe: the graph handle and stream are only ever used from the single scheduler task.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// Replay the captured graph on its stream. Does NOT sync — caller must sync before reading results.
    pub fn launch(&self) {
        use cudarc::driver::sys;
        unsafe {
            let r = sys::cuGraphLaunch(self.exec, self.stream);
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch failed");
        }
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        use cudarc::driver::sys;
        unsafe { sys::cuGraphExecDestroy(self.exec); }
    }
}

fn state_stride(cfg: &Config) -> usize { cfg.max_position_embeddings }

/// Reusable device-buffer pool.
///
/// TWO PROPERTIES IT DID NOT HAVE, AND AN OOM THAT FOLLOWED.
///
/// It used to key buffers on their EXACT length and never free anything: `release` just pushed onto a
/// list, and `get(n)` only reused a buffer of length exactly `n`. Every prefill activation is sized by
/// the PROMPT LENGTH, so **every distinct prompt length ever seen leaked a full set of buffers,
/// permanently**. That is invisible in a benchmark (which reuses a handful of lengths) and fatal in real
/// use: a tool-eval suite sending hundreds of different prompt lengths exhausted GPU memory and the
/// allocation panicked — killing the scheduler task, after which every request returned zero tokens.
///
/// So:
///   * **Best-fit reuse.** Hand back the smallest cached buffer that is big enough (and not absurdly
///     oversized). Callers only ever touch the first `n` elements — the extra capacity is inert. This
///     is what collapses "one bucket per prompt length" into a handful of buckets.
///   * **A hard cap on cached bytes.** Past the cap, a released buffer is DROPPED (freed) rather than
///     hoarded. The pool is an optimisation; it must not be able to consume the device.
///
/// Reused buffers are NOT re-zeroed — only fresh allocations are (cudarc's `alloc_zeros` is
/// `cuMemAllocAsync` and does not actually zero). Every kernel that reads a pool buffer must therefore
/// write it first, which was already the contract.
pub struct Pool {
    dev: Arc<CudaDevice>,
    /// (buffer, BUCKET capacity) — never the caller's exact `n`. See `bucket`.
    free: Vec<(S, usize)>,
    free_bf16: Vec<(B, usize)>,
    cached_bytes: usize,
    cap_bytes: usize,
}

/// Past this many bytes of CACHED (idle) buffers, released buffers are freed rather than hoarded.
/// Generous next to a 6 GB model + KV on a 128 GB box, while making unbounded growth impossible.
const POOL_CAP_BYTES: usize = 3 << 30;


/// What a request for `n` elements NEEDS: the next power-of-two bucket (min 256).
///
/// Bucketing is what makes reuse possible at all. Buffers used to be keyed on the caller's EXACT
/// length, and every prefill activation is sized by the PROMPT LENGTH — so each distinct prompt length
/// got its own permanent bucket and nothing was ever reused. Costs <2x slack, buys bounded growth.
fn bucket_up(n: usize) -> usize { n.max(256).next_power_of_two() }

/// What a buffer of TRUE capacity `cap` can SERVE: the largest power-of-two bucket that fits inside it
/// (0 = too small to pool at all).
///
/// `release` must use this, never `bucket_up(n)` on the caller's `n`. Not every buffer handed to the
/// pool was allocated BY the pool: `embed_batch` allocates the prefill residual at EXACTLY `h*n`, and
/// `admit()` releases it. Filing that under `bucket_up(h*n)` claims it can serve up to 2x its real
/// size — so the pool later handed a 1033-column residual (4,231,168 elems) to a request that needed
/// 8,388,608 and wrote 16 MB into a 8 MB allocation. That surfaced as `CUDA_ERROR_INVALID_VALUE` on an
/// innocent memcpy, hours into a benchmark. The pool must trust the ALLOCATION, not the caller's claim.
fn bucket_down(cap: usize) -> usize {
    if cap < 256 { return 0; }
    1usize << (usize::BITS - 1 - cap.leading_zeros()) as usize
}

impl Pool {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        Pool { dev, free: vec![], free_bf16: vec![], cached_bytes: 0, cap_bytes: POOL_CAP_BYTES }
    }

    pub fn get(&mut self, n: usize) -> S {
        let b = bucket_up(n);
        if let Some(pos) = self.free.iter().position(|x| x.1 == b) {
            let (buf, _) = self.free.swap_remove(pos);
            self.cached_bytes -= buf.len() * 4;
            assert!(buf.len() >= n, "pool handed out {} f32 for a request of {n}", buf.len());
            return buf;
        }
        // cudarc's alloc_zeros is cuMemAllocAsync — it does NOT zero. Zero fresh allocations explicitly
        // so any pool buffer handed out is deterministic (a stale read would otherwise pick up GPU
        // garbage that varies run-to-run). Reused buffers are NOT re-zeroed: every kernel that reads a
        // pool buffer must write it first, which was already the contract.
        let mut buf = self.alloc_f32(b);
        self.dev.memset_zeros(&mut buf).unwrap();
        self.dev.synchronize().unwrap(); // ensure zeroing is visible to the compute stream
        buf
    }

    /// Allocate, and if the device is out of memory, DROP THE CACHE AND TRY AGAIN before giving up.
    ///
    /// An OOM here used to panic on a tokio worker, which killed the scheduler task — after which the
    /// server kept accepting requests and returning ZERO TOKENS forever. A zombie is worse than a crash.
    /// The pool is only an optimisation: under pressure it should surrender its idle buffers, not the
    /// process.
    fn alloc_f32(&mut self, n: usize) -> S {
        match self.dev.alloc_zeros::<f32>(n) {
            Ok(b) => b,
            Err(_) => {
                let freed = self.cached_bytes;
                self.free.clear();
                self.free_bf16.clear();
                self.cached_bytes = 0;
                let _ = self.dev.synchronize();
                eprintln!("[pool] OUT OF MEMORY allocating {} f32 — dropped {:.1} MB of cached buffers, retrying",
                          n, freed as f64 / 1e6);
                self.dev.alloc_zeros::<f32>(n).expect("alloc f32 after dropping the pool cache")
            }
        }
    }

    fn alloc_bf16(&mut self, n: usize) -> B {
        match self.dev.alloc_zeros::<half::bf16>(n) {
            Ok(b) => b,
            Err(_) => {
                let freed = self.cached_bytes;
                self.free.clear();
                self.free_bf16.clear();
                self.cached_bytes = 0;
                let _ = self.dev.synchronize();
                eprintln!("[pool] OUT OF MEMORY allocating {} bf16 — dropped {:.1} MB of cached buffers, retrying",
                          n, freed as f64 / 1e6);
                self.dev.alloc_zeros::<half::bf16>(n).expect("alloc bf16 after dropping the pool cache")
            }
        }
    }

    /// Return a buffer to the pool. ALWAYS caches — never frees here.
    ///
    /// Freeing at release is a use-after-free waiting to happen: `release` is called IMMEDIATELY after
    /// launching the kernels that read the buffer, long before they run. Dropping it there hands the
    /// memory back to CUDA while queued work still points at it, and the next allocation reuses the
    /// address — which surfaces as a baffling `CUDA_ERROR_INVALID_VALUE` on some later, innocent memcpy.
    /// (I did exactly this and it blew up on the first run.) Eviction happens in `trim`, which syncs.
    pub fn release(&mut self, s: S, _n: usize) {
        let b = bucket_down(s.len());   // TRUE capacity — see bucket_down
        if b == 0 { return; }
        self.cached_bytes += s.len() * 4;
        self.free.push((s, b));
    }

    pub fn get_bf16(&mut self, n: usize) -> B {
        let b = bucket_up(n);
        if let Some(pos) = self.free_bf16.iter().position(|x| x.1 == b) {
            let (buf, _) = self.free_bf16.swap_remove(pos);
            self.cached_bytes -= buf.len() * 2;
            assert!(buf.len() >= n, "pool handed out {} bf16 for a request of {n}", buf.len());
            return buf;
        }
        let mut buf = self.alloc_bf16(b);
        self.dev.memset_zeros(&mut buf).unwrap();
        self.dev.synchronize().unwrap();
        buf
    }

    pub fn release_bf16(&mut self, s: B, _n: usize) {
        let b = bucket_down(s.len());   // TRUE capacity — see bucket_down
        if b == 0 { return; }
        self.cached_bytes += s.len() * 2;
        self.free_bf16.push((s, b));
    }

    /// Bytes currently idle in the pool.
    pub fn cached_bytes(&self) -> usize { self.cached_bytes }

    /// Evict idle buffers down to the cap. Call ONLY where it is safe to free device memory — this
    /// SYNCHRONIZES first, because a cached buffer may still be the target of kernels that were launched
    /// before it was released and have not run yet.
    ///
    /// This is what keeps the pool bounded. Without it, every distinct prompt length permanently
    /// retained its own set of prefill activations (90–600 MB each on 9B) and a tool-eval suite with
    /// hundreds of prompt lengths exhausted the GPU.
    pub fn trim(&mut self) {
        if self.cached_bytes <= self.cap_bytes { return; }
        let before = self.cached_bytes;
        self.dev.synchronize().unwrap();      // no queued kernel may still reference what we free
        // Drop the biggest buffers first: they are the ones a rare long prompt left behind.
        self.free.sort_by_key(|x| x.0.len());
        self.free_bf16.sort_by_key(|x| x.0.len());
        while self.cached_bytes > self.cap_bytes {
            let f32_big = self.free.last().map(|x| x.0.len() * 4).unwrap_or(0);
            let bf_big = self.free_bf16.last().map(|x| x.0.len() * 2).unwrap_or(0);
            if f32_big == 0 && bf_big == 0 { break; }
            if f32_big >= bf_big {
                self.free.pop();                          // dropped here -> device memory freed
                self.cached_bytes -= f32_big;
            } else {
                self.free_bf16.pop();
                self.cached_bytes -= bf_big;
            }
        }
        eprintln!("[pool] trimmed {:.1} GB -> {:.1} GB of idle buffers",
                  before as f64 / 1e9, self.cached_bytes as f64 / 1e9);
    }
}
