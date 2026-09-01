//! S4F — the INTEGRATED DFlash2 draft round (K-DF2-2/3): trunk taps → fc/hidden-norm →
//! 5-layer block pass → borrowed LM-head logits → top-16 → selector chain → 7 draft tokens.
//!
//! # What this module owns (workdoc §3.2–§3.4)
//!
//! * **Incremental ctx injection** at M≤8 via `gemm_dsp_b` (DECISION-B row-chunkability —
//!   replaces the S3F probe's M=C `gemm_tiled_b` path): fc(taps)→hidden_norm per chunk, then
//!   per-layer k/v projections (k_norm + RoPE at TRUE absolute positions) written into a
//!   **2048-deep ring** KV (`[nkv, 2056, hd]`, ctx row j at `j % 2048`, block rows at
//!   `[2048, 2056)`) — window-bounded, no unbounded growth.
//! * **The block pass** (S3F's kernels, `gemm_dsp_b` everywhere) with the ring attention
//!   (`gqa_attn_band_ring_b`) — the visit order/softmax trees are IDENTICAL to S3F's linear
//!   kernel, so at C ≤ 2048 the two agree bit-for-bit (the probe's control).
//! * **The borrowed head**: the TARGET's NVFP4 lm_head (MMA-repacked at trunk load), run at
//!   N=7 directly on `h_final` columns 1..7 (`x = h_final + 5120` is a `[5120,7]` col-major
//!   tensor) via the trunk's own `gemm_mma_fp4_b` — NO new head GEMM.
//! * **The selector**: `top16_b` (radix, (logit desc, id asc), deterministic) + the fused
//!   `df2_sel_walk_b` greedy chain.
//!
//! # Numerics contract (the S3F mirror discipline, extended)
//!
//! The diff reference is `mirror.rs` extended through fc/head/selector (`round_mirror_*`):
//! bf16 staging at every device boundary + the device's exact reduction orders. The ORACLE
//! (`oracle.rs`) stays untouched. The probe's EXACT gates run on IDENTICAL inputs (device
//! logits → both top16s; device hp/cand/unary → both walks); the head is gated at rel-L2
//! (the mma's internal accumulation order cannot be mirrored — documented, not a defect).
//!
//! # Streams
//!
//! Own blocking stream (AGENTS §2.1). The trunk's capture D2D runs on the TRUNK's stream; the
//! round is launched after the trunk's forward completes in the probe (host-ordered). S5F's
//! graph capture will fold both into one stream-ordered chain.

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::collections::HashMap;
use std::sync::Arc;

use crate::dflash2::capture::Df2TapSink;
use crate::dflash2::gpu::{fork_blocking_stream, upload_bf16, upload_norm};
use crate::dflash2::band_smem;
use crate::dflash2::{BLOCK, CONV_GROUP, CONV_GROUPS, CONV_KERNEL, HEAD_DIM, HIDDEN, INTER, N_LAYERS,
                     NUM_HEADS, NUM_KV_HEADS, RMS_EPS, SELECTOR_RANK, TAP_CONCAT_DIM, VOCAB};

/// The ring depth (the sliding window; all 5 layers `sliding_attention`).
pub const RING: usize = crate::dflash2::SLIDING_WINDOW;      // 2048
/// Ring KV stride: ring rows + the 8 block rows above them.
pub const RING_STRIDE: usize = RING + BLOCK;                  // 2056

fn d<T>(s: &CudaSlice<T>) -> u64 {
    *s.device_ptr()
}
fn grid(n: usize) -> (u32, u32, u32) {
    (((n + 255) / 256) as u32, 1, 1)
}
fn fbits(x: f32) -> u64 {
    x.to_bits() as u64
}

/// A borrowed NVFP4 tensor's device pointers (MMA-repacked qweight/scales + the per-tile
/// f32 gs vector). The trunk owns the allocations; they outlive the round.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4Ptrs {
    pub qweight: u64,
    pub scales: u64,
    pub gs: u64,
}

/// P2 (round sharding): the round's TP all-reduce context — a Copy snapshot of the trunk's
/// transport surfaces (`GpuModel::df2_ar_ctx`), taken at round load (AFTER `attach_tp` on both
/// the head and the zero-config node). The round drives the SAME wire protocol as the trunk's
/// `tp_all_reduce_bf16` world>2 branch — K1 `tp_gate_copy_signal` + K2 `tp_wait_add{,_g}` tree
/// (fp32 accumulate, single bf16 rounding, canonical fold) — but launches on the ROUND's
/// blocking stream so the sites capture into the round's CUDA graph. Every field is resolved
/// identically on every rank (pure function of the shipped TpConfig + world), so the barrier
/// sequence stays SPMD-identical.
#[derive(Clone, Copy, Debug)]
pub struct Df2ArCtx {
    /// Device pointer of the transport ctx (epochs, gates, ring counters).
    pub ctx_dptr: u64,
    /// Device pointer of the persistent fp32 all-reduce scratch (GpuModel-owned, process
    /// lifetime — the round's graph captures reference it by address).
    pub f32_scratch: u64,
    /// TP world (engages only at world > 2).
    pub world: i32,
    /// This process's TP rank (the weight-band index for the shard-at-load slices).
    pub rank: i32,
    /// GPU-direct receive on ⇒ K2 = `tp_wait_add_g` (must mirror the trunk's resolution).
    pub k2_gpu_recv: bool,
    /// P3-1 one-shot fanout on (predicate mirrored from the trunk; the round's 80 KB payloads
    /// exceed the default decode payload, so the tree path is the norm).
    pub oneshot: bool,
    /// The transport's default payload width (the one-shot predicate's bound).
    pub payload_bytes: usize,
}

/// The trunk's borrowed lm_head/embed in either supported trunk dtype. The serving NVFP4
/// trunk borrows the MMA-repacked tensors (the `gemm_mma_fp4_b` / `embed_gather_fp4_tiled_b`
/// path); the plain-BF16 trunk class (`/mnt/models/Qwen3.8-27B-BF16` — the S5F2 L0 dtype
/// diagnostic) borrows the raw bf16 tensors (the `gemm_binv_b` / `embed_gather_b` path).
/// The round's embed/head launches dispatch on the variant.
#[derive(Clone, Copy, Debug)]
pub enum BorrowedW {
    Nvfp4(Nvfp4Ptrs),
    Bf16 { ptr: u64 },
}

/// Runtime projection weight. DFlash2 MR-GPTQ artifacts use the engine's ordinary
/// MMA-repacked NVFP4 representation; the optional BF16 copy is the same quantized
/// matrix dequantized once at load and is used only by prompt-prime's large-M GEMM.
enum RoundWeight {
    Bf16(CudaSlice<bf16>),
    Nvfp4 {
        qweight: CudaSlice<u8>,
        scales: CudaSlice<u8>,
        w4_qweight: Option<CudaSlice<u8>>,
        w4_scales: Option<CudaSlice<u8>>,
        gs: CudaSlice<f32>,
        input_gs: Option<f32>,
        prime_bf16: CudaSlice<bf16>,
        m: usize,
        k: usize,
        rotated: bool,
    },
}

struct RoundLayer {
    q_proj: RoundWeight, k_proj: RoundWeight, v_proj: RoundWeight,
    o_proj: RoundWeight, gate_proj: RoundWeight, up_proj: RoundWeight,
    down_proj: RoundWeight,
    q_norm: CudaSlice<f32>, k_norm: CudaSlice<f32>,
    input_ln: CudaSlice<f32>, post_ln: CudaSlice<f32>,
    attn_kp: CudaSlice<bf16>, attn_base: CudaSlice<bf16>,
    mlp_kp: CudaSlice<bf16>, mlp_base: CudaSlice<bf16>,
}

struct RoundGlobal {
    fc: CudaSlice<bf16>,
    hidden_norm: CudaSlice<f32>,
    norm: CudaSlice<f32>,
}

struct CalibScratch {
    n: usize,
    raw: CudaSlice<bf16>, th: CudaSlice<bf16>,
    pos: CudaSlice<i32>, rows: CudaSlice<i32>, slots: CudaSlice<i32>,
    cos: CudaSlice<f32>, sin: CudaSlice<f32>,
    kc: CudaSlice<bf16>, vc: CudaSlice<bf16>,
}

/// Optional block-scaled FP4 activation path for the 35 fixed-N=8 drafter projections. Context
/// prime/injection deliberately stays W4A16: the prime gate proves that mixing its large-M A4
/// path with fixed-N=8 A4 changes the KV ring enough to alter drafts. The standard packed weight
/// layout is retained only when requested; W4A16 remains the zero-overhead fallback.
struct Df2W4a4 {
    quant: CudaFunction,
    gemm: CudaFunction,
    gemm_n8: CudaFunction,
    bq: CudaSlice<u8>,
    sb: CudaSlice<u8>,
    k_max: usize,
}

impl Df2W4a4 {
    fn requested() -> bool {
        std::env::var("GB10_DF2_W4A4").map(|v| {
            let v = v.trim(); !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("off")
        }).unwrap_or(false)
    }

    fn build(dev: &Arc<CudaDevice>) -> Result<Self> {
        let module = "gpu_w4a4_df2";
        let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_w4a4.ptx")?);
        dev.load_ptx(ptx, module, &["w4a4_quant_pack_b", "w4a4_gemm_b", "w4a4_gemm_n8_b", "kernel_build_id"])?;
        crate::gpu::GpuModel::assert_kernel_build_id(dev, module)?;
        let quant = dev.get_func(module, "w4a4_quant_pack_b").context("df2 w4a4 quant kernel")?;
        let gemm = dev.get_func(module, "w4a4_gemm_b").context("df2 w4a4 wide gemm kernel")?;
        let gemm_n8 = dev.get_func(module, "w4a4_gemm_n8_b").context("df2 w4a4 n8 gemm kernel")?;
        let k_max = INTER;
        let bq = dev.htod_sync_copy(&vec![0xFFu8; BLOCK * (k_max / 2)])?;
        let sb = dev.htod_sync_copy(&vec![0xFFu8; BLOCK * (k_max / 4)])?;
        Ok(Self { quant, gemm, gemm_n8, bq, sb, k_max })
    }
}

fn load_round_nvfp4(dev: &Arc<CudaDevice>, rd: &crate::gptq::ShardReader,
                    stem: &str, m: usize, k: usize, rotated: bool, keep_w4: bool) -> Result<RoundWeight> {
    let (pm, qw) = rd.read_bytes(&format!("{stem}.weight_packed"))?;
    let (sm, sc) = rd.read_bytes(&format!("{stem}.weight_scale"))?;
    let (_, gb) = rd.read_bytes(&format!("{stem}.weight_global_scale"))?;
    anyhow::ensure!(pm.shape == vec![m, k / 2] && sm.shape == vec![m, k / 16],
                    "{stem}: malformed NVFP4 shape");
    anyhow::ensure!(gb.len() == 4, "{stem}: malformed global scale");
    let global_scale = f32::from_le_bytes(gb[..4].try_into().unwrap());
    anyhow::ensure!(global_scale.is_finite() && global_scale > 0.0,
                    "{stem}: invalid global scale {global_scale}");
    let input_gs = rd.read_bytes(&format!("{stem}.input_global_scale")).ok().and_then(|(_, b)| {
        (b.len() == 4).then(|| f32::from_le_bytes(b[..4].try_into().unwrap()))
    }).filter(|v| v.is_finite() && *v > 0.0);
    if keep_w4 {
        anyhow::ensure!(input_gs.is_some(),
                        "{stem}: GB10_DF2_W4A4 requires a valid input_global_scale");
    }
    let prime = crate::quant::dequantize_nvfp4(&crate::quant::Nvfp4Tensor {
        qweight: qw.clone(), scales: sc.clone(), global_scale, m, k,
    });
    let (wt, st) = crate::quant::repack_nvfp4_mma(&qw, &sc, m, k);
    let (w4_qweight, w4_scales) = if keep_w4 {
        (Some(dev.htod_sync_copy(&qw)?), Some(dev.htod_sync_copy(&sc)?))
    } else { (None, None) };
    Ok(RoundWeight::Nvfp4 {
        qweight: dev.htod_sync_copy(&wt)?, scales: dev.htod_sync_copy(&st)?,
        w4_qweight, w4_scales,
        gs: dev.htod_sync_copy(&vec![1.0 / global_scale; m / 16])?,
        input_gs, prime_bf16: dev.htod_sync_copy(&prime)?, m, k, rotated,
    })
}

macro_rules! klaunch {
    ($s:expr, $name:expr, $g:expr, $b:expr, $smem:expr, ($($a:expr),+ $(,)?)) => {
        unsafe {
            let (g0, g1, g2) = $g;
            let (b0, b1, b2) = $b;
            let name: &str = $name;
            $s.bk.get(name).cloned().unwrap_or_else(|| panic!("df2 kernel {name}")).launch_on_stream(
                &$s.stream,
                LaunchConfig { grid_dim: (g0, g1, g2), block_dim: (b0, b1, b2), shared_mem_bytes: $smem },
                ($($a),+)
            ).unwrap_or_else(|e| panic!("df2 launch {name}: {e:?}"));
        }
    };
}

/// The integrated round's outputs (host copies).
pub struct Df2RoundOut {
    /// The 7 draft tokens (the greedy chain path).
    pub tokens: Vec<u32>,
    /// Top-16 candidate ids per position, row-major `[7][16]` (the deterministic order).
    pub candidates: Vec<u32>,
    /// Candidate unary logits, `[7][16]` f32.
    pub unary: Vec<f32>,
    /// Final chain scores, `[7][16]` f32.
    pub scores: Vec<f32>,
    /// The post-final-norm block hidden `[8][5120]` (row-major per block row).
    pub h_final: Vec<f32>,
    /// S5F3 dump: the per-layer block hiddens `[5][8][5120]` f32 (post each layer's second
    /// residual add, pre final-norm) — the S1 layer-bisect surface vs the oracle's
    /// `backbone_forward` layer_hiddens. Empty unless the dump-full path fills it.
    pub layer_hiddens: Vec<f32>,
    /// The `[7][VOCAB]` logits (f32 from bf16) — probe-only, `dump_logits` opt-in.
    pub logits: Option<Vec<f32>>,
    /// The selector hidden projection `[7][256]` (bf16 values as f32) — probe-only.
    pub hp: Option<Vec<f32>>,
    /// Per-stage GPU timings (ms), recorded with CUDA events: [inject is timed by the caller]
    /// (block, head, top16, walk). Probe-only.
    pub stage_ms: Option<[f32; 4]>,
}

/// S5F2 — the SAMPLED draft round's outputs (L2's real-q path): the drawn draft chain, its
/// per-position selector probabilities, and the full candidate (token, q) table.
pub struct Df2SampleOut {
    /// The 7 DRAWN draft tokens.
    pub tokens: Vec<u32>,
    /// The drawn candidate's softmax weight per position, `[7]` f32 — the q in the real-q
    /// RS accept (u·q < p).
    pub q_rows: Vec<f32>,
    /// The candidate token ids per position, row-major `[7][16]` u32.
    pub cand_tok: Vec<u32>,
    /// The candidate softmax weights per position, row-major `[7][16]` f32 (sums to 1 per
    /// position) — the q the exact relu(p−q) verify residual needs at every candidate.
    pub cand_q: Vec<f32>,
}

/// The four distinct projection inputs retained after one layer forward. q/k/v share
/// `qkv`, gate/up share `gate_up`; this is exactly the Hessian sharing used by the driver.
pub struct Df2CalibInputs {
    pub qkv: Vec<bf16>,
    pub o: Vec<bf16>,
    pub gate_up: Vec<bf16>,
    pub down: Vec<bf16>,
}

/// A minimal CUDA-event stage timer (marks + elapsed, cudarc result::event).
pub struct EvTimer {
    events: Vec<cudarc::driver::sys::CUevent>,
}

impl EvTimer {
    pub fn new() -> Self {
        EvTimer { events: Vec::new() }
    }
    /// Record a split point on `stream`.
    pub fn mark(&mut self, stream: cudarc::driver::sys::CUstream) {
        use cudarc::driver::result::event;
        let ev = event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)
            .expect("cuEventCreate");
        unsafe { event::record(ev, stream).expect("cuEventRecord"); }
        self.events.push(ev);
    }
    /// Elapsed ms between marks i and i+1 (synchronizes the end event once).
    pub fn elapsed_ms(&self, i: usize) -> f32 {
        use cudarc::driver::sys;
        assert!(i + 1 < self.events.len(), "EvTimer mark {i}+1 missing");
        unsafe {
            let r = sys::cuEventSynchronize(self.events[i + 1]);
            assert!(r == sys::CUresult::CUDA_SUCCESS, "cuEventSynchronize: {r:?}");
        }
        unsafe { cudarc::driver::result::event::elapsed(self.events[i], self.events[i + 1])
            .expect("cuEventElapsedTime") }
    }
}

impl Drop for EvTimer {
    fn drop(&mut self) {
        for &e in &self.events {
            unsafe { let _ = cudarc::driver::result::event::destroy(e); }
        }
    }
}

/// The integrated DFlash2 draft round.
pub struct Df2Round {
    pub dev: Arc<CudaDevice>,
    stream: cudarc::driver::CudaStream,
    bk: HashMap<String, CudaFunction>,
    layers: Vec<RoundLayer>,
    glob: RoundGlobal,
    /// `candidate_selector.hidden_projection` [256, 5120] bf16.
    hp_w: CudaSlice<bf16>,
    /// `predecessor_codebook` / `successor_codebook` [248320, 256] bf16.
    pred_cb: CudaSlice<bf16>,
    succ_cb: CudaSlice<bf16>,
    /// Borrowed trunk tensors (trunk-owned): lm_head + embed, either NVFP4 (MMA-repacked
    /// serving trunk) or BF16 (the plain-BF16 class).
    pub head: Option<BorrowedW>,
    pub embed: Option<BorrowedW>,
    // Whether the borrowed trunk lm_head expects Hadamard16-rotated activations.
    head_hadamard16: bool,
    // Dedicated `[HIDDEN, 7]` scratch: h_final must stay unrotated for the selector.
    head_input: CudaSlice<bf16>,
    /// Shared Hadamard activation scratch. The largest DFlash2 MR input is INTER×BLOCK.
    mr_input: CudaSlice<bf16>,
    /// Large-M MR scratch and calibration workspace are reused across all 512 samples.
    prime_mr: CudaSlice<bf16>,
    calib: Option<CalibScratch>,
    w4a4: Option<Df2W4a4>,
    cos_table: CudaSlice<f32>,
    sin_table: CudaSlice<f32>,
    /// Ring KV: per layer `[nkv, RING_STRIDE, hd]` bf16.
    k_ring: Vec<CudaSlice<bf16>>,
    v_ring: Vec<CudaSlice<bf16>>,
    /// Committed ctx rows (absolute positions 0..nprev are live in the ring).
    nprev: usize,
    /// The fc input chunk buffer `[25600, 8]` col-major (the tap sink's twin or an upload).
    pub staging: CudaSlice<bf16>,
    /// S5F3: the attached TRUNK tap sink (None = no live capture; the round reads its OWN
    /// `staging` — the probe's upload_chunk path). The sink's staging is the LIVE capture
    /// buffer; `sync_staging_from_sink` copies it into `self.staging` before inject in the
    /// serving lanes (cudarc's CudaSlice::clone() is a DEEP copy — the pre-S5F3 attach code
    /// cloned the zeroed sink and inject_dev then consumed stale zeros: the draft-parity
    /// root cause).
    sink: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
    th_raw: CudaSlice<bf16>,
    th: CudaSlice<bf16>,
    kc: CudaSlice<bf16>,
    vc: CudaSlice<bf16>,
    cos_c: CudaSlice<f32>,
    sin_c: CudaSlice<f32>,
    pos_c: CudaSlice<i32>,       // [8] rope positions (absolute)
    wrow_c: CudaSlice<i32>,      // [8] ring write rows
    slot_c: CudaSlice<i32>,      // [8] zeros
    // block scratch (S3F shapes)
    blk: BlkScratch,
    pos_blk: CudaSlice<i32>,     // [8] block rope positions (absolute)
    wrow_blk: CudaSlice<i32>,    // [8] block ring write rows [RING..RING+8)
    wrow_ctl: CudaSlice<i32>,    // [8] block rows at [nprev..nprev+8) — the linear-control write
    toks_blk: CudaSlice<i32>,    // [8] anchor + 7×MASK
    // head + selector scratch
    logits: CudaSlice<bf16>,     // [VOCAB, 7] col-major
    out_vals: CudaSlice<f32>,    // [7*16]
    out_ids: CudaSlice<u32>,     // [7*16]
    hp_bf16: CudaSlice<bf16>,    // [256*7]
    hp_f32: CudaSlice<f32>,      // [256*7]
    unary_ctl: CudaSlice<f32>,   // [7*16] (sign-flip control)
    walk_tokens: CudaSlice<u32>, // [7]
    walk_scores: CudaSlice<f32>, // [7*16]
    // S5F2 sampled-walk outputs + inputs (L2 real-q path). Packed to keep the kernel launch
    // at the 12-arg cudarc cap: [0..7) = tokens / q_rows; [7 + 16p + k) = the candidate
    // (token, q) table rows.
    walk_out_tok: CudaSlice<u32>,  // [7 + 7*16] tokens | candidate ids
    walk_out_q: CudaSlice<f32>,    // [7 + 7*16] q_rows | candidate weights
    walk_seeds: CudaSlice<u32>,    // [7] per-position selector draw seeds
    // ring-vs-linear attention control surfaces (probe-only; layer 0)
    ctl_attn_ref: CudaSlice<bf16>, // ring kernel's layer-0 attention output (re-run copy)
    ctl_attn: CudaSlice<bf16>,     // S3F linear kernel's layer-0 attention on the dual cache
    max_c: usize,
    /// S5F: optional device i32 holding the CURRENT ntot (nprev + BLOCK) for the CUDA-graph
    /// replay path — the ring-attention kernel reads it when non-zero (see the kernel's ntot_dev).
    /// 0 = the eager packed-arg path (bit-identical to the S4F-validated behavior).
    pub ntot_dev: u64,
    /// S5F: optional device u32 holding the CURRENT anchor for the CUDA-graph replay path (the
    /// walk kernel reads it when non-zero). 0 = the eager arg path.
    pub anchor_dev: u64,
    /// S5F: the captured draft-round CUDA graph (embed -> 5 layers -> norm -> head -> top16 ->
    /// hp -> walk), replayed with per-step (anchor, nprev) written to device buffers.
    pub round_graph: Option<crate::gpu::CudaGraph>,
    /// E0: captured ring-injection graphs, one per inject width m in 1..=BLOCK. `inject_dev`'s
    /// launch sequence is width-CONTINGENT (several grids are functions of m), so each m gets its
    /// own graph — the eager call at a fresh m doubles as the pool/warmup pass, then the capture
    /// records the identical sequence for that m. Every buffer the sequence touches is a
    /// persistent `Df2Round` field (fixed addresses — the same discipline as `round_graph`), and
    /// the region contains NO all-reduce (kv projections are rank-local by construction), so
    /// capture/replay is SPMD-trivial. `GB10_NO_DF2_INJECT_GRAPH=1` keeps the eager path.
    pub inject_graphs: std::collections::HashMap<usize, crate::gpu::CudaGraph>,
    /// The per-replay device ints the graph reads (kept alive for the graph's lifetime).
    ntot_buf: Option<CudaSlice<i32>>,
    anchor_buf: Option<CudaSlice<u32>>,
    /// P2: the TP all-reduce context (Some ⇒ the round is SHARDED: per-rank weight slices,
    /// per-head ring KV, and 2 all-reduce sites per layer inside `layer_forward`). None = the
    /// pre-P2 replicated round, byte-identical to the S9F/S10R behavior.
    ar: Option<Df2ArCtx>,
    /// P2: the rank-local GEMM widths (the full constants when unsharded).
    nq_l: usize,      // local q heads  (NUM_HEADS / world)
    nkv_l: usize,     // local kv heads (NUM_KV_HEADS / world)
    ni_l: usize,      // local MLP intermediate (INTER / world)
}

struct BlkScratch {
    h: CudaSlice<bf16>,
    normed: CudaSlice<bf16>,
    x_conv: CudaSlice<bf16>,
    dyn_attn: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    attn_out: CudaSlice<bf16>,
    fin: CudaSlice<bf16>,
    normed2: CudaSlice<bf16>,
    x_conv2: CudaSlice<bf16>,
    dyn_mlp: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    mlp_out: CudaSlice<bf16>,
    fin2: CudaSlice<bf16>,
    h_final: CudaSlice<bf16>,
    cos8: CudaSlice<f32>,
    sin8: CudaSlice<f32>,
    slot_ids: CudaSlice<i32>,
}

// The round (its cudarc stream + device handle) is only ever touched from the single scheduler
// task — the same usage contract as GpuModel's `unsafe impl Send` (src/gpu.rs:960). The scheduler
// moves across a tokio spawn boundary; nothing else touches the round.
unsafe impl Send for Df2Round {}

impl Df2Round {
    /// Load the DFlash2 artifact + upload (per-tensor bf16, norms f32 w−1 — the S3F layout
    /// freeze) + allocate the ring KV + scratch. `head`/`embed` are the TRUNK's borrowed
    /// tensors in either supported dtype (`GpuModel::df2_borrow`). The pre-P2 replicated load
    /// (single-box probes and every non-sharded path).
    pub fn load(dir: &str, head: Option<BorrowedW>, embed: Option<BorrowedW>, max_c: usize)
        -> Result<Self> {
        Self::load_tp(dir, head, embed, max_c, None)
    }

    /// S9F+ (2026-08-29): load with an explicit artifact sha256 pin override.
    /// `None` = the published REAL_SHA256; `Some("off")` = no sha check (inventory/shape/
    /// dtype guard still runs); `Some(hex)` = pin to that hash. Used by the serve path so a
    /// retrained selector artifact (new sha) loads without weakening the default.
    pub fn load_pinned(dir: &str, head: Option<BorrowedW>, embed: Option<BorrowedW>, max_c: usize,
                       sha_pin: Option<&str>) -> Result<Self> {
        Self::load_tp_pinned(dir, head, embed, max_c, None, sha_pin)
    }

    /// P2 — the TP load: `load` plus an all-reduce context (`GpuModel::df2_ar_ctx`, Some only
    /// at world > 2 with `--df2-round-shard on`). When `ar` is Some the BIG per-layer weights
    /// are sliced at load (the proven trunk shard-at-load pattern) and the round carries two
    /// all-reduce sites per layer:
    ///
    /// * q/k/v_proj — HEAD-split rows (q: NUM_HEADS/world heads; k/v: NUM_KV_HEADS/world kv
    ///   heads; GQA's 4:1 ratio is preserved locally). Full-K on the replicated x_conv.
    /// * gate/up_proj — row bands (INTER/world rows), full-K on the replicated normed2.
    /// * o_proj/down_proj — K-split column bands: the rank computes a [5120×B] PARTIAL of the
    ///   true output over its input slice; the all-reduce lands the full sum.
    /// * Everything else (fc, norms, conv kp/base, head/embed borrow, selector) stays
    ///   REPLICATED: the conv coefficients are channel-group-local and the projections after
    ///   them need the full hidden — splitting them would force K-split qkv and extra
    ///   all-reduces for ~0.1 ms/layer of reads; the head must see full-dim inputs (the S9F
    ///   garbage-drafts hazard dies by construction).
    ///
    /// The ring KV shrinks to the rank's kv-heads (`[NUM_KV_HEADS/world, RING_STRIDE, hd]` —
    /// compile-time 2056×128 rows, ctx-free by construction post-S10R). Scratch buffers keep
    /// their FULL sizes (the local slices are prefixes — stale tails are never read by any
    /// consumer, same discipline as `upload_chunk`'s m-padding).
    pub fn load_tp(dir: &str, head: Option<BorrowedW>, embed: Option<BorrowedW>, max_c: usize,
                   ar: Option<Df2ArCtx>) -> Result<Self> {
        Self::load_tp_pinned(dir, head, embed, max_c, ar, None)
    }

    /// `load_tp` with an explicit artifact sha256 pin override (see `load_pinned`).
    pub fn load_tp_pinned(dir: &str, head: Option<BorrowedW>, embed: Option<BorrowedW>, max_c: usize,
                          ar: Option<Df2ArCtx>, sha_pin: Option<&str>) -> Result<Self> {
        let pin: Option<&str> = match sha_pin {
            Some("off") => None,
            Some(hex) => Some(hex),
            None => Some(crate::dflash2::REAL_SHA256),
        };
        let mixed = std::path::Path::new(dir).join("model.safetensors.index.json").exists();
        let art = if mixed { crate::dflash2::load::load_runtime(dir)? }
                  else { crate::dflash2::load::load(dir, pin)? };
        let w = &art.weights;
        let cfg = crate::dflash2::oracle::Dflash2Config::default();
        let max_pos = max_c + BLOCK + 1;

        // Resolve the shard geometry ONCE (a pure function of constants + world ⇒ identical on
        // every rank; a non-divisible world refuses the load loudly — the head's round outcome
        // then ships false via CalibTable and every rank serves MTP in lockstep).
        let (sharded, shard_rank, world) = match &ar {
            Some(a) => {
                let world = a.world as usize;
                anyhow::ensure!(world > 1, "df2 round shard: world {} <= 1", world);
                anyhow::ensure!(NUM_HEADS % world == 0 && NUM_KV_HEADS % world == 0
                                && INTER % world == 0,
                    "df2 round shard: world {world} does not divide heads {NUM_HEADS}/{NUM_KV_HEADS} or inter {INTER}");
                anyhow::ensure!((0..world).contains(&(a.rank as usize)),
                    "df2 round shard: rank {} out of range for world {world}", a.rank);
                (true, a.rank as usize, world)
            }
            None => (false, 0, 1),
        };
        let (nq_l, nkv_l, ni_l) = if sharded {
            (NUM_HEADS / world, NUM_KV_HEADS / world, INTER / world)
        } else {
            (NUM_HEADS, NUM_KV_HEADS, INTER)
        };

        let dev = CudaDevice::new(0).context("CudaDevice")?;
        let stream = fork_blocking_stream(&dev);

        let bptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx")?);
        let mut bfnames = ["rmsnorm_b", "rmsnorm_perhead_b", "rope_b", "gather_rope_b",
            "write_kv_b", "add_residual_b", "silu_mul_b", "kernel_build_id",
            "gemm_mma_fp4_b", "embed_gather_fp4_tiled_b", "dequant_fp4_tiled_b",
            "embed_gather_b", "gemm_binv_b", "gptq_rotate_act_b"].to_vec();
        // P2: the all-reduce handshake kernels live in gpu_batch.cu (the trunk AR path's module);
        // f32tobf16 lives in gpu_kernels.cu — it joins kfnames below (a name from the wrong
        // module is a load-time CUDA_ERROR_NOT_FOUND, caught live at the first SHARD=on boot).
        if ar.is_some() {
            bfnames.extend(["tp_gate_copy_signal", "tp_wait_add", "tp_wait_add_g",
                "tp_wait_add_4way"]);
        }
        // bf16tof32 lives in gpu_kernels.cu (next to the df2 kernels).
        dev.load_ptx(bptx, "gpu_batch", &bfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_batch")?;
        let kptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_kernels.ptx")?);
        // S5F adds gemm_tiled_b (the S3F large-M ctx GEMM) for the prompt-prime path
        // (prime_window runs fc + per-layer k/v at M = the whole prefill window).
        let kfnames = ["gemm_dsp_b_m8_r4", "gemm_tiled_b", "gqa_attn_band_b", "gqa_attn_band_ring_b",
            "top16_b", "df2_sel_walk_b", "df2_sel_walk_sample_b", "bf16tof32", "conv2_dynamic_b",
            "f32tobf16", "kernel_build_id"];
        dev.load_ptx(kptx, "gpu_kernels", &kfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_kernels")?;
        let mut bk = HashMap::new();
        for n in bfnames.iter().chain(kfnames.iter()) {
            let module = if kfnames.contains(n) { "gpu_kernels" } else { "gpu_batch" };
            bk.insert(n.to_string(), dev.get_func(module, n).with_context(|| format!("kernel {n} not in ptx"))?);
        }
        let w4a4 = if mixed && Df2W4a4::requested() {
            anyhow::ensure!(!sharded, "GB10_DF2_W4A4 is not implemented for sharded rounds");
            Some(Df2W4a4::build(&dev)?)
        } else {
            if Df2W4a4::requested() && !mixed {
                eprintln!("[df2] GB10_DF2_W4A4 requested for a BF16 drafter — keeping BF16 projections");
            }
            None
        };

        let mut layers = Vec::with_capacity(N_LAYERS);
        for l in &w.layers {
            // P2 shard-at-load (host-side slices BEFORE the bf16 upload — the trunk's proven
            // pattern). Row-major [out, in]: head/row bands are contiguous row slices; the
            // K-split (o/down) takes a per-row column band. Full-K everywhere a band is an
            // OUTPUT band; K-sliced bands produce PARTIAL outputs the all-reduce lands.
            if sharded {
                let rows = |v: &[f32], cols: usize, r0: usize, r1: usize| v[r0 * cols..r1 * cols].to_vec();
                let cols = |v: &[f32], rows_n: usize, cols_n: usize, c0: usize, c1: usize| -> Vec<f32> {
                    let mut o = Vec::with_capacity(rows_n * (c1 - c0));
                    for r in 0..rows_n { o.extend_from_slice(&v[r * cols_n + c0..r * cols_n + c1]); }
                    o
                };
                let nq_rows = nq_l * HEAD_DIM;    // 1024 @ world=4
                let nkv_rows = nkv_l * HEAD_DIM;  // 256
                layers.push(RoundLayer {
                    q_proj: RoundWeight::Bf16(upload_bf16(&dev, &rows(&l.q_proj, HIDDEN, shard_rank * nq_rows, (shard_rank + 1) * nq_rows))),
                    k_proj: RoundWeight::Bf16(upload_bf16(&dev, &rows(&l.k_proj, HIDDEN, shard_rank * nkv_rows, (shard_rank + 1) * nkv_rows))),
                    v_proj: RoundWeight::Bf16(upload_bf16(&dev, &rows(&l.v_proj, HIDDEN, shard_rank * nkv_rows, (shard_rank + 1) * nkv_rows))),
                    o_proj: RoundWeight::Bf16(upload_bf16(&dev, &cols(&l.o_proj, HIDDEN, NUM_HEADS * HEAD_DIM,
                                                    shard_rank * nq_rows, (shard_rank + 1) * nq_rows))),
                    gate_proj: RoundWeight::Bf16(upload_bf16(&dev, &rows(&l.gate_proj, HIDDEN, shard_rank * ni_l, (shard_rank + 1) * ni_l))),
                    up_proj: RoundWeight::Bf16(upload_bf16(&dev, &rows(&l.up_proj, HIDDEN, shard_rank * ni_l, (shard_rank + 1) * ni_l))),
                    down_proj: RoundWeight::Bf16(upload_bf16(&dev, &cols(&l.down_proj, HIDDEN, INTER,
                                                       shard_rank * ni_l, (shard_rank + 1) * ni_l))),
                    q_norm: upload_norm(&dev, &l.q_norm),
                    k_norm: upload_norm(&dev, &l.k_norm),
                    input_ln: upload_norm(&dev, &l.input_ln),
                    post_ln: upload_norm(&dev, &l.post_ln),
                    attn_kp: upload_bf16(&dev, &l.attention_conv.kernel_projection),
                    attn_base: upload_bf16(&dev, &l.attention_conv.base_kernel),
                    mlp_kp: upload_bf16(&dev, &l.mlp_conv.kernel_projection),
                    mlp_base: upload_bf16(&dev, &l.mlp_conv.base_kernel),
                });
            } else {
                layers.push(RoundLayer {
                    q_proj: RoundWeight::Bf16(upload_bf16(&dev, &l.q_proj)),
                    k_proj: RoundWeight::Bf16(upload_bf16(&dev, &l.k_proj)),
                    v_proj: RoundWeight::Bf16(upload_bf16(&dev, &l.v_proj)),
                    o_proj: RoundWeight::Bf16(upload_bf16(&dev, &l.o_proj)),
                    gate_proj: RoundWeight::Bf16(upload_bf16(&dev, &l.gate_proj)),
                    up_proj: RoundWeight::Bf16(upload_bf16(&dev, &l.up_proj)),
                    down_proj: RoundWeight::Bf16(upload_bf16(&dev, &l.down_proj)),
                    q_norm: upload_norm(&dev, &l.q_norm),
                    k_norm: upload_norm(&dev, &l.k_norm),
                    input_ln: upload_norm(&dev, &l.input_ln),
                    post_ln: upload_norm(&dev, &l.post_ln),
                    attn_kp: upload_bf16(&dev, &l.attention_conv.kernel_projection),
                    attn_base: upload_bf16(&dev, &l.attention_conv.base_kernel),
                    mlp_kp: upload_bf16(&dev, &l.mlp_conv.kernel_projection),
                    mlp_base: upload_bf16(&dev, &l.mlp_conv.base_kernel),
                });
            }
        }
        let glob = RoundGlobal {
            fc: upload_bf16(&dev, &w.fc),
            hidden_norm: upload_norm(&dev, &w.hidden_norm),
            norm: upload_norm(&dev, &w.norm),
        };
        let up_b = |data: &[f32]| -> CudaSlice<bf16> {
            let b: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
            dev.htod_sync_copy(&b).expect("upload bf16")
        };
        let hp_w = up_b(&w.hidden_projection);
        let pred_cb = up_b(&w.predecessor_codebook);
        let succ_cb = up_b(&w.successor_codebook);
        anyhow::ensure!(head.is_some(), "Df2Round needs the trunk's lm_head (df2_borrow)");
        anyhow::ensure!(embed.is_some(), "Df2Round needs the trunk's embed (df2_borrow)");

        let inv = crate::dflash2::mirror::inv_freq(&cfg);
        let (cos_t, sin_t) = crate::dflash2::mirror::rope_tables(&cfg, &inv, max_pos);
        let cos_table = dev.htod_sync_copy(&cos_t)?;
        let sin_table = dev.htod_sync_copy(&sin_t)?;

        let dev2 = dev.clone();
        let alloc_z = move |n: usize| dev2.alloc_zeros::<bf16>(n).expect("alloc bf16");
        let dev3 = dev.clone();
        let alloc_zf = move |n: usize| dev3.alloc_zeros::<f32>(n).expect("alloc f32");
        let dev4 = dev.clone();
        let alloc_zi = move |n: usize| dev4.alloc_zeros::<i32>(n).expect("alloc i32");
        let dev5 = dev.clone();
        let dev6 = dev.clone();
        let mut k_ring = Vec::with_capacity(N_LAYERS);
        let mut v_ring = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            // P2: the ring holds ONLY this rank's kv-heads post-shard (heads are independent —
            // no cross-rank attention traffic; the ring rows stay compile-time 2056×128,
            // ctx-free by construction post-S10R).
            k_ring.push(alloc_z(nkv_l * RING_STRIDE * HEAD_DIM));
            v_ring.push(alloc_z(nkv_l * RING_STRIDE * HEAD_DIM));
        }
        // Explicit zeroing of the persistent buffers a first round might read partially
        // (AGENTS §2.2: alloc_zeros does NOT zero).
        let zero8: Vec<bf16> = vec![bf16::default(); TAP_CONCAT_DIM * BLOCK];
        let staging = dev.htod_sync_copy(&zero8)?;
        let blk = BlkScratch {
            h: alloc_z(HIDDEN * BLOCK),
            normed: alloc_z(HIDDEN * BLOCK),
            x_conv: alloc_z(HIDDEN * BLOCK),
            dyn_attn: alloc_z(2 * CONV_KERNEL * CONV_GROUPS * BLOCK),
            q: alloc_z(NUM_HEADS * HEAD_DIM * BLOCK),
            k: alloc_z(NUM_KV_HEADS * HEAD_DIM * BLOCK),
            v: alloc_z(NUM_KV_HEADS * HEAD_DIM * BLOCK),
            attn: alloc_z(NUM_HEADS * HEAD_DIM * BLOCK),
            attn_out: alloc_z(HIDDEN * BLOCK),
            fin: alloc_z(HIDDEN * BLOCK),
            normed2: alloc_z(HIDDEN * BLOCK),
            x_conv2: alloc_z(HIDDEN * BLOCK),
            dyn_mlp: alloc_z(2 * CONV_KERNEL * CONV_GROUPS * BLOCK),
            gate: alloc_z(INTER * BLOCK),
            up: alloc_z(INTER * BLOCK),
            mlp_out: alloc_z(HIDDEN * BLOCK),
            fin2: alloc_z(HIDDEN * BLOCK),
            // gemm_dsp (m8) reads X rows 1..=8 at +HIDDEN for the selector projection: the
            // 9th row is a permanently-zero guard row (row 8 never written, only read by the
            // hp GEMM's 8th column, which no consumer reads).
            h_final: dev6.htod_sync_copy(&vec![bf16::default(); HIDDEN * (BLOCK + 1)])
                .expect("h_final zeroed 9 rows"),
            cos8: alloc_zf(BLOCK * HEAD_DIM),
            sin8: alloc_zf(BLOCK * HEAD_DIM),
            slot_ids: dev.htod_sync_copy(&vec![0i32; BLOCK])?,
        };
        let pos_blk: Vec<i32> = (0..BLOCK).map(|b| b as i32).collect();
        let pos_blk = dev.htod_sync_copy(&pos_blk)?;
        let wrow_blk: Vec<i32> = (0..BLOCK).map(|b| (RING + b) as i32).collect();
        let wrow_blk = dev.htod_sync_copy(&wrow_blk)?;
        let wrow_ctl: Vec<i32> = (0..BLOCK).map(|b| b as i32).collect();
        let wrow_ctl = dev.htod_sync_copy(&wrow_ctl)?;
        let toks_blk = dev.htod_sync_copy(&vec![0i32; BLOCK])?;

        dev.synchronize()?;
        if sharded {
            eprintln!("[df2] round SHARDED across {world} ranks (rank {shard_rank}): \
                       qkv/gate/up col-split ({}q/{}kv heads, {} inter rows/rank), o/down K-split, \
                       ring KV [{} x {} x {}]/layer, 2 all-reduce sites/layer on the round stream",
                      nq_l, nkv_l, ni_l, nkv_l, RING_STRIDE, HEAD_DIM);
        }
        let mut round = Self {
            dev, stream, bk, layers, glob, hp_w, pred_cb, succ_cb, head, embed,
            head_hadamard16: false,
            head_input: alloc_z(HIDDEN * 7),
            mr_input: alloc_z(INTER * BLOCK),
            prime_mr: alloc_z(HIDDEN * max_c),
            calib: None,
            w4a4,
            cos_table, sin_table, k_ring, v_ring, nprev: 0,
            staging,
            sink: None,
            th_raw: alloc_z(HIDDEN * BLOCK),
            th: alloc_z(HIDDEN * BLOCK),
            kc: alloc_z(NUM_KV_HEADS * HEAD_DIM * BLOCK),
            vc: alloc_z(NUM_KV_HEADS * HEAD_DIM * BLOCK),
            cos_c: alloc_zf(BLOCK * HEAD_DIM),
            sin_c: alloc_zf(BLOCK * HEAD_DIM),
            pos_c: alloc_zi(BLOCK),
            wrow_c: alloc_zi(BLOCK),
            slot_c: dev5.htod_sync_copy(&vec![0i32; BLOCK])?,
            blk, pos_blk, wrow_blk, wrow_ctl, toks_blk,
            logits: alloc_z(VOCAB * 7),
            out_vals: alloc_zf(7 * 16),
            out_ids: dev5.alloc_zeros::<u32>(7 * 16).expect("alloc u32"),
            hp_bf16: alloc_z(SELECTOR_RANK * 8),   // m8 kernel writes 8 cols (2048); walk reads 7
            hp_f32: alloc_zf(SELECTOR_RANK * 8),
            unary_ctl: alloc_zf(7 * 16),
            walk_tokens: dev5.alloc_zeros::<u32>(7).expect("alloc u32"),
            ctl_attn_ref: alloc_z(NUM_HEADS * HEAD_DIM * BLOCK),
            ctl_attn: alloc_z(NUM_HEADS * HEAD_DIM * BLOCK),
            walk_scores: alloc_zf(7 * 16),
            walk_out_tok: dev5.alloc_zeros::<u32>(7 + 7 * 16).expect("alloc u32"),
            walk_out_q: alloc_zf(7 + 7 * 16),
            walk_seeds: dev5.alloc_zeros::<u32>(7).expect("alloc u32"),
            max_c,
            ntot_dev: 0,
            anchor_dev: 0,
            round_graph: None,
            inject_graphs: std::collections::HashMap::new(),
            ntot_buf: None,
            anchor_buf: None,
            ar,
            nq_l,
            nkv_l,
            ni_l,
        };
        if mixed {
            anyhow::ensure!(!sharded, "MR-GPTQ DFlash2 round sharding is not implemented yet");
            round.install_quantized_projections(dir)?;
            eprintln!("[df2] loaded mixed MR-GPTQ/NVFP4 drafter artifact (35 projections, {})",
                      if round.w4a4.is_some() { "round W4A4 / context W4A16" } else { "W4A16" });
        }
        Ok(round)
    }

    fn install_quantized_projections(&mut self, dir: &str) -> Result<()> {
        let rd = crate::gptq::ShardReader::open(std::path::Path::new(dir))?;
        let rotated = std::fs::read_to_string(std::path::Path::new(dir).join("config.json"))
            .ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .map(|j| {
                let t = &j["quantization_config"]["transform"];
                t.as_str() == Some("hadamard16") || t["type"].as_str() == Some("hadamard16")
            }).unwrap_or(false);
        let specs = [
            ("self_attn.q_proj", NUM_HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.k_proj", NUM_KV_HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.v_proj", NUM_KV_HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.o_proj", HIDDEN, NUM_HEADS * HEAD_DIM),
            ("mlp.gate_proj", INTER, HIDDEN), ("mlp.up_proj", INTER, HIDDEN),
            ("mlp.down_proj", HIDDEN, INTER),
        ];
        for li in 0..N_LAYERS {
            for &(suffix, m, k) in &specs {
                let stem = format!("layers.{li}.{suffix}");
                anyhow::ensure!(rd.metas.contains_key(&format!("{stem}.weight_packed")),
                                "mixed DFlash2 artifact misses {stem}.weight_packed");
                let w = load_round_nvfp4(&self.dev, &rd, &stem, m, k, rotated, self.w4a4.is_some())?;
                match suffix {
                    "self_attn.q_proj" => self.layers[li].q_proj = w,
                    "self_attn.k_proj" => self.layers[li].k_proj = w,
                    "self_attn.v_proj" => self.layers[li].v_proj = w,
                    "self_attn.o_proj" => self.layers[li].o_proj = w,
                    "mlp.gate_proj" => self.layers[li].gate_proj = w,
                    "mlp.up_proj" => self.layers[li].up_proj = w,
                    "mlp.down_proj" => self.layers[li].down_proj = w,
                    _ => unreachable!(),
                }
            }
        }
        self.dev.synchronize()?;
        Ok(())
    }

    /// Attach the TRUNK's tap sink as this round's fc input (the capture staging is then read
    /// directly by [`inject_dev`] — no intermediate copy).
    pub fn attach_sink(&mut self, sink: &Arc<Df2TapSink>) {
        // S5F3 fix: cudarc's CudaSlice::clone() is a DEEP copy (try_clone = fresh alloc +
        // dtod), NOT a shared handle. Cloning here snapshot the sink's then-zeroed buffer and
        // the trunk's live captures never reached the round: every inject consumed ZERO taps.
        // Keep the Arc; the serving lanes call `sync_staging_from_sink` before each inject.
        self.sink = Some(sink.clone());
    }

    /// S5F3: copy the attached sink's LIVE staging (the trunk's capture columns [0..8)) into
    /// the round's own staging, stream-ordered on the round's stream. The caller must have
    /// synced the trunk's capture first (the lane calls this after the verify returns).
    /// No-op when no sink is attached (the probe's upload_chunk path writes self.staging).
    pub fn sync_staging_from_sink(&mut self) -> Result<()> {
        if let Some(sink) = &self.sink {
            use cudarc::driver::sys;
            let cp = sys::CUDA_MEMCPY2D {
                srcXInBytes: 0, srcY: 0,
                srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: std::ptr::null(), srcDevice: *sink.staging.device_ptr() as u64,
                srcArray: std::ptr::null_mut(), srcPitch: TAP_CONCAT_DIM * 2,
                dstXInBytes: 0, dstY: 0,
                dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstHost: std::ptr::null_mut(), dstDevice: *self.staging.device_ptr() as u64,
                dstArray: std::ptr::null_mut(), dstPitch: TAP_CONCAT_DIM * 2,
                WidthInBytes: TAP_CONCAT_DIM * 2, Height: BLOCK,
            };
            unsafe {
                let r = sys::cuMemcpy2DAsync_v2(&cp, self.stream.stream);
                if r != sys::CUresult::CUDA_SUCCESS {
                    anyhow::bail!("df2 sync staging D2D failed: {r:?}");
                }
            }
            self.dev.synchronize()?;
        }
        Ok(())
    }

    pub fn nprev(&self) -> usize { self.nprev }

    /// P2: the rank-local kv-head count (NUM_KV_HEADS / world when sharded).
    pub fn nkv(&self) -> usize { self.nkv_l }

    /// P2 — world>2 all-reduce of a bf16 buffer ON THE ROUND'S STREAM: the faithful mirror of
    /// `GpuModel::tp_all_reduce_bf16`'s world>2 branch — one-shot (P3-1 predicate permitting)
    /// or the recursive-doubling tree (K1 `tp_gate_copy_signal` + K2 `tp_wait_add{,_g}`, fp32
    /// accumulate in the persistent scratch, ONE bf16 rounding per chunk via `f32tobf16`, the
    /// canonical fold that keeps the sum bitwise rank-identical). The round's payloads
    /// ([5120×8] bf16 = 80 KB) always fit a single chunk (slot ~1 MB). Launched inside
    /// `layer_forward`, so the sites capture into the round's CUDA graph exactly like the
    /// trunk's verify-graph ARs; the device-side epoch counter (not a kernel arg) advances
    /// identically on every rank — SPMD by construction.
    fn tp_ar_bf16(&self, ptr: u64, n: usize) {
        let Some(ar) = &self.ar else { panic!("df2 tp_ar_bf16: no AR ctx (unsharded round)") };
        debug_assert!(ar.world > 2, "df2 round shard engages only at world > 2");
        if ar.oneshot && n * 2 <= ar.payload_bytes.max(1) {
            klaunch!(self, "tp_gate_copy_signal", (1u32, 1, 1), (1024, 1, 1), 0,
                (ar.ctx_dptr, ptr, (n * 2) as u32));
            klaunch!(self, "tp_wait_add_4way", (1u32, 1, 1), (1024, 1, 1), 0,
                (ar.ctx_dptr, ptr, ptr, n as i32, 0i32));
            return;
        }
        let sp = ar.f32_scratch;
        let rounds = ar.world.trailing_zeros() as usize;   // log2(world); world>2 ⇒ ≥2
        let chunk = ((crate::tp::TP_SLOT_BYTES - 64) & !7) / 4;   // fp32 elems per ring slot
        let k2: &str = if ar.k2_gpu_recv { "tp_wait_add_g" } else { "tp_wait_add" };
        let mut off = 0usize;
        while off < n {
            let c = (n - off).min(chunk);
            let bp = ptr + (off * 2) as u64;   // bf16 partial source (round-0 K1 input)
            klaunch!(self, "tp_gate_copy_signal", (1u32, 1, 1), (1024, 1, 1), 0,
                (ar.ctx_dptr, bp, (c * 2) as u32));
            klaunch!(self, k2, (1u32, 1, 1), (1024, 1, 1), 0,
                (ar.ctx_dptr, sp, bp, c as i32, 3i32));
            for _ in 1..rounds {
                klaunch!(self, "tp_gate_copy_signal", (1u32, 1, 1), (1024, 1, 1), 0,
                    (ar.ctx_dptr, sp, (c * 4) as u32));
                klaunch!(self, k2, (1u32, 1, 1), (1024, 1, 1), 0,
                    (ar.ctx_dptr, sp, sp, c as i32, 2i32));
            }
            // Round this chunk's fp32 sum to bf16 ONCE (the single rounding boundary).
            klaunch!(self, "f32tobf16", grid(c), (256, 1, 1), 0, (bp, sp, c as i32));
            off += c;
        }
    }

    /// The drafter's ring-KV allocation ranges (ptr, bytes) per layer — the S5F no-alias probe
    /// asserts these never intersect the trunk's KV-cache ranges (the drafter's ring is
    /// drafter-private; an overlap would corrupt trunk KV on write).
    pub fn ring_kv_ptr_ranges(&self) -> Vec<(u64, u64)> {
        let bytes = self.nkv_l * RING_STRIDE * HEAD_DIM * 2;
        (0..N_LAYERS).flat_map(|li| {
            [(d(&self.k_ring[li]), bytes as u64), (d(&self.v_ring[li]), bytes as u64)]
        }).collect()
    }

    /// Gather ONLY the block input (anchor + 7×MASK) from the trunk's embed — the probe's
    /// embed gate, isolated from the layer stack.
    pub fn embed_probe(&mut self, anchor: u32) -> Result<Vec<f32>> {
        let toks: Vec<i32> = [anchor as i32].iter().copied()
            .chain(std::iter::repeat(crate::dflash2::MASK_TOKEN_ID as i32).take(BLOCK - 1)).collect();
        self.dev.htod_sync_copy_into(&toks, &mut self.toks_blk)?;
        self.embed_gather();
        self.dev.synchronize()?;
        let h: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.h)?;
        Ok(h.iter().map(|x| x.to_f32()).collect())
    }

    /// Embed the block input `toks_blk` from the TRUNK's embed, dispatching on the borrowed
    /// dtype: NVFP4 (the MMA-repacked serving embed) or BF16 (the plain-BF16 trunk class).
    fn embed_gather(&self) {
        match self.embed.expect("embed ptrs") {
            BorrowedW::Nvfp4(p) => {
                klaunch!(self, "embed_gather_fp4_tiled_b", grid(HIDDEN * BLOCK), (256, 1, 1), 0,
                    (d(&self.blk.h), p.qweight, p.scales, p.gs, d(&self.toks_blk), HIDDEN as i32, BLOCK as i32));
            }
            BorrowedW::Bf16 { ptr } => {
                klaunch!(self, "embed_gather_b", grid(HIDDEN * BLOCK), (256, 1, 1), 0,
                    (d(&self.blk.h), ptr, d(&self.toks_blk), HIDDEN as i32, BLOCK as i32));
            }
        }
    }

    /// The borrowed head at N=7 on h_final cols 1..7 (`[5120,7]` col-major at +5120), dispatching
    /// on the borrowed dtype: NVFP4 (the trunk's own persistent `gemm_mma_fp4_b`; N<=16 legal via
    /// the N-clamped X reads) or BF16 (`gemm_binv_b`, NC=7 specialization, the batch-invariant
    /// fixed-order reduction — the same kernel the bf16 serving chain's logits use).
    fn head_logits(&self) {
        let h_final = d(&self.blk.h_final) + (HIDDEN * 2) as u64;
        let head_input = if self.head_hadamard16 {
            let blocks = HIDDEN * 7 / 16;
            klaunch!(self, "gptq_rotate_act_b", grid(blocks), (256, 1, 1), 0,
                (d(&self.head_input), h_final, blocks as i64));
            d(&self.head_input)
        } else {
            h_final
        };
        match self.head.expect("head ptrs") {
            BorrowedW::Nvfp4(p) => {
                let persistent = (crate::gpu::GB10_SMS * 6).min(VOCAB / 16) as u32;
                klaunch!(self, "gemm_mma_fp4_b", (persistent, 1, 1), (256, 1, 1), 0,
                    (d(&self.logits), p.qweight, p.scales, p.gs,
                     head_input,
                     VOCAB as i32, HIDDEN as i32, 7i32, 0u64, 0i32));
            }
            BorrowedW::Bf16 { ptr } => {
                let smem = (7 * 256 * 4) as u32;
                klaunch!(self, "gemm_binv_b", (VOCAB as u32, 1, 1), (256, 1, 1), smem,
                    (d(&self.logits), ptr, head_input,
                     VOCAB as i32, HIDDEN as i32, 7i32));
            }
        }
    }

    /// Configure the activation transform expected by the borrowed trunk lm_head.
    pub fn set_head_hadamard16(&mut self, enabled: bool) {
        self.head_hadamard16 = enabled;
        if enabled {
            eprintln!("[df2] borrowed lm_head uses hadamard16; rotating DFlash2 h_final before the head GEMM");
        }
    }

    /// Restart the round at position 0 (the ring contents are simply re-overwritten in
    /// position order; un-committed rows are never read).
    pub fn reset(&mut self) {
        self.nprev = 0;
    }

    /// Probe-only: decode tile-row `mt` of an NVFP4 tensor via the trunk's OWN
    /// `dequant_fp4_tiled_b` (fp4_tiled_at over the repacked tiles), rows [0,16) x K.
    pub fn dequant_probe(&self, mt: usize, w: u64, sc: u64, gs: u64, k: usize) -> Result<Vec<f32>> {
        let out = self.dev.alloc_zeros::<bf16>(16 * k).context("dequant probe alloc")?;
        let base = (mt * (k / 16) * 128) as u64;
        let sbase = (mt * (k / 16) * 16) as u64;
        klaunch!(self, "dequant_fp4_tiled_b", grid((16 * k + 255) / 256), (256, 1, 1), 0,
            (d(&out), w + base, sc + sbase, gs, 16i32, k as i32));
        self.dev.synchronize()?;
        let o: Vec<bf16> = self.dev.dtoh_sync_copy(&out)?;
        Ok(o.iter().map(|x| x.to_f32()).collect())
    }

    /// The ring-vs-linear control pair (layer 0): (ring reference, S3F linear) attn outputs.
    /// Requires the last round to have run with `ctl_dual_write`.
    pub fn dump_ctl_pair(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let r: Vec<bf16> = self.dev.dtoh_sync_copy(&self.ctl_attn_ref)?;
        let l: Vec<bf16> = self.dev.dtoh_sync_copy(&self.ctl_attn)?;
        Ok((r.iter().map(|x| x.to_f32()).collect(), l.iter().map(|x| x.to_f32()).collect()))
    }

    /// The last chunk's k/v scratch AFTER inject (layer N-1's k post-rope / raw v),
    /// `[8, nkv*hd]` row-major — the probe's k-path gate.
    pub fn dump_kvc(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let k: Vec<bf16> = self.dev.dtoh_sync_copy(&self.kc)?;
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.vc)?;
        Ok((k.iter().map(|x| x.to_f32()).collect(), v.iter().map(|x| x.to_f32()).collect()))
    }

    /// Probe-only: rewind the committed position (the perf loop's steady-state cycle).
    pub fn rollback_nprev(&mut self, n: usize) {
        assert!(n <= self.nprev);
        self.nprev = n;
    }

    /// The committed th rows `[nprev, 5120]` (bf16 values as f32; probe gate).
    pub fn dump_th(&self) -> Result<Vec<f32>> {
        let t: Vec<bf16> = self.dev.dtoh_sync_copy(&self.th)?;
        Ok(t.iter().map(|x| x.to_f32()).collect())
    }
    pub fn max_c(&self) -> usize { self.max_c }

    /// Upload `m ≤ 8` tap columns (col-major `[25600, m]`, host bf16) into the staging buffer.
    /// Only the first m columns are written; cols m..8 keep their previous bytes (their GEMM
    /// outputs are garbage columns that nothing downstream reads — column-independent math).
    pub fn upload_chunk(&mut self, cols: &[bf16], m: usize) -> Result<()> {
        // htod_sync_copy_into requires EQUAL lens, so the copy is always the full BLOCK width
        // (row-major [8, 25600]); rows >= m are garbage-but-unread (column-independent math).
        assert!(m <= BLOCK && m >= 1);
        assert!(cols.len() >= TAP_CONCAT_DIM * BLOCK, "upload_chunk needs the full {} rows", BLOCK);
        self.dev.htod_sync_copy_into(&cols[..TAP_CONCAT_DIM * BLOCK], &mut self.staging)
            .context("upload tap chunk")?;
        Ok(())
    }

    // ---- kernel helpers (S3F's launch shapes) --------------------------------

    fn gemm_dsp(&self, out: &CudaSlice<bf16>, w: &CudaSlice<bf16>, x_ptr: u64, outn: usize, inn: usize) {
        let g = ((outn + 3) / 4) as u32; // R=4
        klaunch!(self, "gemm_dsp_b_m8_r4", (g, 1, 1), (256, 1, 1), 0,
            (d(out), d(w), x_ptr, outn as i32, inn as i32));
    }

    /// Launch one fixed-N=8 W4A4 projection from row-major activation/output pointers.
    fn gemm_w4a4(&self, out_ptr: u64, w4: &Df2W4a4, wq: &CudaSlice<u8>, ws: &CudaSlice<u8>,
                  gs: &CudaSlice<f32>, x_ptr: u64, outn: usize, inn: usize,
                  rows: usize, xgs: f32) {
        assert!(rows == BLOCK && inn <= w4.k_max);
        assert!(inn % 64 == 0 && outn % 16 == 0);
        let pad8 = rows.div_ceil(8) * 8;
        unsafe {
            w4.quant.clone().launch_on_stream(
                &self.stream,
                LaunchConfig {
                    grid_dim: ((inn / 64) as u32, (pad8 / 8) as u32, 1),
                    block_dim: (256, 1, 1), shared_mem_bytes: 0,
                },
                (x_ptr, inn as i32, rows as i32, pad8 as i32,
                 d(&w4.bq), d(&w4.sb), xgs),
            ).expect("df2 w4a4 quant launch");
            if crate::w4a4::n8_on() {
                w4.gemm_n8.clone().launch_on_stream(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: (outn.div_ceil(128) as u32, 1, 1),
                        block_dim: (128, 1, 1), shared_mem_bytes: crate::w4a4::W4_N8_SMEM,
                    },
                    (d(wq), d(ws), d(&w4.bq), d(&w4.sb), d(gs), out_ptr,
                     outn as i32, rows as i32, inn as i32, 1.0f32 / xgs),
                ).expect("df2 w4a4 n8 gemm launch");
            } else {
                w4.gemm.clone().launch_on_stream(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: (crate::w4a4::W4a4State::dense_grid(outn, rows), 1, 1),
                        block_dim: (256, 1, 1), shared_mem_bytes: crate::w4a4::W4_SMEM,
                    },
                    (d(wq), d(ws), d(&w4.bq), d(&w4.sb), d(gs), out_ptr,
                     outn as i32, rows as i32, inn as i32, 1.0f32 / xgs),
                ).expect("df2 w4a4 wide gemm launch");
            }
        }
    }

    /// Fixed-width (N=8) projection dispatch. MR tensors rotate their activation first; an
    /// explicit DFlash2 A4 policy uses the same quantized activation path as prompt prime.
    fn gemm_weight_impl(&self, out: &CudaSlice<bf16>, w: &RoundWeight, x_ptr: u64,
                        outn: usize, inn: usize, allow_w4a4: bool) {
        match w {
            RoundWeight::Bf16(w) => self.gemm_dsp(out, w, x_ptr, outn, inn),
            RoundWeight::Nvfp4 { qweight, scales, w4_qweight, w4_scales, gs, input_gs,
                                 m, k, rotated, .. } => {
                assert_eq!((*m, *k), (outn, inn), "df2 nvfp4 projection shape");
                let xp = if *rotated {
                    let blocks = inn * BLOCK / 16;
                    klaunch!(self, "gptq_rotate_act_b", grid(blocks), (256, 1, 1), 0,
                        (d(&self.mr_input), x_ptr, blocks as i64));
                    d(&self.mr_input)
                } else { x_ptr };
                if let (true, Some(w4), Some(wq), Some(ws), Some(xgs)) =
                    (allow_w4a4, self.w4a4.as_ref(), w4_qweight.as_ref(), w4_scales.as_ref(), *input_gs) {
                    self.gemm_w4a4(d(out), w4, wq, ws, gs, xp, outn, inn, BLOCK, xgs);
                    return;
                }
                let persistent = (crate::gpu::GB10_SMS * 6).min(outn / 16) as u32;
                klaunch!(self, "gemm_mma_fp4_b", (persistent, 1, 1), (256, 1, 1), 0,
                    (d(out), d(qweight), d(scales), d(gs), xp,
                     outn as i32, inn as i32, BLOCK as i32, 0u64, 0i32));
            }
        }
    }

    fn gemm_weight(&self, out: &CudaSlice<bf16>, w: &RoundWeight, x_ptr: u64,
                   outn: usize, inn: usize) {
        self.gemm_weight_impl(out, w, x_ptr, outn, inn, true);
    }

    fn gemm_weight_context(&self, out: &CudaSlice<bf16>, w: &RoundWeight, x_ptr: u64,
                           outn: usize, inn: usize) {
        self.gemm_weight_impl(out, w, x_ptr, outn, inn, false);
    }

    /// Large-M GEMM (S3F's ctx-side kernel): out [outn, m] = w [outn, k] x x [k, m] col-major bf16.
    fn gemm_tiled(&self, out: &CudaSlice<bf16>, w: &CudaSlice<bf16>, x_ptr: u64, outn: usize, k: usize, m: usize) {
        let mx = ((m + 127) / 128) as u32;
        let nx = ((outn + 127) / 128) as u32;
        klaunch!(self, "gemm_tiled_b", (mx, nx, 1), (16, 16, 1), 0,
            (d(out), d(w), x_ptr, outn as i32, k as i32, m as i32));
    }

    /// Large-M prompt-prime dispatch. The quantized matrix is dequantized once at load; only the
    /// activation Hadamard is performed here, preserving both the MR basis and the W4A16 context
    /// contract shared with incremental injection.
    fn gemm_tiled_weight(&self, out: &CudaSlice<bf16>, w: &RoundWeight, x_ptr: u64,
                         outn: usize, k: usize, m: usize) {
        match w {
            RoundWeight::Bf16(w) => self.gemm_tiled(out, w, x_ptr, outn, k, m),
            RoundWeight::Nvfp4 { prime_bf16, m: wm, k: wk, rotated, .. } => {
                assert_eq!((*wm, *wk), (outn, k), "df2 nvfp4 prime shape");
                if *rotated {
                    assert!(k * m <= self.prime_mr.len(), "df2 prime MR scratch capacity");
                    let blocks = k * m / 16;
                    klaunch!(self, "gptq_rotate_act_b", grid(blocks), (256, 1, 1), 0,
                        (d(&self.prime_mr), x_ptr, blocks as i64));
                    self.gemm_tiled(out, prime_bf16, d(&self.prime_mr), outn, k, m);
                } else {
                    self.gemm_tiled(out, prime_bf16, x_ptr, outn, k, m);
                }
            }
        }
    }

    fn rmsnorm(&self, out: &CudaSlice<bf16>, x: &CudaSlice<bf16>, w: &CudaSlice<f32>, n: usize, b: usize) {
        klaunch!(self, "rmsnorm_b", (b as u32, 1, 1), (1024, 1, 1), 4096,
            (d(out), d(x), d(w), n as i32, b as i32, fbits(RMS_EPS)));
    }

    fn rmsnorm_perhead(&self, x: &CudaSlice<bf16>, w: &CudaSlice<f32>, heads: usize, b: usize) {
        klaunch!(self, "rmsnorm_perhead_b", ((b * heads) as u32, 1, 1), (HEAD_DIM as u32, 1, 1), (HEAD_DIM * 4) as u32,
            (d(x), d(x), d(w), heads as i32, HEAD_DIM as i32, b as i32, fbits(RMS_EPS)));
    }

    fn rope(&self, x: &CudaSlice<bf16>, cos: &CudaSlice<f32>, sin: &CudaSlice<f32>, heads: usize, b: usize) {
        klaunch!(self, "rope_b", grid(b * heads * (HEAD_DIM / 2)), (256, 1, 1), 0,
            (d(x), d(cos), d(sin), heads as i32, HEAD_DIM as i32, HEAD_DIM as i32, b as i32));
    }

    fn gather_rope(&self, out_cos: &CudaSlice<f32>, out_sin: &CudaSlice<f32>, pos: u64, b: usize) {
        klaunch!(self, "gather_rope_b", grid(b * HEAD_DIM), (256, 1, 1), 0,
            (d(out_cos), d(out_sin), d(&self.cos_table), d(&self.sin_table), pos, HEAD_DIM as i32, b as i32));
    }

    fn conv2(&self, out: &CudaSlice<bf16>, x: &CudaSlice<bf16>, dyn_all: &CudaSlice<bf16>,
             base_ptr: u64, n: usize, side: usize) {
        let dyn_side = (side * CONV_KERNEL * CONV_GROUPS) as i32;
        let dyn_stride = (2 * CONV_KERNEL * CONV_GROUPS) as i32;
        klaunch!(self, "conv2_dynamic_b", grid(n * HIDDEN), (256, 1, 1), 0,
            (d(out), d(x), d(dyn_all), base_ptr, HIDDEN as i32, n as i32,
             CONV_GROUPS as i32, CONV_GROUP as i32, dyn_side, dyn_stride));
    }

    // ---- §3.2 incremental ctx injection (M ≤ 8, gemm_dsp path) ---------------

    /// Inject `m ≤ 8` committed tap columns (read from `self.staging` cols `[0, m)` — the
    /// trunk capture's staging or an upload). Positions are ABSOLUTE `[nprev, nprev+m)`:
    /// RoPE at true positions; ring write rows `pos % RING`. `timer` (optional) brackets the
    /// whole injection (fc + norm + 5×(k/v proj + norm + rope + write)).
    pub fn inject_dev(&mut self, m: usize, mut timer: Option<&mut EvTimer>) -> Result<()> {
        assert!(m <= BLOCK, "inject chunk {m} > BLOCK {BLOCK}");
        assert!(self.nprev + m <= self.max_c, "nprev {} + {m} > max_c {}", self.nprev, self.max_c);
        let n0 = self.nprev;
        let ring = RING;

        // rope positions (absolute) + ring write rows for this chunk (padded to BLOCK width:
        // htod_sync_copy_into needs equal lens; rows >= m are garbage-but-unread)
        let mut pos: Vec<i32> = (0..m).map(|j| (n0 + j) as i32).collect();
        let mut wrow: Vec<i32> = (0..m).map(|j| ((n0 + j) % ring) as i32).collect();
        pos.resize(BLOCK, 0);
        wrow.resize(BLOCK, 0);
        self.dev.htod_sync_copy_into(&pos, &mut self.pos_c)?;
        self.dev.htod_sync_copy_into(&wrow, &mut self.wrow_c)?;
        if let Some(t) = timer.as_deref_mut() { t.mark(self.stream.stream); }

        // E0: replay the captured injection graph when this width has one (the sequence is
        // width-contingent, so the map is keyed by m). Same kernels, same order, same buffers —
        // the probe's graph-vs-eager determinism assert covers the pair. The eager call at a
        // fresh m below doubles as the warmup; the capture that follows only RECORDS.
        if self.inject_graphs.contains_key(&m) {
            if let Some(g) = self.inject_graphs.get(&m) { g.launch(); }
            if let Some(t) = timer.as_deref_mut() { t.mark(self.stream.stream); }
            self.nprev = n0 + m;
            return Ok(());
        }
        self.inject_kernels(m);
        if std::env::var("GB10_NO_DF2_INJECT_GRAPH").is_err() {
            let _ = self.capture_inject_graph(m);
        }
        if let Some(t) = timer.as_deref_mut() { t.mark(self.stream.stream); }
        self.nprev = n0 + m;
        Ok(())
    }

    /// The pure-launch body of `inject_dev` (fc + hidden_norm at M=m, then per-layer
    /// k/v projections + ring writes). Eager execution and CUDA-graph capture call the SAME
    /// function — the capture records exactly the eager sequence. No allocs, no syncs, no
    /// all-reduce (kv projections are rank-local) — capture-safe by construction.
    fn inject_kernels(&self, m: usize) {
        // fc + hidden_norm at M=m (gemm_dsp; cols ≥ m are garbage-but-unread)
        self.gemm_dsp(&self.th_raw, &self.glob.fc, d(&self.staging), HIDDEN, TAP_CONCAT_DIM);
        self.rmsnorm(&self.th, &self.th_raw, &self.glob.hidden_norm, HIDDEN, m);

        // per-layer k/v: k = rope(k_norm(k_proj(th))) @ ring rows; v raw
        // P2: the projections run at the rank-local kv-head width (heads are independent —
        // no all-reduce; each rank writes only ITS kv-heads' ring rows).
        let nkv = self.nkv_l;
        self.gather_rope(&self.cos_c, &self.sin_c, d(&self.pos_c), m);
        for li in 0..N_LAYERS {
            let l = &self.layers[li];
            self.gemm_weight_context(&self.kc, &l.k_proj, d(&self.th), nkv * HEAD_DIM, HIDDEN);
            self.rmsnorm_perhead(&self.kc, &l.k_norm, nkv, m);
            self.rope(&self.kc, &self.cos_c, &self.sin_c, nkv, m);
            self.gemm_weight_context(&self.vc, &l.v_proj, d(&self.th), nkv * HEAD_DIM, HIDDEN);
            klaunch!(self, "write_kv_b", grid(m * nkv * HEAD_DIM), (256, 1, 1), 0,
                (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&self.kc), d(&self.vc),
                 d(&self.wrow_c), RING_STRIDE as i32, nkv as i32, HEAD_DIM as i32,
                 m as i32, d(&self.slot_c)));
        }
    }

    /// E0: capture `inject_kernels(m)` into a per-width CUDA graph (the `capture_round_graph`
    /// pattern). The DF2 stream is quiesced first (the eager inject just enqueued); the capture
    /// pass records only. Failure is non-fatal: the eager path stays.
    fn capture_inject_graph(&mut self, m: usize) -> bool {
        use cudarc::driver::sys;
        if self.dev.synchronize().is_err() { return false; }
        let stream = self.stream.stream;
        let r = unsafe { sys::cuStreamBeginCapture_v2(stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) };
        if r != sys::CUresult::CUDA_SUCCESS { return false; }
        self.inject_kernels(m);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r = unsafe { sys::cuStreamEndCapture(stream, &mut graph) };
        if r != sys::CUresult::CUDA_SUCCESS { return false; }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r = unsafe { sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        unsafe { sys::cuGraphDestroy(graph); }
        if r != sys::CUresult::CUDA_SUCCESS { return false; }
        self.inject_graphs.insert(m, crate::gpu::CudaGraph::from_exec(exec, stream));
        true
    }

    /// S5F — engine prompt-prime: inject `n` tap columns at ONE large M (the whole prefill
    /// window) via the S3F tiled GEMM, instead of n/8 skinny-M chunks. `taps` is the prefill
    /// capture buffer `[TAP_CONCAT_DIM, n]` col-major bf16 (the same per-layer-row layout as the
    /// sink's staging), cols [0, n) = the window's taps. Positions are ABSOLUTE
    /// `[pos_start, pos_start+n)`; ring write rows `pos % RING`. Sets `nprev = pos_start + n`.
    /// Bit-identical k/v to the probe's chunked path (same kernels, same reduction orders — the
    /// per-chunk probe asserts th/k bitwise; M only changes the GEMM tile count, not the math per
    /// element). Runs on the round's stream; the caller must have synced the capture D2Ds.
    pub fn prime_window(&mut self, taps: &CudaSlice<bf16>, n: usize, pos_start: usize) -> Result<()> {
        assert!(n >= 1 && n <= 8192, "prime window {n} out of range (1..8192)");
        assert!(pos_start + n <= self.max_c, "prime {}..{} > max_c {}",
                pos_start, pos_start + n - 1, self.max_c);
        let ring = RING;

        let pos: Vec<i32> = (0..n).map(|j| (pos_start + j) as i32).collect();
        let wrow: Vec<i32> = (0..n).map(|j| ((pos_start + j) % ring) as i32).collect();
        // slot ids are ALL ZERO (single-slot ring) but `write_kv_b` indexes slot_ids[b] for
        // b in 0..B — a buffer sized for BLOCK=8 would be an OOB READ at n > 8 (the first real
        // S5F bug this session caught by the lossless gate's crash).
        let slots: Vec<i32> = vec![0i32; n];
        let pos_dev = self.dev.htod_sync_copy(&pos).context("prime pos")?;
        let wrow_dev = self.dev.htod_sync_copy(&wrow).context("prime wrow")?;
        let slot_dev = self.dev.htod_sync_copy(&slots).context("prime slots")?;
        let cos_c = self.dev.alloc_zeros::<f32>(n * HEAD_DIM).context("prime cos")?;
        let sin_c = self.dev.alloc_zeros::<f32>(n * HEAD_DIM).context("prime sin")?;
        self.gather_rope(&cos_c, &sin_c, d(&pos_dev), n);

        // fc + hidden_norm at M=n (one weight read for the whole window).
        let th_raw = self.dev.alloc_zeros::<bf16>(HIDDEN * n).context("prime th_raw")?;
        let th = self.dev.alloc_zeros::<bf16>(HIDDEN * n).context("prime th")?;
        self.gemm_tiled(&th_raw, &self.glob.fc, d(taps), HIDDEN, TAP_CONCAT_DIM, n);
        self.rmsnorm(&th, &th_raw, &self.glob.hidden_norm, HIDDEN, n);

        // per-layer k/v at M=n, written at ring rows (pos_start+j) % RING.
        // P2: rank-local kv-head width (same as inject_dev).
        let nkv = self.nkv_l;
        let kc = self.dev.alloc_zeros::<bf16>(nkv * HEAD_DIM * n).context("prime k")?;
        let vc = self.dev.alloc_zeros::<bf16>(nkv * HEAD_DIM * n).context("prime v")?;
        for li in 0..N_LAYERS {
            let l = &self.layers[li];
            self.gemm_tiled_weight(&kc, &l.k_proj, d(&th), nkv * HEAD_DIM, HIDDEN, n);
            self.rmsnorm_perhead(&kc, &l.k_norm, nkv, n);
            self.rope(&kc, &cos_c, &sin_c, nkv, n);
            self.gemm_tiled_weight(&vc, &l.v_proj, d(&th), nkv * HEAD_DIM, HIDDEN, n);
            klaunch!(self, "write_kv_b", grid(n * nkv * HEAD_DIM), (256, 1, 1), 0,
                (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&kc), d(&vc),
                 d(&wrow_dev), RING_STRIDE as i32, nkv as i32, HEAD_DIM as i32,
                 n as i32, d(&slot_dev)));
        }
        self.dev.synchronize()?;
        self.nprev = pos_start + n;
        Ok(())
    }

    /// Project one target tap window through the drafter's BF16 `fc + hidden_norm` and return
    /// the compact `[HIDDEN,n]` context representation used by every layer's k/v projections.
    fn ensure_calib_scratch(&mut self, n: usize) -> Result<()> {
        if self.calib.as_ref().map(|c| c.n) == Some(n) { return Ok(()); }
        self.dev.synchronize()?;
        self.calib = Some(CalibScratch {
            n, raw: self.dev.alloc_zeros::<bf16>(HIDDEN * n)?,
            th: self.dev.alloc_zeros::<bf16>(HIDDEN * n)?,
            pos: self.dev.alloc_zeros::<i32>(n)?, rows: self.dev.alloc_zeros::<i32>(n)?,
            slots: self.dev.alloc_zeros::<i32>(n)?,
            cos: self.dev.alloc_zeros::<f32>(n * HEAD_DIM)?, sin: self.dev.alloc_zeros::<f32>(n * HEAD_DIM)?,
            kc: self.dev.alloc_zeros::<bf16>(NUM_KV_HEADS * HEAD_DIM * n)?,
            vc: self.dev.alloc_zeros::<bf16>(NUM_KV_HEADS * HEAD_DIM * n)?,
        });
        Ok(())
    }

    pub fn project_taps_host(&mut self, taps: &CudaSlice<bf16>, n: usize) -> Result<Vec<bf16>> {
        anyhow::ensure!((1..=8192).contains(&n), "df2 projected window {n} out of range");
        self.ensure_calib_scratch(n)?;
        let c = self.calib.as_ref().unwrap();
        self.gemm_tiled(&c.raw, &self.glob.fc, d(taps), HIDDEN, TAP_CONCAT_DIM, n);
        self.rmsnorm(&c.th, &c.raw, &self.glob.hidden_norm, HIDDEN, n);
        self.dev.synchronize()?;
        Ok(self.dev.dtoh_sync_copy(&c.th)?.to_vec())
    }

    /// Prompt-prime from a cached projected context. This is the calibration counterpart of
    /// `prime_window`: it avoids retaining 5× larger raw target taps between layer passes.
    pub fn prime_projected(&mut self, th_host: &[bf16], n: usize, pos_start: usize) -> Result<()> {
        anyhow::ensure!(th_host.len() == HIDDEN * n, "df2 projected cache shape");
        anyhow::ensure!(pos_start + n <= self.max_c, "df2 projected prime exceeds max_c");
        self.ensure_calib_scratch(n)?;
        let pos: Vec<i32> = (0..n).map(|j| (pos_start + j) as i32).collect();
        let rows: Vec<i32> = (0..n).map(|j| ((pos_start + j) % RING) as i32).collect();
        let slots = vec![0i32; n];
        {
            let c = self.calib.as_mut().unwrap();
            self.dev.htod_sync_copy_into(th_host, &mut c.th)?;
            self.dev.htod_sync_copy_into(&pos, &mut c.pos)?;
            self.dev.htod_sync_copy_into(&rows, &mut c.rows)?;
            self.dev.htod_sync_copy_into(&slots, &mut c.slots)?;
        }
        let c = self.calib.as_ref().unwrap();
        self.gather_rope(&c.cos, &c.sin, d(&c.pos), n);
        let nkv = self.nkv_l;
        for li in 0..N_LAYERS {
            let l = &self.layers[li];
            self.gemm_tiled_weight(&c.kc, &l.k_proj, d(&c.th), nkv * HEAD_DIM, HIDDEN, n);
            self.rmsnorm_perhead(&c.kc, &l.k_norm, nkv, n);
            self.rope(&c.kc, &c.cos, &c.sin, nkv, n);
            self.gemm_tiled_weight(&c.vc, &l.v_proj, d(&c.th), nkv * HEAD_DIM, HIDDEN, n);
            klaunch!(self, "write_kv_b", grid(n * nkv * HEAD_DIM), (256, 1, 1), 0,
                (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&c.kc), d(&c.vc), d(&c.rows),
                 RING_STRIDE as i32, nkv as i32, HEAD_DIM as i32, n as i32, d(&c.slots)));
        }
        self.dev.synchronize()?;
        self.nprev = pos_start + n;
        Ok(())
    }

    /// Run through `layer` and snapshot its four distinct projection inputs. Earlier layers use
    /// whichever BF16/NVFP4 weights are currently installed, enabling sequential GPTQ.
    pub fn capture_layer_inputs(&mut self, anchor: u32, layer: usize) -> Result<Df2CalibInputs> {
        anyhow::ensure!(layer < N_LAYERS && self.nprev > 0, "df2 calibration layer/prefix");
        self.refresh_block_pos()?;
        let _ = self.draft_round_depth(anchor, crate::dflash2::SLIDING_WINDOW,
                                       false, false, false, layer + 1)?;
        Ok(Df2CalibInputs {
            qkv: self.dev.dtoh_sync_copy(&self.blk.x_conv)?.to_vec(),
            o: self.dev.dtoh_sync_copy(&self.blk.attn)?.to_vec(),
            gate_up: self.dev.dtoh_sync_copy(&self.blk.x_conv2)?.to_vec(),
            down: self.dev.dtoh_sync_copy(&self.blk.gate)?.to_vec(),
        })
    }

    /// Install one freshly quantized projection during the sequential calibration pass.
    pub fn install_calibrated_projection(&mut self, layer: usize, suffix: &str,
                                         qw: &[u8], sc: &[u8], global_scale: f32,
                                         m: usize, k: usize, rotated: bool) -> Result<()> {
        let prime = crate::quant::dequantize_nvfp4(&crate::quant::Nvfp4Tensor {
            qweight: qw.to_vec(), scales: sc.to_vec(), global_scale, m, k,
        });
        let (wt, st) = crate::quant::repack_nvfp4_mma(qw, sc, m, k);
        let w = RoundWeight::Nvfp4 {
            qweight: self.dev.htod_sync_copy(&wt)?, scales: self.dev.htod_sync_copy(&st)?,
            w4_qweight: None, w4_scales: None, input_gs: None,
            gs: self.dev.htod_sync_copy(&vec![1.0 / global_scale; m / 16])?,
            prime_bf16: self.dev.htod_sync_copy(&prime)?, m, k, rotated,
        };
        match suffix {
            "self_attn.q_proj" => self.layers[layer].q_proj = w,
            "self_attn.k_proj" => self.layers[layer].k_proj = w,
            "self_attn.v_proj" => self.layers[layer].v_proj = w,
            "self_attn.o_proj" => self.layers[layer].o_proj = w,
            "mlp.gate_proj" => self.layers[layer].gate_proj = w,
            "mlp.up_proj" => self.layers[layer].up_proj = w,
            "mlp.down_proj" => self.layers[layer].down_proj = w,
            _ => anyhow::bail!("unknown DFlash2 projection {suffix}"),
        }
        Ok(())
    }

    // ---- §3.3/§3.4 the block pass + borrowed head + selector -----------------

    /// One draft round. Requires `nprev ≥ 1`. `window` = the attention band (2048; the
    /// negative control passes a huge value). `flip_unary` negates the unary term inside the
    /// walk (the selector sign-flip control). `ctl_dual_write` additionally writes the block
    /// k/v at LINEAR rows `[nprev, nprev+8)` so the S3F `gqa_attn_band_b` control can read
    /// the same cache. `dump` collects probe surfaces (logits, hp) + stage timings.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_round(&mut self, anchor: u32, window: usize, flip_unary: bool,
                       ctl_dual_write: bool, dump: bool) -> Result<Df2RoundOut> {
        self.draft_round_depth(anchor, window, flip_unary, ctl_dual_write, dump, N_LAYERS)
    }

    /// Probe-only layer bisect: run the round with the first `nlayers` block layers (the
    /// block h after the last run layer is what the head/top16 see — for the determinism
    /// hunt we only compare `h_final`).
    pub fn draft_round_depth(&mut self, anchor: u32, window: usize, flip_unary: bool,
                             ctl_dual_write: bool, dump: bool, nlayers: usize) -> Result<Df2RoundOut> {
        self.draft_round_stages(anchor, window, flip_unary, ctl_dual_write, dump, nlayers, true)
    }

    /// Determinism bisection: `post` = false stops after the final norm (no head/top16/hp/walk).
    pub fn draft_round_stages(&mut self, anchor: u32, window: usize, flip_unary: bool,
                              ctl_dual_write: bool, dump: bool, nlayers: usize, post: bool)
                              -> Result<Df2RoundOut> {
        let ntot = self.nprev + BLOCK;
        let mut timer = EvTimer::new();
        timer.mark(self.stream.stream);

        // ---- block input: [anchor, MASK×7] from the TRUNK's real embed table ----
        let toks: Vec<i32> = [anchor as i32].iter().copied()
            .chain(std::iter::repeat(crate::dflash2::MASK_TOKEN_ID as i32).take(BLOCK - 1)).collect();
        self.dev.htod_sync_copy_into(&toks, &mut self.toks_blk)?;
        self.embed_gather();

        // ---- 5-layer backbone (S3F sequence; ring attention) ----
        for li in 0..nlayers.min(N_LAYERS) {
            self.layer_forward(li, ntot, window, ctl_dual_write);
        }
        // final norm
        self.rmsnorm(&self.blk.h_final, &self.blk.h, &self.glob.norm, HIDDEN, BLOCK);
        timer.mark(self.stream.stream);
        if !post {
            let mut hf: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.h_final)?;
            hf.truncate(HIDDEN * BLOCK);   // the 9th row is the guard row — not part of the output
            return Ok(Df2RoundOut {
                tokens: vec![0; 7], candidates: vec![0; 7 * 16], unary: vec![0.0; 7 * 16],
                scores: vec![0.0; 7 * 16], h_final: hf.iter().map(|x| x.to_f32()).collect(),
                layer_hiddens: Vec::new(),
                logits: None, hp: None, stage_ms: None,
            });
        }

        // ---- borrowed head at N=7 on h_final cols 1..7 ([5120,7] col-major at +5120) ----
        // Dtype dispatch: NVFP4 (persistent-grid gemm_mma_fp4_b; N<=16 legal) or BF16
        // (gemm_binv_b, NC=7 — the batch-invariant fixed-order reduction).
        self.head_logits();
        timer.mark(self.stream.stream);
        // ---- top-16 on the 7 MASK rows ----
        klaunch!(self, "top16_b", (7u32, 1, 1), (256, 1, 1), 0,
            (d(&self.out_vals), d(&self.out_ids), d(&self.logits), VOCAB as i32, 7i32));
        timer.mark(self.stream.stream);

        // ---- hidden_projection [256,5120] × h_sel [5120,7] (rows 1..7) → hp ----
        self.gemm_dsp(&self.hp_bf16, &self.hp_w, d(&self.blk.h_final) + (HIDDEN * 2) as u64, SELECTOR_RANK, HIDDEN);
        klaunch!(self, "bf16tof32", grid(SELECTOR_RANK * 7), (256, 1, 1), 0,
            (d(&self.hp_f32), d(&self.hp_bf16), (SELECTOR_RANK * 7) as i32));

        // ---- the greedy chain (sign-flip control negates the unary term) ----
        let unary_src: &CudaSlice<f32> = if flip_unary {
            let vals: Vec<f32> = self.dev.dtoh_sync_copy(&self.out_vals)?;
            let neg: Vec<f32> = vals.iter().map(|&v| -v).collect();
            self.dev.htod_sync_copy_into(&neg, &mut self.unary_ctl)?;
            &self.unary_ctl
        } else {
            &self.out_vals
        };
        klaunch!(self, "df2_sel_walk_b", (1u32, 1, 1), (256, 1, 1), 0,
            (d(&self.walk_tokens), d(&self.walk_scores), d(&self.hp_f32), d(&self.out_ids),
             d(unary_src), d(&self.pred_cb), d(&self.succ_cb), anchor, self.anchor_dev, SELECTOR_RANK as i32));
        timer.mark(self.stream.stream);

        // ---- read back ----
        let tokens: Vec<u32> = self.dev.dtoh_sync_copy(&self.walk_tokens)?.to_vec();
        let scores: Vec<f32> = self.dev.dtoh_sync_copy(&self.walk_scores)?.to_vec();
        let candidates: Vec<u32> = self.dev.dtoh_sync_copy(&self.out_ids)?.to_vec();
        let unary: Vec<f32> = self.dev.dtoh_sync_copy(&self.out_vals)?.to_vec();
        let mut hf: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.h_final)?;
        hf.truncate(HIDDEN * BLOCK);   // guard row never leaves the device
        let h_final: Vec<f32> = hf.iter().map(|x| x.to_f32()).collect();
        let (logits, hp) = if dump {
            let lg: Vec<bf16> = self.dev.dtoh_sync_copy(&self.logits)?;
            let mut hp16: Vec<bf16> = self.dev.dtoh_sync_copy(&self.hp_bf16)?;
            hp16.truncate(SELECTOR_RANK * 7);   // 8th gemm column never leaves the device
            (Some(lg.iter().map(|x| x.to_f32()).collect()),
             Some(hp16.iter().map(|x| x.to_f32()).collect()))
        } else {
            (None, None)
        };
        // marks: 0 start, 1 block-end, 2 head-end, 3 top16-end, 4 walk-end
        let stage_ms = Some([timer.elapsed_ms(0), timer.elapsed_ms(1), timer.elapsed_ms(2), timer.elapsed_ms(3)]);
        Ok(Df2RoundOut { tokens, candidates, unary, scores, h_final, layer_hiddens: Vec::new(),
                         logits, hp, stage_ms })
    }

    /// S5F — the ENGINE draft round (lean): the same kernel sequence as `draft_round` with the
    /// probe-only surfaces removed (no EvTimer, no dumps, no ctl/dual-write, no flip) and ONLY the
    /// 7 walk tokens read back. This is the serving-callable round the decode loop drives.
    /// Requires `nprev >= 1` (the committed prefix incl. the prompt prime). The block pass reads
    /// the RING (ctx) — the sink staging is NOT read here (its fresh contents are consumed by
    /// [`inject_dev`] after the trunk verify). Stream-ordered on the round's stream.
    pub fn draft_round_dev(&mut self, anchor: u32) -> Result<Vec<u32>> {
        // ---- block input: [anchor, MASK×7] from the TRUNK's real embed table ----
        let toks: Vec<i32> = [anchor as i32].iter().copied()
            .chain(std::iter::repeat(crate::dflash2::MASK_TOKEN_ID as i32).take(BLOCK - 1)).collect();
        self.dev.htod_sync_copy_into(&toks, &mut self.toks_blk)?;
        // The EAGER path must use the packed-arg kernels (ntot/anchor from the launch args, NOT
        // the graph's device ints): after a capture, self.ntot_dev/anchor_dev point at the graph
        // buffers — leaving them set would make the eager kernels read a STALE ntot (the last
        // replay's) against an eager-sized smem allocation (the graph-capture smem bug class,
        // caught by the probe's varying-nprev replays).
        let (ntot_save, anchor_save) = (self.ntot_dev, self.anchor_dev);
        self.ntot_dev = 0;
        self.anchor_dev = 0;
        let r = self.draft_round_kernels(anchor, false, None);
        self.ntot_dev = ntot_save;
        self.anchor_dev = anchor_save;
        r?;
        // ---- read back ONLY the 7 tokens ----
        self.dev.synchronize()?;
        let tokens: Vec<u32> = self.dev.dtoh_sync_copy(&self.walk_tokens)?.to_vec();
        Ok(tokens)
    }

    /// S5F3 — the DUMP draft round: the same eager kernel sequence as `draft_round_dev` (incl.
    /// the packed-arg zeroing so a captured graph's stale device ints can't leak in) with the
    /// FULL probe surfaces read back: the 7 tokens, the top-16 candidates/unary, the chain
    /// scores, the block h_final `[8, 5120]` f32, and (dump=true) the [7, vocab] logits + hp.
    /// The step-dump analysis compares these against the S2F oracle's run_round surfaces on the
    /// same taps (the S1 bisect). Behavior-neutral vs `draft_round_dev`: same kernels, same
    /// inputs — only the readbacks differ. Callers must refresh_block_pos() first (as the
    /// serving eager path does).
    pub fn draft_round_full(&mut self, anchor: u32) -> Result<Df2RoundOut> {
        let (ntot_save, anchor_save) = (self.ntot_dev, self.anchor_dev);
        self.ntot_dev = 0;
        self.anchor_dev = 0;
        let r = self.draft_round_stages(anchor, crate::dflash2::SLIDING_WINDOW, false, false,
                                        true, crate::dflash2::N_LAYERS, true);
        self.ntot_dev = ntot_save;
        self.anchor_dev = anchor_save;
        r
    }

    /// The kernel-only draft-round sequence (embed gather -> 5 layers -> final norm -> head ->
    /// top16 -> hp -> walk). All inputs must be device-resident: `toks_blk` (anchor+MASK),
    /// `pos_blk` + `cos8`/`sin8` (refreshed by `refresh_block_pos`), the RING (ctx), and the
    /// per-replay device ints (`ntot_dev`, `anchor_dev` — 0 on the eager path). Shared by the
    /// eager path and the CUDA-graph capture (the capture records exactly this sequence).
    /// `graph_mode` = true for the capture/replay (the walk reads the anchor from `anchor_dev`);
    /// the eager path passes the real `anchor` as the launch arg with `anchor_dev` = 0.
    /// `sample` = Some((seeds, temperature)) runs the S5F2 SAMPLED selector walk
    /// (`df2_sel_walk_sample_b` — multinomial at temperature, q_rows + candidate table out)
    /// instead of the greedy chain; the sampled path is EAGER-only (per-step seeds are host
    /// inputs, not graph-replay-safe) — callers must pass None when capturing.
    fn draft_round_kernels(&mut self, anchor: u32, graph_mode: bool,
                           sample: Option<(&[u32], f32)>) -> Result<()> {
        let ntot = self.nprev + BLOCK;
        self.embed_gather();
        // ---- 5-layer backbone (ring attention) ----
        for li in 0..N_LAYERS {
            self.layer_forward(li, ntot, crate::dflash2::SLIDING_WINDOW, false);
        }
        // final norm
        self.rmsnorm(&self.blk.h_final, &self.blk.h, &self.glob.norm, HIDDEN, BLOCK);
        // ---- borrowed head at N=7 on h_final cols 1..7 ----
        self.head_logits();
        // ---- top-16 on the 7 MASK rows ----
        klaunch!(self, "top16_b", (7u32, 1, 1), (256, 1, 1), 0,
            (d(&self.out_vals), d(&self.out_ids), d(&self.logits), VOCAB as i32, 7i32));
        // ---- hidden_projection [256,5120] × h_sel [5120,7] → hp ----
        self.gemm_dsp(&self.hp_bf16, &self.hp_w, d(&self.blk.h_final) + (HIDDEN * 2) as u64, SELECTOR_RANK, HIDDEN);
        klaunch!(self, "bf16tof32", grid(SELECTOR_RANK * 7), (256, 1, 1), 0,
            (d(&self.hp_f32), d(&self.hp_bf16), (SELECTOR_RANK * 7) as i32));
        // ---- the chain: greedy (device unary, no flip) or S5F2 sampled ----
        match sample {
            Some((seeds, temperature)) => {
                // The sampled walk's per-position seeds: host writes them into the round's
                // scratch; the walk draws the multinomial and writes tokens + q_rows + the
                // candidate (token, q) table for the real-q verify's exact residual.
                self.dev.htod_sync_copy_into(seeds, &mut self.walk_seeds).expect("htod walk seeds");
                klaunch!(self, "df2_sel_walk_sample_b", (1u32, 1, 1), (256, 1, 1), 0,
                    (d(&self.walk_out_tok), d(&self.walk_out_q),
                     d(&self.hp_f32), d(&self.out_ids), d(&self.out_vals),
                     d(&self.pred_cb), d(&self.succ_cb),
                     anchor, if graph_mode { self.anchor_dev } else { 0 },
                     d(&self.walk_seeds), temperature, SELECTOR_RANK as i32));
            }
            None => {
                klaunch!(self, "df2_sel_walk_b", (1u32, 1, 1), (256, 1, 1), 0,
                    (d(&self.walk_tokens), d(&self.walk_scores), d(&self.hp_f32), d(&self.out_ids),
                     d(&self.out_vals), d(&self.pred_cb), d(&self.succ_cb),
                     anchor, if graph_mode { self.anchor_dev } else { 0 }, SELECTOR_RANK as i32));
            }
        }
        Ok(())
    }

    /// S5F2 — the SAMPLED draft round (L2's selector path): the S4F round sequence with the
    /// multinomial walk at `temperature`. Returns the drawn draft tokens, their per-position
    /// selector probabilities (`q_rows`), and the full candidate (token, q) table — the inputs
    /// the real-q verify (`verify_forward_sample_rq`) needs for the accept (u·q < p) and the
    /// exact relu(p−q) residual. Eager-only (the greedy round stays graph-captured).
    pub fn draft_round_dev_sample(&mut self, anchor: u32, seeds: &[u32], temperature: f32)
        -> Result<Df2SampleOut> {
        assert_eq!(seeds.len(), 7, "draft_round_dev_sample needs 7 selector seeds");
        let toks: Vec<i32> = [anchor as i32].iter().copied()
            .chain(std::iter::repeat(crate::dflash2::MASK_TOKEN_ID as i32).take(BLOCK - 1)).collect();
        self.dev.htod_sync_copy_into(&toks, &mut self.toks_blk)?;
        let (ntot_save, anchor_save) = (self.ntot_dev, self.anchor_dev);
        self.ntot_dev = 0;
        self.anchor_dev = 0;
        let r = self.draft_round_kernels(anchor, false, Some((seeds, temperature)));
        self.ntot_dev = ntot_save;
        self.anchor_dev = anchor_save;
        r?;
        self.dev.synchronize()?;
        let out_tok: Vec<u32> = self.dev.dtoh_sync_copy(&self.walk_out_tok)?.to_vec();
        let out_q: Vec<f32> = self.dev.dtoh_sync_copy(&self.walk_out_q)?.to_vec();
        let tokens: Vec<u32> = out_tok[..7].to_vec();
        let cand_tok: Vec<u32> = out_tok[7..7 + 7 * 16].to_vec();
        let q_rows: Vec<f32> = out_q[..7].to_vec();
        let cand_q: Vec<f32> = out_q[7..7 + 7 * 16].to_vec();
        Ok(Df2SampleOut { tokens, q_rows, cand_tok, cand_q })
    }

    /// S5F — capture the draft-round kernel sequence as a CUDA graph (the MTP verify-graph
    /// pattern): the captured region is kernel-only (embed -> layers -> norm -> head -> top16 ->
    /// hp -> walk); per-replay inputs (anchor, nprev) are device ints written before each replay.
    /// The attention smem is allocated at the MAX ntot (max_c + BLOCK) so every replay fits.
    /// Returns None when capture is unsupported (the eager path stays).
    pub fn capture_round_graph(&mut self) -> bool {
        use cudarc::driver::sys;
        if self.round_graph.is_some() { return true; }
        // allocate the per-replay device ints + the anchor buffer
        let dev = self.dev.clone();
        let (mut ntot_int, mut anchor_buf) = match (dev.alloc_zeros::<i32>(1), dev.alloc_zeros::<u32>(1)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return false,
        };
        if dev.htod_sync_copy_into(&[(self.max_c + BLOCK) as i32], &mut ntot_int).is_err() { return false; }
        if dev.htod_sync_copy_into(&[0u32], &mut anchor_buf).is_err() { return false; }
        self.ntot_dev = *ntot_int.device_ptr() as u64;
        self.anchor_dev = *anchor_buf.device_ptr() as u64;
        // position the block at max_c so the recorded launches (smem, grid) cover every replay.
        let pos_max: Vec<i32> = (0..BLOCK).map(|b| (self.max_c + b) as i32).collect();
        if dev.htod_sync_copy_into(&pos_max, &mut self.pos_blk).is_err() { return false; }
        self.gather_rope(&self.blk.cos8, &self.blk.sin8, d(&self.pos_blk), BLOCK);
        if dev.synchronize().is_err() { return false; }
        let stream = self.stream.stream;
        // The recorded launch configs must cover EVERY replay. S10R: the attention's dynamic
        // smem is ctx-INDEPENDENT (band_smem(), 8,864 B constant) and the grid is
        // BLOCK*NUM_HEADS — but capture still positions the block at max_c and sets nprev to
        // the max so the recorded packed args mirror the worst case, then restores. (The
        // kernels are only RECORDED here — nothing executes, so the ring contents are
        // irrelevant; ntot_dev makes the replayed ntot device-driven.)
        let nprev_save = self.nprev;
        self.nprev = self.max_c;
        let r = unsafe { sys::cuStreamBeginCapture_v2(stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) };
        if r != sys::CUresult::CUDA_SUCCESS { self.nprev = nprev_save; return false; }
        let ok = self.draft_round_kernels(0, true, None).is_ok();
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r = unsafe { sys::cuStreamEndCapture(stream, &mut graph) };
        self.nprev = nprev_save;
        if !ok || r != sys::CUresult::CUDA_SUCCESS { return false; }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r = unsafe { sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        unsafe { sys::cuGraphDestroy(graph); }
        if r != sys::CUresult::CUDA_SUCCESS { return false; }
        self.round_graph = Some(crate::gpu::CudaGraph::from_exec(exec, stream));
        // keep the device ints alive for the lifetime of the graph (they are captured by addr)
        self.ntot_buf = Some(ntot_int);
        self.anchor_buf = Some(anchor_buf);
        true
    }

    /// S5F — replay the captured draft round with the given anchor. Host writes the per-replay
    /// device inputs (anchor, nprev, block positions) + gathers the block RoPE eagerly, then ONE
    /// graph launch, then the 7-token readback. Bit-identical to the eager path (same kernels,
    /// same order — the graph determinism is asserted by the probe). Falls back to the eager
    /// path when no graph is captured.
    pub fn draft_round_graph(&mut self, anchor: u32) -> Result<Vec<u32>> {
        let Some(g) = self.round_graph.as_ref() else {
            return self.draft_round_dev(anchor);
        };
        let ntot = self.nprev + BLOCK;
        // device inputs for this replay
        let toks: Vec<i32> = [anchor as i32].iter().copied()
            .chain(std::iter::repeat(crate::dflash2::MASK_TOKEN_ID as i32).take(BLOCK - 1)).collect();
        self.dev.htod_sync_copy_into(&toks, &mut self.toks_blk)?;
        let pos: Vec<i32> = (0..BLOCK).map(|b| (self.nprev + b) as i32).collect();
        self.dev.htod_sync_copy_into(&pos, &mut self.pos_blk)?;
        self.dev.htod_sync_copy_into(&[ntot as i32], self.ntot_buf.as_mut().unwrap())?;
        self.dev.htod_sync_copy_into(&[anchor], self.anchor_buf.as_mut().unwrap())?;
        self.gather_rope(&self.blk.cos8, &self.blk.sin8, d(&self.pos_blk), BLOCK);
        // gather_rope and the graph replay use the same blocking stream, so stream ordering is
        // sufficient here. The synchronous token D2H below is the single completion fence.
        g.launch();
        let tokens: Vec<u32> = self.dev.dtoh_sync_copy(&self.walk_tokens)?.to_vec();
        Ok(tokens)
    }

    /// One draft layer on the ring KV (S3F's sequence; the attention is the ring variant).
    /// P2 sharded mode: the q/k/v/o + gate/up/down launches run at the RANK-LOCAL widths and
    /// `o_proj`/`down_proj` produce PARTIAL [5120×B] outputs, landed by an all-reduce at each
    /// sublayer boundary (the trunk's reduce-site pattern — the next sublayer's rmsnorm needs
    /// the full-hidden sum, so the AR cannot be deferred past it). The conv/norm/head path and
    /// every buffer address are IDENTICAL to the unsharded launch sequence, so the captured
    /// graph keeps the same node structure per rank.
    fn layer_forward(&self, li: usize, ntot: usize, window: usize, ctl_dual_write: bool) {
        let l = &self.layers[li];
        let blk = &self.blk;
        let base1_off = (CONV_KERNEL * HIDDEN * 2) as u64;
        // P2: local widths (the full constants when unsharded — the dims below are then
        // byte-identical to the pre-P2 launches).
        let (nq, nkv, ni) = (self.nq_l, self.nkv_l, self.ni_l);
        let sharded = self.ar.is_some();
        // attention sublayer
        self.rmsnorm(&blk.normed, &blk.h, &l.input_ln, HIDDEN, BLOCK);
        self.gemm_dsp(&blk.dyn_attn, &l.attn_kp, d(&blk.normed), 2 * CONV_KERNEL * CONV_GROUPS, HIDDEN);
        self.conv2(&blk.x_conv, &blk.normed, &blk.dyn_attn, d(&l.attn_base), BLOCK, 0);
        self.gemm_weight(&blk.q, &l.q_proj, d(&blk.x_conv), nq * HEAD_DIM, HIDDEN);
        self.rmsnorm_perhead(&blk.q, &l.q_norm, nq, BLOCK);
        self.rope(&blk.q, &blk.cos8, &blk.sin8, nq, BLOCK);
        self.gemm_weight(&blk.k, &l.k_proj, d(&blk.x_conv), nkv * HEAD_DIM, HIDDEN);
        self.rmsnorm_perhead(&blk.k, &l.k_norm, nkv, BLOCK);
        self.rope(&blk.k, &blk.cos8, &blk.sin8, nkv, BLOCK);
        self.gemm_weight(&blk.v, &l.v_proj, d(&blk.x_conv), nkv * HEAD_DIM, HIDDEN);
        // block rows → ring rows [RING, RING+8) (+ the linear control copy at [nprev, nprev+8))
        klaunch!(self, "write_kv_b", grid(BLOCK * nkv * HEAD_DIM), (256, 1, 1), 0,
            (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&blk.k), d(&blk.v),
             d(&self.wrow_blk), RING_STRIDE as i32, nkv as i32, HEAD_DIM as i32,
             BLOCK as i32, d(&blk.slot_ids)));
        if ctl_dual_write && ntot <= RING {
            klaunch!(self, "write_kv_b", grid(BLOCK * nkv * HEAD_DIM), (256, 1, 1), 0,
                (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&blk.k), d(&blk.v),
                 d(&self.wrow_ctl), RING_STRIDE as i32, nkv as i32, HEAD_DIM as i32,
                 BLOCK as i32, d(&blk.slot_ids)));
        }
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let smem = crate::dflash2::band_smem(window, ntot);
        let nh_packed = ((nq << 20) | (HEAD_DIM << 10) | nkv) as i32;
        let packed = (((ntot as u64) << 32) | ((RING as u64) << 16) | RING_STRIDE as u64);
        let window_b = ((window << 4) | BLOCK) as i32;
        klaunch!(self, "gqa_attn_band_ring_b", ((BLOCK * nq) as u32, 1, 1), (HEAD_DIM as u32, 1, 1), smem as u32,
            (d(&blk.attn), d(&blk.q), d(&self.k_ring[li]), d(&self.v_ring[li]),
             d(&self.pos_blk), packed, self.ntot_dev, nh_packed, window_b, fbits(scale)));
        if ctl_dual_write && li == 0 {
            // the ring-vs-linear control: (a) the ring kernel again into a reference buffer
            // (deterministic -> bitwise copy of what just ran), (b) the S3F linear kernel over
            // the same cache's dual-written linear rows.
            klaunch!(self, "gqa_attn_band_ring_b", ((BLOCK * nq) as u32, 1, 1), (HEAD_DIM as u32, 1, 1), smem as u32,
                (d(&self.ctl_attn_ref), d(&blk.q), d(&self.k_ring[li]), d(&self.v_ring[li]),
                 d(&self.pos_blk), packed, self.ntot_dev, nh_packed, window_b, fbits(scale)));
            let lin_stride = (((ntot as u64) << 16) | RING_STRIDE as u64) as i64;
            klaunch!(self, "gqa_attn_band_b", ((BLOCK * nq) as u32, 1, 1), (HEAD_DIM as u32, 1, 1), smem as u32,
                (d(&self.ctl_attn), d(&blk.q), d(&self.k_ring[li]), d(&self.v_ring[li]),
                 d(&self.pos_blk), lin_stride, nh_packed, window_b, fbits(scale)));
        }
        self.gemm_weight(&blk.attn_out, &l.o_proj, d(&blk.attn), HIDDEN, nq * HEAD_DIM);
        if sharded {
            // AR site 1: land the rank's o_proj partial (K-split over q heads) — conv2's
            // channel-local finish and the residual then run on the full sum.
            self.tp_ar_bf16(d(&blk.attn_out), HIDDEN * BLOCK);
        }
        self.conv2(&blk.fin, &blk.attn_out, &blk.dyn_attn, d(&l.attn_base) + base1_off, BLOCK, 1);
        klaunch!(self, "add_residual_b", grid(HIDDEN * BLOCK), (256, 1, 1), 0,
            (d(&blk.h), d(&blk.h), d(&blk.fin), (HIDDEN * BLOCK) as i32));
        // mlp sublayer
        self.rmsnorm(&blk.normed2, &blk.h, &l.post_ln, HIDDEN, BLOCK);
        self.gemm_dsp(&blk.dyn_mlp, &l.mlp_kp, d(&blk.normed2), 2 * CONV_KERNEL * CONV_GROUPS, HIDDEN);
        self.conv2(&blk.x_conv2, &blk.normed2, &blk.dyn_mlp, d(&l.mlp_base), BLOCK, 0);
        self.gemm_weight(&blk.gate, &l.gate_proj, d(&blk.x_conv2), ni, HIDDEN);
        self.gemm_weight(&blk.up, &l.up_proj, d(&blk.x_conv2), ni, HIDDEN);
        klaunch!(self, "silu_mul_b", grid(ni * BLOCK), (256, 1, 1), 0,
            (d(&blk.gate), d(&blk.gate), d(&blk.up), (ni * BLOCK) as i32));
        self.gemm_weight(&blk.mlp_out, &l.down_proj, d(&blk.gate), HIDDEN, ni);
        if sharded {
            // AR site 2: land the rank's down_proj partial (K-split over inter rows).
            self.tp_ar_bf16(d(&blk.mlp_out), HIDDEN * BLOCK);
        }
        self.conv2(&blk.fin2, &blk.mlp_out, &blk.dyn_mlp, d(&l.mlp_base) + base1_off, BLOCK, 1);
        klaunch!(self, "add_residual_b", grid(HIDDEN * BLOCK), (256, 1, 1), 0,
            (d(&blk.h), d(&blk.h), d(&blk.fin2), (HIDDEN * BLOCK) as i32));
    }

    /// Refresh the per-round block position arrays (call when nprev changes, before the
    /// round): block RoPE positions = ABSOLUTE `[nprev, nprev+8)`; the linear-control rows
    /// = `[nprev, nprev+8)`.
    pub fn refresh_block_pos(&mut self) -> Result<()> {
        let n = self.nprev;
        let pos: Vec<i32> = (0..BLOCK).map(|b| (n + b) as i32).collect();
        self.dev.htod_sync_copy_into(&pos, &mut self.pos_blk)?;
        let ctl: Vec<i32> = (0..BLOCK).map(|b| (n + b) as i32).collect();
        self.dev.htod_sync_copy_into(&ctl, &mut self.wrow_ctl)?;
        self.gather_rope(&self.blk.cos8, &self.blk.sin8, d(&self.pos_blk), BLOCK);
        self.dev.synchronize()?;
        Ok(())
    }

    /// The ring-vs-linear attention control (probe-only): re-run LAYER 0's attention with the
    /// S3F `gqa_attn_band_b` kernel over the SAME ring cache (block rows must have been
    /// dual-written) and return the output for a bit-for-bit diff vs the ring kernel's.
    pub fn ctl_linear_attn(&mut self, window: usize) -> Result<Vec<f32>> {
        let ntot = self.nprev + BLOCK;
        assert!(ntot <= RING, "the linear control needs ntot {} <= RING {RING}", ntot);
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let smem = crate::dflash2::band_smem(window, ntot);
        let (nq, nkv) = (self.nq_l, self.nkv_l);
        let nh_packed = ((nq << 20) | (HEAD_DIM << 10) | nkv) as i32;
        let ntot_stride = ((ntot as u64) << 16) | RING_STRIDE as u64;
        let window_b = ((window << 4) | BLOCK) as i32;
        klaunch!(self, "gqa_attn_band_b", ((BLOCK * nq) as u32, 1, 1), (HEAD_DIM as u32, 1, 1), smem as u32,
            (d(&self.blk.attn), d(&self.blk.q), d(&self.k_ring[0]), d(&self.v_ring[0]),
             d(&self.pos_blk), ntot_stride, nh_packed, window_b, fbits(scale)));
        let a: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.attn)?;
        Ok(a.iter().map(|x| x.to_f32()).collect())
    }

    /// Read back the ring k/v rows for one layer (probe diff surfaces), as
    /// `[RING_STRIDE * nkv * hd]` bf16-as-f32.
    /// True per-head ring regions: block rows [C_ring, C_ring+8) of EVERY kv head, plus a
    /// committed-row sample (rows 0..4 and C-4..C) — the write_kv targets and their neighbors.
    pub fn dump_ring_regions(&self, li: usize, c: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        const STRIDE_R: usize = RING_STRIDE; // 2056
        const RING_R: usize = RING;          // 2048
        let nkv = self.nkv_l;
        let mut grab = |buf: &CudaSlice<bf16>| -> Vec<f32> {
            let full: Vec<bf16> = self.dev.dtoh_sync_copy(buf).unwrap();
            let mut out = Vec::with_capacity(nkv * 16 * 128);
            for h in 0..nkv {
                let base = h * STRIDE_R * 128;
                for r in 0..4usize { out.extend(full[base + r * 128..base + r * 128 + 128].iter().map(|x| x.to_f32())); }
                for r in (c - 4)..c { out.extend(full[base + r * 128..base + r * 128 + 128].iter().map(|x| x.to_f32())); }
                for r in RING_R..(RING_R + 8) { out.extend(full[base + r * 128..base + r * 128 + 128].iter().map(|x| x.to_f32())); }
            }
            out
        };
        Ok((grab(&self.k_ring[li]), grab(&self.v_ring[li])))
    }

    /// Re-run ONLY the ring attention (layer 0's exact launch) into ctl_attn_ref — the
    /// determinism discriminator: if this reproduces the dumped attn, the INPUTS changed;
    /// if it reproduces round-0's value, blk.attn was overwritten after attention.
    pub fn ctl_ring_attn(&mut self, window: usize) -> Result<Vec<f32>> {
        let ntot = self.nprev + BLOCK;
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let smem = crate::dflash2::band_smem(window, ntot);
        let (nq, nkv) = (self.nq_l, self.nkv_l);
        let nh_packed = ((nq << 20) | (HEAD_DIM << 10) | nkv) as i32;
        let packed = (((ntot as u64) << 32) | ((RING as u64) << 16) | RING_STRIDE as u64);
        let window_b = ((window << 4) | BLOCK) as i32;
        klaunch!(self, "gqa_attn_band_ring_b", ((BLOCK * nq) as u32, 1, 1), (HEAD_DIM as u32, 1, 1), smem as u32,
            (d(&self.ctl_attn_ref), d(&self.blk.q), d(&self.k_ring[0]), d(&self.v_ring[0]),
             d(&self.pos_blk), packed, self.ntot_dev, nh_packed, window_b, fbits(scale)));
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.ctl_attn_ref)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// The dual-write control's attention snapshot (recomputed right after attention ran,
    /// BEFORE o_proj/MLP — the pre-corruption reference for the determinism hunt).
    pub fn dump_ctl_attn_ref(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.ctl_attn_ref)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// S5F3 dump: read back selected RING rows (k+v, bf16-as-f32) for one layer — the ctx
    /// rows `[c_lo, c_hi)` + the block rows `[RING, RING+8)`. The S1 ring-bisect surface.
    pub fn dump_ring_rows(&self, li: usize, c_lo: usize, c_hi: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let nkv = self.nkv_l;
        let hd = crate::dflash2::HEAD_DIM;
        let rows: Vec<usize> = (c_lo..c_hi.min(RING)).chain(RING..RING + 8).collect();
        // The ring's writers run on the ROUND's stream — the dtoh (NULL stream) must not race
        // them: a device-wide sync first (dump-only path).
        self.dev.synchronize()?;
        let mut grab = |buf: &CudaSlice<bf16>| -> Result<Vec<f32>> {
            let full: Vec<bf16> = self.dev.dtoh_sync_copy(buf)?;
            let mut out = Vec::with_capacity(rows.len() * nkv * hd);
            for h in 0..nkv {
                let base = h * RING_STRIDE * hd;
                for &r in &rows {
                    out.extend(full[base + r * hd..base + r * hd + hd].iter().map(|x| x.to_f32()));
                }
            }
            Ok(out)
        };
        Ok((grab(&self.k_ring[li])?, grab(&self.v_ring[li])?))
    }

    /// S5F3: the round's fc projection output `th` [5120, 8] bf16-as-f32 (the inject's fc
    /// result — the S1 GEMM-bisect surface vs the oracle's tap_project).
        pub fn dump_ring_kv(&self, li: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let k: Vec<bf16> = self.dev.dtoh_sync_copy(&self.k_ring[li])?;
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.v_ring[li])?;
        Ok((k.iter().map(|x| x.to_f32()).collect(), v.iter().map(|x| x.to_f32()).collect()))
    }

    /// The staging buffer read back as f32 (probe).
    pub fn dump_staging(&self) -> Result<Vec<f32>> {
        let s: Vec<bf16> = self.dev.dtoh_sync_copy(&self.staging)?;
        Ok(s.iter().map(|x| x.to_f32()).collect())
    }

    /// S5F probe debug: the post-final-norm block hidden read back (row-major [8][HIDDEN]).
    pub fn dump_h_final(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.h_final)?;
        Ok(v[..HIDDEN * BLOCK].iter().map(|x| x.to_f32()).collect())
    }

    /// The block hidden (pre-final-norm) read back (probe).
    pub fn dump_block_h(&self) -> Result<Vec<f32>> {
        let h: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.h)?;
        Ok(h.iter().map(|x| x.to_f32()).collect())
    }

    /// The block's last-layer k/v scratch BEFORE write_kv (probe determinism localization).
    pub fn dump_block_kv(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let k: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.k)?;
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.v)?;
        Ok((k.iter().map(|x| x.to_f32()).collect(), v.iter().map(|x| x.to_f32()).collect()))
    }

    /// The block's last-layer k/v scratch BEFORE write_kv (probe determinism localization).
    /// The last-run layer's post-input_ln normed buffer (probe determinism localization).
    pub fn dump_block_normed(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.normed)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// The last-run layer's dyn conv coefficients (probe determinism localization).
    pub fn dump_block_dyn(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.dyn_attn)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    pub fn dump_block_q(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.q)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// The block's last-layer conv input scratch (probe determinism localization).
    pub fn dump_block_x(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.x_conv)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// Access the block scratch attention buffer (control diff source).
    pub fn dump_attn_scratch(&self) -> Result<Vec<f32>> {
        let a: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.attn)?;
        Ok(a.iter().map(|x| x.to_f32()).collect())
    }
}
