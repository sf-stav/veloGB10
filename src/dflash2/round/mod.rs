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
//!
//! Map of this directory (`Df2Round`'s impl is cut by phase; every item keeps its `crate::dflash2::round::` path):
//! - `mod.rs` (this file): ring geometry helpers, weight / context / output types, the `klaunch!` macro and the `Df2Round` struct.
//! - `load.rs`: `Df2Round::load*` (plain, pinned, tensor-parallel).
//! - `state.rs`: tap-sink attach / staging sync, ring bookkeeping, reset / rewind and diagnostic probes.
//! - `inject.rs`: chunk upload, GEMM / norm / rope kernel wrappers, incremental ctx injection and `prime_window`.
//! - `draft.rs`: the draft rounds (host, device, sampled, graph) and the 5-layer `layer_forward`.
//! - `dumps.rs`: block-position refresh and the `dump_*` / `ctl_*` diagnostic readbacks.

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::collections::HashMap;
use std::sync::Arc;

use crate::dflash2::capture::{copy_row_into, Df2TapSink};
use crate::dflash2::gpu::{fork_blocking_stream, upload_bf16, upload_fp8, upload_norm, upload_nvfp4, Df2W, GpuGlobal, GpuLayer};

/// Which ring kv projection (the prime-twin helper's selector).
#[derive(Clone, Copy)]
enum Kv { K, V }
use crate::dflash2::band_smem;
use crate::dflash2::{block, levels, conv_group, conv_groups, conv_kernel, head_dim, hidden, inter, n_layers,
                     num_heads, num_kv_heads, rms_eps, selector_rank, tap_concat_dim, tap_layers, vocab};

/// The ring depth (the sliding window; all 5 layers `sliding_attention`).
pub fn ring() -> usize { crate::dflash2::sliding_window() } // 2048 for both shipped artifacts
/// Ring KV stride: ring rows + the 8 block rows above them.
/// Ring KV stride = ring() + the live draft block (2056 at block 8, 2064 at block 16).
/// Runtime, not const: the ring row count is indexed by block position.
#[inline]
pub fn ring_stride() -> usize { ring() + block() }

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
    /// F8-CATCHUP two-stage draft head: coarse NVFP4-mma pass (0.68 GB) -> top-256 shortlist ->
    /// EXACT bf16 re-rank of the shortlist rows against the ORIGINAL hiddens. Downstream
    /// (top16_b/walk) sees bf16-exact logits whenever the shortlist contains the bf16 top-16,
    /// so τ tracks the plain bf16 path while the head read drops 2.54 GB -> ~0.68 GB.
    TwoStage { q: Nvfp4Ptrs, bf16_ptr: u64 },
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
    layers: Vec<GpuLayer>,
    glob: GpuGlobal,
    /// `candidate_selector.hidden_projection` [256, 5120] bf16.
    hp_w: CudaSlice<bf16>,
    /// `predecessor_codebook` / `successor_codebook` [248320, 256] bf16.
    pred_cb: CudaSlice<bf16>,
    succ_cb: CudaSlice<bf16>,
    /// Borrowed trunk tensors (trunk-owned): lm_head + embed, either NVFP4 (MMA-repacked
    /// serving trunk) or BF16 (the plain-BF16 class).
    pub head: Option<BorrowedW>,
    pub embed: Option<BorrowedW>,
    /// DFLASH2-GEN mask override: the artifact's exact BF16 mask row + a zeros token table,
    /// staged when the trunk's quantized mask row measures DAMAGED (rel-L2 > 0.25 at load).
    /// `embed_gather` then overwrites the block's MASK rows (1..block) with this copy.
    mask_ovr_row: Option<CudaSlice<bf16>>,
    mask_ovr_toks: Option<CudaSlice<i32>>,
    cos_table: CudaSlice<f32>,
    sin_table: CudaSlice<f32>,
    /// Ring KV: per layer `[nkv, RING_STRIDE, hd]` bf16.
    k_ring: Vec<CudaSlice<bf16>>,
    v_ring: Vec<CudaSlice<bf16>>,
    /// Committed ctx rows (absolute positions 0..nprev are live in the ring).
    nprev: usize,
    /// DF2_CARRY: which physical prefix-cache slot's committed prefix the ring rows currently
    /// reflect (None = unproven), and the frontier they are valid up to.
    ///
    /// The ring is a MODULO ring (`pos % ring()`, ring() = 2048), unlike `DsparkRound`'s ring which
    /// is LINEAR (`c_ring = max_c`, no wrap) — so "the ring covered this prefix" is not by
    /// itself enough: rows are recycled, and the caller must also know that the rows it still
    /// needs have not been overwritten. `ring_len` is the frontier the caller passes to
    /// `BatchScheduler`'s carry guard, which is where the full condition lives.
    ring_slot: Option<usize>,
    ring_len: usize,
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
    /// PERFFIX1: the ring frontier `pos_blk` / `cos8` / `sin8` / `wrow_ctl` were last refreshed for (None = unknown:
    /// boot, or the round-graph capture parked them at max_c). `refresh_block_pos_if_stale` skips a refresh that
    /// would rewrite the same values (the head drafts, then verifies, at one frontier).
    pos_at: Option<usize>,
    wrow_blk: CudaSlice<i32>,    // [8] block ring write rows [ring()..ring()+8)
    wrow_ctl: CudaSlice<i32>,    // [8] block rows at [nprev..nprev+8) — the linear-control write
    toks_blk: CudaSlice<i32>,    // [8] anchor + 7×MASK
    // head + selector scratch
    logits: CudaSlice<bf16>,     // [vocab(), 7] col-major
    coarse_idx: CudaSlice<u32>,  // [7, 256] the TwoStage shortlist (ids per token column)
    h2d_stage: CudaSlice<i32>,   // [2*block()+2] merged per-replay inputs (toks|pos|ntot|anchor)
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
    /// capture/replay is SPMD-trivial. `--no-df2-inject-graph=1` keeps the eager path.
    pub inject_graphs: std::collections::HashMap<usize, crate::gpu::CudaGraph>,
    /// The per-replay device ints the graph reads (kept alive for the graph's lifetime).
    ntot_buf: Option<CudaSlice<i32>>,
    anchor_buf: Option<CudaSlice<u32>>,
    /// P2: the TP all-reduce context (Some ⇒ the round is SHARDED: per-rank weight slices,
    /// per-head ring KV, and 2 all-reduce sites per layer inside `layer_forward`). None = the
    /// pre-P2 replicated round, byte-identical to the S9F/S10R behavior.
    ar: Option<Df2ArCtx>,
    /// P2: the rank-local GEMM widths (the full constants when unsharded).
    nq_l: usize,      // local q heads  (num_heads() / world)
    nkv_l: usize,     // local kv heads (num_kv_heads() / world)
    ni_l: usize,      // local MLP intermediate (inter() / world)
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

mod load;
mod state;
mod inject;
mod draft;
mod dumps;

