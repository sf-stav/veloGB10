//! S-A3-d — EXL3 whole-model OFFLINE forward (probe-only, --bench-exl3-forward).
//!
//! Assembles the qwen4_exp trunk from the EXL3 pack via the S-A3-b loader and the
//! S-A3-c kernels (dense + new grouped/per-expert variants), plus fp16 glue ports
//! of the engine's hc/GDN/attention/MoE math. No server, no scheduler, no MTP, no
//! TP. PLE (layer-1 n-gram injection) is Phase 0b — NOT in this build; the ladder
//! records the layer-1 PLE delta vs the reference until it lands.
//! Weight path served AS STORED: trellis decoded in-kernel; fp16 tensors bit-copied;
//! bf16 plain tensors value-converted to f16 at load (loss-guarded, one-time).
//!
//! Modes:
//!   generate — prompt prefill + greedy decode (ids + detok text)
//!   ladder   — dump {embed, stratified layer residuals, final hidden, logits}
//!   bench    — decode round timing at width W (+ width sweep), byte model
//!
//! File map (the former single `exl3_forward.rs`; every item keeps its `crate::exl3_forward::` path via `pub use self::<sub>::*`;
//! this file: shared types, launch macros (textual: keep above every `mod`), child-module declarations):
//! - `attn.rs`: decode attention: QSA sparse + dense v3 / split-K steps
//! - `attn_gates.rs`: attention gates, dv3 plan, QSA tunables
//! - `bench.rs`: offline runners: FwdArgs, run, bench_mtp_exl3
//! - `chain.rs`: prefill tunables, moe_pf helpers, exl3_chain_*
//! - `consts.rs`: MTP / copy / KV-format constants, MtpOff, AFlags
//! - `cublaslt.rs`: cuBLASLt plan cache and the Lt matmul variants
//! - `df2.rs`: DFlash2 hooks (df2_round, prime, boot pointers)
//! - `draft.rs`: MTP draft chain: draft_block / pass, mtp_draft_*
//! - `forward_step.rs`: forward_step, kernel sequence, step graph, head_forward
//! - `graph_keys.rs`: CUDA-graph key functions, precapture statics
//! - `hc.rs`: hyper-connection (hc) helpers and methods
//! - `ladder.rs`: ladder diagnostics: LadderCtx, SpDigest, dumps
//! - `lmh.rs`: LM-head GEMM: tunables, selection, launch
//! - `load.rs`: FwdModel::load / load_tp (one item, kept whole)
//! - `moe.rs`: MoE forward: moe, moe_split, moe_inner
//! - `moe_wp20.rs`: WP20 expert selection / launch, histogram dumps
//! - `mpk.rs`: MPK persistent MoE kernels, W4MOE options
//! - `mtp_tap.rs`: MTP tap capture, prime, WP12 cross-checks
//! - `pack.rs`: packed two-lane verify, pack_* methods
//! - `ple.rs`: PLE: row decoders, staging, sidecar, PleDev
//! - `prefill.rs`: prefill driver: prefill_chunk / segs / segs_inner
//! - `prefill_attn.rs`: prefill attention: QSA rows, KV writes, flash launches
//! - `prefill_hc.rs`: prefill hc rows (pre / post / sp)
//! - `prefill_moe.rs`: prefill MLP / MoE and WP18 selection
//! - `prefill_warmup.rs`: prefill_prompt, warmups, lt_prewarm_start, precapture
//! - `probes_lt.rs`: probes: pinned split-point Lt plans, Lt algorithm rows
//! - `probes_pack23.rs`: probes: pack3 / pack2
//! - `probes_packstate.rs`: probes: packstate
//! - `probes_prefillx.rs`: probes: prefill-vs-forward, pack probe entries
//! - `psk.rs`: PSK chain knobs and exl3_chain_pair
//! - `reprime.rs`: MTP head re-prime runtime, head_reprime*
//! - `rounds.rs`: round drivers: mtp_round*, copy_round, mtp_verify
//! - `router_r16.rs`: router entry selection (R16), smem opt-ins, WP10 / PDL
//! - `sampling.rs`: SampParams, RqStage, PenParams, DraftTail
//! - `scratch.rs`: impl Scratch, qsa sizing, host-wait helpers
//! - `slot_state.rs`: sampler setters, slot reset / snapshot / spill
//! - `spec_pass.rs`: SpecPass state, spec_* methods
//! - `tensors.rs`: plain tensor reader, K-split pickers, spill math
//! - `tune_hc.rs`: hc / WP09 / WP11 / W4HC tunables, launches
//! - `verify.rs`: verify round: verify_shadow, verify_kernels, commit, blocks
//! - `vision.rs`: RoPE tables, mrope row gather, slot vision upload
//! - `widths.rs`: trellis widths, wide-arm refusal, PFX1 thresholds
//! - `wp21.rs`: WP21 fused two-quad chain
//! - `wp_levers.rs`: GDN ring / WP12 / WP20 / WP18 lever code
//! - one file per former inline `#[cfg(test)]` module, named after it (21).

use crate::exl3::{Exl3Pack, ModuleMeta};
use crate::gpu::fork_blocking_stream;
use anyhow::{bail, Context, Result};
use cudarc::cublaslt::{result as lt, sys as ltsys};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::f16;
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::Arc;

const MODULE: &str = "exl3_bench";

// TUNE (PLAN/AUTOTUNE_DESIGN.md §4): the tunable registry. Knob reads go through `tune::get`
// (env alias -> tuner overlay -> frozen table -> built-in default == today's behaviour).
pub use crate::exl3_tune as tune;
macro_rules! gblaunch {
    ($l:expr, $name:expr, $grid:expr, $block:expr, $smem:expr, ($($arg:expr),* $(,)?)) => {{
        let f = $l
            .dev
            .get_func("gpu_batch", $name)
            .ok_or_else(|| anyhow::anyhow!("kernel {} missing from gpu_batch", $name))?;
        unsafe {
            f.clone()
                .launch_on_stream($l.stream, LaunchConfig { grid_dim: $grid, block_dim: $block, shared_mem_bytes: $smem }, ($($arg),*))
                .with_context(|| format!("launch {}", $name))
        }
    }};
}

/// TP-4M1: xq_had_suh_rows_live — items (row x 128-block) in flight per warp, and the persistent grid size.
/// XQ_HSL_KERNEL names the U-variant of kernels/exl3_bench.cu (xq_had_suh_rows_live_u{1,2,4}); XQ_HSL_U must match it.
pub(crate) const XQ_HSL_KERNEL: &str = "xq_had_suh_rows_live_u2";
pub(crate) const XQ_HSL_U: usize = 2;
pub(crate) const XQ_HSL_BLOCKS: usize = 288;
/// The persistent grid of the replicated (non-EP) xq_had_suh_rows_live call (all rows live): 576
/// blocks, GitHub andersc's PACK-PF default.
pub(crate) const XQ_HSL_FULL_BLOCKS: usize = 576;

macro_rules! xqlaunch {
    ($l:expr, $name:expr, $grid:expr, $block:expr, $smem:expr, ($($arg:expr),* $(,)?)) => {{
        // A5 D14: programmatic dependent launch (--exl3-pdl=1) through the raw module handle;
        // every exl3_bench entry kernel opens with XQ_PDL_ENTRY (wait -> launch_dependents).
        let pdl_f = if pdl_on() { xq_raw_fn_smem($name, $smem) } else { None };
        if let Some(rf) = pdl_f {
            let (g0, g1, g2) = $grid;
            let (b0, b1, b2) = $block;
            // params MUST be built inline in the launch statement: as_kernel_param() points INTO
            // each argument expression's temporary (the gpu.rs E9 lesson).
            unsafe { launch_pss_raw($l.stream.stream, rf, (g0, g1, g2), (b0, b1, b2), $smem,
                                    &mut [$(($arg).as_kernel_param()),*]) }
                .with_context(|| format!("launch {} (pdl)", $name))
        } else {
        let f = $l
            .dev
            .get_func(MODULE, $name)
            .ok_or_else(|| anyhow::anyhow!("kernel {} missing from {}", $name, MODULE))?;
        unsafe {
            f.clone()
                .launch_on_stream($l.stream, LaunchConfig { grid_dim: $grid, block_dim: $block, shared_mem_bytes: $smem }, ($($arg),*))
                .with_context(|| format!("launch {}", $name))
        }
        }
    }};
}

/// TP-G: xqlaunch! for entries with more parameters than cudarc's 12-tuple launch: always through
/// the raw module handle (PDL attribute when pdl_on(), else a plain cuLaunchKernel).
macro_rules! xqlaunch_raw {
    ($l:expr, $name:expr, $grid:expr, $block:expr, $smem:expr, ($($arg:expr),* $(,)?)) => {{
        let rf = xq_raw_fn_smem($name, $smem)
            .ok_or_else(|| anyhow::anyhow!("{} raw fn unavailable (stale PTX?)", $name))?;
        let (g0, g1, g2) = $grid;
        let (b0, b1, b2) = $block;
        if pdl_on() {
            unsafe { launch_pss_raw($l.stream.stream, rf, (g0, g1, g2), (b0, b1, b2), $smem,
                                    &mut [$(($arg).as_kernel_param()),*]) }
                .with_context(|| format!("launch {} (pdl)", $name))
        } else {
            let r = unsafe {
                cudarc::driver::sys::cuLaunchKernel(rf, g0, g1, g2, b0, b1, b2, $smem, $l.stream.stream,
                    [$(($arg).as_kernel_param()),*].as_mut_ptr(), std::ptr::null_mut())
            };
            if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS { Ok(()) }
            else { Err(anyhow::anyhow!("launch {} ({r:?})", $name)) }
        }
    }};
}

/// TP-4T1: the prefill TAIL-chunk plain fp16 GEMM `out[m,n] = sum_k w[n,k] x[m,k]` (c < recon_min_rows). m <= 16 and
/// K % 8 == 0 take `xq_gemm_f16_tail` (coalesced smem-staged tile, BITWISE equal per output to `xq_gemm_f16`: same
/// ascending-k FFMA chain, same final rounding); anything else keeps the original per-row `xq_gemm_f16` launch.
macro_rules! xq_gemm_f16_small {
    ($l:expr, $out:expr, $w:expr, $x:expr, $m:expr, $n:expr, $k:expr) => {{
        let (m_, n_, k_) = ($m as usize, $n as usize, $k as usize);
        if m_ <= 16 && k_ % 8 == 0 {
            xqlaunch!($l, "xq_gemm_f16_tail", (((n_ as u32) + 31) / 32, ((m_ as u32) + 3) / 4, 1), (128, 1, 1), 0,
                     ($out, $w, $x, m_ as i32, n_ as i32, k_ as i32))
        } else {
            xqlaunch!($l, "xq_gemm_f16", (((n_ as u32) + 127) / 128, m_ as u32, 1), (128, 1, 1), 0,
                     ($out, $w, $x, m_ as i32, n_ as i32, k_ as i32))
        }
    }};
}

/// LANEWIDE-EXACT: xq_gemm_f16_tail at any M (grid (ceil(N/32), ceil(M/4))): per output xq_gemm_f16's exact
/// arithmetic, so a row's result does not depend on M or on the other rows. K % 8 == 0.
macro_rules! xq_gemm_f16_tail {
    ($l:expr, $out:expr, $w:expr, $x:expr, $m:expr, $n:expr, $k:expr) => {{
        let (m_, n_, k_) = ($m as usize, $n as usize, $k as usize);
        anyhow::ensure!(k_ % 8 == 0, "xq_gemm_f16_tail: K {} not a multiple of 8", k_);
        xqlaunch!($l, "xq_gemm_f16_tail", (((n_ as u32) + 31) / 32, ((m_ as u32) + 3) / 4, 1), (128, 1, 1), 0,
                 ($out, $w, $x, m_ as i32, n_ as i32, k_ as i32))
    }};
}

// DHEAD (w2): the draft lm_head diet (src/exl3_forward/dhead.rs) — declared after xqlaunch!.
mod dhead;
// TUNE T0c: the DHEADP registry entries (class D, declared in dhead.rs)
pub(crate) use dhead::{T_DHEADP_OFF, T_DHEADP_CAP, T_SPEC_RATIO_DHEAD};
// W4/SMALL (wave 4): five small-kernel levers (src/exl3_forward/w4s.rs) — after xqlaunch! too.
mod w4s;
use w4s::{w4s_on, W4S_ROUTER};
// WP16: intermediate recurrent checkpoints (src/exl3_forward/wp16.rs) — after xqlaunch! too.
mod wp16;
// COPYPF1: --copy-span-prefill device step (prefill-width copy verify) + --probe-copy-prefill
mod cpv;
pub use cpv::{probe_copy_prefill, CpvBufs, CpvOut};
// LANEWIDE: --probe-lane-wide-cost (step 1 cost shapes) and the many-lane wide verify
mod lwv;
pub use lwv::{probe_lane_wide_cost, probe_lane_wide_pin, probe_lane_wide_xclass, WideBufs, WideLane, WideMs};
// LANEWIDE-DRES: --exl3-dense-resident (kept fp16 dense matrices for the reconstruct arm) + --probe-dense-resident
pub mod dres;
pub use dres::probe_dense_resident;
// PACKPF: --probe-exl3-packpf (packed multi-prompt prefill vs per-prompt prefill, bit for bit + time)
mod packpf;
pub use packpf::probe_pack_pf;
// TP-A: EXL3 tensor parallelism (slice oracle, EP shard/filter/combine/all-reduce, the xtp
// probe and the TP head/node run) — src/exl3_forward/xtp.rs, after xqlaunch! too.
pub mod xtp;
// FP4L: nvidia/Qwen3.8-Flash-Next-FP4-Local weight kinds on this trunk (src/exl3_forward/fp4l_fwd.rs).
mod fp4l_fwd;
/// FP4L: `--probe-fp4l-gpu --model-dir <FP4-Local dir>` (kernel truth on real tensors; PLAN/FP4L_GPU_RUNBOOK.md).
pub fn probe_fp4l_gpu(dir: &str) -> Result<()> { fp4l_fwd::probe_gpu(dir) }
pub use wp16::{absorb_grid, absorb_on, absorb_rmin, plan_grid, merge_cap, merge_grid, merge_grid_cap, probe_wp16, run_grid, tail_ckpt_extra_chunk, tail_ckpt_on, tail_point, wp16_due, wp16_off, wp16_xcheck_on, CkptDiff, TAIL_BACKOFF,
               CkptLayout, CkptStore, RecurCkpt, Wp16ProbeArgs, TAIL_MIN_GAIN, WP16_STRIDE};
pub(crate) use wp16::T_PREFIX_TAIL_CKPT;
// TUNE T0c: the W4/SMALL registry entries (declared next to their launch sites in w4s.rs)
pub(crate) use w4s::{T_W4S_ROUTER, T_W4S_GDN, T_W4S_DATTN, T_W4S_PLE, T_W4S_DRAFT};
/// W4S (router) launch geometry for the binv probe (exl3_bench::probe_router_fold).
pub(crate) const W4S_ROUTER_NT: u32 = w4s::W4R_NT;
pub(crate) fn w4s_router_grid(dev: &Arc<CudaDevice>, ne: usize) -> u32 { w4s::w4r_grid(dev, ne) }
pub(crate) fn w4s_router_smem(m: usize, h: usize) -> usize { w4s::w4r_smem(m, h) }
// W4/DENSE: the dense split-K chains on the LMH streaming design (src/exl3_forward/dense.rs).
pub(crate) mod dense;
// TUNE T1 (PLAN/AUTOTUNE_DESIGN.md §5): the in-round measurement core of `--autotune`. A child
// module (declared after the launch macros) so it drives the served units — verify_shadow /
// verify_kernels / draft_block / forward_step_kernels — without widening their visibility.
#[path = "../exl3_autotune/mod.rs"]
pub mod autotune;
// WP25: the q8 host reference + --probe-exl3-kvq / --probe-exl3-kvnll. A child module so the
// teacher-forced verify step reaches the private verify/prefill internals without widening them.
#[path = "../exl3_kvnll.rs"]
mod kvnll;
pub use kvnll::{probe_kv_nll, probe_kv_q8_selftest, KvNllArgs};
pub(crate) use kvnll::q4k_quant_row;
// S-B9-REL-PPL: `--perplexity` on an EXL3 pack — the served chunked prefill with an all-rows lm_head.
#[path = "../exl3_ppl/mod.rs"]
mod ppl;
pub use ppl::{perplexity_exl3, PplArgs};
// S-B9-REL-QSAREGIME: --probe-exl3-regime (the mixed-regime decode-step instrument; same child-module reason).
#[path = "../exl3_regime.rs"]
mod regime;
pub use regime::{probe_regime, RegimeArgs};

mod attn;
mod attn_gates;
pub use self::attn_gates::*;
mod bench;
pub use self::bench::*;
mod chain;
pub use self::chain::*;
mod consts;
pub use self::consts::*;
mod cublaslt;
pub(crate) use self::cublaslt::*;
mod df2;
mod draft;
mod forward_step;
mod graph_keys;
pub use self::graph_keys::*;
mod hc;
pub use self::hc::*;
mod ladder;
pub use self::ladder::*;
mod lmh;
pub(crate) use self::lmh::*;
mod load;
mod moe;
mod moe_wp20;
mod mpk;
pub(crate) use self::mpk::*;
mod mtp_tap;
mod pack;
pub use self::pack::*;
mod ple;
pub(crate) use self::ple::*;
mod prefill;
mod prefill_attn;
mod prefill_hc;
mod prefill_moe;
mod prefill_warmup;
mod probes_lt;
pub use self::probes_lt::*;
mod probes_pack23;
mod probes_packstate;
mod probes_prefillx;
pub use self::probes_prefillx::*;
mod psk;
pub(crate) use self::psk::*;
mod reprime;
pub(crate) use self::reprime::*;
mod rounds;
mod router_r16;
pub(crate) use self::router_r16::*;
mod sampling;
pub use self::sampling::*;
mod scratch;
pub use self::scratch::*;
mod slot_state;
mod spec_pass;
pub(crate) use self::spec_pass::*;
mod tensors;
pub(crate) use self::tensors::*;
mod tune_hc;
pub(crate) use self::tune_hc::*;
mod verify;
mod vision;
pub(crate) use self::vision::*;
mod widths;
pub use self::widths::*;
mod wp21;
pub(crate) use self::wp21::*;
mod wp_levers;
pub(crate) use self::wp_levers::*;

// ---------------------------------------------------------------------------
// Device model
// ---------------------------------------------------------------------------

struct Quad {
    tr: CudaSlice<u16>,
    suh: CudaSlice<u16>,
    svh: CudaSlice<u16>,
    bits: i32,
    /// Half-bit units (2*K + half): the kernel-side switch index. bits = k2/2 floor.
    k2: i32,
    k: i32,
    n: i32,
    // S-A3-f-f ITEM 2: K-split factor for the m<=16 decode chain (grid (N/128, ks)).
    // 1 = the original single-block-per-tile path (byte-identical contract, e.g. lm_head).
    // Fixed per shape -> m-independent -> batch invariance preserved; the reduction
    // order changes vs ks=1, so those chains moved to the re-baseline protocol (brief §5).
    ks: u32,
    // W3/LMH: the lm_head class (full head + the pruned draft slice). m <= 16 chains of such a
    // Quad take the persistent exl3_lmh_* GEMM (bitwise == exl3_hmma_gemm [+ exl3_had_svh]).
    lmh: bool,
    // Split-K fixup arrival counters, one u32 per 128-column N-tile (max(n/128, 1)), explicitly
    // zeroed at load (AGENTS §2.2) and left at 0 by every launch that arrives on them (the last
    // arrival per tile re-arms): WP21's exl3_hmma_gemm_ks_fx(_x2) (default) and W4/DENSE's
    // exl3_dks_* (opt-in). A chain launch takes ONE of the two paths, both leave the words at 0,
    // and launches of one Quad are stream-ordered — per Quad => distinct counters for every chain
    // in flight (WP27 forks: see the WP21 concurrency note at wp21_on).
    cnt: CudaSlice<u32>,
    // TP-B: ShardSpec::Row — this Quad holds a K (row) slice of the full tensor, so its chain
    // output is a PARTIAL that must be all-reduced (FwdModel::chain_o). Sharding is a tensor
    // property: the forward keys the reduce site off THIS flag, never off `world`.
    rowpar: bool,
    // FP4L (fp4l_fwd.rs): Some = an FP4-Local weight (MXFP8 / f16) — tr/suh/svh are 1-element
    // placeholders and every chain dispatcher branches to fp4l_fwd FIRST. None = a trellis Quad.
    fx: Option<Box<fp4l_fwd::FxW>>,
}

/// Prefix-cache snapshot buffers, laid out like the live per-slot state they copy.
struct PrefixSnap {
    conv: Vec<CudaSlice<f32>>,
    s: Vec<CudaSlice<f32>>,
    ple: CudaSlice<u16>,
    carry: CudaSlice<f32>,
    hist: std::sync::Mutex<Vec<i32>>,
}

struct HcDev {
    norm: CudaSlice<u16>,
    down: CudaSlice<u16>,
    up: CudaSlice<u16>,
    winj: Option<CudaSlice<u16>>,
    /// S-A3-p (--hc-int8, opt-in): int8 copies of down/up for the decode/verify mixer.
    q8: Option<HcQ8>,
    /// FP4L --fp4l-norms f32: the STORED fp32 norm weights (bf16 ones widened exactly) [hc][h] for the
    /// xf_*_f twins; `norm` keeps its f16 copy (diagnostics). None = the f16 norm everywhere.
    norm_f: Option<CudaSlice<f32>>,
    /// FP4L --fp4l-plain exact: down / up / winj hold the STORED bf16 bits — only the xf_*_bf twins and
    /// the stored-bf16 GEMMs may read them (every mixer launch site branches on this). false = f16 bits.
    bf: bool,
}

/// S-A3-p: symmetric int8 mixer weights, one fp32 scale per OUTPUT row along the contracted
/// dim (the reference implementation's EXL3_GR_INT8 recipe): down [lr][rw] -> down_s[lr], up [rw][lr] -> up_s[rw].
struct HcQ8 {
    down_q: CudaSlice<i8>,
    down_s: CudaSlice<f32>,
    up_q: CudaSlice<i8>,
    up_s: CudaSlice<f32>,
    /// WP11 R3: up_q holds the k-major relayout [lr/16][rw][16] (same bytes, permuted) —
    /// consumed ONLY by the WP11 twins (xq_hc_fuse_i8k / xq_hc_fuse_i8_rbk).
    up_kmajor: bool,
    /// WP11 --hc-step-check: the original row-major up_q, kept only for the eager old-path
    /// diff (None otherwise — no extra memory in serving).
    up_q_rm: Option<CudaSlice<i8>>,
}

struct GdnLayer {
    in_qkv: Quad,
    in_z: Quad,
    out_proj: Quad,
    a_w: CudaSlice<u16>,
    b_w: CudaSlice<u16>,
    a_log: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    conv_w: CudaSlice<f32>,
    norm: CudaSlice<u16>,
    /// FP4L: in_proj_a / in_proj_b as exact MXFP8 Quads + their PRIVATE split-K workspace (the a|b branch
    /// runs beside in_qkv in the WP27 fork, so it cannot share chain_ws). None = the plain f16 a_w / b_w.
    ab_fx: Option<Box<(Quad, Quad, CudaSlice<f32>)>>,
}

struct AttnLayer {
    q_proj: Quad, // 2560 -> 12288 b5 (q|gate per head)
    k_proj: Quad, // 2560 -> 512 b5
    v_proj: Quad, // 2560 -> 512 b5
    o_proj: Quad, // 6144 -> 2560 b5
    qnkn: CudaSlice<u16>, // f16 [512] = q_norm | k_norm
    // S-A3-i: the layer's QSA indexer (None = pack has no indexer / dense attention).
    idx: Option<IndexerDev>,
}

/// S-A3-i: QSA indexer device state for one full-attention layer. `qk` is the EXL3
/// 3-bpw index_qk_proj (2560 -> 640: 4 normed+roped q heads | one raw 128-D key);
/// `qn`/`kn` are the indexer layernorms as f32; `params` mirrors gpu.rs's QsaParams
/// block ([cos_t, sin_t, kw, qw] u64 ptrs, eps f32, then hd/heads/rdim/ratio/topk/
/// sel_max/pad i32s — kernels/gpu_batch.cu). The raw-key plane is FwdModel::qsa_keys.
struct IndexerDev {
    qk: Quad,
    qn: CudaSlice<f32>,
    kn: CudaSlice<f32>,
    params: Option<CudaSlice<u8>>,
}

/// EXL27B: the routed experts of a Qwen3.5-class MoE pack. The gate tables live in MoeLayer::gu_tr /
/// suh_gu / svh_gu (stacked per expert, NOT interleaved with up), the down tables in d_tr / suh_d /
/// svh_d; these are the up tables plus the per-layer prefill offsets (e * words: the pack mixes 3-
/// and 4-bit layers, so the scratch's model-wide offset tables cannot address them).
struct SplitGu {
    up_tr: CudaSlice<u16>,
    suh_u: CudaSlice<u16>,
    svh_u: CudaSlice<u16>,
    offs_e: CudaSlice<u64>,
    offs_d: CudaSlice<u64>,
}

struct MoeLayer {
    router: CudaSlice<u16>,
    sg: CudaSlice<u16>,
    gu_tr: CudaSlice<u16>,
    d_tr: CudaSlice<u16>,
    suh_gu: CudaSlice<u16>,
    svh_gu: CudaSlice<u16>,
    suh_d: CudaSlice<u16>,
    svh_d: CudaSlice<u16>,
    sh_gate: Quad,
    sh_up: Quad,
    sh_down: Quad,
    gu_words: u64,
    d_words: u64,
    /// TP2BIT5 #26 Fix 3: THIS layer's per-expert trellis offset tables for the device-routed
    /// prefill arm (identity over the layer's expert space -- local ids under EP). The scratch
    /// tables they replace were built from LAYER 0's width and misaddressed every other width
    /// in a mixed pack (47xK3+1xK4 / 47xK4+1xK5).
    pf_offs_gu: CudaSlice<u64>,
    pf_offs_d: CudaSlice<u64>,
    /// CF-P1g: the experts' trellis bit width K (3 = the 3.05 pack, 4 = the 4.05 pack). The WP20 / A-once fast paths are
    /// K = 3 only so far; every other width runs the K-generic kernels (xq_gemm_grouped_xh, the pf entries).
    bits: usize,
    /// P7-1P5: k2 = 2K (+1 fractional) — the arg exl3_reconstruct/dump8 take.
    k2: usize,
    /// TP-A: expert-parallel shard of this layer's ROUTED experts (None = all experts resident,
    /// the TP=1 layout and the replicated T1 MTP head). Sharding is a tensor property: the forward
    /// keys the EP route filter + partial combine + all-reduce off THIS field, never off `world`.
    ep: Option<xtp::EpShard>,
    /// TP-I: trunk layer index (usize::MAX = the MTP head) — keys the per-layer EP deal and the
    /// --tp-ep-hist routing histogram.
    li: usize,
    /// EXL27B: a Qwen3.5-class MoE pack's routed experts (Qwen3.6-35B-A3B): gate and up are separate
    /// stacked tables with their OWN suh / svh (moe_split / moe_split_prefill). FN: always None.
    split: Option<SplitGu>,
    /// EXL27B: a dense pack's MLP (Qwen3.5-class 27B/9B): gate / up / down are the three plain quads
    /// in sh_gate / sh_up / sh_down — each with its OWN input transform (suh), output scales (svh) and
    /// bit width — computed by dense_mlp / dense_mlp_prefill (copies of the shared-expert chain code);
    /// the routed-expert buffers are 1-element placeholders and no MoE code runs. FN: always false.
    dense: bool,
    /// FP4L: the NVFP4 routed experts of an FP4-Local pack (the trellis tables are 1-element
    /// placeholders; routing, shared expert and combine stay the trunk's). None = trellis experts.
    fx: Option<fp4l_fwd::FxMoe>,
    /// FP4L --fp4l-plain exact: `router` holds the STORED bf16 bits (the router GEMM runs on
    /// xf_bf16w_gemm / xf_bf16_wide and the legacy top-k chain), `sg` holds the xf_sg_gate channel
    /// constants and `sg_bf` the stored bf16 shared_expert_gate. false / None = f16 (trunk kernels).
    router_bf: bool,
    sg_bf: Option<CudaSlice<u16>>,
}

struct PleDev {
    key_w: CudaSlice<u16>,      // [10240, 2560] f16
    value_w: CudaSlice<u16>,    // [2560, 2560] f16
    // WP09: k-major [K/16][N][16] copies for the decode/verify projections (xq_gemm_f16_rows_km,
    // coalesced, bitwise). None when --wp09-off/PLE=0. The row-major pair stays for prefill
    // (cuBLASLt opA=T / xq_gemm_f16) — +65.5 MB.
    key_km: Option<CudaSlice<u16>>,
    value_km: Option<CudaSlice<u16>>,
    norm_key_w: CudaSlice<u16>, // [10240] f16 bits, (1+w)
    norm_query_w: CudaSlice<u16>,
    norm_conv_w: CudaSlice<u16>,
    conv_w: CudaSlice<u16>,     // [10240*4] f16, row-major [d][k]
    shards: Vec<std::fs::File>, // 128 trellis shard files
    // RAM-resident n-gram table (owner flag 2026-09-25; the reference implementation's EXL3_NGRAM_STREAM=0): each
    // shard's data section read once at load. None = per-token pread through the page cache —
    // measured 0.35 ms/round host when the cache holds the working set, 2.0-2.7 ms when GPU
    // allocations crowd it out (262K window: 1% of the 30 GiB table resident).
    // CF-P1d: a OnceLock so `auto` can promote the table to RAM AFTER the boot has allocated every lane's KV,
    // scratch, graphs and the vision tower (FwdModel::ple_promote_auto) — deciding at weight-load time,
    // before any of that exists, let `--max-batch 8` at 131K exhaust the box (mem watchdog at boot).
    ram: std::sync::OnceLock<Vec<Vec<u8>>>,
    shard_data_off: Vec<u64>,   // absolute byte offset of each shard's data section
    rows_per_shard: i64,
    /// CF-P1g: group width K of the n-gram rows (5 = 3.05 pack, 6 = 4.05 pack) and the row size in bytes.
    row_k: usize,
    row_bytes: usize,
    head_offsets: Vec<i64>,     // [16]
    head_vocab_sizes: Vec<i64>, // [16]
    layer_multipliers: [i64; 3],
    head_bias_bits: Vec<u16>,   // f16 [16*160]
    head_bias_f32: Vec<f32>,    // WP05: head_bias_bits widened once at load (f16 -> f32 is exact)
    codebook: Vec<f32>,         // [65536]
    eos: i32,
    /// FP4L: the NVFP4 n-gram table (90-B rows: 80 nibbles + 10 E4M3 scales, one multiplier).
    /// None = the trellis rows above. With Some, `shards` / `shard_data_off` / `row_bytes` describe
    /// the NIBBLE planes and `fx.s_off` the scale planes; the RAM table holds interleaved 90-B rows.
    fx: Option<fp4l_fwd::PleFx>,
    /// FP4L --fp4l-plain exact: key_w / value_w hold the STORED bf16 bits (no k-major copies) and
    /// `conv_f` the bf16 conv taps widened to fp32 (exact). --fp4l-norms f32: `norm_f` = the stored
    /// fp32 [key, query, conv] norm weights. false / None = the f16 copies (trunk kernels).
    bf: bool,
    conv_f: Option<CudaSlice<f32>>,
    norm_f: Option<Box<[CudaSlice<f32>; 3]>>,
}

enum Mixer {
    Gdn(GdnLayer),
    Attn(AttnLayer),
}

struct LayerDev {
    hc_attn: HcDev,
    hc_mlp: HcDev,
    mixer: Mixer,
    moe: MoeLayer,
}

/// S-A3-e M1 — the pack's own nextn draft head (`mtp.*`), loaded as EXL3 through the SAME
/// weight path (trellis decode in-kernel). Mirrors their Qwen4ExpForCausalLMMTP:
///   streams = fc_hidden(grouped_norm(trunk tap)) + bcast(fc_embedding(norm(embed)))
/// (stream_tap=True, pre_fc_norm_* weights, float add — their out_dtype), then ONE
/// hyper-connected attn+MoE block, then the head's own mixer -> hidden -> trunk lm_head.
/// The head keeps its OWN full-window KV per lane (slot = lane, absolute positions), the
/// single-slot-per-lane convention of AGENTS 2.5. The trunk's attention indexer is NOT
/// modeled (same posture as the verified trunk forward); at draft-chain KV sizes (<=9 rows)
/// top-k selection is the full set, so the head is unaffected.
pub struct DraftHead {
    pub hc_attn: HcDev,
    pub hc_mlp: HcDev,
    pub attn: AttnLayer,
    pub moe: MoeLayer,
    fc_hidden: Quad,
    fc_embed: Quad,
    pre_fc_norm_hidden: CudaSlice<u16>,    // [hc*h]
    pre_fc_norm_embedding: CudaSlice<u16>, // [hc*h] (tiled hc x at load)
    /// FP4L --fp4l-norms f32: (pre_fc_norm_hidden, pre_fc_norm_embedding) as their stored bf16 values in fp32
    /// (exact) for xf_hc_norm_f. None = the f16 pair above.
    pre_f: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
    mixer: HcDev,
    kv: CudaSlice<u8>,                     // K1 row-strided bytes, same layout/format as the trunk k_cache
    // Head QSA (2026-09-26): the MTP layer is a full qwen4_exp block WITH its own indexer
    // (mtp.layers.0.self_attn.indexer.*) — the reference implementation's draft runs QSA past the budget. Raw-key
    // plane per slot, bf16 [width][max_pos][hdx], like the trunk's qsa_keys.
    qsa_keys: Option<CudaSlice<u16>>,
    // pooled-key plane (bf16 [width][max_pos/ratio][hdx]); see FwdModel::qsa_pool
    qsa_pool: Option<CudaSlice<u16>>,
}

/// TP-H #4: one rank's vocab shard of the lm_head. `q` is the lm_head class (lmh, ks 1, fused
/// Hadamard) over columns [base, base + n_sh) — bitwise the replicated head's columns.
struct VpHead {
    pub q: Quad,
    /// global id of this shard's column 0 (rank * n_sh)
    pub base: i32,
    pub n_sh: i32,
}

pub struct FwdModel {
    dev: Arc<CudaDevice>,
    pub cfg: crate::qwen::Config,
    /// YARN1: the trunk's rope (per-dim frequencies + mscale) — the source cos_tab, the
    /// prefill host loop, the QSA indexer tables and the vision mrope rows were ALL built from.
    pub(crate) rope: RopeFreqs,
    // TP-B: the TRUNK's per-rank geometry (attention / GDN head counts divided by the world when
    // the trunk is head-sharded; == cfg at TP=1). Every trunk mixer / state site reads its head
    // counts from here; the replicated MTP head keeps `cfg` (T1 contract, design §4.1).
    pub tc: crate::qwen::Config,
    width: usize,
    embed: CudaSlice<u16>,
    /// FP4L --fp4l-plain exact: `embed` holds the STORED bf16 bits (xf_embed_resid_bf; every embedding
    /// gather site branches on this). false = f16 bits.
    embed_bf: bool,
    /// FP4L precision context (None = not an FP4-Local pack, or every lever at its f16 end).
    fxc: Option<fp4l_fwd::FxCtx>,
    layers: Vec<LayerDev>,
    hc_mixer: HcDev,
    lm_head: Quad,
    // S-A3-f-d: pruned draft lm_head (reference patch 0002, EXL3_MTP_HEAD_N, default
    // 65536): an EXL3 column slice of the shared head — draft-only; verification
    // always uses the FULL head, so an out-of-slice draft is a rejection, not an
    // error, and accepted output bits are unchanged (lossless by construction).
    lm_head_draft: Option<Quad>,
    /// FP4L --fp4l-draft-head screen: the int4 screen of lm_head_draft's rows (draft-only; TP = 1)
    fx_dhs: Option<fp4l_fwd::FxDhs>,
    // TP-H #4 (--tp-vp-head): this rank's column shard of the lm_head (world 2) — the vocab-
    // parallel greedy tail (xtp.rs vp_greedy_tail). None = the replicated head everywhere.
    lm_head_vp: Option<VpHead>,
    // DHEAD (w2): B-bit screen + exact rescore of lm_head_draft for the graphed m = 1 draft
    // passes (draft-only; dhead.rs). None = the old head (--dhead-off=1 / no slice).
    dhead: Option<dhead::DHead>,
    conv_state: Vec<CudaSlice<f32>>,
    s_state: Vec<CudaSlice<f32>>,
    // W4S (gdn): device table [S_0..S_{L-1}, conv_0..conv_{L-1}] of the live GDN state planes for
    // the one-launch verify commit (xq_gdn_commit_all_w4); fixed for the model's lifetime.
    w4s_lptr: CudaSlice<u64>,
    // K1 (PLAN/KV_CACHE_FORMATS.md): row-strided BYTES [W][K|V][nkv][max_pos] rows of
    // kv_rowbytes(fmt, hd); the format rides the top 4 bits of the kernels' max_pos (mpf()).
    k_cache: Vec<CudaSlice<u8>>,
    /// KVPOOL (fork): the shared physical pool behind the position planes (None = stock slabs), and per slot the
    /// rows currently mapped (the `vmm_ensure` fast path).
    vmm: Option<std::sync::Mutex<crate::kv_vmm::KvVmm>>,
    vmm_rows: Vec<std::sync::atomic::AtomicUsize>,
    /// KVPOOL: per slot, the rows of the POOLED-key planes mapped (>= vmm_rows: a multi-lane scorer reads a short
    /// lane's pooled blocks out to the longest lane of its group — `vmm_ensure_pooled`).
    vmm_pool_rows: Vec<std::sync::atomic::AtomicUsize>,
    kv_fmt: u32,
    max_pos: usize,
    pub bytes_per_round: u64,
    ple: Option<PleDev>,
    ple_state: CudaSlice<u16>,              // [W*10240*9] f16 conv state (rolled per step)
    ple_hist: std::sync::Mutex<Vec<i32>>, // [W*2] ring: last two ids per lane
    /// The PLE history fill / reset token (248046 for trellis packs; config.eos_token_id for FP4L).
    ple_eos: i32,
    // Prefix cache (2026-09-26): per-slot snapshot of the RECURRENT state at a prefill checkpoint
    // (everything positional — KV, QSA key/pooled planes, head KV — stays valid for positions
    // below it). Allocated only when [exl3-prefix]=1 (the server's --prefix-cache).
    snap: Option<PrefixSnap>,
    pub mtp: Option<DraftHead>,
    // S-A3-f Item 1: ONE blocking compute stream for the model's lifetime (the per-step
    // fork/destroy of the same kind of stream was pure host overhead), and the full
    // host-computed RoPE table [max_pos][2*rdim] (the per-step powf/cos/sin loop, gone).
    stream: CudaStream,
    cos_tab: CudaSlice<f32>,
    // VIS-2: per-slot mrope for image requests, read by xq_cos_gather_v at every main-attention RoPE
    // gather: rope_map [width][max_pos] i32 = the cos row of KV position p inside the slot's prompt
    // (p < rope_plen[slot]); past it the row is p + rope_delta[slot]; rows >= max_pos index mrope_tab
    // [width * VIS_MROPE_ROWS][2*rdim] (the slot's image-token rows, host-built per request).
    // rope_plen = rope_delta = 0 (every text-only request) reads cos_tab[p] exactly as before.
    rope_map: CudaSlice<i32>,
    rope_plen: CudaSlice<i32>,
    rope_delta: CudaSlice<i32>,
    mrope_tab: CudaSlice<f32>,
    // VIS-3: the indexer's per-slot image rows (qsa_cos/qsa_sin layout), the QsaMrope block the
    // QsaParams point at (kept alive here), and whether the indexer table carries both rope halves.
    qm_cos: CudaSlice<f32>,
    qm_sin: CudaSlice<f32>,
    #[allow(dead_code)]
    qsa_mrope: CudaSlice<u8>,
    qsa_full: bool,
    // WP05: cos_tab reproduced the prefill's host RoPE loop bit-for-bit at load (sampled rows),
    // so the prefill prologue may gather from it (xq_cos_gather) instead of recomputing.
    cos_gather_ok: bool,
    // S-A3-i: QSA state — per-full-attn-layer raw-key planes [width][max_pos][hdx]
    // (bf16 bits in u16) and the deinterleaved indexer rope tables [max_pos][rdim]
    // the bf16 selection kernels read through QsaParams (cos_tab is interleaved).
    qsa_keys: Vec<CudaSlice<u16>>,
    // Pooled block-key planes (2026-09-26), one per trunk attention layer (qsa_keys' indexing):
    // bf16 [width][max_pos/ratio][hdx] = qsa_block_key of each block, maintained incrementally so
    // scoring reads one key per block (exact: qsa_block_key's outputs are bf16-rounded). Empty =
    // off (--qsa-pool=0, or the gpu_batch module lacks the pool kernels) -> re-pooling scorer.
    qsa_pool: Vec<CudaSlice<u16>>,
    qsa_cos: CudaSlice<f32>,
    qsa_sin: CudaSlice<f32>,
    // S-A3-f Item 1b: one captured decode-step graph per width (replay = one launch).
    graphs: std::sync::Mutex<std::collections::HashMap<usize, crate::gpu::CudaGraph>>,
    // S-A3-f-f ITEM 1: total co-resident block capacity for xq_hc_fuse at block=256,
    // from cuOccupancyMaxActiveBlocksPerMultiprocessor x SM count (0 = probe failed or
    // kernel missing -> fused path disabled for ALL m; the old 3-kernel path is used
    // consistently, so batch invariance is never split across two hc paths).
    hc_fuse_cap: u32,
    // WP11: co-resident capacities (blocks, 256 t) of the k-major twins (xq_hc_fuse_i8k,
    // xq_hc_fuse_i8_rbk); (0, _) => no k-major relayout at load => old kernels throughout.
    wp11_cap: (u32, u32),
    // HC (w4): grid (CTAs) of xq_hc_w4 = clamp(--w4hc-g or 96, ceil(h/32), co-residency x 5/6);
    // 0 = the W4 mixer is off for ALL m (escape, shape outside its contract, or the probe failed).
    // TUNE T0c: the load-time grid (the availability flag); a launch recomputes it from w4hc_cap
    // with the width's w4hc.g (w4hc_grid_now) — equal to this one at the default.
    w4hc_grid: u32,
    // TUNE T0c: xq_hc_w4's measured co-residency (blocks, 256 t, carveout max); 0 = not probed.
    w4hc_cap: u32,
    // S-A3-f-g: rc_of_t LUT (tmap inverse, 256 B) for the reconstruct kernel
    // (exl3_decode_dump) of the prefill reconstruct→cuBLASLt dense path.
    rc_lut: CudaSlice<u8>,
    // S-A3-o diagnostic (--exl3-esel-hist=1): per-(m, live-expert-count) histogram of
    // every MoE call, dumped at drop. Off by default (no launch, no buffer).
    esel_hist: Option<CudaSlice<i32>>,
    // TP-I #6 diagnostic (--tp-ep-hist=<path>): per-(layer, expert) routing histogram of every
    // decode/verify MoE call, [layers][2][ne] (live-per-call, row picks); dumped as JSON to <path>
    // by the scheduler at every request finish (cumulative). Off by default (no launch, no buffer).
    expert_hist: Option<(CudaSlice<i32>, String)>,
    /// CF-P1e step 0 diagnostic (--exl3-route-log): (record ring, counter, dir, file sequence).
    route_log: Option<(CudaSlice<i32>, CudaSlice<u32>, String, std::sync::atomic::AtomicUsize)>,
    // S-A3-r diagnostic (--qsa-dump-layer=k): selection lists + block scores of QSA layer k,
    // decode ([sel_max] + [nblk]) and prefill (all chunk rows), copied after each top-k.
    qsa_dbg: Option<(usize, CudaSlice<i32>, CudaSlice<f32>, CudaSlice<i32>, CudaSlice<f32>)>,
    // S-A3-s diagnostic (--qsa-rowdump-layer=k): QSA layer k's selection list for EVERY sparse
    // row, indexed by absolute position (< QSA_ROWDUMP_CAP) — decode (sel, psel) and prefill (sel, psel).
    qsa_rowdbg: Option<(usize, CudaSlice<i32>, CudaSlice<i32>, CudaSlice<i32>, CudaSlice<i32>)>,
    // TP-A: the tensor-parallel attachment (rank, world, the doorbell ctx, the fp32 partial
    // buffer). None = TP=1. Reduce sites key off the TENSOR (MoeLayer::ep), this only carries
    // the transport.
    tp: Option<xtp::TpState>,
    // EXL27C-DF2 (fwd): the DFlash2 draft-round capture state - DEFAULT OFF (all None; every
    // hook below is a dead `Option` check: zero launches, zero host work). `set_df2_capture`
    // arms the sinks (Arc-shared with the Df2Round the serve agent owns; their staging
    // addresses are fixed for the process lifetime, so the taps are CUDA-graph-safe).
    df2_sink: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
    df2_prime: Option<std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>>,
    // The boot-time DF2 head ([vocab, hidden] bf16 row-major - the dequantized + Hadamard-
    // folded lm_head quad) and the bf16 twin of the f16 embed staging (the DF2 round's
    // embed_gather_b / gemm_binv_b read bf16; the EXL3 forward stages f16). Both are built
    // lazily ONCE by `df2_boot_ptrs` - which takes &self per the interface contract, hence
    // the interior mutability; the locks are only ever taken at boot.
    // (Type deviation from the brief's plain `Option<...>`: documented in the report.)
    df2_head_bf16: std::sync::Mutex<Option<cudarc::driver::CudaSlice<half::bf16>>>,
    df2_embed_bf16: std::sync::Mutex<Option<cudarc::driver::CudaSlice<half::bf16>>>,
    /// S-EXL3-DF2Q: the quantized (NVFP4-mma) twin of `df2_head_bf16` — the coarse stage of the
    /// two-stage df2 head (gpu.rs `build_df2_q_head` recipe, ported 2026-10-18). Plain bf16
    /// borrow costs ~126 ms/round through `gemm_binv_b`; coarse-NVFP4 + exact bf16 re-rank ~3 ms.
    df2_q_head: std::sync::OnceLock<Option<(cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<f32>)>>,
}
// CUstream is an opaque handle; the stream is created by this model, used only from
// the owning scheduler thread, and destroyed with the device. Same contract as
// `unsafe impl Send for GpuModel` (gpu.rs) — the raw pointer itself crosses no thread.
unsafe impl Send for FwdModel {}
unsafe impl Sync for FwdModel {}

impl Drop for FwdModel {
    fn drop(&mut self) {
        // Destroy graph execs while the CUDA context is guaranteed alive: `dev` is an
        // Arc field still held during drop(), whereas CudaGraph holds a bare
        // CUgraphExec with no context reference — dropping it after context teardown
        // segfaulted at shutdown (observed 2026-09-21). Fields drop AFTER this body.
        self.graphs.lock().unwrap().clear();
        self.dump_esel_hist();
        // KVPOOL: the position planes are VMM ranges, not cuMemAlloc blocks — CudaSlice::drop would cuMemFree them.
        // Leak the slices; KvVmm::drop (a later field) unmaps, frees the ranges and releases the pool.
        if self.vmm.is_some() {
            let _ = self.dev.synchronize();
            let _ = unsafe { cudarc::driver::result::stream::synchronize(self.stream.stream) };
            for s in std::mem::take(&mut self.k_cache) { let _ = s.leak(); }
            for s in std::mem::take(&mut self.qsa_keys) { let _ = s.leak(); }
            for s in std::mem::take(&mut self.qsa_pool) { let _ = s.leak(); }
            if let Some(head) = self.mtp.as_mut() {
                if let Ok(z) = self.dev.null::<u8>() { let _ = std::mem::replace(&mut head.kv, z).leak(); }
                if let Some(k) = head.qsa_keys.take() { let _ = k.leak(); }
                if let Some(k) = head.qsa_pool.take() { let _ = k.leak(); }
            }
        }
    }
}

/// KVPOOL: one u8 position plane — a VMM range when the pool is on, else the stock allocation (not zeroed either way:
/// `alloc_zeros` is cuMemAllocAsync here, and the KV planes are write-before-read per position).
fn vmm_plane_u8(dev: &Arc<CudaDevice>, vmm: &mut Option<crate::kv_vmm::KvVmm>, len: usize) -> Result<CudaSlice<u8>> {
    match vmm {
        Some(v) if len > 0 => { let base = v.reserve(len, false)?; Ok(unsafe { dev.upgrade_device_ptr::<u8>(base, len) }) }
        _ => Ok(dev.alloc_zeros::<u8>(len)?),
    }
}
/// KVPOOL: one bf16 indexer-key plane, really zero (the stock boot uploads zeros: `qsa_pool_update_b` reads a whole
/// `ratio`-row block, rows past the lane's position included). Under VMM every freshly mapped page is zero-filled.
fn vmm_plane_u16z(dev: &Arc<CudaDevice>, vmm: &mut Option<crate::kv_vmm::KvVmm>, len: usize) -> Result<CudaSlice<u16>> {
    match vmm {
        Some(v) if len > 0 => { let base = v.reserve(len * 2, true)?; Ok(unsafe { dev.upgrade_device_ptr::<u16>(base, len) }) }
        _ => Ok(dev.htod_sync_copy(&vec![0u16; len])?),
    }
}
const ESEL_HIST_W: usize = 161;
/// VIS-2: image-token RoPE rows per slot (the most image tokens one request may carry).
pub const VIS_MROPE_ROWS: usize = 16384;
const QSA_ROWDUMP_CAP: usize = 4096;

pub struct Scratch {
    resid: CudaSlice<f32>,
    hn: CudaSlice<u16>,
    dd: CudaSlice<u16>,
    uu: CudaSlice<u16>,
    inj: CudaSlice<f32>,
    x: CudaSlice<u16>,
    chain_xh: CudaSlice<u16>,
    chain_yraw: CudaSlice<u16>,
    qkv: CudaSlice<u16>,
    qg: CudaSlice<u16>,
    kp: CudaSlice<u16>,
    vp: CudaSlice<u16>,
    zbuf: CudaSlice<u16>,
    a_out: CudaSlice<u16>,
    b_out: CudaSlice<u16>,
    core: CudaSlice<u16>,
    normed: CudaSlice<u16>,
    mixer_out: CudaSlice<u16>,
    logits_r: CudaSlice<f32>,
    // S-A3-f-d step 0 lever 1: router K-split fp32 partials [ks][m][ne].
    router_ws: CudaSlice<f32>,
    route_cnt: CudaSlice<i32>, // S-A3-z: xq_router_topk_route last-block counter (self re-arming)
    fold_cnt: CudaSlice<i32>,  // WP10: xq_router_fold last-block counter (self re-arming)
    ids: CudaSlice<i32>,
    wts: CudaSlice<f32>,
    slotmap: CudaSlice<i32>,
    offs_gu: CudaSlice<u64>,
    offs_d: CudaSlice<u64>,
    idxmap: CudaSlice<i32>,
    esel: CudaSlice<i32>, // S-A3-f Item 2: device live-expert count ([1])
    xh_e: CudaSlice<u16>,
    ygu_raw: CudaSlice<u16>,
    ygu: CudaSlice<u16>,
    din: CudaSlice<u16>,
    xhd: CudaSlice<u16>,
    yd_raw: CudaSlice<u16>,
    yd: CudaSlice<u16>,
    sh1: CudaSlice<u16>,
    sh2: CudaSlice<u16>,
    ysh_raw: CudaSlice<u16>,
    ysh: CudaSlice<u16>,
    moe_out: CudaSlice<u16>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    toks: CudaSlice<i32>,
    pos: CudaSlice<i32>,
    slots: CudaSlice<i32>,
    // S-A3-i QSA decode/verify scratch (planes are model-owned; these are per-row pipes).
    qsa_qk: CudaSlice<u16>,    // [w][640] f16 indexer projection (chain output)
    qsa_qkb: CudaSlice<u16>,   // [w][640] bf16 (selection kernels are bf16)
    qsa_scores: CudaSlice<f32>,// [w][nblk_stride]
    qsa_sel: CudaSlice<i32>,   // [w][sel_max] ascending selection lists
    qsa_psel: CudaSlice<i32>,  // [w] last-valid index per row
    slot_ids: CudaSlice<i32>,  // [w] deinterleaved slots for the bf16 selection kernels
    qstage: CudaSlice<f32>,    // [w][nh][hd] normed+roped q (f32)
    qsa_pm: CudaSlice<f32>,    // [w*nh*QSA_SPLITS]
    qsa_pl: CudaSlice<f32>,
    qsa_pacc: CudaSlice<f32>,  // [w*nh*QSA_SPLITS*hd]
    logits: CudaSlice<u16>,
    argmax: CudaSlice<i32>,
    // WP09: xq_argmax_rows cross-block keys [w][XQ_AM_NB_MAX] + per-row last-block counters [w]
    // (self re-arming: ONE real zero at allocation, every launch leaves them at 0).
    am_part: CudaSlice<u64>,
    am_cnt: CudaSlice<u32>,
    // TP-H #4: the vocab-parallel greedy tail's row keys (own shard key -> merged key), u64 x 16
    // (the doorbell payload; padding rows stay zero on both ranks).
    vp_keys: CudaSlice<u64>,
    // TP-H2 (--tp-vp-sampled): the shard GEMM's rows [w][n_sh] (f16) that the row all-gather ships;
    // fully written before read. n_sh = 0 (no shard: TP=1) -> a 1-element placeholder, never read.
    vp_rows: CudaSlice<u16>,
    ple_emb: CudaSlice<u16>,
    ple_key16: CudaSlice<u16>,
    ple_val: CudaSlice<u16>,
    ple_kn: CudaSlice<f32>,
    ple_qn: CudaSlice<f32>,
    ple_gated: CudaSlice<u16>,
    ple_normed: CudaSlice<u16>,
    // S-A3-f-d: chain-verify round buffers.
    // d_dev: drafted ids [8]; accept_out: single readback [8] (emitted, -1 pad);
    // acc2: {slot, a} device commit key; taps_keep: resid snapshot for the next
    // round's draft-extend tap ("tap-before-token" pairing).
    d_dev: CudaSlice<i32>,
    accept_out: CudaSlice<i32>,
    acc2: CudaSlice<i32>,
    // PACK1 LEG 1 (S-B9-CF-PACK1): packed-verify metadata + packed-only buffers.
    // seg: k_seg x {slot, row_start, len, flags}; acc_seg: {slot, a} x k_seg (device
    // commit keys, xq_accept_seg). gdn_ring_p: the packed ring, 16 rows per GDN layer
    // (the lone path keeps its MTP_MAX_K+1-deep plane stride; offsets unchanged).
    seg: CudaSlice<i32>,
    acc_seg: CudaSlice<i32>,
    gdn_ring_p: CudaSlice<f32>,
    taps_keep: CudaSlice<f32>,
    // Verify shadow states (one slot): GDN recurrent + conv ring per layer, PLE ring.
    s_state_sh: Vec<CudaSlice<f32>>,
    conv_sh: Vec<CudaSlice<f32>>,
    ple_ring_sh: CudaSlice<u16>,
    // S-A3-f-d Item 1b: raw (pre-conv) qkv rows per GDN layer, saved during the
    // verify chunk — the conv commit must replay the state window with the RAW
    // projections (the chunk overwrites qkv in place with silu outputs).
    qkv_save: CudaSlice<u16>,
    // S-A3-f-d Item 1b FIX-2: per-layer saves for the GDN commit replay —
    // verify_commit runs as a separate pass where the shared qkv/a_out/b_out
    // hold only the LAST GDN layer's rows.
    qkv_post: CudaSlice<u16>, // post-conv qkv rows (xq_gdn_token's q/k/v input)
    a_save: CudaSlice<u16>,   // gate a rows [n_gdn, SAVE_PLANE_ROWS, nh]
    b_save: CudaSlice<u16>,   // gate b rows [n_gdn, SAVE_PLANE_ROWS, nh]
    // S-A3-o G1: per-token rank-1 GDN update factors written by the verify chunk
    // [n_gdn, MTP_MAX_K+1, nh, XQ_GDN_RS] f32 (delta | krow | g); read by the ring commit.
    gdn_ring: CudaSlice<f32>,
    // S-A3-p: per-round draft-chain inputs {b, p, slot, 0} (the graphed chain reads these)
    draft_meta: CudaSlice<i32>,
    // REPRIME (w2): runtime of the fixed-width device re-prime (meta, pinned accept readback +
    // event, the pending hand-off to head_reprime). None = eager re-prime (allocation failed).
    rp: Option<RpRt>,
    // API-parity G2: per-row sampler params [w][8] read by xq_sample_rows (all-zero rows =
    // greedy: the argmax id is kept). samp_live = host shadow "some row may be non-zero",
    // so all-greedy traffic never pays an upload.
    samp: CudaSlice<u32>,
    samp_live: bool,
    // WP15: per-row penalty params [w][8] read by xq_pen_rows (row 0 also by the draft mirror
    // xq_pen_draft). pen_live = host shadow "some staged row is penalized"; it is part of every
    // decode / verify / draft graph key, so unpenalized traffic replays graphs with no penalty
    // node at all. pen_hist = per-slot token ring [w][PEN_RING] (index = position & (PEN_RING-1)):
    // seeded from the prompt tail at admit, then committed on device by xq_pen_rows.
    // pen_draft_i = the draft pass whose logits the next head_launches penalizes (None = none).
    // DHEADP (w3): pen_mono = every staged row's params are monotone (PenParams::monotone): a
    // penalized draft then takes DHEAD's screen + bound-expanded rescore (dhead.rs); else the old
    // head. Part of the draft graph keys (pen_draft_key).
    pen: CudaSlice<u32>,
    pen_live: bool,
    pen_mono: bool,
    // W2-EXL3: per-row allowed-token bitsets [w][vmask_words(V)] + per-row flags [w]
    // read by xq_vmask_rows (JSON-schema constrained decoding). vmask_live = host shadow "the
    // NEXT forward_step masks some row"; ONE-SHOT: forward_step clears it after the step, so a
    // later caller that did not stage a mask can never inherit one. Part of the decode graph key
    // (bit 47), so unmasked steps replay graphs with no mask node at all.
    vmask: CudaSlice<u32>,
    vmask_flags: CudaSlice<i32>,
    vmask_live: bool,
    /// Logprobs at TP (owner 2026-11-02): a lane of this step reports log probabilities, so the LM-head tail must
    /// leave FULL logits rows (the TP row gather / the replicated head, as for a sampled row) instead of the
    /// vocabulary-parallel greedy key exchange. Set per step by the scheduler from its lanes (identical on every
    /// rank: the lanes are mirrored). Part of the graph key through `vp_key`. Always false at TP=1 (no vp tail).
    pub(crate) lp_live: bool,
    pen_hist: CudaSlice<i32>,
    pen_draft_i: Option<usize>,
    // API-parity DDS: per-draft-pass confidence (the draft head's max logit) [MTP_MAX_K + 1]
    dconf: CudaSlice<f32>,
    // Long-context decode: the QSA block-scoring grid for the NEXT decode/verify capture or
    // launch, in blocks (a power-of-two bucket >= the blocks any row of the step can see; 0 =
    // the full window). Part of the graph key, so one graph serves a whole context bucket.
    qsa_grid_nblk: usize,
    // A5 D5: draft-head split-K attention partials [w][nh][DRAFT_ATTN_SPLITS](+[hd])
    dpm: CudaSlice<f32>,
    dpl: CudaSlice<f32>,
    dpacc: CudaSlice<f32>,
    gdn_chk: CudaSlice<f32>, // ring-check copy of one state plane (1 elem unless checking)
    // MTP draft-head scratch (S-A3-e): entry chains run at m' = m*hc rows.
    hcx: CudaSlice<u16>,   // chain_xh at m*hc rows (k = h)
    hcy: CudaSlice<u16>,   // chain_yraw at m*hc rows (n = h)
    hf: CudaSlice<u16>,    // fc_hidden output [m*hc, h]
    he32: CudaSlice<f32>,  // embed gather (f32, broadcast rows) [m, hc, h]
    // S-A3-f-f: fused hc-mixer barrier counter (self-resetting; ONE real zero at
    // allocation, then every launch leaves it at 0 — no memset node in the graph).
    hc_bar: CudaSlice<i32>,
    // S-A3-f-f ITEM 2: split-K fp32 partials [ks][m][n] (fully written before read).
    chain_ws: CudaSlice<f32>,
    // S-A3-o C1: second member's buffers for a paired chain launch (exl3_chain_pair).
    pair_xh: CudaSlice<u16>,
    pair_yraw: CudaSlice<u16>,
    pair_ws: CudaSlice<f32>,
    // WP05: pinned staging for the DDS round (per-pass readback, PLE rows, verify uploads).
    // None (allocation failed) = the pre-WP05 pageable path.
    wp05: Option<Wp05Stage>,
    // A5 WP20: MoE epilogue last-arrival counters [emax * (mi/128) gate/up pairs | h/128 down
    // tiles] (self re-arming: ONE real zero here, every launch leaves them 0 — no memset node)
    // and the per-row shared-expert gate sigmoid [w] (gate/up kernel -> down combine).
    moe_cnt: CudaSlice<u32>,
    moe_sgv: CudaSlice<f32>,
    // WP13 dense attention v3: raw-score plane [w][nh][DV3_TS], chunk maxima of
    // fl(raw*scale) [w][nh][DV3_NCH], XCHECK side output [w][nh*hd] + counters [4].
    dv3_s: CudaSlice<f32>,
    dv3_pm: CudaSlice<f32>,
    dv3_xc: CudaSlice<u16>,
    dv3_xst: CudaSlice<i32>,
    // WP27: the GDN in_z chain's own split-K scratch (xh [w, h] / yraw [w, value_dim] / ws
    // [ks, w, value_dim]) so it can be a sibling branch of the in_qkv chain (which owns chain_*)
    // in a captured graph; and the graph class of the body being launched (exl3_wp27::CLS_*,
    // set at the top of forward_step_kernels / verify_kernels / head_launches).
    z_xh: CudaSlice<u16>,
    z_yraw: CudaSlice<u16>,
    z_ws: CudaSlice<f32>,
    wp27_cls: u8,
    // TP-C: this Scratch's graph-key id (sk): graphs are keyed by (width, ..., scratch)
    gid: u32,
    // WIDEVERIFY1 refinement 2: rows of ONE lone verify this scratch's per-round planes hold — the
    // gdn_ring plane stride (rows per GDN layer), the taps_keep window (rows per slot), the accept /
    // REPRIME readback words. MTP_MAX_K + 1 (= every neural round) unless the server armed the wide
    // copy verify (COPY_VROWS); a default scratch is byte-for-byte the pre-refinement layout.
    pub(crate) vrows: usize,
    // WP24 (real-q speculative sampling): rq_list [lane_pack::rq_list_rows(w) * RQ_ROW] u32 = per
    // draft pass i its top-32 candidate ids | logits (f32) | q' (f32, the exact law the pass
    // sampled from; 0 = outside it), written by xq_rq_draft; rq_part / rq_cnt = its cross-block
    // keys + self re-arming last-block counter (one real zero at allocation); rq_dsamp [8] =
    // the draft sampler {1, tau_d, top_p, top_k, seed_lo, seed_hi, 0, 0} (host shadow
    // rq_dsamp_host: upload only on change). rq_mode = the NEXT draft passes' mode (0 = today's
    // draft tail, 1 = + the top-32 list for the --wp24-dump rung, 2 = real-q: the full-slice
    // head samples its draft from q') and rq_live = the next verify runs xq_sample_rows_rq; both
    // are graph-key bits. rq_dctr = the draft RNG counter base (draft_meta[3]); rq_draft_i = the
    // pass the next head_launches serves.
    rq_list: CudaSlice<u32>,
    rq_part: CudaSlice<u64>,
    rq_cnt: CudaSlice<u32>,
    rq_dsamp: CudaSlice<u32>,
    rq_dsamp_host: [u32; 8],
    rq_mode: u8,
    rq_live: bool,
    rq_dctr: i32,
    rq_draft_i: Option<usize>,
    /// FP4L --fp4l-plain exact: the stored-bf16 GEMMs' split-K slabs / fixup counters (router, n-gram
    /// projections, hc mixer fallbacks: three disjoint sets, because the WP27 PLE fork runs the n-gram
    /// projections beside layer 0's routed branch) and the shared-gate channels xg [w][h]. None otherwise.
    fxs: Option<fp4l_fwd::FxScratch>,
}

struct Launcher<'a> {
    dev: &'a Arc<CudaDevice>,
    stream: &'a CudaStream,
}

impl Launcher<'_> {
    /// TUNE §4.1: the launch context (family, width, regime) knob lookups key on — the innermost
    /// family scope entered by the entry points (forward_step_kernels, verify_block, draft_block /
    /// draft_pass, head_reprime_launches, prefill_chunk); Ctx::UTIL outside them.
    fn cx(&self) -> tune::Ctx { tune::current() }
}

/// PACKPF: the most prompts one packed pass carries.
pub const PACK_PF_MAX: usize = 16;

/// PACKPF: one prompt segment of a packed prefill pass — rows [r0, r0+n) of the pass are
/// positions [pos0, pos0+n) of `slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PfSeg { pub r0: usize, pub n: usize, pub pos0: usize, pub slot: usize }

/// C-row scratch for the chunked prefill sweep (separate from the decode
/// `Scratch`, which is width-scaled for serving lanes; the compact MoE row
/// lists make the per-expert buffers O(C*topk), not O(E*C)).
pub struct PrefillScratch {
    pub c: usize,
    sids: CudaSlice<i32>,
    // per-row activation buffers ([C, ...])
    resid: CudaSlice<f32>,
    hn: CudaSlice<u16>,
    dd: CudaSlice<u16>,
    uu: CudaSlice<u16>,
    inj: CudaSlice<f32>,
    x: CudaSlice<u16>,
    x_last: CudaSlice<u16>, // row C-1 copy for the final lm_head chain
    // S-A3-t: MTP-head KV fill over the prompt (allocated only when a draft head exists):
    // shifted trunk streams / embed streams (f32 [c][hc*h]), the two fc outputs (f16 [c][hc*h]),
    // and the carry = trunk streams of the previous chunk's last row (f32 [hc*h]).
    hfill_s: CudaSlice<f32>,
    hfill_a: CudaSlice<u16>,
    hfill_b: CudaSlice<u16>,
    hfill_carry: CudaSlice<f32>,
    /// PACKPF: one head-fill carry per slot (slot k at [k*rw, (k+1)*rw)) for prompts whose prefill
    /// runs in packed passes (several prompts in flight at once; psc.hfill_carry holds only one).
    hfill_carry_slots: CudaSlice<f32>,
    // S-A3-u D: device-side MoE routing — tile list [(expert, super-row)] + its length, the
    // identity expert map, per-expert trellis offsets, and the identity row map (down side).
    pf_tiles: CudaSlice<i32>,
    /// A5-P1 WP19: 4 per-MT-class tile lists (stride = the single list's capacity) + counts.
    pf_tiles4: CudaSlice<i32>,
    pf_ntiles4: CudaSlice<i32>,
    pf_ntiles: CudaSlice<i32>,
    pf_ident: CudaSlice<i32>,
    pf_dident: CudaSlice<i32>,
    chain_xh: CudaSlice<u16>,
    chain_yraw: CudaSlice<u16>,
    // S-A3-f-f: split-K workspace for the ks arm — the DECODE branch DOES run here
    // when a final chunk has <= 16 rows (see prefill_scratch sizing note).
    chain_ws: CudaSlice<f32>,
    // S-A3-f-g: reconstruct staging (largest trunk dense matrix, f16, written
    // whole by exl3_decode_dump before any read) + cuBLASLt workspace.
    recon_ws: CudaSlice<u16>,
    lt_ws: CudaSlice<u8>,
    qkv: CudaSlice<u16>,
    qg: CudaSlice<u16>,
    kp: CudaSlice<u16>,
    vp: CudaSlice<u16>,
    zbuf: CudaSlice<u16>,
    a_out: CudaSlice<u16>,
    b_out: CudaSlice<u16>,
    core: CudaSlice<u16>,
    normed: CudaSlice<u16>,
    mixer_out: CudaSlice<u16>,
    logits_r: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    wts: CudaSlice<f32>,
    idxmap: CudaSlice<i32>,
    moe_out: CudaSlice<u16>,
    sh1: CudaSlice<u16>,
    sh2: CudaSlice<u16>,
    ysh: CudaSlice<u16>,
    cos: CudaSlice<f32>,
    toks: CudaSlice<i32>,
    // compact MoE buffers ([R = C*topk, ...]) + row lists
    xh_e: CudaSlice<u16>,
    ygu_raw: CudaSlice<u16>,
    ygu: CudaSlice<u16>,
    din: CudaSlice<u16>,
    xhd: CudaSlice<u16>,
    yd_raw: CudaSlice<u16>,
    yd: CudaSlice<u16>,
    offs_gu: CudaSlice<u64>,
    offs_d: CudaSlice<u64>,
    // TP-B: expert-parallel prefill — the chunk's routed ids in this rank's LOCAL expert space
    // (-1 = remote), [c * topk]
    ids_ep: CudaSlice<i32>,
    row_tok: CudaSlice<i32>,
    row_eidx: CudaSlice<i32>,
    cand: CudaSlice<i32>,
    offs_row: CudaSlice<i32>,
    cnt: CudaSlice<i32>,
    moerows: CudaSlice<i32>,
    // PLE
    ple_emb: CudaSlice<u16>,
    ple_key16: CudaSlice<u16>,
    ple_val: CudaSlice<u16>,
    ple_kn: CudaSlice<f32>,
    ple_qn: CudaSlice<f32>,
    ple_gated: CudaSlice<u16>,
    ple_normed: CudaSlice<u16>,
    // S-A3-i QSA prefill pipes (row-range [t0, c) of a chunk rides the selection path)
    qsa_qk: CudaSlice<u16>,      // [c][640] f16 indexer projection
    qsa_qkb: CudaSlice<u16>,     // [c][640] bf16
    qsa_blocks: CudaSlice<u16>,  // [nblk][128] bf16 pooled block keys
    qsa_scores: CudaSlice<f32>,  // [c][nblk_stride] (WP18 on: empty unless --wp18-xcheck)
    qsa_sel: CudaSlice<i32>,     // [c][sel_max]
    qsa_psel: CudaSlice<i32>,    // [c]
    // WP18 rung A: bounded prefill selection (None = the full-matrix path)
    wp18: Option<Wp18Scratch>,
    pos_rows: CudaSlice<i32>,    // [c] logical positions (pos0 + i)
    // VIS-2: the image request being prefilled (None = text): its merged image embeddings
    // [n_img][hidden] (fp16-rounded f32) and, per prompt position, the image row it takes (-1 = a
    // token embedding); vis_rows [c] is the per-chunk upload of that map for xq_splice_rows.
    pub vis_img: Option<CudaSlice<f32>>,
    pub vis_src: Vec<i32>,
    vis_rows: CudaSlice<i32>,
    slot_rows: CudaSlice<i32>,   // [c] slot ids
    slotpos_rows: CudaSlice<i32>,// [c][2] (slot, pos) for prep/splitk
    qstage: CudaSlice<f32>,      // [c][nh][hd]
    /// PFTC: the tensor-core dense prefill's staged q (f16 bits) [c][nh][hd]; 1 element on packs with an indexer
    /// (Flash-Next never takes PFTC).
    pf_qh: CudaSlice<u16>,
    qsa_pm: CudaSlice<f32>,      // [c*nh*QSA_SPLITS]
    qsa_pl: CudaSlice<f32>,
    qsa_pacc: CudaSlice<f32>,    // [c*nh*QSA_SPLITS*hd]
    pub logits: CudaSlice<u16>,
    argmax: CudaSlice<i32>,
    // WP05: pinned prologue tables + PLE rows (None = pre-WP05 htod_sync_copy path)
    wp05: Option<Wp05PfStage>,
    // TP-I2 item 1: the MoE reduce's rows-landed hook (set by prefill_chunk around moe_prefill) and
    // whether the EP combine's reduce ran it (then the next hc_pre skips its inject+norm)
    pf_hook: Option<xtp::PfHook>,
    pf_hooked: bool,
    /// LANEWIDE-EXACT: a wide verify's scratch under --lane-wide-exact: the hc down / up take the fixed-sequence tail
    /// GEMM at any row count (row-independent). Always false on prompt prefill scratch.
    row_indep: bool,
    // TP-SP: this rank's row block while a sequence-parallel chunk's MoE runs (set by prefill_chunk
    // around moe_prefill: the EP combine then reduce-SCATTERS into the own rows)
    sp_rows: Option<xtp::SpRows>,
    // S-B9-REL-PPL: Some only inside `--perplexity` (exl3_ppl.rs): every row of the chunk goes through
    // the lm_head and is scored. None everywhere else — serving never constructs it.
    ppl: Option<ppl::PplSink>,
    // COPYPF1: Some only inside FwdModel::copy_prefill_verify / --probe-copy-prefill (exl3_forward/cpv.rs): the
    // chunk's final-mixed rows (f16) and its last row's streams (the tap that predicted the next token) are
    // copied out before the head fill overwrites them. None everywhere else: serving without the flag never sets it.
    cpv: Option<cpv::CpvStash>,
    // LEG-B (2026-10-30): fractional-k2 packs (1.66bpw: 82% of experts are k2=3, K=1.5) have no
    // prefill expert kernel (both pf arms decode integer-window trellises). This decode-width
    // Scratch hosts the row-blocked fallback that runs the chunk through the PROVEN decode
    // moe() sequence (xq_gemm_grouped_xh: runtime-k2 dq — the same path tonight's green gates
    // ran on this pack). None on integer-width packs (zero cost).
    frac_sc: Option<Scratch>,
    // FP4L --fp4l-plain exact: the prefill shared-gate channels xg [c][h] (columns >= 6 stay zero).
    fxg: Option<CudaSlice<u16>>,
}

/// Everything cuBLASLt needs for ONE (k, n, m) shape: the heuristic-pinned algo
/// plus the descriptors. Created once per shape (first call, or PFX1's background
/// pre-population), reused forever — per-call descriptor creation would both cost
/// host time and leak.
struct LtPlan {
    algo: ltsys::cublasLtMatmulAlgo_t,
    desc: ltsys::cublasLtMatmulDesc_t,
    a_layout: ltsys::cublasLtMatrixLayout_t,
    b_layout: ltsys::cublasLtMatrixLayout_t,
    c_layout: ltsys::cublasLtMatrixLayout_t,
}

/// S-B9-CF-PACK4: one request's drafted chain awaiting the packed verify (`FwdModel::pack_draft`).
pub struct PackDraft {
    pub slot: usize,
    /// the bonus token at position `p`
    pub b: i32,
    pub p: usize,
    /// the drafts the verify will judge (>= 1)
    pub drafts: Vec<i32>,
    /// per-pass confidences (the DDS calibrator's labels; empty for a fixed-depth chain)
    pub confs: Vec<f32>,
    spec: bool,
    /// PACK4S: the lane's sampler for its verify rows (greedy = `SegSamp::GREEDY`) and the
    /// real-q draft sampler its chain ran with (None = an argmax chain)
    pub samp: crate::lane_pack::SegSamp,
    pub rq: Option<RqStage>,
    /// S-B9-CF-PACK5: head draft passes this rank issued for the chain (loop + speculative
    /// launches; a fixed-depth chain = its depth): the pre-verify word the ranks compare
    pub launched: usize,
    /// a speculative pass was launched and then stopped by the DDS: its collective epochs may
    /// still be in flight when the pack's host exchange runs (`pack_drain_wasted`)
    pub wasted: bool,
}

/// S-B9-CF-PACK4: what the draft half of an adaptive round hands to its verify.
struct DraftHalf {
    /// the w verified drafts, clamped to the vocabulary
    drafts: Vec<i32>,
    /// per-pass confidences of every pass that ran
    confs: Vec<f32>,
    w: usize,
    /// head draft passes this rank issued (loop + speculative launches): the TP pre-verify word
    launched: usize,
    /// position of a launched pass the DDS then stopped before
    wasted: Option<usize>,
    /// the speculative next pass was live this round
    spec: bool,
    tp0: std::time::Instant,
    tp1: std::time::Instant,
}

/// WP05: page-locked host memory (cuMemHostAlloc) for the per-round / per-chunk staging and the
/// per-pass DDS readback. Copies from/to it are enqueued on the device's NULL (legacy) stream —
/// the same stream cudarc's htod_copy_into / dtoh_sync_copy use, so ordering against the
/// blocking compute stream is unchanged — and every host rewrite of a region happens only after
/// a NULL-stream synchronize has retired the copies that read it (callers track `inflight`).
pub(crate) struct PinnedHost {
    ptr: *mut u8,
    len: usize,
    dev: Arc<CudaDevice>,
}

/// WP05: decode-side pinned staging owned by a Scratch (width w). Byte layout:
/// toks [w] i32 | pos [w] i32 | slots [2w] i32 | slot_ids [w] i32 | draft_meta [4] i32 |
/// readback [4] u32 (dconf_i, d_i) | ple_emb [w*2560] u16.
pub(crate) struct Wp05Stage {
    pin: PinnedHost,
    w: usize,
    /// a copy from `pin` may still be in flight (set on enqueue, cleared by a synchronize)
    /// HOST caveat: the DDS loop also clears it after a readback-EVENT wait; in a round whose DDS
    /// stopped with a speculative pass in flight, that pass's 8-byte DtoH into the OTHER readback
    /// slot (off_rb()+0/+8) may still be landing. No host code writes the rb region (only reads a
    /// slot after its own event) and every later rb DtoH is stream-ordered behind it, so this is
    /// safe today — a future HOST WRITE into the rb region must synchronize first.
    inflight: bool,
    /// PLE rows 0..toks.len() already built into the ple region for (slot, hist pair) —
    /// the DDS round builds row i while draft pass i runs; mtp_verify builds the rest.
    ple_pre: Option<(usize, (i32, i32), Vec<i32>)>,
    /// HOST / RO-5(a): speculative-next-pass state (created on the first DDS round that uses it)
    spec: Option<SpecPass>,
}

/// WP05: prefill-side pinned staging owned by a PrefillScratch (chunk width c). Byte layout:
/// toks [c] | pos_rows [c] | slot_rows [c] | slotpos_rows [2c] | moerows [4] (i32) |
/// ple_emb [c*2560] u16 (0 rows when the pack has no PLE).
pub(crate) struct Wp05PfStage {
    pin: PinnedHost,
    c: usize,
    // per REGION (disjoint bytes): the prologue's table copies may still be queued when the
    // layer-1 PLE rows are written — one shared flag would force a mid-chunk host drain there.
    inflight_tab: bool,
    inflight_ple: bool,
}

struct PlainTensor {
    dtype: String,
    offset: usize,
    n_bytes: usize,
    shard: String,
    shape: Vec<usize>,
}

/// Buffers a caller must carry so `exl3_chain` can take the reconstruct+Lt arm.
/// Raw device pointers only (no borrows): the staging is written whole by
/// exl3_decode_dump before any read, the Lt workspace by cuBLASLt itself.
/// Decode paths pass None and keep byte-identical launches.
pub(crate) struct ReconBufs {
    ws: u64,       // staging device pointer (f16, [k_max * n_max])
    ws_len: usize, // staging capacity in u16 elements
    rc: u64,       // rc_of_t LUT device pointer
    ltws: u64,     // cuBLASLt workspace device pointer
    ltws_len: usize,
}

/// A rung-2 selection for one MoE call (FwdModel::wp20_select -> moe_wp20): the boot-verified
/// suh sequence id (logged only — WP20-v1c pins the sequence in the kernels) and the exact
/// entries to launch (the down one compiled for `topk`).
#[derive(Clone, Copy)]
struct Wp20Sel {
    conv: u32,
    gu: &'static str,
    dn: &'static str,
    topk: usize,
}

/// WP18 buffers of one PrefillScratch: [rows][WP18_T] slab + [rows][topk] carried lists, and the
/// XCHECK reference lists/counters.
pub(crate) struct Wp18Scratch {
    rows: usize,
    slab: CudaSlice<f32>,
    ck: CudaSlice<u32>,
    ci: CudaSlice<i32>,
    x: Option<(CudaSlice<i32>, CudaSlice<i32>, CudaSlice<i32>)>,
}

#[cfg(test)]
mod rel_gates2_reprime_words_tests;
#[cfg(test)]
mod pfx1_tests;
#[cfg(test)]
mod reprime_tests;
#[cfg(test)]
mod ple_ngram_tests;
#[cfg(test)]
mod wp05_tests;
#[cfg(test)]
mod ple_k_tests;
#[cfg(test)]
mod served_window_tests;
#[cfg(test)]
mod rel_fixb_router_selection;
#[cfg(test)]
mod l10_prefillx_tests;
#[cfg(test)]
mod wp18_tests;
#[cfg(test)]
mod qsaregime_tests;
#[cfg(test)]
mod tp4g_dv3_tests;
/// PQ8-v2: host emulation of the flash prefill kernel (a3fg_flash_body<HD, BT, BKV, ..>) to pin
/// down WHAT the K/V tile staging puts in shared memory — the only thing that differs between
/// xq_attn_prefill_flash256 (scalar stager) and its flash256v twin. The math (fma score chain
/// from +0 ascending, online softmax, shfl_down sum tree, fma PV) is one shared routine, so any
/// output difference is a staging difference.
///
/// The scalar stager (blockDim == hd == 256) gives tile element (jj, e) to thread e, i.e. warp
/// e >> 5, and zeroes it when jj >= THAT warp's jn = min(16, pmax_w + 1 - jt) — its OWN q row's
/// causal limit, not the block's; a warp whose row loop has ended (jt > pmax_w) stages nothing,
/// leaving the previous tile's values. So on a block's diagonal tile the key/value rows a later
/// q row needs have the dims of earlier warps zeroed (or stale). flash256v v1 staged (row w and
/// w+8, all dims of group lane) under warp w's jn instead — a DIFFERENT smem image (it happens
/// to be the fully staged, causally correct one when pos0 % 8 == 0), hence the ~48% f16
/// mismatch on the GPU. v2 reproduces the scalar image exactly: owner warp of dims grp*8.. is
/// grp >> 2, stage iff jt <= pbase + (grp >> 2), zero iff jt + jj > pbase + (grp >> 2), and all
/// warps run the block's trip count (pbase + 7) so the owner's elements get staged.
///
/// w5/PFIX: the emulation is shape-generic. The hd=128 shape (xq_attn_prefill_flash: BT 16,
/// BKV 32, blockDim 512 != hd) hands element (jj, e) to thread (jj*128 + e) % 512, i.e. warp
/// (jj % 4) * 4 + (e >> 5) — the same defect under a row-and-dim owner map; the CAUSAL image
/// (xq_attn_prefill_flash128c / flash256c) matches the causal softmax at both shapes.
#[cfg(test)]
mod flash256_stage_tests;
#[cfg(test)]
mod wp24_key_tests;
#[cfg(test)]
mod vis2_tests;
#[cfg(test)]
mod ple_headroom_tests;
/// S-B9-REL-INTEG D1: a wide trellis launch on a bit width its switch has no case for writes nothing (the
/// kernels end in `default: break;`) and the request is answered 200 with garbage. Reachable only off the
/// default flags (`--exl3-recon-min 0`, or a call site without reconstruct buffers; `--pf-moe-kernel` on a
/// pack with 5-bit routed experts); the host refuses before the first launch. The GPU check is G6 of the
/// integration gates (the 4.05 pack with `--exl3-recon-min 0`).
#[cfg(test)]
mod wide_bits_guard_tests;
/// P7 B3 (K = 5 experts): the host picks the b5 SHFL entries — the a1b5 grouped rung-1 entry,
/// the K-chunked b5 gate/up (diet-independent: the word diet is b3/b4-only by static_assert),
/// and the top-10-only b5 down epilogue. Pins the arm so a future re-gate cannot silently
/// drop K = 5 back to the K-generic chain.
#[cfg(test)]
mod wp20_b5_tests;
/// YARN1 rope unit tests (host-only): the EXL3 tables' freqs/mscale against an INDEPENDENTLY
/// derived reference (the YaRN formula computed in this module, not a call into the dspark
/// oracle — AGENTS §3: same-class consistency proves nothing), the factor-1.0 bit-identity
/// with the legacy native loop, and the mscale formula.
#[cfg(test)]
mod rope_tests;
/// ISSUES 122: a spill region is one slot's share of a `[width x per_slot]` plane in BYTES. The capture used the element
/// count as the byte count: it kept 1/4 of every f32 plane (GDN conv/S state, head-fill carry) and 1/2 of every u16 plane
/// (QSA keys/pool, PLE ring), at the wrong offset for slot > 0, so a restore left the previous occupant's state in the rest.
#[cfg(test)]
mod spill_region_tests;
#[cfg(test)]
mod dsplitk_tests;
