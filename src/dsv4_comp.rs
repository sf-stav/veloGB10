//! DSV4 Phase-3 LANE 3B: the §B.5 compressor (CSA overlap ratio-4 + HCA ratio-128)
//! and the §B.6 indexer on GPU, built on the spine (`dsv4_gpu`) and the lane module
//! `kernels/gpu_dsv4_comp.cu` (`src/ptx/gpu_dsv4_comp.ptx`).
//!
//! # Semantics target
//!
//! The G1-proven CPU reference (`dsv4_cpu::Compressor` / `dsv4_cpu::indexer_forward`),
//! replicated operation-for-operation (see the .cu module docs): the fp32 wkv/wgate
//! GEMM uses `dot_tree`'s exact pairwise-tree order; the softmax-pool uses the CPU's
//! serial ascending-j order with no FMA contraction and double-rounded exp; the decode
//! state machine (CSA slots 4..7 + 8-tap cat/shift, HCA slot start_pos%128) is one
//! block; the indexer score chain replicates dot8 + the bf16 rounding chain
//! (Lane-B-finding-#2) exactly. Post-pool finishing rides the spine's G1/spine-proven
//! kernels: `dsv4_rmsnorm_b` → `dsv4_rope_last_b` (block's FIRST token position) →
//! QAT-sim (attention compressor: strided FP8 act_quant_sim group 64 on `:448`;
//! indexer compressor rotate=True: `dsv4_fwht_rotate` + `dsv4_fp4_act_quant_sim`).
//!
//! # Batch discipline
//!
//! B=1 (the CPU reference and the oracle are single-sequence; verify-width N is a
//! token count, not a batch). Every per-token result is a fixed-order reduction, so
//! outputs are pure functions of the committed prefix (§12.B.2 batch-invariance):
//! prefill-vs-decode-vs-verify-width cannot change any token's compressor row or
//! indexer selection. The N>1 decode path (chunked prefill / verify) is a sequential
//! per-token loop — exactly the one-shot-equivalent semantics §12.B.5 demands (the
//! CPU reference's decode branch is s=1-only; equivalence is proven by the chunked-
//! prefill gate in tests/dsv4_comp_test.rs).
//!
//! # Frontier snapshot/restore (§12.B.4)
//!
//! `kv_state`/`score_state` (fp32, score init −inf) are the ONLY recurrent state.
//! `snapshot()` copies them to a device-side `CompSnapshot` (D2D on the compute
//! stream — rides prefix-cache snapshots without a host round-trip); `restore()`
//! rewinds. `reset()` re-initializes (kernel — memset can't write the −inf pattern).

use anyhow::{anyhow, Result};
use cudarc::driver::result;
use cudarc::driver::sys;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, DeviceSlice};
use half::bf16;
use std::sync::Arc;

use crate::dsv4_cpu;
use crate::dsv4_gpu::{self, Dsv4Arg, Dsv4Kernels, B, S};
use crate::dsv4_launch;

/// fp32 device buffer of i32 positions/flags.
type I32Dev = CudaSlice<i32>;

// `Dsv4Arg` for device VIEWS (row-offset sub-slices) — the pointer marshalling is the
// same as for CudaSlice (address OF the CUdeviceptr field, §4 gotcha #1). The trait is
// crate-local, so these impls live here (dsv4_gpu.rs stays untouched).
impl<T> Dsv4Arg for CudaView<'_, T> {
    fn ptr(&self) -> *mut std::ffi::c_void {
        self.device_ptr() as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}
impl<T> Dsv4Arg for CudaViewMut<'_, T> {
    fn ptr(&self) -> *mut std::ffi::c_void {
        self.device_ptr() as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

// -----------------------------------------------------------------------------------------------
// Spec + module loading
// -----------------------------------------------------------------------------------------------

/// Compressor geometry (mirrors dsv4_cpu::CompressorWeights' shape fields).
#[derive(Debug, Clone, Copy)]
pub struct CompSpec {
    pub ratio: usize,     // 4 (CSA/indexer) / 128 (HCA)
    pub head_dim: usize,  // d: 512 (attention) / 128 (indexer)
    pub rope_dim: usize,  // rd: 64
    pub overlap: bool,    // ratio == 4
    pub rotate: bool,     // indexer compressor: Hadamard + FP4 sim on full d
    pub sim_group: usize, // 64 (FP8 attn) / 32 (FP4 indexer)
    pub dim: usize,       // input dim (4096)
}

impl CompSpec {
    pub fn coff(&self) -> usize {
        1 + self.overlap as usize
    }
    /// R5 packed-cache row layout (code bytes, scale bytes) — the QAT-native form:
    /// rotate (indexer, FP4 g32): (d/2, d/32); non-rotate (attn, FP8 g64 nope): (nope, nope/64).
    pub fn packed_layout(&self) -> (usize, usize) {
        if self.rotate {
            (self.head_dim / 2, self.head_dim / self.sim_group)
        } else {
            let nope = self.head_dim - self.rope_dim;
            (nope, nope / self.sim_group)
        }
    }
    /// coff·d — the wkv/wgate output width and one state row's length.
    pub fn cd(&self) -> usize {
        self.coff() * self.head_dim
    }
    /// State tensor rows: coff·ratio.
    pub fn state_rows(&self) -> usize {
        self.coff() * self.ratio
    }

    /// CSA attention compressor (§B.5, ratio 4, overlap, FP8-sim on :448).
    pub fn csa_attn(dim: usize, rope_dim: usize) -> Self {
        CompSpec { ratio: 4, head_dim: 512, rope_dim, overlap: true, rotate: false, sim_group: 64, dim }
    }
    /// HCA attention compressor (§B.5, ratio 128, no overlap, FP8-sim on :448).
    pub fn hca_attn(dim: usize, rope_dim: usize) -> Self {
        CompSpec { ratio: 128, head_dim: 512, rope_dim, overlap: false, rotate: false, sim_group: 64, dim }
    }
    /// CSA indexer compressor (§B.6: ratio 4, overlap, rotate, FP4-sim on full 128).
    pub fn indexer(dim: usize, rope_dim: usize) -> Self {
        CompSpec { ratio: 4, head_dim: 128, rope_dim, overlap: true, rotate: true, sim_group: 32, dim }
    }
}

impl From<&dsv4_cpu::CompressorWeights> for CompSpec {
    fn from(w: &dsv4_cpu::CompressorWeights) -> Self {
        CompSpec {
            ratio: w.ratio,
            head_dim: w.head_dim,
            rope_dim: w.rope_dim,
            overlap: w.overlap,
            rotate: w.rotate,
            sim_group: w.sim_group,
            dim: w.dim,
        }
    }
}

/// Both PTX modules the lane needs: the lane's own `gpu_dsv4_comp` and the spine
/// `gpu_dsv4` (rmsnorm / rope_last / fwht / fp4-sim / topk / act_quant_g128).
pub struct CompKernels {
    pub comp: Dsv4Kernels,
    pub spine: Dsv4Kernels,
}

impl CompKernels {
    pub fn load(dev: &Arc<CudaDevice>) -> Result<Self> {
        let comp = Dsv4Kernels::load_module(
            dev,
            "src/ptx/gpu_dsv4_comp.ptx",
            &[
                "dsv4_comp_state_init_b",
                "dsv4_comp_gemm_tree_f32w_b",
                "dsv4_comp_gemm_tree_bf16w_bf16out_b",
                "dsv4_comp_gemm_tc_b",
                "dsv4_comp_gemm_tc_pair_b",
                "dsv4_comp_gemm_fast_pair_b",
                "dsv4_comp_pad16_b",
                "dsv4_comp_prefill_pool_b",
                "dsv4_comp_prefill_stash_b",
                "dsv4_comp_decode_b",
                "dsv4_comp_round_bf16_b",
                "dsv4_comp_act_quant_sim_g64s_b",
                "dsv4_comp_act_quant_g64s_b",
                "dsv4_comp_copy_rows_b",
                "dsv4_comp_wscale_b",
                "dsv4_comp_index_score_b",
                "dsv4_comp_index_score_fp4_b",
                "dsv4_comp_idx_remask_b",
                "dsv4_score_gather_b",
                "dsv4_idx_remap_b",
                "dsv4_idx_offset_place_b",
                "dsv4_comp_index_score_tile_b",
                "dsv4_f32_place_b",
                "dsv4_f32_gather_place_b",
            ],
        )?;
        let spine = Dsv4Kernels::load(
            dev,
            &[
                "dsv4_rmsnorm_b",
                "dsv4_rope_last_b",
                "dsv4_iota_b",
                "dsv4_fwht_rotate",
                "dsv4_fp4_act_quant_sim",
                "dsv4_fp4_act_quant",
                "dsv4_act_quant_g128",
                "dsv4_topk",
            ],
        )?;
        Ok(CompKernels { comp, spine })
    }
}

/// Device-side RoPE table (fp32 cos/sin [positions, rd/2]).
pub struct DevRope {
    pub cos: S,
    pub sin: S,
    pub rd: usize,
}

impl DevRope {
    pub fn from_cpu(dev: &Arc<CudaDevice>, table: &dsv4_cpu::RopeTable) -> Result<Self> {
        let cos = dev.htod_sync_copy(&table.cos).map_err(|e| anyhow!("rope cos htod: {e}"))?;
        let sin = dev.htod_sync_copy(&table.sin).map_err(|e| anyhow!("rope sin htod: {e}"))?;
        Ok(DevRope { cos, sin, rd: table.rd })
    }
}

// -----------------------------------------------------------------------------------------------
// GpuCompressor — §B.5 on GPU (both variants + the indexer's rotate one)
// -----------------------------------------------------------------------------------------------

/// Device-side compressor. Owns its output cache ([cache_rows, head_dim] bf16 — for the
/// attention compressor this models the aliased rows 128.. of the layer kv_cache; the
/// integration layer can hand us a view later) and the fp32 frontier state.
pub struct GpuCompressor {
    pub spec: CompSpec,
    norm_eps: f32,
    // weights (fp32 device — bf16-sourced, used fp32 per §A.2). The scalar tree GEMM
    // reads these; kept for backward compat + the indexer's bf16-out GEMM variant.
    wkv: S,
    wgate: S,
    /// bf16-cast weights for the WMMA tensor-core GEMM (dsv4_comp_gemm_tc_b). Cast
    /// from the same bf16-valued f32 at upload — lossless (§A.2 store-bf16/compute-fp32).
    wkv_bf: B,
    wgate_bf: B,
    /// Pre-padded activation panel [16, dim] bf16 for the WMMA pair GEMM's partial-row
    /// tiles (R3A.1: replaces the per-K-tile a_pad smem detour — E0c measured it at 5x).
    /// Filled by dsv4_comp_pad16_b once per gemm_pair call when s % 16 != 0.
    x_pad: B,
    norm: S,
    ape: S,
    // frontier state (fp32; score_state init −inf) — the snapshot/restore target
    pub kv_state: S,
    pub score_state: S,
    // scratch
    kv_full: S,
    score_full: S,
    pooled: S,
    pooled_bf: B,
    pooled_bf2: B, // rotate path (fwht out)
    sim_scales: CudaSlice<u8>,
    fire_dev: CudaSlice<u32>,
    /// compressor output cache [cache_rows, head_dim] bf16 (post-QAT-sim values).
    pub cache: B,
    pub cache_rows: usize,
    /// R5a/R5b: the QAT-native packed form of the cache — rotate (indexer): FP4-e2m1 codes
    /// [cache_rows, d/2] + UE8M0 scales [cache_rows, d/32]; non-rotate (attn): FP8-e4m3
    /// codes [cache_rows, nope] + scales [cache_rows, nope/64] (the 64 rope dims stay in
    /// the bf16 cache — the packed reader covers the nope span only). Written by the
    /// epilogue BEFORE the in-place sim (same body math, same inputs ⇒ dequant(packed) ==
    /// the bf16 cache rows' simmed span, BITWISE).
    pub packed_codes: Option<CudaSlice<u8>>,
    pub packed_scales: Option<CudaSlice<u8>>,
    /// When nonzero, the epilogue writes cache rows to this device address INSTEAD of
    /// `cache` — the kv_cache/attn-cache alias (Item 3). The attention compressor's
    /// cache is aliased to `kv_cache[win..]`, eliminating the per-step d2d mirror and
    /// recovering ~6 GB at 1M. snapshot/restore skip the cache when aliased (the
    /// kv_cache snapshot already covers the tail). 0 = use `cache` (the indexer's
    /// compressor and all test paths).
    pub cache_alias_ptr: u64,
    s_max: usize,
}

impl GpuCompressor {
    /// `w` is the CPU-side weight struct (already fp32: loader upcast). `s_max` sizes
    /// the GEMM scratch (the largest prefill chunk this instance will see).
    pub fn new(
        dev: &Arc<CudaDevice>,
        ks: &CompKernels,
        stream: &CudaStream,
        spec: CompSpec,
        w: &dsv4_cpu::CompressorWeights,
        norm_eps: f32,
        cache_rows: usize,
        s_max: usize,
    ) -> Result<Self> {
        let cd = spec.cd();
        let rows = spec.state_rows();
        anyhow::ensure!(w.wkv.len() == cd * spec.dim && w.wgate.len() == cd * spec.dim, "wkv/wgate shape");
        anyhow::ensure!(w.norm.len() == spec.head_dim && w.ape.len() == spec.ratio * cd, "norm/ape shape");
        anyhow::ensure!(spec.head_dim == 512 || spec.head_dim == 128, "head_dim {} unsupported", spec.head_dim);
        anyhow::ensure!(spec.ratio == 4 || spec.ratio == 128, "ratio {} unsupported", spec.ratio);
        if !spec.rotate {
            anyhow::ensure!(spec.sim_group == 64 && (spec.head_dim - spec.rope_dim) % 64 == 0, "fp8 sim geometry");
        }
        let nb_max = s_max / spec.ratio + 1;
        let wkv_bf: Vec<bf16> = w.wkv.iter().map(|&v| bf16::from_f32(v)).collect();
        let wgate_bf: Vec<bf16> = w.wgate.iter().map(|&v| bf16::from_f32(v)).collect();
        let me = GpuCompressor {
            spec,
            norm_eps,
            wkv: dev.htod_sync_copy(&w.wkv).map_err(|e| anyhow!("wkv htod: {e}"))?,
            wgate: dev.htod_sync_copy(&w.wgate).map_err(|e| anyhow!("wgate htod: {e}"))?,
            wkv_bf: dev.htod_sync_copy(&wkv_bf).map_err(|e| anyhow!("wkv_bf htod: {e}"))?,
            wgate_bf: dev.htod_sync_copy(&wgate_bf).map_err(|e| anyhow!("wgate_bf htod: {e}"))?,
            x_pad: dev.alloc_zeros::<bf16>(16 * spec.dim).map_err(|e| anyhow!("x_pad alloc: {e}"))?,
            norm: dev.htod_sync_copy(&w.norm).map_err(|e| anyhow!("norm htod: {e}"))?,
            ape: dev.htod_sync_copy(&w.ape).map_err(|e| anyhow!("ape htod: {e}"))?,
            kv_state: dev.alloc_zeros::<f32>(rows * cd).map_err(|e| anyhow!("kv_state alloc: {e}"))?,
            score_state: dev.alloc_zeros::<f32>(rows * cd).map_err(|e| anyhow!("score_state alloc: {e}"))?,
            kv_full: dev.alloc_zeros::<f32>(s_max.max(1) * cd).map_err(|e| anyhow!("kv_full alloc: {e}"))?,
            score_full: dev.alloc_zeros::<f32>(s_max.max(1) * cd).map_err(|e| anyhow!("score_full alloc: {e}"))?,
            pooled: dev.alloc_zeros::<f32>(nb_max * spec.head_dim).map_err(|e| anyhow!("pooled alloc: {e}"))?,
            pooled_bf: dev.alloc_zeros::<bf16>(nb_max * spec.head_dim).map_err(|e| anyhow!("pooled_bf alloc: {e}"))?,
            pooled_bf2: dev.alloc_zeros::<bf16>(nb_max * spec.head_dim).map_err(|e| anyhow!("pooled_bf2 alloc: {e}"))?,
            sim_scales: dev.alloc_zeros::<u8>(nb_max * (spec.head_dim / spec.sim_group).max(1)).map_err(|e| anyhow!("sim scales alloc: {e}"))?,
            fire_dev: dev.alloc_zeros::<u32>(1).map_err(|e| anyhow!("fire alloc: {e}"))?,
            cache: dev.alloc_zeros::<bf16>(cache_rows * spec.head_dim).map_err(|e| anyhow!("cache alloc: {e}"))?,
            packed_codes: {
                let (cb, _sb) = spec.packed_layout();
                Some(dev.alloc_zeros::<u8>(cache_rows * cb).map_err(|e| anyhow!("packed codes alloc: {e}"))?)
            },
            packed_scales: {
                let (_cb, sb) = spec.packed_layout();
                Some(dev.alloc_zeros::<u8>(cache_rows * sb).map_err(|e| anyhow!("packed scales alloc: {e}"))?)
            },
            cache_rows,
            cache_alias_ptr: 0,
            s_max,
        };
        // alloc_zeros does NOT zero (AGENTS §2.2): the state init kernel establishes
        // kv=0 / score=−inf on the compute stream, then syncs so later reads are safe.
        me.reset(ks, stream)?;
        dev.synchronize().map_err(|e| anyhow!("post-init sync: {e}"))?;
        Ok(me)
    }

    /// Re-initialize the frontier state (kv=0, score=−inf) on the compute stream.
    pub fn reset(&self, ks: &CompKernels, stream: &CudaStream) -> Result<()> {
        let n = self.kv_state.len() as i64;
        dsv4_launch!(ks.comp, "dsv4_comp_state_init_b", stream.stream,
            (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&self.kv_state, &self.score_state, &n))?;
        Ok(())
    }

    /// The two GEMMs (wkv/wgate) for s rows of x → kv_full/score_full. The attention
    /// compressor (rotate=false) uses the WMMA tensor-core kernel for production speed
    /// — its output feeds the gather (tolerance-level). The INDEXER compressor
    /// (rotate=true) keeps the scalar tree GEMM — its output feeds the topk score chain
    /// where reorder-level noise flips near-tie block selections at long context.
    fn gemm_pair<X: Dsv4Arg + ?Sized>(&self, ks: &CompKernels, stream: &CudaStream, x: &X, s: usize) -> Result<()> {
        let spec = &self.spec;
        let (si, ki, ni) = (s as i32, spec.dim as i32, spec.cd() as i32);
        if spec.rotate {
            // Indexer compressor: scalar tree GEMM (bit-exact, preserves topk precision).
            dsv4_launch!(ks.comp, "dsv4_comp_gemm_tree_f32w_b", stream.stream,
                (spec.cd() as u32, s as u32, 1), (256, 1, 1), 0,
                (x, &self.wkv, &self.kv_full, &si, &ki, &ni))?;
            dsv4_launch!(ks.comp, "dsv4_comp_gemm_tree_f32w_b", stream.stream,
                (spec.cd() as u32, s as u32, 1), (256, 1, 1), 0,
                (x, &self.wgate, &self.score_full, &si, &ki, &ni))?;
        } else {
            // Attention compressor: WMMA tensor-core GEMM (tolerance-level, feeds gather).
            // R3.3: the wkv+wgate pair now goes in ONE fused launch (dsv4_comp_gemm_tc_pair_b) —
            // each tile's k-loop is the identical WMMA sequence, so §12.B.5 cross-width bitwise
            // holds exactly; at decode the two sequential 32-CTA launches' latency halves.
            // R3A.1: partial-row tiles read the pre-padded x_pad panel (filled once per call by
            // dsv4_comp_pad16_b) instead of the per-K-tile a_pad smem detour — the tile the mma
            // sees is bit-identical; only the load schedule changed.
            // (A decode-width GEMV replacement was tried and REVERTED: the compressor state is
            // cross-width bitwise-contracted; only reduction-order-preserving changes are
            // admissible here.)
            let tiles_m = (s + 15) / 16;
            let tiles_n = (spec.cd() + 15) / 16;
            let rc = s - (tiles_m - 1) * 16;
            if rc < 16 {
                let (row0, rci, ki2) = ((tiles_m - 1) as i32 * 16, rc as i32, spec.dim as i32);
                dsv4_launch!(ks.comp, "dsv4_comp_pad16_b", stream.stream,
                    (((spec.dim + 255) / 256) as u32, 16, 1), (256, 1, 1), 0,
                    (&self.x_pad, x, &row0, &rci, &ki2))?;
            }
            if crate::dsv4_attn::exact_gemm_enabled() {
                dsv4_launch!(ks.comp, "dsv4_comp_gemm_tc_pair_b", stream.stream,
                    ((2 * tiles_n) as u32, tiles_m as u32, 1), (32, 1, 1), 0,
                    (&self.kv_full, &self.score_full, x, &self.x_pad, &self.wkv_bf, &self.wgate_bf, &si, &ki, &ni))?;
            } else {
                // Item 2.5 fast path (default): big-tile 4-warp bf16→fp32 GEMM — same inputs,
                // scheduler-chosen reduction order (BITWISE == tc_pair in practice: the
                // per-element wmma chain is unchanged; the tolerance class degrades to exact,
                // gated). Isolated bench (rotating weights): 2.01x at s=1, 1.44x at s=6 —
                // the DSpark step's two shapes — but 0.74-0.76x at prefill width (the
                // [16,64] tile's A-slab re-read vs tc_pair's 128-CTA spread), so the fast
                // kernel dispatches only for s <= 16 (decode/verify), with the under-filled
                // HCA corner (16 CTAs) keeping tc_pair.
                let cd = spec.cd();
                anyhow::ensure!(cd % 64 == 0, "fast compressor pair needs cd % 64 == 0 (cd={cd})");
                let tiles64 = cd / 64;
                if s <= 16 && (2 * tiles64) * tiles_m >= 32 {
                    dsv4_launch!(ks.comp, "dsv4_comp_gemm_fast_pair_b", stream.stream,
                        ((2 * tiles64) as u32, tiles_m as u32, 1), (128, 1, 1), 0,
                        (&self.kv_full, &self.score_full, x, &self.x_pad, &self.wkv_bf, &self.wgate_bf, &si, &ki, &ni))?;
                } else {
                    dsv4_launch!(ks.comp, "dsv4_comp_gemm_tc_pair_b", stream.stream,
                        ((2 * tiles_n) as u32, tiles_m as u32, 1), (32, 1, 1), 0,
                        (&self.kv_full, &self.score_full, x, &self.x_pad, &self.wkv_bf, &self.wgate_bf, &si, &ki, &ni))?;
                }
            }
        }
        Ok(())
    }

    /// Post-pool finishing (§B.5 :368-382): bf16 round → RMSNorm_d → RoPE at each row's
    /// block-first-token position → QAT-sim (FP8 g64 on :448, or Hadamard+FP4 when
    /// rotate) → cache write at rows [cache_row0, cache_row0+nrows). R5a/R5b: BEFORE the
    /// in-place sim, the CODES variant of the same quant body also writes the QAT-native
    /// packed rows (packed_codes/packed_scales — lossless vs the bf16 cache, gated).
    /// R3A.4 P3: positions are `start + (i*mul)/div` generated ON DEVICE (the callers used
    /// to upload a host Vec per epilogue — a full cuCtxSynchronize each).
    fn epilogue<I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        dev: &Arc<CudaDevice>,
        ks: &CompKernels,
        stream: &CudaStream,
        rope: &DevRope,
        nrows: usize,
        pos_spec: (i32, i32, i32),
        cache_row0: usize,
    ) -> Result<()> {
        let spec = &self.spec;
        let d = spec.head_dim;
        let n = (nrows * d) as i32;
        let rows_i = nrows as i32;
        let d_i = d as i32;
        // round fp32 → bf16
        dsv4_launch!(ks.comp, "dsv4_comp_round_bf16_b", stream.stream,
            (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&self.pooled_bf, &self.pooled, &n))?;
        // RMSNorm (in-place: the kernel reads each element before writing it)
        let eps = self.norm_eps;
        dsv4_launch!(ks.spine, "dsv4_rmsnorm_b", stream.stream,
            (nrows as u32, 1, 1), (256, 1, 1), 0,
            (&self.pooled_bf, &self.pooled_bf, &self.norm, &rows_i, &d_i, &eps))?;
        // RoPE at the block's first-token position (last rd dims)
        let (pstart, pmul, pdiv) = pos_spec;
        let pos_dev = crate::dsv4_gpu::iota_positions::<I>(dev, &ks.spine, stream, pstart, pmul, pdiv, nrows)?;
        let rd_i = spec.rope_dim as i32;
        let inv0 = 0i32;
        dsv4_launch!(ks.spine, "dsv4_rope_last_b", stream.stream,
            (((nrows + 7) / 8) as u32, 1, 1), (256, 1, 1), 0,
            (&self.pooled_bf, &rope.cos, &rope.sin, &pos_dev, &rows_i, &d_i, &rd_i, &inv0))?;
        // QAT-sim + cache write. The cache destination is either the compressor's own
        // `cache` buffer or, when aliased (cache_alias_ptr != 0), the external buffer
        // (the kv_cache tail — Item 3 alias). Both are device addresses; DevPtr wraps
        // either for the kernel launch.
        let row0_i = cache_row0 as i32;
        let cache_dst = crate::dsv4_gpu::DevPtr {
            dptr: if self.cache_alias_ptr != 0 { self.cache_alias_ptr } else { *self.cache.device_ptr() },
        };
        if spec.rotate {
            dsv4_launch!(ks.spine, "dsv4_fwht_rotate", stream.stream,
                (((nrows + 7) / 8) as u32, 1, 1), (256, 1, 1), 0,
                (&self.pooled_bf, &self.pooled_bf2, &rows_i))?;
            let warps = nrows * (d / 32);
            // R5a: write the FP4-packed cache rows FIRST (codes variant reads pooled_bf2
            // unmodified; the in-place sim then runs the IDENTICAL body on the IDENTICAL
            // inputs ⇒ dequant(packed) == the bf16 cache rows, bitwise — the R5a-1 gate
            // asserts exactly that).
            if crate::dsv4_gpu::env_flag_once(crate::opt!("packed-cache")) {
                if let (Some(pc), Some(ps)) = (&self.packed_codes, &self.packed_scales) {
                let codes_dst = crate::dsv4_gpu::DevPtr {
                    dptr: *pc.device_ptr() + (cache_row0 * (d / 2)) as u64,
                };
                let scales_dst = crate::dsv4_gpu::DevPtr {
                    dptr: *ps.device_ptr() + (cache_row0 * (d / spec.sim_group)) as u64,
                };
                dsv4_launch!(ks.spine, "dsv4_fp4_act_quant", stream.stream,
                    (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                    (&self.pooled_bf2, &codes_dst, &scales_dst, &rows_i, &d_i))?;
                }
            }
            dsv4_launch!(ks.spine, "dsv4_fp4_act_quant_sim", stream.stream,
                (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                (&self.pooled_bf2, &self.sim_scales, &rows_i, &d_i))?;
            dsv4_launch!(ks.comp, "dsv4_comp_copy_rows_b", stream.stream,
                (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                (&cache_dst, &row0_i, &self.pooled_bf2, &rows_i, &d_i))?;
        } else {
            let nope = (d - spec.rope_dim) as i32;
            let warps = nrows * ((d - spec.rope_dim) / 64);
            // R5b: FP8-packed cache rows FIRST (codes variant, const input — the in-place
            // sim then runs the identical body on identical inputs; dequant == bf16 rows'
            // nope span bitwise, the R5b-1 gate).
            if crate::dsv4_gpu::env_flag_once(crate::opt!("packed-cache")) {
                if let (Some(pc), Some(ps)) = (&self.packed_codes, &self.packed_scales) {
                let nope_u = d - spec.rope_dim;
                let codes_dst = crate::dsv4_gpu::DevPtr {
                    dptr: *pc.device_ptr() + (cache_row0 * nope_u) as u64,
                };
                let scales_dst = crate::dsv4_gpu::DevPtr {
                    dptr: *ps.device_ptr() + (cache_row0 * (nope_u / 64)) as u64,
                };
                dsv4_launch!(ks.comp, "dsv4_comp_act_quant_g64s_b", stream.stream,
                    (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                    (&self.pooled_bf, &codes_dst, &scales_dst, &rows_i, &nope, &d_i))?;
                }
            }
            dsv4_launch!(ks.comp, "dsv4_comp_act_quant_sim_g64s_b", stream.stream,
                (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                (&self.pooled_bf, &self.sim_scales, &rows_i, &nope, &d_i))?;
            dsv4_launch!(ks.comp, "dsv4_comp_copy_rows_b", stream.stream,
                (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                (&cache_dst, &row0_i, &self.pooled_bf, &rows_i, &d_i))?;
        }
        Ok(())
    }

    /// §B.5 prefill (start_pos == 0): full-block pooling + frontier stash + epilogue.
    /// `x` is [s, dim] bf16 (the attn_norm output). Returns the number of pooled
    /// blocks (0 when s < ratio — the reference returns None).
    pub fn prefill<X: Dsv4Arg + ?Sized, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        dev: &Arc<CudaDevice>,
        ks: &CompKernels,
        stream: &CudaStream,
        x: &X,
        s: usize,
        rope: &DevRope,
    ) -> Result<usize> {
        self.prefill_at::<X, I>(dev, ks, stream, x, s, 0, rope)
    }

    /// §B.5 chunked prefill at absolute `start_pos` (0 for chunk 0, >0 for continuation
    /// chunks). When start_pos > 0 and coff == 2, block 0's overlap rows read the carried
    /// frontier state (kv_state/score_state[0..ratio)) via the kernel's `carry` flag —
    /// bitwise-identical to the one-shot prefill over the same span (§12.B.5: the frontier
    /// IS the previous block's overlap rows, stashed by the prior chunk). The cache write
    /// offset and RoPE positions advance to start_pos. Replaces the sequential per-token
    /// decode loop (`forward_tokens`) for chunks 2+ — one batched pool launch instead of
    /// s sequential decode+sync round-trips (the parallel chunk-prefill speedup).
    pub fn prefill_at<X: Dsv4Arg + ?Sized, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        dev: &Arc<CudaDevice>,
        ks: &CompKernels,
        stream: &CudaStream,
        x: &X,
        s: usize,
        start_pos: usize,
        rope: &DevRope,
    ) -> Result<usize> {
        let spec = &self.spec;
        anyhow::ensure!(s <= self.s_max, "prefill s={s} > s_max={}", self.s_max);
        let ratio = spec.ratio;
        let cutoff = s - s % ratio;
        let nb = cutoff / ratio;
        self.gemm_pair(ks, stream, x, s)?;
        let do_stash = (spec.coff() == 2 && cutoff >= ratio) as i32;
        let carry = (start_pos > 0 && spec.coff() == 2) as i32;
        let (si, cutoff_i, ratio_i, d_i, coff_i, nb_i) =
            (s as i32, cutoff as i32, ratio as i32, spec.head_dim as i32, spec.coff() as i32, nb as i32);
        // Pool and stash are split into two launches to avoid a race: block 0 READS
        // kv_state[0..ratio*cd) for the carry while the stash WRITES it. Sequential
        // launches on the blocking stream eliminate the hazard.
        if nb > 0 {
            dsv4_launch!(ks.comp, "dsv4_comp_prefill_pool_b", stream.stream,
                (nb as u32, 1, 1), (spec.head_dim as u32, 1, 1), 0,
                (&self.kv_full, &self.score_full, &self.ape, &self.pooled,
                 &self.kv_state, &self.score_state,
                 &si, &cutoff_i, &ratio_i, &d_i, &coff_i, &nb_i, &carry))?;
        }
        dsv4_launch!(ks.comp, "dsv4_comp_prefill_stash_b", stream.stream,
            (1u32, 1, 1), (spec.head_dim as u32, 1, 1), 0,
            (&self.kv_full, &self.score_full, &self.ape,
             &self.kv_state, &self.score_state,
             &si, &cutoff_i, &ratio_i, &d_i, &coff_i, &do_stash))?;
        if nb > 0 {
            let cache_row0 = start_pos / ratio;
            // positions[b] = start_pos + b*ratio → iota (start_pos, ratio, 1) on device.
            self.epilogue::<I>(dev, ks, stream, rope, nb, (start_pos as i32, ratio as i32, 1), cache_row0)?;
        }
        Ok(nb)
    }

    /// §B.5 decode for ONE token at `start_pos`. `x_row` is [dim] bf16. Returns true
    /// when a compression fired (cache row start_pos//ratio written).
    pub fn decode<X: Dsv4Arg + ?Sized, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        dev: &Arc<CudaDevice>,
        ks: &CompKernels,
        stream: &CudaStream,
        x_row: &X,
        start_pos: usize,
        rope: &DevRope,
    ) -> Result<bool> {
        let spec = &self.spec;
        anyhow::ensure!(start_pos > 0, "decode path requires start_pos > 0");
        self.gemm_pair(ks, stream, x_row, 1)?;
        let (sp_i, ratio_i, d_i, coff_i) =
            (start_pos as i32, spec.ratio as i32, spec.head_dim as i32, spec.coff() as i32);
        dsv4_launch!(ks.comp, "dsv4_comp_decode_b", stream.stream,
            (1u32, 1, 1), (spec.cd() as u32, 1, 1), 0,
            (&self.kv_full, &self.score_full, &self.ape,
             &self.kv_state, &self.score_state, &self.pooled, &self.fire_dev,
             &sp_i, &ratio_i, &d_i, &coff_i))?;
        dev.synchronize().map_err(|e| anyhow!("decode sync: {e}"))?;
        let fired = dev.dtoh_sync_copy(&self.fire_dev).map_err(|e| anyhow!("fire dtoh: {e}"))?[0] != 0;
        if fired {
            // positions = [start_pos + 1 - ratio] (single row) → iota (start, 0, 1).
            self.epilogue::<I>(dev, ks, stream, rope, 1,
                ((start_pos + 1 - spec.ratio) as i32, 0, 1), start_pos / spec.ratio)?;
        }
        Ok(fired)
    }

    /// Chunk/verify path: `s` sequential decode steps from `start_pos` (x is [s, dim]).
    /// Exactly equivalent to one-shot prefill over the same span (§12.B.5 — gated).
    /// Returns the number of fired compressions.
    ///
    /// R4 (verify economics): the old form ran `decode()` per token — a FULL wkv+wgate
    /// GEMM per token (6× weight re-reads at verify) plus a host synchronize per token.
    /// The state machine's per-token state update depends only on that token's kv/score
    /// slot, and the pool (and epilogue) only fire on block completion — so: ONE batched
    /// `gemm_pair` over all s tokens (width-invariant per row vs s sequential s=1 GEMMs —
    /// the existing §12.B.5 gates prove it), then the SAME sequential per-token
    /// `dsv4_comp_decode_b` chain (identical state updates, identical order), fire flags
    /// into a per-token buffer, ONE host sync, and the fired epilogues deferred to the end
    /// (they write only the output cache, never the frontier state). Bitwise-identical to
    /// the old flow by construction; gates are the comp suite + rollback + MECHANICS.
    pub fn forward_tokens<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>, U: dsv4_gpu::Dsv4Buf<u32>>(
        &self,
        dev: &Arc<CudaDevice>,
        ks: &CompKernels,
        stream: &CudaStream,
        x: &X,
        s: usize,
        start_pos: usize,
        rope: &DevRope,
    ) -> Result<usize> {
        let spec = &self.spec;
        let (dim, cd, hd, ratio) = (spec.dim, spec.cd(), spec.head_dim, spec.ratio);
        // 1. ONE batched GEMM for all s tokens (per-row bitwise == s sequential s=1 GEMMs).
        self.gemm_pair(ks, stream, x, s)?;
        // 2. The SAME sequential per-token state-machine chain, fire flags per token.
        let fire_flags = U::alloc_zeros(dev, stream.stream, s).map_err(|e| anyhow!("fire flags alloc: {e}"))?;
        // Per-token pool slots (each decode_b writes its own; only fire rows are consumed).
        // Separate per-call buffer: the struct's `pooled` holds nb_max BLOCK slots, which s
        // can exceed on long continuation chunks.
        let pooled_snap = F::alloc_zeros(dev, stream.stream, s * hd).map_err(|e| anyhow!("pooled snap alloc: {e}"))?;
        let (ratio_i, d_i, coff_i) = (ratio as i32, hd as i32, spec.coff() as i32);
        for j in 0..s {
            let sp_i = (start_pos + j) as i32;
            let kv_row = self.kv_full.slice(j * cd..(j + 1) * cd);
            let sc_row = self.score_full.slice(j * cd..(j + 1) * cd);
            let fire_j = fire_flags.view(j, 1);
            let pooled_j = pooled_snap.view(j * hd, hd);
            dsv4_launch!(ks.comp, "dsv4_comp_decode_b", stream.stream,
                (1u32, 1, 1), (cd as u32, 1, 1), 0,
                (&kv_row, &sc_row, &self.ape,
                 &self.kv_state, &self.score_state, &pooled_j, &fire_j,
                 &sp_i, &ratio_i, &d_i, &coff_i))?;
        }
        // 3. Deferred epilogues for the fired subset (cache-only writes). Fire is
        // POSITION-DETERMINISTIC — the kernel's own condition is ((sp+1)%ratio==0)
        // (dsv4_comp_decode_b, gpu_dsv4_comp.cu §4) — so the host computes the subset
        // itself: NO sync, NO flags readback (was: dev.synchronize() + dtoh per call =
        // one full pipeline drain per layer per token at decode). The stream-ordered
        // kernel sequence is identical (decode_b(j) writes pooled_snap(j) before the
        // epilogue reads it — same stream), so this is bitwise-neutral; the fire_flags
        // buffer stays only because the kernel writes *fire unconditionally.
        let mut fired = 0usize;
        for j in 0..s {
            if (start_pos + j + 1) % ratio == 0 {
                fired += 1;
                // epilogue reads `pooled` at offset 0 — restore this fire's pool there first.
                unsafe {
                    result::memcpy_dtod_async(
                        *self.pooled.device_ptr(),
                        pooled_snap.dptr() + (j * hd * 4) as u64,
                        hd * 4,
                        stream.stream,
                    )
                    .map_err(|e| anyhow!("pooled restore dtod: {e}"))?;
                }
                self.epilogue::<I>(dev, ks, stream, rope, 1,
                    ((start_pos + j + 1 - ratio) as i32, 0, 1),
                    (start_pos + j) / ratio)?;
            }
        }
        Ok(fired)
    }

    // ---- frontier snapshot/restore (§12.B.4) ----

    /// Device-side snapshot of the frontier state (D2D on the compute stream — no host
    /// round-trip, rides the prefix-cache snapshot machinery).
    pub fn snapshot(&self, dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<CompSnapshot> {
        let n = self.kv_state.len();
        let kv = dev.alloc_zeros::<f32>(n).map_err(|e| anyhow!("snap alloc: {e}"))?;
        let sc = dev.alloc_zeros::<f32>(n).map_err(|e| anyhow!("snap alloc: {e}"))?;
        let bytes = n * 4;
        unsafe {
            result::memcpy_dtod_async(*kv.device_ptr(), *self.kv_state.device_ptr(), bytes, stream.stream)
                .map_err(|e| anyhow!("snap kv dtod: {e}"))?;
            result::memcpy_dtod_async(*sc.device_ptr(), *self.score_state.device_ptr(), bytes, stream.stream)
                .map_err(|e| anyhow!("snap score dtod: {e}"))?;
        }
        Ok(CompSnapshot { kv_state: kv, score_state: sc })
    }

    /// Rewind the frontier state to a snapshot (DSpark verify rollback).
    pub fn restore(&self, snap: &CompSnapshot, stream: &CudaStream) -> Result<()> {
        let bytes = self.kv_state.len() * 4;
        unsafe {
            result::memcpy_dtod_async(*self.kv_state.device_ptr(), *snap.kv_state.device_ptr(), bytes, stream.stream)
                .map_err(|e| anyhow!("restore kv dtod: {e}"))?;
            result::memcpy_dtod_async(*self.score_state.device_ptr(), *snap.score_state.device_ptr(), bytes, stream.stream)
                .map_err(|e| anyhow!("restore score dtod: {e}"))?;
        }
        Ok(())
    }

    // ---- test accessors (blocking dtoh) ----

    /// Cache rows [0, rows) as bf16-valued f32.
    pub fn cache_host(&self, dev: &Arc<CudaDevice>, rows: usize) -> Result<Vec<f32>> {
        dev.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        let view = self.cache.slice(0..rows * self.spec.head_dim);
        let mut out = vec![bf16::ZERO; rows * self.spec.head_dim];
        dev.dtoh_sync_copy_into(&view, &mut out).map_err(|e| anyhow!("cache dtoh: {e}"))?;
        Ok(out.iter().map(|b| b.to_f32()).collect())
    }

    /// Frontier state as (kv_state, score_state) fp32 host copies.
    pub fn state_host(&self, dev: &Arc<CudaDevice>) -> Result<(Vec<f32>, Vec<f32>)> {
        dev.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        let kv = dev.dtoh_sync_copy(&self.kv_state).map_err(|e| anyhow!("kv_state dtoh: {e}"))?;
        let sc = dev.dtoh_sync_copy(&self.score_state).map_err(|e| anyhow!("score_state dtoh: {e}"))?;
        Ok((kv, sc))
    }
}

/// Device-side frontier snapshot (see GpuCompressor::snapshot).
pub struct CompSnapshot {
    pub kv_state: S,
    pub score_state: S,
}

/// Full persistent-state snapshot (frontier + cache) for DSpark verify rollback — restores the
/// compressor to an exact earlier sequence position so the committed prefix can be re-advanced
/// (or the state fully rewound). The cache is append-only and stale rows beyond `nb_committed`
/// are ignored, but a full restore guarantees a clean rewind (the forced-mismatch gate compares
/// bitwise). D2D on the compute stream — no host round-trip.
pub struct CompFullSnapshot {
    pub kv_state: S,
    pub score_state: S,
    pub cache: B,
}

impl GpuCompressor {
    /// Set the cache alias — the epilogue writes cache rows to `dptr` instead of `self.cache`.
    /// Used by `Dsv4AttnState::new_state` to alias the attention compressor's cache to the
    /// `kv_cache[win..]` tail (Item 3: eliminates the per-step d2d mirror).
    pub fn set_cache_alias(&mut self, dptr: u64) {
        self.cache_alias_ptr = dptr;
    }

    pub fn snapshot_full(&self, dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<CompFullSnapshot> {
        let n = self.kv_state.len();
        let kv = dev.alloc_zeros::<f32>(n).map_err(|e| anyhow!("full-snap kv alloc: {e}"))?;
        let sc = dev.alloc_zeros::<f32>(n).map_err(|e| anyhow!("full-snap sc alloc: {e}"))?;
        let cn = self.cache_rows * self.spec.head_dim;
        let ca = dev.alloc_zeros::<bf16>(cn).map_err(|e| anyhow!("full-snap cache alloc: {e}"))?;
        let cache_src = if self.cache_alias_ptr != 0 { self.cache_alias_ptr } else { *self.cache.device_ptr() };
        unsafe {
            result::memcpy_dtod_async(*kv.device_ptr(), *self.kv_state.device_ptr(), n * 4, stream.stream)
                .map_err(|e| anyhow!("full-snap kv dtod: {e}"))?;
            result::memcpy_dtod_async(*sc.device_ptr(), *self.score_state.device_ptr(), n * 4, stream.stream)
                .map_err(|e| anyhow!("full-snap sc dtod: {e}"))?;
            result::memcpy_dtod_async(*ca.device_ptr(), cache_src, cn * 2, stream.stream)
                .map_err(|e| anyhow!("full-snap cache dtod: {e}"))?;
        }
        Ok(CompFullSnapshot { kv_state: kv, score_state: sc, cache: ca })
    }

    pub fn restore_full(&self, snap: &CompFullSnapshot, stream: &CudaStream) -> Result<()> {
        let n = self.kv_state.len();
        let cn = self.cache_rows * self.spec.head_dim;
        let cache_dst = if self.cache_alias_ptr != 0 { self.cache_alias_ptr } else { *self.cache.device_ptr() };
        unsafe {
            result::memcpy_dtod_async(*self.kv_state.device_ptr(), *snap.kv_state.device_ptr(), n * 4, stream.stream)
                .map_err(|e| anyhow!("full-restore kv dtod: {e}"))?;
            result::memcpy_dtod_async(*self.score_state.device_ptr(), *snap.score_state.device_ptr(), n * 4, stream.stream)
                .map_err(|e| anyhow!("full-restore sc dtod: {e}"))?;
            result::memcpy_dtod_async(cache_dst, *snap.cache.device_ptr(), cn * 2, stream.stream)
                .map_err(|e| anyhow!("full-restore cache dtod: {e}"))?;
        }
        Ok(())
    }
}

impl GpuIndexer {
    /// Full snapshot of the indexer's persistent state (= its rotate compressor's full state).
    pub fn snapshot_full(&self, dev: &Arc<CudaDevice>, stream: &CudaStream) -> Result<CompFullSnapshot> {
        self.comp.snapshot_full(dev, stream)
    }
    pub fn restore_full(&self, snap: &CompFullSnapshot, stream: &CudaStream) -> Result<()> {
        self.comp.restore_full(snap, stream)
    }
}

// -----------------------------------------------------------------------------------------------
// Streaming indexer top-k (DSV4_LONG_CONTEXT_1M §4 — the 1M enabler)
// -----------------------------------------------------------------------------------------------

/// Streaming top-k scratch, pre-allocated at construction (the 3C production-hardening note —
/// NO per-call allocations on the request path). Peak memory is `s_max · nb_tile · 4` for the
/// score stripe + `s_max · k · 24` for the merge buffers — INDEPENDENT of `nblocks` (context).
/// Per-call usage is dense row-major: the stripe buffer is used as `[s, tc]` (tc = stripe
/// width), the stripe index buffer as `[s, kc]` (kc = min(k, tc)).
pub struct TopkScratch {
    /// Score stripe `[s_max, nb_tile]` f32 (dense `[s, tc]` per stripe).
    pub scores_tile: S,
    stripe_idx: I32Dev,   // [s_max, k] (dense [s, kc] per stripe)
    carry_idx: I32Dev,    // [s_max, k] — the running partial top-k (global block ids)
    carry_scores: S,      // [s_max, k] — ... and their scores (carried: old stripes are gone)
    merged_idx: I32Dev,   // [s_max, 2k] — [carry | stripe+t0]
    merged_scores: S,     // [s_max, 2k] — [carry_scores | stripe scores]
    sel: I32Dev,          // [s_max, k] — merged top-k output (positions into merged)
    /// Stripe width over the block axis (≤ 16384, the dsv4_topk capacity class).
    pub nb_tile: usize,
    k: usize,
    s_max: usize,
}

impl TopkScratch {
    pub fn new(dev: &Arc<CudaDevice>, s_max: usize, k: usize, nb_tile: usize) -> Result<Self> {
        anyhow::ensure!(nb_tile > 0 && nb_tile <= 16384, "nb_tile {nb_tile} outside 1..=16384 (dsv4_topk capacity)");
        anyhow::ensure!(k > 0 && k <= 512, "topk k {k} outside 1..=512");
        Ok(Self {
            scores_tile: dev.alloc_zeros::<f32>(s_max * nb_tile).map_err(|e| anyhow!("scores_tile: {e}"))?,
            stripe_idx: dev.alloc_zeros::<i32>(s_max * k).map_err(|e| anyhow!("stripe_idx: {e}"))?,
            carry_idx: dev.alloc_zeros::<i32>(s_max * k).map_err(|e| anyhow!("carry_idx: {e}"))?,
            carry_scores: dev.alloc_zeros::<f32>(s_max * k).map_err(|e| anyhow!("carry_scores: {e}"))?,
            merged_idx: dev.alloc_zeros::<i32>(s_max * 2 * k).map_err(|e| anyhow!("merged_idx: {e}"))?,
            merged_scores: dev.alloc_zeros::<f32>(s_max * 2 * k).map_err(|e| anyhow!("merged_scores: {e}"))?,
            sel: dev.alloc_zeros::<i32>(s_max * k).map_err(|e| anyhow!("sel: {e}"))?,
            nb_tile,
            k,
            s_max,
        })
    }
}

/// Streaming deterministic top-k over the §B.6 index scores: `q_fwht` [s, NH·HD] bf16 against
/// `kv_cache` [nblocks, HD] bf16 with head `weights` [s, NH] bf16 → the top-`k` global block
/// indices per row, written to `out_idx` [s, k]. Scores stripes of ≤ nb_tile blocks
/// (`dsv4_comp_index_score_tile_b` — bitwise-identical scores to the full-matrix kernel), takes
/// each stripe's deterministic top-k, and merges into the running carry. The merge orders the
/// carry (all globals < t0) BEFORE the stripe (globals ≥ t0), so dsv4_topk's position-asc tie-
/// break IS the global-index-asc tie-break: the final selection (set AND order) is identical to
/// the materialized-matrix top-k (§12.B.2 — gated at 64K/250K in tests/dsv4_comp_test.rs).
/// Peak memory is independent of `nblocks`. `start_pos` + `ratio` give the absolute causal limit.
#[allow(clippy::too_many_arguments)]
pub fn index_topk_streaming(
    ks: &CompKernels,
    stream: &CudaStream,
    q_fwht: &B,
    kv_cache: &B,
    weights: &B,
    scr: &TopkScratch,
    out_idx: &I32Dev,
    s: usize,
    nblocks: usize,
    k: usize,
    start_pos: usize,
    ratio: usize,
) -> Result<()> {
    anyhow::ensure!(s <= scr.s_max, "streaming topk: s={s} > scratch s_max={}", scr.s_max);
    anyhow::ensure!(k <= scr.k, "streaming topk: k={k} > scratch k={}", scr.k);
    anyhow::ensure!(nblocks > 0 && k > 0, "streaming topk: empty range");
    let (s_i, k_i, sp_i, ratio_i) = (s as i32, k as i32, start_pos as i32, ratio as i32);
    let m_i = (2 * k) as i32;
    let grid_sk = (((s * k) + 255) / 256) as u32;
    let mut t0 = 0usize;
    let mut first = true;
    while t0 < nblocks {
        let tc = (nblocks - t0).min(scr.nb_tile);
        let kc = k.min(tc);
        let (t0_i, tc_i, nb_i, kc_i) = (t0 as i32, tc as i32, nblocks as i32, kc as i32);
        // 1. score the stripe (dense [s, tc]) — bitwise == the full matrix's columns [t0, t0+tc)
        //    grid.y > 1 parallelizes the block axis (the 1M decode latency lever).
        let score_grid_y = ((tc + 1023) / 1024).max(1) as u32;
        dsv4_launch!(ks.comp, "dsv4_comp_index_score_tile_b", stream.stream,
            (s as u32, score_grid_y, 1), (256, 1, 1), 0,
            (q_fwht, kv_cache, weights, &scr.scores_tile, &t0_i, &tc_i, &nb_i, &sp_i, &ratio_i))?;
        // 2. stripe top-kc (dense [s, kc] — local block ids, tie-break local-asc == global-asc)
        dsv4_launch!(ks.spine, "dsv4_topk", stream.stream,
            (s as u32, 1, 1), (256, 1, 1), 0,
            (&scr.scores_tile, &scr.stripe_idx, &s_i, &tc_i, &kc_i))?;
        if first {
            // carry := stripe 0 (stripe-0 always has kc == k: k = min(index_topk, nblocks) ≤ tc)
            anyhow::ensure!(kc == k, "stripe-0 kc={kc} < k={k} (impossible: k ≤ nblocks, tc ≥ k)");
            dsv4_launch!(ks.comp, "dsv4_idx_offset_place_b", stream.stream,
                (grid_sk, 1, 1), (256, 1, 1), 0,
                (&scr.carry_idx, &scr.stripe_idx, &s_i, &k_i, &k_i, &0i32, &0i32))?;
            dsv4_launch!(ks.comp, "dsv4_score_gather_b", stream.stream,
                (grid_sk, 1, 1), (256, 1, 1), 0,
                (&scr.carry_scores, &scr.scores_tile, &scr.stripe_idx, &s_i, &k_i, &tc_i))?;
            first = false;
        } else {
            // 3. merged [s, k+kc] = [carry (globals < t0) | stripe+t0 (globals ≥ t0)]
            dsv4_launch!(ks.comp, "dsv4_idx_offset_place_b", stream.stream,
                (grid_sk, 1, 1), (256, 1, 1), 0,
                (&scr.merged_idx, &scr.carry_idx, &s_i, &k_i, &m_i, &0i32, &0i32))?;
            let grid_skc = (((s * kc) + 255) / 256) as u32;
            dsv4_launch!(ks.comp, "dsv4_idx_offset_place_b", stream.stream,
                (grid_skc, 1, 1), (256, 1, 1), 0,
                (&scr.merged_idx, &scr.stripe_idx, &s_i, &kc_i, &m_i, &k_i, &t0_i))?;
            dsv4_launch!(ks.comp, "dsv4_f32_place_b", stream.stream,
                (grid_sk, 1, 1), (256, 1, 1), 0,
                (&scr.merged_scores, &scr.carry_scores, &s_i, &k_i, &m_i, &0i32))?;
            dsv4_launch!(ks.comp, "dsv4_f32_gather_place_b", stream.stream,
                (grid_skc, 1, 1), (256, 1, 1), 0,
                (&scr.merged_scores, &scr.scores_tile, &scr.stripe_idx, &s_i, &kc_i, &m_i, &k_i, &tc_i))?;
            // 4. merged top-k over T = k+kc → sel (carry-first positions == global-idx tie-break)
            let tm_i = (k + kc) as i32;
            dsv4_launch!(ks.spine, "dsv4_topk", stream.stream,
                (s as u32, 1, 1), (256, 1, 1), 0,
                (&scr.merged_scores, &scr.sel, &s_i, &tm_i, &k_i))?;
            // 5. carry := remap(sel) / gather(merged_scores, sel)
            dsv4_launch!(ks.comp, "dsv4_idx_remap_b", stream.stream,
                (grid_sk, 1, 1), (256, 1, 1), 0,
                (&scr.carry_idx, &scr.sel, &scr.merged_idx, &s_i, &k_i, &m_i))?;
            dsv4_launch!(ks.comp, "dsv4_score_gather_b", stream.stream,
                (grid_sk, 1, 1), (256, 1, 1), 0,
                (&scr.carry_scores, &scr.merged_scores, &scr.sel, &s_i, &k_i, &m_i))?;
        }
        t0 += tc;
    }
    // publish: out_idx := carry_idx (both dense [s, k] — one D2D on the compute stream)
    unsafe {
        result::memcpy_dtod_async(*out_idx.device_ptr(), *scr.carry_idx.device_ptr(), s * k * 4, stream.stream)
            .map_err(|e| anyhow!("streaming topk publish dtod: {e}"))?;
    }
    Ok(())
}

// -----------------------------------------------------------------------------------------------
// GpuIndexer — §B.6 on GPU (CSA only)
// -----------------------------------------------------------------------------------------------

/// Device-side indexer. Shares the attention `qr` (input), owns its ratio-4 rotate
/// compressor (its kv_cache IS the indexer kv cache [max_seq//4, 128]) and produces
/// the deterministic top-min(512, end//4) block selections (§12.B.2).
pub struct GpuIndexer {
    pub comp: GpuCompressor,
    pub n_heads: usize,     // 64
    pub head_dim: usize,    // 128
    pub q_lora_rank: usize, // 1024
    pub index_topk: usize,  // 512
    // wq_b fp8_bsb operands (MMA-repacked codes + UE8M0 block scales)
    wq_b_wt: CudaSlice<u8>,
    wq_b_sb: CudaSlice<u8>,
    weights_proj: B, // [64, dim] bf16
    wscale: f32,
    // scratch
    qr_codes: CudaSlice<u8>,  // [16, q_lora_rank]
    qr_scales: CudaSlice<u8>, // [16, q_lora_rank/128]
    q_tile: B,                // [16, n_heads*head_dim] fp8_bsb out (per-tile scratch)
    q_rot: B,                 // [s_max*nh, hd] post-rope
    q_fwht: B,                // [s_max*nh, hd] post-fwht/fp4-sim
    q_sim_scales: CudaSlice<u8>,
    weights: B,               // [s_max, nh]
    /// Streaming top-k scratch (score stripe + merge buffers — peak memory independent of
    /// context; replaces the [s_max, nb_max] full score matrix, DSV4_LONG_CONTEXT_1M §4).
    pub topk_scratch: TopkScratch,
    // ---- small-s full-matrix path (decode/verify, s ≤ DEC_FULL_SMAX) ----
    // At s ≤ 16 the materialized score matrix is ≤ 16·nb_max·4 (16 MB at 1M) — memory is NOT
    // the constraint at decode widths, latency is, and the hierarchical full-matrix top-k is
    // ~1.6× faster than streaming at 250K blocks (measured, tests/dsv4_comp_test.rs §9c).
    // Selections are IDENTICAL across paths (same dot8 scores, same value-desc/idx-asc order).
    scores_full: S,           // [DEC_FULL_SMAX, nb_max]
    chunk_scratch: S,         // [DEC_FULL_SMAX, 16384] — strided chunk copy
    stage1_idx: I32Dev,       // [DEC_FULL_SMAX, m_max]
    gathered: S,              // [DEC_FULL_SMAX, m_max]
    stage2_idx: I32Dev,       // [DEC_FULL_SMAX, k]
    /// stage1 row width: ceil(nb_max/16384) chunks × k.
    m_max: usize,
    idx: I32Dev,              // [s_max, index_topk]
    s_max: usize,
    /// indexer kv-cache row capacity (max_seq_len // 4).
    pub nb_max: usize,
}

/// Decode/verify width cap for the full-matrix top-k path (see GpuIndexer::scores_full).
pub const DEC_FULL_SMAX: usize = 16;
/// dsv4_topk capacity class (T ≤ 16384) — the stripe/chunk width for both top-k paths.
const TOPK_CHUNK: usize = 16384;

impl GpuIndexer {
    /// `wq_b_wt`/`wq_b_sb`: the indexer wq_b [nh*hd, qlr] in fp8_bsb form
    /// (quant::repack_fp8_mma codes + [M/128, K/128] UE8M0 scales). `weights_proj`
    /// [nh, dim] bf16-valued f32 (checkpoint dtype). The head-weights scale is computed
    /// internally with the CPU reference's exact expression
    /// `((hd as f64).powf(-0.5) * (nh as f64).powf(-0.5)) as f32`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dev: &Arc<CudaDevice>,
        ks: &CompKernels,
        stream: &CudaStream,
        dim: usize,
        rope_dim: usize,
        q_lora_rank: usize,
        n_heads: usize,
        head_dim: usize,
        index_topk: usize,
        comp_w: &dsv4_cpu::CompressorWeights, // rotate=true, ratio 4, head_dim 128
        norm_eps: f32,
        wq_b_wt: &[u8],
        wq_b_sb: &[u8],
        weights_proj: &[f32],
        max_seq_len: usize,
        s_max: usize,
    ) -> Result<Self> {
        anyhow::ensure!(n_heads == 64 && head_dim == 128, "indexer score kernel is NH=64/HD=128 specialized");
        let spec = CompSpec::indexer(dim, rope_dim);
        anyhow::ensure!(CompSpec::from(comp_w).cd() == spec.cd() && comp_w.rotate, "indexer compressor spec mismatch");
        let nb_max = max_seq_len / 4;
        let comp = GpuCompressor::new(dev, ks, stream, spec, comp_w, norm_eps, nb_max, s_max)?;
        let wscale = ((head_dim as f64).powf(-0.5) * (n_heads as f64).powf(-0.5)) as f32;
        let wp: Vec<bf16> = weights_proj.iter().map(|&v| bf16::from_f32(v)).collect();
        anyhow::ensure!(wp.len() == n_heads * dim, "weights_proj shape");
        Ok(GpuIndexer {
            comp,
            n_heads,
            head_dim,
            q_lora_rank,
            index_topk,
            wq_b_wt: dev.htod_sync_copy(wq_b_wt).map_err(|e| anyhow!("wq_b wt htod: {e}"))?,
            wq_b_sb: dev.htod_sync_copy(wq_b_sb).map_err(|e| anyhow!("wq_b sb htod: {e}"))?,
            weights_proj: dev.htod_sync_copy(&wp).map_err(|e| anyhow!("weights_proj htod: {e}"))?,
            wscale,
            qr_codes: dev.alloc_zeros::<u8>(16 * q_lora_rank).map_err(|e| anyhow!("qr codes: {e}"))?,
            qr_scales: dev.alloc_zeros::<u8>(16 * (q_lora_rank / 128)).map_err(|e| anyhow!("qr scales: {e}"))?,
            q_tile: dev.alloc_zeros::<bf16>(16 * n_heads * head_dim).map_err(|e| anyhow!("q tile: {e}"))?,
            q_rot: dev.alloc_zeros::<bf16>(s_max * n_heads * head_dim).map_err(|e| anyhow!("q_rot: {e}"))?,
            q_fwht: dev.alloc_zeros::<bf16>(s_max * n_heads * head_dim).map_err(|e| anyhow!("q_fwht: {e}"))?,
            q_sim_scales: dev.alloc_zeros::<u8>(s_max * n_heads * (head_dim / 32)).map_err(|e| anyhow!("q sim scales: {e}"))?,
            weights: dev.alloc_zeros::<bf16>(s_max * n_heads).map_err(|e| anyhow!("weights: {e}"))?,
            topk_scratch: TopkScratch::new(dev, s_max, index_topk, 16384)?,
            scores_full: dev.alloc_zeros::<f32>(DEC_FULL_SMAX * nb_max).map_err(|e| anyhow!("scores_full: {e}"))?,
            chunk_scratch: dev.alloc_zeros::<f32>(DEC_FULL_SMAX * TOPK_CHUNK).map_err(|e| anyhow!("chunk_scratch: {e}"))?,
            stage1_idx: dev.alloc_zeros::<i32>(DEC_FULL_SMAX * nb_max.div_ceil(TOPK_CHUNK) * index_topk.max(1)).map_err(|e| anyhow!("stage1_idx: {e}"))?,
            gathered: dev.alloc_zeros::<f32>(DEC_FULL_SMAX * nb_max.div_ceil(TOPK_CHUNK) * index_topk.max(1)).map_err(|e| anyhow!("gathered: {e}"))?,
            stage2_idx: dev.alloc_zeros::<i32>(DEC_FULL_SMAX * index_topk.max(1)).map_err(|e| anyhow!("stage2_idx: {e}"))?,
            m_max: nb_max.div_ceil(TOPK_CHUNK) * index_topk,
            idx: dev.alloc_zeros::<i32>(s_max * index_topk.max(1)).map_err(|e| anyhow!("idx: {e}"))?,
            s_max,
            nb_max,
        })
    }

    /// §B.6 forward. `x` [s, dim] bf16 (attn_norm out), `qr` [s, q_lora_rank] bf16 (the
    /// attention's shared q-lora latent), `offset` = s (prefill) / window (decode).
    /// `f_bsb` is the cudarc handle for `gemm_dsv4_fp8_bsb2` (gpu_batch module; R3A.1 moved
    /// production to the two-tiles-per-CTA pair kernel — identical per-element chains).
    /// Returns k = min(index_topk, end//4) (0 → empty selection, nothing written).
    /// Takes &mut self only for the fp8_bsb tile scratch (the cudarc wrapper needs a
    /// concrete &mut CudaSlice); all semantics are pure functions of the prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn forward<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>, U: dsv4_gpu::Dsv4Buf<u32>>(
        &mut self,
        dev: &Arc<CudaDevice>,
        ks: &CompKernels,
        stream: &CudaStream,
        f_bsb: &CudaFunction,
        x: &X,
        qr: &X,
        s: usize,
        start_pos: usize,
        offset: usize,
        rope: &DevRope,
    ) -> Result<usize> {
        let (nh, hd, qlr, ratio) = (self.n_heads, self.head_dim, self.q_lora_rank, 4usize);
        let dim = self.comp.spec.dim;
        anyhow::ensure!(s <= self.s_max, "indexer s={s} > s_max");
        let end_pos = start_pos + s;
        // 1. own compressor updates the indexer kv cache. Batched pool when the chunk
        //    boundary aligns (start_pos==0 OR (s>=ratio AND start_pos%ratio==0)) — the
        //    parallel chunk-prefill speedup. Sequential decode otherwise (verify/decode with
        //    small s or non-aligned start_pos — the batched pool cannot run the per-token
        //    state machine: slot write, fire check, CSA shift).
        let ratio = self.comp.spec.ratio;
        if start_pos == 0 || (s >= ratio && start_pos % ratio == 0) {
            self.comp.prefill_at::<X, I>(dev, ks, stream, x, s, start_pos, rope)?;
        } else {
            self.comp.forward_tokens::<X, F, I, U>(dev, ks, stream, x, s, start_pos, rope)?;
        }
        let nblocks = end_pos / ratio;
        let k = self.index_topk.min(nblocks);
        if k == 0 {
            return Ok(0); // prefill s < ratio: no committed blocks, empty selection
        }
        // 2. q = wq_b(qr) in ≤16-row tiles (fp8_bsb host contract), tiles → q_rot
        for t0 in (0..s).step_by(16) {
            let tn = (s - t0).min(16);
            let rows_i = tn as i32;
            let n_i = qlr as i32;
            let qr_tile = qr.view(t0 * qlr, tn * qlr);
            let warps = tn * (qlr / 128);
            dsv4_launch!(ks.spine, "dsv4_act_quant_g128", stream.stream,
                (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                (&qr_tile, &self.qr_codes, &self.qr_scales, &rows_i, &n_i))?;
            dsv4_gpu::launch_fp8_bsb2(
                f_bsb, stream, &mut self.q_tile, &self.wq_b_wt, &self.wq_b_sb,
                &self.qr_codes, &self.qr_scales, nh * hd, qlr, tn, None,
            )?;
            // tile → q_rot rows [t0*nh*hd, (t0+tn)*nh*hd) on the compute stream
            unsafe {
                result::memcpy_dtod_async(
                    *self.q_rot.device_ptr() + (t0 * nh * hd * 2) as u64,
                    *self.q_tile.device_ptr(),
                    tn * nh * hd * 2,
                    stream.stream,
                )
                .map_err(|e| anyhow!("q tile dtod: {e}"))?;
            }
        }
        // 3. RoPE last 64 dims on [s*nh, hd] rows (position shared across a token's heads)
        let rows = s * nh;
        let pos_dev = crate::dsv4_gpu::iota_positions::<I>(dev, &ks.spine, stream, start_pos as i32, 1, nh as i32, rows)?;
        let (rows_i, hd_i, rd_i, inv0) = (rows as i32, hd as i32, self.comp.spec.rope_dim as i32, 0i32);
        dsv4_launch!(ks.spine, "dsv4_rope_last_b", stream.stream,
            (((rows + 7) / 8) as u32, 1, 1), (256, 1, 1), 0,
            (&self.q_rot, &rope.cos, &rope.sin, &pos_dev, &rows_i, &hd_i, &rd_i, &inv0))?;
        // 4. Hadamard + FP4 sim (q_fwht holds the result)
        dsv4_launch!(ks.spine, "dsv4_fwht_rotate", stream.stream,
            (((rows + 7) / 8) as u32, 1, 1), (256, 1, 1), 0,
            (&self.q_rot, &self.q_fwht, &rows_i))?;
        let warps = rows * (hd / 32);
        dsv4_launch!(ks.spine, "dsv4_fp4_act_quant_sim", stream.stream,
            (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&self.q_fwht, &self.q_sim_scales, &rows_i, &hd_i))?;
        // 5. head weights: weights_proj(x) → bf16 → ×wscale → bf16
        {
            let (si, ki, ni) = (s as i32, dim as i32, nh as i32);
            dsv4_launch!(ks.comp, "dsv4_comp_gemm_tree_bf16w_bf16out_b", stream.stream,
                (nh as u32, s as u32, 1), (256, 1, 1), 0,
                (x, &self.weights_proj, &self.weights, &si, &ki, &ni))?;
            let nw = (s * nh) as i32;
            dsv4_launch!(ks.comp, "dsv4_comp_wscale_b", stream.stream,
                (((nw + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                (&self.weights, &self.wscale, &nw))?;
        }
        // 6-7. deterministic top-k, two regimes with IDENTICAL selections (same dot8 score
        // chain, same value-desc / global-idx-asc total order, §12.B.2 — cross-path SET+order
        // equality gated in tests/dsv4_comp_test.rs §9):
        //   - s ≤ 16 (decode/verify): the materialized score matrix is ≤ 16 MB even at 1M —
        //     memory is not the constraint; the full-matrix hierarchical top-k is ~1.6× faster
        //     than streaming at 250K blocks (measured, §9c).
        //   - s > 16 (prefill chunks): streaming over ≤nb_tile-block stripes — peak memory
        //     s·nb_tile·4, INDEPENDENT of context length (the 1M enabler). Supersedes the 3C
        //     hierarchical path (whose helpers are the merge primitives reused in both arms).
        let (nb_i, sp_i, ratio_i) = (nblocks as i32, start_pos as i32, ratio as i32);
        if s <= DEC_FULL_SMAX {
            self.topk_full_matrix(ks, stream, s, nblocks, k, start_pos)?;
        } else {
            index_topk_streaming(
                ks, stream, &self.q_fwht, &self.comp.cache, &self.weights,
                &self.topk_scratch, &self.idx, s, nblocks, k, start_pos, ratio,
            )?;
        }
        let s_i = s as i32;
        let k_i = k as i32;
        // 8. re-mask / offset (per-row block-causal limit inside)
        let off_i = offset as i32;
        dsv4_launch!(ks.comp, "dsv4_comp_idx_remask_b", stream.stream,
            (((s * k + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&self.idx, &s_i, &k_i, &sp_i, &ratio_i, &nb_i, &off_i))?;
        Ok(k)
    }

    /// Small-s (decode/verify) top-k: materialize the [s, nblocks] score matrix (≤ 16 MB at
    /// 1M for s ≤ 16) and run the 3C hierarchical top-k (≤16384 chunks + gather/merge) with
    /// PRE-ALLOCATED scratch (the 3C production-hardening note — no per-call allocs). The
    /// selection is identical to the streaming path's (same dot8 scores, same value-desc /
    /// global-idx-asc total order — gated §9c in tests/dsv4_comp_test.rs).
    fn topk_full_matrix(&mut self, ks: &CompKernels, stream: &CudaStream, s: usize, nblocks: usize, k: usize, start_pos: usize) -> Result<()> {
        let (s_i, k_i, nb_i, sp_i, ratio_i) = (s as i32, k as i32, nblocks as i32, start_pos as i32, 4i32);
        // grid.y > 1 parallelizes the block axis — the single-CTA scorer dominated 1M
        // decode at ~33 ms/250K blocks; tiling across CTAs fills the GPU's SMs.
        let score_grid_y = ((nblocks + 1023) / 1024).max(1) as u32;
        // R5a-2 (env-hatched, --packed-cache=1): read the FP4-packed cache (¼ bytes, same
        // values bit-for-bit — the dsv4_comp_index_score_fp4_b chains are unchanged).
        // Requires the packed buffers to exist (rotate compressor) — else the bf16 reader.
        let use_fp4 = crate::dsv4_gpu::env_flag_once(crate::opt!("packed-cache"))
            && self.comp.packed_codes.is_some() && self.comp.packed_scales.is_some();
        if use_fp4 {
            let pc = self.comp.packed_codes.as_ref().unwrap();
            let ps = self.comp.packed_scales.as_ref().unwrap();
            dsv4_launch!(ks.comp, "dsv4_comp_index_score_fp4_b", stream.stream,
                (s as u32, score_grid_y, 1), (256, 1, 1), 0,
                (&self.q_fwht, pc, ps, &self.weights, &self.scores_full, &nb_i, &sp_i, &ratio_i))?;
        } else {
            dsv4_launch!(ks.comp, "dsv4_comp_index_score_b", stream.stream,
                (s as u32, score_grid_y, 1), (256, 1, 1), 0,
                (&self.q_fwht, &self.comp.cache, &self.weights, &self.scores_full, &nb_i, &sp_i, &ratio_i))?;
        }
        if nblocks <= TOPK_CHUNK {
            dsv4_launch!(ks.spine, "dsv4_topk", stream.stream,
                (s as u32, 1, 1), (256, 1, 1), 0,
                (&self.scores_full, &self.idx, &s_i, &nb_i, &k_i))?;
            return Ok(());
        }
        // Hierarchical: chunk into ≤16384 (strided copy into the dense chunk scratch), per-chunk
        // topk, offset+place, gather scores, stage-2 topk, remap to global block ids.
        let n_chunks = nblocks.div_ceil(TOPK_CHUNK);
        let m = n_chunks * k;
        anyhow::ensure!(m <= self.m_max, "stage1 overflow: m={m} > m_max={} (nb_max grew?)", self.m_max);
        for c in 0..n_chunks {
            let base = c * TOPK_CHUNK;
            let cs = (nblocks - base).min(TOPK_CHUNK);
            for row in 0..s {
                unsafe {
                    result::memcpy_dtod_async(
                        *self.chunk_scratch.device_ptr() + (row * cs * 4) as u64,
                        (*self.scores_full.device_ptr()) + ((row * nblocks + base) * 4) as u64,
                        cs * 4,
                        stream.stream,
                    )
                    .map_err(|e| anyhow!("chunk score dtod: {e}"))?;
                }
            }
            let (cs_i, ck_i) = (cs as i32, k as i32);
            dsv4_launch!(ks.spine, "dsv4_topk", stream.stream,
                (s as u32, 1, 1), (256, 1, 1), 0,
                (&self.chunk_scratch, &self.stage2_idx, &s_i, &cs_i, &ck_i))?;
            let (m_i, col_off_i, off_i) = (m as i32, (c * k) as i32, base as i32);
            dsv4_launch!(ks.comp, "dsv4_idx_offset_place_b", stream.stream,
                (((s * k + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                (&self.stage1_idx, &self.stage2_idx, &s_i, &ck_i, &m_i, &col_off_i, &off_i))?;
        }
        let (m_i, nb_i2) = (m as i32, nblocks as i32);
        dsv4_launch!(ks.comp, "dsv4_score_gather_b", stream.stream,
            (((s * m + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&self.gathered, &self.scores_full, &self.stage1_idx, &s_i, &m_i, &nb_i2))?;
        dsv4_launch!(ks.spine, "dsv4_topk", stream.stream,
            (s as u32, 1, 1), (256, 1, 1), 0,
            (&self.gathered, &self.stage2_idx, &s_i, &m_i, &k_i))?;
        dsv4_launch!(ks.comp, "dsv4_idx_remap_b", stream.stream,
            (((s * k + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&self.idx, &self.stage2_idx, &self.stage1_idx, &s_i, &k_i, &m_i))?;
        Ok(())
    }

    /// Host copy of the topk indices [s, k] (blocking).
    pub fn idx_host(&self, dev: &Arc<CudaDevice>, s: usize, k: usize) -> Result<Vec<i32>> {
        dev.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        let view = self.idx.slice(0..s * k);
        let mut out = vec![0i32; s * k];
        dev.dtoh_sync_copy_into(&view, &mut out).map_err(|e| anyhow!("idx dtoh: {e}"))?;
        Ok(out)
    }

    /// Device-side reference to the full topk index buffer [s_max, index_topk]. Lane 3C
    /// uses this to place the indexer's [s, k] output into the unified window++compress
    /// index list (the `dsv4_idxs_place_b` kernel reads from this buffer).
    pub fn idx_dev(&self) -> &I32Dev {
        &self.idx
    }

    /// Host copy of the raw (pre-topk, post-mask) index scores [s, nblocks] bf16-valued f32.
    /// Reads the buffer the LAST forward wrote: `scores_full` for s ≤ DEC_FULL_SMAX (any
    /// nblocks ≤ nb_max), else the streaming tile — valid only for nblocks ≤ nb_tile (a
    /// single stripe; multi-stripe forwards leave only the LAST stripe in it).
    pub fn scores_host(&self, dev: &Arc<CudaDevice>, s: usize, nblocks: usize) -> Result<Vec<f32>> {
        dev.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        let (view, n) = if s <= DEC_FULL_SMAX {
            anyhow::ensure!(nblocks <= self.nb_max, "scores_host: nblocks {nblocks} > nb_max {}", self.nb_max);
            (self.scores_full.slice(0..s * nblocks), s * nblocks)
        } else {
            anyhow::ensure!(
                nblocks <= self.topk_scratch.nb_tile,
                "scores_host: nblocks {nblocks} > nb_tile {} (multi-stripe: full matrix no longer exists)",
                self.topk_scratch.nb_tile
            );
            (self.topk_scratch.scores_tile.slice(0..s * nblocks), s * nblocks)
        };
        let mut out = vec![0.0f32; n];
        dev.dtoh_sync_copy_into(&view, &mut out).map_err(|e| anyhow!("scores dtoh: {e}"))?;
        Ok(out)
    }

    /// Host copy of the indexer q [rows, 128] post-rope/FWHT/FP4-sim (diagnostics).
    pub fn q_host(&self, dev: &Arc<CudaDevice>, rows: usize) -> Result<Vec<f32>> {
        dev.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        let view = self.q_fwht.slice(0..rows * self.head_dim);
        let mut out = vec![bf16::ZERO; rows * self.head_dim];
        dev.dtoh_sync_copy_into(&view, &mut out).map_err(|e| anyhow!("q dtoh: {e}"))?;
        Ok(out.iter().map(|b| b.to_f32()).collect())
    }
}
