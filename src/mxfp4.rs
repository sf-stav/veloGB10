//! MXFP4-native mode support (Phase 0: --probe-mxfp4 standalone microbench).
//!
//! Storage stays NVFP4 (packed E2M1 nibbles + UE4M3 per-16-element scales + per-tensor fp32
//! global scale) exactly as shipped. The only artifact-side change is a LOSSLESS byte
//! permutation of the production MMA-repacked buffers into the fragment/scale layout the
//! sm_121a native instruction expects:
//!
//!   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
//!       .f32.e2m1.e2m1.f32.ue4m3
//!
//! (SASS: OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X). The layouts below were EMPIRICALLY VERIFIED
//! on GB10 by the calibration probe in kernels/mxfp4_bench.cu's header comment (1899/1899
//! checks pass, 2026-08-06).
//!
//! Production input layouts (quant.rs repack_nvfp4_mma, per 16-row tile):
//!   wt: 128 B/tile, u32 (tile, lane), byte j = elements of row (g + 8*(j&1)) at columns
//!       2t + 8*(j>>1) and +1 (low/high nibble). lane = g*4 + t, g = r&7, t = ((c&7)>>1).
//!   st: 16 B/tile, byte r = UE4M3 scale of row r of the tile's K-block.
//!   gs: one fp32 per 16-row tile (segment-local reciprocal tensor scale) — unchanged.
//!
//! OMMA output layouts (per 16-row tile mt, 64-K kstep ks):
//!   Aimg: 512 B/(tile,kstep) = 128 u32, u32 = (mt*nks+ks)*128 + lane*4 + r. Byte m' of that
//!       u32 = production byte (tile mt*nblk + ks*4 + 2*(r>>1) + (t>>1), lane g*4 + m',
//!       byte (r&1) | ((t&1)<<1)) — nibble positions preserved, pure byte gather.
//!   SFAw: 64 B/(tile,kstep) = 16 u32, u32 = (mt*nks+ks)*16 + g*2 + (t&1), valid t<=1; byte v
//!       = production scale byte (mt*nblk + ks*4 + v, row g + 8*(t&1)).
//!   Bp/SFB (activations, 8 token rows x K, per 64-K kstep): Bp u32 (ks*32+lane)*2 + r: nibble
//!       j = e2m1 code of X[token g][ks*64 + 8t + j + 32r]; SFB u32 ks*32+lane (t==0 only):
//!       byte v = UE4M3 scale of (token g, kblock ks*4 + v). Only lane t==0 is read by HW.

use half::bf16;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr};
use cudarc::nvrtc::Ptx;

/// A tensor's OMMA repack as the native dispatch sees it. Two ownership modes:
/// - `Owned`: the OMMA bytes live ONLY here (dual-layout — the W::Nvfp4 holds the STANDARD
///   repack; this twin is separate). `shard_mxfp4_*` must dtoh+reshard these under TP attach.
/// - `Ptr`: economy mode — the W::Nvfp4's qweight/scales ARE the OMMA repack, so we store only
///   the device POINTERS (no clone). `CudaSlice::clone` deep-copies device memory, so cloning
///   here would double the resident footprint and OOM the box on big models (the hy3 84 GB
///   rank-local → 168 GB). Economy tensors are sharded at LOAD (tp_sharded_at_load), so the TP
///   attach's `shard_mxfp4_*` never sees a `Ptr` entry.
pub enum OmmaEntry {
    Owned(CudaSlice<u8>, CudaSlice<u8>),
    Ptr(u64, u64),
}

impl OmmaEntry {
    /// The (aimg, sfa) device pointers the native GEMM / prefill-dequant kernels launch against.
    pub fn ptrs(&self) -> (u64, u64) {
        match self {
            OmmaEntry::Owned(a, s) => (*a.device_ptr() as u64, *s.device_ptr() as u64),
            OmmaEntry::Ptr(a, s) => (*a, *s),
        }
    }
}

/// MXFP4-native serving-mode state (`--mxfp4=on`). Present iff the mode is on; `None` = the
/// default bf16 chain runs (byte-identical contract, non-destruction §6).
pub struct Mxfp4State {
    /// Per-tensor OMMA repacks keyed by the `W::Nvfp4` qweight DEVICE POINTER — the only
    /// stable identity a dispatch site has. Built at load (workers), never mutated after
    /// (except the TP attach's `shard_mxfp4_*` re-key).
    pub w: HashMap<u64, OmmaEntry>,
    /// Sensitive-tensor allowlist (design §8.1): qweight ptrs that run the bf16 chain even in
    /// native mode (GB10_MXFP4_MTP_BF16 — the MTP head's own activation quant erodes draft
    /// acceptance). Explicit set — a tensor missing from BOTH maps is a loader bug, panics.
    /// RwLock: runtime-mutable (the cross-chain probe --probe-mxfp4-xchain flips the whole set
    /// between passes; Phase B's sensitive-site allowlist extension needs it too). Production
    /// costs one read lock per GEMM (~ns, semantics unchanged).
    pub allow_bf16: RwLock<HashSet<u64>>,
    /// Mixer projections cleared for the W4A4 v2 PREFILL GEMM despite being on the bf16
    /// chain for decode/verify (--pf-mixer4 at load: 'safe' = out_proj/o_proj/qkv_proj,
    /// 'all' = + in_proj; unset = none). Gate-arbitrated 2026-08-26: 'all' MISMATCHes the
    /// losslessness fuzz at ctx 7418 (same seed GREEN without it) — R4's ruling holds.
    pub mixer4: RwLock<HashSet<u64>>,
    /// Activation-pack scratch: Bp/SFB for MXFP4_GROUPS eight-token groups at the max K.
    pub bp: CudaSlice<u32>,
    pub sfb: CudaSlice<u32>,
    /// (gemv, quant) from the gpu_mxfp4 module (sm_121a).
    pub gemv: CudaFunction,
    pub quant: CudaFunction,
    /// Prefill dequant (mxfp4_dequant_tiled_b — OMMA-layout element accessor): the prefill path
    /// dequantizes fp4 weights to bf16 scratch before cuBLAS. MUST read the OMMA repack (not the
    /// standard qweight bytes — in dual-layout mode those are the standard MMA repack, in economy
    /// mode they ARE the repack; `st.w` resolves it either way).
    pub dequant: CudaFunction,
    /// Economy-mode embed gather (OMMA-layout element access — embed_gather_fp4_tiled_b reads
    /// the standard layout, which is never uploaded in economy mode).
    pub embed: CudaFunction,
    /// MoE native path (122B grouped/slot experts): kernel handles + scratch from the
    /// gpu_mxfp4_moe module. `moe_bp/moe_sfb` sized for MOE_GROUPS_MAX 8-token groups at the
    /// max K (the grouped arm's worst case: ppad_max = MAX_VERIFY·topk + ne·16 rows).
    pub moe_gemv: CudaFunction,      // mxfp4_gemm_moe_grouped_native_b
    pub moe_slot: CudaFunction,      // mxfp4_gemm_moe_slot_native_b
    pub quant_ng: CudaFunction,      // mxfp4_quant_pack_ng_b
    pub moe_bp: CudaSlice<u32>,
    pub moe_sfb: CudaSlice<u32>,
    /// Fused activation-quant GEMMs (EXPERT_FUSED_QUANT_RESPONSE.md — F1/F2/F3). Each replaces
    /// the quant (+silu) + GEMM pair at its launch sites when --mxfp4-fused is not "0";
    /// byte-identical C and Bp/SFB fragments by construction (§3, §7). The unfused handles
    /// above stay live for the A/B escape. Suffixes: _b1/_b8/_b16 = NR (dense), _b0/_b1 =
    /// XSILU off/on (slot/grouped MoE).
    pub gemv_fused1: CudaFunction,   // mxfp4_gemv_native_fused_b1  (dense, N==1)
    pub gemv_fused8: CudaFunction,   // mxfp4_gemv_native_fused_b8  (dense, 2..=8)
    pub gemv_fused16: CudaFunction,  // mxfp4_gemv_native_fused_b16 (dense, 9..=16)
    pub moe_slot_fused0: CudaFunction,   // mxfp4_gemm_moe_slot_fused_b0   (gu: X = x)
    pub moe_slot_fused1: CudaFunction,   // mxfp4_gemm_moe_slot_fused_b1   (dn: silu in stage)
    pub moe_grouped_fused0: CudaFunction,  // mxfp4_gemm_moe_grouped_fused_b0 (gu: X = x_perm)
    pub moe_grouped_fused1: CudaFunction,  // mxfp4_gemm_moe_grouped_fused_b1 (dn: silu in stage)
    /// P4 B2 W4A4 prefill path (kernels/gpu_mxfp4.cu, 2026-08-18): quant (K-B) + OMMA GEMM
    /// (K-A, the perf6 lineage). Used by gemm_quant_prefill when --mxfp4-prefill is set
    /// and batch >= MXFP4_PREFILL_MIN_BATCH. Scratch: pf_bq/pf_sb sized for PREFILL_CHUNK
    /// rows at MXFP4_MAX_K.
    pub pf_quant: CudaFunction,      // mxfp4_quant_prefill_b
    pub pf_gemm: CudaFunction,       // mxfp4_gemm_prefill_b
    /// v2 (2026-08-25, PLAN/13): OMMA-native fragment-order prefill pair at 213-253 TF/s.
    /// Reads the RESIDENT Aimg/SFAw directly (no repack); acts via the batched OMMA packer
    /// (arithmetic statement-identical to the decode packer). Same env flag, same gates.
    pub pf_quant2: CudaFunction,     // mxfp4_quant_pack_prefill_b
    pub pf_gemm2: CudaFunction,      // mxfp4_gemm_prefill_v2_b
    pub pf_repack: CudaFunction,     // mxfp4_repack_rm_b (tiled -> row-major, once per tensor)
    pub pf_bq: CudaSlice<u8>,
    pub pf_sb: CudaSlice<u8>,
    /// Row-major repack cache keyed by qweight device ptr -> (wrm, srm, m, k). Built lazily
    /// on first prefill use of the tensor; guarded by pf_rm_lock.
    pub pf_rm: RwLock<HashMap<u64, (CudaSlice<u8>, CudaSlice<u8>, usize, usize)>>,
    pub pf_rm_lock: std::sync::Mutex<()>,
}

/// Minimum prefill batch (tokens) for the W4A4 OMMA path; below it, dequant+cuBLAS wins.
pub const MXFP4_PREFILL_MIN_BATCH: usize = 512;

/// Upper bound on the largest K any fp4 linear can have in this family (27B down-proj 17408;
/// fused heads smaller). Scratch is sized from this; a bigger K panics loudly at dispatch.
pub const MXFP4_MAX_K: usize = 32768;
/// Eight-token OMMA groups for N <= 16 (Phase 2 verify width; decode always uses group 0).
pub const MXFP4_GROUPS: usize = 2;

impl Mxfp4State {
    /// Loads the gpu_mxfp4 module (sm_121a: the mxf4nvf4 mma and cvt.e2m1x2 reject plain
    /// sm_121), runs the KERNEL_BUILD_ID handshake (new kernels join the manifest, never
    /// weakened), allocates the pack scratch, and adopts the per-tensor repacks.
    pub fn build(on: bool, omma_map: HashMap<u64, OmmaEntry>,
                 allow_bf16: HashSet<u64>, mixer4: HashSet<u64>,
                 dev: &Arc<CudaDevice>,
                 cfg: &crate::qwen::Config) -> anyhow::Result<Option<Self>> {
        if !on {
            return Ok(None);
        }
        let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_mxfp4.ptx")?);
        dev.load_ptx(ptx, "gpu_mxfp4", &["mxfp4_gemv_native_b", "mxfp4_quant_pack_b",
                                         "mxfp4_gemv_native_fused_b1", "mxfp4_gemv_native_fused_b8",
                                         "mxfp4_gemv_native_fused_b16",
                                         "mxfp4_dequant_tiled_b", "mxfp4_embed_gather_tiled_b",
                                         "mxfp4_quant_prefill_b", "mxfp4_gemm_prefill_b", "mxfp4_repack_rm_b",
                                         "mxfp4_quant_pack_prefill_b", "mxfp4_gemm_prefill_v2_b",
                                         "kernel_build_id"])?;
        crate::gpu::GpuModel::assert_kernel_build_id(dev, "gpu_mxfp4")?;
        let nks_max = MXFP4_MAX_K / 64;
        let bp = dev.alloc_zeros::<u32>(MXFP4_GROUPS * nks_max * 64).unwrap();
        let sfb = dev.alloc_zeros::<u32>(MXFP4_GROUPS * nks_max * 32).unwrap();
        // MoE module: grouped/slot GEMVs + the N-group quant. Sized from the grouped arm's worst
        // case (ppad_max = MAX_VERIFY·topk + ne·16 rows → ppad/8 eight-token groups).
        let ptx_moe = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_mxfp4_moe.ptx")?);
        dev.load_ptx(ptx_moe, "gpu_mxfp4_moe",
                     &["mxfp4_gemm_moe_grouped_native_b", "mxfp4_gemm_moe_slot_native_b",
                       "mxfp4_quant_pack_ng_b",
                       "mxfp4_gemm_moe_slot_fused_b0", "mxfp4_gemm_moe_slot_fused_b1",
                       "mxfp4_gemm_moe_grouped_fused_b0", "mxfp4_gemm_moe_grouped_fused_b1",
                       "kernel_build_id"])?;
        crate::gpu::GpuModel::assert_kernel_build_id(dev, "gpu_mxfp4_moe")?;
        // MoE pack scratch: the grouped arm's worst case (verify ppad = MAX_VERIFY·topk + ne·16)
        // AND the grouped-prefill window (PREFILL_CHUNK·topk + ne·16 rows, ppad/8 8-token groups),
        // AND the slot arm's down-side groups (batch·topk). The 122B prefill window dominates.
        let v_groups = (crate::gpu::MAX_VERIFY * cfg.num_experts_per_tok + cfg.num_experts * 16) / 8;
        let pf_groups = (crate::batch::PREFILL_CHUNK * cfg.num_experts_per_tok + cfg.num_experts * 16) / 8;
        let moe_groups_max = v_groups.max(pf_groups).max(crate::gpu::MAX_VERIFY * cfg.num_experts_per_tok);
        let moe_bp = dev.alloc_zeros::<u32>(moe_groups_max * nks_max * 64).unwrap();
        let moe_sfb = dev.alloc_zeros::<u32>(moe_groups_max * nks_max * 32).unwrap();
        // P4 B2 prefill scratch: Bq/SB for PREFILL_CHUNK rows at max K.
        let pf_rows = crate::batch::PREFILL_CHUNK;
        let pf_bq = dev.htod_sync_copy(&vec![0xFFu8; pf_rows * (MXFP4_MAX_K / 2)]).unwrap();
        // SFB layout (mxfp4_quant_pack_prefill_b): [(q*nks + ks)*32 + lane] u32 = 32 u32 per
        // (8-row group, 64-col chunk) = nks*128 B per group = nks*16 B PER ROW. The old
        // rows*(MAX_K/16) sizing (2048 B/row) only covered nks <= 128 (K <= 8192): the 27B
        // down_proj (K=17408, nks=272, 4352 B/row) scribbled ~19MB past the buffer on every
        // prefill quant — the 2026-08-26 flake (allocation-dependent corruption of neighbors,
        // memcheck: 264 invalid global writes at mxfp4_quant_pack_prefill_b+0x1560).
        let pf_sb = dev.htod_sync_copy(&vec![0xFFu8; pf_rows * (MXFP4_MAX_K / 64) * 16]).unwrap();
        // Poison the fast-path scratch (expert R3 decisive test): alloc_zeros does NOT zero
        // (AGENTS §2.2); fill with 0xFF so any GEMM read of never-quantized rows changes
        // the output deterministically instead of silently reading stale bytes.
        // poison via a tiny host copy (portable; no raw CUDA API import)

        let pf_quant = dev.get_func("gpu_mxfp4", "mxfp4_quant_prefill_b")
            .expect("mxfp4_quant_prefill_b not in gpu_mxfp4 module");
        let pf_gemm = dev.get_func("gpu_mxfp4", "mxfp4_gemm_prefill_b")
            .expect("mxfp4_gemm_prefill_b not in gpu_mxfp4 module");
        let pf_repack = dev.get_func("gpu_mxfp4", "mxfp4_repack_rm_b")
            .expect("mxfp4_repack_rm_b not in gpu_mxfp4 module");
        Ok(Some(Mxfp4State {
            w: omma_map,
            allow_bf16: RwLock::new(allow_bf16),
            mixer4: RwLock::new(mixer4),
            bp,
            sfb,
            gemv: dev.get_func("gpu_mxfp4", "mxfp4_gemv_native_b")
                .expect("mxfp4_gemv_native_b not in gpu_mxfp4 module"),
            quant: dev.get_func("gpu_mxfp4", "mxfp4_quant_pack_b")
                .expect("mxfp4_quant_pack_b not in gpu_mxfp4 module"),
            pf_quant2: dev.get_func("gpu_mxfp4", "mxfp4_quant_pack_prefill_b")
                .expect("mxfp4_quant_pack_prefill_b not in gpu_mxfp4 module"),
            pf_gemm2: dev.get_func("gpu_mxfp4", "mxfp4_gemm_prefill_v2_b")
                .expect("mxfp4_gemm_prefill_v2_b not in gpu_mxfp4 module"),
            dequant: dev.get_func("gpu_mxfp4", "mxfp4_dequant_tiled_b")
                .expect("mxfp4_dequant_tiled_b not in gpu_mxfp4 module"),
            embed: dev.get_func("gpu_mxfp4", "mxfp4_embed_gather_tiled_b")
                .expect("mxfp4_embed_gather_tiled_b not in gpu_mxfp4 module"),
            moe_gemv: dev.get_func("gpu_mxfp4_moe", "mxfp4_gemm_moe_grouped_native_b")
                .expect("mxfp4_gemm_moe_grouped_native_b not in gpu_mxfp4_moe module"),
            moe_slot: dev.get_func("gpu_mxfp4_moe", "mxfp4_gemm_moe_slot_native_b")
                .expect("mxfp4_gemm_moe_slot_native_b not in gpu_mxfp4_moe module"),
            quant_ng: dev.get_func("gpu_mxfp4_moe", "mxfp4_quant_pack_ng_b")
                .expect("mxfp4_quant_pack_ng_b not in gpu_mxfp4_moe module"),
            moe_bp,
            moe_sfb,
            gemv_fused1: dev.get_func("gpu_mxfp4", "mxfp4_gemv_native_fused_b1")
                .expect("mxfp4_gemv_native_fused_b1 not in gpu_mxfp4 module"),
            gemv_fused8: dev.get_func("gpu_mxfp4", "mxfp4_gemv_native_fused_b8")
                .expect("mxfp4_gemv_native_fused_b8 not in gpu_mxfp4 module"),
            gemv_fused16: dev.get_func("gpu_mxfp4", "mxfp4_gemv_native_fused_b16")
                .expect("mxfp4_gemv_native_fused_b16 not in gpu_mxfp4 module"),
            moe_slot_fused0: dev.get_func("gpu_mxfp4_moe", "mxfp4_gemm_moe_slot_fused_b0")
                .expect("mxfp4_gemm_moe_slot_fused_b0 not in gpu_mxfp4_moe module"),
            moe_slot_fused1: dev.get_func("gpu_mxfp4_moe", "mxfp4_gemm_moe_slot_fused_b1")
                .expect("mxfp4_gemm_moe_slot_fused_b1 not in gpu_mxfp4_moe module"),
            moe_grouped_fused0: dev.get_func("gpu_mxfp4_moe", "mxfp4_gemm_moe_grouped_fused_b0")
                .expect("mxfp4_gemm_moe_grouped_fused_b0 not in gpu_mxfp4_moe module"),
            moe_grouped_fused1: dev.get_func("gpu_mxfp4_moe", "mxfp4_gemm_moe_grouped_fused_b1")
                .expect("mxfp4_gemm_moe_grouped_fused_b1 not in gpu_mxfp4_moe module"),
            pf_quant,
            pf_gemm,
            pf_repack,
            pf_bq,
            pf_sb,
            pf_rm: RwLock::new(HashMap::new()),
            pf_rm_lock: std::sync::Mutex::new(()),
        }))
    }

    /// Cross-chain probe (`--probe-mxfp4-xchain`): mark EVERY repacked tensor allowlisted so the
    /// dispatch runs the bf16 chain end-to-end (both layouts are resident in non-economy mode).
    /// Returns the number of tensors switched.
    pub fn allow_all_bf16(&self) -> usize {
        let mut g = self.allow_bf16.write().unwrap();
        let before = g.len();
        g.extend(self.w.keys().copied());
        g.len() - before
    }

    /// Undo `allow_all_bf16` (back to the load-time allowlist, e.g. the MTP head).
    pub fn clear_allow_all_bf16(&self) {
        let mut g = self.allow_bf16.write().unwrap();
        let keys: HashSet<u64> = self.w.keys().copied().collect();
        g.retain(|p| !keys.contains(p));
    }
}

/// The production MMA repack constants (mirrors quant.rs — imported here to avoid a dep on
/// that module's internals; both must agree with kernels/gpu_batch.cu).
const MMA_M: usize = 16;
const MMA_K: usize = 16;

/// Lossless repack of production MMA-repacked NVFP4 (wt, st) into the OMMA A/SFA layouts.
///
/// Pure byte permutation: no arithmetic, no re-quantization, nibbles preserved exactly.
/// Requires M % 16 == 0, K % 64 == 0, K % 32 == 0 (production invariant; K%64 is OMMA's).
/// Returns (Aimg, SFAw) sized ntm*nks*512 and ntm*nks*64.
pub fn repack_nvfp4_omma(wt: &[u8], st: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(m % MMA_M == 0 && k % (MMA_K * 4) == 0 && k % 32 == 0,
            "OMMA repack needs M%16==0, K%64==0 (got {m}x{k})");
    let (ntm, nblk, nks) = (m / MMA_M, k / MMA_K, k / (MMA_K * 4));
    assert_eq!(wt.len(), ntm * nblk * 128, "wt size");
    assert_eq!(st.len(), ntm * nblk * MMA_M, "st size");
    let mut aimg = vec![0u8; ntm * nks * 512];
    let mut sfa = vec![0u8; ntm * nks * 64];

    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let tiles_per = ntm.div_ceil(nthreads).max(1);
    let a_chunk_len = tiles_per * nks * 512;
    let s_chunk_len = tiles_per * nks * 64;
    std::thread::scope(|sc| {
        for (t, (a_chunk, s_chunk)) in aimg.chunks_mut(a_chunk_len).zip(sfa.chunks_mut(s_chunk_len)).enumerate() {
            let mt0 = t * tiles_per;
            let mt1 = (mt0 + tiles_per).min(ntm);
            if mt0 >= mt1 { break; }
            sc.spawn(move || {
                for mt in 0..(mt1 - mt0) {
                    let mt = mt0 + mt;
                    let a_local = &mut a_chunk[(mt - mt0) * nks * 512..][..nks * 512];
                    let s_local = &mut s_chunk[(mt - mt0) * nks * 64..][..nks * 64];
                    for ks in 0..nks {
                        for lane in 0..32u32 {
                            let (g, tq) = (lane >> 2, lane & 3);
                            for r in 0..4u32 {
                                let kb = ks as u32 * 4 + 2 * (r >> 1) + (tq >> 1);
                                let jb = ((r & 1) | ((tq & 1) << 1)) as usize;
                                let src_base = ((mt * nblk + kb as usize) * 32 + g as usize * 4) * 4 + jb;
                                let dst_base = (ks * 128 + lane as usize * 4 + r as usize) * 4;
                                // u32 as 4 bytes, byte m' of dst = byte jb of production u32 (lane g*4+m').
                                for m in 0..4usize {
                                    a_local[dst_base + m] = wt[src_base + m * 4];
                                }
                            }
                            if tq <= 1 {
                                let dst = (ks * 16 + g as usize * 2 + tq as usize) * 4;
                                for v in 0..4usize {
                                    s_local[dst + v] = st[(mt * nblk + ks * 4 + v) * MMA_M + g as usize + 8 * tq as usize];
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    (aimg, sfa)
}

/// ECONOMY-mode variant: OMMA repack DIRECTLY from the raw NVFP4 tensor (qw [M, K/2] packed
/// nibbles, sc [M, K/16] E4M3 scales) — the composition of the two permutations (raw → standard
/// MMA repack → OMMA), so the standard repack is never materialized. The 122B's assembly workers
/// otherwise hold the fused raw AND the standard repack AND the OMMA repack at once (~2.25× per
/// job), which trips earlyoom on the unified pool (measured: SIGTERM/SIGKILL of the load AND
/// bystander processes, 2026-08-07). Byte-exact same result as
/// repack_nvfp4_omma(repack_nvfp4_mma(qw, sc)).
///
/// Composition (Aimg u32 = (tile mt, kstep ks, lane (g,t), reg r), byte m'):
///   production standard byte = (tile mt*nblk + kb, lane g*4+m', byte jb)
///   with kb = ks*4 + 2*(r>>1) + (t>>1), jb = (r&1) | ((t&1)<<1)
///   = raw byte qw[(mt*16 + g + 8*(jb&1))*(k/2) + (kb*16 + 2*m' + 8*(jb>>1))/2] (nibbles as-is).
///   SFA u32 (lane (g,t<=1)) byte v = raw scale sc[(mt*16 + g + 8*t)*(k/16) + ks*4 + v].
pub fn repack_nvfp4_omma_raw(qw: &[u8], sc: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(m % MMA_M == 0 && k % (MMA_K * 4) == 0 && k % 32 == 0,
            "OMMA repack needs M%16==0, K%64==0 (got {m}x{k})");
    let (ntm, nblk, nks) = (m / MMA_M, k / MMA_K, k / (MMA_K * 4));
    assert_eq!(qw.len(), m * k / 2, "qw size");
    assert_eq!(sc.len(), m * k / 16, "sc size");
    let mut aimg = vec![0u8; ntm * nks * 512];
    let mut sfa = vec![0u8; ntm * nks * 64];

    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let tiles_per = ntm.div_ceil(nthreads).max(1);
    let a_chunk_len = tiles_per * nks * 512;
    let s_chunk_len = tiles_per * nks * 64;
    std::thread::scope(|scope| {
        for (t, (a_chunk, s_chunk)) in aimg.chunks_mut(a_chunk_len).zip(sfa.chunks_mut(s_chunk_len)).enumerate() {
            let mt0 = t * tiles_per;
            let mt1 = (mt0 + tiles_per).min(ntm);
            if mt0 >= mt1 { break; }
            scope.spawn(move || {
                for mt in 0..(mt1 - mt0) {
                    let mt = mt0 + mt;
                    let a_local = &mut a_chunk[(mt - mt0) * nks * 512..][..nks * 512];
                    let s_local = &mut s_chunk[(mt - mt0) * nks * 64..][..nks * 64];
                    for ks in 0..nks {
                        for lane in 0..32u32 {
                            let (g, tq) = (lane >> 2, lane & 3);
                            for r in 0..4u32 {
                                let jb = (r & 1) | ((tq & 1) << 1);
                                let kb = ks as u32 * 4 + 2 * (r >> 1) + (tq >> 1);
                                let row = (mt * MMA_M + g as usize + 8 * (jb & 1) as usize) * (k / 2);
                                let dst_base = (ks * 128 + lane as usize * 4 + r as usize) * 4;
                                for mp in 0..4usize {
                                    let c0 = (kb as usize) * 16 + 2 * mp + 8 * (jb >> 1) as usize;
a_local[dst_base + mp] = qw[row + c0 / 2];
                                }
                            }
                            if tq <= 1 {
                                let dst = (ks * 16 + g as usize * 2 + tq as usize) * 4;
                                for v in 0..4usize {
                                    s_local[dst + v] = sc[(mt * MMA_M + g as usize + 8 * tq as usize) * (k / 16) + ks * 4 + v];
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    (aimg, sfa)
}

/// Host-side B (activation) pack for the Phase-0 bench: 8 token rows of bf16 [8][K] ->/// (Bp, SFB) in the OMMA layouts above. Mirrors the verified device kernel
/// mxfp4_quant_pack_b (kernels/mxfp4_bench.cu) — used only to feed the bench kernel;
/// the production path packs on device with hardware cvt.rn.satfinite.e2m1x2.
pub fn pack_b_host(x: &[bf16], k: usize) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(x.len(), 8 * k);
    assert!(k % 64 == 0);
    let nks = k / 64;
    let mut bp = vec![0u32; nks * 64];
    let mut sfb = vec![0u32; nks * 32];
    let mut scales = [[0u8; 4]; 8];
    let mut codes = [[0u8; 64]; 8];
    for ks in 0..nks {
        for n in 0..8usize {
            let row = &x[n * k + ks * 64..][..64];
            for v in 0..4 {
                let blk = &row[v * 16..][..16];
                let amax = blk.iter().map(|b| b.to_f32().abs()).fold(0.0f32, f32::max);
                scales[n][v] = e4m3_ceil(amax / 6.0);
                let inv = if scales[n][v] == 0 { 0.0 } else { 1.0 / ue4m3_f(scales[n][v]) };
                for (i, b) in blk.iter().enumerate() {
                    codes[n][v * 16 + i] = e2m1_rn(b.to_f32() * inv);
                }
            }
        }
        for lane in 0..32usize {
            let (g, t) = (lane >> 2, lane & 3);
            let mut b0 = 0u32;
            let mut b1 = 0u32;
            for j in 0..8usize {
                b0 |= (codes[g][8 * t + j] as u32) << (4 * j);
                b1 |= (codes[g][32 + 8 * t + j] as u32) << (4 * j);
            }
            bp[ks * 64 + lane * 2] = b0;
            bp[ks * 64 + lane * 2 + 1] = b1;
            if t == 0 {
                sfb[ks * 32 + lane] = u32::from_le_bytes(scales[g]);
            }
        }
    }
    (bp, sfb)
}

/// UE4M3 encode of |x| rounded UP (smallest representable magnitude >= |x|); sign bit 0
/// (the OMMA ignores the ue4m3 sign bit). Mirrors e4m3_ceil in kernels/mxfp4_bench.cu.
pub fn e4m3_ceil(x: f32) -> u8 {
    if !(x > 0.0) { return 0x00; }
    if x >= 448.0 { return 0x7F; }
    let (m, e) = libm_shim::frexp(x);
    let e4 = e + 6;
    let mant = (m - 0.5) * 16.0;
    let mut mant = mant.ceil() as i32;
    let mut e4 = e4;
    if mant >= 8 { mant = 0; e4 += 1; }
    if e4 < 0 {
        // subnormal: value = mant * 2^-9
        let sm = (x * 512.0).ceil() as i32;
        return (sm.min(7)) as u8;
    }
    if e4 > 14 { return 0x7F; }
    ((e4 << 3) | mant) as u8
}

/// UE4M3 decode (sign bit ignored; exp==0 -> mant * 2^-9). Mirrors ue4m3_f.
pub fn ue4m3_f(s: u8) -> f32 {
    let (e, m) = (s >> 3, s & 7);
    if e == 0 { m as f32 * 0.001953125 }
    else { (1.0 + m as f32 / 8.0) * libm_shim::exp2(e as f32 - 7.0) }
}

/// e2m1 nibble -> float (the 16 codes, sign bit 3). Mirrors gpu_batch.cu's e2m1_f / the device
/// table in gpu_mxfp4.cu.
pub fn e2m1_f(c: u8) -> f32 {
    const V: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    V[(c & 15) as usize]
}

/// e2m1 round-to-nearest-even of a float (saturates at 6.0). Mirrors cvt.rn.satfinite.e2m1x2
/// in the f32 -> code direction (values 0,0.5,1,1.5,2,3,4,6; bit 3 = sign).
/// EXACT arithmetic (partition_point + exact distance comparisons, ties to the even-mantissa
/// code) — the previous tolerance version used a 1e-7 epsilon that mis-rounded values within
/// the epsilon band of a code midpoint (the probe's fragment oracle diverged from the
/// hardware cvt; 2026-08-12). Same rule as dsv4_cpu::f32_to_e2m1_rne, proven equivalent to
/// the hardware on non-zero values.
pub fn e2m1_rn(x: f32) -> u8 {
    const T: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    if x.is_nan() {
        return 0x7; // e2m1fn NaN (S.111)
    }
    let sign = if x.is_sign_negative() { 0x8u8 } else { 0u8 };
    let a = x.abs().min(6.0);
    if a == 0.0 {
        return sign;
    }
    let hi = T.partition_point(|&v| v < a);
    let code = if hi == 0 {
        0usize
    } else if hi >= 8 {
        7usize
    } else {
        let d_hi = T[hi] - a;
        let d_lo = a - T[hi - 1];
        if d_hi < d_lo {
            hi
        } else if d_hi > d_lo {
            hi - 1
        } else if hi % 2 == 0 {
            hi
        } else {
            hi - 1
        }
    };
    sign | code as u8
}

mod libm_shim {
    /// (mantissa in [0.5, 1), exponent) such that x = m * 2^e. Bit-trick version, no libm.
    pub fn frexp(x: f32) -> (f32, i32) {
        if x == 0.0 { return (0.0, 0); }
        let bits = x.to_bits();
        let e = ((bits >> 23) & 0xFF) as i32 - 127;
        let m = f32::from_bits((bits & 0x807F_FFFF) | (126 << 23));   // exp 126 -> [0.5, 1)
        (m, e + 1)
    }
    /// 2^x for integer-exponent uses only (e4m3 decode). Exact.
    pub fn exp2(x: f32) -> f32 {
        f32::from_bits(((x as i32 + 127).clamp(0, 254) as u32) << 23)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The ECONOMY raw repack must be byte-identical to the two-step path
    // (raw -> repack_nvfp4_mma -> repack_nvfp4_omma): both are pure permutations of the same
    // nibbles, and any drift here silently corrupts the native chain (acceptance -> 0%).
    #[test]
    fn raw_repack_matches_two_step() {
        for (m, k) in [(16usize, 256usize), (128, 1024), (16, 64), (32, 320)] {
            let mut qw = vec![0u8; m * k / 2];
            let mut sc = vec![0u8; m * k / 16];
            let mut s = 0x9E3779B9u32;
            for b in qw.iter_mut() { s = s.wrapping_mul(1664525).wrapping_add(1013904223); *b = (s >> 24) as u8; }
            for b in sc.iter_mut() { s = s.wrapping_mul(1664525).wrapping_add(1013904223); *b = (s >> 24) as u8; }
            let (wt, st) = crate::quant::repack_nvfp4_mma(&qw, &sc, m, k);
            let (a2, s2) = repack_nvfp4_omma(&wt, &st, m, k);
            let (a1, s1) = repack_nvfp4_omma_raw(&qw, &sc, m, k);
            for (i, (x, y)) in a1.iter().zip(&a2).enumerate() {
                if x != y { panic!("Aimg first mismatch at {m}x{k} idx {i} (byte {x:02x} vs {y:02x})"); }
            }
            for (i, (x, y)) in s1.iter().zip(&s2).enumerate() {
                if x != y { panic!("SFA first mismatch at {m}x{k} idx {i} (byte {x:02x} vs {y:02x})"); }
            }
        }
    }
}

#[cfg(test)]
mod tp_shard_tests {
    use super::*;

    fn rng_bytes(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 24) as u8 }).collect()
    }

    /// The shard-twins' invariant: slicing the OMMA repack by the TP shard math must equal
    /// repacking each rank's shard. Both directions, every world, multi-segment col.
    #[test]
    fn omma_shard_matches_shard_omma() {
        for world in [1usize, 2, 4, 8, 16] {
            // m=512 (=2·256) and k=1024 so col and row shards divide at every rung up to 16.
            let (m, k) = (512usize, 1024usize);
            let qw = rng_bytes(m * k / 2, 7);
            let sc = rng_bytes(m * k / 16, 9);
            let (wt, st) = crate::quant::repack_nvfp4_mma(&qw, &sc, m, k);
            let (aimg, sfa) = repack_nvfp4_omma(&wt, &st, m, k);
            let gsv: Vec<f32> = (0..m / 16).map(|i| i as f32 * 0.5).collect();
            let nks = k / 64;

            // ---- COLUMN-parallel (whole-M single segment + a 2-segment split) ----
            let seg_w = 256;
            for segs in [vec![(0usize, m, true)], vec![(0, seg_w, true), (seg_w, seg_w, true)]] {
                for rank in 0..world {
                    // shard of the OMMA (the twin's math, host-side)
                    let mut a_new = Vec::new();
                    let mut s_new = Vec::new();
                    for &(off, cnt, split) in &segs {
                        let take = if split { cnt / world } else { cnt };
                        let tile_lo = (off + if split { rank * take } else { 0 }) / 16;
                        let ntl = take / 16;
                        a_new.extend_from_slice(&aimg[tile_lo * nks * 512..(tile_lo + ntl) * nks * 512]);
                        s_new.extend_from_slice(&sfa[tile_lo * nks * 64..(tile_lo + ntl) * nks * 64]);
                    }
                    // OMMA of the shard (via the loader's host shard + repack)
                    let (qw2, sc2, gsv2, ml) =
                        crate::gpu::host_shard_nvfp4_col_segs(&wt, &st, &gsv, k, &segs, rank, world);
                    let (a_ref, s_ref) = repack_nvfp4_omma(&qw2, &sc2, ml, k);
                    assert_eq!(a_new, a_ref, "col Aimg mismatch segs={segs:?} world={world} rank={rank}");
                    assert_eq!(s_new, s_ref, "col SFA mismatch segs={segs:?} world={world} rank={rank}");
                    let mut g_expected: Vec<f32> = Vec::new();
                    for &(off, cnt, split) in &segs {
                        let take = if split { cnt / world } else { cnt };
                        let tile_lo = (off + if split { rank * take } else { 0 }) / 16;
                        g_expected.extend_from_slice(&gsv[tile_lo..tile_lo + take / 16]);
                    }
                    assert_eq!(gsv2, g_expected, "col gs mismatch segs={segs:?} world={world} rank={rank}");
                }
            }

            // ---- ROW-parallel (K split) ----
            for rank in 0..world {
                let k_local = k / world;
                let nks_local = k_local / 64;
                let mut a_new = Vec::new();
                let mut s_new = Vec::new();
                for mt in 0..(m / 16) {
                    let a_src = (mt * nks + rank * nks_local) * 512;
                    a_new.extend_from_slice(&aimg[a_src..a_src + nks_local * 512]);
                    let s_src = (mt * nks + rank * nks_local) * 64;
                    s_new.extend_from_slice(&sfa[s_src..s_src + nks_local * 64]);
                }
                let (qw2, sc2, kl) = crate::gpu::host_shard_nvfp4_row(&wt, &st, m, k, rank, world);
                let (a_ref, s_ref) = repack_nvfp4_omma(&qw2, &sc2, m, kl);
                assert_eq!(a_new, a_ref, "row Aimg mismatch world={world} rank={rank}");
                assert_eq!(s_new, s_ref, "row SFA mismatch world={world} rank={rank}");
            }
        }
    }
}
