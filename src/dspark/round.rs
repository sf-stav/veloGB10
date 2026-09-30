//! `dspark::round` — the serving GPU round for the Qwen3.8-27B-DSpark block drafter (WI1).
//!
//! Port of the proven `dflash2::round::Df2Round` template onto the DSpark anatomy: 5 plain
//! transformer layers (NO conv — the DSpark artifact has no conv tensors), block 7
//! (anchor + 6 MASK → 7 drafted tokens at verify width 8, the exact DF2 chain shape), GQA
//! 40q/8kv @ hd 128, MLP 10240, YaRN rope (factor from the artifact config: 32 base / 64
//! for the -512K variants), and FULL-CONTEXT draft attention (`sliding_window: null`) via
//! `dspark_attn_full_ring_b` (gpu_dspark.ptx) — the band kernel's three-pass schedule with
//! the scores row in a global scratch. The draft chain adds the Markov head with REFERENCE
//! semantics (vLLM `_sample_sequential` + SpecForge `apply_block_logits`): prev is seeded
//! with the ANCHOR token and the bias `W2 @ W1[prev]` is added at EVERY position 0..6,
//! teacher-forced — d0 = argmax(logits row 0 + W2@W1[anchor]);
//! d_k = argmax(logits0_k + W2 @ W1[d_{k-1}]) (oracle DECISIONS L/E, revised), fanned over
//! blocks by `dspark_chain_{a,b}`.
//!
//! Artifact conventions (verified against the published reference code + checkpoint values,
//! 2026-09-06 session): norm gains are FULL (`w·x`, file values ≈1.0 — the trunk's ≈0-mean
//! `(1+w)` convention does NOT apply to this artifact); rope pairing is NeoX half-split
//! (`rotate_half`); YaRN mscale (0.1·ln(factor)+1) folds into the cos/sin tables; the YaRN
//! correction range is low=find(beta_fast), high=find(beta_slow). Legacy arms:
//! --dspark-norm-g1 / --dspark-d0-plain / --dspark-rope-il / --dspark-nomscale.
//!
//! The borrowed head/embed contract is DF2's verbatim (`GpuModel::df2_borrow`, including the
//! F8 two-stage head). Greedy losslessness is verify-protected: the trunk's argmax stream is
//! untouched by whatever the drafter proposes.
//!
//! V1 (this file): BF16 artifact, eager launches, replicated round (TP=2 loads it on both
//! ranks; the drafts are bit-identical because the taps are bit-identical from the
//! all-reduced hiddens). Round/inject CUDA graphs and the temp>0 sampled lane follow the
//! DF2 pattern and land after the eager path is oracle-proven.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::bf16;

use crate::dflash2::capture::Df2TapSink;
use crate::dflash2::gpu::fork_blocking_stream;
use crate::dflash2::round::{BorrowedW, Nvfp4Ptrs};
use crate::dspark::{
    load, BLOCK, HIDDEN, HEAD_DIM, INTER, MARKOV_RANK, N_LAYERS, NUM_HEADS, NUM_KV_HEADS, RMS_EPS,
    TAP_CONCAT_DIM, VOCAB,
};

/// DSpark block: anchor + 6 MASK = 7 positions → 7 drafted tokens (verify width 8 = 1 + 7,
/// identical to the DF2 chain lane; MAX_VERIFY=16 untouched). Re-exported for the serving
/// lane's verify-width math (= crate::dspark::BLOCK).
pub use crate::dspark::BLOCK as DSPARK_BLOCK;
/// F8/B2: tile rows staged through smem in dspark_attn_full_ring_c (see gpu_dspark.cu).
pub const DSPARK_ATTN_TILE: usize = 128;

fn d<T>(s: &CudaSlice<T>) -> u64 {
    *s.device_ptr() as u64
}
fn grid(n: usize) -> (u32, u32, u32) {
    (((n + 255) / 256) as u32, 1, 1)
}
fn fbits(x: f32) -> u64 {
    x.to_bits() as u64
}

macro_rules! klaunch {
    ($s:expr, $name:expr, $g:expr, $b:expr, $smem:expr, ($($a:expr),+ $(,)?)) => {
        unsafe {
            let (g0, g1, g2) = $g;
            let (b0, b1, b2) = $b;
            let name: &str = $name;
            $s.bk.get(name).cloned().unwrap_or_else(|| panic!("dspark kernel {name}")).launch_on_stream(
                &$s.stream,
                LaunchConfig { grid_dim: (g0, g1, g2), block_dim: (b0, b1, b2), shared_mem_bytes: $smem },
                ($($a),+)
            ).unwrap_or_else(|e| panic!("dspark launch {name}: {e:?}"));
        }
    };
}

/// One draft-backbone layer's device weights (BF16 artifact; norms f32 as w−1 — the same
/// (1+w)·x convention the rmsnorm kernels and the oracle both use).
struct GpuLayer {
    q_proj: CudaSlice<bf16>,   // [5120, 5120]
    k_proj: CudaSlice<bf16>,   // [1024, 5120]
    v_proj: CudaSlice<bf16>,   // [1024, 5120]
    o_proj: CudaSlice<bf16>,   // [5120, 5120]
    gate_proj: CudaSlice<bf16>, // [10240, 5120]
    up_proj: CudaSlice<bf16>,   // [10240, 5120]
    down_proj: CudaSlice<bf16>, // [5120, 10240]
    q_norm: CudaSlice<f32>,     // [128] (w−1)
    k_norm: CudaSlice<f32>,     // [128] (w−1)
    input_ln: CudaSlice<f32>,   // [5120] (w−1)
    post_ln: CudaSlice<f32>,    // [5120] (w−1)
}

struct GpuGlobal {
    fc: CudaSlice<bf16>,         // [5120, 25600]
    hidden_norm: CudaSlice<f32>, // [5120] (w−1)
    norm: CudaSlice<f32>,        // [5120] (w−1)
}

/// The rope config the artifact ships (YaRN). Parsed from config.json at load so the base
/// (factor 32) and -512K (factor 64) artifacts both bind exactly.
#[derive(Clone, Debug)]
pub struct RopeParams {
    pub theta: f32,
    pub factor: f32,
    pub orig_ctx: usize,
    pub beta_fast: u32,
    pub beta_slow: u32,
}

pub struct DsparkRound {
    dev: Arc<CudaDevice>,
    stream: cudarc::driver::CudaStream,
    bk: HashMap<String, CudaFunction>,
    #[allow(dead_code)]
    cfg: crate::dspark::oracle::DsparkConfig,
    head: Option<BorrowedW>,
    embed: Option<BorrowedW>,

    layers: Vec<GpuLayer>,
    glob: GpuGlobal,
    /// Markov head (host copy kept for the confidence weights' readback-free host math).
    w1: CudaSlice<bf16>, // [VOCAB, MARKOV_RANK]
    w2: CudaSlice<bf16>, // [VOCAB, MARKOV_RANK]
    /// Transposed Markov W2 [MARKOV_RANK, VOCAB] — dspark_chain_c's coalesced layout.
    w2t: CudaSlice<bf16>,
    conf_w: Vec<f32>,    // [HIDDEN + MARKOV_RANK] host
    conf_b: f32,

    cos_table: CudaSlice<f32>, // [max_pos, HEAD_DIM/2]
    sin_table: CudaSlice<f32>,

    /// Full-context KV ring. Capacity C_ring = max_c (rows never wrap at valid nprev); rows
    /// [C_ring, C_ring+BLOCK) are the fixed block region. stride = C_ring + 8 (< 2^21 —
    /// the packed-arg field width; the 512K context class fits).
    k_ring: Vec<CudaSlice<bf16>>,
    v_ring: Vec<CudaSlice<bf16>>,
    c_ring: usize,
    ring_stride: usize,
    /// Global scores scratch for dspark_attn_full_ring_b/_c: [BLOCK*NUM_HEADS, score_stride].
    scores_g: CudaSlice<f32>,
    score_stride: usize,
    /// F8/B2 flash-decode partials for dspark_attn_full_ring_d:
    /// [NUM_KV_HEADS][DSPARK_QSEG][BLOCK*QG][2+HEAD_DIM] f32 (1.2 MB — vs the 73 MB
    /// scores_g round-trip the _b path pays at max ring).
    attn_part: CudaSlice<f32>,

    nprev: usize,
    /// A6/DF2_CARRY identity: which slot's committed prefix the ring rows currently reflect
    /// (`None` = unproven). A length is NOT enough — the round is one shared object while the
    /// per-slot bookkeeping is per slot, so `--max-batch > 1` can carry a foreign ring and lose
    /// tau (verify still rejects, so no losslessness gate sees it).
    /// See PLAN/DSPARK_RING_IDENTITY_SPEC.md; the DFlash2 twin carries the same field.
    ring_slot: Option<usize>,
    max_c: usize,

    // block scratch ([*, 8] col-major; the M=8 GEMM width, 7 live rows)
    toks_blk: CudaSlice<i32>,   // [8]
    consts_dev: CudaSlice<i32>, // [4] {anchor, nprev, c_ring, shift}
    consts_pin: Vec<i32>,       // pinned-source mirror of consts_dev (async htod)
    staging: CudaSlice<bf16>,   // [TAP_CONCAT_DIM, 8] tap staging (probe uploads / sink sync)
    h: CudaSlice<bf16>,
    normed: CudaSlice<bf16>,
    normed2: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    attn_out: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    mlp_out: CudaSlice<bf16>,
    h_final: CudaSlice<bf16>,
    th_raw: CudaSlice<bf16>, // inject: fc out [5120, 8]
    th: CudaSlice<bf16>,     // inject: hidden_norm out [5120, 8]
    kc: CudaSlice<bf16>,     // inject/prime: [1024, 8]
    vc: CudaSlice<bf16>,
    cos8: CudaSlice<f32>, sin8: CudaSlice<f32>,       // block rope [128, 8]
    pos_blk: CudaSlice<i32>, wrow_blk: CudaSlice<i32>, slot_blk: CudaSlice<i32>,
    cos_c: CudaSlice<f32>, sin_c: CudaSlice<f32>,     // inject rope [128, 8]
    pos_c: CudaSlice<i32>, wrow_c: CudaSlice<i32>, slot_c: CudaSlice<i32>,

    logits: CudaSlice<bf16>,     // [VOCAB, 7] col-major (borrowed head at rows=7)
    coarse_idx: CudaSlice<u32>,  // [7, 256] the TwoStage shortlist (ids per token column)
    prev_dev: CudaSlice<u32>,    // [1] device chain state
    walk_tokens: CudaSlice<u32>, // [7]
    latents: CudaSlice<f32>,     // [6, MARKOV_RANK]
    cand_s: CudaSlice<f32>,      // [CHAIN_BLOCKS]
    cand_o: CudaSlice<u32>,      // [CHAIN_BLOCKS]

    /// Attached trunk tap sink (the decode/verify capture staging). The serving lane calls
    /// `sync_staging_from_sink` before each inject — the DF2 S5F3 deep-copy lesson.
    sink: Option<Arc<crate::dflash2::capture::Df2TapSink>>,
}

unsafe impl Send for DsparkRound {}

/// The chain fan-out (vocab scan blocks). 512 blocks × 256 threads ≈ 2 vocab rows/thread.
const CHAIN_BLOCKS: usize = 512;

/// Rope table rows = max_c + BLOCK + 1 (mirrors DF2's max_pos choice).
fn rope_rows(max_c: usize) -> usize {
    max_c + BLOCK + 1
}

impl DsparkRound {
    /// Bind the ring/capacity geometry: the caller passes the served context budget. The
    /// packed-arg geometry fields are 21 bits (512K context class fits; gpu_dspark.cu).
    pub fn load(dir: &str, head: Option<BorrowedW>, embed: Option<BorrowedW>, max_c: usize) -> Result<Self> {
        Self::load_pinned(dir, head, embed, max_c, None)
    }

    pub fn load_pinned(dir: &str, head: Option<BorrowedW>, embed: Option<BorrowedW>, max_c: usize,
                       sha_pin: Option<&str>) -> Result<Self> {
        let pin: Option<&str> = match sha_pin {
            Some("off") => None,
            Some(hex) => Some(hex),
            None => Some(crate::dspark::REAL_SHA256),
        };
        let art = load::load(dir, pin)?;
        let w = &art.weights;
        let cfg = crate::dspark::oracle::DsparkConfig::default();
        let rope = Self::rope_from_dir(dir)?;

        anyhow::ensure!(head.is_some(), "DsparkRound needs the trunk's lm_head (df2_borrow)");
        anyhow::ensure!(embed.is_some(), "DsparkRound needs the trunk's embed (df2_borrow)");
        // F8 acceptance A2: the packed-arg geometry fields are 21 bits wide (gpu_dspark.cu) —
        // the round carries the full 512K context class (524288 + block + slack < 2^21).
        anyhow::ensure!(max_c >= 1024 && max_c + BLOCK + 8 < (1 << 21),
            "dspark max_c {max_c} out of the packed-arg range (1024..={}); \
             larger contexts need the ntot_dev arm", (1 << 21) - BLOCK - 9);

        let dev = CudaDevice::new(0).context("CudaDevice")?;
        let stream = fork_blocking_stream(&dev);

        // ---- modules (the DF2 round's name lists, minus its selector/tree extras, plus ours)
        let bptx = cudarc::nvrtc::Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx")?);
        // gemm_mma_fp4_b: the TwoStage coarse head (the REAL trunk head is NVFP4 TwoStage —
        // the probe's synthetic head is Bf16 and never exercises it, so this only shows at
        // serving; DF2's list carries the same name).
        let bfnames = ["rmsnorm_b", "rmsnorm_perhead_b", "rope_b", "gather_rope_b",
            "write_kv_b", "add_residual_b", "silu_mul_b", "kernel_build_id",
            "embed_gather_b", "gemm_binv_b", "gemm_mma_fp4_b"].to_vec();
        dev.load_ptx(bptx, "gpu_batch", &bfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_batch")?;
        let kptx = cudarc::nvrtc::Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_kernels.ptx")?);
        let kfnames = ["gemm_dsp_b_m8_r4", "gemm_tiled_b", "bf16tof32", "f32tobf16",
            "kernel_build_id", "df2_top256_b", "df2_head_rerank_b"];
        dev.load_ptx(kptx, "gpu_kernels", &kfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_kernels")?;
        let sptx = cudarc::nvrtc::Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_dspark.ptx")?);
        let sfnames = ["kernel_build_id", "dspark_attn_full_ring_b", "dspark_attn_full_ring_c",
                      "dspark_attn_full_ring_d", "dspark_attn_full_ring_e", "dspark_attn_merge_d", "dspark_chain_a",
            "dspark_chain_b", "dspark_chain_c", "dspark_row0_argmax_b", "dspark_rope_b", "dspark_scalars_b"];
        dev.load_ptx(sptx, "gpu_dspark", &sfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_dspark")?;
        let mut bk = HashMap::new();
        for (n, module) in bfnames.iter().map(|n| (n, "gpu_batch"))
            .chain(kfnames.iter().map(|n| (n, "gpu_kernels")))
            .chain(sfnames.iter().map(|n| (n, "gpu_dspark"))) {
            bk.insert(n.to_string(), dev.get_func(module, n).with_context(|| format!("kernel {n} not in ptx"))?);
        }

        let up_b = |data: &[f32]| -> CudaSlice<bf16> {
            let b: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
            dev.htod_sync_copy(&b).expect("upload bf16")
        };
        let up_norm = |data: &[f32]| -> CudaSlice<f32> {
            // DSpark stores FULL GAINS (every drafter norm weight has mean ≈ +1.0; SpecForge
            // trains with HF Qwen3RMSNorm = plain w·x; the TRUNK's ≈0-mean weights are the
            // (1+w) family — the §7 norm-convention trap cuts BOTH ways). The rmsnorm kernels
            // compute (1 + uploaded)·x, so upload w−1 to get w_file·x — DF2's convention after
            // all. --dspark-norm-g1=1 restores the old (1+w_file)·x upload for A/B.
            let legacy = crate::opts::var(crate::opt!("dspark-norm-g1")).map(|v| v == "1" || v == "on").unwrap_or(false);
            let m: Vec<f32> = if legacy {
                data.to_vec()
            } else {
                data.iter().map(|&x| x - 1.0f32).collect()
            };
            dev.htod_sync_copy(&m).expect("upload norm")
        };

        let mut layers = Vec::with_capacity(N_LAYERS);
        for l in w.layers.iter() {
            layers.push(GpuLayer {
                q_proj: up_b(&l.q_proj),
                k_proj: up_b(&l.k_proj),
                v_proj: up_b(&l.v_proj),
                o_proj: up_b(&l.o_proj),
                gate_proj: up_b(&l.gate_proj),
                up_proj: up_b(&l.up_proj),
                down_proj: up_b(&l.down_proj),
                q_norm: up_norm(&l.q_norm),
                k_norm: up_norm(&l.k_norm),
                input_ln: up_norm(&l.input_ln),
                post_ln: up_norm(&l.post_ln),
            });
        }
        let glob = GpuGlobal {
            fc: up_b(&w.fc),
            hidden_norm: up_norm(&w.hidden_norm),
            norm: up_norm(&w.norm),
        };
        let w1 = up_b(&w.w1);
        let w2 = up_b(&w.w2);
        // F8: transposed Markov W2 [rank, VOCAB] for dspark_chain_c — the coalesced chain
        // (chain_a's broadcast 2 B loads ran at 2.8% of roofline; 19 ms × 7/step = the
        // round's 147 ms cost center). One-time 127 MB host transpose at load (threads,
        // row-strided writes; ~0.2 s).
        let mut w2t = vec![0f32; w.w2.len()];
        {
            let (v, r) = (crate::dspark::VOCAB, MARKOV_RANK);
            let src = &w.w2;
            std::thread::scope(|s| {
                let nthreads = 8usize;
                let seg = r / nthreads;   // r=256 divisible by 8
                let mut rest = &mut w2t[..];
                for t in 0..nthreads {
                    let (dseg, tail) = rest.split_at_mut(seg * v);
                    rest = tail;
                    s.spawn(move || {
                        for (ri, dstrow) in dseg.chunks_mut(v).enumerate() {
                            let i = t * seg + ri;
                            for o in 0..v { dstrow[o] = src[o * r + i]; }
                        }
                    });
                }
            });
        }
        let w2t = up_b(&w2t);
        // confidence.weight [1, 5376] = [hidden half (5120) | latent half (256)] — kept on the
        // host for the readback-free host confidence math (6 tiny dots per round).
        let conf_w = w.confidence_w.clone();
        let conf_b = w.confidence_b;

        // ---- rope tables (YaRN, artifact params — the oracle's exact freq/table math)
        let max_pos = rope_rows(max_c);
        let freqs = crate::dspark::oracle::yaarn_freqs(
            HEAD_DIM, rope.theta, rope.factor, rope.orig_ctx, rope.beta_fast, rope.beta_slow);
        // YaRN mscale folds into the cos/sin cache (vLLM yarn_scaling_rope.py:
        // cos = freqs.cos() * mscale, sin likewise; q and k both scale, scores scale by
        // mscale² = 1.813 at factor 32). --dspark-nomscale=1 restores the old unscaled
        // tables for A/B.
        let mscale = if crate::opts::var(crate::opt!("dspark-nomscale")).map(|v| v == "1" || v == "on").unwrap_or(false) {
            1.0f32
        } else if rope.factor > 1.0 {
            0.1f32 * rope.factor.ln() + 1.0f32
        } else {
            1.0f32
        };
        // [max_pos, rdim=HEAD_DIM] with the duplicated-freqs second half — the DEVICE's
        // gather_rope_b/rope_b convention (DF2's rope_tables); a [max_pos, half] table makes
        // the gather cross into the next position's row (a growing-with-position rope error).
        let half = HEAD_DIM / 2;
        let rdim = HEAD_DIM;
        let mut cos_t = vec![0.0f32; max_pos * rdim];
        let mut sin_t = vec![0.0f32; max_pos * rdim];
        for p in 0..max_pos {
            let pf = p as f32;
            for (i, f) in freqs.iter().enumerate() {
                let ang = pf * f;
                cos_t[p * rdim + i] = ang.cos() * mscale;
                sin_t[p * rdim + i] = ang.sin() * mscale;
                if i >= half { continue; }
                cos_t[p * rdim + half + i] = cos_t[p * rdim + i];
                sin_t[p * rdim + half + i] = sin_t[p * rdim + i];
            }
        }
        let cos_table = dev.htod_sync_copy(&cos_t)?;
        let sin_table = dev.htod_sync_copy(&sin_t)?;

        // ---- full-context ring + scores scratch
        let c_ring = max_c;
        let ring_stride = c_ring + 8;
        let mut k_ring = Vec::with_capacity(N_LAYERS);
        let mut v_ring = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            k_ring.push(dev.alloc_zeros::<bf16>(NUM_KV_HEADS * ring_stride * HEAD_DIM)?);
            v_ring.push(dev.alloc_zeros::<bf16>(NUM_KV_HEADS * ring_stride * HEAD_DIM)?);
        }
        let score_stride = max_c + BLOCK;
        let scores_g = dev.alloc_zeros::<f32>(BLOCK * NUM_HEADS * score_stride)?;

        // ---- scratch ([*, 8] col-major; the M=8 GEMM width, 7 live rows)
        let a8 = |n: usize| -> Result<CudaSlice<bf16>> { Ok(dev.alloc_zeros::<bf16>(n * 8)?) };
        let toks_blk = dev.alloc_zeros::<i32>(8)?;
        let consts_dev = dev.alloc_zeros::<i32>(4)?;
        let consts_pin = vec![0i32; 4];
        let staging = dev.alloc_zeros::<bf16>(TAP_CONCAT_DIM * 8)?;
        let h = a8(HIDDEN)?;
        let normed = a8(HIDDEN)?;
        let normed2 = a8(HIDDEN)?;
        let q = a8(NUM_HEADS * HEAD_DIM)?;
        let k = a8(NUM_KV_HEADS * HEAD_DIM)?;
        let v = a8(NUM_KV_HEADS * HEAD_DIM)?;
        let attn = a8(NUM_HEADS * HEAD_DIM)?;
        let attn_out = a8(HIDDEN)?;
        let gate = a8(INTER)?;
        let up = a8(INTER)?;
        let mlp_out = a8(HIDDEN)?;
        let h_final = a8(HIDDEN)?;
        let th_raw = a8(HIDDEN)?;
        let th = a8(HIDDEN)?;
        let kc = a8(NUM_KV_HEADS * HEAD_DIM)?;
        let vc = a8(NUM_KV_HEADS * HEAD_DIM)?;
        let f8 = || dev.alloc_zeros::<f32>(HEAD_DIM * 8);
        let cos8 = f8()?;
        let sin8 = f8()?;
        let cos_c = f8()?;
        let sin_c = f8()?;
        let i8 = || dev.alloc_zeros::<i32>(8);
        let pos_blk = i8()?;
        let wrow_blk = i8()?;
        let slot_blk = i8()?;
        let pos_c = i8()?;
        let wrow_c = i8()?;
        let slot_c = i8()?;
        let logits = dev.alloc_zeros::<bf16>(VOCAB * BLOCK)?;
        let coarse_idx = dev.alloc_zeros::<u32>(BLOCK * 256)?;
        let prev_dev = dev.alloc_zeros::<u32>(1)?;
        let walk_tokens = dev.alloc_zeros::<u32>(BLOCK)?;
        let latents = dev.alloc_zeros::<f32>((BLOCK - 1) * MARKOV_RANK)?;
        let cand_s = dev.alloc_zeros::<f32>(CHAIN_BLOCKS)?;
        let cand_o = dev.alloc_zeros::<u32>(CHAIN_BLOCKS)?;
        // F8/B2 flash-decode partials (see attn_part). Sized for SEG=16 (_e) — _d (SEG=8)
        // uses the low half. [NUM_KV_HEADS][16][BLOCK*QG][2+HEAD_DIM] f32 (2.3 MB — vs the
        // 73 MB scores_g of _b).
        const DSPARK_QSEG: usize = 16;
        let nq = BLOCK * (NUM_HEADS / NUM_KV_HEADS);
        let attn_part = dev.alloc_zeros::<f32>(
            NUM_KV_HEADS * DSPARK_QSEG * nq * (2 + HEAD_DIM))?;

        eprintln!("[dspark] round RESIDENT — block {BLOCK}, {} layers, rope yarn factor {} \
                   (theta {}, orig {}), ring {max_c} rows, scores {} MB, weights {} MB",
                  N_LAYERS, rope.factor, rope.theta, rope.orig_ctx,
                  BLOCK * NUM_HEADS * score_stride * 4 / 1_000_000,
                  art.n_params * 2 / 1_000_000);

        Ok(Self {
            dev, stream, bk, cfg, head, embed,
            layers, glob, w1, w2, w2t, conf_w, conf_b,
            cos_table, sin_table,
            k_ring, v_ring, c_ring, ring_stride, scores_g, score_stride, attn_part,
            nprev: 0, ring_slot: None, max_c,
            toks_blk, consts_dev, consts_pin, staging, h, normed, normed2, q, k, v, attn, attn_out, gate, up,
            mlp_out, h_final, th_raw, th, kc, vc,
            cos8, sin8, pos_blk, wrow_blk, slot_blk, cos_c, sin_c, pos_c, wrow_c, slot_c,
            logits, coarse_idx, prev_dev, walk_tokens, latents, cand_s, cand_o,
            sink: None,
        })
    }

    /// Parse the artifact's rope params from config.json (block_size/mask/id live in the
    /// constants; only the rope knobs differ between the base and -512K artifacts).
    fn rope_from_dir(dir: &str) -> Result<RopeParams> {
        let path = std::path::Path::new(dir).join("config.json");
        let txt = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&txt)?;
        let rp = v.get("rope_parameters").context("config.json: no rope_parameters")?;
        Ok(RopeParams {
            theta: rp.get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(1e7) as f32,
            factor: rp.get("factor").and_then(|x| x.as_f64()).unwrap_or(32.0) as f32,
            orig_ctx: rp.get("original_max_position_embeddings")
                .and_then(|x| x.as_u64()).unwrap_or(8192) as usize,
            beta_fast: rp.get("beta_fast").and_then(|x| x.as_f64()).unwrap_or(32.0) as u32,
            beta_slow: rp.get("beta_slow").and_then(|x| x.as_f64()).unwrap_or(1.0) as u32,
        })
    }

    pub fn nprev(&self) -> usize { self.nprev }
    pub fn max_c(&self) -> usize { self.max_c }

    /// DF2 S5F3 lesson: keep the Arc and copy the sink's LIVE staging before each inject.
    pub fn attach_sink(&mut self, sink: &Arc<crate::dflash2::capture::Df2TapSink>) {
        self.sink = Some(sink.clone());
    }

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
                // ALL 8 staging columns: the verify capture writes 8 (anchor + 7 drafts) and a
                // full accept injects nacc+1 = 8 — copying only BLOCK=7 left col 7 stale.
                WidthInBytes: TAP_CONCAT_DIM * 2, Height: 8,
            };
            unsafe {
                let r = sys::cuMemcpy2DAsync_v2(&cp, self.stream.stream);
                if r != sys::CUresult::CUDA_SUCCESS {
                    anyhow::bail!("dspark sync staging D2D failed: {r:?}");
                }
            }
            self.dev.synchronize()?;
        }
        Ok(())
    }

    // ---- kernel helpers (the DF2 launch shapes verbatim) --------------------

    fn embed_gather(&self) {
        match self.embed.expect("embed ptrs") {
            BorrowedW::TwoStage { .. } => {
                panic!("TwoStage borrow on the embed (contract violation)");
            }
            BorrowedW::Nvfp4(p) => {
                klaunch!(self, "embed_gather_fp4_tiled_b", grid(HIDDEN * BLOCK), (256, 1, 1), 0,
                    (d(&self.h), p.qweight, p.scales, p.gs, d(&self.toks_blk), HIDDEN as i32, BLOCK as i32));
            }
            BorrowedW::Bf16 { ptr } => {
                klaunch!(self, "embed_gather_b", grid(HIDDEN * BLOCK), (256, 1, 1), 0,
                    (d(&self.h), ptr, d(&self.toks_blk), HIDDEN as i32, BLOCK as i32));
            }
        }
    }

    fn gemm_dsp(&self, out: &CudaSlice<bf16>, w: &CudaSlice<bf16>, x_ptr: u64, outn: usize, inn: usize) {
        let g = ((outn + 3) / 4) as u32; // R=4
        klaunch!(self, "gemm_dsp_b_m8_r4", (g, 1, 1), (256, 1, 1), 0,
            (d(out), d(w), x_ptr, outn as i32, inn as i32));
    }

    /// Large-M GEMM (prime path): out [outn, m] = w [outn, k] x x [k, m] col-major bf16.
    fn gemm_tiled(&self, out: &CudaSlice<bf16>, w: &CudaSlice<bf16>, x_ptr: u64, outn: usize, k: usize, m: usize) {
        let mx = ((m + 127) / 128) as u32;
        let nx = ((outn + 127) / 128) as u32;
        klaunch!(self, "gemm_tiled_b", (mx, nx, 1), (16, 16, 1), 0,
            (d(out), d(w), x_ptr, outn as i32, k as i32, m as i32));
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
        // REFERENCE = HALF-SPLIT (NeoX `rotate_half`): SpecForge dflash.py imports rotate_half
        // from HF qwen3 and vLLM's get_rope defaults is_neox_style=True — both references
        // pair (j, j+half). DF2's rope_b implements exactly that. The interleaved kernel
        // (dspark_rope_b, pairs (2j, 2j+1)) is kept as a diagnostic arm:
        // --dspark-rope-il=1 selects it; GB10_DSPARK_ROPE_HS=1 is accepted as an explicit
        // no-op (half-split is now the default).
        let il = crate::opts::var(crate::opt!("dspark-rope-il")).map(|v| v == "1" || v == "on").unwrap_or(false);
        if il {
            klaunch!(self, "dspark_rope_b", grid(b * heads * (HEAD_DIM / 2)), (256, 1, 1), 0,
                (d(x), d(cos), d(sin), heads as i32, HEAD_DIM as i32, HEAD_DIM as i32, b as i32));
        } else {
            klaunch!(self, "rope_b", grid(b * heads * (HEAD_DIM / 2)), (256, 1, 1), 0,
                (d(x), d(cos), d(sin), heads as i32, HEAD_DIM as i32, HEAD_DIM as i32, b as i32));
        }
    }

    fn gather_rope(&self, out_cos: &CudaSlice<f32>, out_sin: &CudaSlice<f32>, pos: u64, b: usize) {
        klaunch!(self, "gather_rope_b", grid(b * HEAD_DIM), (256, 1, 1), 0,
            (d(out_cos), d(out_sin), d(&self.cos_table), d(&self.sin_table), pos, HEAD_DIM as i32, b as i32));
    }

    /// The borrowed head at rows=7 on h_final cols 0..6 — DSpark differs from DF2 here: row 0
    /// (the anchor row) PRODUCES a draft (d0 = its argmax), so the head reads col 0 (no
    /// +HIDDEN*2 offset). Dispatch mirrors DF2's, TwoStage included.
    fn head_logits(&self) {
        match self.head.expect("head ptrs") {
            BorrowedW::TwoStage { q, bf16_ptr } => {
                // Stage 1: coarse NVFP4 pass over the full vocab (0.68 GB) on h_final cols 0..6.
                let persistent = (crate::gpu::GB10_SMS * 6).min(VOCAB / 16) as u32;
                klaunch!(self, "gemm_mma_fp4_b", (persistent, 1, 1), (256, 1, 1), 0,
                    (d(&self.logits), q.qweight, q.scales, q.gs,
                     d(&self.h_final), VOCAB as i32, HIDDEN as i32, BLOCK as i32, 0u64, 0i32));
                // Stage 1b: per-column top-256 shortlist (deterministic; recall@256 is the only
                // quality requirement — stage 2 supplies the exact ranking).
                klaunch!(self, "df2_top256_b", (BLOCK as u32, 1, 1), (256, 1, 1), 0,
                    (d(&self.coarse_idx), d(&self.logits), VOCAB as i32, BLOCK as i32));
                // Stage 2: exact bf16 re-rank of the shortlist against the ORIGINAL hiddens.
                klaunch!(self, "df2_head_rerank_b", ((BLOCK * 256) as u32, 1, 1), (256, 1, 1), 0,
                    (d(&self.logits), d(&self.coarse_idx), bf16_ptr,
                     d(&self.h_final), VOCAB as i32, HIDDEN as i32, BLOCK as i32));
            }
            BorrowedW::Nvfp4(p) => {
                let persistent = (crate::gpu::GB10_SMS * 6).min(VOCAB / 16) as u32;
                klaunch!(self, "gemm_mma_fp4_b", (persistent, 1, 1), (256, 1, 1), 0,
                    (d(&self.logits), p.qweight, p.scales, p.gs,
                     d(&self.h_final), VOCAB as i32, HIDDEN as i32, BLOCK as i32, 0u64, 0i32));
            }
            BorrowedW::Bf16 { ptr } => {
                let smem = (BLOCK * 256 * 4) as u32;
                klaunch!(self, "gemm_binv_b", (VOCAB as u32, 1, 1), (256, 1, 1), smem,
                    (d(&self.logits), ptr, d(&self.h_final),
                     VOCAB as i32, HIDDEN as i32, BLOCK as i32));
            }
        }
    }

    // ---- context maintenance (prime + inject) --------------------------------

    /// Engine prompt-prime: `n` tap columns at one large M via gemm_tiled (the DF2 body).
    pub fn prime_window(&mut self, taps: &CudaSlice<bf16>, n: usize, pos_start: usize) -> Result<()> {
        assert!(n >= 1, "prime window {n} out of range");
        assert!(pos_start + n <= self.max_c, "prime {}..{} > max_c {}",
                pos_start, pos_start + n - 1, self.max_c);
        let pos: Vec<i32> = (0..n).map(|j| (pos_start + j) as i32).collect();
        let wrow: Vec<i32> = (0..n).map(|j| (pos_start + j) as i32).collect(); // no wrap: c_ring = max_c
        let slots: Vec<i32> = vec![0i32; n];
        let pos_dev = self.dev.htod_sync_copy(&pos).context("prime pos")?;
        let wrow_dev = self.dev.htod_sync_copy(&wrow).context("prime wrow")?;
        let slot_dev = self.dev.htod_sync_copy(&slots).context("prime slots")?;
        let cos_c = self.dev.alloc_zeros::<f32>(n * HEAD_DIM)?;
        let sin_c = self.dev.alloc_zeros::<f32>(n * HEAD_DIM)?;
        self.gather_rope(&cos_c, &sin_c, d(&pos_dev), n);

        // n <= 8 runs DIRECTLY on the round's own th_raw/th scratch ([5120, 8] — the probe's
        // dump_th reads them); wider windows allocate locals (the gemm is width-agnostic).
        let _local_raw: CudaSlice<bf16>;
        let _local_th: CudaSlice<bf16>;
        let (th_raw_buf, th_buf): (&CudaSlice<bf16>, &CudaSlice<bf16>) = if n <= 8 {
            (&self.th_raw, &self.th)
        } else {
            _local_raw = self.dev.alloc_zeros::<bf16>(HIDDEN * n)?;
            _local_th = self.dev.alloc_zeros::<bf16>(HIDDEN * n)?;
            (&_local_raw, &_local_th)
        };
        self.gemm_tiled(th_raw_buf, &self.glob.fc, d(taps), HIDDEN, TAP_CONCAT_DIM, n);
        self.rmsnorm(th_buf, th_raw_buf, &self.glob.hidden_norm, HIDDEN, n);

        let kc = self.dev.alloc_zeros::<bf16>(NUM_KV_HEADS * HEAD_DIM * n)?;
        let vc = self.dev.alloc_zeros::<bf16>(NUM_KV_HEADS * HEAD_DIM * n)?;
        for li in 0..N_LAYERS {
            let l = &self.layers[li];
            self.gemm_tiled(&kc, &l.k_proj, d(th_buf), NUM_KV_HEADS * HEAD_DIM, HIDDEN, n);
            self.rmsnorm_perhead(&kc, &l.k_norm, NUM_KV_HEADS, n);
            self.rope(&kc, &cos_c, &sin_c, NUM_KV_HEADS, n);
            self.gemm_tiled(&vc, &l.v_proj, d(th_buf), NUM_KV_HEADS * HEAD_DIM, HIDDEN, n);
            klaunch!(self, "write_kv_b", grid(n * NUM_KV_HEADS * HEAD_DIM), (256, 1, 1), 0,
                (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&kc), d(&vc),
                 d(&wrow_dev), self.ring_stride as i32, NUM_KV_HEADS as i32, HEAD_DIM as i32,
                 n as i32, d(&slot_dev)));
        }
        self.dev.synchronize()?;
        self.nprev = pos_start + n;
        Ok(())
    }

    /// Inject `m ≤ BLOCK` committed tap columns from `self.staging` cols [0, m) — the
    /// per-step DF2 inject (M=8 gemm_dsp path; cols ≥ m garbage-but-unread). Absolute
    /// positions [nprev, nprev+m); ring rows [nprev, nprev+m) (no wrap at valid nprev).
    pub fn inject_dev(&mut self, m: usize) -> Result<()> {
        // Serving injects nacc+1 which reaches 8 on a full accept (7 drafts + the bonus's
        // committed tap). The M=8 gemm width, the [*, 8] staging and the ring's 8 block rows
        // all support 8 — the DSpark block being 7 only means the DRAFT round uses 7 of the
        // 8 columns (DF2's BLOCK is 8 for the same role).
        assert!(m <= 8, "inject chunk {m} > 8");
        assert!(self.nprev + m <= self.max_c, "nprev {} + {m} > max_c {}", self.nprev, self.max_c);
        let n0 = self.nprev;
        let pos: Vec<i32> = (0..m).map(|j| (n0 + j) as i32).collect();
        let wrow: Vec<i32> = (0..m).map(|j| (n0 + j) as i32).collect();
        let mut pos8 = pos; pos8.resize(8, 0);
        let mut wrow8 = wrow; wrow8.resize(8, 0);
        self.dev.htod_sync_copy_into(&pos8, &mut self.pos_c)?;
        self.dev.htod_sync_copy_into(&wrow8, &mut self.wrow_c)?;

        // fc + hidden_norm at M=m (gemm_dsp; the fixed 8-col width, cols ≥ m unread)
        self.gemm_dsp(&self.th_raw, &self.glob.fc, d(&self.staging), HIDDEN, TAP_CONCAT_DIM);
        self.rmsnorm(&self.th, &self.th_raw, &self.glob.hidden_norm, HIDDEN, m);
        self.gather_rope(&self.cos_c, &self.sin_c, d(&self.pos_c), m);
        for li in 0..N_LAYERS {
            let l = &self.layers[li];
            self.gemm_dsp(&self.kc, &l.k_proj, d(&self.th), NUM_KV_HEADS * HEAD_DIM, HIDDEN);
            self.rmsnorm_perhead(&self.kc, &l.k_norm, NUM_KV_HEADS, m);
            self.rope(&self.kc, &self.cos_c, &self.sin_c, NUM_KV_HEADS, m);
            self.gemm_dsp(&self.vc, &l.v_proj, d(&self.th), NUM_KV_HEADS * HEAD_DIM, HIDDEN);
            klaunch!(self, "write_kv_b", grid(m * NUM_KV_HEADS * HEAD_DIM), (256, 1, 1), 0,
                (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&self.kc), d(&self.vc),
                 d(&self.wrow_c), self.ring_stride as i32, NUM_KV_HEADS as i32, HEAD_DIM as i32,
                 m as i32, d(&self.slot_c)));
        }
        self.dev.synchronize()?;
        self.nprev = n0 + m;
        Ok(())
    }

    // ---- the draft round -----------------------------------------------------

    /// Refresh the block position arrays for the current nprev (call before each round).
    /// F8/P3: the per-step scalars (anchor, nprev, c_ring, block-pos shift) go to the
    /// device as ONE async copy into consts_dev; `dspark_scalars_b` (launched first in
    /// draft_round_kernels_partial) expands toks/pos/wrow/prev on device. This replaced
    /// FOUR blocking htod_sync_copy_into per round that serialized against the round's
    /// own GPU queue (~20 ms/round at 17K ctx). Semantics unchanged (incl. BLOCK_POS).
    fn refresh_block_pos(&mut self) -> Result<()> {
        Ok(())
    }

    fn upload_scalars(&mut self, anchor: u32) -> Result<()> {
        let s: i32 = crate::opts::var(crate::opt!("dspark-block-pos")).ok()
            .and_then(|v| v.parse().ok()).unwrap_or(0);
        self.consts_pin[0] = anchor as i32;
        self.consts_pin[1] = self.nprev as i32;
        self.consts_pin[2] = self.c_ring as i32;
        self.consts_pin[3] = s;
        unsafe {
            use cudarc::driver::result::memcpy_htod_async;
            memcpy_htod_async(*self.consts_dev.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                              &self.consts_pin, self.stream.stream).expect("htod consts");
        }
        Ok(())
    }

    /// The pure-launch body (eager). `graph_mode` packs nothing yet (v1 eager) — the flag
    /// mirrors DF2's for the graph follow-up.
    fn draft_round_kernels(&mut self, anchor: u32) -> Result<()> {
        // --dspark-round-bisect=<n>: run only the first n backbone layers (diagnostics —
        // marginal cost per layer + head/chain residual from the round-trace totals).
        let nl: usize = crate::opts::var(crate::opt!("dspark-round-bisect")).ok()
            .and_then(|v| v.parse().ok()).unwrap_or(N_LAYERS);
        self.draft_round_kernels_partial(anchor, nl.min(N_LAYERS))
    }

    /// Layer-bisect instrument: run embed + the first `nlayers` backbone layers (no head/chain).
    fn draft_round_kernels_partial(&mut self, anchor: u32, nlayers: usize) -> Result<()> {
        let ntot = self.nprev + BLOCK;
        self.upload_scalars(anchor)?;
        klaunch!(self, "dspark_scalars_b", (1, 1, 1), (32, 1, 1), 0,
            (d(&self.toks_blk), d(&self.pos_blk), d(&self.wrow_blk), d(&self.prev_dev),
             d(&self.consts_dev), crate::dspark::MASK_TOKEN_ID as i32, BLOCK as i32));
        // block rope rows for the current nprev (one gather; all 5 layers share it)
        self.gather_rope(&self.cos8, &self.sin8, d(&self.pos_blk), BLOCK);
        self.embed_gather();
        // ---- 5-layer backbone (full-context ring attention) ----
        for li in 0..nlayers {
            self.layer_forward(li, ntot);
        }
        if nlayers < N_LAYERS {
            return Ok(());
        }
        // final norm
        self.rmsnorm(&self.h_final, &self.h, &self.glob.norm, HIDDEN, BLOCK);
        // ---- borrowed head at rows=7 (cols 0..6) ----
        self.head_logits();
        // ---- the markov chain, REFERENCE semantics: the bias W2@W1[prev] applies at EVERY
        // position 0..6 with prev seeded by the ANCHOR (vLLM _sample_sequential seeds
        // prev = input_ids[query_off 0]; dspark.py apply_block_logits is teacher-forced).
        // --dspark-d0-plain=1 restores the old plain-argmax d0 for A/B.
        let legacy_d0 = crate::opts::var(crate::opt!("dspark-d0-plain")).map(|v| v == "1" || v == "on").unwrap_or(false);
        if legacy_d0 {
            klaunch!(self, "dspark_row0_argmax_b", (1, 1, 1), (256, 1, 1), 256,
                (d(&self.walk_tokens), d(&self.prev_dev), d(&self.logits),
                 VOCAB as i32, BLOCK as i32));
            for k in 1..BLOCK {
                klaunch!(self, "dspark_chain_a", (CHAIN_BLOCKS as u32, 1, 1), (256, 1, 1),
                         ((MARKOV_RANK + 64) * 4) as u32,
                         (d(&self.cand_s), d(&self.cand_o), d(&self.logits), d(&self.w1), d(&self.w2),
                          d(&self.prev_dev), VOCAB as i32, MARKOV_RANK as i32, BLOCK as i32, k as i32));
                klaunch!(self, "dspark_chain_b", (1, 1, 1), (256, 1, 1), 256,
                    (d(&self.walk_tokens), d(&self.prev_dev), d(&self.latents),
                     d(&self.cand_s), d(&self.cand_o), CHAIN_BLOCKS as i32,
                     d(&self.w1), MARKOV_RANK as i32, BLOCK as i32, k as i32));
            }
        } else {
            // prev_dev[0] = anchor — already expanded by dspark_scalars_b above.
            // F8: chain_c (coalesced, exact ascending-i order) is the default; the legacy
            // broadcast kernel stays behind --dspark-chain-a=1 for A/B (19 ms vs ~0.6 ms
            // per call at rank 256 / vocab 248320).
            let legacy_chain = crate::opts::var(crate::opt!("dspark-chain-a"))
                .map(|v| v == "1" || v == "on").unwrap_or(false);
            for k in 0..BLOCK {
                if legacy_chain {
                    klaunch!(self, "dspark_chain_a", (CHAIN_BLOCKS as u32, 1, 1), (256, 1, 1),
                             ((MARKOV_RANK + 64) * 4) as u32,
                             (d(&self.cand_s), d(&self.cand_o), d(&self.logits), d(&self.w1), d(&self.w2),
                              d(&self.prev_dev), VOCAB as i32, MARKOV_RANK as i32, BLOCK as i32, k as i32));
                } else {
                    klaunch!(self, "dspark_chain_c", (CHAIN_BLOCKS as u32, 1, 1), (256, 1, 1),
                             ((MARKOV_RANK + 64) * 4) as u32,
                             (d(&self.cand_s), d(&self.cand_o), d(&self.logits), d(&self.w1), d(&self.w2t),
                              d(&self.prev_dev), VOCAB as i32, MARKOV_RANK as i32, BLOCK as i32, k as i32));
                }
                klaunch!(self, "dspark_chain_b", (1, 1, 1), (256, 1, 1), 256,
                    (d(&self.walk_tokens), d(&self.prev_dev), d(&self.latents),
                     d(&self.cand_s), d(&self.cand_o), CHAIN_BLOCKS as i32,
                     d(&self.w1), MARKOV_RANK as i32, BLOCK as i32, k as i32));
            }
        }
        Ok(())
    }

    fn layer_forward(&self, li: usize, ntot: usize) {
        let l = &self.layers[li];
        // attention sublayer (NO conv — DSpark is a plain transformer drafter)
        self.rmsnorm(&self.normed, &self.h, &l.input_ln, HIDDEN, BLOCK);
        self.gemm_dsp(&self.q, &l.q_proj, d(&self.normed), NUM_HEADS * HEAD_DIM, HIDDEN);
        self.rmsnorm_perhead(&self.q, &l.q_norm, NUM_HEADS, BLOCK);
        self.gemm_dsp(&self.k, &l.k_proj, d(&self.normed), NUM_KV_HEADS * HEAD_DIM, HIDDEN);
        self.rmsnorm_perhead(&self.k, &l.k_norm, NUM_KV_HEADS, BLOCK);
        // (block rope was gathered once in draft_round_kernels — cos8/sin8)
        self.rope(&self.q, &self.cos8, &self.sin8, NUM_HEADS, BLOCK);
        self.rope(&self.k, &self.cos8, &self.sin8, NUM_KV_HEADS, BLOCK);
        self.gemm_dsp(&self.v, &l.v_proj, d(&self.normed), NUM_KV_HEADS * HEAD_DIM, HIDDEN);
        // block rows → ring rows [c_ring, c_ring + BLOCK)
        klaunch!(self, "write_kv_b", grid(BLOCK * NUM_KV_HEADS * HEAD_DIM), (256, 1, 1), 0,
            (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&self.k), d(&self.v),
             d(&self.wrow_blk), self.ring_stride as i32, NUM_KV_HEADS as i32, HEAD_DIM as i32,
             BLOCK as i32, d(&self.slot_blk)));
        // FULL-context attention over the ring (the new gpu_dspark kernel)
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let nh_packed = ((NUM_HEADS << 20) | (HEAD_DIM << 10) | NUM_KV_HEADS) as i32;
        // 21-bit geometry fields (see gpu_dspark.cu): ntot [42..64), C_ring [21..42),
        // stride [0..21) — carries the 512K context class.
        debug_assert!(ntot < (1 << 22) && self.c_ring < (1 << 21) && self.ring_stride < (1 << 21));
        let packed = (((ntot as u64) << 42) | ((self.c_ring as u64) << 21) | self.ring_stride as u64);
        // F8/B4: dspark_attn_full_ring_e (flash-decode on mma.sync tensor cores) is the
        // default — 0.56 ms/call at ntot=30K vs _d's 4.78 (KV roofline 0.52), oracle
        // 43/43. --dspark-attn={b,c,d} selects the legacy/scalar kernels explicitly
        // (diagnostic A/B escapes — logged, never silent).
        let attn_sel = crate::opts::var(crate::opt!("dspark-attn")).unwrap_or_default();
        match attn_sel.as_str() {
            "b" => {
                eprintln!("[dspark] --dspark-attn=b — legacy untiled _b (diagnostic)");
                let smem = ((HEAD_DIM + 32) * 4) as u32;
                klaunch!(self, "dspark_attn_full_ring_b", ((BLOCK * NUM_HEADS) as u32, 1, 1),
                         (HEAD_DIM as u32, 1, 1), smem,
                    (d(&self.attn), d(&self.q), d(&self.k_ring[li]), d(&self.v_ring[li]),
                     packed, std::ptr::null_mut::<i32>() as u64, nh_packed,
                     d(&self.scores_g), self.score_stride as i32, fbits(scale)));
            }
            "c" => {
                eprintln!("[dspark] --dspark-attn=c — smem-tiled _c (diagnostic)");
                let pitch = HEAD_DIM + 16;
                let smem = (((HEAD_DIM + 32) * 4) + DSPARK_ATTN_TILE * pitch * 2) as u32;
                klaunch!(self, "dspark_attn_full_ring_c", ((BLOCK * NUM_HEADS) as u32, 1, 1),
                         (HEAD_DIM as u32, 1, 1), smem,
                    (d(&self.attn), d(&self.q), d(&self.k_ring[li]), d(&self.v_ring[li]),
                     packed, std::ptr::null_mut::<i32>() as u64, nh_packed,
                     d(&self.scores_g), self.score_stride as i32, fbits(scale)));
            }
            "d" => {
                // F8/B2 scalar flash-decode (diagnostic escape — the pre-mma default).
                const DSPARK_QSEG: usize = 8;
                let nq = BLOCK * (NUM_HEADS / NUM_KV_HEADS);
                let smem_d = (nq * HEAD_DIM * 4 + 2 * 32 * HEAD_DIM * 2) as u32;
                klaunch!(self, "dspark_attn_full_ring_d",
                         (NUM_KV_HEADS as u32, DSPARK_QSEG as u32, 1), (256, 1, 1), smem_d,
                    (d(&self.attn), d(&self.q), d(&self.k_ring[li]), d(&self.v_ring[li]),
                     packed, std::ptr::null_mut::<i32>() as u64, nh_packed,
                     d(&self.attn_part), nq as i32, fbits(scale)));
                klaunch!(self, "dspark_attn_merge_d", (NUM_KV_HEADS as u32, 1, 1),
                    (256, 1, 1), 0,
                    (d(&self.attn), d(&self.attn_part), nh_packed,
                     (nq | (DSPARK_QSEG << 16)) as i32));
            }
            _ => {
                // F8/B4 default: tensor-core flash-decode. Partial contract shared with _d
                // (SEG rides gridDim.y + the merge's nq_total bits 16..23).
                const SEG_E: usize = 16;
                let nq = BLOCK * (NUM_HEADS / NUM_KV_HEADS);
                klaunch!(self, "dspark_attn_full_ring_e",
                         (NUM_KV_HEADS as u32, SEG_E as u32, 1), (96, 1, 1), 0,
                    (d(&self.attn), d(&self.q), d(&self.k_ring[li]), d(&self.v_ring[li]),
                     packed, std::ptr::null_mut::<i32>() as u64, nh_packed,
                     d(&self.attn_part), nq as i32, fbits(scale)));
                klaunch!(self, "dspark_attn_merge_d", (NUM_KV_HEADS as u32, 1, 1),
                         (256, 1, 1), 0,
                    (d(&self.attn), d(&self.attn_part), nh_packed,
                     (nq | (SEG_E << 16)) as i32))
            }
        }
        self.gemm_dsp(&self.attn_out, &l.o_proj, d(&self.attn), HIDDEN, NUM_HEADS * HEAD_DIM);
        klaunch!(self, "add_residual_b", grid(HIDDEN * BLOCK), (256, 1, 1), 0,
            (d(&self.h), d(&self.h), d(&self.attn_out), (HIDDEN * BLOCK) as i32));
        // mlp sublayer
        self.rmsnorm(&self.normed2, &self.h, &l.post_ln, HIDDEN, BLOCK);
        self.gemm_dsp(&self.gate, &l.gate_proj, d(&self.normed2), INTER, HIDDEN);
        self.gemm_dsp(&self.up, &l.up_proj, d(&self.normed2), INTER, HIDDEN);
        klaunch!(self, "silu_mul_b", grid(INTER * BLOCK), (256, 1, 1), 0,
            (d(&self.gate), d(&self.gate), d(&self.up), (INTER * BLOCK) as i32));
        self.gemm_dsp(&self.mlp_out, &l.down_proj, d(&self.gate), HIDDEN, INTER);
        klaunch!(self, "add_residual_b", grid(HIDDEN * BLOCK), (256, 1, 1), 0,
            (d(&self.h), d(&self.h), d(&self.mlp_out), (HIDDEN * BLOCK) as i32));
    }

    /// Probe instrument: run layer `li`'s ctx-K pipeline (k_proj gemm → perhead norm → rope →
    /// ring write) on the caller's th rows [m, HIDDEN] (token-major) at positions
    /// [pos0, pos0+m), returning each stage's device output (f32) for host comparison.
    pub fn probe_k_pipeline(&mut self, li: usize, th: &[f32], m: usize, pos0: usize)
        -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>
    {
        assert!(m <= 8);
        let thb: Vec<bf16> = th.iter().map(|&x| bf16::from_f32(x)).collect();
        let th_dev = self.dev.htod_sync_copy(&thb)?;
        let kc = self.dev.alloc_zeros::<bf16>(NUM_KV_HEADS * HEAD_DIM * 8)?;
        // stage 1: the gemm (M=8 width; cols >= m unread)
        self.gemm_tiled(&kc, &self.layers[li].k_proj, d(&th_dev), NUM_KV_HEADS * HEAD_DIM, HIDDEN, m);
        self.dev.synchronize()?;
        let g: Vec<bf16> = self.dev.dtoh_sync_copy(&kc)?;
        let gemm_out: Vec<f32> = g.iter().map(|x| x.to_f32()).collect();
        // stage 2: perhead norm
        self.rmsnorm_perhead(&kc, &self.layers[li].k_norm, NUM_KV_HEADS, m);
        self.dev.synchronize()?;
        let nrm: Vec<bf16> = self.dev.dtoh_sync_copy(&kc)?;
        let norm_out: Vec<f32> = nrm.iter().map(|x| x.to_f32()).collect();
        // stage 3: rope at [pos0, pos0+m)
        let cos8 = self.dev.alloc_zeros::<f32>(8 * HEAD_DIM)?;
        let sin8 = self.dev.alloc_zeros::<f32>(8 * HEAD_DIM)?;
        let pos: Vec<i32> = (0..m).map(|j| (pos0 + j) as i32).collect();
        let pos_dev = self.dev.htod_sync_copy(&pos)?;
        self.gather_rope(&cos8, &sin8, d(&pos_dev), m);
        self.rope(&kc, &cos8, &sin8, NUM_KV_HEADS, m);
        self.dev.synchronize()?;
        let rp: Vec<bf16> = self.dev.dtoh_sync_copy(&kc)?;
        let rope_out: Vec<f32> = rp.iter().map(|x| x.to_f32()).collect();
        // NOTE: no stage-4 ring write here — the live ring is sacred (a probe write zeroed the
        // V rows and poisoned every later round in the process). The ring path is gated by the
        // ctx-KV gates (prime/inject) instead; the rope stage IS the ring's k input.
        self.dev.synchronize()?;
        let ring_k = Vec::new();
        Ok((gemm_out, norm_out, rope_out, ring_k))
    }

    /// Probe support: readback the round's staging ([8, TAP_CONCAT_DIM] bf16, token-major).
    pub fn dump_staging(&self) -> Result<Vec<half::bf16>> {
        Ok(self.dev.dtoh_sync_copy(&self.staging)?)
    }

    /// Probe support: readback ring rows [0, rows) of layer `li`'s K/V caches.
    /// Returns (k, v), each element (kvh, r, d) at kvh*(stride*hd) + r*hd + d (f32).
    pub fn dump_kv_rows(&self, li: usize, rows: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let stride = self.ring_stride;
        let k: Vec<bf16> = self.dev.dtoh_sync_copy(&self.k_ring[li])?.to_vec();
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.v_ring[li])?.to_vec();
        let mut ko = Vec::with_capacity(NUM_KV_HEADS * rows * HEAD_DIM);
        let mut vo = Vec::with_capacity(NUM_KV_HEADS * rows * HEAD_DIM);
        for h in 0..NUM_KV_HEADS {
            for r in 0..rows {
                for d in 0..HEAD_DIM {
                    ko.push(k[h * stride * HEAD_DIM + r * HEAD_DIM + d].to_f32());
                    vo.push(v[h * stride * HEAD_DIM + r * HEAD_DIM + d].to_f32());
                }
            }
        }
        Ok((ko, vo))
    }

    /// Probe support: readback the running block hidden h rows [0, rows) (f32, col-major).
    pub fn dump_h(&self, rows: usize) -> Result<Vec<f32>> {
        let h: Vec<bf16> = self.dev.dtoh_sync_copy(&self.h)?.to_vec();
        let mut out = Vec::with_capacity(rows * HIDDEN);
        for r in 0..rows {
            for i in 0..HIDDEN {
                out.push(h[r * HIDDEN + i].to_f32());
            }
        }
        Ok(out)
    }

    /// Probe instrument: refresh + embed + the first `nlayers` layers, then sync (no head).
    pub fn draft_round_partial_dev(&mut self, anchor: u32, nlayers: usize) -> Result<()> {
        self.refresh_block_pos()?;
        self.draft_round_kernels_partial(anchor, nlayers)?;
        self.dev.synchronize()?;
        Ok(())
    }

    /// Run the eager draft round for `anchor`; returns the 7 drafted tokens (readback).
    pub fn draft_round_dev(&mut self, anchor: u32) -> Result<Vec<u32>> {
        // --dspark-round-trace=1: per-phase host timing (first 40 rounds) — the round
        // runs on its own CUDA context, invisible to a default nsys capture, so the split
        // must come from the engine itself.
        let trace = crate::opts::var(crate::opt!("dspark-round-trace")).is_ok();
        static ROUND_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = if trace { ROUND_N.fetch_add(1, std::sync::atomic::Ordering::SeqCst) } else { 0 };
        let t0 = std::time::Instant::now();
        self.refresh_block_pos()?;
        let t1 = std::time::Instant::now();
        self.draft_round_kernels(anchor)?;
        let t2 = std::time::Instant::now();
        self.dev.synchronize()?;
        let t3 = std::time::Instant::now();
        let toks: Vec<u32> = self.dev.dtoh_sync_copy(&self.walk_tokens)?.to_vec();
        let t4 = std::time::Instant::now();
        if trace && n < 40 {
            eprintln!("[round-trace #{:>2}] refresh {:.2} kernels {:.2} sync {:.2} dtoh {:.2} | total {:.2} ms",
                n, (t1-t0).as_secs_f32()*1e3, (t2-t1).as_secs_f32()*1e3,
                (t3-t2).as_secs_f32()*1e3, (t4-t3).as_secs_f32()*1e3, (t4-t0).as_secs_f32()*1e3);
        }
        Ok(toks[..BLOCK].to_vec())
    }

    /// Probe support: readback the tap-projection output th rows [0, rows) (f32). Valid right
    /// after prime_window/inject_dev (the next inject overwrites it at its own M).
    pub fn dump_th(&self, rows: usize) -> Result<Vec<f32>> {
        // the gemm x/out convention is TOKEN-major (column m of an [inn, 8] tensor starts at
        // m*inn), so row r, dim i lives at r*HIDDEN + i.
        let th: Vec<bf16> = self.dev.dtoh_sync_copy(&self.th)?.to_vec();
        let mut out = Vec::with_capacity(rows * HIDDEN);
        for r in 0..rows {
            for i in 0..HIDDEN {
                out.push(th[r * HIDDEN + i].to_f32());
            }
        }
        Ok(out)
    }

    /// Probe support: readback the full logits block [VOCAB, BLOCK] (bf16 → f32). The
    /// round-vs-oracle gate runs the host markov chain on THESE logits so the chain check is
    /// exact at zero tolerance risk (the borrowed-head path is gated separately by rel-L2).
    pub fn dump_logits(&self) -> Result<Vec<f32>> {
        // token-major [7, VOCAB] (the binv C layout); returned as-is, f32.
        let lg: Vec<bf16> = self.dev.dtoh_sync_copy(&self.logits)?.to_vec();
        Ok(lg.iter().map(|x| x.to_f32()).collect())
    }

    /// Probe support: upload committed tap rows directly into the round's staging (the probe
    /// path; serving uses `sync_staging_from_sink`). `rows` is ROW-major [8, TAP_CONCAT_DIM]
    /// — the gemm x convention is token-major (column c at c*inn), so each of the 8 staging
    /// columns is one contiguous 25600-tap row; columns >= the injected m are unread.
    pub fn upload_chunk(&mut self, rows: &[f32]) -> Result<()> {
        assert!(rows.len() >= 8 * TAP_CONCAT_DIM, "upload_chunk needs 8 contiguous tap rows");
        let b: Vec<bf16> = rows[..8 * TAP_CONCAT_DIM].iter().map(|&x| bf16::from_f32(x)).collect();
        self.dev.htod_sync_copy_into(&b, &mut self.staging)?;
        Ok(())
    }

    /// Read back the 6 markov latent rows (W1[d_k], k = 1..6) — the confidence head's inputs.
    pub fn read_latents(&self) -> Result<Vec<f32>> {
        Ok(self.dev.dtoh_sync_copy(&self.latents)?.to_vec())
    }

    /// Read back ALL 7 h_final rows, f32. The confidence head consumes row k for draft k
    /// (reference `compute_confidence`: hidden row i + markov embed of prev_i) — the oracle's
    /// confidence indexes `h[k*hidden .. (k+1)*hidden]` for k = 1..6, i.e. rows 1..6.
    pub fn read_h_final(&self) -> Result<Vec<f32>> {
        // token-major (see dump_th): row r, dim i at r*HIDDEN + i.
        let h: Vec<bf16> = self.dev.dtoh_sync_copy(&self.h_final)?.to_vec();
        let mut out = Vec::with_capacity(BLOCK * HIDDEN);
        for r in 0..BLOCK {
            for i in 0..HIDDEN {
                out.push(h[r * HIDDEN + i].to_f32());
            }
        }
        Ok(out)
    }

    /// The confidence head on the HOST (a 5376-dot per position — no kernel worth it):
    /// p_k = sigmoid(conf_w · [h_k ∥ latent_k] + conf_b); survival = cumprod; k_verify via
    /// the oracle's truncation (adaptive verify — DEFAULT OFF in v1; this is diagnostics).
    pub fn confidence(&self, h: &[f32], latents: &[f32]) -> ([f32; 6], [f32; 6], u8) {
        let rank = MARKOV_RANK;
        let mut p = [0.0f32; 6];
        let mut survival = [1.0f32; 6];
        let mut prod = 1.0f32;
        for k in 1..BLOCK {
            // draft k is scored against hidden row k (the row that produced it) — the
            // oracle/reference mapping; h is the full 7-row readback.
            let hk = &h[k * HIDDEN..(k + 1) * HIDDEN];
            let lk = &latents[(k - 1) * rank..k * rank];
            let mut c = self.conf_b;
            for i in 0..HIDDEN {
                c += self.conf_w[i] * hk[i];
            }
            for i in 0..rank {
                c += self.conf_w[HIDDEN + i] * lk[i];
            }
            let pk = 1.0f32 / (1.0f32 + (-c).exp());
            p[k - 1] = pk;
            prod *= pk;
            survival[k - 1] = prod;
        }
        // oracle truncate(): k+1 drafts survive → width = (k+1)+1; full = 8
        let mut k_verify = 8u8;
        for (kk, &s) in survival.iter().enumerate() {
            if s < 0.15 {
                k_verify = (kk + 2) as u8;
                break;
            }
        }
        (p, survival, k_verify)
    }

    pub fn reset(&mut self) {
        self.nprev = 0;
        // A6 identity (PLAN/DSPARK_RING_IDENTITY_SPEC.md §3.1): a reset drops the ring's identity
        // claim as well as its frontier — rows still physically present must not be carried by a
        // later request. Exact twin of `Df2Round::reset`.
        self.ring_slot = None;
    }

    /// A6 identity: the slot whose committed prefix the ring currently reflects (`None` = unproven).
    pub fn ring_slot(&self) -> Option<usize> { self.ring_slot }

    /// A6 identity: bind the ring to `slot` at the CURRENT committed frontier. Called after the
    /// lane step, so `nprev` is the whole committed sequence this ring holds.
    pub fn note_ring(&mut self, slot: usize) {
        self.ring_slot = Some(slot);
    }

    /// A6 identity: drop the claim — the ring's contents can no longer be attributed to any slot's
    /// prefix (a failed/half-way prime leaves it partially overwritten).
    pub fn invalidate_ring(&mut self) {
        self.ring_slot = None;
    }

    /// A6 prefix-carry: rewind the ring to `nprev` committed rows. The rows [0, nprev) are a
    /// pure function of the token prefix [0, nprev) — when a new request reuses exactly that
    /// prefix, they stay valid and only the suffix needs priming. Rows past nprev are never
    /// read (attention spans [0, nprev + BLOCK) and the suffix prime overwrites them).
    /// NOTE: this only moves the frontier — the identity claim is deliberately NOT touched (the
    /// claim is about WHOSE rows these are, which a rewind does not change). Twin of
    /// `Df2Round::rewind`.
    pub fn rewind(&mut self, nprev: usize) {
        self.nprev = nprev;
    }
}
