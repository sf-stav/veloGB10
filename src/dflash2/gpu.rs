//! S3F — the DFlash2 draft-block forward as device kernels (K-DF2-1).
//!
//! A standalone, single-process GPU pass: load the REAL 81-tensor artifact, upload it per the
//! dossier's layout, and run `tap_project → draft_kv_write → 5-layer backbone → final norm` on
//! device. It diffs against the oracle (via `mirror.rs`'s bf16-staged reference); the selector,
//! LM head, trunk tap capture, CUDA graphs and the round loop are S4F/S5F, NOT here.
//!
//! # Layout (frozen for S4F — see PLAN/B8_S3F_REPORT.md R6)
//!
//! * Activations: col-major `[dim, B]` bf16 (the engine convention).
//! * Weights: row-major `[out, in]` bf16; per-tensor buffers (quantization-friendly — no baked
//!   dtypes in the launch API, the FP8 axis is S8F).
//! * Norm weights: uploaded as f32 `w − 1` — the reused `(1+w)` rmsnorm kernels then compute the
//!   plain `w·x` this family needs (DECISION F; the same `gpu.rs:3272` hy_v3 trick).
//! * RoPE: θ=1e7 half-split `rotate_half` (DECISION D/H), cos/sin `[max_pos, head_dim]` f32 with
//!   the duplicated-freqs layout the engine's `gather_rope_b`/`rope_b` expect; NO YaRN.
//! * Draft KV: per layer `[nkv, ntot, hd]` bf16 (single slot), ctx rows 0..C−1, block rows C..C+7.
//!
//! # Kernels
//!
//! * NEW (gpu_kernels.cu): `gemm_dsp_b_m8_r{2,4}` (skinny M=8 GEMM), `gemm_tiled_b` (ctx-side
//!   large-M GEMM), `conv2_dynamic_b` (grouped causal 2-tap conv), `gqa_attn_band_b` (band-masked
//!   dual-source GQA attention).
//! * REUSED (gpu_batch.cu): `rmsnorm_b`, `rmsnorm_perhead_b`, `rope_b`, `gather_rope_b`,
//!   `write_kv_b`, `add_residual_b`, `silu_mul_b`.

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::collections::HashMap;
use std::sync::Arc;

use crate::dflash2::mirror;
use crate::dflash2::oracle::Dflash2Config;
use crate::dflash2::{BLOCK, CONV_GROUP, CONV_GROUPS, CONV_KERNEL, HEAD_DIM, HIDDEN, INTER, N_LAYERS,
                       NUM_HEADS, NUM_KV_HEADS, RMS_EPS, TAP_CONCAT_DIM};

fn d<T>(s: &CudaSlice<T>) -> u64 {
    *s.device_ptr()
}
fn grid(n: usize) -> (u32, u32, u32) {
    (((n + 255) / 256) as u32, 1, 1)
}
fn fbits(x: f32) -> u64 {
    x.to_bits() as u64
}

pub(crate) fn fork_blocking_stream(dev: &Arc<CudaDevice>) -> cudarc::driver::CudaStream {
    use cudarc::driver::result::stream::{create, destroy, StreamKind};
    let mut s = dev.fork_default_stream().expect("fork stream");
    unsafe {
        destroy(s.stream).expect("destroy nonblocking stream");
        s.stream = create(StreamKind::Default).expect("create blocking stream");
    }
    s
}

/// One NVFP4 weight tensor on device, MMA-repacked (`quant::repack_nvfp4_mma`) and ready for
/// `gemm_mma_fp4_b` (the trunk's persistent fp4 GEMM; the drafter's shapes all satisfy its
/// M%16/K%32 constraints). PLAN/25 §1a: the per-step block pass streams these instead of bf16 —
/// 3.59 GB → 1.01 GB of weight bytes per round.
pub(crate) struct Nvfp4Dev {
    pub(crate) wt: CudaSlice<u8>,
    pub(crate) st: CudaSlice<u8>,
    pub(crate) gs: CudaSlice<f32>,
    pub(crate) m: usize,
    pub(crate) k: usize,
}

/// A drafter linear weight: BF16 (the original artifact, and everything the probe path runs),
/// NVFP4 (`--df2-bake-nvfp4`) or FP8-E4M3 (`--df2-bake-fp8`). Dispatched in `round.rs::gemm_lin`.
pub(crate) enum Df2W {
    Bf16(CudaSlice<bf16>),
    Nvfp4(Nvfp4Dev),
    Fp8(Fp8Dev),
}

impl Df2W {
    /// The BF16 slice (the probe/oracle path only ever builds this arm).
    pub(crate) fn bf16(&self) -> &CudaSlice<bf16> {
        match self { Df2W::Bf16(s) => s, Df2W::Nvfp4(_) | Df2W::Fp8(_) => unreachable!("probe path is bf16-only") }
    }
}

/// Device state for `upload_fp8` — the twin of [`Nvfp4Dev`] for the trunk's
/// `gemm_mma_fp8_b` (8-bit weights + a per-ROW f32 scale multiplier).
pub(crate) struct Fp8Dev {
    pub(crate) wt: CudaSlice<u8>,
    pub(crate) rs: CudaSlice<f32>,
    pub(crate) m: usize,
    pub(crate) k: usize,
}

/// Upload a packed NVFP4 tensor: MMA-repack on host (the trunk's own flow — quantize at bake,
/// repack at load), then the three device buffers `gemm_mma_fp4_b` consumes.
pub(crate) fn upload_nvfp4(dev: &Arc<CudaDevice>, p: &crate::dflash2::load::PackedNvfp4) -> Nvfp4Dev {
    let (wt, st) = crate::quant::repack_nvfp4_mma(&p.qweight, &p.scales, p.m, p.k);
    // The file's weight_global_scale is the trunk --quantize emit convention: a DIVISOR
    // (6*448/amax). The fp4 kernels MULTIPLY by gs and index it PER 16-ROW TILE (gs[mt]) —
    // upload the reciprocal REPLICATED to m/16 entries, byte-for-byte the trunk loader's own
    // convention (gpu.rs: `let gsv = vec![1.0f32 / gs; rn / 16]`).
    Nvfp4Dev {
        wt: dev.htod_sync_copy(&wt).expect("upload fp4 weights"),
        st: dev.htod_sync_copy(&st).expect("upload fp4 scales"),
        gs: dev.htod_sync_copy(&vec![1.0f32 / p.global_scale; p.m / 16]).expect("upload fp4 gs"),
        m: p.m,
        k: p.k,
    }
}

/// Upload a packed FP8 tensor: MMA-repack on host (`quant::repack_fp8_mma`, the trunk's own
/// DSV4-attention flow), then the two device buffers `gemm_mma_fp8_b` consumes. The row scales
/// MULTIPLY (no divisor, no per-tile replication — the epilogue indexes `rs[m]` directly).
pub(crate) fn upload_fp8(dev: &Arc<CudaDevice>, p: &crate::dflash2::load::PackedFp8) -> Fp8Dev {
    let wt = crate::quant::repack_fp8_mma(&p.qweight, p.m, p.k);
    Fp8Dev {
        wt: dev.htod_sync_copy(&wt).expect("upload fp8 weights"),
        rs: dev.htod_sync_copy(&p.row_scale).expect("upload fp8 row scales"),
        m: p.m,
        k: p.k,
    }
}

/// One draft-backbone layer's device weights (row-major; norms f32 as w−1). The 9 linears are
/// `Df2W` (BF16 on the original artifact, NVFP4 on a baked one); `k_proj_bf16`/`v_proj_bf16`
/// carry the BF16 twins `prime_window` needs in nvfp4 mode (`gemm_mma_fp4_b` is N≤16, prime is
/// not; fc's twin lives on `GpuGlobal`).
pub(crate) struct GpuLayer {
    pub(crate) q_proj: Df2W,     // [4096, 5120]
    pub(crate) k_proj: Df2W,     // [1024, 5120]
    pub(crate) k_proj_bf16: Option<CudaSlice<bf16>>,
    pub(crate) v_proj: Df2W,     // [1024, 5120]
    pub(crate) v_proj_bf16: Option<CudaSlice<bf16>>,
    pub(crate) o_proj: Df2W,     // [5120, 4096]
    pub(crate) gate_proj: Df2W,  // [17408, 5120]
    pub(crate) up_proj: Df2W,    // [17408, 5120]
    pub(crate) down_proj: Df2W,  // [5120, 17408]
    pub(crate) q_norm: CudaSlice<f32>,      // [128] (w−1)
    pub(crate) k_norm: CudaSlice<f32>,      // [128] (w−1)
    pub(crate) input_ln: CudaSlice<f32>,    // [5120] (w−1)
    pub(crate) post_ln: CudaSlice<f32>,     // [5120] (w−1)
    pub(crate) attn_kp: Df2W,    // [1280, 5120]
    pub(crate) attn_base: CudaSlice<bf16>,  // [4, 5120] ([2,kernel,hidden] flattened)
    pub(crate) mlp_kp: Df2W,     // [1280, 5120]
    pub(crate) mlp_base: CudaSlice<bf16>,   // [4, 5120]
}

/// The global (non-layer) device weights. `fc_bf16` = the prime twin (see `GpuLayer`).
pub(crate) struct GpuGlobal {
    pub(crate) fc: Df2W,                     // [5120, 25600]
    pub(crate) fc_bf16: Option<CudaSlice<bf16>>,
    pub(crate) hidden_norm: CudaSlice<f32>,  // [5120] (w−1)
    pub(crate) norm: CudaSlice<f32>,         // [5120] (w−1)
}

/// Block-shaped scratch (fixed width BLOCK=8), persistent. Separate attn/mlp buffers so every
/// per-piece diff reads the value the kernel actually wrote (no reuse overwrite).
struct BlockScratch {
    normed: CudaSlice<bf16>,   // [5120, 8] input_layernorm out
    x_conv: CudaSlice<bf16>,   // [5120, 8] attn conv prepare
    dyn_attn: CudaSlice<bf16>,      // [1280, 8] attn kernel_projection out
    q: CudaSlice<bf16>,        // [4096, 8]
    k: CudaSlice<bf16>,        // [1024, 8]
    v: CudaSlice<bf16>,        // [1024, 8]
    attn: CudaSlice<bf16>,     // [4096, 8] attention out (pre-o_proj)
    attn_out: CudaSlice<bf16>, // [5120, 8] o_proj out
    fin: CudaSlice<bf16>,      // [5120, 8] attn conv finish
    normed2: CudaSlice<bf16>,  // [5120, 8] post_attention_layernorm out
    x_conv2: CudaSlice<bf16>,  // [5120, 8] mlp conv prepare
    dyn_mlp: CudaSlice<bf16>,     // [1280, 8] mlp kernel_projection out
    gate: CudaSlice<bf16>,     // [17408, 8]
    up: CudaSlice<bf16>,       // [17408, 8]
    mlp_out: CudaSlice<bf16>,  // [5120, 8] down_proj out
    fin2: CudaSlice<bf16>,     // [5120, 8] mlp conv finish
    h: CudaSlice<bf16>,        // [5120, 8] the residual
    h_final: CudaSlice<bf16>,  // [5120, 8]
    cos8: CudaSlice<f32>,      // [8, 128]
    sin8: CudaSlice<f32>,
    slot_ids: CudaSlice<i32>,  // [8] all 0 (single slot)
}

/// The standalone DFlash2 draft-block pass.
pub struct Df2Gpu {
    pub dev: Arc<CudaDevice>,
    stream: cudarc::driver::CudaStream,
    bk: HashMap<String, CudaFunction>,
    layers: Vec<GpuLayer>,
    glob: GpuGlobal,
    cos_table: CudaSlice<f32>, // [max_pos, head_dim]
    sin_table: CudaSlice<f32>,
    pos_block: CudaSlice<i32>, // [8]
    k_cache: Vec<CudaSlice<bf16>>, // [layer] [8, max_c+8, 128]
    v_cache: Vec<CudaSlice<bf16>>,
    caches_n: usize,
    blk: BlockScratch,
    max_c: usize,
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

pub(crate) fn upload_bf16(dev: &Arc<CudaDevice>, data: &[f32]) -> CudaSlice<bf16> {
    let b: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
    dev.htod_sync_copy(&b).expect("upload bf16")
}
pub(crate) fn upload_norm(dev: &Arc<CudaDevice>, data: &[f32]) -> CudaSlice<f32> {
    let w: Vec<f32> = data.iter().map(|&x| x - 1.0).collect();
    dev.htod_sync_copy(&w).expect("upload norm")
}

impl Df2Gpu {
    /// Load the artifact + upload weights + build RoPE tables + allocate scratch for `max_c`.
    pub fn load(dir: &str, max_c: usize) -> Result<Self> {
        Self::load_pinned(dir, max_c, None)
    }

    /// `load` with an explicit artifact sha256 pin override (see `round::load_pinned`).
    pub fn load_pinned(dir: &str, max_c: usize, sha_pin: Option<&str>) -> Result<Self> {
        let pin: Option<&str> = match sha_pin {
            Some("off") => None,
            Some(hex) => Some(hex),
            None => Some(crate::dflash2::REAL_SHA256),
        };
        let art = crate::dflash2::load::load(dir, pin)?;
        let w = &art.weights;
        let cfg = Dflash2Config::default();
        let max_pos = max_c + BLOCK + 1;

        let dev = CudaDevice::new(0).context("CudaDevice")?;
        let stream = fork_blocking_stream(&dev);

        let bptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx")?);
        let bfnames = ["rmsnorm_b", "rmsnorm_perhead_b", "rope_b", "gather_rope_b",
            "write_kv_b", "add_residual_b", "silu_mul_b", "kernel_build_id"];
        dev.load_ptx(bptx, "gpu_batch", &bfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_batch")?;
        let kptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_kernels.ptx")?);
        let kfnames = ["gemm_dsp_b_m8_r2", "gemm_dsp_b_m8_r4", "gemm_tiled_b",
            "conv2_dynamic_b", "gqa_attn_band_b", "kernel_build_id"];
        dev.load_ptx(kptx, "gpu_kernels", &kfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_kernels")?;
        let mut bk = HashMap::new();
        for n in bfnames.iter().chain(kfnames.iter()) {
            let module = if kfnames.contains(n) { "gpu_kernels" } else { "gpu_batch" };
            bk.insert(n.to_string(), dev.get_func(module, n).with_context(|| format!("kernel {n} not in ptx"))?);
        }

        let mut layers = Vec::with_capacity(N_LAYERS);
        for l in &w.layers {
            layers.push(GpuLayer {
                q_proj: Df2W::Bf16(upload_bf16(&dev, &l.q_proj)),
                k_proj: Df2W::Bf16(upload_bf16(&dev, &l.k_proj)),
                k_proj_bf16: None,
                v_proj: Df2W::Bf16(upload_bf16(&dev, &l.v_proj)),
                v_proj_bf16: None,
                o_proj: Df2W::Bf16(upload_bf16(&dev, &l.o_proj)),
                gate_proj: Df2W::Bf16(upload_bf16(&dev, &l.gate_proj)),
                up_proj: Df2W::Bf16(upload_bf16(&dev, &l.up_proj)),
                down_proj: Df2W::Bf16(upload_bf16(&dev, &l.down_proj)),
                q_norm: upload_norm(&dev, &l.q_norm),
                k_norm: upload_norm(&dev, &l.k_norm),
                input_ln: upload_norm(&dev, &l.input_ln),
                post_ln: upload_norm(&dev, &l.post_ln),
                attn_kp: Df2W::Bf16(upload_bf16(&dev, &l.attention_conv.kernel_projection)),
                attn_base: upload_bf16(&dev, &l.attention_conv.base_kernel),
                mlp_kp: Df2W::Bf16(upload_bf16(&dev, &l.mlp_conv.kernel_projection)),
                mlp_base: upload_bf16(&dev, &l.mlp_conv.base_kernel),
            });
        }
        let glob = GpuGlobal {
            fc: Df2W::Bf16(upload_bf16(&dev, &w.fc)),
            fc_bf16: None,
            hidden_norm: upload_norm(&dev, &w.hidden_norm),
            norm: upload_norm(&dev, &w.norm),
        };

        let inv = mirror::inv_freq(&cfg);
        let (cos_t, sin_t) = mirror::rope_tables(&cfg, &inv, max_pos);
        let cos_table = dev.htod_sync_copy(&cos_t)?;
        let sin_table = dev.htod_sync_copy(&sin_t)?;

        let pos_block: Vec<i32> = (0..BLOCK).map(|b| (max_c + b) as i32).collect();
        let pos_block = dev.htod_sync_copy(&pos_block)?;

        let alloc_z = |n: usize| dev.alloc_zeros::<bf16>(n).expect("alloc bf16");
        let caches_n = max_c + BLOCK;
        let mut k_cache = Vec::with_capacity(N_LAYERS);
        let mut v_cache = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            k_cache.push(alloc_z(NUM_KV_HEADS * caches_n * HEAD_DIM));
            v_cache.push(alloc_z(NUM_KV_HEADS * caches_n * HEAD_DIM));
        }

        let blk = BlockScratch {
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
            h: alloc_z(HIDDEN * BLOCK),
            h_final: alloc_z(HIDDEN * BLOCK),
            cos8: dev.alloc_zeros::<f32>(BLOCK * HEAD_DIM).expect("alloc f32"),
            sin8: dev.alloc_zeros::<f32>(BLOCK * HEAD_DIM).expect("alloc f32"),
            slot_ids: dev.alloc_zeros::<i32>(BLOCK).expect("alloc i32"),
        };

        dev.synchronize()?;
        Ok(Self { dev, stream, bk, layers, glob, cos_table, sin_table, pos_block, k_cache, v_cache, caches_n, blk, max_c })
    }

    /// Re-upload layer 0's q_proj (the sign-flip negative control).
    pub fn set_layer0_q_proj(&mut self, data: &[f32]) {
        self.layers[0].q_proj = Df2W::Bf16(upload_bf16(&self.dev, data));
    }

    fn gemm_dsp(&self, out: &CudaSlice<bf16>, w: &CudaSlice<bf16>, x: &CudaSlice<bf16>, outn: usize, inn: usize) {
        let g = ((outn + 3) / 4) as u32; // R=4
        klaunch!(self, "gemm_dsp_b_m8_r4", (g, 1, 1), (256, 1, 1), 0,
            (d(out), d(w), d(x), outn as i32, inn as i32));
    }

    fn gemm_tiled(&self, out: &CudaSlice<bf16>, w: &CudaSlice<bf16>, x: &CudaSlice<bf16>, n: usize, k: usize, m: usize) {
        let mx = ((m + 127) / 128) as u32;
        let nx = ((n + 127) / 128) as u32;
        klaunch!(self, "gemm_tiled_b", (mx, nx, 1), (16, 16, 1), 0,
            (d(out), d(w), d(x), n as i32, k as i32, m as i32));
    }

    fn conv2(&self, out: &CudaSlice<bf16>, x: &CudaSlice<bf16>, dyn_all: &CudaSlice<bf16>,
             base_ptr: u64, n: usize, side: usize) {
        let dyn_side = (side * CONV_KERNEL * CONV_GROUPS) as i32;
        let dyn_stride = (2 * CONV_KERNEL * CONV_GROUPS) as i32;
        klaunch!(self, "conv2_dynamic_b", grid(n * HIDDEN), (256, 1, 1), 0,
            (d(out), d(x), d(dyn_all), base_ptr, HIDDEN as i32, n as i32,
             CONV_GROUPS as i32, CONV_GROUP as i32, dyn_side, dyn_stride));
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
        let rdim = HEAD_DIM;
        klaunch!(self, "rope_b", grid(b * heads * (rdim / 2)), (256, 1, 1), 0,
            (d(x), d(cos), d(sin), heads as i32, HEAD_DIM as i32, rdim as i32, b as i32));
    }

    fn gather_rope(&self, out_cos: &CudaSlice<f32>, out_sin: &CudaSlice<f32>, pos: u64, b: usize) {
        let rdim = HEAD_DIM;
        klaunch!(self, "gather_rope_b", grid(b * rdim), (256, 1, 1), 0,
            (d(out_cos), d(out_sin), d(&self.cos_table), d(&self.sin_table), pos, rdim as i32, b as i32));
    }

    /// The ctx-side k/v for one layer: k = k_norm(RoPE(k_proj(th))), v = v_proj(th) — written to
    /// cache rows 0..C−1.
    fn draft_kv_write(&self, li: usize, c: usize, th: &CudaSlice<bf16>,
                      k_out: &CudaSlice<bf16>, v_out: &CudaSlice<bf16>,
                      cos_c: &CudaSlice<f32>, sin_c: &CudaSlice<f32>,
                      pos_ctx: &CudaSlice<i32>, slot_ids: &CudaSlice<i32>) {
        let l = &self.layers[li];
        self.gemm_tiled(k_out, l.k_proj.bf16(), th, NUM_KV_HEADS * HEAD_DIM, HIDDEN, c);
        self.rmsnorm_perhead(k_out, &l.k_norm, NUM_KV_HEADS, c);
        self.rope(k_out, cos_c, sin_c, NUM_KV_HEADS, c);
        self.gemm_tiled(v_out, l.v_proj.bf16(), th, NUM_KV_HEADS * HEAD_DIM, HIDDEN, c);
        klaunch!(self, "write_kv_b", grid(c * NUM_KV_HEADS * HEAD_DIM), (256, 1, 1), 0,
            (d(&self.k_cache[li]), d(&self.v_cache[li]), d(k_out), d(v_out),
             d(pos_ctx), self.caches_n as i32, NUM_KV_HEADS as i32, HEAD_DIM as i32, c as i32,
             d(slot_ids)));
    }

    /// Run one full pass. `taps` is `[c, 25600]` row-major f32; `anchor` is the block's first
    /// token. `band_window` = the attention sliding window (2048; the negative control passes a
    /// huge value to drop the band). Returns the whole-pass surfaces + (if `dump_pieces`) layer-0
    /// pieces.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&mut self, taps: &[f32], anchor: u32, c: usize, band_window: usize, dump_pieces: bool)
        -> Result<Df2PassOut> {
        assert!(c <= self.max_c, "c {c} > max_c {}", self.max_c);
        let tap_dim = TAP_CONCAT_DIM;

        // ---- taps [c, 25600] row-major f32 == col-major [25600, c] bf16 (same flat layout) ----
        let taps_b: Vec<bf16> = taps.iter().map(|&x| bf16::from_f32(x)).collect();
        let taps_dev = self.dev.htod_sync_copy(&taps_b).context("upload taps")?;

        // ---- per-c device arrays (exact size) ----
        let pos_ctx: Vec<i32> = (0..c).map(|i| i as i32).collect();
        let pos_ctx_dev = self.dev.htod_sync_copy(&pos_ctx).context("pos_ctx")?;
        let pos_block: Vec<i32> = (0..BLOCK).map(|b| (c + b) as i32).collect();
        self.dev.htod_sync_copy_into(&pos_block, &mut self.pos_block).context("pos_block")?;
        let slot_ids: Vec<i32> = vec![0i32; c];
        let slot_ids_dev = self.dev.htod_sync_copy(&slot_ids).context("slot_ids")?;
        let th_raw = self.dev.alloc_zeros::<bf16>(HIDDEN * c).expect("th_raw");
        let th = self.dev.alloc_zeros::<bf16>(HIDDEN * c).expect("th");
        let k_ctx = self.dev.alloc_zeros::<bf16>(NUM_KV_HEADS * HEAD_DIM * c).expect("k_ctx");
        let v_ctx = self.dev.alloc_zeros::<bf16>(NUM_KV_HEADS * HEAD_DIM * c).expect("v_ctx");
        let cos_c = self.dev.alloc_zeros::<f32>(c * HEAD_DIM).expect("cos_c");
        let sin_c = self.dev.alloc_zeros::<f32>(c * HEAD_DIM).expect("sin_c");

        // ---- per-seq RoPE tables ----
        self.gather_rope(&self.blk.cos8, &self.blk.sin8, d(&self.pos_block), BLOCK);
        self.gather_rope(&cos_c, &sin_c, d(&pos_ctx_dev), c);

        // ---- tap projection: th = hidden_norm(fc(taps)) over c rows ----
        self.gemm_tiled(&th_raw, self.glob.fc.bf16(), &taps_dev, HIDDEN, tap_dim, c);
        self.rmsnorm(&th, &th_raw, &self.glob.hidden_norm, HIDDEN, c);

        // ---- draft KV write for all 5 layers (ctx rows 0..C-1) ----
        let mut layer0_k: Option<Vec<f32>> = None;
        let mut layer0_v: Option<Vec<f32>> = None;
        for li in 0..N_LAYERS {
            self.draft_kv_write(li, c, &th, &k_ctx, &v_ctx, &cos_c, &sin_c, &pos_ctx_dev, &slot_ids_dev);
            if li == 0 && dump_pieces {
                let k = self.dev.dtoh_sync_copy(&k_ctx)?;
                let v = self.dev.dtoh_sync_copy(&v_ctx)?;
                layer0_k = Some(k.iter().map(|x| x.to_f32()).collect());
                layer0_v = Some(v.iter().map(|x| x.to_f32()).collect());
            }
        }

        // ---- block input: anchor + 7× MASK synthetic embeddings ----
        {
            let synth = crate::dflash2::synth::SyntheticTables::new(crate::dflash2::SYNTH_EMBED_HEAD_SEED);
            let emb = mirror::block_emb_mirror(&Dflash2Config::default(), &synth, anchor);
            let emb_b: Vec<bf16> = emb.iter().map(|&x| bf16::from_f32(x)).collect();
            self.dev.htod_sync_copy_into(&emb_b, &mut self.blk.h).context("upload emb")?;
        }

        // ---- 5-layer backbone ----
        let mut layer_hiddens = Vec::with_capacity(N_LAYERS);
        let mut layer0_pieces: Option<Df2Pieces> = None;
        for li in 0..N_LAYERS {
            self.layer_forward(li, c, band_window);
            if li == 0 && dump_pieces {
                layer0_pieces = Some(self.dump_pieces(layer0_k.clone(), layer0_v.clone()));
            }
            let h = self.dev.dtoh_sync_copy(&self.blk.h)?;
            layer_hiddens.push(h.iter().map(|x| x.to_f32()).collect());
        }

        // ---- final norm ----
        self.rmsnorm(&self.blk.h_final, &self.blk.h, &self.glob.norm, HIDDEN, BLOCK);

        self.dev.synchronize()?;
        let th = self.dev.dtoh_sync_copy(&th)?;
        let th: Vec<f32> = th.iter().map(|x| x.to_f32()).collect();
        let h = self.dev.dtoh_sync_copy(&self.blk.h_final)?;
        let h: Vec<f32> = h.iter().map(|x| x.to_f32()).collect();
        Ok(Df2PassOut { th, layer_hiddens, h, pieces: layer0_pieces })
    }

    /// One draft layer: attention sublayer + mlp sublayer (with the grouped dynamic convs).
    fn layer_forward(&self, li: usize, c: usize, band_window: usize) {
        let l = &self.layers[li];
        let blk = &self.blk;
        let ntot = c + BLOCK;
        let base1_off = (CONV_KERNEL * HIDDEN * 2) as u64; // byte offset of side 1 in base_kernel
        // attention sublayer
        self.rmsnorm(&blk.normed, &blk.h, &l.input_ln, HIDDEN, BLOCK);
        self.gemm_dsp(&blk.dyn_attn, l.attn_kp.bf16(), &blk.normed, 2 * CONV_KERNEL * CONV_GROUPS, HIDDEN);
        self.conv2(&blk.x_conv, &blk.normed, &blk.dyn_attn, d(&l.attn_base), BLOCK, 0);
        self.gemm_dsp(&blk.q, l.q_proj.bf16(), &blk.x_conv, NUM_HEADS * HEAD_DIM, HIDDEN);
        self.rmsnorm_perhead(&blk.q, &l.q_norm, NUM_HEADS, BLOCK);
        self.rope(&blk.q, &blk.cos8, &blk.sin8, NUM_HEADS, BLOCK);
        self.gemm_dsp(&blk.k, l.k_proj.bf16(), &blk.x_conv, NUM_KV_HEADS * HEAD_DIM, HIDDEN);
        self.rmsnorm_perhead(&blk.k, &l.k_norm, NUM_KV_HEADS, BLOCK);
        self.rope(&blk.k, &blk.cos8, &blk.sin8, NUM_KV_HEADS, BLOCK);
        self.gemm_dsp(&blk.v, l.v_proj.bf16(), &blk.x_conv, NUM_KV_HEADS * HEAD_DIM, HIDDEN);
        klaunch!(self, "write_kv_b", grid(BLOCK * NUM_KV_HEADS * HEAD_DIM), (256, 1, 1), 0,
            (d(&self.k_cache[li]), d(&self.v_cache[li]), d(&blk.k), d(&blk.v),
             d(&self.pos_block), self.caches_n as i32, NUM_KV_HEADS as i32, HEAD_DIM as i32, BLOCK as i32,
             d(&self.slot_ids())));
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let smem = crate::dflash2::band_smem(band_window, ntot);
        let nh_packed = ((NUM_HEADS << 20) | (HEAD_DIM << 10) | NUM_KV_HEADS) as i32;
        let ntot_stride = ((ntot as u64) << 16) | (self.caches_n as u64);
        let window_b = ((band_window << 4) | BLOCK) as i32;
        klaunch!(self, "gqa_attn_band_b", ((BLOCK * NUM_HEADS) as u32, 1, 1), (HEAD_DIM as u32, 1, 1), smem as u32,
            (d(&blk.attn), d(&blk.q), d(&self.k_cache[li]), d(&self.v_cache[li]),
             d(&self.pos_block), ntot_stride, nh_packed, window_b, fbits(scale)));
        self.gemm_dsp(&blk.attn_out, l.o_proj.bf16(), &blk.attn, HIDDEN, NUM_HEADS * HEAD_DIM);
        self.conv2(&blk.fin, &blk.attn_out, &blk.dyn_attn, d(&l.attn_base) + base1_off, BLOCK, 1);
        klaunch!(self, "add_residual_b", grid(HIDDEN * BLOCK), (256, 1, 1), 0,
            (d(&blk.h), d(&blk.h), d(&blk.fin), (HIDDEN * BLOCK) as i32));
        // mlp sublayer
        self.rmsnorm(&blk.normed2, &blk.h, &l.post_ln, HIDDEN, BLOCK);
        self.gemm_dsp(&blk.dyn_mlp, l.mlp_kp.bf16(), &blk.normed2, 2 * CONV_KERNEL * CONV_GROUPS, HIDDEN);
        self.conv2(&blk.x_conv2, &blk.normed2, &blk.dyn_mlp, d(&l.mlp_base), BLOCK, 0);
        self.gemm_dsp(&blk.gate, l.gate_proj.bf16(), &blk.x_conv2, INTER, HIDDEN);
        self.gemm_dsp(&blk.up, l.up_proj.bf16(), &blk.x_conv2, INTER, HIDDEN);
        klaunch!(self, "silu_mul_b", grid(INTER * BLOCK), (256, 1, 1), 0,
            (d(&blk.gate), d(&blk.gate), d(&blk.up), (INTER * BLOCK) as i32));
        self.gemm_dsp(&blk.mlp_out, l.down_proj.bf16(), &blk.gate, HIDDEN, INTER);
        self.conv2(&blk.fin2, &blk.mlp_out, &blk.dyn_mlp, d(&l.mlp_base) + base1_off, BLOCK, 1);
        klaunch!(self, "add_residual_b", grid(HIDDEN * BLOCK), (256, 1, 1), 0,
            (d(&blk.h), d(&blk.h), d(&blk.fin2), (HIDDEN * BLOCK) as i32));
    }

    /// The single-slot (all-zero) id array for the block write (persistent, size 8).
    fn slot_ids(&self) -> &CudaSlice<i32> {
        &self.blk.slot_ids
    }

    /// dtoh layer-0 pieces for the per-piece diff.
    fn dump_pieces(&self, k_ctx: Option<Vec<f32>>, v_ctx: Option<Vec<f32>>) -> Df2Pieces {
        let g = |s: &CudaSlice<bf16>, n: usize| -> Vec<f32> {
            let v = self.dev.dtoh_sync_copy(s).expect("dtoh piece");
            v[..n].iter().map(|x| x.to_f32()).collect()
        };
        let blk = &self.blk;
        Df2Pieces {
            input_ln_out: g(&blk.normed, HIDDEN * BLOCK),
            dyn_attn: g(&blk.dyn_attn, 2 * CONV_KERNEL * CONV_GROUPS * BLOCK),
            q: g(&blk.q, NUM_HEADS * HEAD_DIM * BLOCK),
            k: g(&blk.k, NUM_KV_HEADS * HEAD_DIM * BLOCK),
            v: g(&blk.v, NUM_KV_HEADS * HEAD_DIM * BLOCK),
            attn: g(&blk.attn, NUM_HEADS * HEAD_DIM * BLOCK),
            o: g(&blk.attn_out, HIDDEN * BLOCK),
            x_conv: g(&blk.x_conv, HIDDEN * BLOCK),
            fin: g(&blk.fin, HIDDEN * BLOCK),
            post_ln_out: g(&blk.normed2, HIDDEN * BLOCK),
            x_conv2: g(&blk.x_conv2, HIDDEN * BLOCK),
            dyn_mlp: g(&blk.dyn_mlp, 2 * CONV_KERNEL * CONV_GROUPS * BLOCK),
            mlp_out: g(&blk.mlp_out, HIDDEN * BLOCK),
            fin2: g(&blk.fin2, HIDDEN * BLOCK),
            k_ctx: k_ctx.unwrap_or_default(),
            v_ctx: v_ctx.unwrap_or_default(),
        }
    }
}

/// The whole-pass outputs (col-major [dim, B] flattened).
pub struct Df2PassOut {
    pub th: Vec<f32>,                 // [c, HIDDEN]
    pub layer_hiddens: Vec<Vec<f32>>, // [5][BLOCK*HIDDEN]
    pub h: Vec<f32>,                  // [BLOCK*HIDDEN]
    pub pieces: Option<Df2Pieces>,
}

/// Layer-0 per-piece dumps (for the §3.3 per-piece gates).
pub struct Df2Pieces {
    pub input_ln_out: Vec<f32>,
    pub dyn_attn: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn: Vec<f32>,
    pub o: Vec<f32>,
    pub x_conv: Vec<f32>,
    pub fin: Vec<f32>,
    pub post_ln_out: Vec<f32>,
    pub x_conv2: Vec<f32>,
    pub dyn_mlp: Vec<f32>,
    pub mlp_out: Vec<f32>,
    pub fin2: Vec<f32>,
    pub k_ctx: Vec<f32>,
    pub v_ctx: Vec<f32>,
}
