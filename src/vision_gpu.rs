//! GPU vision tower — the FP32 port of `vision_encoder::VisualTower::forward_cpu`.
//!
//! This is the P1 implementation of the vision GPU port (PLAN/W2_GPU_PORT_HANDOFF.md). It reproduces
//! the CPU reference forward EXACTLY in data type: the on-disk BF16 weights are loaded to FP32 (the
//! CPU `forward_cpu` does the same) and every matmul / layernorm / attention runs in FP32. This is
//! the only way to sit inside the binding per-block rel-L2 envelope (PLAN/W2_PREPROC_SPEC.md §10,
//! ~2.9e-5) — a BF16 GEMM path is ~1e-3 rel-L2 off the FP32 oracle. The vision tower NEVER touches
//! the NVFP4/FP8 dequant chain (AGENTS §2.4 / §6): it is the plain BF16-weight -> FP32 class.
//!
//! GEMMs are batched over the token count N via the engine's cuBLAS `GemmConfig` pattern (the
//! `gemm_act` gpu.rs:4693 batch>2 arm); activations are row-major `[N, D]`, which cuBLAS reads as a
//! column-major `[D, N]` B matrix with `ldb = D` — no transposes needed. Attention is the new
//! `vision_attn` kernel (head_dim 72, parameterized via template instantiation).

use crate::gpu::{fork_blocking_stream, Pool, S};
use crate::vision_tower::{VisualBlock, VisualTower};
use anyhow::Result;
use cudarc::cublas::{sys::cublasOperation_t as OP, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaFunction, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::Arc;

/// Per-ViT-block GPU weights (f32, uploaded from the CPU bf16->f32 tower).
struct GpuBlock {
    norm1_w: S, norm1_b: S, norm2_w: S, norm2_b: S,
    qkv_w: S, qkv_b: S, proj_w: S, proj_b: S,
    fc1_w: S, fc1_b: S, fc2_w: S, fc2_b: S,
}

const LAYERNORM_EPS: f32 = 1e-6;

macro_rules! vlaunch {
    ($s:expr, $name:expr, $g:expr, $b:expr, $smem:expr, ($($a:expr),+ $(,)?)) => {
        unsafe {
            let f = $s.k.get($name).cloned().unwrap_or_else(|| panic!("vision kernel {}", $name));
            f.launch_on_stream(&$s.stream,
                LaunchConfig { grid_dim: $g, block_dim: $b, shared_mem_bytes: $smem },
                ($($a),+)
            ).unwrap_or_else(|e| panic!("vision launch {}: {:?}", $name, e));
        }
    };
}

/// The GPU vision tower: weights on device + the FP32 kernels + a stream/pool.
pub struct GpuVisualTower {    dev: Arc<CudaDevice>,
    blas: CudaBlas,
    stream: cudarc::driver::CudaStream,
    pool: Pool,
    k: HashMap<String, CudaFunction>,
    /// CPU weights retained for the host-side pos-embed / rotary tables.
    host: VisualTower,
    patch_w: S, patch_b: S,
    blocks: Vec<GpuBlock>,
    merger_norm_w: S, merger_norm_b: S,
    merger_fc1_w: S, merger_fc1_b: S,
    merger_fc2_w: S, merger_fc2_b: S,
}

fn d<T>(s: &cudarc::driver::CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }
fn grid(n: usize) -> (u32, u32, u32) { (((n + 255) / 256) as u32, 1, 1) }
fn fbits(x: f32) -> u64 { x.to_bits() as u64 }

// The cudarc `CudaStream`/`CudaBlas` wrap raw CUDA handles (auto !Send). The serving pattern
// (`unsafe impl Send for GpuModel`, gpu.rs:981) moves these across the scheduler thread; here the
// GpuVisualTower is shared across server workers behind a Mutex, so all CUDA access is serialized
// and the tower is never touched concurrently. Same safety argument as GpuModel.
unsafe impl Send for GpuVisualTower {}

impl GpuVisualTower {
    /// Build the GPU tower on the given device, uploading the CPU tower's weights as FP32.
    pub fn new(dev: Arc<CudaDevice>, tower: &VisualTower) -> Result<Self> {
        let stream = fork_blocking_stream(&dev);
        let blas = CudaBlas::new(dev.clone())?;
        unsafe { blas.set_stream(Some(&stream))?; }
        let pool = Pool::new(dev.clone());

        // Load the FP32 vision kernels.
        let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_vision.ptx")?);
        let fnames = [
            "vision_layernorm", "vision_add_inplace", "vision_bias_add",
            "vision_gelu_tanh", "vision_gelu", "vision_attn", "vision_attn_generic",
            "vision_rope_split_transpose", "vision_softmax_rows", "vision_o_write",
            "vision_kernel_build_id",
        ];
        dev.load_ptx(ptx, "gpu_vision", &fnames)?;
        let mut k = HashMap::new();
        for n in &fnames {
            if let Some(f) = dev.get_func("gpu_vision", n) { k.insert(n.to_string(), f); }
        }

        let up = { let dev = dev.clone(); move |v: &[f32]| -> S { dev.htod_sync_copy(v).unwrap() } };
        let blocks = tower.blocks.iter().map(|b| GpuBlock {
            norm1_w: up(&b.norm1_w), norm1_b: up(&b.norm1_b),
            norm2_w: up(&b.norm2_w), norm2_b: up(&b.norm2_b),
            qkv_w: up(&b.qkv_w), qkv_b: up(&b.qkv_b),
            proj_w: up(&b.proj_w), proj_b: up(&b.proj_b),
            fc1_w: up(&b.fc1_w), fc1_b: up(&b.fc1_b),
            fc2_w: up(&b.fc2_w), fc2_b: up(&b.fc2_b),
        }).collect();

        Ok(GpuVisualTower {
            dev, blas, stream, pool, k, host: tower.clone(),
            patch_w: up(&tower.patch_embed_w), patch_b: up(&tower.patch_embed_b),
            blocks,
            merger_norm_w: up(&tower.merger_norm_w), merger_norm_b: up(&tower.merger_norm_b),
            merger_fc1_w: up(&tower.merger_fc1_w), merger_fc1_b: up(&tower.merger_fc1_b),
            merger_fc2_w: up(&tower.merger_fc2_w), merger_fc2_b: up(&tower.merger_fc2_b),
        })
    }

    /// Batched FP32 GEMM: `out[N,outn] = x[N,inn] @ w[outn,inn]^T` (row-major activations / weights).
    fn gemm(&self, w: &S, x: &S, out: &mut S, inn: usize, outn: usize, n: usize) -> Result<()> {
        let cfg = GemmConfig::<f32> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: outn as i32, n: n as i32, k: inn as i32,
            alpha: 1.0, lda: inn as i32, ldb: inn as i32, beta: 0.0, ldc: outn as i32,
        };
        unsafe { self.blas.gemm(cfg, w, x, out).map_err(|e| anyhow::anyhow!("vision gemm f32: {e:?}")) }
    }

    fn layernorm(&self, out: &mut S, x: &S, w: &S, b: &S, n: usize, dim: usize) {
        let smem = (2 * 256 * 4) as u32;
        vlaunch!(self, "vision_layernorm", (n as u32, 1, 1), (256, 1, 1), smem,
            (d(out), d(x), d(w), d(b), n as i32, dim as i32, fbits(LAYERNORM_EPS)));
    }

    fn bias_add(&self, out: &mut S, bias: &S, rows: usize, outn: usize) {
        let total = rows * outn;
        vlaunch!(self, "vision_bias_add", grid(total), (256, 1, 1), 0,
            (d(out), d(bias), rows as i32, outn as i32));
    }

    fn add_inplace(&self, out: &mut S, src: &S, n: usize) {
        vlaunch!(self, "vision_add_inplace", grid(n), (256, 1, 1), 0, (d(out), d(src), n as i32));
    }

    fn gelu_tanh(&self, out: &mut S, n: usize) {
        vlaunch!(self, "vision_gelu_tanh", grid(n), (256, 1, 1), 0, (d(out), n as i32));
    }

    fn gelu(&self, out: &mut S, n: usize) {
        vlaunch!(self, "vision_gelu", grid(n), (256, 1, 1), 0, (d(out), n as i32));
    }

    /// Full non-causal MHA over N tokens (one packed chunk, no KV cache). qkv [N, 3*hidden].
    /// cuBLAS-based: per-head [N, hd] row-major q/k/v (transpose+RoPE kernel), then the engine's
    /// batched GEMM pattern. S = scale*(Q·K^T), softmax over keys, O = P·V. This replaces the scalar
    /// flash kernel that was O(N²) serial and ~35 s on a 12K-patch image; cuBLAS runs the same O(N²)
    /// math on the tensor/FP32 path far faster. Confined to the vision tower (AGENTS §2 — the text
    /// prefill/decode attention path is untouched).
    fn attention(&self, qkv: &S, cos: &S, sin: &S, out: &mut S, n: usize,
                 qo: &S, ko: &S, vo: &S, s: &mut S, o: &mut S) -> Result<()> {
        let (heads, hd, hidden) = (self.host.dims.heads, self.host.dims.head_dim(), self.host.dims.hidden);
        let scale = (hd as f32).powf(-0.5);
        let nhd = n * hd;          // per-head elems
        let nhead = heads * nhd;   // all heads' q/k/v elems
        let nn = n * n;            // score elems per head
        let n_o = n * hd;          // O elems per head

        // qkv [N, 3*hidden] -> per-head contiguous [N, hd] roped q/k/v (transpose + RoPE, vision-only).
        vlaunch!(self, "vision_rope_split_transpose", grid(nhead), (256, 1, 1), 0,
            (d(qkv), d(cos), d(sin), d(qo), d(ko), d(vo),
             n as i32, heads as i32, hidden as i32, hd as i32));

        for h in 0..heads {
            let st = h * nhd;
            let (q_view, k_view, v_view) = (qo.slice(st..st + nhd), ko.slice(st..st + nhd),
                                            vo.slice(st..st + nhd));
            // S = scale * (Q @ K^T): Q,K are [N, hd] row-major -> transa=T / transb=N.
            let cfg = GemmConfig::<f32> {
                transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
                m: n as i32, n: n as i32, k: hd as i32,
                alpha: scale, lda: hd as i32, ldb: hd as i32, beta: 0.0, ldc: n as i32,
            };
            unsafe { self.blas.gemm(cfg, &q_view, &k_view, s).map_err(|e| anyhow::anyhow!("vision QK gemm: {e:?}"))?; }
            // softmax over the key dim (in place S -> P). Coalesced N-thread-per-query kernel.
            vlaunch!(self, "vision_softmax_rows", grid(n), (256, 1, 1), 0, (d(s), n as i32));
            // O = P @ V: P [N,N] col-major, V [N, hd] row-major -> transb=T. C is [N, hd]
            // column-major at o[tq + d*N], so ldc = N (must be >= m=N).
            let cfg2 = GemmConfig::<f32> {
                transa: OP::CUBLAS_OP_N, transb: OP::CUBLAS_OP_T,
                m: n as i32, n: hd as i32, k: n as i32,
                alpha: 1.0, lda: n as i32, ldb: hd as i32, beta: 0.0, ldc: n as i32,
            };
            unsafe { self.blas.gemm(cfg2, s, &v_view, o).map_err(|e| anyhow::anyhow!("vision PV gemm: {e:?}"))?; }
            vlaunch!(self, "vision_o_write", grid(n_o), (256, 1, 1), 0,
                (d(o), d(out), n as i32, hidden as i32, hd as i32, h as i32));
        }
        Ok(())
    }

    /// Forward the vision tower. Returns the merged image embeddings `[N/merge^2, OUT_HIDDEN]` as f32.
    /// When `trace` is set, also returns the oracle-ordered hidden states: states[0] = pre_blocks
    /// (post patch_embed + pos_embed), states[1+k] = block_k (k = 0..26).
    pub fn forward(&mut self, pixel_values: &[f32], gh: usize, gw: usize, trace: bool) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        self.dev.bind_to_thread()?;
        let d = self.host.dims;
        let (hidden, inter, merge) = (d.hidden, d.inter, d.merge);
        let (hd, mi) = (d.head_dim(), d.merge_inter());
        let wpv = d.wpv();
        let n = gh * gw;
        let tn = n / (merge * merge);
        assert_eq!(pixel_values.len(), n * wpv, "pixel_values len");

        // Host tables (pos-embed bilinear + rotary cos/sin) — the same functions the CPU path uses.
        let pe = crate::vision_encoder::pos_embed_bilinear(
            &self.host.pos_embed_w, gh, gw, d.num_side(), hidden, merge);
        let (cos, sin) = crate::vision_encoder::vision_cos_sin(gh, gw, hd, merge);

        let pv = self.dev.htod_sync_copy(pixel_values)?;
        let pe_g = self.dev.htod_sync_copy(&pe)?;
        let cos_g = self.dev.htod_sync_copy(&cos)?;
        let sin_g = self.dev.htod_sync_copy(&sin)?;

        let mut h = self.pool.get(n * hidden);
        self.gemm(&self.patch_w, &pv, &mut h, wpv, hidden, n)?;
        self.bias_add(&mut h, &self.patch_b, n, hidden);
        // pos-embed
        self.add_inplace(&mut h, &pe_g, n * hidden);

        let mut states = Vec::new();
        if trace { states.push(self.to_host(&h, n * hidden)?); }   // pre_blocks

        // cuBLAS-attention scratch (reused across blocks). Sized for the current N.
        let (nhd_s, nhead_s, nn_s, n_o_s) = (n * hd, d.heads * n * hd, n * n, n * hd);
        let mut aq = self.pool.get(nhead_s);
        let mut ak = self.pool.get(nhead_s);
        let mut av = self.pool.get(nhead_s);
        let mut as_ = self.pool.get(nn_s);
        let mut ao = self.pool.get(n_o_s);

        for blk in &self.blocks {
            let mut norm1 = self.pool.get(n * hidden);
            self.layernorm(&mut norm1, &h, &blk.norm1_w, &blk.norm1_b, n, hidden);
            let mut qkv = self.pool.get(n * 3 * hidden);
            self.gemm(&blk.qkv_w, &norm1, &mut qkv, hidden, 3 * hidden, n)?;
            self.bias_add(&mut qkv, &blk.qkv_b, n, 3 * hidden);
            let mut attn = self.pool.get(n * hidden);
            self.attention(&qkv, &cos_g, &sin_g, &mut attn, n, &aq, &ak, &av, &mut as_, &mut ao)?;
            self.pool.release(qkv, n * 3 * hidden);
            let mut proj = self.pool.get(n * hidden);
            self.gemm(&blk.proj_w, &attn, &mut proj, hidden, hidden, n)?;
            self.bias_add(&mut proj, &blk.proj_b, n, hidden);
            self.pool.release(attn, n * hidden);
            self.add_inplace(&mut h, &proj, n * hidden);
            self.pool.release(proj, n * hidden);

            let mut norm2 = self.pool.get(n * hidden);
            self.layernorm(&mut norm2, &h, &blk.norm2_w, &blk.norm2_b, n, hidden);
            let mut fc1 = self.pool.get(n * inter);
            self.gemm(&blk.fc1_w, &norm2, &mut fc1, hidden, inter, n)?;
            self.bias_add(&mut fc1, &blk.fc1_b, n, inter);
            self.gelu_tanh(&mut fc1, n * inter);
            let mut fc2 = self.pool.get(n * hidden);
            self.gemm(&blk.fc2_w, &fc1, &mut fc2, inter, hidden, n)?;
            self.bias_add(&mut fc2, &blk.fc2_b, n, hidden);
            self.pool.release(fc1, n * inter);
            self.add_inplace(&mut h, &fc2, n * hidden);
            self.pool.release(fc2, n * hidden);
            self.pool.release(norm1, n * hidden);
            self.pool.release(norm2, n * hidden);
            if trace { states.push(self.to_host(&h, n * hidden)?); }
        }
        self.pool.release(aq, nhead_s); self.pool.release(ak, nhead_s); self.pool.release(av, nhead_s);
        self.pool.release(as_, nn_s); self.pool.release(ao, n_o_s);

        // merger: layernorm -> view [N,hidden] as [tn,mi] -> fc1 -> gelu -> fc2
        let mut ln = self.pool.get(n * hidden);
        self.layernorm(&mut ln, &h, &self.merger_norm_w, &self.merger_norm_b, n, hidden);
        self.pool.release(h, n * hidden);
        let mut mfc1 = self.pool.get(tn * mi);
        self.gemm(&self.merger_fc1_w, &ln, &mut mfc1, mi, mi, tn)?;
        self.pool.release(ln, n * hidden);
        self.bias_add(&mut mfc1, &self.merger_fc1_b, tn, mi);
        self.gelu(&mut mfc1, tn * mi);
        let mut out = self.pool.get(tn * d.out_hidden);
        self.gemm(&self.merger_fc2_w, &mfc1, &mut out, mi, d.out_hidden, tn)?;
        self.pool.release(mfc1, tn * mi);
        self.bias_add(&mut out, &self.merger_fc2_b, tn, d.out_hidden);

        let merged = self.to_host(&out, tn * d.out_hidden)?;
        self.pool.release(out, tn * d.out_hidden);
        self.sync()?;
        Ok((merged, states))
    }

    fn to_host(&self, buf: &S, n: usize) -> Result<Vec<f32>> {
        self.sync()?;
        let mut v = self.dev.dtoh_sync_copy(buf).map_err(|e| anyhow::anyhow!("dtoh_sync_copy: {e:?}"))?;
        debug_assert!(v.len() >= n);
        v.truncate(n);
        Ok(v)
    }

    fn sync(&self) -> Result<()> {
        unsafe { cudarc::driver::result::stream::synchronize(self.stream.stream) }
            .map_err(|e| anyhow::anyhow!("stream sync: {e:?}"))
    }

    /// The host-side tower (dims + CPU weights retained for pos-embed / rotary tables / preproc).
    pub fn host(&self) -> &VisualTower {
        &self.host
    }

    /// Number of merged image tokens this grid produces.
    pub fn num_tokens(&self, gh: usize, gw: usize) -> usize {
        (gh * gw) / (self.host.dims.merge * self.host.dims.merge)
    }
}
