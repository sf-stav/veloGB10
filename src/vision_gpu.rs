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

use crate::gpu::{fork_blocking_stream, S};
use crate::vision_tower::{VisualBlock, VisualTower};
use anyhow::Result;
use cudarc::cublas::{sys::cublasOperation_t as OP, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{result, sys, CudaDevice, CudaFunction, DevicePtr, DevicePtrMut, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::Arc;

// ---- S-B9-REL-VISIONFIX F1: the tower's own device memory -------------------------------------
//
// The tower runs on a server worker thread while the scheduler thread captures CUDA graphs (at
// boot, and lazily whenever a new step shape is first used). Measured on GB10
// (tests/vision_capture_race_test.rs): while ANY thread has a capture open on a blocking stream, a
// call from another thread that goes through the legacy (NULL) stream is refused with
// CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, the capturing thread's capture is INVALIDATED, and when the
// call races the capture's begin / end the driver has been seen to segfault. Every cudarc "sync"
// helper is such a call (`htod_sync_copy` = cuMemAllocAsync(NULL) + cuMemcpyHtoDAsync(NULL) +
// cuStreamSynchronize(NULL); `dtoh_sync_copy`; `alloc_zeros`; `memset_zeros`; `synchronize`; and
// `CudaSlice::drop` = cuMemFreeAsync(NULL)), and so was the shared `Pool` on a miss. The capture
// mode does not matter (thread-local, global and relaxed behave the same for this).
//
// The same work on the tower's OWN blocking stream is not refused and does not disturb a
// thread-local capture: launches, cuBLAS, cuMemcpy*Async, cuMemsetD8Async, cuMemAllocAsync /
// cuMemFreeAsync and cuStreamSynchronize on that stream. So after construction the tower touches
// device memory only through `TBuf` and the three helpers below (`upload`, `to_host`, `sync`).
// RULE: nothing in `forward` or below it may call a cudarc `*_sync_*` helper, `alloc*`,
// `memset_zeros`, `CudaDevice::synchronize`, or let a `CudaSlice` drop; and no library handle
// (CudaBlas, cuBLASLt) may be created or destroyed there (a destroy is not refused, it invalidates
// another thread's capture: tests/vision_capture_race_test.rs, FACT 7).
//
// INVARIANT (constructor and Drop, S-B9-REL-VISIONFIX2 V2d). Outside the rule above, and by
// design: `new_with_drill` uploads every weight through cudarc's legacy-stream helper
// (`htod_sync_copy(..).unwrap()`: 332 of them on the 27-block tower), creates the cuBLAS handle,
// forks the stream and loads the module; `Drop` frees every weight through the legacy stream
// (`CudaSlice::drop` = cuMemFreeAsync(NULL), 332 of them) and destroys the cuBLAS handle. That is
// safe ONLY because (1) the tower is built before any request exists and before the scheduler can
// have a capture open: the EXL3 server builds it on the boot thread before the scheduler thread
// exists (`vision_boot`, src/exl3_serve.rs), the NVFP4 server builds it before it spawns the
// scheduler task (src/main.rs, before `tokio::spawn(scheduler.run())`); and (2) the tower is never
// dropped while serving: the server holds one `Arc` for the life of the process and the
// poisoned-lock recovery keeps the tower (`into_inner`), it does not rebuild it. A caller that
// builds or drops a `GpuVisualTower` on a serving-time thread, or moves either construction after
// the scheduler can run, breaks this and must first serialize against the scheduler's captures.

/// Device memory owned by the tower: allocated, zeroed, and freed on the tower's own stream.
/// Dropping one is a stream-ordered free on that stream (errors ignored: never a panic in a
/// destructor), so an early `?` return or an unwinding panic in `forward` stays off the legacy stream.
pub(crate) struct TBuf {
    ptr: sys::CUdeviceptr,
    len: usize,
    stream: sys::CUstream,
}

impl TBuf {
    /// `len` f32 on `stream`, zero-filled (the shared pool zeroed its fresh buffers too; reused
    /// buffers are handed out as they are: every kernel writes its output before anything reads it).
    fn alloc(stream: sys::CUstream, len: usize) -> Result<TBuf> {
        let bytes = len * std::mem::size_of::<f32>();
        let ptr = unsafe { result::malloc_async(stream, bytes) }
            .map_err(|e| anyhow::anyhow!("vision tower alloc of {len} f32 on its own stream: {e:?}"))?;
        let buf = TBuf { ptr, len, stream };
        unsafe { result::memset_d8_async(buf.ptr, 0, bytes, stream) }
            .map_err(|e| anyhow::anyhow!("vision tower memset on its own stream: {e:?}"))?;
        Ok(buf)
    }
    /// A borrowed sub-range (per-head q/k/v rows).
    fn view(&self, range: std::ops::Range<usize>) -> TView {
        assert!(range.start <= range.end && range.end <= self.len, "tower view {range:?} outside a {}-f32 buffer", self.len);
        TView { ptr: self.ptr + (range.start * std::mem::size_of::<f32>()) as u64, len: range.end - range.start }
    }
}

impl Drop for TBuf {
    fn drop(&mut self) {
        unsafe { let _ = result::free_async(self.ptr, self.stream); }
    }
}

impl DeviceSlice<f32> for TBuf { fn len(&self) -> usize { self.len } }
impl DevicePtr<f32> for TBuf { fn device_ptr(&self) -> &sys::CUdeviceptr { &self.ptr } }
impl DevicePtrMut<f32> for TBuf { fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr { &mut self.ptr } }

/// A read-only window into a `TBuf` (cuBLAS takes any `DevicePtr`).
#[derive(Clone, Copy)]
pub(crate) struct TView {
    ptr: sys::CUdeviceptr,
    len: usize,
}
impl DeviceSlice<f32> for TView { fn len(&self) -> usize { self.len } }
impl DevicePtr<f32> for TView { fn device_ptr(&self) -> &sys::CUdeviceptr { &self.ptr } }

/// The tower's scratch buffers, kept between requests. Power-of-two buckets (min 256), as the
/// shared `Pool` it replaces, so the reuse pattern and the device memory held are what they were.
pub(crate) struct TPool {
    stream: sys::CUstream,
    free: Vec<TBuf>,
    /// Fresh allocations made (a pool miss). Flat once every bucket a workload needs exists.
    allocs: usize,
}

impl TPool {
    fn new(stream: sys::CUstream) -> Self { TPool { stream, free: Vec::new(), allocs: 0 } }
    fn bucket(n: usize) -> usize { n.max(256).next_power_of_two() }
    fn get(&mut self, n: usize) -> Result<TBuf> {
        let b = Self::bucket(n);
        if let Some(pos) = self.free.iter().position(|x| x.len == b) {
            return Ok(self.free.swap_remove(pos));
        }
        self.allocs += 1;
        // The shared `Pool` this replaced answered an out-of-memory miss by dropping its cached
        // buffers and retrying once (gpu.rs `alloc_f32`). So does this, on the tower's own stream;
        // a second failure is an `Err` (the CPU fallback), never a panic on a worker.
        let (stream, free) = (self.stream, &mut self.free);
        alloc_retry(|| TBuf::alloc(stream, b), |e| {
            let freed: usize = free.iter().map(|x| x.len * 4).sum();
            free.clear(); // stream-ordered frees on the tower's stream
            let _ = unsafe { result::stream::synchronize(stream) };
            // (any allocation error relieves once: the line says what the driver said, it does not assume OOM)
            crate::rprintln!("[vision] tower pool: allocating {b} f32 failed ({e:#}) — dropped {:.1} MB of cached scratch, retrying once",
                             freed as f64 / 1e6);
        })
    }
    fn put(&mut self, buf: TBuf) { self.free.push(buf); }
    fn cached_bytes(&self) -> usize { self.free.iter().map(|b| b.len * 4).sum() }
}

/// `attempt`; on an error `relieve(&error)` once and `attempt` again (the allocation retry of the tower's
/// pool, which answers ANY allocation error that way: pure, so the control flow is unit-tested without a
/// device).
pub(crate) fn alloc_retry<T>(mut attempt: impl FnMut() -> Result<T>, relieve: impl FnOnce(&anyhow::Error)) -> Result<T> {
    match attempt() {
        Ok(v) => Ok(v),
        Err(e) => { relieve(&e); attempt() }
    }
}

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

/// The GPU vision tower: weights on device + the FP32 kernels + its own stream and scratch pool.
pub struct GpuVisualTower {
    dev: Arc<CudaDevice>,
    blas: CudaBlas,
    /// Scratch on the tower's own stream. Declared BEFORE `stream`: fields drop in order, and the
    /// buffers must be freed while their stream still exists.
    pool: TPool,
    stream: cudarc::driver::CudaStream,
    k: HashMap<String, CudaFunction>,
    /// CPU weights retained for the host-side pos-embed / rotary tables.
    host: VisualTower,
    patch_w: S, patch_b: S,
    blocks: Vec<GpuBlock>,
    merger_norm_w: S, merger_norm_b: S,
    merger_fc1_w: S, merger_fc1_b: S,
    merger_fc2_w: S, merger_fc2_b: S,
    /// PR #16: one-shot fault drill (--vision-fail-gemm) — see `vision_gemm_drill`.
    drill: std::cell::Cell<bool>,
}

fn d<P: DevicePtr<f32>>(s: &P) -> u64 { *s.device_ptr() as u64 }
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
        Self::new_with_drill(dev, tower, false)
    }

    /// `new` + the one-shot `--vision-fail-gemm` drill (arms `vision_gemm_drill`).
    pub fn new_with_drill(dev: Arc<CudaDevice>, tower: &VisualTower, fail_first_gemm: bool) -> Result<Self> {
        let stream = fork_blocking_stream(&dev);
        let blas = CudaBlas::new(dev.clone())?;
        // PR #16: pin cuBLAS to the SAME blocking stream the tower's kernels launch on
        // (`fork_blocking_stream` — a StreamKind::Default stream, never a NonBlocking fork;
        // AGENTS §2.1). Unpinned, cuBLAS runs on the NULL stream and every GEMM orders
        // against the kernels only through the legacy-stream barrier; pinned, GEMMs and
        // kernels are simply stream-ordered on one blocking stream.
        unsafe { blas.set_stream(Some(&stream))?; }
        let pool = TPool::new(stream.stream);

        // Load the FP32 vision kernels.
        let ptx = Ptx::from_src(crate::ptx_embed::GPU_VISION);
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

        // Weights: uploaded ONCE here, at boot, through cudarc's legacy-stream helper. This is the
        // only legacy-stream work the tower ever does, and it must stay at construction time (the
        // servers build the tower on the boot thread, before any request can reach a worker).
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
            dev, blas, pool, stream, k, host: tower.clone(),
            patch_w: up(&tower.patch_embed_w), patch_b: up(&tower.patch_embed_b),
            blocks,
            merger_norm_w: up(&tower.merger_norm_w), merger_norm_b: up(&tower.merger_norm_b),
            merger_fc1_w: up(&tower.merger_fc1_w), merger_fc1_b: up(&tower.merger_fc1_b),
            merger_fc2_w: up(&tower.merger_fc2_w), merger_fc2_b: up(&tower.merger_fc2_b),
            drill: std::cell::Cell::new(fail_first_gemm),
        })
    }

    /// Batched FP32 GEMM: `out[N,outn] = x[N,inn] @ w[outn,inn]^T` (row-major activations / weights).
    /// PR #16: returns `Err` on cuBLAS failure — a panic here fired while the tower mutex was
    /// held, poisoning it and killing every later vision request until restart.
    fn gemm<X: DevicePtr<f32>>(&self, w: &S, x: &X, out: &mut TBuf, inn: usize, outn: usize, n: usize) -> Result<()> {
        vision_gemm_drill(&self.drill)?;
        let cfg = GemmConfig::<f32> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: outn as i32, n: n as i32, k: inn as i32,
            alpha: 1.0, lda: inn as i32, ldb: inn as i32, beta: 0.0, ldc: outn as i32,
        };
        unsafe { self.blas.gemm(cfg, w, x, out).map_err(|e| anyhow::anyhow!("vision gemm f32: {e:?}")) }
    }

    fn layernorm(&self, out: &mut TBuf, x: &TBuf, w: &S, b: &S, n: usize, dim: usize) {
        let smem = (2 * 256 * 4) as u32;
        vlaunch!(self, "vision_layernorm", (n as u32, 1, 1), (256, 1, 1), smem,
            (d(out), d(x), d(w), d(b), n as i32, dim as i32, fbits(LAYERNORM_EPS)));
    }

    fn bias_add(&self, out: &mut TBuf, bias: &S, rows: usize, outn: usize) {
        let total = rows * outn;
        vlaunch!(self, "vision_bias_add", grid(total), (256, 1, 1), 0,
            (d(out), d(bias), rows as i32, outn as i32));
    }

    fn add_inplace(&self, out: &mut TBuf, src: &TBuf, n: usize) {
        vlaunch!(self, "vision_add_inplace", grid(n), (256, 1, 1), 0, (d(out), d(src), n as i32));
    }

    fn gelu_tanh(&self, out: &mut TBuf, n: usize) {
        vlaunch!(self, "vision_gelu_tanh", grid(n), (256, 1, 1), 0, (d(out), n as i32));
    }

    fn gelu(&self, out: &mut TBuf, n: usize) {
        vlaunch!(self, "vision_gelu", grid(n), (256, 1, 1), 0, (d(out), n as i32));
    }

    /// Full non-causal MHA over N tokens (one packed chunk, no KV cache). qkv [N, 3*hidden].
    /// cuBLAS-based: per-head [N, hd] row-major q/k/v (transpose+RoPE kernel), then the engine's
    /// batched GEMM pattern. S = scale*(Q·K^T), softmax over keys, O = P·V. This replaces the scalar
    /// flash kernel that was O(N²) serial and ~35 s on a 12K-patch image; cuBLAS runs the same O(N²)
    /// math on the tensor/FP32 path far faster. Confined to the vision tower (AGENTS §2 — the text
    /// prefill/decode attention path is untouched).
    fn attention(&self, qkv: &TBuf, cos: &TBuf, sin: &TBuf, out: &mut TBuf, n: usize,
                 qo: &TBuf, ko: &TBuf, vo: &TBuf, s: &mut TBuf, o: &mut TBuf) -> Result<()> {
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
            let (q_view, k_view, v_view) = (qo.view(st..st + nhd), ko.view(st..st + nhd),
                                            vo.view(st..st + nhd));
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
        // PR #16: the tower runs on spawn_blocking worker threads (VIS-2), which may not have
        // bound the device's primary context yet — cudarc requires bind_to_thread for
        // cross-thread CudaDevice use (no-op when already bound).
        self.dev.bind_to_thread()?;
        let d = self.host.dims;
        let n = gh * gw;
        assert_eq!(pixel_values.len(), n * d.wpv(), "pixel_values len");

        // Host tables (pos-embed bilinear + rotary cos/sin) — the same functions the CPU path uses.
        let pe = crate::vision_encoder::pos_embed_bilinear(
            &self.host.pos_embed_w, gh, gw, d.num_side(), d.hidden, d.merge);
        let (cos, sin) = crate::vision_encoder::vision_cos_sin(gh, gw, d.head_dim(), d.merge);

        let r = self.forward_dev(pixel_values, &pe, &cos, &sin, gh, gw, trace);
        // F1: the uploads are stream-ordered copies out of `pixel_values`, `pe`, `cos` and `sin`.
        // Drain the tower's stream before those host buffers can go away, also after an error.
        let drained = self.sync();
        let out = r?;
        drained?;
        Ok(out)
    }

    /// The device part of `forward`. Everything here stays on the tower's own stream (see the
    /// RULE at the top of this file): scratch comes from `self.pool`, host data moves through
    /// `upload` / `to_host`.
    fn forward_dev(&mut self, pixel_values: &[f32], pe: &[f32], cos: &[f32], sin: &[f32],
                   gh: usize, gw: usize, trace: bool) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let d = self.host.dims;
        let (hidden, inter, merge) = (d.hidden, d.inter, d.merge);
        let (hd, mi) = (d.head_dim(), d.merge_inter());
        let wpv = d.wpv();
        let n = gh * gw;
        let tn = n / (merge * merge);

        let pv = self.upload(pixel_values)?;
        let pe_g = self.upload(pe)?;
        let cos_g = self.upload(cos)?;
        let sin_g = self.upload(sin)?;

        let mut h = self.pool.get(n * hidden)?;
        self.gemm(&self.patch_w, &pv, &mut h, wpv, hidden, n)?;
        self.bias_add(&mut h, &self.patch_b, n, hidden);
        // pos-embed
        self.add_inplace(&mut h, &pe_g, n * hidden);

        let mut states = Vec::new();
        if trace { states.push(self.to_host(&h, n * hidden)?); }   // pre_blocks

        // cuBLAS-attention scratch (reused across blocks). Sized for the current N.
        let (nhead_s, nn_s, n_o_s) = (d.heads * n * hd, n * n, n * hd);
        let aq = self.pool.get(nhead_s)?;
        let ak = self.pool.get(nhead_s)?;
        let av = self.pool.get(nhead_s)?;
        let mut as_ = self.pool.get(nn_s)?;
        let mut ao = self.pool.get(n_o_s)?;

        for bi in 0..self.blocks.len() {
            let mut norm1 = self.pool.get(n * hidden)?;
            let mut qkv = self.pool.get(n * 3 * hidden)?;
            let mut attn = self.pool.get(n * hidden)?;
            {
                let blk = &self.blocks[bi];
                self.layernorm(&mut norm1, &h, &blk.norm1_w, &blk.norm1_b, n, hidden);
                self.gemm(&blk.qkv_w, &norm1, &mut qkv, hidden, 3 * hidden, n)?;
                self.bias_add(&mut qkv, &blk.qkv_b, n, 3 * hidden);
                self.attention(&qkv, &cos_g, &sin_g, &mut attn, n, &aq, &ak, &av, &mut as_, &mut ao)?;
            }
            // PR #16: the per-block QKV scratch goes back to the pool here — unreleased, it
            // grew the pool by 3*n*hidden floats on EVERY block of EVERY image.
            self.pool.put(qkv);
            let mut proj = self.pool.get(n * hidden)?;
            {
                let blk = &self.blocks[bi];
                self.gemm(&blk.proj_w, &attn, &mut proj, hidden, hidden, n)?;
                self.bias_add(&mut proj, &blk.proj_b, n, hidden);
            }
            self.pool.put(attn);
            self.add_inplace(&mut h, &proj, n * hidden);
            self.pool.put(proj);

            let mut norm2 = self.pool.get(n * hidden)?;
            let mut fc1 = self.pool.get(n * inter)?;
            let mut fc2 = self.pool.get(n * hidden)?;
            {
                let blk = &self.blocks[bi];
                self.layernorm(&mut norm2, &h, &blk.norm2_w, &blk.norm2_b, n, hidden);
                // S-B9-REL-VISIONFIX: these two results were DROPPED (no `?`). A cuBLAS failure in
                // an MLP GEMM left `fc1` / `fc2` holding whatever the pool handed out, and the
                // tower returned embeddings computed from it as a success: no error, no fallback.
                self.gemm(&blk.fc1_w, &norm2, &mut fc1, hidden, inter, n)?;
                self.bias_add(&mut fc1, &blk.fc1_b, n, inter);
                self.gelu_tanh(&mut fc1, n * inter);
                self.gemm(&blk.fc2_w, &fc1, &mut fc2, inter, hidden, n)?;
                self.bias_add(&mut fc2, &blk.fc2_b, n, hidden);
            }
            self.pool.put(fc1);
            self.add_inplace(&mut h, &fc2, n * hidden);
            self.pool.put(fc2);
            self.pool.put(norm1);
            self.pool.put(norm2);
            if trace { states.push(self.to_host(&h, n * hidden)?); }
        }
        self.pool.put(aq); self.pool.put(ak); self.pool.put(av);
        self.pool.put(as_); self.pool.put(ao);

        // merger: layernorm -> view [N,hidden] as [tn,mi] -> fc1 -> gelu -> fc2
        let mut ln = self.pool.get(n * hidden)?;
        self.layernorm(&mut ln, &h, &self.merger_norm_w, &self.merger_norm_b, n, hidden);
        self.pool.put(h);
        let mut mfc1 = self.pool.get(tn * mi)?;
        self.gemm(&self.merger_fc1_w, &ln, &mut mfc1, mi, mi, tn)?;
        self.pool.put(ln);
        self.bias_add(&mut mfc1, &self.merger_fc1_b, tn, mi);
        self.gelu(&mut mfc1, tn * mi);
        let mut out = self.pool.get(tn * d.out_hidden)?;
        self.gemm(&self.merger_fc2_w, &mfc1, &mut out, mi, d.out_hidden, tn)?;
        self.pool.put(mfc1);
        self.bias_add(&mut out, &self.merger_fc2_b, tn, d.out_hidden);

        let merged = self.to_host(&out, tn * d.out_hidden)?;
        self.pool.put(out);
        // the uploads' buffers go back too (they used to be dropped: a legacy-stream free per image)
        self.pool.put(pv); self.pool.put(pe_g); self.pool.put(cos_g); self.pool.put(sin_g);
        Ok((merged, states))
    }

    /// Host -> device as a stream-ordered copy on the tower's own stream, into a pool buffer. The
    /// copy reads `host` when the stream reaches it: `host` must outlive the next `sync()`
    /// (`forward` drains the stream before its host tables are dropped).
    fn upload(&mut self, host: &[f32]) -> Result<TBuf> {
        let buf = self.pool.get(host.len())?;
        unsafe { result::memcpy_htod_async(buf.ptr, host, self.stream.stream) }
            .map_err(|e| anyhow::anyhow!("vision tower upload on its own stream: {e:?}"))?;
        Ok(buf)
    }

    /// Device -> host: the first `n` f32 of `buf`, as a stream-ordered copy on the tower's own
    /// stream (after every kernel queued before it), then a drain of that stream.
    fn to_host(&self, buf: &TBuf, n: usize) -> Result<Vec<f32>> {
        assert!(n <= buf.len, "tower download of {n} f32 from a {}-f32 buffer", buf.len);
        let mut v = vec![0f32; n];
        unsafe { result::memcpy_dtoh_async(&mut v, buf.ptr, self.stream.stream) }
            .map_err(|e| anyhow::anyhow!("vision tower download on its own stream: {e:?}"))?;
        self.sync()?;
        Ok(v)
    }

    /// PR #16: sync the tower's OWN (blocking) stream, not the whole device — the serving
    /// compute stream keeps flowing while a vision copy drains.
    fn sync(&self) -> Result<()> {
        unsafe { result::stream::synchronize(self.stream.stream) }
            .map_err(|e| anyhow::anyhow!("stream sync: {e:?}"))
    }

    /// (scratch bytes the tower holds between requests, fresh allocations it has made): the second
    /// is flat once every bucket a workload needs exists (the PR #16 pool-release check).
    pub fn pool_stats(&self) -> (usize, usize) {
        (self.pool.cached_bytes(), self.pool.allocs)
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

impl Drop for GpuVisualTower {
    fn drop(&mut self) {
        // drain, then free the scratch on its own stream before the stream itself is destroyed
        let _ = self.dev.bind_to_thread();
        let _ = self.sync();
        self.pool.free.clear();
        let _ = self.sync();
    }
}

/// PR #16 test drill (--vision-fail-gemm): consumes the one-shot flag and returns the error a
/// REAL cuBLAS failure produces, so the server's GPU->CPU fallback can be exercised on hardware
/// without provoking an actual CUDA fault. Pure function — unit-tested below; the flag is
/// consumed on the first GEMM so the very next request uses the GPU tower again.
pub(crate) fn vision_gemm_drill(armed: &std::cell::Cell<bool>) -> Result<()> {
    if armed.get() {
        armed.set(false);
        Err(anyhow::anyhow!("vision gemm f32: injected cuBLAS failure (--vision-fail-gemm drill)"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod drill_tests {
    use super::*;

    /// S-B9-REL-VISIONFIX F1 + VISIONFIX2 V2c, the RULE at the top of this file as a check. The whole
    /// non-test source of this file is the request path EXCEPT two spans, and every forbidden name is
    /// checked over everything else (the buffer types, the pool, the kernel-launch macro, `gemm`,
    /// `layernorm`, `attention`, `forward`, `forward_dev`, `upload`, `to_host`, `sync`, ...):
    ///   * the constructor `new_with_drill` (boot, on the boot thread, before any request or any
    ///     scheduler capture exists): it must contain exactly the one legacy-stream weight upload
    ///     helper, `htod_sync_copy`, and the cuBLAS-handle / module set-up;
    ///   * `impl Drop for GpuVisualTower`: exempt because it only ever runs when the tower is dropped,
    ///     which never happens while serving (the server holds the tower for the life of the process;
    ///     see the INVARIANT at the top of the file). It is checked to use only own-stream operations
    ///     itself (`sync()` and clearing the pool's `TBuf`s); the fields' own destructors (the
    ///     weights, the cuBLAS handle, the stream) are the part the invariant covers.
    /// (The behaviour itself is tested on a device in tests/vision_capture_race_test.rs.)
    #[test]
    fn the_request_path_names_no_legacy_stream_helper() {
        let src = src_text!("vision_gpu");
        let forbidden: Vec<String> = [("htod_sync", "_copy"), ("dtoh_sync", "_copy"), ("alloc_", "zeros"), ("memset_", "zeros"),
                                      ("dev.synchronize", "()"), ("htod_", "copy("), ("dtoh_", "copy("), ("dev.alloc", "("), ("memcpy_htod_", "sync"),
                                      ("memcpy_dtoh_", "sync"), ("malloc_", "sync"), ("free_", "sync"),
                                      ("CudaBlas", "::new"), ("load_", "ptx("), ("fork_blocking", "_stream"),
                                      ("cublas", "Destroy"), ("cublasLt", "Destroy"), ("destroy_", "handle"), ("memcpy_dtod_", "sync"),
                                      ("dev.alloc", "::<"), ("htod_copy", "_into("), ("dtod_", "copy"), ("memset_d8_", "sync"),
                                      ("ctx::", "synchronize"), (".synchron", "ize()")]
            .iter().map(|(a, b)| [*a, *b].concat()).collect();
        let code_of = |text: &str| -> String {
            // (comments and `use` lines are not calls)
            text.lines().filter(|l| { let t = l.trim_start(); !t.starts_with("//") && !t.starts_with("use ") }).collect::<Vec<_>>().join("\n")
        };
        // everything above the test module
        let end = src.find("#[cfg(test)]\nmod drill_tests").expect("the test module");
        let non_test = &src[..end];
        // span 1: the constructor, from `pub fn new_with_drill` to the doc comment of `gemm`
        let c0 = non_test.find("    pub fn new_with_drill(").expect("new_with_drill");
        let c1 = c0 + non_test[c0..].find("    /// Batched FP32 GEMM:").expect("end of the constructor");
        // span 2: the Drop impl, up to the drill function that follows it
        let d0 = non_test.find("impl Drop for GpuVisualTower").expect("Drop impl");
        let d1 = d0 + non_test[d0..].find("/// PR #16 test drill (--vision-fail-gemm)").expect("end of the Drop impl");
        assert!(c0 < c1 && c1 < d0 && d0 < d1, "span order");
        let request_path = code_of(&format!("{}{}{}", &non_test[..c0], &non_test[c1..d0], &non_test[d1..]));
        // the span check is not vacuous: the request path is the bulk of the file and contains the forward
        for must in ["pub fn forward(", "fn forward_dev(", "fn upload(", "fn to_host(", "fn attention(", "fn gemm<", "struct TBuf", "struct TPool", "macro_rules! vlaunch"] {
            assert!(request_path.contains(must), "the checked span must contain `{must}`");
        }
        for f in &forbidden {
            assert!(!request_path.contains(f.as_str()), "the request path names `{f}`: a legacy-stream or handle call on a worker thread (see the RULE at the top of the file)");
        }
        // the shared legacy-stream pool (`crate::gpu::Pool`, `Pool::get` unwraps its NULL-stream calls) must not come back:
        // any `Pool` identifier that is not `TPool`
        for (at, _) in request_path.match_indices("Pool") {
            assert!(request_path[..at].ends_with('T'), "the request path names the shared `Pool` (offset {at}): use the tower's own `TPool`");
        }
        // the constructor is the one place the legacy-stream upload stays: exactly once, and the exempt span holds the
        // constructor ONLY (a method added between it and `gemm` would otherwise be exempt by position)
        let ctor = code_of(&non_test[c0..c1]);
        assert_eq!(ctor.matches(forbidden[0].as_str()).count(), 1, "the weight upload at construction");
        assert_eq!(ctor.matches("fn ").count(), 1, "the exempt constructor span must hold exactly one fn");
        // nothing but the test module may follow its start (non-test code appended after it would be unscanned)
        for l in src[end..].lines() {
            for top in ["fn ", "pub fn ", "impl ", "struct ", "pub struct ", "static ", "const ", "macro_rules!", "pub(crate) fn "] {
                assert!(!l.starts_with(top), "top-level item after the test module is not scanned: {l}");
            }
        }
        // Drop: only own-stream work in its own body
        let drop_body = code_of(&non_test[d0..d1]);
        for f in &forbidden {
            assert!(!drop_body.contains(f.as_str()), "the Drop body names `{f}`");
        }
        assert!(drop_body.contains("self.sync()") && drop_body.contains("self.pool.free.clear()"), "Drop drains and clears on the tower's own stream");
        // and the request path does use the tower's own stream for its copies and its drain
        for must in ["result::memcpy_htod_async(buf.ptr, host, self.stream.stream)", "result::memcpy_dtoh_async(&mut v, buf.ptr, self.stream.stream)",
                     "result::stream::synchronize(self.stream.stream)"] {
            assert!(request_path.contains(must), "the request path must contain `{must}`");
        }
    }

    /// V4d: the pool's out-of-memory answer is "drop the cache and try once more" and nothing else:
    /// one retry, a second failure is an error (never a panic), no relief on success.
    #[test]
    fn pool_out_of_memory_retries_once_after_relief() {
        use std::cell::Cell;
        let (tries, relieved) = (Cell::new(0), Cell::new(0));
        let ok = alloc_retry(|| { tries.set(tries.get() + 1); if tries.get() == 1 { Err(anyhow::anyhow!("oom")) } else { Ok(7) } },
                             |_| relieved.set(relieved.get() + 1));
        assert_eq!((ok.unwrap(), tries.get(), relieved.get()), (7, 2, 1), "first attempt fails, relief, second succeeds");
        let (tries, relieved) = (Cell::new(0), Cell::new(0));
        let err: Result<i32> = alloc_retry(|| { tries.set(tries.get() + 1); Err(anyhow::anyhow!("oom")) }, |_| relieved.set(relieved.get() + 1));
        assert!(err.is_err() && tries.get() == 2 && relieved.get() == 1, "a second failure is an Err after exactly one retry");
        let (tries, relieved) = (Cell::new(0), Cell::new(0));
        let ok = alloc_retry(|| { tries.set(tries.get() + 1); Ok(1) }, |_| relieved.set(relieved.get() + 1));
        assert!(ok.is_ok() && tries.get() == 1 && relieved.get() == 0, "no relief when the first attempt succeeds");
    }

    /// The scratch pool hands a buffer back for the same bucket, counts fresh allocations, and its
    /// bucket rule is the shared pool's (power of two, at least 256): pure bookkeeping, no device.
    #[test]
    fn tower_pool_buckets() {
        assert_eq!(TPool::bucket(1), 256);
        assert_eq!(TPool::bucket(256), 256);
        assert_eq!(TPool::bucket(257), 512);
        assert_eq!(TPool::bucket(784 * 1152), 1 << 20);
        assert_eq!(TPool::bucket(784 * 784), 1 << 20);
        assert_eq!(TPool::bucket(784 * 4304), 1 << 22);
    }

    #[test]
    fn drill_fires_exactly_once() {
        let armed = std::cell::Cell::new(true);
        let first = vision_gemm_drill(&armed);
        assert!(first.is_err(), "armed drill must fail the first gemm");
        assert!(first.unwrap_err().to_string().contains("injected cuBLAS failure"));
        // consumed: the next request goes back to the GPU path
        assert!(vision_gemm_drill(&armed).is_ok());
        assert!(vision_gemm_drill(&armed).is_ok());
        let disarmed = std::cell::Cell::new(false);
        assert!(vision_gemm_drill(&disarmed).is_ok());
    }
}
