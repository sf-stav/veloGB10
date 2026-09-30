//! DFlashDrafter — the AngelSlim Hy3-DFlash-B8 draft model (E29-B1 of the Hy3 speed plan).
//!
//! A 5-layer qwen3-style transformer CONDITIONED on the target model's hidden states — it is NOT a
//! standalone causal LM. It drafts an 8-token block with NON-causal attention over the concatenated
//! (context keys + block keys).
//!
//! Reference semantics (z-lab/dflash `dflash/model.py`, ported exactly):
//!   1. `target_hidden` = concat of the TARGET's post-layer hiddens at layers {1,20,39,58,77} along
//!      the last dim → [B, L, 20480]. Then `hidden_norm(fc(target_hidden))` (fc 20480→4096, no bias,
//!      RMSNorm) — computed once, shared by every layer.
//!   2. `hidden_states` = the block's token embeddings from the target's `embed_tokens`.
//!   3. Per layer (5): input_layernorm → q = q_proj(hidden) [B,8,8192] → view [B,8,64,128] →
//!      q_norm (per-head RMSNorm) → transpose(1,2). k_ctx = k_proj(target_hidden) [B,L,1024],
//!      k_noise = k_proj(hidden) [B,8,1024]; k = cat([k_ctx,k_noise],dim=1) → view [B,L+8,8,128] →
//!      k_norm → transpose. v likewise (NO v_norm). Rotary (theta 11158840, head_dim 128): q uses
//!      cos[..., -8:, :] (the block's positions), k uses the FULL position range (ctx positions
//!      0..L-1, block positions pos_start..pos_start+7).
//!   4. Attention over the concatenated k/v, NON-causal (is_causal=False, no mask — each block
//!      position attends to ALL context + ALL block positions).
//!   5. o_proj → residual → post_attention_layernorm → swiglu MLP → residual.
//!   6. Final RMSNorm → the LM head (the checkpoint has NO lm_head; the probe feeds the target's
//!      `embed_tokens` as a stand-in — the real loop passes the target's actual head).
//!
//! Engine mapping (the "engine port may keep its own KV layout" allowance from the task): the ctx
//! k/v and block k/v are written into per-layer bf16 KV caches at cache rows 0..L-1 and L..L+7
//! (rank space), and the decode-path attention (`gqa_attn_splitk` + `gqa_attn_reduce`) is run with
//! every block query's position pinned to L+7 — so every query attends to all L+8 keys, exactly the
//! reference's maskless full attention. The ROPE positions are decoupled from the cache rows (the
//! tree-verify convention): ctx keys rotate at positions 0..L-1, block q/k at pos_start..pos_start+7.
//!
//! All activations are col-major [dim, batch] bf16 (the engine convention); weights are row-major
//! [out, in] bf16 (the `gemm_act` convention).
//!
//! Invariants honored (AGENTS.md §2): blocking compute stream; fresh pool buffers are zeroed by
//! `Pool::get` (never rely on `alloc_zeros`); D2D/htod on the compute stream; the KV caches are
//! only ever READ within [0, L+8) and every one of those rows is written before attention runs.
//!
//! Norm weights: the checkpoint's RMSNorm is the PLAIN T5-style `weight * x` (Hy3 family), while
//! the engine's rmsnorm kernels hard-code qwen3_5's zero-centered `(1 + weight) * x`. Store the
//! norm weights as (w - 1) at upload, exactly like the hy_v3 loader (src/gpu.rs:2410) — the
//! kernel then computes `(1 + (w-1)) * x == w * x` losslessly in fp32.

use anyhow::{anyhow, Context, Result};
use cudarc::cublas::{sys::cublasOperation_t as OP, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::bf16;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::gpu::Pool;

/// Compile-time CEILINGS for buffer sizing only. Every live value comes from the artifact's
/// `config.json` (P14: the 35B drafter has 6 layers / 8 taps / block 16; the Hy3 artifact has
/// 5 / 5 / 8). Buffers are sized to the ceiling so a wider artifact never resizes mid-flight;
/// the runtime geometry lives in [`DflashCfg`] / [`DflashDrafter`] and is never assumed.
pub const MAX_BLOCK: usize = 16;
pub const MAX_TAPS: usize = 8;

/// The DFlash v1 draft artifact's geometry, read from `config.json` ONLY (no tensor access).
///
/// Every field is REQUIRED — a missing key is a loud error, never a silent default, because a
/// wrong `h`/`nctx`/`hd` produces plausible-looking garbage instead of a crash (the E29 lesson:
/// a config default that "works" is indistinguishable from a config default that is wrong).
/// Accepted schema variants: `rope_theta` at the top level (Hy3/older exports) or nested under
/// `rope_parameters.rope_theta` (Qwen3.6 exports, which is what the 35B artifact uses — reading
/// only the top-level key silently fell back to the wrong theta before P14).
#[derive(Clone, Debug)]
pub struct DflashCfg {
    pub h: usize,
    pub n_layers: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// `dflash_config.block_size` — the artifact's trained draft block.
    pub block: usize,
    /// `dflash_config.mask_token_id` — the block's no-information filler token.
    pub mask_token_id: u32,
    /// `dflash_config.target_layer_ids` — the TARGET layers whose post-layer hidden states are
    /// concatenated into the conditioning feature (order = feature layout).
    pub tap_layers: Vec<usize>,
    /// Per-layer attention type (`config.layer_types`): `true` = sliding/causal layer, `false` =
    /// full non-causal layer. Empty = no `layer_types` (every layer full non-causal).
    pub sliding: Vec<bool>,
    /// `config.sliding_window` (0 = none).
    pub sliding_window: usize,
}

impl DflashCfg {
    /// Read + validate the artifact geometry. `cfg_dir` is the drafter directory itself or the
    /// TARGET directory (which carries `<name>/config.json`? no — the drafter dir is explicit).
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let txt = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("DFlash: read {}", cfg_path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&txt)
            .with_context(|| format!("DFlash: parse {}", cfg_path.display()))?;
        let d = &v["dflash_config"];
        let req = |loc: &serde_json::Value, k: &str| -> Result<usize> {
            loc.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| {
                anyhow!("DFlash config {}: required key `{k}` missing or not a positive integer",
                        cfg_path.display())
            })
        };
        let h = req(&v, "hidden_size")?;
        let n_layers = req(&v, "num_hidden_layers")?;
        let nh = req(&v, "num_attention_heads")?;
        let nkv = req(&v, "num_key_value_heads")?;
        let hd = req(&v, "head_dim")?;
        let inter = req(&v, "intermediate_size")?;
        let vocab = req(&v, "vocab_size")?;
        let rms_eps = v.get("rms_norm_eps").and_then(|x| x.as_f64())
            .ok_or_else(|| anyhow!("DFlash config: required key `rms_norm_eps` missing"))? as f32;
        let rope_theta = v.get("rope_parameters").and_then(|r| r.get("rope_theta"))
            .or_else(|| v.get("rope_theta"))
            .and_then(|x| x.as_f64())
            .ok_or_else(|| anyhow!(
                "DFlash config {}: no rope theta — expected `rope_theta` or `rope_parameters.rope_theta`",
                cfg_path.display()))? as f32;
        anyhow::ensure!(d.is_object(), "DFlash config {}: `dflash_config` object missing", cfg_path.display());
        let block = req(d, "block_size")?;
        let mask_token_id = req(d, "mask_token_id")? as u32;
        let tap_layers: Vec<usize> = d.get("target_layer_ids")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow!("DFlash config: `dflash_config.target_layer_ids` missing"))?
            .iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect();
        anyhow::ensure!(!tap_layers.is_empty(), "DFlash config: target_layer_ids empty");
        anyhow::ensure!(tap_layers.len() <= MAX_TAPS,
            "DFlash config: {} tap layers exceeds the compiled ceiling MAX_TAPS={MAX_TAPS}", tap_layers.len());
        anyhow::ensure!(block >= 1 && block <= MAX_BLOCK,
            "DFlash config: block_size {block} outside 1..={MAX_BLOCK} (MAX_VERIFY is {MAX_BLOCK})");
        anyhow::ensure!(hd % 32 == 0 && hd <= 512,
            "DFlash config: head_dim {hd} outside the attention kernels' envelope (32 | hd, hd <= 512)");
        let sliding: Vec<bool> = v.get("layer_types").and_then(|x| x.as_array())
            .map(|a| a.iter().map(|t| t.as_str() == Some("sliding_attention")).collect())
            .unwrap_or_default();
        let sliding_window = v.get("sliding_window").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        Ok(DflashCfg { h, n_layers, nh, nkv, hd, inter, vocab, rms_eps, rope_theta,
                       block, mask_token_id, tap_layers, sliding, sliding_window })
    }

    /// The number of conditioning columns (one per tap layer).
    pub fn nctx(&self) -> usize { self.tap_layers.len() }
}

/// Rank-0 staging buffer for the DFlash tap capture. Each target forward (a token-by-token
/// prefill step or an N-row verify block) copies the post-FFN-add residuals of this artifact's
/// tap layers into columns [0, n) of `scratch` ([nctx*h, MAX_BLOCK] bf16, layer-major within a
/// column) — device-side only, on the compute stream, NO dtoh in the loop. The draft loop then
/// D2Ds the columns it needs (the accepted span) into the persistent ctx feature buffer.
///
/// `layers` is the artifact's `dflash_config.target_layer_ids` and `nctx` its length: the trunk's
/// layer loops ask THIS sink which target layers to capture, so a second artifact geometry is a
/// config change, never a recompile (P14; the 35B has 8 taps, the Hy3 artifact 5).
pub struct DflashTapSink {
    pub scratch: CudaSlice<bf16>,
    pub nctx: usize,
    pub layers: Vec<usize>,
    /// OPTIONAL wide target (P14 serving lane): a `[nctx*h, wide_stride]` ctx-feature buffer
    /// addressed by ABSOLUTE position. When armed, every capture lands at the column block
    /// `[wide_pos, wide_pos+n)` instead of the ring staging — so the prefill fills the whole ctx
    /// feature in one pass (no token-by-token prefill) and each verify rewrites the span it just
    /// verified. `None` keeps the E29 probe behavior byte-for-byte.
    pub wide: Option<CudaSlice<bf16>>,
    pub wide_stride: usize,
    /// Absolute position of the next forward's first captured column (device-loop visible).
    pub wide_pos: std::sync::atomic::AtomicUsize,
}

impl DflashTapSink {
    pub fn new(dev: &Arc<CudaDevice>, h: usize, layers: Vec<usize>) -> Self {
        let nctx = layers.len();
        assert!(nctx > 0 && nctx <= MAX_TAPS, "DFlash tap sink: {nctx} taps > MAX_TAPS {MAX_TAPS}");
        DflashTapSink {
            scratch: dev.alloc_zeros::<bf16>(MAX_TAPS * h * MAX_BLOCK).unwrap(),
            nctx,
            layers,
            wide: None,
            wide_stride: 0,
            wide_pos: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    /// Arm the wide ctx-feature target (see the struct docs). `stride` = the number of positions.
    pub fn arm_wide(&mut self, buf: CudaSlice<bf16>, stride: usize) {
        self.wide = Some(buf);
        self.wide_stride = stride;
    }
    #[inline]
    pub fn set_wide_pos(&self, p: usize) {
        self.wide_pos.store(p, std::sync::atomic::Ordering::Relaxed);
    }
    #[inline]
    pub fn wide_ptr(&self) -> u64 { *self.wide.as_ref().expect("dflash wide sink").device_ptr() }
    /// The staging index of target layer `li`, if it is a tap layer.
    #[inline]
    pub fn tap_index(&self, li: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l == li)
    }
}

/// One tap-layer copy: `residual` [h, n] bf16 col-major (the post-FFN-add residual of target
/// layer `TAP_LAYERS[tap_li]`) → `sink.scratch` columns [0, n) at the layer-major offset
/// `tap_li`. Stream-ordered 2D D2D on the compute stream (invariant 3).
/// One tap-layer copy into an arbitrary `[nctx*h, stride]` feature buffer: target columns
/// `[dst_y, dst_y+n)`, layer-row `tap_li`. Both the ring staging (dst_y = 0) and the wide
/// per-position ctx feature (dst_y = the absolute position) are this same call.
pub fn tap_capture_to(dev: &Arc<CudaDevice>, stream: cudarc::driver::sys::CUstream,
                      dst: u64, dst_pitch_elems: usize, dst_y: usize,
                      res_ptr: u64, h: usize, n: usize, tap_li: usize) {
    use cudarc::driver::sys;
    let cp = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0, srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(), srcDevice: res_ptr,
        srcArray: std::ptr::null_mut(), srcPitch: h * 2,
        dstXInBytes: tap_li * h * 2, dstY: dst_y,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(), dstDevice: dst,
        dstArray: std::ptr::null_mut(), dstPitch: dst_pitch_elems * 2,
        WidthInBytes: h * 2, Height: n,
    };
    unsafe {
        let r = sys::cuMemcpy2DAsync_v2(&cp, stream);
        assert!(r == sys::CUresult::CUDA_SUCCESS, "dflash tap capture D2D failed: {r:?}");
    }
}

/// One tap-layer copy into the ring staging (the probe path; see [`DflashTapSink`]).
pub fn tap_capture(dev: &Arc<CudaDevice>, stream: cudarc::driver::sys::CUstream,
                   sink: &DflashTapSink, res_ptr: u64, h: usize, n: usize, tap_li: usize) {
    // HARD asserts, not debug_asserts: a release build (every deployed binary) compiles the debug
    // form out, and an `n` past the ring's 16 columns would write past `scratch` into whatever the
    // allocator placed next — the silent-corruption class this repo has already paid for twice
    // (`gpu.rs`'s pool tripwire records the same lesson). Two integer compares per tapped layer.
    assert!(n <= MAX_BLOCK, "tap capture batch {n} exceeds MAX_BLOCK {MAX_BLOCK}");
    assert!(tap_li < sink.nctx, "tap capture layer {tap_li} >= nctx {}", sink.nctx);
    tap_capture_to(dev, stream, *sink.scratch.device_ptr() as u64, sink.nctx * h, 0,
                   res_ptr, h, n, tap_li);
}

/// One-shot loud warning for a wide-capture write that would fall outside the ctx surface.
///
/// The wide branch of the tap capture guards on `p0 + n <= wide_stride`; when the guard fails the
/// capture is SKIPPED and the ctx feature keeps whatever was there (stale, or an un-zeroed
/// allocation). That is a silent-wrong-answer failure mode — the exact class the P14 lane was
/// caught by — so the skip is reported once per process instead of never. (It is not a panic: the
/// failure is on a live serving path, and the caller-side prime asserts the frontier the lane
/// actually consumes.)
pub fn warn_wide_skip(p0: usize, n: usize, stride: usize) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!("[dflash] WARNING: wide tap capture SKIPPED — window [{p0}, {}) exceeds the ctx \
                   surface stride {stride}; the drafter would condition on stale ctx columns. \
                   (Raise --max-seq-len or shrink the prefill window.)",
                  p0 + n);
    });
}

/// One DFlash transformer layer's weights (all from `model.safetensors`, BF16).
pub struct DflashLayer {
    pub input_ln: CudaSlice<f32>,      // [h]
    pub post_ln: CudaSlice<f32>,       // [h]
    pub q_norm: CudaSlice<f32>,        // [hd]
    pub k_norm: CudaSlice<f32>,        // [hd]
    pub q_proj: CudaSlice<bf16>,       // [nh*hd, h]
    pub k_proj: CudaSlice<bf16>,       // [nkv*hd, h]
    pub v_proj: CudaSlice<bf16>,       // [nkv*hd, h]
    pub o_proj: CudaSlice<bf16>,       // [h, nh*hd]
    pub gate_proj: CudaSlice<bf16>,    // [inter, h]
    pub up_proj: CudaSlice<bf16>,      // [inter, h]
    pub down_proj: CudaSlice<bf16>,    // [h, inter]
}

/// Per-layer context+block KV caches in RANK space: ctx at rows 0..L-1, block at rows L..L+7.
/// `stride` is the allocated row count per (head, layer) plane; forward() requires stride >= L+8.
pub struct DflashKv {
    pub k_cache: Vec<CudaSlice<bf16>>, // [layer] [nkv * stride * hd]
    pub v_cache: Vec<CudaSlice<bf16>>,
    pub stride: usize,
}

impl DflashKv {
    pub fn new(d: &DflashDrafter, stride: usize) -> Self {
        let n = d.nkv * stride * d.hd;
        let mut k_cache = Vec::with_capacity(d.layers.len());
        let mut v_cache = Vec::with_capacity(d.layers.len());
        for _ in 0..d.layers.len() {
            k_cache.push(d.dev.alloc_zeros::<bf16>(n).unwrap());
            v_cache.push(d.dev.alloc_zeros::<bf16>(n).unwrap());
        }
        DflashKv { k_cache, v_cache, stride }
    }
}

/// The DFlash drafter. Owns its device runtime (blocking compute stream + cuBLAS), the checkpoint
/// weights, the rope tables, and the tiny persistent per-forward arrays. Activations are pooled.
pub struct DflashDrafter {
    pub dev: Arc<CudaDevice>,
    stream: cudarc::driver::CudaStream,
    blas: CudaBlas,
    bk: HashMap<String, CudaFunction>,
    pub layers: Vec<DflashLayer>,
    /// The block's embedding table [vocab, h] — doubles as the LM-head stand-in (no lm_head in
    /// the checkpoint). The real loop passes the target's head instead.
    pub embed: CudaSlice<bf16>,
    fc: CudaSlice<bf16>,        // [h, 20480]
    hidden_norm: CudaSlice<f32>,// [h] (w-1 convention)
    norm: CudaSlice<f32>,       // [h] (w-1 convention)
    cos_table: CudaSlice<f32>,  // [cos_max, rdim]
    sin_table: CudaSlice<f32>,
    cos_max: usize,
    // Persistent per-forward device arrays (sized BLOCK).
    toks_dev: CudaSlice<i32>,    // block token ids for embed_gather_b
    write_pos: CudaSlice<i32>,   // KV cache rows for the block (L..L+7)
    slot_ids: CudaSlice<i32>,    // all 0 (rank-space cache base)
    // Geometry.
    pub h: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rdim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// The block's slot/mask token id (`dflash_config.mask_token_id`).
    pub mask_token_id: u32,
    /// The trained draft block length (`dflash_config.block_size`).
    pub block: usize,
    /// Number of conditioning columns (`dflash_config.target_layer_ids.len()`).
    pub nctx: usize,
    /// The target layers whose hiddens feed the feature (order = feature layout).
    pub tap_layers: Vec<usize>,
    /// Per-layer attention type (empty = all layers full non-causal, the pre-P14 behavior).
    pub sliding: Vec<bool>,
    pub sliding_window: usize,
    /// The final normalized hidden of the LAST [`forward_step`] ([h, MAX_BLOCK] bf16). The serving
    /// lane runs the target's own LM head on this; the probe reads it for its logits.
    pub last_hidden: CudaSlice<bf16>,
}

/// Pack `(ctx_len, causal, window)` for `dflash_attn_b` (the launch helper is capped at 12 args).
#[inline]
pub fn dflash_mask_pack(ctx_len: usize, causal: bool, window: usize) -> u64 {
    (ctx_len as u64) | ((causal as u64) << 32) | ((window as u64) << 33)
}

fn d<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() }
fn grid(n: usize) -> (u32, u32, u32) { (((n + 255) / 256) as u32, 1, 1) }
fn fbits(x: f32) -> u64 { x.to_bits() as u64 }

/// Launch a gpu_batch.ptx kernel by name. Mirrors gpu.rs's `blaunch!`.
macro_rules! dlaunch {
    ($s:expr, $name:expr, $g:expr, $b:expr, $smem:expr, ($($a:expr),+ $(,)?)) => {
        unsafe {
            let (g0, g1, g2) = $g;
            let (b0, b1, b2) = $b;
            let name: &str = $name;
            $s.bk.get(name).cloned().unwrap_or_else(|| panic!("dflash kernel {}", name)).launch_on_stream(
                &$s.stream,
                LaunchConfig { grid_dim: (g0, g1, g2), block_dim: (b0, b1, b2), shared_mem_bytes: $smem },
                ($($a),+)
            ).unwrap_or_else(|e| panic!("dflash launch {}: {:?}", name, e));
        }
    };
}

/// Create the engine's compute stream as a BLOCKING stream (AGENTS.md §2 invariant; see the
/// `fork_blocking_stream` note in src/gpu.rs for the cross-stream race this prevents).
fn fork_blocking_stream(dev: &Arc<CudaDevice>) -> cudarc::driver::CudaStream {
    use cudarc::driver::result::stream::{create, destroy, StreamKind};
    let mut s = dev.fork_default_stream().expect("fork stream");
    unsafe {
        destroy(s.stream).expect("destroy nonblocking stream");
        s.stream = create(StreamKind::Default).expect("create blocking stream");
    }
    s
}

fn bf16_slice(data: &[u8]) -> &[bf16] {
    bytemuck::cast_slice(data)
}

impl DflashDrafter {
    /// Load the DFlash drafter from a model directory (config.json + model.safetensors).
    /// `max_pos` sizes the rope tables (must cover the largest block position the probe feeds).
    pub fn load_from_dir(dir: &Path, max_pos: usize) -> Result<Self> {
        let cfg = DflashCfg::load(dir)?;
        let (h, n_layers, nh, nkv, hd, inter, vocab) =
            (cfg.h, cfg.n_layers, cfg.nh, cfg.nkv, cfg.hd, cfg.inter, cfg.vocab);
        let (rms_eps, rope_theta, mask_token_id, block) =
            (cfg.rms_eps, cfg.rope_theta, cfg.mask_token_id, cfg.block);
        let nctx = cfg.nctx();
        let rdim = hd;
        assert!(hd % 32 == 0 && hd <= 512, "head_dim {hd} outside the attention kernels' envelope");

        let sf_path = dir.join("model.safetensors");
        let raw = std::fs::read(&sf_path).with_context(|| format!("read {}", sf_path.display()))?;
        let st = SafeTensors::deserialize(&raw).context("deserialize model.safetensors")?;

        let dev = CudaDevice::new(0)?;
        let stream = fork_blocking_stream(&dev);
        let blas = CudaBlas::new(dev.clone())?;
        unsafe { blas.set_stream(Some(&stream))?; }

        // Load the batch kernels this module uses (gpu_batch.ptx, verified against this binary).
        let bptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx")?);
        let bfnames = ["write_kv_prefill", "write_kv_b", "add_residual_b", "silu_mul_b",
            "embed_gather_b", "gemm_binv_b", "gemm_binv_f32_b", "kernel_build_id"];
        dev.load_ptx(bptx, "gpu_batch", &bfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_batch")?;
        let mut bk = HashMap::new();
        for n in bfnames {
            bk.insert(n.to_string(), dev.get_func("gpu_batch", n)
                .with_context(|| format!("gpu_batch.{n} not in ptx"))?);
        }
        // The DFlash-specific kernels (src/ptx/gpu_dflash.ptx): reference-exact bf16 rounding.
        let dptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_dflash.ptx")?);
        // `load_ptx` needs 'static names, and a name that is missing from the PTX is a LOUD load
        // failure (never a silent fallback). The tiled list MUST match the
        // `DFLASH_TILED_KERNEL(...)` instantiations in kernels/gpu_dflash.cu; the gate in
        // `forward_step` only dispatches to combinations listed here.
        let mut dfnames: Vec<&'static str> = vec!["dflash_rmsnorm_b", "dflash_rope_b",
                                                  "dflash_attn_b", "kernel_build_id"];
        dfnames.extend([
            "dflash_attn_tiled_d1_g1", "dflash_attn_tiled_d1_g2", "dflash_attn_tiled_d1_g4",
            "dflash_attn_tiled_d2_g1", "dflash_attn_tiled_d2_g2", "dflash_attn_tiled_d2_g4",
            "dflash_attn_tiled_d2_g8", "dflash_attn_tiled_d2_g16",
            "dflash_attn_tiled_d4_g1", "dflash_attn_tiled_d4_g2", "dflash_attn_tiled_d4_g4",
            "dflash_attn_tiled_d4_g8", "dflash_attn_tiled_d4_g16",
            "dflash_attn_tiled_d8_g1", "dflash_attn_tiled_d8_g2", "dflash_attn_tiled_d8_g4",
            "dflash_attn_tiled_d8_g8",
        ]);
        dev.load_ptx(dptx, "gpu_dflash", &dfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_dflash")?;
        for n in &dfnames {
            bk.insert(n.to_string(), dev.get_func("gpu_dflash", n)
                .with_context(|| format!("gpu_dflash.{n} not in ptx"))?);
        }

        let tensor = |name: &str| -> Result<CudaSlice<bf16>> {
            let view = st.tensor(name).with_context(|| format!("missing tensor {name}"))?;
            assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{name} not BF16");
            let data = bf16_slice(view.data()).to_vec();
            Ok(dev.htod_sync_copy(&data).with_context(|| format!("upload {name}"))?)
        };
        let norm_f32 = |name: &str, n: usize| -> Result<CudaSlice<f32>> {
            let view = st.tensor(name).with_context(|| format!("missing tensor {name}"))?;
            assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{name} not BF16");
            let data = bf16_slice(view.data());
            assert_eq!(data.len(), n, "{name} shape");
            // RAW bf16 weight values (f32-stored): the dflash_rmsnorm_b kernel reproduces the
            // reference's `weight * x` exactly (no (1+w) transform — that is a qwen3_5 serving
            // convention that does not apply here).
            let fv: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
            Ok(dev.htod_sync_copy(&fv).with_context(|| format!("upload {name}"))?)
        };

        // The 35B artifact has NO `embed_tokens.weight` (the reference drafter consumes the
        // TARGET's embedding and head); the Hy3 artifact carries one. Absent = a 1-element
        // placeholder that must NEVER be used: the caller has to pass the target's embed, and
        // `forward_step` refuses loudly when neither the checkpoint nor the caller supplies one
        // (a silently wrong gather is exactly the mxfp4-permuted-embed class of bug, AGENTS §3).
        let embed = if st.tensor("embed_tokens.weight").is_ok() {
            tensor("embed_tokens.weight")?
        } else {
            eprintln!("[dflash] artifact has no embed_tokens.weight — the caller MUST pass the \
                       target's embed as the noise source (the serving lane always does)");
            dev.alloc_zeros::<bf16>(1)?
        };
        let fc = tensor("fc.weight")?;
        let hidden_norm = norm_f32("hidden_norm.weight", h)?;
        let norm = norm_f32("norm.weight", h)?;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let lp = format!("layers.{i}");
            layers.push(DflashLayer {
                input_ln: norm_f32(&format!("{lp}.input_layernorm.weight"), h)?,
                post_ln: norm_f32(&format!("{lp}.post_attention_layernorm.weight"), h)?,
                q_norm: norm_f32(&format!("{lp}.self_attn.q_norm.weight"), hd)?,
                k_norm: norm_f32(&format!("{lp}.self_attn.k_norm.weight"), hd)?,
                q_proj: tensor(&format!("{lp}.self_attn.q_proj.weight"))?,
                k_proj: tensor(&format!("{lp}.self_attn.k_proj.weight"))?,
                v_proj: tensor(&format!("{lp}.self_attn.v_proj.weight"))?,
                o_proj: tensor(&format!("{lp}.self_attn.o_proj.weight"))?,
                gate_proj: tensor(&format!("{lp}.mlp.gate_proj.weight"))?,
                up_proj: tensor(&format!("{lp}.mlp.up_proj.weight"))?,
                down_proj: tensor(&format!("{lp}.mlp.down_proj.weight"))?,
            });
        }
        dev.synchronize()?;

        let mut s = Self {
            dev: dev.clone(),
            stream,
            blas,
            bk,
            layers,
            embed,
            fc,
            hidden_norm,
            norm,
            cos_table: dev.alloc_zeros::<f32>(1)?,
            sin_table: dev.alloc_zeros::<f32>(1)?,
            cos_max: 0,
            toks_dev: dev.alloc_zeros::<i32>(MAX_BLOCK)?,
            write_pos: dev.alloc_zeros::<i32>(MAX_BLOCK)?,
            slot_ids: dev.alloc_zeros::<i32>(MAX_BLOCK)?,
            h, nh, nkv, hd, inter, vocab, rdim, rms_eps, rope_theta, mask_token_id,
            block, nctx, tap_layers: cfg.tap_layers.clone(), sliding: cfg.sliding.clone(),
            sliding_window: cfg.sliding_window,
            last_hidden: dev.alloc_zeros::<bf16>(h * MAX_BLOCK)?,
        };
        s.ensure_rope(max_pos.max(1024))?;
        Ok(s)
    }

    /// (Re)build the cos/sin rope tables so they cover `max_pos` positions (theta, hd 128).
    /// Matches the z-lab reference's quantization exactly: the transformers pipeline stores
    /// `inv_freq` as a bf16 buffer (model.to(bfloat16)) and the rotary forward returns
    /// `cos.to(x.dtype)/sin.to(x.dtype)` — bf16. The engine's f32 table therefore holds the
    /// bf16-quantized values (lossless upcast), so the rotation uses the reference's angles.
    fn ensure_rope(&mut self, max_pos: usize) -> Result<()> {
        if max_pos <= self.cos_max { return Ok(()); }
        let half = self.rdim / 2;
        let theta = self.rope_theta;
        // transformers compute_default_rope_parameters: fp32 power, then the bf16 buffer round.
        let mut inv = vec![0.0f32; half];
        for i in 0..half {
            let v = 1.0f32 / theta.powf(2.0 * i as f32 / self.rdim as f32);
            inv[i] = half::bf16::from_f32(v).to_f32();
        }
        let mut cos_t = vec![0.0f32; max_pos * self.rdim];
        let mut sin_t = vec![0.0f32; max_pos * self.rdim];
        for p in 0..max_pos {
            let pf = p as f32;
            for i in 0..half {
                let f = pf * inv[i];
                let (c, s) = (f.cos(), f.sin());
                // cos.to(dtype=bf16): quantize like the reference's rotary forward output.
                let c = half::bf16::from_f32(c).to_f32();
                let s = half::bf16::from_f32(s).to_f32();
                cos_t[p * self.rdim + i] = c; sin_t[p * self.rdim + i] = s;
                cos_t[p * self.rdim + i + half] = c; sin_t[p * self.rdim + i + half] = s;
            }
        }
        self.cos_table = self.dev.htod_sync_copy(&cos_t)?;
        self.sin_table = self.dev.htod_sync_copy(&sin_t)?;
        self.cos_max = max_pos;
        Ok(())
    }

    /// bf16 GEMM: out[outn, batch] = W[outn, inn] @ x[inn, batch] (all col-major except W row-major).
    /// batch <= 2 → the deterministic `gemm_binv_b`; larger → cuBLAS (same dispatch as gemm_act).
    fn gemm<X: DevicePtr<bf16>>(&self, w: &CudaSlice<bf16>, x: &X, out: &mut CudaSlice<bf16>,
            inn: usize, outn: usize, batch: usize) {
        if batch <= 2 {
            let smem = (batch * 256 * 4) as u32;
            dlaunch!(self, "gemm_binv_b", (outn as u32, 1, 1), (256, 1, 1), smem,
                (d(out), d(w), *x.device_ptr() as u64, outn as i32, inn as i32, batch as i32));
        } else {
            let cfg = GemmConfig::<bf16> {
                transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
                m: outn as i32, n: batch as i32, k: inn as i32,
                alpha: bf16::from_f32(1.0), lda: inn as i32,
                ldb: inn as i32, beta: bf16::from_f32(0.0), ldc: outn as i32,
            };
            unsafe { self.blas.gemm(cfg, w, x, out).expect("dflash gemm"); }
        }
    }

    /// Batched RMSNorm with the reference's EXACT bf16 semantics (dflash_rmsnorm_b: normalize in
    /// fp32, round to bf16, multiply by the raw bf16 weight, round again — transformers
    /// Qwen3RMSNorm). `nh` = per-head grouping (1 for whole-vector columns); weights are the RAW
    /// checkpoint values. `out`/`x` are raw device pointers so in-place use avoids a borrow clash.
    fn rmsnorm(&self, out: u64, x: u64, w: &CudaSlice<f32>, nh: usize, n: usize, b: usize) {
        let bs = n.min(1024);
        dlaunch!(self, "dflash_rmsnorm_b", ((b * nh) as u32, 1, 1), (bs as u32, 1, 1), (bs * 4) as u32,
            (out, x, d(w), nh as i32, n as i32, b as i32, fbits(self.rms_eps)));
    }

    /// Run the block forward. `ctx` is the conditioning feature [5*h, L] bf16 COL-major (the
    /// concat of the target's hiddens at layers {1,20,39,58,77}, one column per ctx position);
    /// `block_tokens` is the 8 block tokens; `pos_start` is the block's first ROPE position
    /// (the target chain position the block starts at). Returns logits [BLOCK, vocab] f32
    /// row-major (logits[b*vocab + t]).
    ///
    /// `noise_embed`/`head` override the checkpoint's embed_tokens stand-in with the TARGET's
    /// full-vocab bf16 embed + lm_head (the reference `dflash_generate` uses `target.embed_tokens`
    /// for the noise and `target.lm_head` for the logits — the checkpoint's own embed is a
    /// different tensor). `None` keeps the probe stand-in (checkpoint embed for both).
    pub fn forward_step<C: DevicePtr<bf16>, H: DevicePtr<bf16>>(&mut self, pool: &mut Pool,
                   kv: &mut DflashKv, ctx: &C, ncols: usize, ctx_row0: usize,
                   block_tokens: &[u32], pos_start: usize, noise_embed: Option<&H>,
                   noise_block: Option<&CudaSlice<bf16>>) -> Result<()> {
        let (h, nh, nkv, hd, inter) = (self.h, self.nh, self.nkv, self.hd, self.inter);
        let ctx_len = ncols;   // (the body below says `ctx_len` for the columns appended NOW)
        let q = block_tokens.len();
        let k = ctx_row0 + ctx_len + q;
        assert!(q > 0 && q <= self.block && q <= MAX_BLOCK,
                "dflash block must be 1..={} tokens (got {q})", self.block);
        assert!(kv.stride >= k, "dflash kv stride {} < ctx+block {k}", kv.stride);
        assert_eq!(ctx.len(), self.nctx * h * ctx_len, "ctx feature shape");
        assert_eq!(ctx_row0 + ctx_len, pos_start,
                   "dflash ctx append [{ctx_row0}, +{ctx_len}) must end at the block's first position {pos_start}");
        // The noise block is either GATHERED here from an embedding table (the probe path / an
        // artifact that carries its own embed) or handed in pre-gathered by the caller. The engine's
        // target embedding is NVFP4-PACKED, so the serving lane always pre-gathers it through the
        // engine's own embedding path (`embed_batch`) — `W::bf16()` would panic on a packed table.
        let emb_ptr = noise_embed.map(|e| *e.device_ptr() as u64).unwrap_or(*self.embed.device_ptr() as u64);

        self.ensure_rope(pos_start + q)?; // grow the cos/sin tables if needed

        // ---- per-forward device arrays (htod on the compute stream) ----
        let toks: Vec<i32> = block_tokens.iter().map(|&t| t as i32).collect();
        let write_pos: Vec<i32> = (0..q).map(|b| (pos_start + b) as i32).collect(); // block rows
        let slot_ids: Vec<i32> = vec![0i32; q];                                    // rank-space base
        unsafe {
            use cudarc::driver::result::memcpy_htod_async;
            let c = self.stream.stream;
            memcpy_htod_async(*self.toks_dev.device_ptr() as cudarc::driver::sys::CUdeviceptr, &toks, c).expect("htod toks");
            memcpy_htod_async(*self.write_pos.device_ptr() as cudarc::driver::sys::CUdeviceptr, &write_pos, c).expect("htod write_pos");
            memcpy_htod_async(*self.slot_ids.device_ptr() as cudarc::driver::sys::CUdeviceptr, &slot_ids, c).expect("htod slot_ids");
        }

        // ---- conditioning: hidden_norm(fc(target_hidden)) — once, shared by all layers ----
        let mut ctx_cond = pool.get_bf16(h * ctx_len);
        self.gemm(&self.fc, ctx, &mut ctx_cond, self.nctx * h, h, ctx_len);
        let cc = d(&ctx_cond);
        self.rmsnorm(cc, cc, &self.hidden_norm, 1, h, ctx_len);

        // ---- noise embedding: the block's tokens (the TARGET's embed when the loop passes it) ----
        let hidden = pool.get_bf16(h * q);
        if let Some(nb) = noise_block {
            assert_eq!(nb.len(), h * q, "dflash noise block must be [h, block]");
            unsafe {
                let r = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    d(&hidden), *nb.device_ptr() as u64, h * q * 2, self.stream.stream);
                assert!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                        "dflash noise block copy failed: {r:?}");
            }
        } else {
            assert!(noise_embed.is_some() || self.embed.len() > 1,
                    "dflash: no noise embedding — the artifact has no embed_tokens.weight and the \
                     caller passed neither a table nor a pre-gathered block");
            dlaunch!(self, "embed_gather_b", grid(h * q), (256, 1, 1), 0,
                (d(&hidden), emb_ptr, *self.toks_dev.device_ptr() as u64,
                 h as i32, q as i32));
        }

        // Stable-sized per-layer scratch (allocated once).
        let mut normed = pool.get_bf16(h * q);
        let mut qb = pool.get_bf16(nh * hd * q);
        let mut k_noise = pool.get_bf16(nkv * hd * q);
        let mut v_noise = pool.get_bf16(nkv * hd * q);
        let mut attn = pool.get_bf16(nh * hd * q);
        let mut attn_out = pool.get_bf16(h * q);
        let mut gate = pool.get_bf16(inter * q);
        let mut up = pool.get_bf16(inter * q);
        let mut mlp_out = pool.get_bf16(h * q);
        // ctx-sized scratch.
        let mut k_ctx = pool.get_bf16(nkv * hd * ctx_len);
        let mut v_ctx = pool.get_bf16(nkv * hd * ctx_len);

        // --dflash-debug: dump intermediates as f32 for the golden comparison.
        let dbg = crate::opts::var(crate::opt!("dflash-debug")).is_ok();
        let dump = |tag: &str, b: &CudaSlice<bf16>, n: usize| {
            if dbg {
                let v = self.dev.dtoh_sync_copy(b).unwrap();
                let v: Vec<f32> = v[..n].iter().map(|x| x.to_f32()).collect();
                let mut f = std::io::BufWriter::new(std::fs::File::create(format!("/tmp/dflash_{tag}.bin")).unwrap());
                use std::io::Write;
                for x in v { f.write_all(&x.to_le_bytes()).unwrap(); }
            }
        };
        if dbg { dump("ctx_cond", &ctx_cond, h * ctx_len); dump("block", &hidden, h * q); }
        if dbg {
            // dump the cos/sin table rows [pos_start, pos_start+8)
            for (tag, t) in [("cos", &self.cos_table), ("sin", &self.sin_table)] {
                let v = self.dev.dtoh_sync_copy(t).unwrap();
                let v: Vec<f32> = v[pos_start * self.rdim..(pos_start + q) * self.rdim].to_vec();
                let mut f = std::io::BufWriter::new(std::fs::File::create(format!("/tmp/dflash_{tag}tab.bin")).unwrap());
                use std::io::Write;
                for x in v { f.write_all(&x.to_le_bytes()).unwrap(); }
            }
        }
        // raw (pre-norm) q/k buffers
        let mut q_raw = pool.get_bf16(nh * hd * q);
        let mut kn_raw = pool.get_bf16(nkv * hd * q);

        // Rope table pointers: ctx keys rotate at positions 0..L-1 (table base), the block's
        // q/k at pos_start..pos_start+7 (row offset into the table).
        let cos0 = *self.cos_table.device_ptr() as u64;
        let sin0 = *self.sin_table.device_ptr() as u64;
        let block_off = (pos_start * self.rdim * 4) as u64;
        // The ctx columns are the target's hiddens at the ABSOLUTE positions [pos_start-ctx_len,
        // pos_start) — the reference rotates every key at its own absolute position
        // (`position_ids[:, start - target_hidden.shape[1] : start + verify_size]`). The cache ROW
        // is local (0..ctx_len-1); only the rope table row is absolute.
        let ctx_off = ((pos_start.saturating_sub(ctx_len)) * self.rdim * 4) as u64;
        let stride = kv.stride;

        for (li, layer) in self.layers.iter().enumerate() {
            self.rmsnorm(d(&mut normed), d(&hidden), &layer.input_ln, 1, h, q);
            // q / k_noise / v_noise from the block hidden; k_ctx / v_ctx from the conditioning.
            self.gemm(&layer.q_proj, &normed, &mut q_raw, h, nh * hd, q);
            self.gemm(&layer.k_proj, &normed, &mut kn_raw, h, nkv * hd, q);
            self.gemm(&layer.v_proj, &normed, &mut v_noise, h, nkv * hd, q);
            self.gemm(&layer.k_proj, &ctx_cond, &mut k_ctx, h, nkv * hd, ctx_len);
            self.gemm(&layer.v_proj, &ctx_cond, &mut v_ctx, h, nkv * hd, ctx_len);
            if dbg { dump(&format!("normed_{li}"), &normed, h * q); }
            if dbg { dump(&format!("qraw_{li}"), &q_raw, nh * hd * q); }
            if dbg { dump(&format!("knraw_{li}"), &kn_raw, nkv * hd * q); }
            // Per-head q/k norm (into the post-norm buffers), then rotary.
            self.rmsnorm(d(&mut qb), d(&q_raw), &layer.q_norm, nh, hd, q);
            self.rmsnorm(d(&mut k_noise), d(&kn_raw), &layer.k_norm, nkv, hd, q);
            if dbg { dump(&format!("qnorm_{li}"), &qb, nh * hd * q); }
            self.rmsnorm(d(&mut k_ctx), d(&k_ctx), &layer.k_norm, nkv, hd, ctx_len);
            if dbg {
                dump(&format!("knn_{li}"), &k_noise, nkv * hd * q);
                dump(&format!("kcn_{li}"), &k_ctx, nkv * hd * ctx_len);
            }
            dlaunch!(self, "dflash_rope_b", grid(q * nh * (self.rdim / 2)), (256, 1, 1), 0,
                (d(&qb), cos0 + block_off, sin0 + block_off, nh as i32, hd as i32, self.rdim as i32, q as i32));
            dlaunch!(self, "dflash_rope_b", grid(q * nkv * (self.rdim / 2)), (256, 1, 1), 0,
                (d(&k_noise), cos0 + block_off, sin0 + block_off, nkv as i32, hd as i32, self.rdim as i32, q as i32));
            dlaunch!(self, "dflash_rope_b", grid(ctx_len * nkv * (self.rdim / 2)), (256, 1, 1), 0,
                (d(&k_ctx), cos0 + ctx_off, sin0 + ctx_off, nkv as i32, hd as i32, self.rdim as i32, ctx_len as i32));
            if dbg {
                dump(&format!("q_{li}"), &qb, nh * hd * q);
                dump(&format!("kn_{li}"), &k_noise, nkv * hd * q);
                dump(&format!("kc_{li}"), &k_ctx, nkv * hd * ctx_len);
                dump(&format!("v_{li}"), &v_noise, nkv * hd * q);
            }
            // KV: ctx at rows 0..L-1, block at rows L..L+7 (rank space).
            dlaunch!(self, "write_kv_prefill", grid(ctx_len * nkv * hd), (256, 1, 1), 0,
                (d(&kv.k_cache[li]), d(&kv.v_cache[li]), d(&k_ctx), d(&v_ctx),
                 stride as i32, nkv as i32, hd as i32, ctx_len as i32, ctx_row0 as i32));
            dlaunch!(self, "write_kv_b", grid(q * nkv * hd), (256, 1, 1), 0,
                (d(&kv.k_cache[li]), d(&kv.v_cache[li]), d(&k_noise), d(&v_noise),
                 *self.write_pos.device_ptr() as u64, stride as i32, nkv as i32, hd as i32, q as i32,
                 *self.slot_ids.device_ptr() as u64));
            if dbg && li == 0 {
                dump("kcache_0", &kv.k_cache[0], nkv * stride * hd);
                dump("vcache_0", &kv.v_cache[0], nkv * stride * hd);
            }
            // NON-CAUSAL attention over ALL K keys (ctx + block), softmax weights rounded to bf16
            // exactly like the reference's eager attention (dflash_attn_b). Per-layer mask, exactly
            // as the reference derives it from `layer_types` (config.is_causal is unset):
            //   `sliding_attention` → is_causal=True + sliding_window; `full_attention` → no mask.
            let layer_sliding = self.sliding.get(li).copied().unwrap_or(false);
            let window = if layer_sliding { self.sliding_window } else { 0 };
            let tmask = dflash_mask_pack(ctx_row0 + ctx_len, layer_sliding, window);
            // P14: the TILED kernel (row-tiled, K/V streamed through smem, online softmax) is the
            // default — it reads each KV byte from DRAM ~once per (row-tile, kv-head) instead of
            // 3x per (row, q_head) (see the kernel's header comment: ~14.4 GB -> ~0.46 GB per round
            // at 7.2 K). `--dflash-attn-eager=1` restores the reference-shaped eager kernel
            // (first-line repro + the A/B that proves the two agree).
            let dper = hd / 32;                    // dims per lane
            let g_heads = nh / nkv;                // q heads per kv head
            let tiled = crate::opts::var(crate::opt!("dflash-attn-eager")).is_err()
                && hd % 32 == 0
                && matches!((dper, g_heads), (1, 1) | (1, 2) | (1, 4)
                            | (2, 1) | (2, 2) | (2, 4) | (2, 8) | (2, 16)
                            | (4, 1) | (4, 2) | (4, 4) | (4, 8) | (4, 16)
                            | (8, 1) | (8, 2) | (8, 4) | (8, 8));
            if tiled {
                let tk = (8192 / hd).max(8);       // TK*hd == 8192 bf16 -> 32 KB dynamic smem
                let tsmem = (2 * tk * hd * 2) as u32;
                let rt = (q + 3) / 4;              // DFLASH_TQ = 4 rows per block
                let tgrid = ((nkv * rt) as u32, 1, 1);
                let tk_i = tk as i32;
                macro_rules! tiled_launch {
                    ($name:expr) => {
                        dlaunch!(self, $name, tgrid, (128u32, 1, 1), tsmem,
                            (d(&attn), d(&qb), d(&kv.k_cache[li]), d(&kv.v_cache[li]),
                             stride as i32, nh as i32, nkv as i32, hd as i32, k as i32, q as i32,
                             tk_i, tmask))
                    };
                }
                match (dper, g_heads) {
                    (1, 1) => tiled_launch!("dflash_attn_tiled_d1_g1"),
                    (1, 2) => tiled_launch!("dflash_attn_tiled_d1_g2"),
                    (1, 4) => tiled_launch!("dflash_attn_tiled_d1_g4"),
                    (2, 1) => tiled_launch!("dflash_attn_tiled_d2_g1"),
                    (2, 2) => tiled_launch!("dflash_attn_tiled_d2_g2"),
                    (2, 4) => tiled_launch!("dflash_attn_tiled_d2_g4"),
                    (2, 8) => tiled_launch!("dflash_attn_tiled_d2_g8"),
                    (2, 16) => tiled_launch!("dflash_attn_tiled_d2_g16"),
                    (4, 1) => tiled_launch!("dflash_attn_tiled_d4_g1"),
                    (4, 2) => tiled_launch!("dflash_attn_tiled_d4_g2"),
                    (4, 4) => tiled_launch!("dflash_attn_tiled_d4_g4"),
                    (4, 8) => tiled_launch!("dflash_attn_tiled_d4_g8"),
                    (4, 16) => tiled_launch!("dflash_attn_tiled_d4_g16"),
                    (8, 1) => tiled_launch!("dflash_attn_tiled_d8_g1"),
                    (8, 2) => tiled_launch!("dflash_attn_tiled_d8_g2"),
                    (8, 4) => tiled_launch!("dflash_attn_tiled_d8_g4"),
                    (8, 8) => tiled_launch!("dflash_attn_tiled_d8_g8"),
                    _ => unreachable!("tiled gate above enumerates every supported (dper, g)"),
                }
            } else {
                let smem = ((hd / 32) as u32 + 1) * 4;
                dlaunch!(self, "dflash_attn_b", ((q * nh) as u32, 1, 1), (hd as u32, 1, 1), smem,
                    (d(&attn), d(&qb), d(&kv.k_cache[li]), d(&kv.v_cache[li]),
                     stride as i32, nh as i32, nkv as i32, hd as i32, k as i32, q as i32,
                     tmask));
            }
            if dbg { dump(&format!("attn_{li}"), &attn, nh * hd * q); }
            // o_proj → residual → post-attention norm → swiglu MLP → residual.
            self.gemm(&layer.o_proj, &attn, &mut attn_out, nh * hd, h, q);
            if dbg { dump(&format!("attnout_{li}"), &attn_out, h * q); }
            dlaunch!(self, "add_residual_b", grid(h * q), (256, 1, 1), 0,
                (d(&hidden), d(&hidden), d(&attn_out), (h * q) as i32));
            self.rmsnorm(d(&mut normed), d(&hidden), &layer.post_ln, 1, h, q);
            self.gemm(&layer.gate_proj, &normed, &mut gate, h, inter, q);
            self.gemm(&layer.up_proj, &normed, &mut up, h, inter, q);
            dlaunch!(self, "silu_mul_b", grid(inter * q), (256, 1, 1), 0,
                (d(&gate), d(&gate), d(&up), (inter * q) as i32));
            self.gemm(&layer.down_proj, &gate, &mut mlp_out, inter, h, q);
            dlaunch!(self, "add_residual_b", grid(h * q), (256, 1, 1), 0,
                (d(&hidden), d(&hidden), d(&mlp_out), (h * q) as i32));
            if dbg { dump(&format!("layer{li}_hidden"), &hidden, h * q); }
        }

        // ---- final norm → self.last_hidden [h, q] (the caller's head / the probe's logits) ----
        let lh = *self.last_hidden.device_ptr();
        self.rmsnorm(lh, d(&hidden), &self.norm, 1, h, q);
        if dbg { let lh = self.dev.dtoh_sync_copy(&self.last_hidden).unwrap();
                 let v: Vec<f32> = lh[..h * q].iter().map(|x| x.to_f32()).collect();
                 let mut f = std::io::BufWriter::new(std::fs::File::create("/tmp/dflash_final_normed.bin").unwrap());
                 use std::io::Write; for x in v { f.write_all(&x.to_le_bytes()).unwrap(); } }

        pool.release_bf16(ctx_cond, h * ctx_len);
        pool.release_bf16(hidden, h * q);
        pool.release_bf16(normed, h * q);
        pool.release_bf16(qb, nh * hd * q);
        pool.release_bf16(k_noise, nkv * hd * q);
        pool.release_bf16(v_noise, nkv * hd * q);
        pool.release_bf16(attn, nh * hd * q);
        pool.release_bf16(attn_out, h * q);
        pool.release_bf16(gate, inter * q);
        pool.release_bf16(up, inter * q);
        pool.release_bf16(mlp_out, h * q);
        pool.release_bf16(k_ctx, nkv * hd * ctx_len);
        pool.release_bf16(v_ctx, nkv * hd * ctx_len);
        Ok(())
    }

    /// Single-shot probe entry (E29-B1): one ctx block [0, ctx_len) + the block forward, then the
    /// LM head over `self.last_hidden` and a host copy of the logits [block, vocab] f32.
    /// `noise_embed`/`head` override the checkpoint's embed_tokens stand-in (the serving lane uses
    /// the TARGET's embed + head instead — `forward_step` + the engine's own head path).
    pub fn forward<C: DevicePtr<bf16>, H: DevicePtr<bf16>>(&mut self, pool: &mut Pool, kv: &mut DflashKv,
                   ctx: &C, ctx_len: usize, block_tokens: &[u32], pos_start: usize,
                   noise_embed: Option<&H>, head: Option<&H>) -> Result<Vec<f32>> {
        let (h, vocab) = (self.h, self.vocab);
        let q = block_tokens.len();
        self.forward_step(pool, kv, ctx, ctx_len, 0, block_tokens, pos_start, noise_embed,
                          None::<&CudaSlice<bf16>>)?;
        let head_ptr = head.map(|e| *e.device_ptr() as u64).unwrap_or(*self.embed.device_ptr() as u64);
        let logits = pool.get(vocab * q);
        let smem = (q * 256 * 4) as u32;
        dlaunch!(self, "gemm_binv_f32_b", (vocab as u32, 1, 1), (256, 1, 1), smem,
            (d(&logits), head_ptr, d(&self.last_hidden), vocab as i32, h as i32, q as i32));
        let host_full = self.dev.dtoh_sync_copy(&logits).context("dtoh dflash logits")?;
        let host: Vec<f32> = host_full[..vocab * q].to_vec();
        pool.release(logits, vocab * q);
        Ok(host)
    }

    /// Top-1 token per block position over the forward's logits (first-max tie-break, matching
    /// torch's `argmax`).
    pub fn top1(&self, logits: &[f32]) -> Vec<u32> {
        assert_eq!(logits.len(), self.block * self.vocab,
                   "dflash top1: {} logits != block {} x vocab {}", logits.len(), self.block, self.vocab);
        (0..self.block).map(|b| {
            let col = &logits[b * self.vocab..(b + 1) * self.vocab];
            let mut best = 0usize;
            for (t, &x) in col.iter().enumerate() {
                if x > col[best] { best = t; }
            }
            best as u32
        }).collect()
    }
}

/// Probe-level file formats (also documented in the E29-B1 report):
///
/// DFCTX (the recorder's format, tokens embedded) — magic b"DFCTX", then LE u32: version=1, plen,
/// nsteps, h=4096, nctx_layers=5; then (plen+nsteps) u32 tokens (prompt then generated); then
/// (nsteps+1) × nctx_layers × h f32 features (feature i = the target's post-layer hiddens at
/// layers {1,20,39,58,77} at ONE position; feature 0 = prefill's last prompt position).
///
/// DFCT (this module's plain format, for golden/full-ctx tests) — magic b"DFCT", then LE u32:
/// version=1, ctx_len, nfeatures; then nfeatures × ctx_len × (nctx_layers × h) f32 (each feature
/// is a full ctx sequence; per feature: position-major, then layer-major in [1,20,39,58,77] order).
pub struct DflashProbeInput {
    pub plen: usize,
    pub steps: Vec<DflashStep>,
}

pub struct DflashStep {
    pub pos_start: usize,
    /// The conditioning feature [NCTX_LAYERS * h, ctx_len] f32 COL-major (position-major in file).
    pub ctx: Vec<f32>,
    pub ctx_len: usize,
    pub block_tokens: Vec<u32>,
    /// Ground-truth target chain (prompt + generated) for the acceptance comparison, if any.
    pub chain: Option<Vec<u32>>,
}

/// Read a ctx-features file: DFCTX (magic) or the plain DFCT format. `cfg` supplies the geometry
/// the file must agree with (`h`, `nctx`, `block`, `mask_token_id`) — the DFCTX header carries
/// h/nctx/plen/nsteps and a mismatch is a loud error, never a silent reinterpretation.
pub fn read_probe_input(path: &Path, tokens_json: Option<&Path>, cfg: &DflashCfg) -> Result<DflashProbeInput> {
    let buf = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let r32 = |b: &[u8], off: usize| -> u32 {
        u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    };
    let f32at = |b: &[u8], off: usize| -> f32 { f32::from_le_bytes(b[off..off + 4].try_into().unwrap()) };
    // tokens.json (optional; REQUIRED for the plain DFCT format): {"plen": N, "tokens": [...]}
    // — `plen` is the prompt token count, `tokens` the full target chain (prompt then generated).
    let (chain, tj_plen) = match tokens_json {
        Some(tp) => {
            let t = std::fs::read_to_string(tp).with_context(|| format!("read {}", tp.display()))?;
            let v: serde_json::Value = serde_json::from_str(&t).context("parse tokens.json")?;
            let toks: Vec<u32> = v["tokens"].as_array()
                .context("tokens.json missing \"tokens\" array")?
                .iter().filter_map(|x| x.as_u64().map(|u| u as u32)).collect();
            let plen = v["plen"].as_u64().map(|p| p as usize).unwrap_or(0);
            anyhow::ensure!(plen <= toks.len(), "tokens.json plen {plen} > token count {}", toks.len());
            (Some(toks), plen)
        }
        None => (None, 0),
    };
    if buf.len() >= 5 && &buf[..5] == b"DFCTX" {
        let ver = r32(&buf, 5);
        anyhow::ensure!(ver == 1, "DFCTX version {ver} unsupported");
        let plen = r32(&buf, 9) as usize;
        let nsteps = r32(&buf, 13) as usize;
        let h = r32(&buf, 17) as usize;
        let nl = r32(&buf, 21) as usize;
        anyhow::ensure!(h == cfg.h && nl == cfg.nctx(),
            "DFCTX h={h} nctx={nl} but the artifact is h={} nctx={}", cfg.h, cfg.nctx());
        let mut off = 25usize;
        let ntok = plen + nsteps;
        let toks: Vec<u32> = (0..ntok).map(|_| { let v = r32(&buf, off); off += 4; v }).collect();
        let feats = nsteps + 1;
        let need = feats * nl * h;
        anyhow::ensure!(off + need * 4 <= buf.len(), "DFCTX truncated");
        let mut steps = Vec::with_capacity(nsteps);
        for i in 0..nsteps {
            // feature i = one position → ctx_len 1; the feature is stored layer-major [5, h].
            let feat: Vec<f32> = (0..nl * h).map(|_| { let v = f32at(&buf, off); off += 4; v }).collect();
            // col-major [nl*h, 1] == layer-major order as stored.
            let end = (plen + i + cfg.block).min(plen + nsteps);
            let block_tokens: Vec<u32> = (plen + i..end).map(|j| toks[j]).collect();
            let block_tokens = if block_tokens.len() < cfg.block {
                let mut bt = block_tokens;
                bt.resize(cfg.block, cfg.mask_token_id); // mask padding at the chain tail
                bt
            } else { block_tokens };
            steps.push(DflashStep {
                pos_start: plen + i,
                ctx: feat,
                ctx_len: 1,
                block_tokens,
                chain: Some(toks.clone()),
            });
        }
        Ok(DflashProbeInput { plen, steps })
    } else if buf.len() >= 4 && &buf[..4] == b"DFCT" {
        let ver = r32(&buf, 4);
        anyhow::ensure!(ver == 1, "DFCT version {ver} unsupported");
        let ctx_len = r32(&buf, 8) as usize;
        let nfeatures = r32(&buf, 12) as usize;
        let mut off = 16usize;
        let per = ctx_len * cfg.nctx() * cfg.h;
        anyhow::ensure!(off + nfeatures * per * 4 <= buf.len(), "DFCT truncated");
        let mut steps = Vec::with_capacity(nfeatures);
        for i in 0..nfeatures {
            // File order: position-major, then layer-major. Convert to col-major [nl*h, L]:
            // col[(pos * nl + layer) * h + d] = file[pos * (nl*h) + layer * h + d].
            let mut ctx = vec![0.0f32; ctx_len * cfg.nctx() * cfg.h];
            for pos in 0..ctx_len {
                for layer in 0..cfg.nctx() {
                    for d in 0..cfg.h {
                        ctx[(pos * cfg.nctx() + layer) * cfg.h + d] = f32at(&buf, off);
                        off += 4;
                    }
                }
            }
            let plen = if tj_plen > 0 { tj_plen } else {
                anyhow::bail!("plain DFCT format needs tokens.json with \"plen\" (prompt token count)");
            };
            let chain_len = chain.as_ref().map(|c| c.len()).unwrap_or(plen);
            let end = (plen + i + cfg.block).min(chain_len);
            let block_tokens: Vec<u32> = chain.as_ref().map(|c| {
                let mut bt: Vec<u32> = (plen + i..end).map(|j| c[j]).collect();
                bt.resize(cfg.block, cfg.mask_token_id);
                bt
            }).unwrap_or_else(|| (0..cfg.block as u32).collect());
            steps.push(DflashStep {
                pos_start: plen + i,
                ctx,
                ctx_len,
                block_tokens,
                chain: chain.clone(),
            });
        }
        Ok(DflashProbeInput { plen: tj_plen, steps })
    } else {
        Err(anyhow!("unrecognized ctx-features file (magic must be DFCTX or DFCT)"))
    }
}
