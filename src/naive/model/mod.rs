//! Naive-N0.5-Flash GPU model (S4): the first `n_layers` layers of the pack on one GB10, decode /
//! verify rows (M <= 16) at consecutive positions, every op through `kernels/gpu_naive.cu`.
//! Eager launches on the device's default stream; graphs, TP and prefill come later (S5/S6).
//!
//! Gate: `--probe-naive-slice --model-dir <pack> --layers N --tokens T` runs T tokens through the
//! GPU slice and the CPU twin (`CpuModel::load_pack_slice`, dequantized weights) and compares the
//! residual after every layer and the logits.
//!
//! Layout (all row-major, M = rows):
//!   x       f32 [M][H]      the residual stream
//!   h       f16 [M][H]      normed input of a sublayer
//!   q/k/v   f16 [M][heads*192]  (v per-head padded as stored: 192 per kv head, 128 live)
//!   attn    f16 [M][nh*128]
//!   KV caches per layer: kc [kv][S][192], vc [kv][S][128] f16 — S = ring slots for SWA layers
//!   (pos & (S-1)), S = max_pos for DSA layers; DSA layers also keep the indexer key plane
//!   kq [max_pos][128] e4m3 + ks [max_pos] f32.
//!
//! Map of this directory (`GpuModel`'s impl is cut by phase; every item keeps its `crate::naive::model::` path):
//! - `mod.rs` (this file): row / smem constants, GPU-side layer structs, `GpuModel`, pack tensor readers.
//! - `load.rs`: slice / layer loading and upload.
//! - `step.rs`: launch helpers, the dense chain, `step_kernels*` and the per-layer forward.
//! - `moe.rs`: the MoE sublayer (`moe_ws`, wide variant, combine) and the TP reduce.
//! - `rows.rs`: `forward_rows*` (chunked prefill / verify entry points).
//! - `helpers.rs`: chunk partition, geometry packing, device-pointer helpers, the `Ws` bump workspace.
//! - `probes.rs`: the `--probe-naive-*` gate entry points.
//! - `t0c_tests.rs`, `t2d_tests.rs`, `geom_tests.rs`: the inline unit-test modules, one file each.

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use super::config::{AttnKind, NaiveConfig};
use super::gpu::{load_module, upload_quad, GpuQuad, MODULE};
use super::pack::NaivePack;

pub const MAX_ROWS: usize = 16;
/// the widest prefill chunk a step takes (the workspace is sized for it)
pub const MAX_PREFILL_ROWS: usize = 2048;

/// T0-c: the largest dynamic smem the naive path may request WITHOUT a per-function opt-in.
///
/// The naive module is loaded through cudarc (`dev.load_ptx`) and every launch goes through
/// `CudaFunction::launch_on_stream`, and cudarc exposes NO accessor for the raw `CUfunction`
/// (`cu_function` is `pub(crate)`) and does not set the attribute itself — it passes
/// `shared_mem_bytes` straight to `cuLaunchKernel`. So the >48 KB opt-in that `gpu::SMEM_OPTIN_CAP`
/// (99 KB) budgets for CANNOT be applied to the copy that actually launches here. Setting it on a
/// private `cuModuleLoadDataEx` copy, as `src/gpu.rs` and `src/exl3_forward.rs` do for their own
/// raw-fn launches, would be the exact silent-no-op the 2026-10-03 lesson records: the attribute
/// binds per `CUfunction`, so the private copy's attribute would not reach cudarc's copy and a
/// >48 KB launch would fail at runtime while every static gate stayed green.
///
/// T0-c therefore does NOT try to raise the cap. It removes the need to: staging only the CTA's
/// k-window makes the request `ks` times smaller, which is what brings the wide shapes (mlp down,
/// k = 16384) back under 48 KB at spec-block 8. Reaching beyond 48 KB on this path would need
/// cudarc to grow a raw-handle accessor — a separate, deliberately unstarted item.
pub const DEFAULT_SMEM_CAP: usize = 48 * 1024;

/// T0-c: dynamic smem for the STAGED (`a1`) body, in bytes, for `m` rows, a `k`-wide input and a
/// split-K factor of `ks`. Since T0-c the kernel stages only this CTA's k-window — `k / ks` columns —
/// so the staged bytes are `m * ((k/ks) + 8) * 2`, not `m * (k + 8) * 2`. `ks == 1` (the window is the
/// whole of k) is byte-identical to the pre-T0-c number, which is what makes the change safe.
///
/// 0 means "does not fit": the caller must then take the per-step (`nv_gemm_body`) path, which needs
/// no smem at all. It must NEVER return a size the kernel would overflow — the budget is an upper
/// bound, not a check (see `gpu::SMEM_OPTIN_CAP` and the illegal-smem lesson), so the arithmetic here
/// and `nv_gemm_body_a1`'s `SK` must never drift apart. `nv_a1_fits` in kernels/gpu_naive.cu is the
/// device-side twin of the windowing decision and is deliberately NOT the cap.
#[inline]
pub fn a1_smem(m: usize, k: usize, ks: usize) -> u32 {
    if m == 0 || m > MAX_ROWS || ks == 0 || k == 0 || k % 16 != 0 { return 0; }
    if (k / 16) % ks != 0 { return 0; }          // the host already requires kb % ks == 0
    let kb = k / 16;
    let chunk = kb / ks;                         // k-tiles this CTA walks
    if chunk == 0 || chunk % 8 != 0 { return 0; } // window must be 128-wide: the fused `suh` path
                                             // runs nv_had128 over whole 128-blocks, so a narrower
                                             // window would straddle a butterfly group (see T0-c).
    let bytes = m * (chunk * 16 + 8) * 2;
    if bytes <= DEFAULT_SMEM_CAP as usize { bytes as u32 } else { 0 }
}

pub struct GpuIndexer {
    pub wq: GpuQuad,
    pub wk: GpuQuad,
    pub wp: GpuQuad,
    pub kn_w: CudaSlice<u16>,
    pub kn_b: CudaSlice<u16>,
    pub kq: CudaSlice<u8>,   // [max_pos][ihd]
    pub ks: CudaSlice<f32>,  // [max_pos]
}

pub struct GpuMoe {
    pub router: CudaSlice<u16>, // f16 [E][H]
    pub bias: CudaSlice<f32>,   // [E]
    pub g_tr: CudaSlice<u16>, pub u_tr: CudaSlice<u16>, pub d_tr: CudaSlice<u16>,
    pub suh_g: CudaSlice<u16>, pub svh_g: CudaSlice<u16>,
    pub suh_u: CudaSlice<u16>, pub svh_u: CudaSlice<u16>,
    pub suh_d: CudaSlice<u16>, pub svh_d: CudaSlice<u16>,
    pub g_words: u64, pub u_words: u64, pub d_words: u64,
    pub bpb: usize,
    /// trellis word offsets of the LOCAL experts (le * words): the grouped prefill GEMM's table
    pub offs_g: CudaSlice<u64>,
    pub offs_d: CudaSlice<u64>,
    /// routing-weight scale: routed_scaling_factor x the converter's up_proj divisor of this layer
    /// (`NaiveConfig::moe_interm_div`; 128 at layer 47), verified against the pack's suh at load
    pub scale: f32,
    /// expert TP (`--naive-moe-shard tp`): this rank's slice of every expert's intermediate (mi / world); else mi
    pub mi_l: usize,
    /// the experts held on this rank (the local index space of `offs_g` / `offs_d`): all of them under expert TP
    pub el: usize,
}

/// Expert TP resolution for a load: a shipped shard says it in its deal (`NaivePack::etp`); the whole pack (the head,
/// a loopback probe) follows `--naive-moe-shard`. World 1 has nothing to shard.
pub fn etp_of(pack: &NaivePack, shard: &super::shard::Shard) -> bool {
    shard.world > 1 && if pack.shard.is_some() { pack.etp }
                       else { super::shipplan::moe_tp_resolve(crate::opts::var(crate::opt!("naive-moe-shard")).ok().as_deref(), shard.world) }
}

pub enum GpuMlp {
    Dense { gate: GpuQuad, up: GpuQuad, down: GpuQuad },
    Moe(GpuMoe),
}

pub struct GpuLayer {
    pub kind: AttnKind,
    /// this rank's query heads / kv heads (TP: 1/W of the model's)
    pub nh_l: usize,
    pub kv_l: usize,
    /// column offset of the first live column inside the (128-block-rounded) k / v slices
    pub k_off: usize,
    pub v_off: usize,
    pub in_norm: CudaSlice<u16>,
    pub post_norm: CudaSlice<u16>,
    pub q: GpuQuad, pub k: GpuQuad, pub v: GpuQuad, pub o: GpuQuad,
    pub sink: Option<CudaSlice<u16>>,
    pub idx: Option<GpuIndexer>,
    pub mlp: GpuMlp,
    pub kc: CudaSlice<u16>,
    pub vc: CudaSlice<u16>,
    pub slots: usize, // S
}

pub struct GpuModel {
    pub dev: Arc<CudaDevice>,
    /// the (non-blocking, capturable) stream every forward kernel rides
    pub stream: cudarc::driver::CudaStream,
    pub cfg: NaiveConfig,
    /// a zero sink row (the SWA kernel's dummy when a layer has no sinks)
    pub zero_sink: CudaSlice<u16>,
    pub shard: super::shard::Shard,
    /// S5: the all-reduce transport (None = one GPU holds the whole model slice)
    pub tp: Option<super::tp::NaiveTp>,
    /// nv_moe_route's ownership map (identity at world 1)
    pub local_of: CudaSlice<i32>,
    pub layers: Vec<GpuLayer>,
    pub embed: CudaSlice<u16>, // bf16 bits [V][H]
    pub norm: CudaSlice<u16>,
    /// this rank's vocabulary slice of the head: columns [v0, v0 + vl) (the whole head at world 1)
    pub lm_head: GpuQuad,
    pub v0: usize,
    pub vl: usize,
    pub max_pos: usize,
    pub ring: usize,
    /// how many requests may hold cache state at once: the per-request slot dimension in front of
    /// every pos-keyed cache (kc/vc/kq/ks). 1 = today's single-request sizing, byte for byte.
    pub n_slots: usize,
    /// device pointer to the per-row slot-id array ([MAX_ROWS] i32, uploaded per step). A row's
    /// slot_ids[m] selects which request's cache plane m reads/writes.
    pub sids: u64,
    /// owns the allocation `sids` points at (zero-initialised: every row starts on slot 0)
    pub sids_buf: CudaSlice<i32>,
    /// F13: the fused split-K combine's per-tile arrival counters (zeroed once; self-resetting)
    pub fuse_cnt: CudaSlice<i32>,
}

// The stream handle is only ever used from the thread that owns the model (the serve loop / a probe).
unsafe impl Send for GpuModel {}

pub(crate) fn f16_bits_of(pack: &NaivePack, name: &str) -> Result<Vec<u16>> {
    let m = pack.plain_tensor(name)?;
    let raw = m.read_bytes(&pack.dir)?;
    Ok(match m.dtype.as_str() {
        "F16" => raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        "BF16" => raw.chunks_exact(2).map(|c| half::f16::from_f32(half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).to_bits()).collect(),
        "F32" => raw.chunks_exact(4).map(|c| half::f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_bits()).collect(),
        other => bail!("{name}: dtype {other}"),
    })
}
pub(crate) fn f32_of(pack: &NaivePack, name: &str) -> Result<Vec<f32>> {
    let m = pack.plain_tensor(name)?;
    let raw = m.read_bytes(&pack.dir)?;
    Ok(match m.dtype.as_str() {
        "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "F16" => raw.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        "BF16" => raw.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        other => bail!("{name}: dtype {other}"),
    })
}
fn le_u16(raw: &[u8]) -> Vec<u16> { raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() }

// `super::<module>::` paths in the moved method bodies resolve through these re-imports (children sit one level deeper).
use super::{cpu, gpu, shard, stage};

mod load;
mod step;
mod moe;
mod rows;
mod helpers;
mod probes;
pub use self::helpers::*;
pub use self::probes::*;

#[cfg(test)]
mod t0c_tests;

#[cfg(test)]
mod t2d_tests;

/// The `geom` word is packed in Rust and unpacked in CUDA, both by hand, and neither side is
/// type-checked against the other. `2acaa750` packed `S` at bit 48 and unpacked it with
/// `& 0xFFFF`; at the shipped `--max-seq-len 262144` that truncated the per-head KV stride to
/// 0, which aliased every KV head onto head 0 across all DSA layers. The 3-layer world=1 slice
/// probes could not see it (their `max_pos` fits in 16 bits) and `--probe-naive-slots` passed.
/// These tests are the ones that class needed.
#[cfg(test)]
mod geom_tests;
