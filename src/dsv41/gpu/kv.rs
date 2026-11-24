//! DSV4.1 GPU host side — leg KC: the compressor and the cache writes. Kernels: `kernels/gpu_dsv41/4x_kv*.cuh`
//! (band 40). Four entries, all launched on the engine's blocking compute stream, PDL-ready:
//!
//! | entry | what | inputs (other legs' GEMM outputs) | writes |
//! |---|---|---|---|
//! | `d41_kc_compress` | ratio-r pool (softmax over the group, ratio 1: none) + RMSNorm + RoPE + FP4/E4M3-16 | `wkv`/`wgate` f32 (r >= 2) or `wkv` bf16 (r = 1) | pre-RoPE latent (bf16, for the index-K GEMM) + main-KV 288-byte records |
//! | `d41_kc_stash` | the call's last rows into the compressor frontier ring (R = r + 5) | same f32 projections | frontier ring |
//! | `d41_kc_index_k` | RMSNorm + RoPE + FP4/E8M0-32 | `wk` bf16 output on the latent | index-K q / s planes |
//! | `d41_kc_swa_write` | RMSNorm + RoPE + FP8/E8M0-32 | `wkv` bf16 output | SWA ring (last 134 rows, slot = pos % 134) and the call's chunk buffer |
//!
//! Launch order inside a layer: `compress` then `stash` (the first group of a call reads the ring the
//! stash overwrites). One call = one lane. The same code serves decode (1 row), verify (k rows) and
//! chunked prefill (any rows): see `group_range`.
//!
//! All pointers are raw device addresses (`u64`): the caches belong to `state.rs` / the lane allocator.
//! Probe: `--probe-dsv41-kernel --dsv41-kernel-set kv --dsv41-dump-ref <fixture dir>` (`probe.rs`).

pub mod probe;
pub mod rope;

use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, DeviceRepr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Entry kernels of this band; the gpu_dsv41 loader (`super::load_module`) collects them.
pub const FNS: &[&str] = &["d41_kc_compress", "d41_kc_stash", "d41_kc_index_k", "d41_kc_swa_write", "d41_kc_sizes"];

/// `sizeof` of the four argument structs and the offsets the probe checks, as `d41_kc_sizes` reports them.
pub const SIZES: [u32; 6] = [120, 48, 64, 80, 72, 112];

pub const DIM: usize = 512;
pub const INDEX_DIM: usize = 128;
/// Norm weight formats (`D41_FMT_*`): BF16 / F16 / F32.
pub use super::abi::fmt::{BF16 as NORM_BF16, F16 as NORM_F16, F32 as NORM_F32};

/// Groups a call over positions `[p0, p0 + rows)` completes at compress ratio `ratio`: `(first group, count)`.
/// Group g covers `[g r, g r + r)`; it is emitted by the call that holds its LAST position. Ratio 1: every row.
pub fn group_range(p0: u64, rows: u64, ratio: u32) -> (u64, u64) {
    let r = ratio as u64;
    let first = p0 / r;
    let end = (p0 + rows) / r; // groups [first, end) are complete after this call
    (first, end.saturating_sub(first))
}

/// Arguments of `d41_kc_compress` (layout = `d41_kc_compress_args`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CompressArgs {
    pub kv: u64, pub score: u64, pub kv16: u64, pub ring_kv: u64, pub ring_score: u64, pub norm_w: u64, pub inv_freq: u64,
    pub latent_out: u64, pub kv_main: u64,
    pub p0: i64,
    pub rows: i32, pub ratio: i32, pub ring_rows: i32, pub g_first: i32, pub n_groups: i32,
    pub norm_fmt: u32, pub eps: f32,
    pub err: u64,
}
unsafe impl DeviceRepr for CompressArgs {}

/// Arguments of `d41_kc_stash` (layout = `d41_kc_stash_args`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StashArgs { pub kv: u64, pub score: u64, pub ring_kv: u64, pub ring_score: u64, pub p0: i64, pub rows: i32, pub ring_rows: i32 }
unsafe impl DeviceRepr for StashArgs {}

/// Arguments of `d41_kc_index_k` (layout = `d41_kc_index_args`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IndexArgs {
    pub k_raw: u64, pub norm_w: u64, pub inv_freq: u64, pub q_plane: u64, pub s_plane: u64,
    pub ratio: i32, pub g_first: i32, pub n_groups: i32, pub norm_fmt: u32, pub eps: f32,
}
unsafe impl DeviceRepr for IndexArgs {}

/// Arguments of `d41_kc_swa_write` (layout = `d41_kc_swa_args`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SwaArgs {
    pub kv_raw: u64, pub norm_w: u64, pub inv_freq: u64, pub ring_q: u64, pub ring_s: u64, pub chunk_q: u64, pub chunk_s: u64,
    pub p0: i64, pub rows: i32, pub norm_fmt: u32, pub eps: f32,
}
unsafe impl DeviceRepr for SwaArgs {}

/// The launcher: one per device, holds the loaded module.
pub struct KvKernels { dev: Arc<CudaDevice> }

impl KvKernels {
    pub fn new(dev: &Arc<CudaDevice>) -> Result<Self> {
        super::load_module(dev)?;
        Ok(Self { dev: dev.clone() })
    }

    /// Pool + norm + main-KV store of the groups `a.g_first .. a.g_first + a.n_groups` (see `group_range`).
    pub fn compress(&self, a: &CompressArgs) -> Result<()> {
        ensure!(a.n_groups >= 0 && a.ratio >= 1 && a.ratio <= 16 && (a.ratio == 1 || a.ring_rows >= a.ratio + 5), "d41_kc_compress: bad geometry {a:?}");
        if a.n_groups == 0 { return Ok(()); }
        let f = self.dev.get_func(super::MODULE, "d41_kc_compress").context("d41_kc_compress missing")?;
        unsafe { f.launch(LaunchConfig { grid_dim: (a.n_groups as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }, (*a,)) }?;
        Ok(())
    }

    /// Copy the call's last `min(rows, R)` projection rows into the frontier ring (launch AFTER `compress`).
    pub fn stash(&self, a: &StashArgs) -> Result<()> {
        ensure!(a.rows >= 0 && a.ring_rows > 0, "d41_kc_stash: bad geometry {a:?}");
        if a.rows == 0 { return Ok(()); }
        let f = self.dev.get_func(super::MODULE, "d41_kc_stash").context("d41_kc_stash missing")?;
        let n = a.rows.min(a.ring_rows) as u32;
        unsafe { f.launch(LaunchConfig { grid_dim: (n, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }, (*a,)) }?;
        Ok(())
    }

    /// Index-K planes of `a.n_groups` groups (the bf16 `wk` output rows `k_raw[0..n_groups]`).
    pub fn index_k(&self, a: &IndexArgs) -> Result<()> {
        ensure!(a.n_groups >= 0 && a.ratio >= 1, "d41_kc_index_k: bad geometry {a:?}");
        if a.n_groups == 0 { return Ok(()); }
        let f = self.dev.get_func(super::MODULE, "d41_kc_index_k").context("d41_kc_index_k missing")?;
        let blocks = (a.n_groups as u32 + 3) / 4;
        unsafe { f.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (32, 4, 1), shared_mem_bytes: 0 }, (*a,)) }?;
        Ok(())
    }

    /// SWA ring + chunk writes of `a.rows` rows starting at `a.p0`.
    pub fn swa_write(&self, a: &SwaArgs) -> Result<()> {
        ensure!(a.rows >= 0, "d41_kc_swa_write: bad geometry {a:?}");
        if a.rows == 0 { return Ok(()); }
        let f = self.dev.get_func(super::MODULE, "d41_kc_swa_write").context("d41_kc_swa_write missing")?;
        unsafe { f.launch(LaunchConfig { grid_dim: (a.rows as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }, (*a,)) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_kc_group_range_covers_every_chunking_exactly_once() {
        for ratio in [1u32, 2, 4] {
            for n in [0u64, 1, 2, 3, 17, 40, 41] {
                // any chunking: the union of the calls' groups is exactly 0 .. n / r, each once
                for chunk in [1u64, 2, 3, 6, 7, 16] {
                    let (mut p0, mut seen) = (0u64, Vec::new());
                    while p0 < n {
                        let rows = chunk.min(n - p0);
                        let (g, c) = group_range(p0, rows, ratio);
                        seen.extend(g..g + c);
                        p0 += rows;
                    }
                    assert_eq!(seen, (0..n / ratio as u64).collect::<Vec<_>>(), "ratio {ratio} n {n} chunk {chunk}");
                }
            }
        }
    }

    #[test]
    fn dsv41_kc_arg_structs_match_the_cuda_layout() {
        // the device twin (`d41_kc_sizes`) is checked on the GPU by the probe; these are its expected values
        assert_eq!(SIZES, [120, 48, 64, 80, 72, 112]);
        assert_eq!(std::mem::size_of::<CompressArgs>(), 120);
        assert_eq!(std::mem::size_of::<StashArgs>(), 48);
        assert_eq!(std::mem::size_of::<IndexArgs>(), 64);
        assert_eq!(std::mem::size_of::<SwaArgs>(), 80);
        assert_eq!(std::mem::offset_of!(CompressArgs, p0), 72);
        assert_eq!(std::mem::offset_of!(CompressArgs, err), 112);
    }
}
