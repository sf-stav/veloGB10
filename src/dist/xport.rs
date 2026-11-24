//! `FabricXport` — reusable multi-rail RDMA collective transport (v2) + deterministic host loopback
//! simulator for world 1 / 2 / 4 / 8.
//!
//! Any model family (`inkling2`, `dsv41`, `naive`, Llama/Qwen ports) can plug into [`FabricXport`]
//! instead of duplicating `naive/tp.rs`, automatically inheriting:
//! - Phase 0 cooperative abort safety & clean shutdown;
//! - Phase 1A single `TP_F_CPU_DONE` + single-CTA C2C poller w/ GPU-L2 handoff;
//! - Phase 1B 2-WR 16 B inline `{epoch, len_tag}` commit;
//! - Phase 1C `IBV_THREAD_DOMAIN` lock-free `ibv_wr_*` verbs + `S=4` CQ batching + 512 B inline;
//! - Phase 1D 2-rail `Pipe` + fused dual-rail RS+AG (`0.75F`/rail);
//! - Phase 2 2-RTT `agree_ext`, bijective `(e / rounds) % R` slots, and world-8 readiness.

use super::placement::Mesh;
use anyhow::{bail, ensure, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use std::sync::{Arc, Barrier, Mutex};

/// Model-agnostic collective backend trait implemented by [`FabricXport`].
pub trait CollectiveBackend: Send + Sync {
    fn mesh(&self) -> Mesh;
    fn all_reduce_f32(&self, stream: &CudaStream, base: u64, n: usize) -> Result<()>;
    fn reduce_scatter_rows_f32(
        &self,
        stream: &CudaStream,
        base: u64,
        row_len: usize,
        blocks: &[(usize, usize)],
    ) -> Result<usize>;
    fn all_gather_rows_multi(
        &self,
        stream: &CudaStream,
        bufs: &[(u64, usize)],
        blocks: &[(usize, usize)],
    ) -> Result<usize>;
    /// 1-RTT All-to-All across all ranks (powers DeepSpeed-Ulysses sequence-parallel attention,
    /// Jacobs et al., arXiv:2309.14509). `src` and `dst` each hold `world` slices of `slice_bytes`.
    fn all_to_all(&self, stream: &CudaStream, src: u64, dst: u64, slice_bytes: usize) -> Result<()>;
    fn abort_code(&self) -> u64;
    fn describe(&self) -> &str;
}

/// Reusable multi-rail collective transport (v2):
/// - `Solo`: zero-cost TP=1 no-op (exact same model forward runs at TP=1/2/4/8).
/// - `Rdma`: full multi-rail RoCEv2 transport (`crate::dsv41::tp::rdma::RdmaXport`).
pub enum FabricXport {
    Solo { describe: String },
    Rdma(crate::dsv41::tp::rdma::RdmaXport),
}

impl FabricXport {
    /// Zero-cost single-GPU transport (TP=1).
    pub fn solo() -> Self {
        FabricXport::Solo {
            describe: "velo-fabric v2 Solo (TP=1: all collectives are zero-cost no-ops)".into(),
        }
    }

    /// Wrap an already-attached `RdmaXport`.
    pub fn from_rdma(rdma: crate::dsv41::tp::rdma::RdmaXport) -> Self {
        FabricXport::Rdma(rdma)
    }

    /// Attach a brought-up `TpContext` for any `TpBranch` (Dsv41, Naive, Exl3, Inkling, etc.)
    /// using the full Phase 0–2 multi-rail PDL / single-CTA C2C / dual-rail RS+AG transport.
    pub fn attach(
        dev: &Arc<CudaDevice>,
        ctx: crate::tp::TpContext,
        row_bytes: usize,
        branch: &crate::tp::TpBranch,
    ) -> Result<Self> {
        let rdma = crate::dsv41::tp::rdma::RdmaXport::attach_with_branch(dev, ctx, row_bytes, branch)?;
        Ok(FabricXport::Rdma(rdma))
    }

    pub fn dual_on(&self) -> bool {
        match self {
            FabricXport::Solo { .. } => false,
            FabricXport::Rdma(r) => r.dual_on(),
        }
    }

    pub fn sp_ready(&self) -> bool {
        match self {
            FabricXport::Solo { .. } => true,
            FabricXport::Rdma(r) => r.sp_ready(),
        }
    }
}

impl CollectiveBackend for FabricXport {
    fn mesh(&self) -> Mesh {
        match self {
            FabricXport::Solo { .. } => Mesh::solo(),
            FabricXport::Rdma(r) => Mesh { rank: r.rank, world: r.world },
        }
    }

    fn all_reduce_f32(&self, stream: &CudaStream, base: u64, n: usize) -> Result<()> {
        match self {
            FabricXport::Solo { .. } => Ok(()),
            FabricXport::Rdma(r) => r.all_reduce(stream, base, n),
        }
    }

    fn reduce_scatter_rows_f32(
        &self,
        stream: &CudaStream,
        base: u64,
        row_len: usize,
        blocks: &[(usize, usize)],
    ) -> Result<usize> {
        match self {
            FabricXport::Solo { .. } => Ok(0),
            FabricXport::Rdma(r) => r.sp_reduce_scatter(stream, base, row_len, blocks),
        }
    }

    fn all_gather_rows_multi(
        &self,
        stream: &CudaStream,
        bufs: &[(u64, usize)],
        blocks: &[(usize, usize)],
    ) -> Result<usize> {
        match self {
            FabricXport::Solo { .. } => Ok(0),
            FabricXport::Rdma(r) => r.sp_all_gather_multi(stream, bufs, blocks),
        }
    }

    fn all_to_all(&self, stream: &CudaStream, src: u64, dst: u64, slice_bytes: usize) -> Result<()> {
        match self {
            FabricXport::Solo { .. } => {
                if src != dst && slice_bytes > 0 {
                    let r = unsafe { cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, slice_bytes, stream.stream) };
                    ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "Solo all_to_all copy: {r:?}");
                }
                Ok(())
            }
            FabricXport::Rdma(r) => r.all_to_all(stream, src, dst, slice_bytes),
        }
    }

    fn abort_code(&self) -> u64 {
        match self {
            FabricXport::Solo { .. } => 0,
            FabricXport::Rdma(r) => r.abort_code(),
        }
    }

    fn describe(&self) -> &str {
        match self {
            FabricXport::Solo { describe } => describe,
            FabricXport::Rdma(r) => &r.describe,
        }
    }
}

/// Canonical bit-exact FP32 reduction matching the engine's RDMA transport association:
/// - world 1: `p0`
/// - world 2: `p0 + p1`
/// - world 4: `(p0 + p2) + (p1 + p3)`
/// - world 8: `((p0 + p2) + (p4 + p6)) + ((p1 + p3) + (p5 + p7))` (== `hier8_tree_reference` == `rd_tree_reference(8, 0)`)
/// - world 3 / 6: sequential `(((p0 + p1) + p2) + ...)` matching `d41x_xcn_sum_f32`
pub fn canonical_reduce_f32(parts: &[Vec<f32>]) -> Result<Vec<f32>> {
    ensure!(!parts.is_empty(), "canonical_reduce_f32: empty partials");
    let n = parts[0].len();
    ensure!(parts.iter().all(|p| p.len() == n), "canonical_reduce_f32: ragged partial lengths");
    match parts.len() {
        1 => Ok(parts[0].clone()),
        2 => Ok((0..n).map(|i| parts[0][i] + parts[1][i]).collect()),
        4 => crate::tp_xport::single_tree_reference(parts),
        8 => crate::tp_xport::hier8_tree_reference(parts),
        3 | 6 => Ok((0..n).map(|i| {
            let mut acc = parts[0][i];
            for p in &parts[1..] { acc += p[i]; }
            acc
        }).collect()),
        w => bail!("canonical_reduce_f32: unsupported world {w} (expected 1, 2, 3, 4, 6, or 8)"),
    }
}

/// Deterministic in-process multi-rank fabric simulator for box-free testing of sharded models
/// across `world = 1, 2, 3, 4, 6, 8` with bit-exact canonical FP32 sums and All-to-All.
pub struct LoopbackFabric {
    world: usize,
    barrier: Barrier,
    slots: Mutex<Vec<Vec<f32>>>,
    reduced: Mutex<Vec<f32>>,
    collectives: Mutex<usize>,
}

impl LoopbackFabric {
    pub fn new(world: usize) -> Result<Arc<Self>> {
        ensure!(matches!(world, 1 | 2 | 3 | 4 | 6 | 8), "LoopbackFabric: world {world} (expected 1, 2, 3, 4, 6, or 8)");
        Ok(Arc::new(Self {
            world,
            barrier: Barrier::new(world),
            slots: Mutex::new(vec![Vec::new(); world]),
            reduced: Mutex::new(Vec::new()),
            collectives: Mutex::new(0),
        }))
    }

    pub fn world(&self) -> usize {
        self.world
    }

    /// Total number of `all_reduce_f32` collectives executed across the fabric.
    pub fn collectives_run(&self) -> usize {
        *self.collectives.lock().unwrap()
    }

    /// Synchronous multi-thread in-place FP32 all-reduce across all `world` ranks using [`canonical_reduce_f32`].
    pub fn all_reduce_host_f32(&self, rank: usize, buf: &mut [f32]) -> Result<()> {
        ensure!(rank < self.world, "LoopbackFabric: rank {rank} >= world {}", self.world);
        if self.world == 1 {
            return Ok(());
        }
        {
            let mut s = self.slots.lock().unwrap();
            s[rank] = buf.to_vec();
        }
        self.barrier.wait();
        if rank == 0 {
            let parts = self.slots.lock().unwrap().clone();
            let sum = canonical_reduce_f32(&parts)?;
            *self.reduced.lock().unwrap() = sum;
            *self.collectives.lock().unwrap() += 1;
        }
        self.barrier.wait();
        {
            let out = self.reduced.lock().unwrap();
            ensure!(out.len() == buf.len(), "LoopbackFabric: length mismatch {} vs {}", out.len(), buf.len());
            buf.copy_from_slice(&out);
        }
        self.barrier.wait();
        Ok(())
    }

    /// Synchronous multi-thread all-gather of equal- or uneven-length slices across all `world` ranks,
    /// concatenated in rank order `0..world`.
    pub fn all_gather_host_f32(&self, rank: usize, local: &[f32]) -> Result<Vec<f32>> {
        ensure!(rank < self.world, "LoopbackFabric: rank {rank} >= world {}", self.world);
        if self.world == 1 {
            return Ok(local.to_vec());
        }
        {
            let mut s = self.slots.lock().unwrap();
            s[rank] = local.to_vec();
        }
        self.barrier.wait();
        let out: Vec<f32> = {
            let s = self.slots.lock().unwrap();
            s.iter().flat_map(|v| v.iter().copied()).collect()
        };
        if rank == 0 {
            *self.collectives.lock().unwrap() += 1;
        }
        self.barrier.wait();
        Ok(out)
    }

    /// Synchronous multi-thread All-to-All across `world` ranks (`src` holds `world` equal slices of `slice_len`,
    /// output receives rank `p`'s slice destined for `rank` at `[p * slice_len..(p + 1) * slice_len)`).
    /// Implements the DeepSpeed-Ulysses transpose collective (Jacobs et al., arXiv:2309.14509).
    pub fn all_to_all_host_f32(&self, rank: usize, src: &[f32]) -> Result<Vec<f32>> {
        ensure!(rank < self.world, "LoopbackFabric: rank {rank} >= world {}", self.world);
        ensure!(src.len() % self.world == 0, "LoopbackFabric all_to_all: len {} not divisible by world {}", src.len(), self.world);
        if self.world == 1 {
            return Ok(src.to_vec());
        }
        let slice = src.len() / self.world;
        {
            let mut s = self.slots.lock().unwrap();
            s[rank] = src.to_vec();
        }
        self.barrier.wait();
        let mut out = Vec::with_capacity(src.len());
        {
            let s = self.slots.lock().unwrap();
            for p in 0..self.world {
                out.extend_from_slice(&s[p][rank * slice..(rank + 1) * slice]);
            }
        }
        if rank == 0 {
            *self.collectives.lock().unwrap() += 1;
        }
        self.barrier.wait();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_all_to_all_and_reduce_for_worlds_3_and_6() {
        for world in [2usize, 3, 4, 6, 8] {
            let fab = LoopbackFabric::new(world).unwrap();
            let slice = 16;
            std::thread::scope(|sc| {
                let hs: Vec<_> = (0..world)
                    .map(|r| {
                        let f = fab.clone();
                        sc.spawn(move || {
                            let src: Vec<f32> = (0..world * slice)
                                .map(|i| (r * 1000 + (i / slice) * 100 + (i % slice)) as f32)
                                .collect();
                            let got = f.all_to_all_host_f32(r, &src).unwrap();
                            for p in 0..world {
                                for k in 0..slice {
                                    assert_eq!(got[p * slice + k], (p * 1000 + r * 100 + k) as f32);
                                }
                            }
                            let mut buf: Vec<f32> = (0..32).map(|i| (r + 1) as f32 * (i as f32 * 0.25 - 2.0)).collect();
                            f.all_reduce_host_f32(r, &mut buf).unwrap();
                            buf
                        })
                    })
                    .collect();
                let outs: Vec<Vec<f32>> = hs.into_iter().map(|h| h.join().unwrap()).collect();
                for r in 1..world {
                    assert_eq!(outs[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                               outs[r].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                               "world {world}: rank {r} must be bitwise identical to rank 0");
                }
            });
        }
    }

    #[test]
    fn loopback_canonical_reduce_matches_rd_tree_for_worlds_1_2_4_8() {
        for world in [1usize, 2, 4, 8] {
            let n = 256;
            let parts: Vec<Vec<f32>> = (0..world)
                .map(|r| {
                    (0..n)
                        .map(|i| {
                            let v = ((i * 13 + r * 37) % 101) as f32 * 1.25e-2 - 0.6;
                            if (i + r) % 3 == 0 { v * 1e5 } else { v }
                        })
                        .collect()
                })
                .collect();
            let want = if world == 1 {
                parts[0].clone()
            } else {
                crate::tp_xport::rd_tree_reference(world, 0, &parts).unwrap()
            };
            let fabric = LoopbackFabric::new(world).unwrap();
            let mut handles = Vec::new();
            for r in 0..world {
                let f = fabric.clone();
                let mut v = parts[r].clone();
                handles.push(std::thread::spawn(move || {
                    f.all_reduce_host_f32(r, &mut v).unwrap();
                    v
                }));
            }
            for (r, h) in handles.into_iter().enumerate() {
                let got = h.join().unwrap();
                let gb: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
                let wb: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
                assert_eq!(gb, wb, "world {world} rank {r} bits differ from rd_tree_reference");
            }
            assert_eq!(fabric.collectives_run(), usize::from(world > 1));
        }
    }
}
