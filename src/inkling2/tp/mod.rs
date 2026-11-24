//! S-INK2-TP2: tensor-parallel Inkling-2 (TP2, TP4). Design, gates and numbers: PLAN/INKLING2_TP2_REPORT.md.
//!
//! Process model (the engine's cluster, as every family): the head (`--server --model-dir <pack> --tp N`) ships the
//! pack to the nodes through the content-addressed blob cache (a re-run sends nothing), the nodes (`--node`) receive
//! the head's options and config (zero-config), the RDMA link comes up, every rank loads ITS slice (`shard.rs`),
//! the nodes send Ready and the head binds HTTP.
//!
//! Lockstep: the head runs the serve loop (lanes, interleaved prefill, prefix cache, sampling decisions) and, at
//! every GPU entry point of the model (`forward`, `decode_lanes`, `sample`, `logits_host`, `snap_save`,
//! `snap_restore`), first announces the call (`Op`) to every node over the retained control stream; a node replays
//! the calls in order on its slice (`node_loop`). The data plane is two fp32 all-reduces per layer visit (wo, MoE)
//! plus the sampler's (best, id) exchange; every rank derives the same token from the exchanged values. The node
//! checks the head's previous sampled tokens against its own one step late (a broken lockstep fails loudly), and
//! the transport watchdog aborts the link when a peer stops answering (every rank stops; nothing partial is served).
//!
//! | file | what |
//! |---|---|
//! | `shard.rs` | what rank r of W holds (heads, inner channels, vocabulary rows) and the host-side slicing |
//! | `kern.rs` | the TP attention launches at 24 / W heads (d0_tp_attn.cuh, generated) and the vocab-parallel sampler (d8) |
//! | `xport.rs` | the all-reduce over the RDMA doorbell link (raw launches on the model stream: graph-capturable); loopback |
//! | `launch.rs` | head / node bring-up (cluster sync, link, load, Ready), the node's replay loop |
//! | `sim.rs` | `--probe-inkling2-tp-sim`: W ranks as W threads on one GPU (loopback exchanges) vs TP1 |

pub mod kern;
pub mod launch;
pub mod shard;
pub mod sim;
pub mod xport;

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::Arc;

/// One model call, as the head announces it (lanes by creation id; positions explicit).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Op {
    /// `GpuModel::forward`: segments (lane id, p0, m), the ids, the output rows
    /// `img`: image feature rows of this chunk (chunk row, [rows][D] f32): the vision tower runs on the head only
    Forward { segs: Vec<(usize, usize, usize)>, ids: Vec<u32>, out: Vec<usize>, img: Vec<(usize, Vec<f32>)> },
    /// `GpuModel::verify_lanes` (and `decode_lanes`, one token a lane): segments (lane id, tokens, the lane's
    /// position), graph replay or eager
    Verify { segs: Vec<(usize, Vec<u32>, usize)>, graph: bool },
    /// `GpuModel::logits_full`: the full-vocabulary rows 0..n gathered on the device (CPU-sampled truncated accept)
    GatherLogits { n: usize },
    /// `GpuModel::sample`; `check` = the tokens the head drew at the previous `Sample` (the lagging lockstep check)
    Sample { t: Vec<f32>, seeds: Vec<u64>, check: Vec<u32> },
    /// `GpuModel::logits_host` (the full-vocabulary gather of CPU-sampled rows)
    Logits { n: usize },
    /// Drop 1: vocab-parallel multi-lane draft (reqs = (lane, anchor, p, t, seed, cap))
    DraftMulti { reqs: Vec<(usize, u32, usize, f32, u64, usize)> },
    /// Drop 1: vocab-parallel exact GPU accept + drafter tap commit (zero logits_full gather)
    MtpAccept { lane: usize, r0: usize, c: usize, t: f32, seed: u64, p0: usize },
    /// Drop 1: commit `rows` tap rows starting at `tap_off` into drafter `lane` at position `p0`
    MtpCommit { lane: usize, tap_off: usize, rows: usize, p0: usize },
    /// Drop 1: gather `c` rows of drafter `step` logits for a CPU-sampled (`top_p`/`top_k`) lane
    MtpGatherStep { lane: usize, c: usize },
    SnapSave { lane: usize, pos: usize },
    SnapRestore { lane: usize, p: usize },
    Shutdown,
}

/// A rank's TP state, held by `GpuModel::tp`.
pub struct TpRank {
    pub x: xport::Xport,
    /// the head's control streams to the nodes (None on a node)
    pub ctl: Option<RefCell<Vec<std::net::TcpStream>>>,
    /// the tokens this rank drew at its last `sample` (the head sends its own with the next Sample)
    pub last: RefCell<Vec<u32>>,
    /// exchange record scratch [48][world][4] (sampler [m][W][2], draft step [4][W][4], accept [8][W][4])
    pub rec: CudaSlice<f32>,
    /// Drop 2: sequence-parallel conv halo staging buffer `[world * 3][D]` fp32 (last 3 rows of every rank)
    pub halo: CudaSlice<f32>,
    full: RefCell<Option<CudaSlice<f32>>>,
    gather: RefCell<Option<CudaSlice<f32>>>,
}

static OPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn ops_sent() -> u64 { OPS.load(std::sync::atomic::Ordering::Relaxed) }

impl TpRank {
    pub fn new(dev: &Arc<CudaDevice>, x: xport::Xport, ctl: Option<Vec<std::net::TcpStream>>) -> Result<TpRank> {
        let w = x.world;
        let mut rec = dev.alloc_zeros::<f32>(48 * w * 4)?;
        dev.memset_zeros(&mut rec)?;
        let mut halo = dev.alloc_zeros::<f32>(w * 3 * crate::inkling2::model::D)?;
        dev.memset_zeros(&mut halo)?;
        if let Some(v) = &ctl { for s in v { let _ = s.set_nodelay(true); } }
        Ok(TpRank { x, ctl: ctl.map(RefCell::new), last: RefCell::new(Vec::new()), rec, halo, full: RefCell::new(None), gather: RefCell::new(None) })
    }

    /// The head: send `op` to every node (before launching it locally).
    pub fn announce(&self, op: Op) -> Result<()> {
        if let Some(c) = &self.ctl {
            OPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for (i, s) in c.borrow_mut().iter_mut().enumerate() {
                crate::cluster::send_json(s, &op).with_context(|| format!("inkling2 TP: the control stream to node rank {} is gone (node dead?)", i + 1))?;
            }
        }
        Ok(())
    }

    /// The head: broadcast a 1-byte bitmask (e.g. Markov draft early-stop mask) to every node over TCP_NODELAY.
    pub fn broadcast_u8(&self, b: u8) -> Result<()> {
        use std::io::Write;
        if let Some(c) = &self.ctl {
            for (i, s) in c.borrow_mut().iter_mut().enumerate() {
                s.write_all(&[b]).with_context(|| format!("inkling2 TP: broadcast_u8 to node rank {}", i + 1))?;
            }
        }
        Ok(())
    }

    /// A [n] fp32 scratch for the full-vocabulary gather (allocated on first use).
    pub fn full_buf(&self, dev: &Arc<CudaDevice>, n: usize) -> Result<std::cell::Ref<'_, CudaSlice<f32>>> {
        if self.full.borrow().is_none() {
            let b = dev.alloc_zeros::<f32>(n)?;
            *self.full.borrow_mut() = Some(b);
        }
        Ok(std::cell::Ref::map(self.full.borrow(), |o| o.as_ref().unwrap()))
    }

    /// A `[world * MAX_OUT * per]` fp32 staging scratch for 2-rail logits all-gather (allocated on first use).
    pub fn gather_buf(&self, dev: &Arc<CudaDevice>, n: usize) -> Result<std::cell::Ref<'_, CudaSlice<f32>>> {
        if self.gather.borrow().is_none() {
            let mut b = dev.alloc_zeros::<f32>(n)?;
            dev.memset_zeros(&mut b)?;
            *self.gather.borrow_mut() = Some(b);
        }
        Ok(std::cell::Ref::map(self.gather.borrow(), |o| o.as_ref().unwrap()))
    }
}
