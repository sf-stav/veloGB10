//! The vocab-parallel exchange of the DSpark round (T0 §3.4, §3.7): what this leg needs from the TP layer.
//!
//! Each rank holds V / N rows of the head (and of the Markov head): rank r owns ids `[r V / N, (r + 1) V / N)`
//! (`crate::dsv41::rankmap`: TP2 64,640, TP4 32,320, TP8 16,160 per rank). A vocab pass produces ONE 64-byte
//! record per row on every rank ([`super::VRec`]: the raw maximum and its id, the log-sum-exp of the processed
//! distribution, the Gumbel maximum and its id, a probe token's value, the residual sum and maximum). The round
//! needs these records all-gathered: every rank then merges the N records of a row in ascending rank order and
//! takes the same decisions (replicated, no broadcast). No logits cross the wire.
//!
//! Per round: one all-gather per chain step of the drafter (5, each `lanes x 64 B` per rank), one for the verify
//! rows (`lanes x (k + 1) x 64 B`), one for the residual (`lanes x 64 B`): 7 tiny collectives, all on the compute
//! stream, inside the round's graph.
//!
//! **Contract for leg TPC** (an implementation of [`VocabExchange`] over `tp_xport`):
//! * `all_gather(g, send, bytes, recv)`: on the engine's compute stream, after the kernels already queued on it,
//!   `recv[r * bytes .. (r + 1) * bytes]` = rank r's `send[0 .. bytes]` for every r, before any kernel queued after
//!   the call runs; `recv` of this rank's own slot may be written by the transport or copied locally;
//! * graph-capturable (device-side signalling, no host sync, fixed buffers registered once: `send` / `recv` are
//!   the same addresses every round of a shape, sizes <= `max_bytes`);
//! * deterministic placement (rank order), no reduction (the merge is ours, in a fixed order).
//! The stage forward of the drafter also needs the trunk's reduces (wo_b, the FFN partial, the mHC rank part,
//! `main_proj`'s K-split partial riding layer 39's reduce): those are the TP forward's (SL's handoff, T0 §3.7).

use crate::dsv41::fwd::dev::Gpu;
use anyhow::{ensure, Result};

pub trait VocabExchange {
    fn ranks(&self) -> usize;
    fn rank(&self) -> usize;
    /// The most bytes one call sends per rank (the transport registers its buffers for this size).
    fn max_bytes(&self) -> usize;
    /// All-gather `bytes` of `send` from every rank into `recv` ([ranks][bytes], rank order), on the compute stream.
    fn all_gather(&self, g: &Gpu, send: u64, bytes: usize, recv: u64) -> Result<()>;
}

/// This rank's vocab range `[v0, v0 + len)` of a `vocab`-row head split over `ranks` (even split, as rankmap).
pub fn vocab_range(vocab: usize, ranks: usize, rank: usize) -> (usize, usize) {
    let per = vocab / ranks;
    let v0 = rank * per;
    let len = if rank + 1 == ranks { vocab - v0 } else { per };
    (v0, len)
}

/// TP1: one rank, the gathered buffer is the rank's own records (no copy when `recv == send`).
pub struct Local { pub max_bytes: usize }

impl VocabExchange for Local {
    fn ranks(&self) -> usize { 1 }
    fn rank(&self) -> usize { 0 }
    fn max_bytes(&self) -> usize { self.max_bytes }
    fn all_gather(&self, g: &Gpu, send: u64, bytes: usize, recv: u64) -> Result<()> {
        ensure!(bytes <= self.max_bytes, "vocab exchange: {bytes} B > {} B", self.max_bytes);
        if send != recv { g.d2d(recv, send, bytes)?; }
        Ok(())
    }
}

/// A test exchange that simulates N ranks on ONE device: the caller runs each rank's vocab pass into its own slot
/// of `recv` (the records a real all-gather would deliver); `all_gather` is then a no-op. Used by the probe to show
/// that a vocab split over N shards merges to the same decisions as one shard (TP invariance of the sampler).
pub struct Simulated { pub n: usize, pub max_bytes: usize }

impl VocabExchange for Simulated {
    fn ranks(&self) -> usize { self.n }
    fn rank(&self) -> usize { 0 }
    fn max_bytes(&self) -> usize { self.max_bytes }
    fn all_gather(&self, _g: &Gpu, _send: u64, bytes: usize, _recv: u64) -> Result<()> {
        ensure!(bytes <= self.max_bytes, "vocab exchange: {bytes} B > {} B", self.max_bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_ds_vocab_ranges_tile_the_vocab() {
        for n in [1usize, 2, 4, 8] {
            let mut next = 0;
            for r in 0..n {
                let (v0, len) = vocab_range(129_280, n, r);
                assert_eq!(v0, next);
                next = v0 + len;
            }
            assert_eq!(next, 129_280);
        }
        assert_eq!(vocab_range(129_280, 4, 1), (32_320, 32_320));
    }
}
