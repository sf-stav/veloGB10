//! DSV4.1 loading: the shard plan per rank, shipping through the content-addressed cache, the
//! upload into `DenseWeight` / `MoeWeight` (crate::dsv41::weights), the boot memory report —
//! OWNED BY LEG LS. I0 wrote only the boot-output contract below, so every load phase reports the
//! same way the other families do (PLAN/DSV41_ADDENDUM_UX_2026-11-18.md, rows 2-4 and 15).
//!
//! ## Boot-output contract (I0)
//! * A load is a sequence of `boot_report::Stage`s with the fixed names (`config`, `tokenizer`,
//!   `weights`, `calibrate`, `graphs`, `scheduler`, `vision`, `serve`); inside a stage, every piece
//!   of work whose size is known up front is ONE `boot_report::Progress` made by [`progress`] with a
//!   [`Phase`], so the `[boot] <what>: NN%` lines read the same on every rank and every pack.
//! * The totals come from the census (`crate::dsv41::census`): `Phase::Headers` counts files,
//!   `Phase::Hash` / `Phase::Ship` / `Phase::Upload` count the bytes of `census::report::rank_bytes`
//!   for that rank, `Phase::EngramPack` / `Phase::EngramShip` count the rank's Engram rows / bytes
//!   (`rankmap::engram_bytes_for`). A phase that finds its work already done (a warm cache) calls
//!   `finish()` at once: the Progress prints nothing for quick work.
//! * Everything else a loader prints is `crate::dprintln!` / `crate::deprintln!` (debug level) or
//!   `boot_report::{warn, notice, error}`; a direct print in a new file needs a row in the logq
//!   per-file baseline (src/logq.rs) — keep load files free of them.

pub mod cli;
pub mod dryrun;
pub mod engram;
pub mod engram_gen;
pub mod fit;
pub mod metrics_rows;
pub mod plan;
pub mod ship;
pub mod splits;
pub mod stage;

use crate::boot_report::{Progress, Unit};

/// A load phase with a known size (one `[boot] <what>: NN%` line family each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// reading safetensors headers (the census): files
    Headers,
    /// content-hashing the rank's shard for the blob cache: bytes
    Hash,
    /// shipping the rank's shard to a node: bytes
    Ship,
    /// reading + uploading weights into device memory: bytes
    Upload,
    /// packing the rank's Engram head files from the native FP8 tables: bytes
    EngramPack,
    /// shipping the rank's Engram head files: bytes
    EngramShip,
    /// writing a slice pack (`--dsv41-slice-pack`): bytes
    Slice,
    /// materialising the head's own shard as the local fast-reboot stage: bytes
    Stage,
}

impl Phase {
    pub fn unit(self) -> Unit { if self == Phase::Headers { Unit::Items("files") } else { Unit::Bytes } }
    /// The `<what>` of the line: `dsv41 <phase>[ rank r/w]`.
    pub fn label(self, rank: Option<(usize, usize)>) -> String {
        let what = match self {
            Phase::Headers => "reading tensor headers", Phase::Hash => "hashing the shard", Phase::Ship => "shipping the shard",
            Phase::Upload => "loading weights", Phase::EngramPack => "packing Engram heads", Phase::EngramShip => "shipping Engram heads",
            Phase::Slice => "writing the slice", Phase::Stage => "staging the shard locally",
        };
        match rank { Some((r, w)) => format!("dsv41 {what} (rank {r} of {w})"), None => format!("dsv41 {what}") }
    }
}

/// The one way a dsv41 load phase makes its progress reporter.
pub fn progress(p: Phase, rank: Option<(usize, usize)>, total: u64) -> Progress { Progress::new(p.label(rank), total, p.unit()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_load_phase_labels() {
        assert_eq!(Phase::Ship.label(Some((2, 4))), "dsv41 shipping the shard (rank 2 of 4)");
        assert_eq!(Phase::Headers.label(None), "dsv41 reading tensor headers");
        let s = crate::boot_report::progress_str(&Phase::Upload.label(None), 50, 100, Phase::Upload.unit(), 2.0);
        assert!(s.starts_with("[boot] dsv41 loading weights:  50%"), "{s}");
    }
}
