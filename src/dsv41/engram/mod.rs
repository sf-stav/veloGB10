//! DSV4.1 Engram — owned jointly by legs ENGa (hashing: `tokmap.rs`, `layout.rs`, `hash.rs`), ENGb
//! (packer, manifest, read benchmark) and EIO (the runtime reader, row cache, prefetch). I0 left
//! this file as a doc-only stub: each leg adds its own `pub mod` lines here (ENGa / ENGb run in
//! wave 0 and created theirs first; on merge, take their lines).
//!
//! Contracts that already hold (from I0):
//! * the bucket layout used for byte accounting is `crate::dsv41::rankmap::engram_layout` (it must
//!   equal ENGa's `layout.rs`; the config check asserts the primes sum to `engram_num_embeddings`);
//! * head ownership per rank is `crate::dsv41::rankmap::rank_map(..).engram_heads`;
//! * the 264-B row record is `d41_engram_row` in kernels/gpu_dsv41/00_abi.cuh (256 E4M3 + 8 E8M0);
//! * the GPU decode kernels go in band 80 (`kernels/gpu_dsv41/80_engram.cuh`, host side
//!   `crate::dsv41::gpu::engram`).

// ---- ENGa: token map, layout, incremental hash state ----
pub mod hash;
pub mod layout;
pub mod tokmap;

#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod cross_tests;

// ---- ENGb: packed tables on disk (packer, manifest, reader locator, read benchmark) ----
pub mod iobench;
pub mod manifest;
pub mod pack;
pub mod source;
pub mod uring;
pub mod verify;

// ---- EIO: the runtime (reader pool, row cache, prefetch / publish, GPU hand-off, counters, probe) ----
pub mod cache;
pub mod io;
pub mod io_metrics;
pub mod io_probe;
pub mod prefetch;
