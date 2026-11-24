//! DSV4.1 tensor parallelism. The split design and ledger are leg T0's (`PLAN/DSV41_TP_DESIGN.md`); this tree is
//! the communication layer and the head / node launch (leg TPC), on which the TP forward (leg TP) runs.
//!
//! | file | what |
//! |---|---|
//! | `frame.rs` | the sites of a decode round (T0's 93 collectives), the frame layout of each (hidden + mix rider + Engram / `main_proj` riders; head-style records), the association model |
//! | `comm.rs` | **the boundary with the per-layer forward**: `Dsv41Comm` (frame / reduce / head / head_residual / check), `TpComm` (world 1 solo, world 2 / 4 RDMA), `TpRow` / `TpOut` |
//! | `rdma.rs` | the fp32 sum all-reduce over `tp_xport` (the K1 / K2 pair, the world-4 single stage, the pipelined wide path) |
//! | `kern.rs` | host side of `kernels/gpu_dsv41/05_tp.cuh` (head exchange, smoke producers / checkers, the timing gate) |
//! | `smoke.rs` | `--probe-dsv41-tp`: the synthetic-layer smoke (bitwise against single-rank sums, head exchange, rank agreement, latency tables) |
//! | `launch.rs` | the head (`--probe-dsv41-tp --tp N`) and node (`--node`, zero-config) launch over the cluster sync |
//! | `metrics.rs` | the `velogb10_dsv41_tp_*` rows |
//! | `wload.rs` | (leg TP) a rank's weights from its shard dir / stage / the pack, cuts re-derived from the plan |
//! | `fwd.rs` | (leg TP) the 40-layer trunk on a rank: T0's split on SL's pieces, the reduces, the head exchange |
//! | `tpk.rs` | (leg TP) host side of the TP forward glue kernels (`kernels/gpu_dsv41/05_tp_fwd.cuh`) |
//! | `idle.rs` | (leg HT1) the serve program's idle tier: `Op::Park`, the nodes block on the head's control connection, the wake byte |
//! | `xchg.rs` | (leg PX1) the world-2 exchange pair as the external reference recipe runs it: PDL stage / collect copies, no-sleep poll, vector copy-out; the exchange trace |
//! | `xprobe.rs` | (leg PX2) `--probe-dsv41-xchg`: both dsv41 exchange pairs against the shared kernels on fake contexts, one GPU |
//! | `mem.rs` | (leg MEM1) a serving rank's memory: allocation formulas, the scratch need, the ledger, trims, the prefix pool's pressure governor |
//!
//! Starting points from I0: the rank -> box map is `crate::dsv41::rankmap`; the per-rank tensor selection and
//! the vocabulary ranges are leg LS's (`crate::dsv41::load::{plan, splits}`). Every TP option is a CLI flag in
//! `src/opts_table.rs` with `Scope::Spmd`, so it rides `TpConfig`.

pub mod ced;
pub mod comm;
pub mod copy;
pub mod cpv;
pub mod cut;
pub mod frame;
pub mod fwd;
pub mod fwprog;
pub mod gate;
pub mod graph;
pub mod hostprof;
pub mod idle;
pub mod kern;
pub mod launch;
pub mod mem;
pub mod metrics;
pub mod pa1;
pub mod pd1;
pub mod pen;
pub mod pit;
pub mod prog;
pub mod q1tap;
pub mod rdma;
pub mod serve;
pub mod smoke;
pub mod spec;
pub mod sq1;
pub mod tpk;
pub mod width;
pub mod wload;
pub mod xchg;
pub mod xprobe;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod idle_sweep_tests;
#[cfg(test)]
mod pfx2_gpu_tests;
#[cfg(test)]
mod p7copy_gpu_tests;
