//! `crate::dist` — Velo-Fabric v2: GSPMD-style `ShardedTensor`, lazy `PartialSum` propagation,
//! declarative `ShardSpecPlan`, and reusable multi-rail `FabricXport`.
//!
//! Designed as a **purely additive v2 layer** that coexists side-by-side with existing model-family
//! TP implementations (`src/dsv41/tp/`, `src/exl3_forward/xtp/`, `src/naive/tp.rs`) without removing
//! or modifying them. New models (e.g. Inkling-2) or v2 ports opt in by:
//! 1. Describing tensor sharding in ~15 lines via [`ShardSpecPlan`] (e.g. [`ShardSpecPlan::inkling2_default`]);
//! 2. Wrapping activations in [`ShardedTensor`] backed by [`GpuBuf`] (or host slices in loopback tests);
//! 3. Calling [`FabricXport`] (`Solo`, `Rdma`, or `Loopback`), which gives every model family the full
//!    dual-rail PDL + single-CTA C2C + 2-WR + lock-free verbs + dual-rail `Pipe` + dual-rail RS+AG
//!    transport out of the box with zero per-model RDMA boilerplate.
//!
//! # Module map
//! - [`placement`]: `Mesh`, `Placement` (`Replicated`, `Shard { axis }`, `PartialSum`), `DType`, aligned slicing.
//! - [`tensor`]: `ShardedTensor<B>`, `GpuBuf`, shape & placement invariants.
//! - [`xport`]: `CollectiveBackend` trait, `FabricXport` (`Solo` / `Rdma` / `Loopback`), canonical FP32 tree sums for world 1/2/4/8.
//! - [`ops`]: GSPMD placement propagation rules, lazy `PartialSum` materialization, `PendingPartialGroup` (rider packing), `CollectivesStats`.
//! - [`plan`]: Declarative `ShardSpecPlan` ruleset + presets (`dense_transformer`, `inkling2_default`, `moe_hybrid`).

pub mod ops;
pub mod placement;
pub mod plan;
pub mod ship;
pub mod tensor;
pub mod xport;

pub use ops::{
    col_linear_placement, elementwise_nonlinear_placement, linear_add_placement,
    materialize_gpu, row_linear_placement, CollectivesStats, PendingPartialGroup,
};
pub use placement::{aligned_shard_range, DType, Mesh, Placement, ShardRange};
pub use plan::{Pattern, ResolvedSlice, Rule, ShardSpecPlan, TensorSpec};
pub use tensor::{GpuBuf, ShardedTensor};
pub use xport::{canonical_reduce_f32, CollectiveBackend, FabricXport, LoopbackFabric};
