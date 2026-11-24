//! DSV4.1 EXL3 weight backend — OWNED BY LEG KE (I0 left this stub; KE replaces the bodies).
//!
//! What it will hold, per linear: the trellis codes (I16 [in/16, out/16, 16K], K = 2..8 read from
//! the header), the input / output Hadamard vectors (suh F16 [in], svh F16 [out]) and the mul1
//! codebook multiplier — for every EXL3 dense class (attention wq_a / wq_b / wkv / wo_a slices /
//! wo_b, compressor, indexer wq_b / wk, shared experts, Engram wkv, DSpark main_proj, the head),
//! the grouped routed experts (384 x top-6, 5120 -> 2304 -> 5120; K = 3 fast path, K = 2 for
//! layers 18-22 in the EAQ v1 pack) and the DSpark experts (128 x top-3, K = 4). All three row
//! classes (`RowClass`): GEMV (<= 2 rows), cooperative (3..16), reconstruct + tensor-core GEMM
//! (>= 17). Kernels: `kernels/gpu_dsv41/10_trellis.cuh` (band 10-19), copied and re-parameterised
//! from the engine's EXL3 kernels (never edited in place: Flash-Next stays untouched).

pub mod dense;
pub mod dense_dec;
pub mod dense_wide;
pub mod launch;
pub mod moe;
pub mod moe_cp;
pub mod moe_dec;
pub mod moe_wide;
pub mod probe;
pub mod probe_bench;
pub mod probe_moe;
pub mod probe_pd1;
pub mod q1_oracle;

use super::{Accum, Act, ActDtype, BackendKind, DenseWeight, DevPtr, Dsv41Weights, Launch, MoeWeight, Route, RowClass, Scratch};
use crate::dsv41::census::LinearFormat;
use anyhow::{bail, Result};

/// One-time device queries of the KE2 paths (geometry handshakes: a tiny kernel + a sync each). Call once at load, before
/// any graph capture; `moe::prepare_moe` also runs the MoE ones. Every later call is capture-safe.
pub fn prepare() -> Result<()> {
    dense_dec::geometry()?;
    moe_dec::geometry()?;
    moe_cp::geometry()?;
    moe_wide::check_geometry()?;
    dense_wide::check_geometry()
}

/// Entry kernels of band 10 (`kernels/gpu_dsv41/1x_*.cuh`); the gpu_dsv41 loader collects them.
pub const FNS: &[&str] = launch::FNS;

/// EXL3 backend. Unit struct: the per-layer MoE tables live in a process-wide cache (`moe::prepare_moe`).
pub struct Exl3Weights;

impl Dsv41Weights for Exl3Weights {
    fn kind(&self) -> BackendKind { BackendKind::Exl3 }
    fn describe(&self) -> &'static str {
        "EXL3 trellis linears (K = 1.5..8, suh / svh Hadamard-128, mul1 codebook): dense with (kb, nb) slices, grouped decode MoE over inner-width slice tables (routed + shared + DSpark), vocab-sliced head; rows <= 2 / 3..16 / 17+ classes"
    }
    fn holds(&self, f: &LinearFormat) -> bool { matches!(f, LinearFormat::Exl3 { .. }) }
    fn scratch_bytes(&self, w: &DenseWeight, rows: usize) -> usize { dense::scratch_bytes(w, rows) }
    fn dense(&self, l: &Launch, w: &DenseWeight, x: &Act, y: &Act, a: Accum, s: Scratch) -> Result<RowClass> {
        if let LinearFormat::Exl3 { mul1, .. } = &w.format { if !*mul1 { bail!("EXL3 {}: only the mul1 codebook is implemented (this pack uses another)", w.key) } }
        dense::dense(l, w, x, y, a, s)
    }
    fn moe(&self, l: &Launch, w: &MoeWeight, x: &Act, r: &Route, y: &Act, s: Scratch) -> Result<RowClass> { moe::moe(l, w, x, r, y, s) }
    fn gather(&self, _l: &Launch, w: &DenseWeight, _ids: DevPtr, _rows: usize, _o: &Act) -> Result<()> {
        bail!("EXL3 embedding gather {}: V4.1 packs store the embedding plain; the plain backend serves it", w.key)
    }
    fn logits(&self, l: &Launch, w: &DenseWeight, x: &Act, o: &Act, s: Scratch) -> Result<RowClass> {
        if o.dtype != ActDtype::F32 { bail!("EXL3 head {}: logits are f32", w.key) }
        dense::dense(l, w, x, o, Accum::Overwrite, s)
    }
}
