//! DSV4.1 (leg I0): the weight contract — `Dsv41Weights`, the trait every weight backend
//! implements (EXL3 trellis: leg KE; native FP8 / MXFP4: legs KN1 / KN2; plain BF16 / F16 / F32:
//! leg KE, for the plain linears mixed into EXL3 packs), the verify-width dispatch classes, and
//! the `WeightPlan` that maps every linear of a census to the backend that holds it.
//!
//! ## The contract (what every backend guarantees)
//! 1. **Per-linear backend, not per-pack.** A pack may mix formats (EAQ v1.1: 27 BF16
//!    indexer / compressor linears in an EXL3 pack). `WeightPlan::from_census` picks the backend
//!    per linear from its census format; the forward never asks "which pack is this".
//! 2. **Row classes are part of the ABI** (AGENTS §4, the rival's three M-classes): every dense /
//!    head / MoE entry takes a row count and serves it with the class `RowClass::of(rows)`:
//!    `Gemv` (1..=2 rows: weight-streaming GEMV), `Coop` (3..=16 rows: cooperative small-M kernel,
//!    where DSpark verify — 6 rows per lane — lives), `Wide` (>= 17 rows: reconstruct / dequant +
//!    tensor-core GEMM). A backend RETURNS the class that served (the caller counts it on
//!    /metrics and names it in a once-per-boot line); a backend that has no kernel for a class
//!    yet serves it through its generic entry and returns `RowClass::Generic` — counted, never
//!    refused (Revision 1: "a width without a fast twin runs on the generic entry and is counted").
//!    **Verify-width note (binding, from KN1):** the row-class BOUNDARIES above are the nominal ones;
//!    a backend defines its own (KN1 serves rows 3..=32 on its cooperative split-K family). What
//!    binds is the invariant: every serving verify width (<= 24 rows = 4 lanes x 6) produces column 0
//!    bit-identical to a 1-row decode. Callers dispatch through the backend (it returns the class
//!    that served) and never through `RowClass::of`, which is only the nominal map.
//! 3. **Batch invariance** (AGENTS §2.4): row i of an N-row call is bit-identical to a 1-row call
//!    on the same input, within one class; across classes the identity is gated per class.
//! 4. **Math = the reference's**, per backend: native = `act_quant` (FP8 E4M3, ue8m0 per 32) then
//!    the block-scaled GEMM, MXFP4 experts with FP8 activations; EXL3 = trellis decode with the
//!    Hadamard pre / post (suh / svh), f32 accumulation; plain = the stored dtype with f32
//!    accumulation. Outputs are bf16 unless the call says f32 (`Act::dtype`).
//! 5. **Streams**: every launch goes on `Launch::stream`, the engine's BLOCKING compute stream
//!    (AGENTS §2.1); `Launch::pdl` asks for programmatic dependent launch (kernels/gpu_dsv41/00_abi.cuh).
//! 6. **No host sync** inside `dense` / `moe` / `gather` / `logits` (graph-capturable).

pub mod exl3;
pub mod native;
pub mod partials;
pub mod plain;

use crate::dsv41::census::classify::{split_of, Logical, Role, Split, TensorClass};
use crate::dsv41::census::{Census, LinearFormat};
use anyhow::Result;
use std::ops::Range;

/// Rows a call may have and still be served by the GEMV class.
pub const GEMV_MAX_ROWS: usize = 2;
/// Rows a call may have and still be served by the cooperative class.
pub const COOP_MAX_ROWS: usize = 16;

/// The verify-width dispatch class of a call (part of the contract, see the module doc; the
/// boundaries on the variants are nominal, each backend defines its own).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowClass {
    /// 1..=2 rows
    Gemv,
    /// 3..=16 rows
    Coop,
    /// >= 17 rows
    Wide,
    /// a backend without a class-specific kernel served the call through its generic entry
    Generic,
}

impl RowClass {
    /// The NOMINAL class of a call of `rows` rows (I0's boundaries). Backends define their own
    /// boundaries (KN1: coop serves 3..=32); callers must dispatch through the backend and use the
    /// class it returns, never this function. The binding invariant is that every serving verify
    /// width (<= 24 rows) gives column 0 bit-identical to a 1-row decode.
    pub const fn of(rows: usize) -> RowClass {
        if rows <= GEMV_MAX_ROWS { RowClass::Gemv } else if rows <= COOP_MAX_ROWS { RowClass::Coop } else { RowClass::Wide }
    }
    pub fn name(self) -> &'static str {
        match self { RowClass::Gemv => "gemv", RowClass::Coop => "coop", RowClass::Wide => "wide", RowClass::Generic => "generic" }
    }
}

/// Which backend holds a linear.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BackendKind { Exl3, Native, Plain }

impl BackendKind {
    pub fn name(self) -> &'static str { match self { BackendKind::Exl3 => "exl3", BackendKind::Native => "native", BackendKind::Plain => "plain" } }
}

/// A raw device pointer (CUdeviceptr). The contract does not depend on cudarc's slice types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DevPtr(pub u64);

/// The element type of an activation buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActDtype { Bf16, F16, F32, Fp8E4m3 }

/// A row-major activation buffer on the device: `rows` x `cols`, leading dimension `ld` (elements).
#[derive(Clone, Copy, Debug)]
pub struct Act { pub ptr: DevPtr, pub rows: usize, pub cols: usize, pub ld: usize, pub dtype: ActDtype }

/// Where and how a call launches.
#[derive(Clone, Copy, Debug)]
pub struct Launch {
    /// the engine's blocking compute stream (raw CUstream as an integer; 0 = the legacy default stream,
    /// which no dsv41 path may use: AGENTS §2.1)
    pub stream: u64,
    /// programmatic dependent launch (every dsv41 entry kernel opens with D41_PDL_ENTRY)
    pub pdl: bool,
}

/// Scratch the caller provides (sized by `scratch_bytes`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Scratch { pub ptr: DevPtr, pub bytes: usize }

/// Overwrite y, or add into it (row-parallel partial sums, MoE accumulation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accum { Overwrite, Add }

/// A device-resident tensor of a linear (filled by the loader, leg LS, at upload).
#[derive(Clone, Debug, PartialEq)]
pub struct DevTensor { pub role: Role, pub ptr: DevPtr, pub dtype: String, pub shape: Vec<u64>, pub bytes: u64 }

/// This rank's part of a linear under TP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shard {
    Whole,
    /// output rows [range) of the full [out, in]
    Out(Range<u64>),
    /// input columns [range) (a partial sum: the caller reduces)
    In(Range<u64>),
}

/// One dense linear as a backend holds it: what the forward calls `dense` with.
#[derive(Clone, Debug)]
pub struct DenseWeight {
    pub key: String,
    pub class: TensorClass,
    pub layer: Option<u16>,
    pub sub: Option<u16>,
    pub format: LinearFormat,
    /// the full logical shape [out, in]
    pub out: u64,
    pub inn: u64,
    pub shard: Shard,
    pub tensors: Vec<DevTensor>,
}

/// One MoE layer's routed experts (trunk: 384 x top-6; DSpark stage: 128 x top-3).
#[derive(Clone, Debug)]
pub struct MoeWeight {
    pub layer: u16,
    pub n_experts: usize,
    /// the experts this rank holds (EP)
    pub local: Range<usize>,
    pub top_k: usize,
    pub dim: usize,
    pub inter: usize,
    pub swiglu_limit: f32,
    /// [local expert][w1, w2, w3]
    pub experts: Vec<[DenseWeight; 3]>,
}

/// The routing of a call: ids i32 [rows, top_k] (global expert ids) and weights f32 [rows, top_k]
/// (already renormalized and scaled by `routed_scaling_factor`, as the reference's Gate returns them).
#[derive(Clone, Copy, Debug)]
pub struct Route { pub ids: DevPtr, pub weights: DevPtr, pub top_k: usize }

/// The weight backend contract (see the module doc). One implementation per storage family.
pub trait Dsv41Weights: Send + Sync {
    fn kind(&self) -> BackendKind;
    /// What this backend holds, in one sentence (boot line / census).
    fn describe(&self) -> &'static str;
    /// Does this backend hold linears of format `f`?
    fn holds(&self, f: &LinearFormat) -> bool;
    /// Device scratch a call with `rows` rows needs.
    fn scratch_bytes(&self, w: &DenseWeight, rows: usize) -> usize;
    /// y [rows, out_local] (+)= x [rows, in_local] . W^T. Returns the class that served.
    fn dense(&self, l: &Launch, w: &DenseWeight, x: &Act, y: &Act, acc: Accum, s: Scratch) -> Result<RowClass>;
    /// Grouped routed experts: y f32 [rows, dim] += sum over (row, slot) routed to a LOCAL expert
    /// e of weight * W2_e( silu(min(W1_e x, limit)) * clamp(W3_e x, -limit, limit) ).
    /// Rows routed only to other ranks' experts contribute nothing here (the reduce adds them).
    fn moe(&self, l: &Launch, w: &MoeWeight, x: &Act, route: &Route, y: &Act, s: Scratch) -> Result<RowClass>;
    /// out [rows, dim] = E[ids] for ids in this rank's vocab rows, 0 elsewhere (vocab-parallel
    /// embedding: the reduce sums one real row). `ids` i32 [rows].
    fn gather(&self, l: &Launch, w: &DenseWeight, ids: DevPtr, rows: usize, out: &Act) -> Result<()>;
    /// FP32 logits [rows, local vocab] = x . H^T (the reference computes the head in fp32).
    fn logits(&self, l: &Launch, w: &DenseWeight, x: &Act, out: &Act, s: Scratch) -> Result<RowClass>;
}

/// One linear of the plan: which backend holds it and which census tensors it is made of.
#[derive(Clone, Debug)]
pub struct PlannedLinear {
    pub key: String,
    pub class: TensorClass,
    pub layer: Option<u16>,
    pub sub: Option<u16>,
    pub format: LinearFormat,
    pub backend: BackendKind,
    pub out: u64,
    pub inn: u64,
    pub split: Split,
    /// indices into `Census::entries`
    pub tensors: Vec<u32>,
}

/// Every linear of a pack mapped to its backend (built from the census; the loader turns each
/// entry into a `DenseWeight` / `MoeWeight` on the device).
#[derive(Clone, Debug, Default)]
pub struct WeightPlan { pub linears: Vec<PlannedLinear> }

impl WeightPlan {
    pub fn from_census(c: &Census) -> Result<WeightPlan> {
        let mut linears = Vec::new();
        for m in &c.modules {
            if !m.class.is_linear() { continue; }
            let Some(backend) = m.format.backend() else {
                anyhow::bail!("{}: format {} has no backend (census problem)", m.key, m.format.label());
            };
            let Some(Logical::Linear { out, inn }) = m.logical else { anyhow::bail!("{}: no [out, in]", m.key) };
            linears.push(PlannedLinear { key: m.key.clone(), class: m.class, layer: m.layer, sub: m.sub, format: m.format.clone(),
                                         backend, out, inn, split: split_of(m.class), tensors: m.tensors.clone() });
        }
        Ok(WeightPlan { linears })
    }
    pub fn count(&self, b: BackendKind) -> usize { self.linears.iter().filter(|p| p.backend == b).count() }
    /// `exl3 46703, plain 55` etc.
    pub fn summary(&self) -> String {
        let mut v: Vec<String> = Vec::new();
        for b in [BackendKind::Exl3, BackendKind::Native, BackendKind::Plain] {
            let n = self.count(b);
            if n > 0 { v.push(format!("{} {n}", b.name())); }
        }
        v.join(", ")
    }
}

/// The three stub backends (they state what they will hold; every op returns a clear error naming
/// the leg that implements it).
pub fn backends() -> Vec<Box<dyn Dsv41Weights>> {
    vec![Box::new(exl3::Exl3Weights), Box::new(native::NativeWeights), Box::new(plain::PlainWeights)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_row_classes_have_the_contract_boundaries() {
        assert_eq!(RowClass::of(1), RowClass::Gemv);
        assert_eq!(RowClass::of(2), RowClass::Gemv);
        assert_eq!(RowClass::of(3), RowClass::Coop);
        assert_eq!(RowClass::of(6), RowClass::Coop, "one lane's DSpark verify (block 5 + 1)");
        assert_eq!(RowClass::of(16), RowClass::Coop);
        assert_eq!(RowClass::of(17), RowClass::Wide);
        assert_eq!(RowClass::of(24), RowClass::Wide, "4 lanes x verify 6");
    }

    #[test]
    fn dsv41_stub_backends_hold_disjoint_formats_and_refuse_foreign_or_unbacked_ops() {
        let fmts = [LinearFormat::Exl3 { k: 3, mul1: true, suh: true, svh: true },
                    LinearFormat::Fp8Block { bo: 32, bi: 32, scale: "F8_E8M0".into() },
                    LinearFormat::Fp4 { block: 32, scale: "F8_E8M0".into(), scale2: false },
                    LinearFormat::Plain("BF16".into())];
        let bs = backends();
        for f in &fmts {
            let holders: Vec<BackendKind> = bs.iter().filter(|b| b.holds(f)).map(|b| b.kind()).collect();
            assert_eq!(holders, vec![f.backend().unwrap()], "{f:?}");
        }
        let w = DenseWeight { key: "x".into(), class: TensorClass::AttnWqA, layer: Some(0), sub: None, format: fmts[0].clone(),
                              out: 1280, inn: 5120, shard: Shard::Whole, tensors: vec![] };
        let a = Act { ptr: DevPtr(0), rows: 1, cols: 5120, ld: 5120, dtype: ActDtype::Bf16 };
        let l = Launch { stream: 0, pdl: false };
        for b in &bs {
            let e = b.dense(&l, &w, &a, &a, Accum::Overwrite, Scratch::default()).unwrap_err().to_string();
            // The backends are implemented now (KE / KN1 / KN2): the I0 intent survives as "a backend
            // never serves a weight it does not hold (here: an EXL3 weight with no tensors and no
            // device) and says so with an error, instead of panicking or returning a class".
            assert!(!e.is_empty() && !e.contains("leg "), "{e}");
            assert!(!b.describe().is_empty());
        }
    }
}
