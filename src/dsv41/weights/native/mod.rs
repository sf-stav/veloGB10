//! DSV4.1 native weight backend (DeepSeek's checkpoint format). Dense FP8 is leg KN1
//! (`dense.rs`), the MXFP4 experts are leg KN2 (`moe.rs`); I0 left this stub and the two files.
//!
//! What it will hold: FP8 E4M3 linears with one ue8m0 scale per 32 x 32 block (attention, indexer
//! wq_b, shared experts, Engram wkv, DSpark main_proj), applied the reference's way (`act_quant` of
//! the input to FP8 with ue8m0 per 32, then the block-scaled GEMM: GEMV for decode, FP8 MMA for
//! prefill); MXFP4 experts (E2M1 packed two per byte + ue8m0 per 32 inputs) with FP8 activations
//! (W4A8, `kind::f8f6f4` block-scaled MMA on sm_121a). All three row classes.

pub mod dense;
pub mod fp8;
pub mod fp8_probe;
pub mod moe;

use super::{Accum, Act, BackendKind, DenseWeight, DevPtr, Dsv41Weights, Launch, MoeWeight, Route, RowClass, Scratch};
use crate::dsv41::census::LinearFormat;
use anyhow::{bail, Result};

pub struct NativeWeights;

impl Dsv41Weights for NativeWeights {
    fn kind(&self) -> BackendKind { BackendKind::Native }
    fn describe(&self) -> &'static str {
        "native FP8 E4M3 32x32 ue8m0-block linears (act_quant FP8 inputs) and MXFP4 E2M1 / ue8m0-32 experts with FP8 activations"
    }
    fn holds(&self, f: &LinearFormat) -> bool { matches!(f, LinearFormat::Fp8Block { .. } | LinearFormat::Fp4 { .. }) }
    fn scratch_bytes(&self, w: &DenseWeight, rows: usize) -> usize { dense::scratch_bytes(w, rows) }
    fn dense(&self, l: &Launch, w: &DenseWeight, x: &Act, y: &Act, a: Accum, s: Scratch) -> Result<RowClass> { dense::dense(l, w, x, y, a, s) }
    fn moe(&self, l: &Launch, w: &MoeWeight, x: &Act, r: &Route, y: &Act, s: Scratch) -> Result<RowClass> { moe::moe(l, w, x, r, y, s) }
    fn gather(&self, _l: &Launch, w: &DenseWeight, _ids: DevPtr, _rows: usize, _o: &Act) -> Result<()> {
        bail!("native embedding gather {}: the V4.1 embedding is plain BF16 (plain backend, leg KE)", w.key)
    }
    fn logits(&self, _l: &Launch, w: &DenseWeight, _x: &Act, _o: &Act, _s: Scratch) -> Result<RowClass> {
        bail!("native head {}: the V4.1 head is plain BF16 in the native checkpoint (plain backend, leg KE)", w.key)
    }
}
