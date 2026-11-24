//! Leg PK5: the drafter stages' attention launcher (the block's rows over the lane's 128-row ring window plus the
//! block's own 5 keys, model.py DSparkAttention). One entry for every caller (TP1 drafter, FW's TP drafter, the
//! budget harness).
//!
//! The block's 5 rows per lane read 3 key blocks (2 window blocks of 64 + the extra block of 5 keys). KA's prefill
//! entry runs one CTA per (row, 16-head group) with every block in the CTA. KA's decode entry (2 splits, merged by
//! the last CTA of a (row, group)) was measured on the drafter's shapes (PK5 report, budget run 3): bit-identical
//! outputs at TP1 / TP2 / TP4 / TP8, but no faster (TP4 / TP8 equal at ~14 us per stage, TP1 4 lanes 25 -> 36 us),
//! so the prefill entry stays.

use crate::dsv41::fwd::dev::Gpu;
use crate::dsv41::gpu::attn::{self, AttnCall};
use anyhow::Result;

/// The entry the stage attention runs (the engagement receipt).
pub const ENTRY: &str = "d41_attn_prefill";

/// The stage attention of `call.rows` block rows.
pub fn launch(g: &Gpu, call: &AttnCall) -> Result<()> { attn::launch_prefill(g.l(), call) }
