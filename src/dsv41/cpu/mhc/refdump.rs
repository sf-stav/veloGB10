//! DSV4.1 OM oracle: the entry point to leg REFb's reference dumps (`tools/dsv41_ref/refdump.py`).
//!
//! The reader itself lives in `crate::dsv41::dump` ([`RefbDump`], the one reader of REFb's layout; INT3 moved it
//! there so OM, OA and SL share it). This module keeps the names OM's, SL's and the other oracle legs' code uses:
//! `RefDump` is the same type, `RefTensor` the same tensor. Generic: `RefDump::layer(l, "x_in")`.

pub use crate::dsv41::dump::RefTensor;
/// A REFb dump directory (see [`crate::dsv41::dump::RefbDump`]).
pub type RefDump = crate::dsv41::dump::RefbDump;
