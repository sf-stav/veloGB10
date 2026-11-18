//! TUNE T1 (PLAN/AUTOTUNE_DESIGN.md §5): the in-round measurement core of `gb10_inference --autotune`
//! and its self-test (§5.7); TUNE T2 (§6-§8): the search, the final gate and the table writer (the
//! `Tuner`, at the end of this file). A child module of `exl3_forward` (declared after the launch
//! macros), so it drives the SERVED units — verify_shadow / verify_kernels / draft_block /
//! forward_step_kernels — without widening their visibility.
//!
//! # Units and the idempotence contract (§5.1)
//!
//! | unit          | launches (one graph per arm)                                       | mode |
//! |---------------|--------------------------------------------------------------------|------|
//! | `verify_dry`  | verify_shadow -> verify_kernels(m') -> taps snapshot -> xq_accept   | SR   |
//! | `draft(s)`    | draft_block(s) (the tap restore runs eagerly before each replay)   | SR   |
//! | `step`        | forward_step_kernels(m = 1)                                        | LI   |
//!
//! Declared write sets (the digest covers every row a replay can leave behind that the canonical
//! round could read; `C` = comparable with the canonical execution at a different width, `F` = the
//! full set, compared arm against arm at the same width):
//! * verify_dry(m') at position p, slot s — C: logits rows < m', argmax rows < m', taps_keep rows < m',
//!   trunk K/V rows p..p+m'-1 of every attention layer (K and V, every kv-head), trunk indexer raw-key
//!   rows p..p+m'-1; F adds accept_out[0..m'+2) and the pooled indexer blocks the rows touch (a
//!   partial block reads stale keys past the rows, so it is NOT comparable across widths).
//!   Also written (scratch, rewritten before read by the canonical verify): resid/x/qkv/chain
//!   buffers, gdn_ring rows < m', qkv_save/qkv_post/a_save/b_save rows < m', conv/PLE shadows, acc2.
//! * draft(s) at position p — C: d_dev[0..s), dconf[0..s), head K/V rows p..p+s-1, head indexer raw
//!   keys p..p+s-1; F adds the head stream (resid row 0), the last pass's draft logits row, the head
//!   pooled blocks. Also written: toks/pos/slots row 0 (re-staged by xq_draft_setup every pass).
//! * step — LI only (it advances the live GDN/conv state; never replayed).
//!
//! Canonical-last: the canonical (served) execution runs AFTER every replay of its round and its write
//! set covers every replay's (fixed chain: canonical verify m = k+1 >= m', canonical chain k >= s), so
//! a replay can never leave a trace in the trajectory — the §5.7 test 3 checks it empirically (greedy
//! tokens AND [mtp-stats] rounds / drafted / accepted identical with SR on and off).
//!
//! # SR round (§5.2, protocol v2/v3 — see the §5.2 and §5.7 amendments), hooked into the served `mtp_round` / `mtp_verify`
//! 1. `sr_before_draft` (after the tap restore + draft_meta upload): every arm's graph is ensured
//!    FIRST (a capture syncs and idles the device, so it never lands inside the timed block); then
//!    the untimed WARM-UP replay(s) of the base (tap restore, graph, digests — the same kernel
//!    sequence as a timed arm, so position 0's predecessor is a replay like everyone else's); then
//!    per arm in the round's BALANCED order — tap restore, L2 conditioning, event A, the arm's draft
//!    graph, event B, untimed digests; then the tap restore again, so the canonical chain reads
//!    exactly the served tap.
//! 2. the canonical draft chain; `sr_after_draft` digests its C set.
//! 3. the served verify uploads; `sr_before_verify`: graphs first, the base warm-up replay(s), then
//!    per arm at this round's m' (one width per round, rotating) in the balanced order — L2
//!    conditioning, event A, the arm's verify_dry graph, event B, digests; event on the canonical start.
//! 4. the canonical verify (+ commit); `sr_after_verify`: canonical end event + its C digest.
//! Events and digests are read at the next round's first hook (the served sync retired them).
//! Pairing: the base arm runs in every round, so every candidate sample has a base sample on
//! identical inputs (same routing, context and drafts); Δ = t_cand − t_base is paired.
//!
//! Why v2 (the .14 self-test, 2026-09-27, receipts selftest-1790470336 / -1790470533): the FIRST
//! verify replay of every round was ~0.6-1.1 ms slower than the next (median µs by position, m = 6:
//! 39607 / 38803 / 38598), the draft replays showed nothing, clocks steady. The L2 conditioning runs
//! before EVERY timed arm, so it is not what differs at position 0; what differs is the predecessor
//! of the whole block — the verify SR follows the canonical draft chain + the served host sync /
//! readback / PLE build / uploads (the device idles), while the draft SR follows the heavy canonical
//! verify. So the warm-up goes at the START of the block, before the first arm's conditioning (each
//! timed arm keeps its own conditioning right before event A). The v1 per-round random order did
//! not balance first positions (base first 49x vs base2 35x) and biased the paired HL; v2 walks a
//! Williams design per analysis cell (every arm in every position equally often, carry-over
//! balanced), records each replay's position (`apos`), and the analysis can remove a fitted
//! position effect (`tune::position_offsets`, `--autotune-stratify`, default on).
//!
//! # LI (§5.3)
//! `li_step`, hooked into the served `forward_step`: each plain step runs the graph of the arm the
//! balanced (Williams) schedule picks (bitwise arms: the trajectory is the same whichever runs),
//! timed with events around the graph launch — the served conditions, unpaired. The first step(s)
//! of every LI segment run the base as an untimed warm-up.
//!
//! Nothing here runs unless the `--autotune` entry point armed it (`armed()` is false in serving).
//!
//! # File map (this was the single `exl3_autotune.rs`; every item keeps its `exl3_forward::autotune::` path)
//! - `mod.rs` (this file): the arming flag, the thread-local session accessor and the served hooks (`sr_before_draft` .. `li_step`).
//! - `session.rs`: the arm / sample / session types and `impl Session` (the in-round measurement core, balanced block plans).
//! - `digest.rs`: CUDA event helpers, the LI schedule, `run_unit`, region tables and the graph-node hash / write-set digests.
//! - `passes.rs`: the self-test corpus, the MTP / plain passes, the preflight idle check and the GPU environment snapshot.
//! - `cli_main.rs`: the position-effect analysis, protocol args and the `--autotune` entry (`cli`, `cli_inner`).
//! - `tuner_defs.rs`: the T2 search types (`Target`, `Knob`, `Family`, `Tuner`), spec tables and prompts.
//! - `tuner_plan.rs` / `tuner_race.rs` / `tuner_gate.rs`: `impl Tuner` by phase: probe and plan; races, search and the prefill point; the final gate.
//! - `tune_cli.rs`: `pair_obs` and `tune_cli`, the `--autotune` driver. `tests.rs`: the inline unit tests.

use super::*;
use super::tune::{self, Ctx, Fam};
use anyhow::{bail, Result};
use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, DevicePtr};
use std::cell::RefCell;
use std::collections::HashMap;

use std::sync::atomic::{AtomicBool, Ordering};

// File map: see the `//!` list at the top. Glob imports make every moved item reachable from `mod.rs` and, through `use super::*`, from each sibling.
mod cli_main;
mod digest;
mod passes;
mod session;
mod tune_cli;
mod tuner_defs;
mod tuner_gate;
mod tuner_plan;
mod tuner_race;
use self::digest::*;
use self::passes::*;
use self::session::*;
use self::tune_cli::*;
use self::tuner_defs::*;
pub use self::cli_main::cli;
use self::cli_main::*;

// ---------------------------------------------------------------------------------------------
// Arming + the thread-local session
// ---------------------------------------------------------------------------------------------

static ARMED: AtomicBool = AtomicBool::new(false);

/// True only while the `--autotune` harness has a session installed on this process.
#[inline]
pub(crate) fn armed() -> bool { ARMED.load(Ordering::Relaxed) }

thread_local! {
    static SESSION: RefCell<Option<Session>> = RefCell::new(None);
}

fn with_session<T>(f: impl FnOnce(&mut Session) -> Result<T>) -> Result<Option<T>> {
    SESSION.with(|c| match c.borrow_mut().as_mut() {
        Some(s) => f(s).map(Some),
        None => Ok(None),
    })
}

/// The served round's own values at a hook point.
pub(crate) struct SrAt {
    pub slot: usize,
    pub k: usize,
    /// the canonical verify width (k + 1)
    pub m: usize,
    pub p: usize,
    pub tap_row: usize,
    pub qsa: bool,
    pub bucket: usize,
}

pub(crate) fn sr_before_draft(model: &FwdModel, sc: &mut Scratch, head: &DraftHead, at: &SrAt) -> Result<()> {
    with_session(|s| s.before_draft(model, sc, head, at)).map(|_| ())
}
pub(crate) fn sr_after_draft(model: &FwdModel, sc: &mut Scratch, _head: &DraftHead, at: &SrAt) -> Result<()> {
    with_session(|s| s.after_draft(model, sc, at)).map(|_| ())
}
pub(crate) fn sr_before_verify(model: &FwdModel, sc: &mut Scratch, at: &SrAt) -> Result<()> {
    with_session(|s| s.before_verify(model, sc, at)).map(|_| ())
}
pub(crate) fn sr_after_verify(model: &FwdModel, sc: &mut Scratch, at: &SrAt) -> Result<()> {
    with_session(|s| s.after_verify(model, sc, at)).map(|_| ())
}
/// LI plain step: Ok(true) = the harness launched this step's unit (the caller skips its own).
pub(crate) fn li_step(model: &FwdModel, sc: &mut Scratch, m: usize, qsa: bool, bucket: usize) -> Result<bool> {
    Ok(with_session(|s| s.li_step(model, sc, m, qsa, bucket))?.unwrap_or(false))
}

// ---------------------------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests;
