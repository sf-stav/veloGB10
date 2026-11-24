//! PK3: `--dsv41-prefill-bf16-partials on|off` (opt-in, default off, D-DSV41-3: a bit-changing lever). Read by the
//! prefill (> 32 rows) launchers of both expert backends at every call (EXL3 `moe_wide.rs`, native `moe.rs`):
//! with it on, each expert's per-rank output is stored as bf16 (not f32) before the ordered combine.
//! Decode / verify calls (<= 32 rows) never read it: their arithmetic is unchanged in both modes.
//! The option is `Scope::Spmd`, so a TP node runs exactly what the head resolved.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The numeric probe of the opt-in against REFb (`--probe-dsv41-kernel --dsv41-kernel-set trellis|mxfp4 --model-dir <pack>
/// --dsv41-dump-ref <fixture>`).
#[path = "partials_probe.rs"]
pub mod probe;

static CALLS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static SAID: AtomicBool = AtomicBool::new(false);

/// Is the opt-in on for this process?
pub fn on() -> bool { crate::opts::on(crate::opt!("dsv41-prefill-bf16-partials")) }

/// A prefill launcher ran one call (`rows` rows) with bf16 partials: count it; the first one prints one notice line.
pub fn note_call(backend: &str, rows: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    if !SAID.swap(true, Ordering::Relaxed) {
        crate::boot_report::notice(&format!(
            "--dsv41-prefill-bf16-partials on: {backend} prefill expert outputs are rounded to bf16 per expert and rank before the combine \
             (changes output bits; prefill rows no longer equal decode rows bit for bit; decode and verify are unchanged)"));
    }
}

/// Prefill expert calls served with bf16 partials since the start (the engagement counter for `/metrics`).
pub fn calls() -> u64 { CALLS.load(Ordering::Relaxed) }
/// Rows of those calls.
pub fn rows() -> u64 { ROWS.load(Ordering::Relaxed) }
