//! DISPATCH_ASSERT — the machine-checked form of `AGENTS.md` §6's rule:
//! **"a wide verify never dispatches to the prefill dequant path."**
//!
//! Spec: `PLAN/DISPATCH_ASSERT_SPEC.md` (written by the Phase-10/11-prep static session; the doc
//! lives on the `docs/phase1011-prep` branch, not on master — see the K1 report for the note).
//!
//! Why this exists (`PLAN/DISPATCH_ASSERT_SPEC.md` §0.2): the rule used to be prose + a *width*
//! assert + two opt-in prints. A ≤`MAX_VERIFY` batch that took the prefill arm passes the width
//! assert (the width is legal), is invisible to `--probe-binv` (which only sweeps widths 1..=16)
//! and shows up at worst as a step-time regression and at best not at all — the same
//! "every gate stayed green" shape as the MTP regression and the mxfp4 mojibake.
//!
//! Two failing directions, both asserted here:
//!  * `batch <= MAX_VERIFY` **⇒** the arm must not be `Prefill` (perf cliff + cuBLAS is not
//!    batch-invariant, so column 0 stops matching a decode).
//!  * `batch > MAX_VERIFY` **⇒** the arm must be `Prefill` / `Bf16Slice` / the deliberate
//!    opt-in native lane — never the fixed-shape MMA arm.
//!
//! Postures (three, mirroring `PLAN/TRIPWIRE_SPEC.md` §3.3):
//!  * **off** (default) — one relaxed load per dispatch; no record kept. Production.
//!  * **count** (`--dispatch-log=count`) — counters per (format, arm) + max batch per arm.
//!  * **assert** (`--probe-dispatch`, or `--dispatch-log=assert`) — count PLUS the invariant
//!    evaluated per record; a violation panics with the full record. Probe/gate only.
//!
//! The counter is **never** on for timing runs (rule 16): `stats_line()` prints its own posture so
//! an arm's config of record says whether it was enabled.

use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// Arms a `gemm_act` dispatch can take.
pub const ARM_MMA: u8 = 0;
pub const ARM_PREFILL: u8 = 1;
pub const ARM_BF16_BINV: u8 = 2;
pub const ARM_BF16_SLICE: u8 = 3;
pub const ARM_MXFP4_NATIVE: u8 = 4;
pub const N_ARMS: usize = 5;

pub const ARM_NAMES: [&str; N_ARMS] = ["mma", "prefill", "bf16_binv", "bf16_slice", "mxfp4_native"];

/// Weight formats.
pub const FMT_NVFP4: u8 = 0;
pub const FMT_FP8: u8 = 1;
pub const FMT_FP8_BLK: u8 = 2;
pub const FMT_BF16: u8 = 3;
pub const FMT_MXFP4: u8 = 4;
pub const N_FMTS: usize = 5;

pub const FMT_NAMES: [&str; N_FMTS] = ["nvfp4", "fp8", "fp8blk", "bf16", "mxfp4"];

pub const POSTURE_OFF: u8 = 0;
pub const POSTURE_COUNT: u8 = 1;
pub const POSTURE_ASSERT: u8 = 2;

static POSTURE: AtomicU8 = AtomicU8::new(POSTURE_OFF);

#[derive(Default)]
struct Counters {
    counts: [[u64; N_ARMS]; N_FMTS],
    max_batch: [usize; N_ARMS],
    violations: u64,
}

fn counters() -> &'static Mutex<Counters> {
    static C: OnceLock<Mutex<Counters>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(Counters::default()))
}

/// Read the posture from the environment once (`--dispatch-log=count|assert`), unless already
/// forced by `--probe-dispatch` (which sets `assert`). Diagnostics-only knob, so an env var is
/// acceptable (AGENTS §7); the user-facing surface is the `--probe-dispatch` flag.
pub fn init_from_opts() {
    if posture() != POSTURE_OFF {
        return;
    }
    match crate::opts::var(crate::opt!("dispatch-log")).ok().as_deref() {
        Some("1") | Some("count") => set_posture(POSTURE_COUNT),
        Some("assert") => set_posture(POSTURE_ASSERT),
        _ => {}
    }
}

pub fn set_posture(p: u8) { POSTURE.store(p, Relaxed); }
pub fn posture() -> u8 { POSTURE.load(Relaxed) }
pub fn posture_name() -> &'static str {
    match posture() {
        POSTURE_COUNT => "count",
        POSTURE_ASSERT => "assert",
        _ => "off",
    }
}

/// Record one dispatch. The hot path in the default posture is a single relaxed load + a
/// predictable-not-taken branch (the `--trace-wide` idiom already used in `gemm_act`).
///
/// `MAX_VERIFY` is passed in rather than imported so this module stays free of a gpu.rs dependency.
#[inline]
pub fn record(fmt: u8, batch: usize, arm: u8, max_verify: usize, site: &'static str) {
    if POSTURE.load(Relaxed) == POSTURE_OFF {
        return;
    }
    let mut c = counters().lock().unwrap_or_else(|e| e.into_inner());
    let (f, a) = (fmt as usize, arm as usize);
    if f < N_FMTS && a < N_ARMS {
        c.counts[f][a] += 1;
    }
    if a < N_ARMS && batch > c.max_batch[a] {
        c.max_batch[a] = batch;
    }
    if POSTURE.load(Relaxed) == POSTURE_ASSERT {
        let bad = (batch <= max_verify && arm == ARM_PREFILL)
            || (batch > max_verify && arm == ARM_MMA);
        if bad {
            c.violations += 1;
            let rec = format!(
                "DISPATCH_ASSERT VIOLATION: site={site} fmt={} batch={batch} arm={} \
                 (MAX_VERIFY={max_verify}) — batch<=MAX_VERIFY took the prefill dequant arm \
                 (perf cliff + cuBLAS is NOT batch-invariant), or batch>MAX_VERIFY took the \
                 fixed-shape MMA arm. AGENTS §6; PLAN/DISPATCH_ASSERT_SPEC.md.",
                FMT_NAMES.get(f).copied().unwrap_or("?"),
                ARM_NAMES.get(a).copied().unwrap_or("?"));
            drop(c);
            panic!("{rec}");
        }
    }
}

/// Counter delta for one (fmt, arm) — how the probe proves *which* arm a single call took.
pub fn count_of(fmt: u8, arm: u8) -> u64 {
    let c = counters().lock().unwrap_or_else(|e| e.into_inner());
    c.counts[fmt as usize][arm as usize]
}

pub fn violations() -> u64 {
    counters().lock().unwrap_or_else(|e| e.into_inner()).violations
}

/// `[dispatch] …` — the config-of-record line (rule 11 + spec §3.4).
pub fn stats_line() -> String {
    let c = counters().lock().unwrap_or_else(|e| e.into_inner());
    let mut parts = Vec::new();
    for f in 0..N_FMTS {
        for a in 0..N_ARMS {
            if c.counts[f][a] > 0 {
                parts.push(format!("{}.{}={}", FMT_NAMES[f], ARM_NAMES[a], c.counts[f][a]));
            }
        }
    }
    let mut maxb = Vec::new();
    for a in 0..N_ARMS {
        if c.counts.iter().any(|row| row[a] > 0) {
            maxb.push(format!("max_{}={}", ARM_NAMES[a], c.max_batch[a]));
        }
    }
    format!("[dispatch] posture={} {} | {} | violations={}",
            posture_name(),
            if parts.is_empty() { "no-dispatches".to_string() } else { parts.join(" ") },
            maxb.join(" "),
            c.violations)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The invariant's two directions, as a pure predicate over a record — the same shape the
    /// probe asserts at runtime. Direction 2 is what keeps direction 1 from being vacuous.
    fn violates(batch: usize, arm: u8, max_verify: usize) -> bool {
        (batch <= max_verify && arm == ARM_PREFILL) || (batch > max_verify && arm == ARM_MMA)
    }

    #[test]
    fn prefill_under_the_bound_is_a_violation() {
        assert!(violates(1, ARM_PREFILL, 16));
        assert!(violates(16, ARM_PREFILL, 16));
    }

    #[test]
    fn mma_over_the_bound_is_a_violation() {
        assert!(violates(17, ARM_MMA, 16));
        assert!(violates(8192, ARM_MMA, 16));
    }

    #[test]
    fn the_legal_arms_are_legal() {
        for n in 1..=16 {
            assert!(!violates(n, ARM_MMA, 16), "mma at {n}");
            assert!(!violates(n, ARM_BF16_BINV, 16), "bf16 binv at {n}");
        }
        for n in [17usize, 8192] {
            assert!(!violates(n, ARM_PREFILL, 16), "prefill at {n}");
            assert!(!violates(n, ARM_BF16_SLICE, 16), "bf16 slice at {n}");
            assert!(!violates(n, ARM_MXFP4_NATIVE, 16), "mxfp4 native at {n}");
        }
    }

    #[test]
    fn record_counts_and_does_not_panic_in_count_posture() {
        set_posture(POSTURE_COUNT);
        let before = count_of(FMT_NVFP4, ARM_MMA);
        record(FMT_NVFP4, 8, ARM_MMA, 16, "unit");
        assert_eq!(count_of(FMT_NVFP4, ARM_MMA), before + 1);
        set_posture(POSTURE_OFF);
    }
}
