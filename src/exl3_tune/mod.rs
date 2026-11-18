//! TUNE (PLAN/AUTOTUNE_DESIGN.md §3–§4, §7): the tunable registry, the frozen per-process table
//! and its provenance-stamped file format, the boot line, and the EXL3 stale-kernel handshake.
//!
//! Re-exported from `exl3_forward` as `tune`. The ENTRIES are declared next to their launch sites
//! in `exl3_forward.rs` (one `TunableDef` const per knob); the central `REGISTRY` below is an
//! append-only list, one line per entry, whose index must equal each entry's `slot` (checked at
//! compile time, together with the class/scope exactness contract).
//!
//! Resolution order of `get` (§4.2):
//!   1. the override flag (CLI-1: the former env alias, a diagnostic override from the options
//!      registry; printed once at boot as `OVERRIDDEN(flag: …)`),
//!   2. the tuner overlay (only armed under `--autotune`, only while a candidate graph is captured),
//!   3. the frozen table value for (entry, scope key) — S always, D only with `--tune-draft on`,
//!      N never (no numerics profile exists; output-changing profiles need the owner's consent),
//!   4. the built-in default (== today's behaviour, so a missing / mismatched table is bitwise).
//!
//! Context: `Launcher::cx()` (exl3_forward) reads the thread-local family scope set by RAII guards
//! at the family entry points (`enter(Ctx::fam(..))`); the chain helpers add the shape. No struct
//! literal changes, so launch sites elsewhere compile unchanged.
//!
//! The table is loaded and `freeze()`d BEFORE `FwdModel::load` (Load-scope knobs are read there)
//! and never changes for the life of the process, so no captured graph is ever stale. A lookup
//! before the freeze returns env/default and is counted (`early_lookups`, printed on the boot line).
//!
//! # File map (this was the single `exl3_tune.rs`; every item keeps its `exl3_tune::` path via `pub use self::<file>::*`)
//! - `mod.rs` (this file): the class / scope / family enums and `TunableDef`, the `env_*` readers.
//! - `registry.rs`: the append-only `REGISTRY`, the reserved / selftest entries and the compile-time contract checks.
//! - `state.rs`: `Ctx` and its RAII scope guard, `Key`, the frozen process table, `get`, and the `--autotune` overlay.
//! - `table.rs`: the table file format (`Table`, `Decision`, `Fingerprint`) and the build / GPU / model fingerprint.
//! - `boot.rs`: table selection, validation, `boot`, `freeze` and the boot line.
//! - `stats.rs`: paired statistics (Hodges-Lehmann, Wilcoxon), the sequential race and width split (pure CPU).
//! - `order.rs`: the balanced (Williams) arm order, the position model and the plan ordering (pure CPU).
//! - `report.rs`: table file name / writer / human report, and the EXL3 stale-kernel PTX handshake.
//! - `tests.rs`: the inline unit tests.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

mod boot;
mod order;
mod registry;
mod report;
mod state;
mod stats;
mod table;
pub use self::boot::*;
pub use self::order::*;
pub use self::registry::*;
pub use self::report::*;
pub use self::state::*;
pub use self::stats::*;
pub use self::table::*;

/// The table file format this build reads and writes.
pub const TABLE_FORMAT: &str = "gb10-exl3-tune/1";
/// The stock (uncapped) SM-clock profile. Only this profile is tuned for now (owner, 2026-09-26);
/// the clock rides the fingerprint so capped-clock profiles are a tuning RUN later, not a code change.
pub const STOCK_PROFILE_MHZ: u32 = 2400;

// ---------------------------------------------------------------------------------------------
// Classes, scopes, families (§3)
// ---------------------------------------------------------------------------------------------

/// Exactness class (§3). S = schedule (bitwise twins; the default search), D = draft-only (opt-in),
/// N = numerics contract (owner consent; m-independent scopes only), P = host policy.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Class { S, D, N, P }

/// Where a value may vary. N-class entries may only use Global / Load / Shape (m-independent,
/// AGENTS §2.4 batch invariance) — enforced at compile time below.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Scope {
    /// one value per process
    Global,
    /// fixed at FwdModel::load (weight relayouts, scratch sizing, smem opt-ins)
    Load,
    /// per graph family (m-independent within a family)
    Fam,
    /// per (K, N, bits) weight shape
    Shape,
    /// per (family, width m, regime): S and D only
    Width,
    /// per prefill chunk-size class x regime: S only
    Chunk,
}

/// Graph family of a launch (the unit a tuned value is keyed by). `Util` = launches outside every
/// family scope (copies, memsets, slot resets, primes).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Fam { Util = 0, Step = 1, Verify = 2, DraftPass = 3, DraftChain = 4, Reprime = 5, Prefill = 6 }
pub const NFAM: usize = 7;
pub const FAMS: [Fam; NFAM] = [Fam::Util, Fam::Step, Fam::Verify, Fam::DraftPass, Fam::DraftChain,
                               Fam::Reprime, Fam::Prefill];

impl Fam {
    pub fn name(self) -> &'static str {
        match self {
            Fam::Util => "Util", Fam::Step => "Step", Fam::Verify => "Verify",
            Fam::DraftPass => "DraftPass", Fam::DraftChain => "DraftChain",
            Fam::Reprime => "Reprime", Fam::Prefill => "Prefill",
        }
    }
    pub fn parse(s: &str) -> Option<Fam> {
        FAMS.iter().copied().find(|f| f.name().eq_ignore_ascii_case(s))
    }
}

impl Class {
    pub fn name(self) -> &'static str { match self { Class::S => "S", Class::D => "D", Class::N => "N", Class::P => "P" } }
}
impl Scope {
    pub fn name(self) -> &'static str {
        match self { Scope::Global => "Global", Scope::Load => "Load", Scope::Fam => "Fam",
                     Scope::Shape => "Shape", Scope::Width => "Width", Scope::Chunk => "Chunk" }
    }
}

/// One tunable. `default` == today's behaviour, so migrating a knob into the registry is a bitwise
/// no-op. `env_parse` reproduces the legacy env alias EXACTLY from its override flag (CLI-1: the
/// options registry; None = unset / not an override).
pub struct TunableDef {
    /// index in REGISTRY (compile-time checked)
    pub slot: u16,
    /// stable id, appears in the table ("moe.a1b3")
    pub id: &'static str,
    pub class: Class,
    pub scope: Scope,
    /// graph families whose capture reads it
    pub fams: &'static [Fam],
    /// candidate values (compile-time variants / args exist for each)
    pub domain: &'static [i32],
    pub default: i32,
    /// divisibility / smem / occupancy / m-range predicate for a candidate value in a context
    pub valid: fn(&Ctx, i32) -> bool,
    /// the bit-diff harness that proves the S domain ("" = DOMAIN UNPROVEN: never tuned)
    pub xcheck: &'static str,
    /// the override flag(s) for printing, e.g. "--wp11-r1" (CLI-1: were the env alias names; "" = none)
    pub env: &'static str,
    pub env_parse: fn() -> Option<i32>,
    /// bump when the body behind the knob changes (§7.2)
    pub rev: u32,
    /// owning work package
    pub wp: &'static str,
}

/// Accept every domain value (the static domain is the only restriction).
pub fn valid_any(_: &Ctx, _: i32) -> bool { true }
/// No env alias.
pub fn env_none() -> Option<i32> { None }

/// Helpers for the override flags (CLI-1: the former env aliases, same value semantics — an option
/// read here is a registry option set on the command line; unset = no override).
pub fn env_str(o: crate::opts::OptId) -> Option<String> { crate::opts::get(o) }
/// `--x off` / `--x on` style switches (default on): Some(1) unless the value is exactly "0".
pub fn env_on_unless_0(o: crate::opts::OptId) -> Option<i32> { env_str(o).map(|v| (v != "0") as i32) }
/// `--x` style switches (default off): Some(1) iff the value is exactly "1".
pub fn env_on_if_1(o: crate::opts::OptId) -> Option<i32> { env_str(o).map(|v| (v == "1") as i32) }

#[cfg(test)]
mod tests;
