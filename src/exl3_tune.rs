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

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

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

// ---------------------------------------------------------------------------------------------
// The central registry (append-only; slot == index). Entries live next to their launch sites.
// ---------------------------------------------------------------------------------------------

pub const REGISTRY: &[&TunableDef] = &[
    // T0 migrations (the knobs already on HEAD; defaults == today, env aliases honoured)
    &crate::exl3_forward::T_PDL,             // 0  S21 PDL per graph family
    &crate::exl3_forward::T_CHAIN_PAIR,      // 1  S22 paired same-input chains per shape
    &crate::exl3_forward::T_MOE_A1B3,        // 2  S2 A-once 3-bit expert entry
    &crate::exl3_forward::T_MOE_FH,          // 3  S2 fused per-expert Hadamards
    &crate::exl3_forward::T_MOE_COOP,        // 4  S2 cooperative expert stream
    &crate::exl3_forward::T_HC_RB,           // 5  S8 row-batched int8 hc mixer
    &crate::exl3_forward::T_HC_MIX4,         // 6  S9 shared inject reduction tree
    &crate::exl3_forward::T_WIDE_MT,         // 7  S24 prefill wide-M MT
    &crate::exl3_forward::T_QSA_SPLITS_DEC,  // 8  N2 (frozen: numerics)
    &crate::exl3_forward::T_WP09_OFF,        // 9  WP09 umbrella escape
    &crate::exl3_forward::T_WP09_PLE,        // 10 WP09 k-major PLE (load)
    &crate::exl3_forward::T_WP09_ARGMAX,     // 11 WP09 one-launch argmax
    &crate::exl3_forward::T_WP10_FOLD,       // 12 WP10 routing fold
    &crate::exl3_forward::T_WP10_GEMV,       // 13 WP10 GEMV lane remap
    &crate::exl3_forward::T_WP11_OFF,        // 14 WP11 umbrella escape
    &crate::exl3_forward::T_WP11_R1,         // 15 WP11 R1 L2 prefetch mode
    &crate::exl3_forward::T_WP11_R2,         // 16 WP11 R2 regridded mix
    &crate::exl3_forward::T_WP11_R3,         // 17 WP11 R3 k-major q_up (load)
    &crate::exl3_forward::T_WP11_R4,         // 18 WP11 R4 hn staging
    &crate::exl3_forward::T_WP11_CVT,        // 19 WP11 PRMT int8 convert
    &crate::exl3_forward::T_WP12_STEP,       // 20 WP12 register-resident GDN step
    &crate::exl3_forward::T_WP12_DIET,       // 21 WP12 replay-save diet
    &crate::exl3_forward::T_WP12_CONV,       // 22 WP12 nostore verify conv
    &crate::exl3_forward::T_WP12_AB,         // 23 WP12 one-launch a|b
    // T0 migrations, p3c line (w3/PSK, W3/LMH, WP04, WP13, WP20, WP27, REPRIME, WP09 nb, hc grid, ...)
    &crate::exl3_forward::T_PSK_ON,          // 24 PSK opt-in switch (p5e: default off)
    &crate::exl3_forward::T_PSK_G,           // 25 T4e persistent split-K grid G per shape (single)
    &crate::exl3_forward::T_PSK_G2,          // 26 T4e ... (pair kernel)
    &crate::exl3_forward::T_PSK_PF,          // 27 PSK trellis L2 prefetch distance
    &crate::exl3_forward::T_LMH_OFF,         // 28 W3/LMH escape
    &crate::exl3_forward::T_LMH_HAD,         // 29 W3/LMH fused output Hadamard
    &crate::exl3_forward::T_LMH_NST,         // 30 W3/LMH cp.async ring depth (5-bit head)
    &crate::exl3_forward::T_SCORE_MR,        // 31 WP04 bucket-free scorer (structural)
    &crate::exl3_forward::T_SCORE_GENERIC,   // 32 WP04 scorer generic body
    &crate::exl3_forward::T_SCORE_MR_CTAS,   // 33 S18 WP04 scorer CTAs/SM
    &crate::exl3_forward::T_TOPK_V2,         // 34 S19 WP04 top-k asc2
    &crate::exl3_forward::T_SEL_GATHER,      // 35 S17 WP04 gather variant per m
    &crate::exl3_forward::T_SEL_VEC,         // 36 S17 legacy gather float4 twin
    &crate::exl3_forward::T_DENSE_V3,        // 37 S16 WP13 dense attention v3
    &crate::exl3_forward::T_WP20_MODE,       // 38 S2 WP20 rungs
    &crate::exl3_forward::T_WP20_DIET,       // 39 S1 WP20 word diet
    &crate::exl3_wp27::T_WP27_FORKS,         // 40 WP27 capture-DAG forks per family
    &crate::exl3_wp27::T_WP27_GRAPHS,        // 41 WP27 graph classes
    &crate::exl3_wp27::T_WP27_PRIO,          // 42 WP27 node priorities
    &crate::exl3_forward::T_REPRIME_GRAPH,   // 43 REPRIME graphed device re-prime
    &crate::exl3_forward::T_DRAFT_GRAPH,     // 44 draft graphs
    &crate::exl3_forward::T_WP09_AM_NB,      // 45 S12 argmax blocks per row
    &crate::exl3_forward::T_HC_BXR,          // 46 S8 row-batched hc mixer grid
    &crate::exl3_forward::T_ROUTER_FUSED,    // 47 S10 fused router (pre-WP10 path)
    &crate::exl3_forward::T_PF_ROUTER_ROWS,  // 48 S26 prefill router rows32 twin
    &crate::exl3_forward::T_PF_MOE_GROUPED,  // 49 prefill grouped expert launches
    &crate::exl3_forward::T_PF_CONV_PAR,     // 50 prefill parallel conv
    // T1 harness
    &T_SELFTEST_SPIN,                        // 51 §5.7 known-positive probe (harness units only)
    // T0c migrations, p5e line (W4 HC / SMALL / DENSE / MOE / QSA, PQ8-v2, PFX1, DHEADP)
    &crate::exl3_forward::T_W4HC_OFF,        // 52 W4/HC umbrella escape
    &crate::exl3_forward::T_W4HC_FUSE,       // 53 W4/HC (a) regridded mixer per width
    &crate::exl3_forward::T_W4HC_MIX,        // 54 W4/HC (b2) mix fused behind the mixer
    &crate::exl3_forward::T_W4HC_INJ,        // 55 W4/HC (b1) deferred inject + norm
    &crate::exl3_forward::T_W4HC_M1,         // 56 W4/HC at m = 1
    &crate::exl3_forward::T_W4HC_G,          // 57 W4/HC grid target
    &crate::exl3_forward::T_W4HC_PF,         // 58 W4/HC phase-B L2 prefetch mode
    &crate::exl3_forward::T_W4HC_LA,         // 59 W4/HC phase-A L2 look-ahead
    &crate::exl3_forward::T_W4S_ROUTER,      // 60 W4/SMALL router fold w4
    &crate::exl3_forward::T_W4S_GDN,         // 61 W4/SMALL one-launch GDN commit
    &crate::exl3_forward::T_W4S_DATTN,       // 62 W4/SMALL dense attention acc w4
    &crate::exl3_forward::T_W4S_PLE,         // 63 W4/SMALL PLE rows_km w4m<M>
    &crate::exl3_forward::T_W4S_DRAFT,       // 64 W4/SMALL draft-head attention
    &crate::exl3_forward::dense::T_W4DENSE_ON,    // 65 W4/DENSE package (opt-in)
    &crate::exl3_forward::dense::T_W4DENSE_FIX,   // 66 W4/DENSE (b) split-K fixup
    &crate::exl3_forward::dense::T_W4DENSE_SUH,   // 67 W4/DENSE (c) fused suh
    &crate::exl3_forward::dense::T_W4DENSE_MULTI, // 68 W4/DENSE (d) attention groups
    &crate::exl3_forward::dense::T_W4DENSE_SILU,  // 69 W4/DENSE silu-down
    &crate::exl3_forward::dense::T_W4DENSE_NST,   // 70 W4/DENSE ring depth
    &crate::exl3_forward::dense::T_W4DENSE_GSAT,  // 71 W4/DENSE G-picker saturation count
    &crate::exl3_forward::T_W4MOE_PARTS,     // 72 W4/MOE package parts (opt-in)
    &crate::exl3_forward::T_W4MOE_ORDER,     // 73 W4/MOE item order
    &crate::exl3_forward::T_W4MOE_NST,       // 74 W4/MOE ring depth pin
    &crate::exl3_forward::T_W4MOE_G,         // 75 W4/MOE persistent grid pin
    &crate::exl3_forward::T_W4MOE_SHOVL,     // 76 W4/MOE (c) shared-expert overlap
    &crate::exl3_forward::T_W4QSA_ON,        // 77 W4/QSA union gather (opt-in)
    &crate::exl3_forward::T_PQ8_KVW,         // 78 PQ8 row-parallel KV writer (prefill)
    &crate::exl3_forward::T_PQ8_FLASH,       // 79 PQ8-v2 flash256 vs flash256v (prefill)
    &crate::exl3_forward::T_PQ8_GATHER,      // 80 PQ8 q8 prefill gather
    &crate::exl3_forward::T_PQ8_CAUSAL,      // 81 N: flash256c causal staging (output-changing)
    &crate::exl3_forward::T_PFX1_ROWS8,      // 82 PFX1 (d) prefill router rows8
    &crate::exl3_forward::T_DHEADP_OFF,      // 83 D: DHEADP escape
    &crate::exl3_forward::T_DHEADP_CAP,      // 84 D: DHEADP expansion cap
    &crate::exl3_forward::T_MOE_SHOVL,       // 85 A5-K1 shared-expert overlap on the default WP20 path (default on)
    &crate::exl3_forward::T_MOE_ROUTER_COAL, // 86 A5-K2 router fold with coalesced weight staging
    &crate::exl3_forward::T_QSA_SELECT,      // 87 A5-K3 latency-lean QSA top-k select (asc3)
    &T_RESERVED_88,                          // 88 reserved (w6/K4 hc.pb_pf / w6/K5 attn.dense_fill, unmerged): the integrator replaces this line
    &T_RESERVED_89,                          // 89 reserved (w6/K6 dds.ratio_floor, unmerged): the integrator replaces this line
    &crate::exl3_forward::T_MOE_GU_FOLD,     // 90 A5-K7 xq_had_suh_multi folded into the WP20 gate/up prologue
    &crate::exl3_forward::T_PF_MOE_KERNEL,   // 91 A5-P1 pipelined prefill MoE expert kernel
    &crate::exl3_forward::T_PF_MOE_MT_PE,    // 92 A5-P1 WP19 per-expert MT
    &crate::exl3_forward::T_PF_MOE_FOLD,     // 93 A5-P1 MoE glue folded into the expert epilogues
    &crate::exl3_forward::T_PREFIX_TAIL_CKPT, // 94 A5-C1 N: tail (message-boundary) prefix checkpoints (output-changing)
    &crate::exl3_forward::T_PF_HC_FUSE,      // 95 A5-P2 prefill hc inject+norm / mix fused
    &crate::exl3_forward::T_PF_GDN_SCAN,     // 96 A5-L1 chunkwise GDN prefill scan re-scheduled (tc2)
    &crate::exl3_forward::T_PF_MOE_PF2,      // 97 A5-L2 prefill MoE expert GEMMs: smem trellis ring (MT-class mask)
    &crate::exl3_forward::T_SPEC_RATIO_DHEAD, // 98 A5-L4 N: real-q (ratio) draft passes on DHEAD
];

/// Slots 88/89 placeholders (A5-K7 took 90 so the unmerged K4/K5 (88) and K6 (89) entries merge
/// without a renumber): read nothing, never tuned, never loaded from a table.
pub const T_RESERVED_88: TunableDef = TunableDef {
    slot: 88, id: "reserved.88", class: Class::P, scope: Scope::Global, fams: &[Fam::Util],
    domain: &[0], default: 0, valid: valid_any, xcheck: "", env: "", env_parse: env_none, rev: 1, wp: "A5-K7",
};
pub const T_RESERVED_89: TunableDef = TunableDef {
    slot: 89, id: "reserved.89", class: Class::P, scope: Scope::Global, fams: &[Fam::Util],
    domain: &[0], default: 0, valid: valid_any, xcheck: "", env: "", env_parse: env_none, rev: 1, wp: "A5-K7",
};

/// §5.7 test 2 (known positive): a candidate whose harness unit ends with an `xq_spin_ns(v)` node —
/// read only by the `--autotune` unit builders (never by a serving launch site), writes nothing.
pub const T_SELFTEST_SPIN: TunableDef = TunableDef {
    slot: 51, id: "selftest.spin_ns", class: Class::S, scope: Scope::Global,
    fams: &[Fam::Step, Fam::Verify, Fam::DraftChain],
    domain: &[0, 200_000], default: 0, valid: valid_any, xcheck: "trivial: the node writes nothing",
    env: "", env_parse: env_none, rev: 1, wp: "TUNE",
};

const fn domain_has(d: &[i32], v: i32) -> bool {
    let mut i = 0;
    while i < d.len() {
        if d[i] == v { return true; }
        i += 1;
    }
    false
}

// Structural exactness contract (§3): checked by the compiler, not by review.
const _: () = {
    let mut i = 0;
    while i < REGISTRY.len() {
        let t = REGISTRY[i];
        assert!(t.slot as usize == i, "tune REGISTRY: an entry's slot differs from its index");
        assert!(!matches!(t.class, Class::N) || matches!(t.scope, Scope::Global | Scope::Load | Scope::Shape),
                "tune REGISTRY: an N-class (numerics) tunable may only be keyed by shape (never by width/family/chunk)");
        assert!(!matches!(t.scope, Scope::Chunk) || matches!(t.class, Class::S),
                "tune REGISTRY: Chunk scope is S-class only");
        assert!(domain_has(t.domain, t.default), "tune REGISTRY: an entry's default is not in its domain");
        assert!(!t.fams.is_empty(), "tune REGISTRY: an entry must name the families that read it");
        i += 1;
    }
};

// ---------------------------------------------------------------------------------------------
// Context + thread-local family scope
// ---------------------------------------------------------------------------------------------

/// Launch context of a lookup. `m` = the family's width (verify width, chunk rows, 1 for a step or
/// draft pass); regime 0 = dense, 1 = sparse (QSA); shape (k, n, bits) set by the chain helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ctx {
    pub fam: Fam,
    pub m: u16,
    pub regime: u8,
    pub k: i32,
    pub n: i32,
    pub bits: u8,
    pub chunk_class: u8,
}

impl Ctx {
    pub const UTIL: Ctx = Ctx { fam: Fam::Util, m: 0, regime: 0, k: 0, n: 0, bits: 0, chunk_class: 0 };
    pub fn fam(f: Fam, m: usize) -> Ctx { Ctx { fam: f, m: m.min(u16::MAX as usize) as u16, ..Ctx::UTIL } }
    pub fn prefill(c: usize) -> Ctx { Ctx { chunk_class: chunk_class(c), ..Ctx::fam(Fam::Prefill, c) } }
    pub fn regime(self, sparse: bool) -> Ctx { Ctx { regime: sparse as u8, ..self } }
    pub fn shape(self, k: i32, n: i32, bits: i32) -> Ctx { Ctx { k, n, bits: bits.clamp(0, 255) as u8, ..self } }
}

/// Key::C class of a Chunk-scope lookup outside every prefill scope (no table decision names it).
pub const NO_CHUNK_CLASS: u8 = u8::MAX;

/// Prefill chunk-size classes: <=32, <=64, <=128, <=256, <=512, <=1024, larger.
pub fn chunk_class(c: usize) -> u8 {
    match c { 0..=32 => 0, 33..=64 => 1, 65..=128 => 2, 129..=256 => 3, 257..=512 => 4, 513..=1024 => 5, _ => 6 }
}

thread_local! {
    static CUR: RefCell<Vec<Ctx>> = RefCell::new(Vec::with_capacity(8));
}

/// RAII family scope: launches (and knob lookups) inside run with this context. Not Send.
pub struct ScopeGuard(std::marker::PhantomData<*const ()>);

/// Enter a family scope for the life of the returned guard (bind it: `let _tc = tune::enter(..)`).
pub fn enter(cx: Ctx) -> ScopeGuard {
    CUR.with(|c| c.borrow_mut().push(cx));
    ScopeGuard(std::marker::PhantomData)
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        CUR.with(|c| { c.borrow_mut().pop(); });
    }
}

/// The innermost active family scope (Ctx::UTIL outside every scope).
pub fn current() -> Ctx {
    CUR.with(|c| c.borrow().last().copied().unwrap_or(Ctx::UTIL))
}

// ---------------------------------------------------------------------------------------------
// Frozen state + lookups
// ---------------------------------------------------------------------------------------------

/// Scope key of a table value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Key {
    G,
    F(u8),
    S(i32, i32, u8),
    W(u8, u16, u8),
    C(u8, u8),
}

pub fn key_of(scope: Scope, cx: &Ctx) -> Key {
    match scope {
        Scope::Global | Scope::Load => Key::G,
        Scope::Fam => Key::F(cx.fam as u8),
        Scope::Shape => Key::S(cx.k, cx.n, cx.bits),
        Scope::Width => Key::W(cx.fam as u8, cx.m, cx.regime),
        // T0c: a Chunk knob read outside a prefill scope (a head fill, a Util launch) never picks up
        // a chunk-class decision (the default applies): only Prefill keys name a class
        Scope::Chunk => Key::C(if cx.fam == Fam::Prefill { cx.chunk_class } else { NO_CHUNK_CLASS }, cx.regime),
    }
}

struct Entry {
    env: Option<i32>,
    /// resolved value when the table holds no decision for this entry
    fast: Option<i32>,
    dec: HashMap<Key, i32>,
}

/// How the process's table was resolved (the boot line).
#[derive(Clone, Debug)]
pub struct BootStatus {
    /// "OK" | "OFF" | "FALLBACK" | "DEFAULTS"
    pub state: &'static str,
    pub reason: String,
    pub table_path: Option<String>,
    pub table_sha8: Option<String>,
    pub decisions_s: usize,
    pub decisions_d: usize,
    pub ignored: Vec<String>,
    pub profile_mhz: u32,
    pub tp: u32,
}

struct Frozen {
    entries: Vec<Entry>,
    status: BootStatus,
}

static FROZEN: OnceLock<Frozen> = OnceLock::new();
static EARLY: AtomicU64 = AtomicU64::new(0);
static ENV: OnceLock<Vec<Option<i32>>> = OnceLock::new();
static CONSULT: [[AtomicU64; NFAM]; REGISTRY.len()] =
    [const { [const { AtomicU64::new(0) }; NFAM] }; REGISTRY.len()];

fn env_cache() -> &'static [Option<i32>] {
    ENV.get_or_init(|| REGISTRY.iter().map(|t| (t.env_parse)()).collect())
}

/// The value of `t` in context `cx` (O(1); see the module doc for the resolution order).
#[inline]
pub fn get(t: &TunableDef, cx: &Ctx) -> i32 {
    let s = t.slot as usize;
    if TUNER_ARMED.load(Ordering::Relaxed) {
        if let Some(row) = CONSULT.get(s) {
            row[cx.fam as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
    let Some(f) = FROZEN.get() else {
        EARLY.fetch_add(1, Ordering::Relaxed);
        return env_cache().get(s).copied().flatten().unwrap_or(t.default);
    };
    let Some(e) = f.entries.get(s) else { return t.default; };
    if let Some(v) = e.env { return v; }
    if OVERLAY_ON.load(Ordering::Relaxed) {
        if let Some(v) = overlay_lookup(t, cx) { return v; }
    }
    if let Some(v) = e.fast { return v; }
    e.dec.get(&key_of(t.scope, cx)).copied().unwrap_or(t.default)
}

/// `get` in the current thread-local family scope.
#[inline]
pub fn get_cur(t: &TunableDef) -> i32 { get(t, &current()) }

/// Lookups that ran before the freeze (a mis-ordered boot reads defaults silently otherwise).
pub fn early_lookups() -> u64 { EARLY.load(Ordering::Relaxed) }

/// Per-family lookup counts of an entry since process start (§4.8a inertness accounting).
pub fn consulted(t: &TunableDef) -> [u64; NFAM] {
    let mut out = [0u64; NFAM];
    if let Some(row) = CONSULT.get(t.slot as usize) {
        for (i, c) in row.iter().enumerate() { out[i] = c.load(Ordering::Relaxed); }
    }
    out
}

pub fn is_frozen() -> bool { FROZEN.get().is_some() }

/// Every value a lookup of `t` can return in this process: the env alias alone when it is set;
/// else the default, the frozen table's decisions, and (under `--autotune`) the whole domain. Load
/// code uses it to pre-resolve per-value resources (smem opt-ins, raw handles) outside any capture.
pub fn reachable(t: &TunableDef) -> Vec<i32> {
    if let Some(v) = env_cache().get(t.slot as usize).copied().flatten() { return vec![v]; }
    let mut v = vec![t.default];
    if let Some(e) = FROZEN.get().and_then(|f| f.entries.get(t.slot as usize)) {
        v.extend(e.dec.values().copied());
    }
    if TUNER_ARMED.load(Ordering::SeqCst) { v.extend(t.domain.iter().copied()); }
    v.sort_unstable();
    v.dedup();
    v
}
pub fn status() -> Option<BootStatus> { FROZEN.get().map(|f| f.status.clone()) }

/// Active env overrides: (entry id, env alias, value).
pub fn env_overrides() -> Vec<(&'static str, &'static str, i32)> {
    let env = env_cache();
    REGISTRY.iter().zip(env.iter())
        .filter_map(|(t, v)| v.map(|v| (t.id, t.env, v)))
        .collect()
}

// ---- tuner overlay (armed only by the --autotune harness) ----

static OVERLAY_ON: AtomicBool = AtomicBool::new(false);
static TUNER_ARMED: AtomicBool = AtomicBool::new(false);
thread_local! {
    static OVERLAY: RefCell<Vec<(u16, Option<Key>, i32)>> = RefCell::new(Vec::new());
}

fn overlay_lookup(t: &TunableDef, cx: &Ctx) -> Option<i32> {
    OVERLAY.with(|o| {
        let o = o.borrow();
        if o.is_empty() { return None; }
        let k = key_of(t.scope, cx);
        o.iter().rev().find(|(s, key, _)| *s == t.slot && key.map_or(true, |kk| kk == k)).map(|x| x.2)
    })
}

/// Arm the overlay machinery (the `--autotune` entry point only; serving never calls it).
pub fn arm_tuner() { TUNER_ARMED.store(true, Ordering::SeqCst); }

/// A candidate assignment in force for the life of the guard (thread-local; the tuner sets it only
/// while it runs / captures one candidate unit). `key = None` applies in every context.
pub struct OverlayGuard(std::marker::PhantomData<*const ()>);

pub fn overlay(assign: &[(&TunableDef, Option<Key>, i32)]) -> anyhow::Result<OverlayGuard> {
    anyhow::ensure!(TUNER_ARMED.load(Ordering::SeqCst), "tune overlay used outside --autotune");
    for (t, _, v) in assign {
        anyhow::ensure!(domain_has(t.domain, *v), "overlay {}={} outside its domain {:?}", t.id, v, t.domain);
        anyhow::ensure!(!matches!(t.class, Class::N), "overlay: {} is N-class (numerics) — never tuned without consent", t.id);
    }
    OVERLAY.with(|o| {
        let mut o = o.borrow_mut();
        o.clear();
        o.extend(assign.iter().map(|(t, k, v)| (t.slot, *k, *v)));
    });
    OVERLAY_ON.store(true, Ordering::SeqCst);
    Ok(OverlayGuard(std::marker::PhantomData))
}

impl Drop for OverlayGuard {
    fn drop(&mut self) {
        OVERLAY.with(|o| o.borrow_mut().clear());
        OVERLAY_ON.store(false, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------------------------
// Table file format (§7.1) + fingerprint (§7.2)
// ---------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ModelFp {
    pub config_sha256: String,
    pub quant_sha256: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct GpuFp {
    pub name: String,
    pub cc: String,
    pub sms: i32,
    pub l2_bytes: i64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Fingerprint {
    pub tune_build_id: String,
    /// sha256 of src/ptx/<stem>.ptx as read from CWD (the PTX this process loads, or the PTX the
    /// shipped fatbin was compiled from)
    pub ptx_sha256: BTreeMap<String, String>,
    /// sha256 of src/ptx/<stem>.fatbin when present (CUBIN deploys); provenance + match when both sides have it
    #[serde(default)]
    pub fatbin_sha256: BTreeMap<String, String>,
    pub registry_schema: String,
    pub model: ModelFp,
    pub gpu: GpuFp,
    pub driver: String,
    pub cuda: String,
    /// tensor-parallel degree the table was tuned at (per-rank shapes change with TP)
    pub tp: u32,
    /// SM-clock profile (MHz, rounded to 100) the table was tuned at
    pub clock_mhz: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Posture {
    pub lanes: usize,
    pub mtp_max_k: usize,
    pub dds: bool,
    pub kv_fmt: u32,
    pub max_pos: usize,
    pub prefill_chunk: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct DecScope {
    #[serde(default, skip_serializing_if = "Option::is_none")] pub fam: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub m: Option<Vec<u16>>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub regime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub k: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub n: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub bits: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub chunk: Option<Vec<u8>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Decision {
    pub id: String,
    pub rev: u32,
    pub class: String,
    #[serde(default)] pub scope: DecScope,
    pub value: i32,
    #[serde(default)] pub default: Option<i32>,
    #[serde(default)] pub evidence: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Table {
    pub format: String,
    /// "shipped" (build_stable.sh) | "user" (a local `--autotune`; boot prefers it)
    #[serde(default = "origin_default")] pub origin: String,
    pub fingerprint: Fingerprint,
    pub posture: Posture,
    #[serde(default)] pub provenance: serde_json::Value,
    #[serde(default)] pub decisions: Vec<Decision>,
    #[serde(default)] pub families: serde_json::Value,
    #[serde(default)] pub confirm: serde_json::Value,
    /// sha256 of the table serialized with this field empty (`seal`); "" = unsealed (hand-written).
    /// A sealed table whose content no longer matches is refused at boot (tamper / hand edit).
    #[serde(default)] pub table_sha256: String,
}
fn origin_default() -> String { "shipped".into() }

impl Table {
    /// The sha256 a sealed table must carry: over its pretty JSON with `table_sha256` = "".
    pub fn content_sha256(&self) -> String {
        let mut t = self.clone();
        t.table_sha256 = String::new();
        sha256_hex(serde_json::to_string_pretty(&t).unwrap_or_default().as_bytes())
    }
    /// Stamp `table_sha256` (the writer's last step).
    pub fn seal(&mut self) { self.table_sha256 = self.content_sha256(); }
    /// Ok for an unsealed table or a seal that matches the content.
    pub fn seal_ok(&self) -> bool { self.table_sha256.is_empty() || self.table_sha256 == self.content_sha256() }
}

pub fn sha256_hex(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

fn sha256_file(p: &Path) -> Option<String> { std::fs::read(p).ok().map(|b| sha256_hex(&b)) }

/// sha256 over the canonical registry description (id|class|scope|fams|domain|default|rev per entry).
pub fn registry_schema() -> String {
    let mut s = String::from(TABLE_FORMAT);
    for t in REGISTRY {
        s += &format!("\n{}|{}|{}|{:?}|{:?}|{}|{}", t.id, t.class.name(), t.scope.name(),
                      t.fams.iter().map(|f| f.name()).collect::<Vec<_>>(), t.domain, t.default, t.rev);
    }
    sha256_hex(s.as_bytes())
}

/// The kernel modules the EXL3 engine loads (the fingerprint's PTX set).
pub const PTX_STEMS: [&str; 2] = ["exl3_bench", "gpu_batch"];

/// GPU identity via the driver API (cuInit is idempotent). Unqueryable fields stay empty / -1.
pub fn gpu_identity() -> (GpuFp, String) {
    use cudarc::driver::sys;
    let mut g = GpuFp { sms: -1, l2_bytes: -1, ..Default::default() };
    let mut cuda = String::new();
    unsafe {
        if sys::cuInit(0) != sys::CUresult::CUDA_SUCCESS { return (g, cuda); }
        let mut d: sys::CUdevice = 0;
        if sys::cuDeviceGet(&mut d, 0) != sys::CUresult::CUDA_SUCCESS { return (g, cuda); }
        let mut buf = [0 as std::ffi::c_char; 256];
        if sys::cuDeviceGetName(buf.as_mut_ptr(), 255, d) == sys::CUresult::CUDA_SUCCESS {
            g.name = std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().trim().to_string();
        }
        let attr = |a: sys::CUdevice_attribute| -> i64 {
            let mut v: std::ffi::c_int = -1;
            if sys::cuDeviceGetAttribute(&mut v, a, d) == sys::CUresult::CUDA_SUCCESS { v as i64 } else { -1 }
        };
        let (maj, min) = (attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                          attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR));
        g.cc = format!("{maj}.{min}");
        g.sms = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) as i32;
        g.l2_bytes = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
        let mut v: std::ffi::c_int = 0;
        if sys::cuDriverGetVersion(&mut v) == sys::CUresult::CUDA_SUCCESS {
            cuda = format!("{}.{}", v / 1000, (v % 1000) / 10);
        }
    }
    (g, cuda)
}

/// The NVIDIA kernel-module version ("580.173.02") from /proc (no process spawn), or "".
pub fn driver_version() -> String {
    std::fs::read_to_string("/proc/driver/nvidia/version").ok()
        .and_then(|s| s.lines().next().map(|l| l.to_string()))
        .and_then(|l| l.split_whitespace()
            .find(|w| w.chars().filter(|c| *c == '.').count() >= 1
                && w.chars().all(|c| c.is_ascii_digit() || c == '.')
                && w.chars().next().map_or(false, |c| c.is_ascii_digit()))
            .map(|w| w.to_string()))
        .unwrap_or_default()
}

/// This process's fingerprint (reads the PTX / fatbin files and the pack's config files).
pub fn current_fingerprint(model_dir: &str, tp: u32, clock_mhz: u32) -> Fingerprint {
    let d = model_dir.trim_end_matches('/');
    let mut ptx = BTreeMap::new();
    let mut fatbin = BTreeMap::new();
    for stem in PTX_STEMS {
        if let Some(h) = sha256_file(Path::new(&format!("src/ptx/{stem}.ptx"))) { ptx.insert(stem.to_string(), h); }
        if let Some(h) = sha256_file(Path::new(&format!("src/ptx/{stem}.fatbin"))) { fatbin.insert(stem.to_string(), h); }
    }
    let (gpu, cuda) = gpu_identity();
    Fingerprint {
        tune_build_id: env!("TUNE_BUILD_ID").to_string(),
        ptx_sha256: ptx,
        fatbin_sha256: fatbin,
        registry_schema: registry_schema(),
        model: ModelFp {
            config_sha256: sha256_file(Path::new(&format!("{d}/config.json"))).unwrap_or_default(),
            quant_sha256: sha256_file(Path::new(&format!("{d}/quantization_config.json"))).unwrap_or_default(),
        },
        gpu,
        driver: driver_version(),
        cuda,
        tp,
        clock_mhz,
    }
}

/// Strict fingerprint + posture match (clock and driver excluded — handled by the caller).
/// Err = the first mismatching field (the FALLBACK reason).
pub fn strict_match(want: &Fingerprint, have: &Fingerprint, want_p: &Posture, have_p: &Posture)
    -> Result<(), String> {
    if want.tune_build_id != have.tune_build_id { return Err("tune_build_id".into()); }
    if want.ptx_sha256 != have.ptx_sha256 { return Err("ptx_sha256".into()); }
    // a fatbin present on both sides must match (a PTX-only table still loads on a fatbin deploy)
    for (k, v) in &want.fatbin_sha256 {
        if let Some(h) = have.fatbin_sha256.get(k) { if h != v { return Err(format!("fatbin_sha256[{k}]")); } }
    }
    if want.registry_schema != have.registry_schema { return Err("registry_schema".into()); }
    if want.model != have.model { return Err("model".into()); }
    if want.gpu.name != have.gpu.name || want.gpu.cc != have.gpu.cc || want.gpu.sms != have.gpu.sms {
        return Err("gpu".into());
    }
    if want.tp != have.tp { return Err("tp".into()); }
    if want_p != have_p { return Err("posture".into()); }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Boot: select, validate, freeze, print
// ---------------------------------------------------------------------------------------------

/// `--tune-table auto|off|<path>`.
#[derive(Clone, Debug, PartialEq)]
pub enum TableSel { Auto, Off, Path(PathBuf) }

impl TableSel {
    pub fn parse(v: Option<&str>) -> anyhow::Result<TableSel> {
        Ok(match v {
            None | Some("auto") => TableSel::Auto,
            Some("off") => TableSel::Off,
            Some(p) if !p.is_empty() && !p.starts_with("--") => TableSel::Path(PathBuf::from(p)),
            Some(other) => anyhow::bail!("--tune-table must be auto, off or a table path (got '{other}')"),
        })
    }
}

pub struct BootReq<'a> {
    pub sel: TableSel,
    pub model_dir: &'a str,
    pub posture: Posture,
    /// `--tune-profile <MHz>` (None = detect, else the stock profile)
    pub profile_mhz: Option<u32>,
    /// `--tune-draft on`: also apply section D (draft-only; seeded sampled bytes may shift)
    pub draft_on: bool,
    pub tp: u32,
}

fn round100(x: f64) -> u32 { ((x / 100.0).round() * 100.0).max(0.0) as u32 }

/// Best-effort SM-clock profile of this box: the application-clock target from nvidia-smi (2 s cap),
/// rounded to 100 MHz; None when unavailable ("GPU counters are optional, never required").
pub fn detect_clock_profile() -> Option<u32> {
    let out = run_timeout("nvidia-smi", &["--query-gpu=clocks.applications.graphics", "--format=csv,noheader,nounits"],
                          std::time::Duration::from_secs(2))?;
    let v: f64 = out.lines().next()?.trim().parse().ok()?;
    let r = round100(v);
    if (1000..=3200).contains(&r) { Some(r) } else { None }
}

/// Run a command with a wall-clock cap; stdout on success, None on error / timeout (killed).
pub fn run_timeout(cmd: &str, args: &[&str], cap: std::time::Duration) -> Option<String> {
    use std::io::Read;
    let mut child = std::process::Command::new(cmd).args(args)
        .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null()).spawn().ok()?;
    let t0 = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(st)) => {
                let mut s = String::new();
                child.stdout.take()?.read_to_string(&mut s).ok()?;
                return if st.success() { Some(s) } else { None };
            }
            Ok(None) if t0.elapsed() < cap => std::thread::sleep(std::time::Duration::from_millis(20)),
            _ => { let _ = child.kill(); let _ = child.wait(); return None; }
        }
    }
}

/// The table boot picked (select_table).
pub struct Pick {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    pub table: Table,
}

/// Pure table selection (§7.2): parse every candidate file, drop the ones whose format / seal /
/// strict fingerprint + posture does not match `have`, then prefer a user-forced table over a
/// shipped one, then the nearest clock profile, then file order. Err = the FALLBACK reason (the
/// last mismatch seen, or "no table matched").
pub fn select_table(files: Vec<(PathBuf, Vec<u8>)>, have: &Fingerprint, posture: &Posture, profile: u32)
    -> Result<Pick, String> {
    select_table_x(files, have, posture, profile, false)
}

/// `select_table` with `explicit` = the table was named by `--tune-table <path>`: only then may a
/// table whose final gate did not pass (confirm.pass != true: ungated, rejected or hand-written)
/// load — `auto` never picks one (a rejected table must never serve by accident).
pub fn select_table_x(files: Vec<(PathBuf, Vec<u8>)>, have: &Fingerprint, posture: &Posture, profile: u32,
                      explicit: bool) -> Result<Pick, String> {
    if files.is_empty() { return Err("no table in ./tune".into()); }
    let mut best_reason = String::from("no table matched");
    let mut picks: Vec<Pick> = Vec::new();
    for (path, bytes) in files {
        let table: Table = match serde_json::from_slice(&bytes) {
            Ok(t) => t,
            Err(e) => { best_reason = format!("parse error in {}: {e}", path.display()); continue; }
        };
        if table.format != TABLE_FORMAT {
            best_reason = format!("format {} in {}", table.format, path.display());
            continue;
        }
        if !table.seal_ok() {
            best_reason = format!("table_sha256 ({}: content edited after sealing)", path.display());
            continue;
        }
        if !explicit && !confirm_passed(&table) {
            best_reason = format!("unconfirmed ({}: its final gate did not pass — load it only by explicit --tune-table <path>)",
                                  path.display());
            continue;
        }
        if let Err(r) = strict_match(&table.fingerprint, have, &table.posture, posture) {
            best_reason = format!("{r} ({})", path.display());
            continue;
        }
        picks.push(Pick { path, bytes, table });
    }
    picks.sort_by_key(|p| ((p.table.origin != "user") as u8,
                           (p.table.fingerprint.clock_mhz as i64 - profile as i64).abs()));
    picks.into_iter().next().ok_or(best_reason)
}

/// Did the table's final gate pass? `confirm.pass == true` (every writer since L10, including a
/// clean 0-decision run). Back-compat: a table written before L10 by a 0-decision run carries
/// `{"skipped": "no decision adopted"}` with no `pass` field — it is the built-in defaults and
/// passes too, but ONLY with zero decisions and no explicit `pass` (an explicit `pass: false`,
/// e.g. `--autotune-skip-gate` or a rejected gate, is never overridden).
pub fn confirm_passed(t: &Table) -> bool {
    match t.confirm.get("pass") {
        Some(v) => v.as_bool() == Some(true),
        None => t.decisions.is_empty()
            && t.confirm.get("skipped").and_then(|v| v.as_str()) == Some("no decision adopted"),
    }
}

fn scan_tables(sel: &TableSel) -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    match sel {
        TableSel::Off => Ok(Vec::new()),
        TableSel::Path(p) => std::fs::read(p).map(|b| vec![(p.clone(), b)])
            .map_err(|e| format!("cannot read {}: {e}", p.display())),
        TableSel::Auto => {
            let mut v: Vec<PathBuf> = match std::fs::read_dir("tune") {
                Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.path())
                    .filter(|p| p.is_file() && p.extension().map_or(false, |x| x == "json"))
                    .collect(),
                Err(_) => Vec::new(),
            };
            v.sort();
            Ok(v.into_iter().filter_map(|p| std::fs::read(&p).ok().map(|b| (p, b))).collect())
        }
    }
}

/// Validate a table's decisions against the registry; returns (per-slot decision maps, #S, #D, ignored).
fn build_decisions(t: &Table, draft_on: bool) -> (Vec<HashMap<Key, i32>>, usize, usize, Vec<String>) {
    let mut dec: Vec<HashMap<Key, i32>> = (0..REGISTRY.len()).map(|_| HashMap::new()).collect();
    let (mut ns, mut nd) = (0usize, 0usize);
    let mut ignored = Vec::new();
    for d in &t.decisions {
        let Some(def) = REGISTRY.iter().find(|r| r.id == d.id) else {
            ignored.push(format!("{} (unknown id)", d.id));
            continue;
        };
        if def.class.name() != d.class {
            ignored.push(format!("{} (class {} != registry {})", d.id, d.class, def.class.name()));
            continue;
        }
        if d.rev != def.rev {
            ignored.push(format!("{} (rev {} != {})", d.id, d.rev, def.rev));
            continue;
        }
        if def.xcheck.is_empty() {
            // §4.7.3: no bit-diff harness proves this domain — never tuned, never loaded
            ignored.push(format!("{} (DOMAIN UNPROVEN: no xcheck harness)", d.id));
            continue;
        }
        match def.class {
            Class::S => {}
            Class::D if draft_on => {}
            Class::D => { ignored.push(format!("{} (section D: needs --tune-draft on)", d.id)); continue; }
            Class::N => { ignored.push(format!("{} (section N: numerics profiles are never auto-loaded)", d.id)); continue; }
            Class::P => { ignored.push(format!("{} (class P: policy is not a table value)", d.id)); continue; }
        }
        if !domain_has(def.domain, d.value) {
            ignored.push(format!("{}={} (outside domain {:?})", d.id, d.value, def.domain));
            continue;
        }
        let keys = match decision_keys(def, &d.scope) {
            Ok(k) => k,
            Err(e) => { ignored.push(format!("{} ({e})", d.id)); continue; }
        };
        let mut ok = true;
        for k in &keys {
            let cx = ctx_of_key(k);
            if !(def.valid)(&cx, d.value) { ok = false; }
        }
        if !ok { ignored.push(format!("{}={} (fails valid())", d.id, d.value)); continue; }
        for k in keys { dec[def.slot as usize].insert(k, d.value); }
        if matches!(def.class, Class::D) { nd += 1; } else { ns += 1; }
    }
    (dec, ns, nd, ignored)
}

/// Expand a decision's scope into registry keys (lists of m / chunk classes fan out).
pub fn decision_keys(def: &TunableDef, s: &DecScope) -> Result<Vec<Key>, String> {
    let regime = |r: &Option<String>| -> Result<u8, String> {
        match r.as_deref() { None | Some("dense") => Ok(0), Some("sparse") => Ok(1), Some(o) => Err(format!("regime '{o}'")) }
    };
    Ok(match def.scope {
        Scope::Global | Scope::Load => vec![Key::G],
        Scope::Fam => {
            let f = s.fam.as_deref().and_then(Fam::parse).ok_or("Fam scope needs fam")?;
            vec![Key::F(f as u8)]
        }
        Scope::Shape => {
            let (k, n, b) = (s.k.ok_or("Shape scope needs k")?, s.n.ok_or("Shape scope needs n")?, s.bits.ok_or("Shape scope needs bits")?);
            vec![Key::S(k, n, b)]
        }
        Scope::Width => {
            let f = s.fam.as_deref().and_then(Fam::parse).ok_or("Width scope needs fam")?;
            let r = regime(&s.regime)?;
            let ms = s.m.as_ref().filter(|v| !v.is_empty()).ok_or("Width scope needs m")?;
            ms.iter().map(|&m| Key::W(f as u8, m, r)).collect()
        }
        Scope::Chunk => {
            let r = regime(&s.regime)?;
            let cs = s.chunk.as_ref().filter(|v| !v.is_empty()).ok_or("Chunk scope needs chunk")?;
            cs.iter().map(|&c| Key::C(c, r)).collect()
        }
    })
}

/// Would boot accept `value` for `def` at `key`? (the valid() predicate at the key's context)
pub fn key_valid(def: &TunableDef, key: &Key, value: i32) -> bool {
    (def.valid)(&ctx_of_key(key), value)
}

/// The overlay a table puts in force at boot: exactly `build_decisions` (class / rev / domain /
/// unproven / valid() filtering), flattened to keyed assignments. The tuner's final gate runs
/// this, so it measures the table boot will load, not the search's view of it.
pub fn table_assign(t: &Table, draft_on: bool) -> (Vec<(&'static TunableDef, Option<Key>, i32)>, Vec<String>) {
    let (dec, _, _, ignored) = build_decisions(t, draft_on);
    let mut v = Vec::new();
    for (i, m) in dec.iter().enumerate() {
        let mut keys: Vec<(&Key, &i32)> = m.iter().collect();
        keys.sort_by_key(|(k, _)| format!("{k:?}"));
        for (k, val) in keys { v.push((REGISTRY[i], Some(*k), *val)); }
    }
    (v, ignored)
}

fn ctx_of_key(k: &Key) -> Ctx {
    let f = |i: u8| FAMS.get(i as usize).copied().unwrap_or(Fam::Util);
    match *k {
        Key::G => Ctx::UTIL,
        Key::F(i) => Ctx { fam: f(i), ..Ctx::UTIL },
        Key::S(k, n, b) => Ctx { k, n, bits: b, ..Ctx::UTIL },
        Key::W(i, m, r) => Ctx { fam: f(i), m, regime: r, ..Ctx::UTIL },
        Key::C(c, r) => Ctx { fam: Fam::Prefill, chunk_class: c, regime: r, ..Ctx::UTIL },
    }
}

fn freeze_with(dec: Vec<HashMap<Key, i32>>, status: BootStatus) -> bool {
    let env = env_cache();
    let entries: Vec<Entry> = REGISTRY.iter().enumerate().map(|(i, t)| {
        let d = dec.get(i).cloned().unwrap_or_default();
        let e = env.get(i).copied().flatten();
        Entry { env: e, fast: if d.is_empty() { Some(e.unwrap_or(t.default)) } else { None }, dec: d }
    }).collect();
    FROZEN.set(Frozen { entries, status }).is_ok()
}

/// Print the one-line tune status (+ the env overrides) — the first log line of a tuned process.
fn print_boot_line(st: &BootStatus) {
    let ov = env_overrides();
    let head = match st.state {
        "OK" => format!("TUNE: {} OK ({}, profile {} MHz tp{}, S decisions {} in force, D {}; early_lookups {})",
                        st.table_sha8.as_deref().unwrap_or("?"), st.table_path.as_deref().unwrap_or("?"),
                        st.profile_mhz, st.tp, st.decisions_s,
                        if st.decisions_d > 0 { format!("{} in force (--tune-draft on)", st.decisions_d) } else { "off".into() },
                        early_lookups()),
        "OFF" => format!("TUNE: OFF ({}) — built-in defaults; early_lookups {}", st.reason, early_lookups()),
        "DEFAULTS" => format!("TUNE: DEFAULTS ({}) — built-in defaults; early_lookups {}", st.reason, early_lookups()),
        _ => format!("TUNE: FALLBACK({}) — built-in defaults for everything; early_lookups {}", st.reason, early_lookups()),
    };
    println!("{head}");
    if !st.ignored.is_empty() {
        println!("TUNE: {} table decision(s) ignored: {}", st.ignored.len(), st.ignored.join("; "));
    }
    if !ov.is_empty() {
        let s: Vec<String> = ov.iter().map(|(id, env, v)| format!("{env} -> {id}={v}")).collect();
        println!("TUNE: OVERRIDDEN(flag: {}) — override flags are diagnostics; they win over the table", s.join(", "));
    }
    println!("TUNE: registry {} entries, schema {}, tune_build_id {}",
             REGISTRY.len(), &registry_schema()[..12], env!("TUNE_BUILD_ID"));
}

/// Resolve the table for this process, freeze it, print the boot line. Call BEFORE FwdModel::load.
/// Never fails the boot: every problem is a loud FALLBACK to the built-in defaults (bitwise).
pub fn boot(req: &BootReq) {
    if FROZEN.get().is_some() {
        println!("TUNE: boot requested after the table was frozen — keeping the frozen table");
        return;
    }
    let tp = req.tp.max(1);
    let base = BootStatus {
        state: "FALLBACK", reason: String::new(), table_path: None, table_sha8: None,
        decisions_s: 0, decisions_d: 0, ignored: Vec::new(), profile_mhz: STOCK_PROFILE_MHZ, tp,
    };
    if req.sel == TableSel::Off {
        let st = BootStatus { state: "OFF", reason: "--tune-table off".into(), ..base };
        freeze_with(Vec::new(), st.clone());
        print_boot_line(&st);
        return;
    }
    let files = match scan_tables(&req.sel) {
        Ok(f) => f,
        Err(e) => {
            let st = BootStatus { reason: e, ..base };
            freeze_with(Vec::new(), st.clone());
            print_boot_line(&st);
            return;
        }
    };
    if files.is_empty() {
        let st = BootStatus { reason: "no table in ./tune".into(), ..base };
        freeze_with(Vec::new(), st.clone());
        print_boot_line(&st);
        return;
    }
    let (profile, psrc) = match req.profile_mhz {
        Some(p) => (p, "--tune-profile"),
        None => match detect_clock_profile() {
            Some(p) => (p, "nvidia-smi application clock"),
            None => (STOCK_PROFILE_MHZ, "stock default"),
        },
    };
    let have = current_fingerprint(req.model_dir, tp, profile);
    // prefer a user-forced table over a shipped one, then the nearest clock profile, then file order
    let explicit = matches!(req.sel, TableSel::Path(_));
    let pick = match select_table_x(files, &have, &req.posture, profile, explicit) {
        Ok(p) => p,
        Err(best_reason) => {
            let st = BootStatus { reason: best_reason, profile_mhz: profile, ..base };
            freeze_with(Vec::new(), st.clone());
            print_boot_line(&st);
            return;
        }
    };
    if pick.table.fingerprint.clock_mhz != profile {
        println!("TUNE: no {profile} MHz profile table ({psrc}); nearest is {} MHz — loading it",
                 pick.table.fingerprint.clock_mhz);
    }
    if pick.table.fingerprint.driver != have.driver || pick.table.fingerprint.cuda != have.cuda {
        println!("TUNE: WARNING driver/CUDA differ (table {} / {}, box {} / {}) — loading (S values are \
                  output-neutral by construction; re-tune recommended: a new driver JIT can change codegen)",
                 pick.table.fingerprint.driver, pick.table.fingerprint.cuda, have.driver, have.cuda);
    }
    let (dec, ns, nd, ignored) = build_decisions(&pick.table, req.draft_on);
    let st = BootStatus {
        state: "OK", reason: String::new(),
        table_path: Some(pick.path.display().to_string()),
        table_sha8: Some(sha256_hex(&pick.bytes)[..8].to_string()),
        decisions_s: ns, decisions_d: nd, ignored, profile_mhz: pick.table.fingerprint.clock_mhz, tp,
    };
    freeze_with(dec, st.clone());
    print_boot_line(&st);
}

/// §4.1 `freeze()`: make the resolved table immutable for the life of the process. `boot` freezes
/// what it resolved; a bare `freeze()` (nothing booted) freezes the built-in defaults. Idempotent.
pub fn freeze() { ensure_frozen("tune::freeze"); }

/// Freeze the built-in defaults if no entry point booted a table (probe / bench paths). Idempotent.
pub fn ensure_frozen(who: &str) {
    if FROZEN.get().is_some() { return; }
    let st = BootStatus {
        state: "DEFAULTS", reason: format!("{who}: this entry point loads no table"), table_path: None,
        table_sha8: None, decisions_s: 0, decisions_d: 0, ignored: Vec::new(),
        profile_mhz: STOCK_PROFILE_MHZ, tp: 1,
    };
    if freeze_with(Vec::new(), st.clone()) { print_boot_line(&st); }
}

// ---------------------------------------------------------------------------------------------
// Statistics (§6.3): Hodges–Lehmann estimates with distribution-free 95% intervals
// ---------------------------------------------------------------------------------------------

/// An estimate with its 95% interval (ms or µs — the caller's unit).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Est {
    pub n: usize,
    pub est: f64,
    pub lo: f64,
    pub hi: f64,
}

fn median_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Largest k with P(T+ <= k) <= alpha/2 under H0 for the Wilcoxon signed-rank statistic of n
/// pairs (exact DP for n <= 300, normal approximation above). None when no such k exists (n tiny).
pub fn wilcoxon_crit(n: usize, alpha: f64) -> Option<usize> {
    if n == 0 { return None; }
    let nn = n * (n + 1) / 2;
    if n <= 300 {
        // p[k] = P(T+ = k) after adding ranks 1..i (each rank in or out with prob 1/2)
        let mut p = vec![0f64; nn + 1];
        p[0] = 1.0;
        for i in 1..=n {
            let top = i * (i + 1) / 2;
            for k in (0..=top).rev() {
                let with = if k >= i { p[k - i] } else { 0.0 };
                p[k] = 0.5 * (p[k] + with);
            }
        }
        let mut cdf = 0f64;
        let mut best: Option<usize> = None;
        for (k, &pk) in p.iter().enumerate() {
            cdf += pk;
            if cdf <= alpha / 2.0 + 1e-12 { best = Some(k); } else { break; }
        }
        best
    } else {
        let z = 1.959963984540054;
        let sd = ((n * (n + 1) * (2 * n + 1)) as f64 / 24.0).sqrt();
        let k = (nn as f64 / 2.0 - z * sd - 0.5).floor();
        if k < 0.0 { None } else { Some(k as usize) }
    }
}

/// One-sample (paired) Hodges–Lehmann estimate = median of the Walsh averages (d_i + d_j)/2, i <= j,
/// with the Wilcoxon signed-rank 95% interval [W_(k+1), W_(N-k)] (k = wilcoxon_crit).
pub fn hl_paired(d: &[f64]) -> Option<Est> {
    let n = d.len();
    if n == 0 || d.iter().any(|x| !x.is_finite()) { return None; }
    let mut w: Vec<f64> = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in i..n { w.push(0.5 * (d[i] + d[j])); }
    }
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let est = median_sorted(&w);
    let nn = w.len();
    let (lo, hi) = match wilcoxon_crit(n, 0.05) {
        Some(k) if k < nn / 2 => (w[k], w[nn - 1 - k]),
        _ => (f64::NEG_INFINITY, f64::INFINITY), // too few pairs for a 95% interval
    };
    Some(Est { n, est, lo, hi })
}

/// Two-sample (unpaired, LI) Hodges–Lehmann shift of y over x = median of all y_j - x_i, with the
/// Mann–Whitney 95% interval (normal approximation) [D_(k+1), D_(nm-k)].
pub fn hl_two_sample(x: &[f64], y: &[f64]) -> Option<Est> {
    let (n, m) = (x.len(), y.len());
    if n == 0 || m == 0 || x.iter().chain(y.iter()).any(|v| !v.is_finite()) { return None; }
    let mut d: Vec<f64> = Vec::with_capacity(n * m);
    for &a in x { for &b in y { d.push(b - a); } }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let est = median_sorted(&d);
    let nm = (n * m) as f64;
    let z = 1.959963984540054;
    let k = (nm / 2.0 - z * (nm * (n + m + 1) as f64 / 12.0).sqrt() - 0.5).floor();
    let (lo, hi) = if k >= 0.0 && (k as usize) < d.len() / 2 {
        (d[k as usize], d[d.len() - 1 - k as usize])
    } else { (f64::NEG_INFINITY, f64::INFINITY) };
    Some(Est { n: n.min(m), est, lo, hi })
}

// ---------------------------------------------------------------------------------------------
// T2 (§6): the decision rule, the sequential race, plan ordering, the table writer and report.
// Pure CPU — the GPU driver (exl3_autotune) feeds it paired observations.
// ---------------------------------------------------------------------------------------------

/// §6.3 practical threshold: τ = max(0.02 ms, 0.1% of the base unit's median time).
pub fn tau_ms(base_median_ms: f64) -> f64 {
    if !base_median_ms.is_finite() { return 0.02; }
    (0.001 * base_median_ms.abs()).max(0.02)
}

/// Median of a sample (None when empty).
pub fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() { return None; }
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() { return None; }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(median_sorted(&s))
}

/// The §6.3 decision on a paired estimate Δ = t_cand − t_base: adopt only when the WHOLE 95% CI
/// lies below −τ (a preference for the canonical default: hysteresis, reproducible picks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// CI_upper < −τ: faster beyond the practical threshold
    Adopt,
    /// not shown faster beyond τ: the default stays
    Default,
    /// CI_lower > 0: clearly slower
    Slower,
}

pub fn verdict(e: &Est, tau: f64) -> Verdict {
    if e.hi < -tau { Verdict::Adopt } else if e.lo > 0.0 { Verdict::Slower } else { Verdict::Default }
}

/// One paired observation: (width tag, Δ ms) — the tag is the verify width m' of the round (0 for a
/// pooled/unit-less observation), so a race pools widths while the width split can separate them.
pub type Obs = (u16, f64);

/// What a sampler returns for one candidate: fresh paired observations, or the reason it leaves
/// the race (a digest mismatch = a kernel bug on live activations; an identical graph = inert).
#[derive(Clone, Debug)]
pub enum Sampled {
    Obs(Vec<Obs>),
    Out(String),
}

/// Race parameters (§6.3): stage 1 = `n_stage1` pairs for every candidate; stage 2 = sequential
/// batches of `step` pairs up to `n_max`; confirmation = `n_confirm` pairs on fresh rounds.
#[derive(Clone, Copy, Debug)]
pub struct RaceCfg {
    pub n_stage1: usize,
    pub n_max: usize,
    pub n_confirm: usize,
    pub step: usize,
}

impl Default for RaceCfg {
    fn default() -> Self { RaceCfg { n_stage1: 8, n_max: 40, n_confirm: 24, step: 4 } }
}

#[derive(Clone, Debug)]
pub struct CandOut {
    pub idx: usize,
    pub fate: String,
    pub est: Option<Est>,
    pub obs: Vec<Obs>,
}

#[derive(Clone, Debug)]
pub struct RaceOut {
    pub cands: Vec<CandOut>,
    /// adopted AND confirmed on fresh rounds
    pub winner: Option<usize>,
    /// the best estimate after stage 2 (adopted or not) — the report's "leader"
    pub leader: Option<usize>,
    pub confirm: Option<Est>,
    pub confirm_obs: Vec<Obs>,
    pub verdict: String,
    /// sampler calls made (each call = fresh rounds on the device)
    pub calls: usize,
}

/// Pooled paired estimate of an observation list.
pub fn est_of(obs: &[Obs]) -> Option<Est> {
    let d: Vec<f64> = obs.iter().map(|o| o.1).collect();
    hl_paired(&d)
}

/// The §6.3 race over `n` candidates (indices 0..n; the base is implicit — every observation is
/// already a paired difference against it). `sample(active, want, confirm)` returns, per active
/// candidate in order, at least `want` NEW observations (fresh rounds) or its exit reason;
/// `confirm` marks the confirmation draw (the driver widens it to every served width).
///   stage 1: `n_stage1` pairs each; drop CI_lo > 0 (clearly slower); keep the best ceil(C/2).
///   stage 2: batches of `step` until each survivor stops (CI half-width < τ/2, or n >= n_max) or
///            drops (its CI lies wholly above the leader's).
///   adopt the leader only if its CI_upper < −τ; then CONFIRM on `n_confirm` fresh pairs — it
///   stands only if the confirmation CI lies wholly below 0 (two independent significant results:
///   the winner's-curse control over the ~400 comparisons of a full run).
pub fn race(n: usize, cfg: &RaceCfg, tau: f64,
            sample: &mut dyn FnMut(&[usize], usize, bool) -> Vec<Sampled>) -> RaceOut {
    let mut out = RaceOut { cands: (0..n).map(|i| CandOut { idx: i, fate: String::new(), est: None, obs: Vec::new() })
                                .collect(),
                            winner: None, leader: None, confirm: None, confirm_obs: Vec::new(),
                            verdict: String::new(), calls: 0 };
    if n == 0 {
        out.verdict = "DEFAULT (no candidates)".into();
        return out;
    }
    let mut take = |active: &[usize], want: usize, out: &mut RaceOut| {
        let r = sample(active, want, false);
        out.calls += 1;
        for (j, &i) in active.iter().enumerate() {
            match r.get(j) {
                // a call that adds no observation can never tighten a CI: out, never re-queued
                Some(Sampled::Obs(o)) if o.is_empty() => out.cands[i].fate = "OUT (no new samples this call)".into(),
                Some(Sampled::Obs(o)) => out.cands[i].obs.extend(o.iter().copied()),
                Some(Sampled::Out(why)) => out.cands[i].fate = why.clone(),
                None => out.cands[i].fate = "OUT (no samples)".into(),
            }
        }
    };
    // ---- stage 1
    let all: Vec<usize> = (0..n).collect();
    take(&all, cfg.n_stage1, &mut out);
    let mut live: Vec<usize> = Vec::new();
    for c in out.cands.iter_mut() {
        if !c.fate.is_empty() { continue; }
        c.est = est_of(&c.obs);
        match c.est {
            Some(e) if e.lo > 0.0 => c.fate = "SLOWER (stage 1: CI wholly above 0)".into(),
            Some(_) => live.push(c.idx),
            None => c.fate = "OUT (no estimate)".into(),
        }
    }
    live.sort_by(|a, b| out.cands[*a].est.unwrap().est.partial_cmp(&out.cands[*b].est.unwrap().est).unwrap());
    let keep = (n + 1) / 2;
    for &i in live.iter().skip(keep) { out.cands[i].fate = "PRUNED (stage 1: outside the best half)".into(); }
    live.truncate(keep);
    // ---- stage 2 (sequential; bounded even if a sampler under-delivers)
    let mut stopped: Vec<usize> = Vec::new();
    let max_iters = (cfg.n_max.saturating_sub(cfg.n_stage1)) / cfg.step.max(1) + 2;
    let mut iters = 0usize;
    loop {
        iters += 1;
        if iters > max_iters {
            stopped.extend(live.drain(..));
            break;
        }
        // leader = best estimate among the survivors (stopped or still sampling)
        let pool: Vec<usize> = live.iter().chain(stopped.iter()).copied().collect();
        let leader = pool.iter().copied().min_by(|a, b| out.cands[*a].est.unwrap().est
            .partial_cmp(&out.cands[*b].est.unwrap().est).unwrap());
        let Some(ld) = leader else { break };
        let lhi = out.cands[ld].est.unwrap().hi;
        let mut next: Vec<usize> = Vec::new();
        for &i in &live {
            let e = out.cands[i].est.unwrap();
            if i != ld && e.lo > lhi {
                out.cands[i].fate = "DROPPED (stage 2: CI wholly above the leader's)".into();
            } else if e.n >= cfg.n_max || (e.hi - e.lo) / 2.0 < tau / 2.0 {
                stopped.push(i);
            } else {
                next.push(i);
            }
        }
        live = next;
        if live.is_empty() { break; }
        take(&live, cfg.step, &mut out);
        let mut still: Vec<usize> = Vec::new();
        for &i in &live {
            if !out.cands[i].fate.is_empty() { continue; }
            match est_of(&out.cands[i].obs) {
                Some(e) => { out.cands[i].est = Some(e); still.push(i); }
                None => out.cands[i].fate = "OUT (no estimate)".into(),
            }
        }
        live = still;
    }
    let leader = stopped.iter().copied().min_by(|a, b| out.cands[*a].est.unwrap().est
        .partial_cmp(&out.cands[*b].est.unwrap().est).unwrap());
    out.leader = leader;
    for &i in &stopped {
        if Some(i) != leader && out.cands[i].fate.is_empty() { out.cands[i].fate = "RUNNER-UP".into(); }
    }
    let Some(ld) = leader else {
        out.verdict = "DEFAULT (no surviving candidate)".into();
        return out;
    };
    let e = out.cands[ld].est.unwrap();
    match verdict(&e, tau) {
        Verdict::Adopt => {}
        v => {
            out.cands[ld].fate = format!("LEADER, not adopted ({v:?}: CI [{:+.4}, {:+.4}] vs tau {tau:.4})", e.lo, e.hi);
            out.verdict = "DEFAULT (no candidate beats tau)".into();
            return out;
        }
    }
    // ---- confirmation on fresh rounds
    let r = sample(&[ld], cfg.n_confirm, true);
    out.calls += 1;
    match r.into_iter().next() {
        Some(Sampled::Obs(o)) => {
            out.confirm = est_of(&o);
            out.confirm_obs = o;
        }
        Some(Sampled::Out(why)) => {
            out.cands[ld].fate = why;
            out.verdict = "DEFAULT (winner left the race during confirmation)".into();
            return out;
        }
        None => {}
    }
    match out.confirm {
        Some(c) if c.hi < 0.0 => {
            out.cands[ld].fate = "ADOPTED (confirmed)".into();
            out.winner = Some(ld);
            out.verdict = "ADOPTED".into();
        }
        c => {
            out.cands[ld].fate = format!("ADOPTED then REJECTED by confirmation ({:?})", c.map(|c| (c.est, c.lo, c.hi)));
            out.verdict = "DEFAULT (confirmation failed)".into();
        }
    }
    out
}

/// Per-width view of a winner's observations (§6.2 width groups): (m', estimate, keep). A width is
/// split back to the default only when its own CI lies wholly above 0 (clearly slower there).
pub fn width_split(obs: &[Obs]) -> Vec<(u16, Option<Est>, bool)> {
    let mut ws: Vec<u16> = obs.iter().map(|o| o.0).collect();
    ws.sort_unstable();
    ws.dedup();
    ws.into_iter().map(|w| {
        let d: Vec<f64> = obs.iter().filter(|o| o.0 == w).map(|o| o.1).collect();
        let e = hl_paired(&d);
        let keep = !matches!(e, Some(x) if x.lo > 0.0);
        (w, e, keep)
    }).collect()
}

// ---------------------------------------------------------------------------------------------
// Measurement protocol v2 (§5.2 amendment, 2026-09-27): an exactly balanced arm order and the
// position model. Pure CPU. The v1 protocol re-randomized the arm order every round; the .14
// self-test receipts (selftest-1790470336 / -1790470533) showed the first verify replay of every
// round ~0.6-1.1 ms slower than the next, a random order that did not balance first positions
// (base first 49x vs base2 35x), and a paired HL biased by the imbalance.
// ---------------------------------------------------------------------------------------------

/// Rows in one period of the balanced order over `n` arms: n for even n, 2n for odd n.
pub fn williams_period(n: usize) -> usize {
    if n <= 1 { 1 } else if n % 2 == 0 { n } else { 2 * n }
}

/// Row `r` (mod the period) of a Williams design over arms 0..n: every row is a permutation; over one
/// period every arm sits at every position exactly period/n times (once for even n, twice for odd
/// n: the cyclic square plus its mirror) and every ordered pair of distinct arms is adjacent exactly
/// period/n times (first-order carry-over balance: each arm follows every other arm equally often).
pub fn williams_row(n: usize, r: usize) -> Vec<usize> {
    if n <= 1 { return vec![0; n]; }
    let r = r % williams_period(n);
    // first row 0, 1, n-1, 2, n-2, ...; row r adds r (mod n); odd n: rows n..2n are mirrored
    let row: Vec<usize> = (0..n).map(|j| {
        let a = if j == 0 { 0 } else if j % 2 == 1 { (j + 1) / 2 } else { n - j / 2 };
        (a + r % n) % n
    }).collect();
    if r >= n { row.into_iter().rev().collect() } else { row }
}

/// Per-cell cursors over the balanced order. `next(cell, n)` returns the next row for that analysis
/// cell (a unit x width x regime; the caller encodes it). Cursors are keyed by (cell, n) and persist
/// across arm-set changes, so short race phases that revisit a cell CONTINUE the design instead of
/// restarting it (a restart at a fixed row would put the same arm first every time). The start row
/// of a new cursor is derived from the seed — deterministic, and balanced over every complete period
/// from any start.
#[derive(Clone, Debug, Default)]
pub struct BalancedOrder {
    seed: u64,
    cur: HashMap<(u64, usize), usize>,
}

impl BalancedOrder {
    pub fn new(seed: u64) -> BalancedOrder { BalancedOrder { seed, cur: HashMap::new() } }

    pub fn next(&mut self, cell: u64, n: usize) -> Vec<usize> {
        let seed = self.seed;
        let c = self.cur.entry((cell, n)).or_insert_with(|| {
            let mut z = seed ^ cell.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (n as u64).rotate_left(32);
            z = (z ^ (z >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (z >> 17) as usize % williams_period(n)
        });
        let row = williams_row(n, *c);
        *c += 1;
        row
    }
}

/// One timed replay for the position model: `round` (the pairing unit: same inputs), `arm` (the
/// configuration, unique per arm-set generation), `pos` (0 = the first TIMED replay of the block).
#[derive(Clone, Copy, Debug)]
pub struct PosObs {
    pub round: u64,
    pub arm: u64,
    pub pos: usize,
    pub ms: f64,
}

/// Position offsets f[p] (ms; mean 0 over the positions that have data; unseen positions read 0) of
/// ONE analysis cell (same unit, width, regime and number of positions), fitted by median polish of
/// the additive model  t = round + arm + f(position) + noise  — the round absorbs the content
/// (routing, context, drafts), the arm its configuration. A position-stratified paired difference
/// is then (t_c - f[p_c]) - (t_b - f[p_b]).
/// * With a balanced order, arm and position are orthogonal and the fit is the position effect.
/// * It never invents an effect: the polish starts at f = 0; rounds with < 2 replays and arms seen
///   in < 2 rounds carry no information about f and are dropped; a confounded design (an arm always
///   at one position) leaves the effect with the arm (f stays ~0 = no adjustment).
pub fn position_offsets(obs: &[PosObs], n_pos: usize) -> Vec<f64> {
    let mut f = vec![0.0f64; n_pos];
    if n_pos < 2 || obs.len() < 4 { return f; }
    // informative subset: rounds with >= 2 replays, arms in >= 2 such rounds (twice, to a fixpoint-ish)
    let mut keep: Vec<PosObs> = obs.iter().copied().filter(|o| o.pos < n_pos && o.ms.is_finite()).collect();
    for _ in 0..2 {
        let mut per_round: HashMap<u64, usize> = HashMap::new();
        for o in &keep { *per_round.entry(o.round).or_default() += 1; }
        keep.retain(|o| per_round[&o.round] >= 2);
        let mut arm_rounds: HashMap<u64, std::collections::BTreeSet<u64>> = HashMap::new();
        for o in &keep { arm_rounds.entry(o.arm).or_default().insert(o.round); }
        keep.retain(|o| arm_rounds[&o.arm].len() >= 2);
    }
    if keep.len() < 4 { return f; }
    let idx = |ids: Vec<u64>| -> (Vec<usize>, usize) {
        let mut map: HashMap<u64, usize> = HashMap::new();
        let v: Vec<usize> = ids.into_iter().map(|x| { let n = map.len(); *map.entry(x).or_insert(n) }).collect();
        let n = map.len();
        (v, n)
    };
    let (ri, nr) = idx(keep.iter().map(|o| o.round).collect());
    let (ai, na) = idx(keep.iter().map(|o| o.arm).collect());
    let group = |ix: &[usize], n: usize| -> Vec<Vec<usize>> {
        let mut g = vec![Vec::new(); n];
        for (i, &k) in ix.iter().enumerate() { g[k].push(i); }
        g
    };
    let (gr, ga) = (group(&ri, nr), group(&ai, na));
    let gp = group(&keep.iter().map(|o| o.pos).collect::<Vec<_>>(), n_pos);
    let (mut mu, mut beta) = (vec![0.0f64; nr], vec![0.0f64; na]);
    let med = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        median_sorted(&v)
    };
    for _ in 0..16 {
        for (r, g) in gr.iter().enumerate() {
            mu[r] = med(g.iter().map(|&i| keep[i].ms - beta[ai[i]] - f[keep[i].pos]).collect());
        }
        for (a, g) in ga.iter().enumerate() {
            beta[a] = med(g.iter().map(|&i| keep[i].ms - mu[ri[i]] - f[keep[i].pos]).collect());
        }
        for (p, g) in gp.iter().enumerate() {
            if g.is_empty() { continue; }
            f[p] = med(g.iter().map(|&i| keep[i].ms - mu[ri[i]] - beta[ai[i]]).collect());
        }
        let seen: Vec<usize> = (0..n_pos).filter(|&p| !gp[p].is_empty()).collect();
        let c = seen.iter().map(|&p| f[p]).sum::<f64>() / seen.len().max(1) as f64;
        for &p in &seen { f[p] -= c; }
    }
    f
}

/// One family of the search plan (a knob, or a declared interacting pair) with its ledger value.
#[derive(Clone, Debug)]
pub struct PlanItem {
    pub name: String,
    /// expected realizable gain per round (ms, the §10 mid estimate)
    pub ev_ms: f64,
    /// estimated tuning seconds (candidates x rounds x round cost)
    pub est_s: f64,
}

/// §6.2 ledger-weighted order: descending expected value density ev / seconds (stable on ties).
pub fn order_by_ev(items: &mut [PlanItem]) {
    items.sort_by(|a, b| {
        let da = a.ev_ms / a.est_s.max(1e-3);
        let db = b.ev_ms / b.est_s.max(1e-3);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// §6.6 budget planner: the prefix of `items` (in order) whose cumulative estimate fits
/// `budget_s - spent_s`; the rest are `UNTUNED (budget)`.
pub fn plan_fits(items: &[PlanItem], spent_s: f64, budget_s: f64) -> usize {
    let mut t = spent_s;
    for (i, it) in items.iter().enumerate() {
        t += it.est_s;
        if t > budget_s { return i; }
    }
    items.len()
}

/// The table's canonical file name: tune/<model8>.<posture8>.tp<N>.<MHz>.json (the posture hash
/// keeps a 4K-context table from shadowing a 256K one; boot scans ./tune/*.json either way).
pub fn table_file_name(t: &Table) -> String {
    let m8 = t.fingerprint.model.config_sha256.get(..8).unwrap_or("nomodel").to_string();
    let p8 = sha256_hex(serde_json::to_string(&t.posture).unwrap_or_default().as_bytes())[..8].to_string();
    format!("tune/{m8}.{p8}.tp{}.{}.json", t.fingerprint.tp, t.fingerprint.clock_mhz)
}

/// Seal and write a table (pretty JSON); returns (path, file sha256).
pub fn write_table(t: &mut Table, path: &str) -> anyhow::Result<String> {
    t.seal();
    let s = serde_json::to_string_pretty(t)?;
    if let Some(dir) = Path::new(path).parent() {
        if !dir.as_os_str().is_empty() { std::fs::create_dir_all(dir)?; }
    }
    std::fs::write(path, s.as_bytes())?;
    Ok(sha256_hex(s.as_bytes()))
}

/// `--autotune-report <table>`: the human view — provenance, decisions with Δ / CI / runner-up,
/// the family statuses (INERT, UNPROVEN, UNTUNED (budget), OVERRIDDEN, ...), the final LI gate.
pub fn report(t: &Table, raw: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let fp = &t.fingerprint;
    let _ = writeln!(s, "TUNE TABLE {} (file sha256 {}, seal {})", TABLE_FORMAT, sha256_hex(raw),
                     if t.table_sha256.is_empty() { "none".to_string() }
                     else if t.seal_ok() { format!("{} OK", &t.table_sha256[..12.min(t.table_sha256.len())]) }
                     else { "MISMATCH (edited after sealing: boot refuses it)".to_string() });
    let _ = writeln!(s, "  origin {} | tune_build_id {} | registry {} | tp {} | clock profile {} MHz",
                     t.origin, fp.tune_build_id, fp.registry_schema.get(..12).unwrap_or(""), fp.tp, fp.clock_mhz);
    let _ = writeln!(s, "  gpu {} cc {} sms {} | driver {} cuda {} | model config {}",
                     fp.gpu.name, fp.gpu.cc, fp.gpu.sms, fp.driver, fp.cuda, fp.model.config_sha256.get(..12).unwrap_or(""));
    let _ = writeln!(s, "  posture {}", serde_json::to_string(&t.posture).unwrap_or_default());
    let p = &t.provenance;
    let g = |k: &str| p.get(k).map(|v| v.to_string()).unwrap_or_else(|| "-".into());
    let _ = writeln!(s, "  provenance: commit {} dirty {} binary {} box {} date {} elapsed_min {} budget_min {}",
                     g("commit"), g("dirty"), g("binary_sha256"), g("box"), g("date"), g("elapsed_min"), g("budget_min"));
    if let Some(nf) = p.get("noise_floor_ms") { let _ = writeln!(s, "  noise floor (ms): {nf}"); }
    if let Some(pr) = p.get("protocol") { let _ = writeln!(s, "  measurement protocol: {pr}"); }
    // §5.2 v2: the replay-position effect the run measured (per unit x width x regime x positions)
    if let Some(pe) = p.get("position_effect").and_then(|v| v.as_array()) {
        let _ = writeln!(s, "POSITION EFFECT ({} cell(s); ms; position 0 = the first timed replay after the warm-up):", pe.len());
        let arr = |v: Option<&serde_json::Value>| v.and_then(|x| x.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().map_or("-".into(), |x| format!("{x:+.3}"))).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        let arr_abs = |v: Option<&serde_json::Value>| v.and_then(|x| x.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().map_or("-".into(), |x| format!("{x:.3}"))).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        for c in pe {
            let g = |k: &str| c.get(k).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_else(|| "-".into());
            let _ = writeln!(s, "  {:<10} m{:<2} {:<6} n_pos {:<2} rounds {:<5} warm-up {} | median by position [{}] | fitted offsets [{}] | spread {}",
                             g("unit"), g("m"), g("regime"), g("n_pos"), g("rounds"),
                             c.get("warmup_median_ms").and_then(|v| v.as_f64()).map_or("-".into(), |x| format!("{x:.3}")),
                             arr_abs(c.get("median_ms_by_pos")), arr(c.get("offsets_ms")),
                             c.get("spread_ms").and_then(|v| v.as_f64()).map_or("-".into(), |x| format!("{x:.3}")));
        }
    }
    let _ = writeln!(s, "DECISIONS ({}):", t.decisions.len());
    for d in &t.decisions {
        let ev = &d.evidence;
        let f = |k: &str| ev.get(k).and_then(|v| v.as_f64());
        let ci = ev.get("ci95_ms").and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().map_or("?".into(), |x| format!("{x:+.4}"))).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        let conf = ev.get("confirm").map(|c| c.to_string()).unwrap_or_default();
        let _ = writeln!(s, "  {:<22} = {:<6} (default {}) scope {} | unit {} n {} HL {} CI95 [{}] base {} ms | runner-up {} | digests ok {} | confirm {}",
                         d.id, d.value, d.default.map_or("-".into(), |v| v.to_string()), serde_json::to_string(&d.scope).unwrap_or_default(),
                         ev.get("unit").and_then(|v| v.as_str()).unwrap_or("?"),
                         ev.get("n").map(|v| v.to_string()).unwrap_or_default(),
                         f("hl_delta_ms").map_or("?".into(), |x| format!("{x:+.4}")), ci,
                         f("base_ms").map_or("?".into(), |x| format!("{x:.3}")),
                         ev.get("runner_up").map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
                         ev.get("digest_ok").map(|v| v.to_string()).unwrap_or_default(), conf);
    }
    if let Some(fams) = t.families.as_object() {
        let _ = writeln!(s, "FAMILIES ({}):", fams.len());
        for (k, v) in fams {
            let st = v.get("status").and_then(|x| x.as_str()).unwrap_or("?");
            let _ = writeln!(s, "  {:<28} {}", k, st);
        }
    }
    if !t.confirm.is_null() {
        let _ = writeln!(s, "FINAL GATE: {}", t.confirm);
    }
    s
}

// ---------------------------------------------------------------------------------------------
// Stale-kernel handshake for the EXL3 modules (finding §14.1)
// ---------------------------------------------------------------------------------------------

/// The KERNEL_BUILD_ID baked into a PTX text's `kernel_build_id` entry (every integer literal of
/// the entry body is a candidate; the one equal to `expect` wins, else the first non-trivial one).
pub fn ptx_build_id(ptx: &str, expect: u64) -> Option<u64> {
    let at = ptx.find(".entry kernel_build_id(")?;
    let body_start = at + ptx[at..].find('{')?;
    let body_end = body_start + ptx[body_start..].find('}')?;
    let body = &ptx[body_start..body_end];
    let mut found: Vec<u64> = Vec::new();
    for tok in body.split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == '[' || c == ']') {
        // nvcc prints the 64-bit immediate as a SIGNED decimal (`mov.b64 %rd3, -7963248470597812263;`)
        let v = if let Some(h) = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X")) {
            u64::from_str_radix(h.trim_end_matches(['U', 'L', 'u', 'l']), 16).ok()
        } else if let Some(d) = tok.strip_prefix('-').filter(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit())) {
            d.parse::<u64>().ok().filter(|&x| x <= 1u64 << 63).map(|x| (x as i64).wrapping_neg() as u64)
        } else if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
            tok.parse::<u64>().ok()
        } else { None };
        if let Some(v) = v { found.push(v); }
    }
    if found.contains(&expect) { return Some(expect); }
    found.into_iter().find(|&v| v > 0xffff)
}

/// gpu_batch half of the EXL3 handshake. When the loaded `gpu_batch` module exposes
/// `kernel_build_id` (another loader registered it), the check LAUNCHES it (machine code actually
/// loaded — the GpuModel::assert_kernel_build_id contract). Otherwise the EXL3 qsa-subset load did
/// not register it, and the check reads the stamp from `src/ptx/gpu_batch.ptx` — the exact file that
/// load read (a CUBIN deploy's fatbin is pinned by its own hash check; this covers the PTX path).
pub fn assert_gpu_batch_build_id(dev: &std::sync::Arc<cudarc::driver::CudaDevice>) -> anyhow::Result<()> {
    if dev.get_func("gpu_batch", "kernel_build_id").is_some() {
        return crate::gpu::GpuModel::assert_kernel_build_id(dev, "gpu_batch");
    }
    let expect = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap_or(0);
    let ptx = std::fs::read_to_string("src/ptx/gpu_batch.ptx")
        .map_err(|e| anyhow::anyhow!("src/ptx/gpu_batch.ptx unreadable for the build-ID handshake: {e}"))?;
    match ptx_build_id(&ptx, expect) {
        Some(got) if got == expect => Ok(()),
        Some(got) => anyhow::bail!(
            "STALE KERNELS: src/ptx/gpu_batch.ptx was built from different kernel sources than this binary \
             (ptx={got:016x}, binary={expect:016x}). A deploy is the binary + src/ptx/*.ptx from ONE build. \
             Run `cargo build --release && ./scripts/build_stable.sh`."),
        None => anyhow::bail!(
            "STALE KERNELS: src/ptx/gpu_batch.ptx has no kernel_build_id entry — it predates the build-ID \
             stamp and cannot be verified against this binary. Run `cargo build --release`."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_ids_unique_and_well_formed() {
        let mut seen = std::collections::HashSet::new();
        for (i, t) in REGISTRY.iter().enumerate() {
            assert_eq!(t.slot as usize, i);
            assert!(seen.insert(t.id), "duplicate tune id {}", t.id);
            assert!(!t.id.is_empty() && !t.id.contains(char::is_whitespace), "bad id {:?}", t.id);
            assert!(domain_has(t.domain, t.default));
            let _ = (t.env_parse)(); // must not panic
        }
    }

    /// T0 migration exactness: for every legacy env alias and a spread of values (unset, "0", "1",
    /// "2", "", junk), the registry value (env parse, else default) reproduces the pre-registry
    /// comparison bit for bit.
    /// Tests that set process env vars hold this lock (cargo runs tests on parallel threads).
    pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn legacy_env_semantics_are_exact() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::exl3_forward as f;
        let vals: [Option<&str>; 16] = [None, Some("0"), Some("1"), Some("2"), Some(""), Some("yes"),
                                        Some("off"), Some("8"), Some("33"), Some("4"), Some(" 2 "),
                                        Some("generic"), Some("legacy"), Some("hg1"), Some("hg3"), Some("3")];
        // the old wp_env_flag(name, default_on = true)
        fn flag(v: Option<&str>) -> i32 { match v { Some("0") | Some("off") => 0, _ => 1 } }
        // (def, old semantics of the raw env value -> the registry-equivalent i32)
        type Old = fn(Option<&str>) -> i32;
        let cases: Vec<(&TunableDef, Old)> = vec![
            (&f::T_MOE_FH, |v| (v == Some("1")) as i32),
            (&f::T_HC_MIX4, |v| (v != Some("0")) as i32),
            (&f::T_WP09_OFF, |v| (v == Some("1")) as i32),
            (&f::T_WP09_PLE, |v| (v != Some("0")) as i32),
            (&f::T_WP09_ARGMAX, |v| (v != Some("0")) as i32),
            (&f::T_HC_RB, |v| (v != Some("0")) as i32),
            (&f::T_WP11_OFF, |v| (v == Some("1")) as i32),
            (&f::T_WP11_R1, |v| match v { Some("0") => 0, Some("2") => 2, _ => 1 }),
            (&f::T_WP11_R2, |v| (v != Some("0")) as i32),
            (&f::T_WP11_R3, |v| (v != Some("0")) as i32),
            (&f::T_WP11_R4, |v| (v != Some("0")) as i32),
            (&f::T_WP11_CVT, |v| (v != Some("0")) as i32), // value 0 == i2f (old: CVT == "0")
            (&f::T_WP10_FOLD, |v| (v != Some("1")) as i32),
            (&f::T_WP10_GEMV, |v| (v != Some("0")) as i32),
            (&f::T_PDL, |v| (v == Some("1")) as i32),
            (&f::T_MOE_A1B3, |v| (v != Some("0")) as i32),
            (&f::T_CHAIN_PAIR, |v| (v != Some("0")) as i32),
            (&f::T_MOE_COOP, |v| v.map(|s| s != "0").unwrap_or(false) as i32),
            (&f::T_QSA_SPLITS_DEC, |v| v.and_then(|s| s.parse::<usize>().ok())
                .filter(|n| (1..=32).contains(n)).unwrap_or(32) as i32),
            // p3c line
            (&f::T_PSK_PF, |v| v.and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(1).clamp(0, 127)),
            // psk.g: any set value hands the decision to the old --psk-g parse (marker -1)
            (&f::T_PSK_G, |v| if v.is_some() { -1 } else { 0 }),
            (&f::T_PSK_G2, |v| if v.is_some() { -1 } else { 0 }),
            (&f::T_LMH_OFF, |v| (v == Some("1")) as i32),
            (&f::T_LMH_HAD, |v| (v != Some("0")) as i32),
            (&f::T_LMH_NST, |v| match v { Some("4") => 4, Some("8") => 8, _ => 6 }),
            (&f::T_SCORE_MR, flag),
            (&f::T_SCORE_GENERIC, |v| (v == Some("generic")) as i32),
            (&f::T_SCORE_MR_CTAS, |v| v.and_then(|s| s.parse::<u32>().ok())
                .filter(|n| (1..=24).contains(n)).unwrap_or(4) as i32),
            (&f::T_TOPK_V2, flag),
            (&f::T_SEL_GATHER, |v| match v { Some("legacy") | Some("0") => 0, Some("hg1") | Some("1") => 1,
                                             Some("hg3") | Some("3") => 3, _ => -1 }),
            (&f::T_SEL_VEC, |v| (v != Some("0")) as i32),
            (&f::T_DENSE_V3, |v| (v != Some("0")) as i32),
            (&f::T_WP20_DIET, |v| (v != Some("0")) as i32),
            (&f::T_ROUTER_FUSED, |v| (v != Some("0")) as i32),
            (&f::T_PF_ROUTER_ROWS, |v| (v != Some("0")) as i32),
            (&f::T_PF_MOE_GROUPED, |v| (v != Some("0")) as i32),
            (&f::T_PF_CONV_PAR, |v| v.is_none() as i32), // the old `.is_err()` (set = serial)
            (&f::T_REPRIME_GRAPH, |v| (v != Some("0")) as i32),
            (&f::T_DRAFT_GRAPH, |v| (v != Some("0")) as i32),
            (&crate::exl3_wp27::T_WP27_PRIO, |v| (v == Some("1")) as i32),
        ];
        for (def, old) in cases {
            let id = crate::opts::lookup(def.env).unwrap_or_else(|| panic!("{}: no option {}", def.id, def.env));
            let saved = crate::opts::get(id);
            for v in vals {
                crate::opts::test_set(id, v);
                let new = (def.env_parse)().unwrap_or(def.default);
                assert_eq!(new, old(v), "{} ({}={:?})", def.id, def.env, v);
            }
            crate::opts::test_set(id, saved.as_deref());
        }
        // --wp12-off parts (the parse is cached per process; test the pure parser)
        let all = 1 | 2 | 4 | 8;
        for (v, want) in [("1", 0u8), ("all", 0), ("0", all), ("", all), ("step", all & !1), ("diet,conv", all & !6),
                          ("ab", all & !8), ("step, ab", all & !9), ("bogus", all)] {
            assert_eq!(f::wp12_mask_of(v), want, "--wp12-off={v:?}");
        }
        // WP20: two variables, the old precedence (OFF=1 -> 0, else EPI=0 -> 1, else 2)
        let two: [Option<&str>; 4] = [None, Some("0"), Some("1"), Some("x")];
        for off in two {
            for epi in two {
                for (k, v) in [(crate::opt!("wp20-off"), off), (crate::opt!("wp20-epi"), epi)] {
                    crate::opts::test_set(k, v)
                }
                let old = if off == Some("1") { 0 } else if epi == Some("0") { 1 } else { 2 };
                let new = (f::T_WP20_MODE.env_parse)().unwrap_or(f::T_WP20_MODE.default);
                assert_eq!(new, old, "--wp20-off={off:?} --wp20-epi={epi:?}");
            }
        }
        crate::opts::unset(crate::opt!("wp20-off"));
        crate::opts::unset(crate::opt!("wp20-epi"));
        // WP27: OFF=1 zeroes forks + graphs; FORKS / GRAPHS lists (ple masked); unset -> defaults
        use crate::exl3_wp27 as w;
        for off in [None, Some("1"), Some("0")] {
            for forks in [None, Some("moe"), Some("gdn,ple"), Some("ple"), Some("none"), Some("all")] {
                for graphs in [None, Some("verify"), Some("step,draft")] {
                    for (k, v) in [(crate::opt!("wp27-off"), off), (crate::opt!("wp27-forks"), forks), (crate::opt!("wp27-graphs"), graphs)] {
                        crate::opts::test_set(k, v)
                    }
                    let parse = |s: Option<&str>, names: &[(&str, i32)], all: i32| -> i32 {
                        match s { None => all, Some(s) => s.split(',').map(|t| t.trim())
                            .map(|t| if t == "all" { all } else { names.iter().find(|n| n.0 == t).map_or(0, |n| n.1) })
                            .fold(0, |a, b| a | b) }
                    };
                    let (old_f, old_g) = if off == Some("1") { (0, 0) } else {
                        (parse(forks, &[("moe", 1), ("gdn", 2), ("ple", 4)], 7) & !4,
                         parse(graphs, &[("verify", 1), ("draft", 2), ("step", 4)], 7))
                    };
                    let nf = (w::T_WP27_FORKS.env_parse)().unwrap_or(w::T_WP27_FORKS.default);
                    let ng = (w::T_WP27_GRAPHS.env_parse)().unwrap_or(w::T_WP27_GRAPHS.default);
                    assert_eq!((nf, ng), (old_f, old_g), "OFF={off:?} FORKS={forks:?} GRAPHS={graphs:?}");
                }
            }
        }
        for k in [crate::opt!("wp27-off"), crate::opt!("wp27-forks"), crate::opt!("wp27-graphs")] { crate::opts::unset(k); }
        // p5e PSK: opt-in (--psk == "1") unless the escape (--psk-off == "1"); unset -> default off
        let pv: [Option<&str>; 5] = [None, Some("1"), Some("0"), Some(""), Some("yes")];
        for on in pv {
            for esc in pv {
                for (k, v) in [(crate::opt!("psk"), on), (crate::opt!("psk-off"), esc)] {
                    crate::opts::test_set(k, v)
                }
                let old = (on == Some("1") && esc != Some("1")) as i32;
                let new = (f::T_PSK_ON.env_parse)().unwrap_or(f::T_PSK_ON.default);
                assert_eq!(new, old, "--psk={on:?} --psk-off={esc:?}");
            }
        }
        for k in [crate::opt!("psk"), crate::opt!("psk-off")] { crate::opts::unset(k); }
        // p5e prefill.wide_mt: the RESOLVED MT (dense wide chains: value 0 = 4) equals PFX1's
        // wide_mt_override().unwrap_or(4) for every alias value (trimmed; only 1|2|4|8 count)
        let saved = crate::opts::get(crate::opt!("exl3-wide-mt"));
        for v in vals.iter().copied().chain([Some(" 8 "), Some("16"), Some("-1")]) {
            match v { Some(s) => crate::opts::set(crate::opt!("exl3-wide-mt"), s), None => crate::opts::unset(crate::opt!("exl3-wide-mt")) }
            let old = match v.map(|s| s.trim().parse::<usize>()) { Some(Ok(mt @ (1 | 2 | 4 | 8))) => mt as i32, _ => 4 };
            let raw = (f::T_WIDE_MT.env_parse)().unwrap_or(f::T_WIDE_MT.default);
            let new = if matches!(raw, 1 | 2 | 4 | 8) { raw } else { 4 };
            assert_eq!(new, old, "--exl3-wide-mt={v:?}");
        }
        match saved { Some(s) => crate::opts::set(crate::opt!("exl3-wide-mt"), s), None => crate::opts::unset(crate::opt!("exl3-wide-mt")) }
    }

    #[test]
    fn wilcoxon_critical_values_match_the_table() {
        // two-sided alpha 0.05 signed-rank critical values (reject when T <= c)
        for (n, c) in [(6usize, 0usize), (7, 2), (8, 3), (9, 5), (10, 8), (12, 13), (15, 25), (20, 52)] {
            assert_eq!(wilcoxon_crit(n, 0.05), Some(c), "n = {n}");
        }
        assert_eq!(wilcoxon_crit(5, 0.05), None);
        // the normal approximation continues the exact table smoothly
        let (a, b) = (wilcoxon_crit(300, 0.05).unwrap() as f64, wilcoxon_crit(301, 0.05).unwrap() as f64);
        assert!(b > a && b - a < 400.0);
    }

    #[test]
    fn hodges_lehmann_recovers_shifts() {
        // symmetric noise around a shift: the estimate lands on the shift and the interval covers it
        let noise: Vec<f64> = (0..60).map(|i| ((i * 37 % 61) as f64 - 30.0) / 300.0).collect();
        for shift in [0.0, 0.2, -0.05] {
            let d: Vec<f64> = noise.iter().map(|x| x + shift).collect();
            let e = hl_paired(&d).unwrap();
            assert!((e.est - shift).abs() < 0.01, "est {} vs {shift}", e.est);
            assert!(e.lo <= shift && shift <= e.hi && e.lo < e.hi);
        }
        // exact constant data: a degenerate interval at the value
        let e = hl_paired(&[0.2; 12]).unwrap();
        assert_eq!((e.est, e.lo, e.hi), (0.2, 0.2, 0.2));
        assert!(hl_paired(&[]).is_none());
        // two-sample shift
        let x: Vec<f64> = noise.iter().map(|v| 40.0 + v).collect();
        let y: Vec<f64> = noise.iter().rev().map(|v| 40.2 + v).collect();
        let e2 = hl_two_sample(&x, &y).unwrap();
        assert!((e2.est - 0.2).abs() < 0.02 && e2.lo <= 0.2 && 0.2 <= e2.hi, "{e2:?}");
    }

    /// T0c (p5e): the line's new aliases, through the registry, reproduce their pre-registry reads
    /// exactly (single-variable aliases swept; the multi-variable rules as products; the cached
    /// parses through their pure parsers).
    #[test]
    fn p5e_legacy_env_semantics_are_exact() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::exl3_forward as f;
        use crate::exl3_forward::dense as d;
        let vals: [Option<&str>; 20] = [None, Some("0"), Some("1"), Some("2"), Some(""), Some("yes"), Some("off"),
                                        Some("OFF"), Some("8"), Some("33"), Some("4"), Some(" 2 "), Some(" 36 "),
                                        Some("il"), Some("IL"), Some("causal"), Some("-1"), Some("96"), Some("512"),
                                        Some("99999999999")];
        type Old = fn(Option<&str>) -> i32;
        let on_unless_0: Old = |v| (v != Some("0")) as i32;
        let cases: Vec<(&TunableDef, Old)> = vec![
            (&f::T_W4HC_OFF, |v| (v == Some("1")) as i32),
            (&f::T_W4HC_FUSE, on_unless_0),
            (&f::T_W4HC_MIX, on_unless_0),
            (&f::T_W4HC_INJ, on_unless_0),
            (&f::T_W4HC_M1, on_unless_0),
            (&f::T_W4HC_G, |v| v.and_then(|s| s.parse::<u32>().ok()).filter(|&g| g >= 1).unwrap_or(96).min(i32::MAX as u32) as i32),
            (&f::T_W4HC_PF, |v| match v { Some("0") => 0, Some("2") => 2, _ => 1 }),
            (&f::T_W4HC_LA, |v| v.and_then(|s| s.parse::<i32>().ok()).unwrap_or(0).clamp(0, 7)),
            (&d::T_W4DENSE_FIX, on_unless_0),
            (&d::T_W4DENSE_SUH, on_unless_0),
            (&d::T_W4DENSE_MULTI, on_unless_0),
            (&d::T_W4DENSE_SILU, on_unless_0),
            (&d::T_W4DENSE_NST, |v| match v { Some("4") => 4, Some("8") => 8, _ => 6 }),
            (&d::T_W4DENSE_GSAT, |v| v.and_then(|s| s.trim().parse::<usize>().ok()).filter(|&x| x >= 1).unwrap_or(36)
                .min(i32::MAX as usize) as i32),
            (&f::T_W4MOE_ORDER, |v| matches!(v, Some("il") | Some("IL") | Some("1")) as i32),
            // 0 = mpk_plan's rule (the old w4moe_nst_env() None)
            (&f::T_W4MOE_NST, |v| match v { Some("4") => 4, Some("8") => 8, _ => 0 }),
            (&f::T_W4MOE_G, |v| v.and_then(|s| s.trim().parse::<u32>().ok()).filter(|&g| g > 0)
                .map_or(0, |g| g.min(i32::MAX as u32) as i32)),
            (&f::T_W4MOE_SHOVL, |v| (v == Some("1")) as i32),
            (&f::T_MOE_SHOVL, on_unless_0),
            (&f::T_MOE_ROUTER_COAL, on_unless_0),
            (&f::T_QSA_SELECT, on_unless_0),
            (&f::T_MOE_GU_FOLD, |v| v.map_or(6, |s| s.trim().parse::<i32>().unwrap_or(0).clamp(0, 16))), // default 6 since e2e7bd2
            (&f::T_PQ8_CAUSAL, |v| (v == Some("causal")) as i32),
            (&f::T_PREFIX_TAIL_CKPT, |v| match v { None => 1, Some("1") => 1, Some("2") => 2, _ => 0 }), // default 1 (owner 2026-09-28)
            (&f::T_DHEADP_OFF, |v| v.map_or(false, |s| !s.is_empty() && s != "0") as i32),
            (&f::T_DHEADP_CAP, |v| v.and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(256).min(i32::MAX as usize) as i32),
            (&f::T_SPEC_RATIO_DHEAD, |v| v.map_or(true, |s| !s.is_empty() && s != "0") as i32), // default 1 (owner 2026-09-28)
        ];
        for (def, old) in cases {
            let id = crate::opts::lookup(def.env).unwrap_or_else(|| panic!("{}: no option {}", def.id, def.env));
            let saved = crate::opts::get(id);
            for v in vals {
                crate::opts::test_set(id, v);
                let new = (def.env_parse)().unwrap_or(def.default);
                assert_eq!(new, old(v), "{} ({}={:?})", def.id, def.env, v);
            }
            crate::opts::test_set(id, saved.as_deref());
        }
        let set = |k: crate::opts::OptId, v: Option<&str>| crate::opts::test_set(k, v);
        let two: [Option<&str>; 6] = [None, Some("1"), Some("0"), Some(""), Some("off"), Some(" 1 ")];
        // W4/DENSE: opt-in (set, trimmed not "", "0", "off", "OFF") unless --w4dense-off == "1"
        for on in two {
            for esc in two {
                set(crate::opt!("w4dense"), on);
                set(crate::opt!("w4dense-off"), esc);
                let old = (on.is_some_and(|v| !matches!(v.trim(), "" | "0" | "off" | "OFF")) && esc != Some("1")) as i32;
                let new = (d::T_W4DENSE_ON.env_parse)().unwrap_or(d::T_W4DENSE_ON.default);
                assert_eq!(new, old, "--w4dense={on:?} --w4dense-off={esc:?}");
            }
        }
        for k in [crate::opt!("w4dense"), crate::opt!("w4dense-off")] { crate::opts::unset(k); }
        // W4/QSA: wp_env_flag(--w4qsa, off) && !wp_env_flag(--w4qsa-off, off) && no SEL_GATHER pin
        let flag0 = |v: Option<&str>| match v { None => false, Some("0") | Some("off") => false, Some(_) => true };
        for on in two {
            for esc in two {
                for pin in [None, Some("legacy"), Some("hg3"), Some("3"), Some("auto")] {
                    set(crate::opt!("w4qsa"), on);
                    set(crate::opt!("w4qsa-off"), esc);
                    set(crate::opt!("exl3-sel-gather"), pin);
                    let pinned = matches!(pin, Some("legacy") | Some("0") | Some("hg1") | Some("1") | Some("hg3") | Some("3"));
                    let old = (flag0(on) && !flag0(esc) && !pinned) as i32;
                    let new = (f::T_W4QSA_ON.env_parse)().unwrap_or(f::T_W4QSA_ON.default);
                    assert_eq!(new, old, "--w4qsa={on:?} --w4qsa-off={esc:?} SEL_GATHER={pin:?}");
                }
            }
        }
        for k in [crate::opt!("w4qsa"), crate::opt!("w4qsa-off"), crate::opt!("exl3-sel-gather")] { crate::opts::unset(k); }
        // PQ8: item i = !wp_env_flag(--pq8-off, off) && wp_env_flag(<item>, on)
        let flag1 = |v: Option<&str>| match v { None => true, Some("0") | Some("off") => false, Some(_) => true };
        for (def, name) in [(&f::T_PQ8_KVW, crate::opt!("pq8-kvw")), (&f::T_PQ8_FLASH, crate::opt!("pq8-flash")),
                            (&f::T_PQ8_GATHER, crate::opt!("pq8-gather"))] {
            for off in two {
                for item in [None, Some("0"), Some("1"), Some("off"), Some("causal")] {
                    set(crate::opt!("pq8-off"), off);
                    set(name, item);
                    let old = (!flag0(off) && flag1(item)) as i32;
                    let new = (def.env_parse)().unwrap_or(def.default);
                    assert_eq!(new, old, "{} --pq8-off={off:?} {name}={item:?}", def.id);
                }
            }
            crate::opts::unset(name);
        }
        crate::opts::unset(crate::opt!("pq8-off"));
        // W4/MOE parts (the alias parse is cached per process: its pure parser, against the old body)
        let (gu, dn, fold) = (f::MPK_GU, f::MPK_DN, f::MPK_FOLD);
        let all = gu | dn | fold;
        for (opt, off, want) in [(None, None, 0u8), (Some("1"), None, all), (Some("0"), None, 0), (Some(" off "), None, 0),
                                 (Some("yes"), None, all), (Some("1"), Some("1"), 0), (Some("1"), Some("all"), 0),
                                 (Some("1"), Some("gu"), dn), (Some("1"), Some("dn"), gu | fold),
                                 (Some("1"), Some("fold"), gu | dn), (Some("1"), Some("GU, fold"), dn),
                                 (Some("1"), Some("0"), all), (Some("1"), Some("bogus"), all), (None, Some("gu"), 0)] {
            assert_eq!(f::w4moe_parts_of(opt, off), want, "--w4moe={opt:?} --w4moe-off={off:?}");
            assert!(f::T_W4MOE_PARTS.domain.contains(&(want as i32)), "mask {want} outside w4moe.parts' domain");
        }
    }

    #[test]
    fn chunk_classes_are_monotone() {
        let mut last = 0u8;
        for c in [1usize, 16, 26, 32, 33, 72, 128, 160, 256, 512, 1024, 2048, 4096] {
            let k = chunk_class(c);
            assert!(k >= last);
            last = k;
        }
        assert_eq!(chunk_class(26), 0);
        assert_eq!(chunk_class(72), 2);
        assert_eq!(chunk_class(2048), 6);
    }

    #[test]
    fn scope_guards_nest_and_unwind() {
        assert_eq!(current(), Ctx::UTIL);
        {
            let _a = enter(Ctx::fam(Fam::Verify, 6).regime(true));
            assert_eq!(current().fam, Fam::Verify);
            assert_eq!(current().m, 6);
            {
                let _b = enter(Ctx::fam(Fam::Reprime, 3));
                assert_eq!(current().fam, Fam::Reprime);
            }
            assert_eq!(current().fam, Fam::Verify);
            assert_eq!(current().regime, 1);
        }
        assert_eq!(current(), Ctx::UTIL);
    }

    #[test]
    fn empty_table_parses_and_keys_expand() {
        let fp = Fingerprint { tp: 1, clock_mhz: 2400, ..Default::default() };
        let t = Table { format: TABLE_FORMAT.into(), origin: "shipped".into(), fingerprint: fp.clone(),
                        posture: Posture::default(), provenance: serde_json::Value::Null,
                        decisions: Vec::new(), families: serde_json::Value::Null, confirm: serde_json::Value::Null,
                        table_sha256: String::new() };
        let s = serde_json::to_string(&t).unwrap();
        let back: Table = serde_json::from_str(&s).unwrap();
        assert_eq!(back.fingerprint, fp);
        assert!(back.decisions.is_empty());
        // a minimal hand-written table (origin defaults to shipped, decisions optional)
        let min = format!(r#"{{"format":"{TABLE_FORMAT}","fingerprint":{},"posture":{}}}"#,
                          serde_json::to_string(&fp).unwrap(), serde_json::to_string(&Posture::default()).unwrap());
        let t2: Table = serde_json::from_str(&min).unwrap();
        assert_eq!(t2.origin, "shipped");
        let (dec, ns, nd, ign) = build_decisions(&t2, false);
        assert_eq!((ns, nd, ign.len()), (0, 0, 0));
        assert!(dec.iter().all(|d| d.is_empty()));
    }

    #[test]
    fn strict_match_names_the_field() {
        let a = Fingerprint { tune_build_id: "x".into(), tp: 1, clock_mhz: 2400, ..Default::default() };
        let mut b = a.clone();
        let p = Posture::default();
        assert!(strict_match(&a, &b, &p, &p).is_ok());
        b.tune_build_id = "y".into();
        assert_eq!(strict_match(&a, &b, &p, &p).unwrap_err(), "tune_build_id");
        let b2 = Fingerprint { tp: 2, ..a.clone() };
        assert_eq!(strict_match(&a, &b2, &p, &p).unwrap_err(), "tp");
        let p2 = Posture { lanes: 4, ..Posture::default() };
        assert_eq!(strict_match(&a, &a, &p, &p2).unwrap_err(), "posture");
        // the clock profile is NOT a strict field (nearest-profile selection happens in boot)
        let b3 = Fingerprint { clock_mhz: 2000, ..a.clone() };
        assert!(strict_match(&a, &b3, &p, &p).is_ok());
    }

    #[test]
    fn ptx_build_id_parses_decimal_and_hex() {
        let ptx = ".visible .entry kernel_build_id(\n .param .u64 kernel_build_id_param_0\n)\n{\n .reg .b64 %rd<4>;\n\
                   ld.param.u64 %rd1, [kernel_build_id_param_0];\n cvta.to.global.u64 %rd2, %rd1;\n\
                   mov.u64 %rd3, 12345678901234567;\n st.global.u64 [%rd2], %rd3;\n ret;\n}\n";
        assert_eq!(ptx_build_id(ptx, 12345678901234567), Some(12345678901234567));
        assert_eq!(ptx_build_id(ptx, 7), Some(12345678901234567));
        let hex = ptx.replace("12345678901234567", "0x2BDC545D6B4B87");
        assert_eq!(ptx_build_id(&hex, 0x2BDC545D6B4B87), Some(0x2BDC545D6B4B87));
        assert_eq!(ptx_build_id("no entry here", 1), None);
        // nvcc's real spelling: the u64 stamp as a signed decimal immediate
        let neg = ptx.replace("12345678901234567", "-7963248470597812263");
        assert_eq!(ptx_build_id(&neg, 0x917c_dbb5_03d8_3fd9), Some(0x917c_dbb5_03d8_3fd9));
    }

    /// The PTX this build just wrote carries THIS binary's KERNEL_BUILD_ID (no GPU needed).
    #[test]
    fn built_ptx_stamps_match_the_binary() {
        let expect = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap();
        for stem in PTX_STEMS {
            let Ok(ptx) = std::fs::read_to_string(format!("src/ptx/{stem}.ptx")) else { continue };
            assert_eq!(ptx_build_id(&ptx, expect), Some(expect), "src/ptx/{stem}.ptx stamp");
        }
    }

    // ---- T2 (pure CPU): statistics on synthetic data, the race, the table, fallback ----

    /// Deterministic pseudo-noise (xorshift -> symmetric triangular, mean 0, |x| <= amp).
    fn noise(seed: &mut u64, amp: f64) -> f64 {
        let mut u = || { *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17;
                         (*seed >> 11) as f64 / (1u64 << 53) as f64 };
        (u() + u() - 1.0) * amp
    }

    #[test]
    fn wilcoxon_interval_has_nominal_coverage_under_the_null() {
        // A/A: 400 experiments of n = 24 symmetric-noise pairs; the 95% interval must cover 0 in
        // about 95% of them (exact signed-rank bounds: coverage >= 95% by construction, a bit more)
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut cover = 0usize;
        for _ in 0..400 {
            let d: Vec<f64> = (0..24).map(|_| noise(&mut seed, 0.08)).collect();
            let e = hl_paired(&d).unwrap();
            if e.lo <= 0.0 && 0.0 <= e.hi { cover += 1; }
        }
        let rate = cover as f64 / 400.0;
        assert!((0.92..=0.995).contains(&rate), "A/A coverage {rate}");
    }

    #[test]
    fn hl_sees_a_known_positive_through_noise() {
        // the self-test's known positive: +0.200 ms on top of +-0.08 ms noise, n = 24
        let mut seed = 42u64;
        let d: Vec<f64> = (0..24).map(|_| 0.2 + noise(&mut seed, 0.08)).collect();
        let e = hl_paired(&d).unwrap();
        assert!((e.est - 0.2).abs() < 0.03, "{e:?}");
        assert!(e.lo > 0.1 && e.hi < 0.3, "{e:?}");
        // and the two-sample (LI) form on unpaired samples with a large common offset
        let x: Vec<f64> = (0..120).map(|_| 50.0 + noise(&mut seed, 1.5)).collect();
        let y: Vec<f64> = (0..120).map(|_| 49.0 + noise(&mut seed, 1.5)).collect();
        let e2 = hl_two_sample(&x, &y).unwrap();
        assert!((e2.est + 1.0).abs() < 0.3 && e2.hi < 0.0, "{e2:?}");
    }

    #[test]
    fn verdict_prefers_the_default() {
        let tau = tau_ms(48.0); // 0.048 ms
        assert!((tau - 0.048).abs() < 1e-12);
        assert_eq!(tau_ms(6.0), 0.02);
        let e = |est, lo, hi| Est { n: 30, est, lo, hi };
        assert_eq!(verdict(&e(-0.2, -0.3, -0.1), tau), Verdict::Adopt);
        // significant but inside tau: the default stays (hysteresis)
        assert_eq!(verdict(&e(-0.04, -0.045, -0.03), tau), Verdict::Default);
        assert_eq!(verdict(&e(0.0, -0.1, 0.1), tau), Verdict::Default);
        assert_eq!(verdict(&e(0.1, 0.05, 0.2), tau), Verdict::Slower);
    }

    /// A synthetic sampler: candidate i has true shift `shift[i]` ms, noise +-amp; `out` marks a
    /// candidate that fails its digest on the first call.
    fn synth(shift: Vec<f64>, amp: f64, out: Option<usize>) -> impl FnMut(&[usize], usize, bool) -> Vec<Sampled> {
        let mut seed = 7u64;
        move |active: &[usize], want: usize, _confirm: bool| {
            active.iter().map(|&i| {
                if Some(i) == out { return Sampled::Out("XCHECK FAIL (synthetic)".into()); }
                Sampled::Obs((0..want).map(|j| ((3 + (j % 4)) as u16, shift[i] + noise(&mut seed, amp))).collect())
            }).collect()
        }
    }

    #[test]
    fn race_adopts_a_real_winner_and_confirms_it() {
        let cfg = RaceCfg::default();
        let tau = 0.05;
        // candidates: slower, neutral, clearly faster (-0.3), slightly faster (-0.1)
        let mut s = synth(vec![0.4, 0.0, -0.3, -0.1], 0.08, None);
        let r = race(4, &cfg, tau, &mut s);
        assert_eq!(r.winner, Some(2), "{r:#?}");
        assert_eq!(r.verdict, "ADOPTED");
        assert!(r.confirm.unwrap().hi < 0.0);
        assert!(r.cands[0].fate.starts_with("SLOWER") || r.cands[0].fate.starts_with("PRUNED"), "{}", r.cands[0].fate);
        assert!(r.calls >= 2);
    }

    #[test]
    fn race_keeps_the_default_on_noise_and_inside_tau() {
        let cfg = RaceCfg::default();
        // pure noise: nothing may be adopted
        let mut s = synth(vec![0.0, 0.0, 0.0], 0.1, None);
        let r = race(3, &cfg, 0.05, &mut s);
        assert_eq!(r.winner, None, "{r:#?}");
        assert!(r.verdict.starts_with("DEFAULT"));
        // a real but sub-threshold gain (-0.02 ms against tau 0.05): the default stays
        let mut s2 = synth(vec![-0.02], 0.01, None);
        let r2 = race(1, &cfg, 0.05, &mut s2);
        assert_eq!(r2.winner, None, "{r2:#?}");
        // no candidates at all
        let mut s3 = synth(vec![], 0.1, None);
        assert_eq!(race(0, &cfg, 0.05, &mut s3).verdict, "DEFAULT (no candidates)");
    }

    /// A sampler that stops delivering (memory cap, a new unit key it cannot capture) must not
    /// livelock stage 2: the candidate leaves the race.
    #[test]
    fn race_terminates_when_a_sampler_underdelivers() {
        let cfg = RaceCfg::default();
        let mut calls = 0usize;
        let mut seed = 11u64;
        let mut s = |active: &[usize], want: usize, _c: bool| -> Vec<Sampled> {
            calls += 1;
            active.iter().map(|_| if calls == 1 {
                // stage 1 delivers a wide, undecided sample; later calls deliver nothing
                Sampled::Obs((0..want).map(|_| (3u16, noise(&mut seed, 1.0))).collect())
            } else { Sampled::Obs(Vec::new()) }).collect()
        };
        let r = race(2, &cfg, 0.01, &mut s);
        assert!(r.calls < 10, "{}", r.calls);
        assert_eq!(r.winner, None);
    }

    #[test]
    fn table_assign_matches_boot() {
        let mut t = table_fixture("user", 2400);
        t.decisions.push(Decision { id: "hc.bxr".into(), rev: 1, class: "S".into(),
                                    scope: DecScope { fam: Some("DraftChain".into()), m: Some(vec![1]), ..Default::default() },
                                    value: 48, default: Some(40), evidence: serde_json::Value::Null });
        let (a, ign) = table_assign(&t, false);
        // the am_nb decision (2 widths) is in force; the invalid hc.bxr at m=1 is not
        assert_eq!(a.len(), 2, "{ign:?}");
        assert!(a.iter().all(|(d, k, v)| d.id == "wp09.am_nb" && *v == 32 && k.is_some()));
        assert!(!key_valid(&crate::exl3_forward::T_HC_BXR, &Key::W(Fam::DraftChain as u8, 1, 0), 48));
        assert!(key_valid(&crate::exl3_forward::T_HC_BXR, &Key::W(Fam::Verify as u8, 6, 0), 48));
    }

    #[test]
    fn race_disqualifies_a_digest_failure_even_if_fast() {
        let cfg = RaceCfg::default();
        let mut s = synth(vec![-1.0, -0.3], 0.05, Some(0));
        let r = race(2, &cfg, 0.05, &mut s);
        assert_eq!(r.winner, Some(1));
        assert!(r.cands[0].fate.contains("XCHECK"), "{}", r.cands[0].fate);
    }

    #[test]
    fn width_split_only_reverts_a_clearly_slower_width() {
        let mut seed = 3u64;
        let mut obs: Vec<Obs> = Vec::new();
        for _ in 0..12 {
            obs.push((3, -0.2 + noise(&mut seed, 0.02)));
            obs.push((6, -0.2 + noise(&mut seed, 0.02)));
            obs.push((8, 0.3 + noise(&mut seed, 0.02)));
        }
        let sp = width_split(&obs);
        let keep: Vec<(u16, bool)> = sp.iter().map(|x| (x.0, x.2)).collect();
        assert_eq!(keep, vec![(3, true), (6, true), (8, false)]);
    }

    #[test]
    fn plan_orders_by_value_density_and_respects_the_budget() {
        let mut v = vec![
            PlanItem { name: "a".into(), ev_ms: 0.1, est_s: 60.0 },
            PlanItem { name: "b".into(), ev_ms: 1.0, est_s: 60.0 },
            PlanItem { name: "c".into(), ev_ms: 0.5, est_s: 10.0 },
        ];
        order_by_ev(&mut v);
        assert_eq!(v.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(), vec!["c", "b", "a"]);
        assert_eq!(plan_fits(&v, 0.0, 75.0), 2);
        assert_eq!(plan_fits(&v, 70.0, 75.0), 0);
        assert_eq!(plan_fits(&v, 0.0, 1e9), 3);
    }

    fn fp_fixture() -> Fingerprint {
        let mut ptx = BTreeMap::new();
        ptx.insert("exl3_bench".to_string(), "aa".repeat(32));
        Fingerprint {
            tune_build_id: "0123456789abcdef".into(), ptx_sha256: ptx, fatbin_sha256: BTreeMap::new(),
            registry_schema: registry_schema(),
            model: ModelFp { config_sha256: "c0".repeat(32), quant_sha256: "q0".repeat(32) },
            gpu: GpuFp { name: "NVIDIA GB10".into(), cc: "12.1".into(), sms: 48, l2_bytes: 25165824 },
            driver: "580.173.02".into(), cuda: "13.0".into(), tp: 1, clock_mhz: 2400,
        }
    }

    fn table_fixture(origin: &str, clock: u32) -> Table {
        let posture = Posture { lanes: 8, mtp_max_k: 7, dds: true, kv_fmt: 3, max_pos: 4097, prefill_chunk: 2048 };
        Table {
            format: TABLE_FORMAT.into(), origin: origin.into(),
            fingerprint: Fingerprint { clock_mhz: clock, ..fp_fixture() }, posture,
            provenance: serde_json::json!({"commit": "x", "box": "test"}),
            decisions: vec![Decision {
                id: "wp09.am_nb".into(), rev: 1, class: "S".into(),
                scope: DecScope { fam: Some("Verify".into()), m: Some(vec![3, 4]), regime: Some("dense".into()), ..Default::default() },
                value: 32, default: Some(0),
                evidence: serde_json::json!({"unit": "verify_dry", "n": 30, "hl_delta_ms": -0.05, "ci95_ms": [-0.07, -0.03]}),
            }],
            families: serde_json::json!({"wp09.am_nb": {"status": "tuned"}}),
            confirm: serde_json::json!({"li_round_ms": {}, "pass": true}),
            table_sha256: String::new(),
        }
    }

    #[test]
    fn table_round_trips_sealed_and_detects_edits() {
        let dir = std::env::temp_dir().join(format!("gb10_tune_test_{}", std::process::id()));
        let mut t = table_fixture("user", 2400);
        let path = dir.join(table_file_name(&t).trim_start_matches("tune/"));
        let fsha = write_table(&mut t, path.to_str().unwrap()).unwrap();
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(sha256_hex(&raw), fsha);
        let back: Table = serde_json::from_slice(&raw).unwrap();
        assert!(back.seal_ok() && !back.table_sha256.is_empty());
        assert_eq!(back.decisions.len(), 1);
        assert_eq!(back.fingerprint, t.fingerprint);
        assert_eq!(back.posture, t.posture);
        // the decision validates against the registry and expands to its width keys
        let (dec, ns, nd, ign) = build_decisions(&back, false);
        assert_eq!((ns, nd), (1, 0), "{ign:?}");
        let slot = crate::exl3_forward::T_WP09_AM_NB.slot as usize;
        assert_eq!(dec[slot].get(&Key::W(Fam::Verify as u8, 3, 0)), Some(&32));
        assert_eq!(dec[slot].get(&Key::W(Fam::Verify as u8, 4, 0)), Some(&32));
        assert_eq!(dec[slot].get(&Key::W(Fam::Verify as u8, 5, 0)), None);
        // a hand edit after sealing is refused
        let edited = String::from_utf8(raw.clone()).unwrap().replace("\"value\": 32", "\"value\": 16");
        assert_ne!(edited.as_bytes(), &raw[..]);
        let tt: Table = serde_json::from_str(&edited).unwrap();
        assert!(!tt.seal_ok());
        // the report names the decision and the seal
        let rep = report(&back, &raw);
        assert!(rep.contains("wp09.am_nb") && rep.contains(" OK"), "{rep}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fingerprint_mismatch_falls_back_with_the_field_named() {
        let t = table_fixture("shipped", 2400);
        let raw = serde_json::to_vec(&t).unwrap();
        let have = fp_fixture();
        let file = |b: Vec<u8>| vec![(PathBuf::from("tune/x.json"), b)];
        // exact match -> picked
        assert!(select_table(file(raw.clone()), &have, &t.posture, 2400).is_ok());
        // a new build (tune_build_id) -> FALLBACK naming the field
        let have2 = Fingerprint { tune_build_id: "fedcba9876543210".into(), ..have.clone() };
        let e = select_table(file(raw.clone()), &have2, &t.posture, 2400).err().unwrap();
        assert!(e.starts_with("tune_build_id"), "{e}");
        // a different kernel PTX
        let mut have3 = have.clone();
        have3.ptx_sha256.insert("exl3_bench".into(), "bb".repeat(32));
        assert!(select_table(file(raw.clone()), &have3, &t.posture, 2400).err().unwrap().starts_with("ptx_sha256"));
        // another TP degree, another posture
        let have4 = Fingerprint { tp: 2, ..have.clone() };
        assert!(select_table(file(raw.clone()), &have4, &t.posture, 2400).err().unwrap().starts_with("tp"));
        let p2 = Posture { max_pos: 262145, ..t.posture.clone() };
        assert!(select_table(file(raw.clone()), &have, &p2, 2400).err().unwrap().starts_with("posture"));
        // a corrupt file and an empty directory
        assert!(select_table(file(b"{not json".to_vec()), &have, &t.posture, 2400).err().unwrap().starts_with("parse error"));
        assert_eq!(select_table(Vec::new(), &have, &t.posture, 2400).err().unwrap(), "no table in ./tune");
        // an unconfirmed table (gate not passed / hand-written): never auto-loaded, loads when named
        let mut unconf = t.clone();
        unconf.confirm = serde_json::json!({"pass": false});
        let e = select_table(file(serde_json::to_vec(&unconf).unwrap()), &have, &t.posture, 2400).err().unwrap();
        assert!(e.starts_with("unconfirmed"), "{e}");
        assert!(select_table_x(file(serde_json::to_vec(&unconf).unwrap()), &have, &t.posture, 2400, true).is_ok());
        // a sealed-then-edited table
        let mut sealed = t.clone();
        sealed.seal();
        sealed.decisions[0].value = 16;
        let e = select_table(file(serde_json::to_vec(&sealed).unwrap()), &have, &t.posture, 2400).err().unwrap();
        assert!(e.starts_with("table_sha256"), "{e}");
        // the clock is not strict: a user table wins over a shipped one; among equals the nearest
        // clock profile wins
        let u = table_fixture("user", 2000);
        let files = vec![(PathBuf::from("tune/a.json"), raw.clone()),
                         (PathBuf::from("tune/b.json"), serde_json::to_vec(&u).unwrap())];
        let pick = select_table(files, &have, &t.posture, 2300).unwrap();
        assert_eq!(pick.table.origin, "user");
        let s2 = table_fixture("shipped", 2000);
        let files = vec![(PathBuf::from("tune/a.json"), raw.clone()),
                         (PathBuf::from("tune/b.json"), serde_json::to_vec(&s2).unwrap())];
        assert_eq!(select_table(files, &have, &t.posture, 2100).unwrap().table.fingerprint.clock_mhz, 2000);
    }

    #[test]
    fn zero_decision_tables_load_failed_confirms_do_not() {
        let have = fp_fixture();
        let file = |b: Vec<u8>| vec![(PathBuf::from("tune/x.json"), b)];
        let sealed = |mut t: Table| { t.seal(); serde_json::to_vec_pretty(&t).unwrap() };
        let mut zero = table_fixture("user", 2400);
        zero.decisions.clear();
        // L10 writer: a clean 0-decision run writes pass = true with its reason -> auto loads it
        zero.confirm = serde_json::json!({"skipped": "no decision adopted", "pass": true, "known_answer": "holds",
            "reason": "0 decisions: the table is the built-in defaults", "hashes_identical": true, "ignored": []});
        let p = zero.posture.clone();
        assert!(select_table(file(sealed(zero.clone())), &have, &p, 2400).is_ok());
        // pre-L10 0-decision table (no pass field, sealed as written then): still validates + loads
        let mut legacy = zero.clone();
        legacy.confirm = serde_json::json!({"skipped": "no decision adopted"});
        assert!(select_table(file(sealed(legacy.clone())), &have, &p, 2400).is_ok());
        // ...but the legacy shape never excuses a table that carries decisions
        let mut legacy_dec = table_fixture("user", 2400);
        legacy_dec.confirm = legacy.confirm.clone();
        let e = select_table(file(sealed(legacy_dec)), &have, &p, 2400).err().unwrap();
        assert!(e.starts_with("unconfirmed"), "{e}");
        // a 0-decision run that did NOT pass (ignored decisions / deviating known answer), a
        // rejected gate and an ungated table: all refused by auto
        let mut failed0 = zero.clone();
        failed0.confirm = serde_json::json!({"skipped": "no decision adopted", "pass": false, "known_answer": "deviates"});
        let mut rejected = table_fixture("user", 2400);
        rejected.confirm = serde_json::json!({"li_round_ms": {}, "beats_defaults": false, "pass": false});
        let mut ungated = zero.clone();
        ungated.confirm = serde_json::json!({"skipped": "--autotune-skip-gate", "pass": false});
        for t in [failed0, rejected, ungated] {
            let e = select_table(file(sealed(t)), &have, &p, 2400).err().unwrap();
            assert!(e.starts_with("unconfirmed"), "{e}");
        }
        // tampering: flipping a failed confirm to pass after sealing is refused by the seal
        let mut tamper = zero.clone();
        tamper.confirm = serde_json::json!({"skipped": "no decision adopted", "pass": false});
        let raw = String::from_utf8(sealed(tamper)).unwrap().replace("\"pass\": false", "\"pass\": true");
        let e = select_table(file(raw.into_bytes()), &have, &p, 2400).err().unwrap();
        assert!(e.starts_with("table_sha256"), "{e}");
        // ...and so is stripping the pass field to fake the legacy shape
        let raw = String::from_utf8(sealed(zero.clone())).unwrap();
        let t2: Table = serde_json::from_str(&raw).unwrap();
        let mut stripped = t2.clone();
        stripped.confirm = serde_json::json!({"skipped": "no decision adopted"});
        let e = select_table(file(serde_json::to_vec_pretty(&stripped).unwrap()), &have, &p, 2400).err().unwrap();
        assert!(e.starts_with("table_sha256"), "{e}");
    }

    #[test]
    fn unproven_and_non_s_decisions_are_ignored_at_boot() {
        let mut t = table_fixture("user", 2400);
        t.decisions.push(Decision { id: "pdl".into(), rev: 1, class: "S".into(),
                                    scope: DecScope { fam: Some("Verify".into()), ..Default::default() },
                                    value: 1, default: Some(0), evidence: serde_json::Value::Null });
        t.decisions.push(Decision { id: "qsa.splits_dec".into(), rev: 1, class: "N".into(),
                                    scope: DecScope::default(), value: 16, default: Some(32),
                                    evidence: serde_json::Value::Null });
        t.decisions.push(Decision { id: "hc.bxr".into(), rev: 1, class: "S".into(),
                                    scope: DecScope { fam: Some("Verify".into()), m: Some(vec![1]), ..Default::default() },
                                    value: 48, default: Some(40), evidence: serde_json::Value::Null });
        let (_, ns, _, ign) = build_decisions(&t, false);
        assert_eq!(ns, 1, "{ign:?}");
        assert!(ign.iter().any(|x| x.contains("pdl") && x.contains("UNPROVEN")), "{ign:?}");
        assert!(ign.iter().any(|x| x.contains("qsa.splits_dec") && x.contains("section N")), "{ign:?}");
        assert!(ign.iter().any(|x| x.contains("hc.bxr") && x.contains("valid")), "{ign:?}");
    }

    // ---- §5.2 v2 (pure CPU): the balanced order and the position model ----

    /// Over one period (from ANY start row) every row is a permutation, every arm sits at every
    /// position exactly period/n times, and every ordered pair of distinct arms is adjacent exactly
    /// period/n times (first-order carry-over balance) — for every arm count the tuner can run.
    #[test]
    fn balanced_order_covers_every_position_equally() {
        for n in 1..=16usize {
            let p = williams_period(n);
            assert_eq!(p % n, 0);
            for start in [0usize, 1, 5, 17] {
                let mut at = vec![vec![0usize; n]; n]; // at[arm][pos]
                let mut adj = vec![vec![0usize; n]; n]; // adj[x][y]: y right after x
                for r in start..start + p {
                    let row = williams_row(n, r);
                    let mut s = row.clone();
                    s.sort_unstable();
                    assert_eq!(s, (0..n).collect::<Vec<_>>(), "n {n} row {r} {row:?} is not a permutation");
                    for (pos, &a) in row.iter().enumerate() { at[a][pos] += 1; }
                    for w in row.windows(2) { adj[w[0]][w[1]] += 1; }
                }
                for a in 0..n {
                    for pos in 0..n { assert_eq!(at[a][pos], p / n, "n {n} start {start}: arm {a} at position {pos}"); }
                    for b in 0..n {
                        if a != b { assert_eq!(adj[a][b], p / n, "n {n} start {start}: {a} -> {b} adjacency"); }
                    }
                }
            }
        }
    }

    /// Cursors are per (cell, n) and persist: interleaved cells each walk their own design; a cell
    /// revisited after another arm count continues where it stopped (never restarts at a fixed row).
    #[test]
    fn balanced_order_cursors_are_per_cell_and_persist() {
        let mut o = BalancedOrder::new(0xABCD);
        let n = 3;
        let p = williams_period(n);
        // two cells interleaved round by round (the verify widths rotate per round): each cell must
        // be exactly balanced on its own after p visits
        let mut first = [[0usize; 3]; 2];
        for _ in 0..p {
            for (ci, cell) in [7u64, 9].iter().enumerate() {
                first[ci][o.next(*cell, n)[0]] += 1;
            }
        }
        for ci in 0..2 { assert_eq!(first[ci], [p / n; 3], "cell {ci}"); }
        // short phases (1 visit each, n = 2, with n = 3 phases in between) still alternate
        let mut firsts = Vec::new();
        for _ in 0..6 {
            firsts.push(o.next(11, 2)[0]);
            let _ = o.next(11, 3);
        }
        assert_eq!(firsts.iter().filter(|&&a| a == 0).count(), 3, "{firsts:?}");
        // deterministic in the seed
        let mut a = BalancedOrder::new(5);
        let mut b = BalancedOrder::new(5);
        for _ in 0..10 { assert_eq!(a.next(3, 4), b.next(3, 4)); }
    }

    /// The position model recovers a known position effect under round (content) and arm effects,
    /// and invents none when there is none — or when the design cannot identify it.
    #[test]
    fn position_offsets_recover_and_never_invent() {
        let mut seed = 99u64;
        let n = 3usize;
        let truth = [0.6, 0.1, -0.7];
        let arms = [0.0, 0.0, 0.2];
        let mut with = Vec::new();
        let mut without = Vec::new();
        let mut confounded = Vec::new();
        for r in 0..120u64 {
            let content = 30.0 + 10.0 * (noise(&mut seed, 0.5) + 0.5);
            let row = williams_row(n, r as usize);
            for (pos, &a) in row.iter().enumerate() {
                let e = noise(&mut seed, 0.05);
                with.push(PosObs { round: r, arm: a as u64, pos, ms: content + arms[a] + truth[pos] + e });
                without.push(PosObs { round: r, arm: a as u64, pos, ms: content + arms[a] + e });
                // confounded: arm a always at position a (the design cannot separate them): the
                // combined effect must stay with the arm, not be split into f
                confounded.push(PosObs { round: r, arm: a as u64, pos: a, ms: content + arms[a] + truth[a] + e });
            }
        }
        let f = position_offsets(&with, n);
        for p in 0..n { assert!((f[p] - truth[p]).abs() < 0.03, "{f:?}"); }
        let f0 = position_offsets(&without, n);
        assert!(f0.iter().all(|x| x.abs() < 0.03), "{f0:?}");
        let fc = position_offsets(&confounded, n);
        assert!(fc.iter().all(|x| x.abs() < 0.03), "confounded design must not move the arm effect into f: {fc:?}");
        // uninformative inputs: no adjustment
        assert_eq!(position_offsets(&with[..3], n), vec![0.0; n]);
        assert_eq!(position_offsets(&with, 1), vec![0.0]);
    }
}
