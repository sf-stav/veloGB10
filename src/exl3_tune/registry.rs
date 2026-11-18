//! The tunable registry: one `TunableDef` slot per knob (append-only; index == `slot`), the reserved and selftest entries,
//! and the compile-time class / scope exactness contract. See `mod.rs` for the file map.

use super::*;

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

pub(super) const fn domain_has(d: &[i32], v: i32) -> bool {
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
