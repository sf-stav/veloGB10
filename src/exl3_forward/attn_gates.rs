//! Predicates and geometry gates that choose the attention kernels (head attention, dense split-K, PFTC, dense
//! v3 plan), plus the QSA split / score / top-k / select / W4QSA / PQ8 tunables and their cross-check flags.
//! See `mod.rs` for the file map and `attn.rs` for the code that consumes them.

use super::*;

/// Attention-target sentinel for the MTP draft head in the shared QSA helpers.
pub(super) const HEAD_ATT: usize = usize::MAX;
/// A5 D5: fixed split count of the draft head's dense attention (graph-safe launch constant;
/// the per-split key range is derived on device from the row's position).
pub const DRAFT_ATTN_SPLITS: usize = 64;

/// --exl3-head-qsa=0: draft head stays dense everywhere and appends no indexer keys (A/B).
pub(super) fn head_qsa_off() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-head-qsa")).as_deref() == Ok("0"))
}

/// DSPLITK: --exl3-dense-splitk, ON by default (owner 2026-11-07, after the GPU results: 35B EXL3 decode 3-5.7x
/// faster from 64K up, LOSSLESS_OK). The summation order differs from xq_attn_decode, so output bytes can differ
/// from `--exl3-dense-splitk off`, which keeps the old kernels for A/B.
pub(super) fn dense_splitk_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(crate::opts::var(crate::opt!("exl3-dense-splitk")).as_deref(), Ok("off") | Ok("0")))
}

/// PFTC (owner 2026-11-07): --exl3-dense-pf-tc, the tensor-core dense prefill for the packs without an indexer.
/// ON by default (the owner's #1 priority for these packs); off = flash256 (A/B). Not bitwise to flash256.
/// POSTURE1 `--exl3-dense-pf-tc-qsa on|off` (default off): see `pf_tc_qsa_ok`.
pub(crate) fn pf_tc_qsa_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::on(crate::opt!("exl3-dense-pf-tc-qsa")))
}
pub static PF_TC_QSA_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) fn dense_pf_tc_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(crate::opts::var(crate::opt!("exl3-dense-pf-tc")).as_deref(), Ok("off") | Ok("0")))
}

/// DSPLITK round 2b: --exl3-dense-splitk-w4 on|off (default on): the trunk's split-K dense attention stages K/V through
/// the cp.async smem ring (the W4S draft-head kernels); off = the plain loads. Bitwise either way.
pub(super) fn dense_splitk_w4_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(crate::opts::var(crate::opt!("exl3-dense-splitk-w4")).as_deref(), Ok("off") | Ok("0")))
}

/// DSPLITK round 2: rows per block of xq_attn_dense_splitk4_vr (mirrors the kernel's XQ_VR_ROWS).
pub(super) const XQ_VR_ROWS: usize = 4;
/// DSPLITK round 3: the ring-staged shared-read verify (_vr_w4) takes every one-slot verify of this many rows or
/// more on f16 / f32 caches (the plain-load _vr keeps 4+: its 2-3-row blocks under-filled the GPU, 35B f16 at 48K d2
/// 36.7 -> 40.1; _vr_w4 at 3 rows: 31.7 -> 29.7).
pub(super) const XQ_VR_MIN_ROWS_W4: usize = 2;
/// DSPLITK round 3: on q8 / fp8 caches the per-row deep-ring _t4 serves verifies below this many rows and _vr_w4 from
/// here up. Measured at 48K (TP=1, LOSSLESS_OK, ms/round per-row _t4 vs shared-read _vr_w4, both with the 4-key
/// interleave): 35B q8 d1 22.9 / 23.1, d2 27.3 / 27.6, d3 32.4 / 31.9, d4 37.9 / 36.6, d6 47.0 / 45.8;
/// 27B q8 d1 85.9 / 86.8, d2 96.2 / 95.7, d3 106.6 / 104.2, d4 116.5 / 114.6, d6 136.3 / 130.6.
pub(super) const XQ_VR_MIN_ROWS_TILED: usize = 3;
/// DSPLITK round 3: cache formats whose split-K rows are decoded once per block into an f32 tile (an unpack with a
/// scale per value: q8 and fp8); f16 / f32 keep the per-warp read.
/// DSPLITK round 3: the raw ring slot of xq_attn_dense_splitk4_t4 (mirrors the kernel's XQ_T4_RB: q8 / fp8 at hd 256).
pub(super) const XQ_T4_RB: usize = 272;
pub(super) fn kv_split_tiled(fmt: u32) -> bool {
    fmt != KV_F16 && fmt != KV_F32
}
/// DSPLITK round 2: --exl3-dense-splitk-vr on|off (default on): one verify's rows share every K/V read; off = the per-row
/// kernel. Bitwise either way (each row keeps its own slices and order).
pub(super) fn dense_splitk_vr_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(crate::opts::var(crate::opt!("exl3-dense-splitk-vr")).as_deref(), Ok("off") | Ok("0")))
}

/// PFTC: --exl3-pf-tc-xcheck (diagnostic): every PFTC launch also runs flash256 and prints the difference.
pub(super) fn pf_tc_xcheck_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-pf-tc-xcheck")).is_ok())
}

/// PFTC: the geometries xq_attn_pf_tc256 covers — head_dim 256 and a GQA group dividing its 96-row block.
pub(crate) fn pf_tc_geometry_ok(nh: usize, nkv: usize, hd: usize) -> bool {
    hd == 256 && nkv > 0 && nh % nkv == 0 && 96 % (nh / nkv) == 0
}

/// DSPLITK: the geometries xq_attn_dense_splitk4 covers — head_dim 256 (32 lanes x 8 dims) and at most 16 query
/// heads per KV head (8 warps x 2 passes). The 35B (16 / 2), 27B (24 / 4) and 9B (16 / 4) packs all fit.
pub(crate) fn dense_splitk_geometry_ok(nh: usize, nkv: usize, hd: usize) -> bool {
    hd == 256 && nkv > 0 && nh % nkv == 0 && nh / nkv <= 16
}

/// --exl3-draft-attn=legacy restores the one-block-per-query-head `xq_attn_decode` scan.
pub(super) fn draft_attn_legacy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-draft-attn")).as_deref() == Ok("legacy"))
}

/// WP13: dense-regime trunk attention v3 (xq_attn_dense_prep -> _dots -> _acc{1,4}), bitwise
/// to xq_attn_decode_group (verify) / xq_attn_decode (decode step). Default ON;
/// --exl3-dense-v3=0 restores the old per-(head, row) two-pass scan (A/B + bisect escape).
pub(crate) fn dense_v3_on() -> bool {
    tune::get_cur(&T_DENSE_V3) != 0
}
/// TUNE T0 (p3c): WP13's switch. Its geometry (DV3_TS / DV3_NCH / DV3_G) is compile-time, mirrored
/// from the kernel — no runtime schedule knob exists yet (a T4 twin would add one).
pub(crate) const T_DENSE_V3: tune::TunableDef = tune::TunableDef {
    slot: 37, id: "attn.dense_v3", class: tune::Class::S, scope: tune::Scope::Width,
    fams: &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::Util],
    domain: &[0, 1], default: 1, valid: tune::valid_any,
    xcheck: "--probe-exl3-binv EXL3-ATTN (v3 vs v2, f32/f16/fp8/q8/q4k) + --exl3-dense-xcheck",
    env: "--exl3-dense-v3", env_parse: || tune::env_on_unless_0(crate::opt!("exl3-dense-v3")), rev: 1, wp: "WP13",
};
/// WP13 XCHECK (diagnostic, --exl3-dense-xcheck=1): after every v3 dense attention the
/// old kernel re-runs into a side buffer and xq_dv3_xcheck diffs the f16 outputs bitwise on
/// device — "[dense-v3-xcheck] MISMATCH ..." per event, a heartbeat every 1024 checks.
/// Graph-safe (device printf); costs the old kernel's time on top of v3's.
pub(super) fn dense_xcheck_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-dense-xcheck")).as_deref() == Ok("1"))
}
/// WP13 geometry — mirrors XQ_DV3_TS / XQ_DV3_NCH / XQ_DV3_G / XQ_DV3_G6 in kernels/exl3_bench.cu.
pub(super) const DV3_TS: usize = 2304;
pub(super) const DV3_NCH: usize = 36;
pub(super) const DV3_G: usize = 12;
/// TP-4G: the second instantiation — 6 q heads per kv head (W=4 / TP=4, kv replicated).
pub(super) const DV3_G6: usize = 6;

/// TP-4G: q heads per kv head the dense-v3 kernels are instantiated for, or None (the layer
/// stays on v2). Pure; `dense_v3_ok` adds the engine-state conditions on top.
pub(crate) fn dv3_group(num_heads: usize, num_kv_heads: usize) -> Option<usize> {
    if num_kv_heads == 0 { None }
    else if num_heads == DV3_G * num_kv_heads { Some(DV3_G) }
    else if num_heads == DV3_G6 * num_kv_heads { Some(DV3_G6) }
    else { None }
}

/// TP-4G / W4S dattn: the 5-stage-ring twin of a dense-v3 accumulate kernel (bitwise to it).
pub(crate) fn dv3_w4_twin(old: &'static str) -> &'static str {
    match old {
        "xq_attn_dense_acc4" => "xq_attn_dense_acc4_w4",
        "xq_attn_dense_acc1" => "xq_attn_dense_acc1_w4",
        "xq_attn_dense_acc4_g6" => "xq_attn_dense_acc4_w4_g6",
        "xq_attn_dense_acc1_g6" => "xq_attn_dense_acc1_w4_g6",
        k => k,
    }
}

/// TP-4G: launch plan of the dense-v3 dots + accumulate kernels for group size `g` (6 | 12) and
/// `m` rows: (dots name, accumulate name BEFORE the W4S twin mapping, accumulate dim slice SD,
/// accumulate block threads = 16 * g). The G=12 names are the pre-TP-4G kernels unchanged.
pub(crate) fn dv3_plan(g: usize, m: usize) -> (&'static str, &'static str, usize, u32) {
    let g6 = g == DV3_G6;
    let dots = if g6 { "xq_attn_dense_dots_g6" } else { "xq_attn_dense_dots" };
    let (acc, sd) = match (m == 1, g6) {
        (true, false) => ("xq_attn_dense_acc1", 16),
        (false, false) => ("xq_attn_dense_acc4", 64),
        (true, true) => ("xq_attn_dense_acc1_g6", 16),
        (false, true) => ("xq_attn_dense_acc4_g6", 64),
    };
    (dots, acc, sd, 16 * g as u32)
}


/// S-A3-o G1: ring floats per (token, head) — mirrors XQ_GDN_RS in exl3_bench.cu.
pub(super) const GDN_RS: usize = 260;

/// S-A3-i: fixed split count of the gathered QSA decode (graph-safe: the split count
/// is a launch constant; each split no-ops past pos_sel).
pub const QSA_SPLITS: usize = 4;
/// Decode/verify split count of the gathered QSA attention (graph-safe launch constant).
/// 4 splits gave rows x 2 kv heads x 4 = 8 CTAs at m=1, each walking ~513 selected tokens
/// serially: 6.4 ms per plain step at 32K context, latency-bound (m=6 cost the same). 32
/// splits = 64 CTAs at m=1. Decode and verify share it (MTP stays bitwise lossless); prefill
/// keeps QSA_SPLITS (2,048-row chunks are already parallel; its partials would be 8x larger),
/// so chunked-prefill vs token-by-token decode now agree to fp32 reassociation, not bits.
pub const QSA_SPLITS_DEC: usize = 32;

/// TUNE T0 (N2): the decode/verify QSA split count is a NUMERICS knob (it moves the reduction
/// grouping) — class N, Global (m-independent: lossless MTP needs decode == verify grouping), so
/// the table never sets it without the owner's consent; the legacy alias keeps its exact parse.
pub(crate) const T_QSA_SPLITS_DEC: tune::TunableDef = tune::TunableDef {
    slot: 8, id: "qsa.splits_dec", class: tune::Class::N, scope: tune::Scope::Global,
    fams: &[tune::Fam::Step, tune::Fam::Verify],
    domain: &[4, 8, 16, 32], default: QSA_SPLITS_DEC as i32, valid: tune::valid_any, xcheck: "drift table (N)",
    env: "--exl3-qsa-splits-dec",
    env_parse: || tune::env_str(crate::opt!("exl3-qsa-splits-dec")).and_then(|v| v.parse::<usize>().ok())
        .filter(|n| (1..=QSA_SPLITS_DEC).contains(n)).map(|n| n as i32),
    rev: 1, wp: "S-A3-qsa",
};

/// Runtime decode/verify split count: QSA_SPLITS_DEC, or --exl3-qsa-splits-dec=4 to restore
/// the pre-2026-09-26 reduction grouping exactly (buffers are sized for the maximum).
pub(super) fn qsa_splits_dec() -> usize {
    // TUNE T0: N-frozen registry value (env alias or the built-in 32; never a table value)
    (tune::get_cur(&T_QSA_SPLITS_DEC).clamp(1, QSA_SPLITS_DEC as i32)) as usize
}

// ---- WP04 + WP14 (PLAN/SURPASS_PLAN_2026-09-26.md): long-context QSA decode + boot capture ----
// Every switch below is a diagnostic escape (AGENTS §1b): the new paths are bit-identical to the
// old ones, the old ones stay reachable for A/B and bisection.
fn wp_env_flag(o: crate::opts::OptId, default_on: bool) -> bool {
    match crate::opts::var(o).as_deref() {
        Ok("0") | Ok("off") => false,
        Ok(_) => true,
        Err(_) => default_on,
    }
}
/// --exl3-score-mr=0: the bucketed `qsa_score_pooled_b` scorer (and bucket-keyed graphs).
pub(super) fn score_mr_on() -> bool {
    tune::get_cur(&T_SCORE_MR) != 0
}
/// `wp_env_flag` as a registry env parser: "0" / "off" -> 0, any other value -> 1, unset -> None.
fn env_flag_parse(o: crate::opts::OptId) -> Option<i32> {
    tune::env_str(o).map(|v| !(v == "0" || v == "off") as i32)
}
// TUNE T0 (p3c): the WP04 / WP14 QSA decode switches (bit-identical twins; eager XCHECKs).
pub(crate) const FAMS_QSA: &[tune::Fam] = &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass,
                                            tune::Fam::DraftChain, tune::Fam::Util];
pub(crate) const T_SCORE_MR: tune::TunableDef = tune::TunableDef {
    slot: 31, id: "qsa.score_mr", class: tune::Class::S, scope: tune::Scope::Global, fams: FAMS_QSA,
    // structural: 0 brings back the bucket-keyed graphs (qsa_bucket_free) — never searched
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--exl3-score-xcheck (pooled_mr vs pooled_b)",
    env: "--exl3-score-mr", env_parse: || env_flag_parse(crate::opt!("exl3-score-mr")), rev: 1, wp: "WP04",
};
pub(crate) const T_SCORE_GENERIC: tune::TunableDef = tune::TunableDef {
    slot: 32, id: "qsa.score_generic", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_QSA,
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "--exl3-score-xcheck (generic body)",
    env: "--exl3-score-mr",
    env_parse: || tune::env_str(crate::opt!("exl3-score-mr")).map(|v| (v == "generic") as i32), rev: 1, wp: "WP04",
};
pub(crate) const T_SCORE_MR_CTAS: tune::TunableDef = tune::TunableDef {
    slot: 33, id: "qsa.score_mr_ctas", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_QSA,
    // CTAs per SM of qsa_score_pooled_mr (a grid-size launch constant: bucket-free, one graph per regime)
    domain: &[2, 3, 4, 6, 8], default: 4, valid: tune::valid_any,
    xcheck: "--exl3-score-xcheck per value (+ the per-sample live digest)",
    env: "--exl3-score-mr-ctas",
    env_parse: || tune::env_str(crate::opt!("exl3-score-mr-ctas")).and_then(|v| v.parse::<u32>().ok())
        .filter(|n| (1..=24).contains(n)).map(|n| n as i32),
    rev: 1, wp: "WP04",
};
pub(crate) const T_TOPK_V2: tune::TunableDef = tune::TunableDef {
    slot: 34, id: "qsa.topk_v2", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_QSA,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--exl3-topk-xcheck (asc2 vs asc)",
    env: "--exl3-topk-v2", env_parse: || env_flag_parse(crate::opt!("exl3-topk-v2")), rev: 1, wp: "WP04",
};
/// qsa.sel_gather value -1 = auto (hg1 at m = 1, hg3 at m >= 2 — the built-in rule).
pub(crate) const T_SEL_GATHER: tune::TunableDef = tune::TunableDef {
    slot: 35, id: "qsa.sel_gather", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_QSA,
    domain: &[-1, 0, 1, 3], default: -1, valid: tune::valid_any, xcheck: "--exl3-gather-xcheck (hg1/hg3 vs splitk4)",
    env: "--exl3-sel-gather",
    env_parse: || tune::env_str(crate::opt!("exl3-sel-gather")).map(|v| match v.as_str() {
        "legacy" | "0" => 0, "hg1" | "1" => 1, "hg3" | "3" => 3, _ => -1 }),
    rev: 1, wp: "WP04",
};
pub(crate) const T_SEL_VEC: tune::TunableDef = tune::TunableDef {
    slot: 36, id: "qsa.sel_vec", class: tune::Class::S, scope: tune::Scope::Width,
    fams: &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass, tune::Fam::DraftChain, tune::Fam::Prefill,
            tune::Fam::Util],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--exl3-sel-xcheck (splitk4 vs splitk)",
    env: "--exl3-sel-vec", env_parse: || tune::env_on_unless_0(crate::opt!("exl3-sel-vec")), rev: 1, wp: "S-A3-v",
};
/// WP04-v2: `qsa_score_pooled_mr` geom flag (bit 62) — --exl3-score-mr=generic runs the kernel's
/// generic body instead of the paired-block 4-head fast body (A/B; both bit-identical to pooled_b).
pub(super) fn score_mr_geom_flag() -> u64 {
    if tune::get_cur(&T_SCORE_GENERIC) != 0 { 1u64 << 62 } else { 0 }
}
/// --exl3-topk-v2=0: the original `xq_qsa_topk_asc` (serial bucket walk, two exscans/chunk).
pub(super) fn topk_v2_on() -> bool {
    tune::get_cur(&T_TOPK_V2) != 0
}
/// A5-K3 (w6/K3, p6d ledger #5a): `xq_qsa_topk_asc3` — asc2's selection rows bit for bit (same
/// (key desc, index asc) kept set, same ascending emission, same pos_sel) with a latency-lean
/// schedule: row min/max -> 11-bit digits below the common prefix -> crossing-bin compaction into
/// shared memory -> warp-segment two-pass emission (see kernels/exl3_bench.cu 3b2). Engages only
/// where the sparse QSA step runs (above the dense threshold). Registry attn.qsa_select (S class, per
/// width); DEFAULT ON after the in-round A/B (128K -0.67 ms/round, every other class ties;
/// PLAN/notes_2026-09-27/K3.md); escape --qsa-select=0 = asc2. Takes precedence over qsa.topk_v2.
pub(crate) const T_QSA_SELECT: tune::TunableDef = tune::TunableDef {
    slot: 87, id: "attn.qsa_select", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_QSA,
    domain: &[0, 1], default: 1, valid: tune::valid_any,
    xcheck: "--probe-exl3-binv EXL3-QSA-SELECT (asc3 vs asc2 + CPU reference) + --exl3-topk-xcheck (eager, vs asc)",
    env: "--qsa-select", env_parse: || tune::env_on_unless_0(crate::opt!("qsa-select")), rev: 1, wp: "A5-K3",
};
pub(super) fn qsa_select_on() -> bool {
    tune::get_cur(&T_QSA_SELECT) != 0
}
/// Decode/verify QSA gather: --exl3-sel-gather = legacy (5a, `xq_attn_sel_splitk4`, or the
/// scalar splitk with --exl3-sel-vec=0) | hg1 (12 warps x 1 head) | hg3 (4 warps x 3 heads)
/// | auto (default: hg1 at m = 1 — 64 CTAs, per-entry latency decides — and hg3 at m >= 2 —
/// 5 CTAs/SM of 128 threads vs 3 of 384). All bit-identical partials, so the choice may differ
/// between a decode step and a verify without touching losslessness. Returns heads per warp
/// for this m (0 = legacy).
pub(super) fn sel_gather_hpw(m: usize) -> usize {
    match tune::get_cur(&T_SEL_GATHER) {
        0 => 0,
        1 => 1,
        3 => 3,
        _ => if m == 1 { 1 } else { 3 },
    }
}
/// W4/QSA (decode wave 4, gap #4): the row-shared union gather `xq_attn_sel_union` for multi-row
/// verifies (one slot, m >= 2) — each selected K/V row staged ONCE for all the rows that picked it,
/// bit-identical partials to the hg/splitk family.
/// p5d DEFAULT = OFF (the old sel_gather_hpw dispatch: hg1 at m = 1, hg3 at m >= 2). Bitwise, but
/// slower IN-ROUND (d13b94c on .13, 2026-09-27: nsys per call at m = 4..8 union 193 -> 452 us vs
/// hg3 131 -> 201 us — the union list grows ~m x 64 entries walked serially on 192 CTAs vs hg3's
/// 512; 32K verify_drain 52.55 vs 50.08 ms (+2.5 ms/round), 128K 58.6 vs 54.8 ms (+3.8)).
///   --w4qsa=1       opt in to the union gather.
///   --w4qsa-off=1   escape: the old dispatch everywhere (wins over --w4qsa=1).
/// An explicit --exl3-sel-gather=legacy|hg1|hg3 also pins the old kernels (so the WP14 A/B
/// commands keep their meaning).
/// TUNE T0c (p5e): w4qsa.on per width (sparse verifies; bitwise, so a full tune re-evaluates it
/// in-round at the sparse point); the aliases (and an explicit --exl3-sel-gather pin) decide
/// exactly as before whenever one is set.
pub(crate) fn w4qsa_on() -> bool {
    tune::get_cur(&T_W4QSA_ON) != 0
}
/// The aliases' rule: (on, why) — the pre-registry logic verbatim.
fn w4qsa_env_rule() -> (bool, &'static str) {
    let pinned = matches!(crate::opts::var(crate::opt!("exl3-sel-gather")).as_deref(),
                          Ok("legacy") | Ok("0") | Ok("hg1") | Ok("1") | Ok("hg3") | Ok("3"));
    let escape = wp_env_flag(crate::opt!("w4qsa-off"), false);
    let opt_in = wp_env_flag(crate::opt!("w4qsa"), false);
    let on = opt_in && !escape && !pinned;
    let why = if on { "row-shared union (xq_attn_sel_union; opt-in --w4qsa=1)" }
        else if escape { if opt_in { "OFF (--w4qsa-off=1 overrides --w4qsa=1)" } else { "OFF (--w4qsa-off=1)" } }
        else if pinned { "OFF (--exl3-sel-gather pins the old kernels)" }
        else { "OFF (default) — hg1/hg3 dispatch; --w4qsa=1 = row-shared union" };
    (on, why)
}
/// FwdModel::load: the W4/QSA boot line (the feature marker; unchanged text), once.
pub(super) fn w4qsa_announce() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let (on, why) = w4qsa_env_rule();
        let xc = w4qsa_xcheck();
        crate::dprintln!("[w4qsa] QSA verify gather: {why}{}",
                 if on && xc { " | XCHECK on (eager bitwise new-vs-old per call)" }
                 else if xc { " (--w4qsa-xcheck ignored while OFF)" } else { "" });
    });
}
pub(crate) const T_W4QSA_ON: tune::TunableDef = tune::TunableDef {
    slot: 77, id: "w4qsa.on", class: tune::Class::S, scope: tune::Scope::Width, fams: &[tune::Fam::Verify],
    domain: &[0, 1], default: 0, valid: |cx, _| cx.m != 1,
    xcheck: "--probe-exl3-binv EXL3-ATTN-SPARSE (union vs hg3 / hg1 / splitk4) + --w4qsa-xcheck",
    env: "--w4qsa|--w4qsa-off",
    // an alias (or a SEL_GATHER pin) set = the old rule exactly; none = the default (off)
    env_parse: || {
        let pinned = matches!(tune::env_str(crate::opt!("exl3-sel-gather")).as_deref(),
                              Some("legacy") | Some("0") | Some("hg1") | Some("1") | Some("hg3") | Some("3"));
        (pinned || tune::env_str(crate::opt!("w4qsa")).is_some() || tune::env_str(crate::opt!("w4qsa-off")).is_some())
            .then(|| w4qsa_env_rule().0 as i32)
    },
    rev: 1, wp: "W4/QSA",
};
/// --w4qsa-xcheck=1: eager bitwise diff of every union launch against the old dispatch.
pub(super) fn w4qsa_xcheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| wp14_xcheck(crate::opt!("w4qsa-xcheck")))
}
/// Mirrors of kernels/exl3_bench.cu XQ_QU_* (the union gather's contract and smem carve).
pub(crate) const QU_HPW: usize = 4;
pub(crate) const QU_S: usize = 2;
pub(crate) const QU_NE: usize = 8;
pub(crate) const QU_MAXR: usize = 16;
pub(crate) const QU_MAXL: usize = 128;
pub(crate) const QU_WW: usize = 8192 / 32;
/// Dynamic smem of `xq_attn_sel_union` for (KV format, hd, rows m, split_len L): ring NE*2*rb
/// | f32 tile S*2*hd*4 (non-f32 formats) | sel, u, msk m*L each | bm, wp WW each | n, slot MAXR
/// each | red 32 — the kernel's carve, byte for byte.
pub(crate) fn qsa_union_smem(fmt: u32, hd: usize, m: usize, l: usize) -> usize {
    QU_NE * 2 * kv_rowbytes(fmt, hd) + if fmt == KV_F32 { 0 } else { QU_S * 2 * hd * 4 }
        + 4 * (3 * m * l + 2 * QU_WW + 2 * QU_MAXR + 32)
}
/// The union gather's launch contract (else the old dispatch): hd 256 (lane owns 8 dims), whole
/// 4-head groups, 1..=QU_MAXR rows, split_len <= QU_MAXL, K/V rows <= 1024 B, smem <= 48 KiB
/// (no opt-in needed). Returns the dynamic smem bytes.
pub(crate) fn qsa_union_fits(fmt: u32, nh: usize, nkv: usize, hd: usize, m: usize, l: usize) -> Option<usize> {
    let gqa = nh / nkv.max(1);
    let smem = qsa_union_smem(fmt, hd, m, l);
    (hd == 256 && nkv >= 1 && gqa * nkv == nh && gqa % QU_HPW == 0 && (1..=QU_MAXR).contains(&m)
        && (1..=QU_MAXL).contains(&l) && kv_rowbytes(fmt, hd) <= 1024 && smem <= 48 * 1024).then_some(smem)
}
/// PQ8 (prefill q8 diet + SURPASS_PLAN WP06 item 1) — three bit-identical twins in the chunked
/// prefill, each with its old kernel kept behind an escape:
///   0 = --pq8-kvw    : xq_attn_kv_prefill_rows (grid c*nkv) for the serial xq_attn_kv_prefill
///                         (every format; trunk layers + the MTP head fill);
///   1 = --pq8-flash  : xq_attn_prefill_flash256v (vector staging + float4 scores) for
///                         xq_attn_prefill_flash256 (every format; the dense rows <= qsa_limit).
///                         p5d: DEFAULT ON again — PQ8-v2 is bitwise on the GPU (.14, 22022ab:
///                         XCHECK flash differ 0 every layer f32 + q8, all 8 hashes exact); p3
///                         had it OFF because v1 was not (p4c @ 6300674: 6,039,782 of 12,582,912
///                         halves differed). `=0` = flash256;
///                         `=causal` (opt-in, NOT bitwise): xq_attn_prefill_flash256c instead,
///                         and (w5/PFIX) xq_attn_prefill_flash128c for hd=128 dense rows t >= 16
///                         (default hd=128: the sweep xq_attn_prefill_q, unchanged);
///   2 = --pq8-gather : xq_attn_sel_pq8 for xq_attn_sel_splitk4 (the q8 and q4k layouts; the
///                         QSA-sparse rows).
/// `=0` turns one item off; --pq8-off=1 turns all three off (= the pre-PQ8 kernels exactly).
/// TUNE T0c (p5e): the three items are registry entries per prefill chunk class (pq8.kvw /
/// pq8.flash / pq8.gather — class S, bitwise twins); the aliases decide exactly as before whenever one
/// is set, and the boot line still prints once, on the first lookup. flash256c (=causal) is the
/// N-class pq8.causal (output-changing: never from a table, never searched).
pub(super) fn pq8_item(i: usize) -> bool {
    pq8_announce();
    let t = match i { 0 => &T_PQ8_KVW, 1 => &T_PQ8_FLASH, 2 => &T_PQ8_GATHER, _ => return false };
    tune::get_cur(t) != 0
}
const PQ8_NAMES: [crate::opts::OptId; 3] = [crate::opt!("pq8-kvw"), crate::opt!("pq8-flash"), crate::opt!("pq8-gather")];
/// The aliases' value of item i (the pre-registry parse, verbatim).
fn pq8_env_item(i: usize) -> bool {
    !wp_env_flag(crate::opt!("pq8-off"), false) && wp_env_flag(PQ8_NAMES[i], true)
}
/// Registry env parser of item i: the old value when --pq8-off or the item's alias is set.
fn pq8_env_parse(i: usize) -> Option<i32> {
    (tune::env_str(crate::opt!("pq8-off")).is_some() || tune::env_str(PQ8_NAMES[i]).is_some()).then(|| pq8_env_item(i) as i32)
}
const PQ8_XCHECK: &str = "--pq8-xcheck (old kernel reference + poisoned new output, bit/byte diff per call, eager)";
pub(crate) const T_PQ8_KVW: tune::TunableDef = tune::TunableDef {
    slot: 78, id: "pq8.kvw", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: PQ8_XCHECK,
    env: "--pq8-kvw|--pq8-off", env_parse: || pq8_env_parse(0), rev: 1, wp: "PQ8",
};
pub(crate) const T_PQ8_FLASH: tune::TunableDef = tune::TunableDef {
    slot: 79, id: "pq8.flash", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    // 0 = xq_attn_prefill_flash256, 1 = its bitwise twin flash256v (PQ8-v2, .14 22022ab: differ 0)
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: PQ8_XCHECK,
    env: "--pq8-flash|--pq8-off", env_parse: || pq8_env_parse(1), rev: 1, wp: "PQ8-v2",
};
pub(crate) const T_PQ8_GATHER: tune::TunableDef = tune::TunableDef {
    slot: 80, id: "pq8.gather", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: PQ8_XCHECK,
    env: "--pq8-gather|--pq8-off", env_parse: || pq8_env_parse(2), rev: 1, wp: "PQ8",
};
/// N (numerics): flash256c's causal staging CHANGES the served bytes (an owner-consent quality A/B).
pub(crate) const T_PQ8_CAUSAL: tune::TunableDef = tune::TunableDef {
    slot: 81, id: "pq8.causal", class: tune::Class::N, scope: tune::Scope::Global, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "drift table (N: output-changing, owner consent)",
    env: "--pq8-flash", env_parse: || tune::env_str(crate::opt!("pq8-flash")).map(|v| (v == "causal") as i32),
    rev: 1, wp: "PQ8-v2",
};
/// The PQ8 boot line (unchanged text, from the aliases), once.
fn pq8_announce() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let off = wp_env_flag(crate::opt!("pq8-off"), false);
        let v = [pq8_env_item(0), pq8_env_item(1), pq8_env_item(2)];
        let s = |b: bool| if b { "on" } else { "OFF" };
        let explicit = crate::opts::var_os(crate::opt!("pq8-flash")).is_some();
        crate::deprintln!("[pq8] prefill: kv writer rows {} | flash256v {} ({}){} | q8 gather {} (8-bit KV only){}", s(v[0]), s(v[1]),
                  if off { "--pq8-off=1" } else if !v[1] { "--pq8-flash=0 = flash256" }
                  else if pq8_flash_causal() { if explicit { "--pq8-flash=causal" } else { "DEFAULT causal (owner 2026-09-27); --pq8-flash=1 = bitwise twin" } }
                  else if explicit { "--pq8-flash set; bitwise flash256 twin (old bytes)" }
                  else { "default" },
                  if v[1] && pq8_flash_causal() { " (CAUSAL staging: flash256c / hd=128 flash128c, opt-in, not bitwise)" } else { "" },
                  s(v[2]),
                  if pq8_xcheck() { " | XCHECK on (eager bitwise new-vs-old per call)" } else { "" });
    });
}
/// --pq8-flash=causal: serve the dense prefill rows with xq_attn_prefill_flash256c — the flash256v
/// body with causal (block-wide) K/V tile staging. flash256 (and its bitwise twin flash256v) zero
/// the dims of earlier warps in the diagonal tile's key/value rows (PQ8-v2 note in exl3_bench.cu),
/// so this CHANGES the served bytes. Owner ruling 2026-09-27: causal is now the DEFAULT
/// (escape --pq8-flash=1 = the bitwise flash256v twin = the pre-p6c served bytes).
/// w5/PFIX: the same switch covers hd=128 — dense rows (t >= 16) take xq_attn_prefill_flash128c
/// (the same body at the hd=128 shape) instead of the sweep xq_attn_prefill_q. The defective scalar
/// hd=128 flash (xq_attn_prefill_flash) has no caller; the hd=128 default stays the sweep. (No EXL3
/// pack on the boxes has hd=128 today: the Flash-Next trunk and its MTP head are hd=256.)
pub(super) fn pq8_flash_causal() -> bool {
    // Owner ruling 2026-09-27: CAUSAL prefill staging is the DEFAULT (a correctness fix — the
    // bitwise flash256v twin zeroes the dims of earlier warps in the diagonal tile's K/V rows).
    // Escapes: --pq8-flash=1 = the bitwise flash256v twin (old served bytes),
    // --pq8-flash=0 = flash256 (the pre-PQ8-v2 kernel), --pq8-flash=causal = explicit.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match crate::opts::var(crate::opt!("pq8-flash")).as_deref() {
        Ok("causal") | Err(_) => true, // owner ruling 2026-09-27: causal is the DEFAULT
        Ok("1") | Ok("true") | Ok("flash256v") => false,
        Ok("0") | Ok("off") | Ok("false") => false, // handled by pq8_item as item-off (flash256)
        Ok(other) => {
            crate::deprintln!("[pq8] --pq8-flash='{other}' is not a known value — using the BITWISE flash256v twin (old bytes), not causal");
            false
        }
    })
}
/// --pq8-xcheck=1: per prefill call, run the OLD kernel as the reference, poison the new
/// kernel's output region, run the NEW kernel, and count differing bytes/bits (eager only).
/// POSTURE1 `--exl3-sel-tc on|off` (default off): the prefill sparse gather on tensor cores (xq_attn_sel_tc) instead of
/// xq_attn_sel_pq8. Output-changing (f16 operands, summation order); the decode / verify gathers are untouched.
pub(crate) fn sel_tc_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::on(crate::opt!("exl3-sel-tc")))
}
/// seltc6 `--exl3-sel-tc-xcheck` (diagnostic): every tensor-core gather launch also runs today's gather and prints the difference.
pub(super) fn sel_tc_xcheck_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-sel-tc-xcheck")).is_ok())
}
pub static SEL_TC_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// seltc6: the gather launches served by xq_attn_sel_tcu8 (6 query heads per KV head: Flash-Next at TP=4).
pub static SEL_TC8_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SEL_TC_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// 1 = --exl3-sel-tc on was asked but this boot cannot serve it (boot line says why); today's gather runs.
pub static SEL_TC_FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Prefill gather launches that ran today's kernel although --exl3-sel-tc on was asked (the fallback in action).
pub static SEL_TC_FALLBACK_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Once per boot: which entry serves the prefill sparse gather (the first sparse prefill chunk says it).
pub(super) fn sel_tc_announce(entry: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let tc = entry.starts_with("xq_attn_sel_tc");
    ONCE.call_once(|| crate::rprintln!("[exl3] prefill sparse-attention gather served by {entry} (--exl3-sel-tc {}); {}",
        if sel_tc_on() { "on" } else { "off" },
        if tc { "/metrics velogb10_exl3_sel_tc_rows_total counts the rows served on the tensor cores" }
        else if sel_tc_on() { "the tensor-core gather is NOT serving: /metrics velogb10_exl3_sel_tc_fallback_calls_total counts these launches" }
        else { "today's gather (the default)" }));
}
pub(super) fn pq8_xcheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| wp14_xcheck(crate::opt!("pq8-xcheck")))
}
/// Eager-only bitwise cross-checks (new vs old kernel, printed per call with a mismatch, and the
/// first few calls). They need host syncs, so they are skipped while a CUDA graph is being
/// captured — run them with --exl3-no-graph=1 (and --graph-precapture off) for full coverage.
pub(super) fn wp14_xcheck(o: crate::opts::OptId) -> bool {
    crate::opts::var(o).map_or(false, |v| !v.is_empty() && v != "0")
}

/// --qsa-key-rope full|half (VIS-3; see the option's help).
pub fn qsa_key_rope_full() -> bool {
    crate::opts::var(crate::opt!("qsa-key-rope")).map_or(true, |v| v.trim() != "half")
}
