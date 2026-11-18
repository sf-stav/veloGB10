//! Tunables (`tune::TunableDef` entries) and their gate predicates for moe-fh, hc mix4 / rb, WP09, WP11 and the
//! W4HC (hc fusion) levers, plus the W4HC capacity / launch helpers. The registry itself is `exl3_tune`.
//! See `mod.rs` for the file map.

use super::*;

/// Fused per-expert Hadamards in the served expert GEMM — exact but measured NEUTRAL (each of an
/// expert's CTAs recomputes the input transform; the separate launches were off the critical
/// path: verify 46.49/46.61 fused vs 46.47/46.32 ms). Default OFF; --moe-fh=1 enables.
pub(super) fn moe_fh_on() -> bool {
    tune::get_cur(&T_MOE_FH) != 0
}
// TUNE T0 migrations: each legacy env switch is a registry entry whose env parser reproduces the old
// comparison exactly and whose default is today's value (a bitwise no-op; the alias still wins).
pub(crate) const T_MOE_FH: tune::TunableDef = tune::TunableDef {
    slot: 3, id: "moe.fh", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "--probe-exl3-binv (grouped a1b3_fh vs a1b3)",
    env: "--moe-fh", env_parse: || tune::env_on_if_1(crate::opt!("moe-fh")), rev: 1, wp: "A5-D3",
};
/// Decode families that run the MoE / hc / routing glue (every m <= 16 forward).
pub(crate) const FAMS_DECODE: &[tune::Fam] = &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass,
                                               tune::Fam::DraftChain, tune::Fam::Util];

/// xq_hc_mix4 (shared inject reduction tree, bitwise) — default on; --hc-mix4=0 = xq_hc_mix.
pub(super) fn hc_mix4_on() -> bool {
    tune::get_cur(&T_HC_MIX4) != 0
}
pub(crate) const T_HC_MIX4: tune::TunableDef = tune::TunableDef {
    slot: 6, id: "hc.mix4", class: tune::Class::S, scope: tune::Scope::Width,
    fams: &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass, tune::Fam::DraftChain,
            tune::Fam::Reprime, tune::Fam::Prefill, tune::Fam::Util],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--hc-step-check (mix4 vs mix)",
    env: "--hc-mix4", env_parse: || tune::env_on_unless_0(crate::opt!("hc-mix4")), rev: 1, wp: "S-A3-hc",
};

/// WP09 escapes (default ON, bitwise either way). --wp09-off=1 restores BOTH old paths;
/// --wp09-ple=0 / --wp09-argmax=0 restore one. PLE: read at LOAD (off = no k-major weight
/// copies, the row-major xq_gemm_f16_rows pair). ARGMAX: off = m single-block xq_argmax launches.
fn wp09_off() -> bool {
    tune::get_cur(&T_WP09_OFF) != 0
}
pub(super) fn wp09_ple_on() -> bool {
    !wp09_off() && tune::get_cur(&T_WP09_PLE) != 0
}
pub(super) fn wp09_argmax_on() -> bool {
    !wp09_off() && tune::get_cur(&T_WP09_ARGMAX) != 0
}
pub(crate) const T_WP09_OFF: tune::TunableDef = tune::TunableDef {
    slot: 9, id: "wp09.off", class: tune::Class::S, scope: tune::Scope::Global,
    fams: &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass, tune::Fam::DraftChain, tune::Fam::Util],
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "--wp09-xcheck",
    env: "--wp09-off", env_parse: || tune::env_on_if_1(crate::opt!("wp09-off")), rev: 1, wp: "WP09",
};
pub(crate) const T_WP09_PLE: tune::TunableDef = tune::TunableDef {
    slot: 10, id: "wp09.ple", class: tune::Class::S, scope: tune::Scope::Load, fams: &[tune::Fam::Util],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--wp09-xcheck (ple_key / ple_val)",
    env: "--wp09-ple", env_parse: || tune::env_on_unless_0(crate::opt!("wp09-ple")), rev: 1, wp: "WP09",
};
pub(crate) const T_WP09_ARGMAX: tune::TunableDef = tune::TunableDef {
    slot: 11, id: "wp09.argmax", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--wp09-xcheck (argmax)",
    env: "--wp09-argmax", env_parse: || tune::env_on_unless_0(crate::opt!("wp09-argmax")), rev: 1, wp: "WP09",
};
/// WP09 diagnostic: --wp09-xcheck=1 re-runs the OLD kernels after every EAGER (non-captured)
/// new-path launch into side buffers and prints a bitwise diff (use with --exl3-no-graph=1 —
/// captured launches are skipped: the check syncs the device).
pub(super) fn wp09_xcheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("wp09-xcheck")).as_deref() == Ok("1"))
}
/// WP09: blocks per row of xq_argmax_rows (kernel cap 32; Scratch.am_part holds w * 32 keys).
pub(super) const XQ_AM_NB_MAX: usize = 32;
pub(super) fn wp09_am_nb(m: usize) -> usize {
    match tune::get_cur(&T_WP09_AM_NB) {
        v if v > 0 => (v as usize).min(XQ_AM_NB_MAX),
        _ => if m <= 2 { 32 } else { 16 },
    }
}
/// TUNE T0 (p3c): blocks per row of xq_argmax_rows (0 = the built-in rule: 32 at m <= 2, else 16).
/// The same ids for every nb (total comparator order; probe_argmax_rows sweeps nb {1,7,16,32}).
pub(crate) const T_WP09_AM_NB: tune::TunableDef = tune::TunableDef {
    slot: 45, id: "wp09.am_nb", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 16, 32], default: 0, valid: tune::valid_any,
    xcheck: "--probe-exl3-binv WP09 argmax_rows (nb sweep) + --wp09-xcheck",
    env: "", env_parse: tune::env_none, rev: 1, wp: "WP09",
};

/// WP09: [N][K] row-major f16 -> [K/16][N][16] (xq_gemm_f16_rows_km's layout). Pure permutation.
pub(crate) fn relayout_km16(src: &[u16], n: usize, k: usize) -> Vec<u16> {
    assert!(k % 16 == 0 && src.len() == n * k, "relayout_km16: {} != {n}x{k}", src.len());
    let mut dst = vec![0u16; n * k];
    for c in 0..k / 16 {
        for r in 0..n {
            dst[(c * n + r) * 16..(c * n + r) * 16 + 16].copy_from_slice(&src[r * k + c * 16..r * k + c * 16 + 16]);
        }
    }
    dst
}

/// WP09 XCHECK: bitwise diff of two f16 planes (new vs old path), one line.
pub(super) fn wp09_diff_u16(tag: &str, m: usize, new: &[u16], old: &[u16]) {
    let (mut mism, mut maxd, mut first) = (0usize, 0f32, None);
    for (i, (a, b)) in new.iter().zip(old.iter()).enumerate() {
        if a != b {
            mism += 1;
            if first.is_none() { first = Some((i, *a, *b)); }
            let d = (f16::from_bits(*a).to_f32() - f16::from_bits(*b).to_f32()).abs();
            if d.is_nan() || d > maxd { maxd = d; }
        }
    }
    crate::dprintln!("WP09_XCHECK {tag} m={m} n={} mism={mism} max_abs_diff={maxd:e} first(idx,new,old)={first:?}",
             new.len().min(old.len()));
}

/// Row-batched int8 hc mixer for m >= 2 (default on; --hc-rb=0 = the per-row grid).
pub(super) fn hc_rb_on() -> bool {
    tune::get_cur(&T_HC_RB) != 0
}
/// Row-batched hc mixer grid (xq_hc_fuse_i8_rb / _rbk): CTAs, clamped by the caller to the measured
/// co-residency (the grid barrier needs every CTA resident). The bodies grid-stride their outputs
/// with a per-output warp reduction, so the CTA count never changes a bit.
pub(super) fn hc_bxr() -> u32 {
    tune::get_cur(&T_HC_BXR).max(1) as u32
}
pub(crate) const T_HC_BXR: tune::TunableDef = tune::TunableDef {
    slot: 46, id: "hc.bxr", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[24, 32, 40, 48], default: 40, valid: |cx, _| cx.m >= 2 && cx.m <= 8,
    xcheck: "--hc-step-check per value (+ the per-sample live digest)",
    env: "", env_parse: tune::env_none, rev: 1, wp: "S-A3-hc",
};
pub(crate) const T_HC_RB: tune::TunableDef = tune::TunableDef {
    slot: 5, id: "hc.rb", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--hc-step-check (row-batched vs m-grid fuse)",
    env: "--hc-rb", env_parse: || tune::env_on_unless_0(crate::opt!("hc-rb")), rev: 1, wp: "S-A3-hc",
};

// ---- WP11 (2026-09-26): hc mixer rungs — every rung is bitwise (kernels/exl3_bench.cu WP11
// header). --wp11-off=1 = the exact pre-WP11 hc path (row-major q_up, xq_hc_fuse_i8 /
// xq_hc_fuse_i8_rb, xq_hc_mix4). Per-rung A/B (all default ON):
//   --wp11-r1=0|1|2  L2 prefetch of the phase-B weights at fuse entry (off / per-thread
//                       prefetch.global.L2 / cp.async.bulk.prefetch.L2)
//   --wp11-r2=0      xq_hc_mix4 instead of the regridded xq_hc_mix_rg
//   --wp11-r3=0      no k-major q_up relayout (load-time) => the old fuse kernels (R1/R4 moot)
//   --wp11-r4=0      no hn smem staging in the row-batched twin
//   --wp11-cvt=0     I2F instead of the exact PRMT int8 convert in the m-grid twin
// --hc-step-check=1: eager bit-diff (old path vs new) of dd/uu/x/inj on every NON-captured
// hc_pre call (run with --exl3-no-graph=1 for full coverage); prints HC_STEP_CHECK lines.
pub(super) fn wp11_off() -> bool {
    tune::get_cur(&T_WP11_OFF) != 0
}
pub(super) fn wp11_pf_mode() -> i32 {
    tune::get_cur(&T_WP11_R1)
}
pub(super) fn wp11_r2_on() -> bool {
    !wp11_off() && tune::get_cur(&T_WP11_R2) != 0
}
pub(super) fn wp11_r3_on() -> bool {
    !wp11_off() && tune::get_cur(&T_WP11_R3) != 0
}
pub(super) fn wp11_r4_on() -> bool {
    tune::get_cur(&T_WP11_R4) != 0
}
pub(super) fn wp11_cvt_i2f() -> bool {
    tune::get_cur(&T_WP11_CVT) == 0
}
// TUNE T0: the WP11 switches (the composites — OFF gating R2/R3 — stay in the helpers above).
pub(crate) const T_WP11_OFF: tune::TunableDef = tune::TunableDef {
    slot: 14, id: "wp11.off", class: tune::Class::S, scope: tune::Scope::Global,
    fams: &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass, tune::Fam::DraftChain,
            tune::Fam::Reprime, tune::Fam::Util],
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "--hc-step-check",
    env: "--wp11-off", env_parse: || tune::env_on_if_1(crate::opt!("wp11-off")), rev: 1, wp: "WP11",
};
pub(crate) const T_WP11_R1: tune::TunableDef = tune::TunableDef {
    slot: 15, id: "wp11.r1", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1, 2], default: 1, valid: tune::valid_any, xcheck: "--hc-step-check",
    env: "--wp11-r1",
    env_parse: || tune::env_str(crate::opt!("wp11-r1")).map(|v| match v.as_str() { "0" => 0, "2" => 2, _ => 1 }),
    rev: 1, wp: "WP11",
};
pub(crate) const T_WP11_R2: tune::TunableDef = tune::TunableDef {
    slot: 16, id: "wp11.r2", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--hc-step-check (mix_rg vs mix4)",
    env: "--wp11-r2", env_parse: || tune::env_on_unless_0(crate::opt!("wp11-r2")), rev: 1, wp: "WP11",
};
pub(crate) const T_WP11_R3: tune::TunableDef = tune::TunableDef {
    slot: 17, id: "wp11.r3", class: tune::Class::S, scope: tune::Scope::Load, fams: &[tune::Fam::Util],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--hc-step-check (k-major twins)",
    env: "--wp11-r3", env_parse: || tune::env_on_unless_0(crate::opt!("wp11-r3")), rev: 1, wp: "WP11",
};
pub(crate) const T_WP11_R4: tune::TunableDef = tune::TunableDef {
    slot: 18, id: "wp11.r4", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--hc-step-check",
    env: "--wp11-r4", env_parse: || tune::env_on_unless_0(crate::opt!("wp11-r4")), rev: 1, wp: "WP11",
};
pub(crate) const T_WP11_CVT: tune::TunableDef = tune::TunableDef {
    slot: 19, id: "wp11.cvt_prmt", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--hc-step-check (prmt vs i2f)",
    env: "--wp11-cvt", env_parse: || tune::env_on_unless_0(crate::opt!("wp11-cvt")), rev: 1, wp: "WP11",
};
pub(super) fn hc_step_check_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("hc-step-check")).is_ok_and(|v| v != "0"))
}

// ---- HC (decode wave 4, 2026-09-27): regridded + mix-fused hc mixer (xq_hc_w4) and the fused
// inject+norm (xq_hc_inj_norm) — both bitwise twins of the p4c path (kernels/exl3_bench.cu HC
// header). Escapes (all default ON; read once):
//   --w4hc-off=1   the exact p4c hc path (WP11 fuse twins + xq_hc_mix_rg + xq_hc_inject/xq_hc_norm)
//   --w4hc-fuse=0  lever (a) off: WP11 fuse twins (then the mix is separate too)
//   --w4hc-mix=0   lever (b2) off: xq_hc_w4 writes uu, the old xq_hc_mix_rg follows
//   --w4hc-inj=0   lever (b1) off: separate xq_hc_inject + xq_hc_norm
//   --w4hc-m1=0    m = 1 (decode/draft) stays on xq_hc_fuse_i8k (+ mix_rg)
// Geometry A/B (no effect on bits): --w4hc-g=<ctas> (default 96; raised to the phase-B minimum
// ceil(h/32) = 80, never above the measured co-residency x 5/6 — else the mixer is OFF), --w4hc-pf=0|1|2 (phase-B L2 prefetch: off / per-sector / bulk; default 1),
// --w4hc-la=0..7 (phase-A L2 look-ahead in 1024-k chunks; default 0).
// --w4hc-xcheck=1: on every NON-captured hc_pre (use --exl3-no-graph=1) run the p4c path and
// the new one on the same inputs (resid and inj restored in between) and bit-diff resid/hn/dd/uu/x/inj.
// TUNE T0c (p5e): the switches and the geometry are registry entries (per width, every value
// bitwise); the aliases above keep their exact parses and still win (diagnostics).
fn w4hc_off() -> bool {
    tune::get_cur(&T_W4HC_OFF) != 0
}
pub(super) fn w4hc_fuse_on() -> bool {
    !w4hc_off() && tune::get_cur(&T_W4HC_FUSE) != 0
}
pub(super) fn w4hc_mix_on() -> bool {
    tune::get_cur(&T_W4HC_MIX) != 0
}
pub(super) fn w4hc_inj_on() -> bool {
    !w4hc_off() && tune::get_cur(&T_W4HC_INJ) != 0
}
pub(super) fn w4hc_m1_on() -> bool {
    tune::get_cur(&T_W4HC_M1) != 0
}
fn w4hc_grid_target() -> u32 {
    tune::get_cur(&T_W4HC_G).max(1) as u32
}
/// Decode families that run the hc mixers (the trunk forward and the draft head).
const FAMS_HC: &[tune::Fam] = FAMS_DECODE;
const W4HC_XCHECK: &str = "--probe-exl3-binv EXL3-HC-W4 + --w4hc-xcheck (p4c vs w4, eager)";
pub(crate) const T_W4HC_OFF: tune::TunableDef = tune::TunableDef {
    slot: 52, id: "w4hc.off", class: tune::Class::S, scope: tune::Scope::Global, fams: FAMS_HC,
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: W4HC_XCHECK,
    env: "--w4hc-off", env_parse: || tune::env_on_if_1(crate::opt!("w4hc-off")), rev: 1, wp: "W4/HC",
};
pub(crate) const T_W4HC_FUSE: tune::TunableDef = tune::TunableDef {
    slot: 53, id: "w4hc.fuse", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_HC,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: W4HC_XCHECK,
    env: "--w4hc-fuse", env_parse: || tune::env_on_unless_0(crate::opt!("w4hc-fuse")), rev: 1, wp: "W4/HC",
};
pub(crate) const T_W4HC_MIX: tune::TunableDef = tune::TunableDef {
    slot: 54, id: "w4hc.mix", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_HC,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: W4HC_XCHECK,
    env: "--w4hc-mix", env_parse: || tune::env_on_unless_0(crate::opt!("w4hc-mix")), rev: 1, wp: "W4/HC",
};
pub(crate) const T_W4HC_INJ: tune::TunableDef = tune::TunableDef {
    slot: 55, id: "w4hc.inj", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_HC,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: W4HC_XCHECK,
    env: "--w4hc-inj", env_parse: || tune::env_on_unless_0(crate::opt!("w4hc-inj")), rev: 1, wp: "W4/HC",
};
pub(crate) const T_W4HC_M1: tune::TunableDef = tune::TunableDef {
    slot: 56, id: "w4hc.m1", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_HC,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: W4HC_XCHECK,
    env: "--w4hc-m1", env_parse: || tune::env_on_unless_0(crate::opt!("w4hc-m1")), rev: 1, wp: "W4/HC",
};
/// xq_hc_w4 CTA target (raised to ceil(h/32), clamped to co-residency x 5/6 at the launch — a value
/// the clamp maps onto another one's grid is an identical graph: INERT in the tuner).
pub(crate) const T_W4HC_G: tune::TunableDef = tune::TunableDef {
    slot: 57, id: "w4hc.g", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_HC,
    domain: &[80, 96, 112, 128], default: 96, valid: tune::valid_any, xcheck: W4HC_XCHECK,
    env: "--w4hc-g",
    // the old parse exactly: an unparsable / zero value means the default 96
    env_parse: || tune::env_str(crate::opt!("w4hc-g"))
        .map(|v| v.parse::<u32>().ok().filter(|&g| g >= 1).unwrap_or(96).min(i32::MAX as u32) as i32),
    rev: 1, wp: "W4/HC",
};
pub(crate) const T_W4HC_PF: tune::TunableDef = tune::TunableDef {
    slot: 58, id: "w4hc.pf", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_HC,
    // phase-B L2 prefetch: off / per-sector / bulk
    domain: &[0, 1, 2], default: 1, valid: tune::valid_any, xcheck: W4HC_XCHECK,
    env: "--w4hc-pf",
    env_parse: || tune::env_str(crate::opt!("w4hc-pf")).map(|v| match v.as_str() { "0" => 0, "2" => 2, _ => 1 }),
    rev: 1, wp: "W4/HC",
};
pub(crate) const T_W4HC_LA: tune::TunableDef = tune::TunableDef {
    slot: 59, id: "w4hc.la", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_HC,
    // phase-A L2 look-ahead in 1024-k chunks (the kernel takes 0..7)
    domain: &[0, 1, 2, 4], default: 0, valid: tune::valid_any, xcheck: W4HC_XCHECK,
    env: "--w4hc-la",
    env_parse: || tune::env_str(crate::opt!("w4hc-la")).map(|v| v.parse::<i32>().ok().unwrap_or(0).clamp(0, 7)),
    rev: 1, wp: "W4/HC",
};
/// HC: the xq_hc_w4 grid for a measured co-residency `cap` and hidden size `h`: --w4hc-g
/// (default 96) raised to the phase-B contract minimum ceil(h/32) (a CTA owns <= 32 columns) and
/// clamped to cap x 5/6 (the S-A3-f-f residency margin every grid-barrier kernel here keeps);
/// 0 when the minimum does not fit under the margin (=> the W4 mixer is off for ALL m).
pub(crate) fn w4hc_default_grid(cap: u32, h: usize) -> u32 {
    let gmin = h.div_ceil(32) as u32;
    let gmax = (cap * 5) / 6;
    let g = w4hc_grid_target().max(gmin).min(gmax);
    if g >= gmin { g } else { 0 }
}
/// xq_hc_w4 flags bits 2-6 (prefetch mode, look-ahead) — pure geometry, never a bit of output.
/// TUNE T0c: w4hc.pf / w4hc.la in the current tune context (per width).
pub(crate) fn w4hc_geom_flags() -> i32 {
    let pf = tune::get_cur(&T_W4HC_PF).clamp(0, 2);
    let la = tune::get_cur(&T_W4HC_LA).clamp(0, 7);
    (pf << 2) | (la << 4)
}
pub(super) fn w4hc_xcheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("w4hc-xcheck")).is_ok_and(|v| v != "0"))
}

/// HC: raw handle of xq_hc_w4 (15 params: past the typed launcher's cap) with the shared-memory
/// carveout pinned to the maximum BEFORE any occupancy query or launch, so the co-residency the
/// grid barrier relies on is measured on the configuration every launch actually gets (a driver-
/// chosen smaller carveout could otherwise fit fewer CTAs/SM than the occupancy API reported).
/// None = stale PTX or the attribute refused => the W4 mixer is never launched.
pub(crate) fn w4hc_fn() -> Option<cudarc::driver::sys::CUfunction> {
    use cudarc::driver::sys;
    #[derive(Clone, Copy)]
    struct FnH(sys::CUfunction);
    unsafe impl Send for FnH {}
    unsafe impl Sync for FnH {}
    static F: std::sync::OnceLock<Option<FnH>> = std::sync::OnceLock::new();
    F.get_or_init(|| {
        let f = xq_raw_fn("xq_hc_w4")?;
        let r = unsafe {
            sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100)
        };
        if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[w4hc] carveout pin failed ({r:?})"); return None; }
        Some(FnH(f))
    }).map(|h| h.0)
}

/// HC: co-resident capacity of xq_hc_w4 (blocks/SM at 256 threads, 0 dynamic smem, on the pinned
/// handle) x SM count. 0 on any failure => the W4 mixer is disabled for ALL m.
pub(crate) fn w4hc_capacity() -> u32 {
    use cudarc::driver::sys;
    let Some(f) = w4hc_fn() else { return 0; };
    let mut bps: std::ffi::c_int = 0;
    if unsafe { sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut bps, f, 256, 0) } != sys::CUresult::CUDA_SUCCESS {
        return 0;
    }
    let mut dev_h: sys::CUdevice = 0;
    if unsafe { sys::cuCtxGetDevice(&mut dev_h) } != sys::CUresult::CUDA_SUCCESS { return 0; }
    let mut sms: std::ffi::c_int = 0;
    let r = unsafe {
        sys::cuDeviceGetAttribute(&mut sms, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_h)
    };
    if r != sys::CUresult::CUDA_SUCCESS { return 0; }
    (bps.max(0) as u32).saturating_mul(sms.max(0) as u32)
}

/// HC: xq_hc_w4 launch (raw; PDL-aware like xqlaunch!). Pointers as u64; winj 0 = no inject gates
/// (the final mixer). flags: bit 0 MIX, bit 1 write uu, bits 2-6 geometry (w4hc_geom_flags).
#[allow(clippy::too_many_arguments)]
pub(crate) fn w4hc_launch(stream: cudarc::driver::sys::CUstream, grid: u32,
                          ptrs: [u64; 11], hcn: f32, lr: usize, m: usize, h: usize, hc: usize, flags: i32) -> Result<()> {
    let f = w4hc_fn().ok_or_else(|| anyhow::anyhow!("xq_hc_w4 raw fn unavailable (stale PTX?)"))?;
    let [mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8, mut a9, mut a10] = ptrs;
    let mut a_hcn = hcn;
    let mut a_lrm = (lr as i32) | ((m as i32) << 16);
    let mut a_hhc = (h as i32) | ((hc as i32) << 16);
    let mut a_fl = flags;
    let mut args: [*mut std::ffi::c_void; 15] = [
        &mut a0 as *mut u64 as *mut _, &mut a1 as *mut u64 as *mut _, &mut a2 as *mut u64 as *mut _,
        &mut a3 as *mut u64 as *mut _, &mut a4 as *mut u64 as *mut _, &mut a5 as *mut u64 as *mut _,
        &mut a6 as *mut u64 as *mut _, &mut a7 as *mut u64 as *mut _, &mut a8 as *mut u64 as *mut _,
        &mut a9 as *mut u64 as *mut _, &mut a10 as *mut u64 as *mut _,
        &mut a_hcn as *mut f32 as *mut _, &mut a_lrm as *mut i32 as *mut _, &mut a_hhc as *mut i32 as *mut _,
        &mut a_fl as *mut i32 as *mut _,
    ];
    if pdl_on() {
        unsafe { launch_pss_raw(stream, f, (grid, 1, 1), (256, 1, 1), 0, &mut args) }
            .context("launch xq_hc_w4 (pdl)")
    } else {
        let r = unsafe {
            cudarc::driver::sys::cuLaunchKernel(f, grid, 1, 1, 256, 1, 1, 0, stream, args.as_mut_ptr(), std::ptr::null_mut())
        };
        anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "xq_hc_w4 launch ({r:?})");
        Ok(())
    }
}

/// TP-I3 (b): the probe-only timestamp twin xq_hc_w4_prof (same body; `prof` = [grid][16] u64:
/// entry, after the PDL wait, after phase A, after the gates, after the grid barrier, after the phase-B
/// up GEMV, end, %smid, after the silu staging, clock64 at entry / end), with the same max carveout pin
/// as w4hc_fn (grid-barrier co-residency). Plain launch (no PDL).
#[allow(clippy::too_many_arguments)]
pub(crate) fn w4hc_prof_launch(stream: cudarc::driver::sys::CUstream, grid: u32, ptrs: [u64; 11], prof: u64,
                               hcn: f32, lr: usize, m: usize, h: usize, hc: usize, flags: i32) -> Result<()> {
    use cudarc::driver::sys;
    #[derive(Clone, Copy)]
    struct FnH(sys::CUfunction);
    unsafe impl Send for FnH {}
    unsafe impl Sync for FnH {}
    static F: std::sync::OnceLock<Option<FnH>> = std::sync::OnceLock::new();
    let f = F.get_or_init(|| {
        let f = xq_raw_fn("xq_hc_w4_prof")?;
        let r = unsafe { sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100) };
        (r == sys::CUresult::CUDA_SUCCESS).then_some(FnH(f))
    }).map(|h| h.0).ok_or_else(|| anyhow::anyhow!("xq_hc_w4_prof unavailable"))?;
    let [mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8, mut a9, mut a10] = ptrs;
    let mut a_hcn = hcn;
    let mut a_lrm = (lr as i32) | ((m as i32) << 16);
    let mut a_hhc = (h as i32) | ((hc as i32) << 16);
    let mut a_fl = flags;
    let mut a_pr = prof;
    let mut args: [*mut std::ffi::c_void; 16] = [
        &mut a0 as *mut u64 as *mut _, &mut a1 as *mut u64 as *mut _, &mut a2 as *mut u64 as *mut _,
        &mut a3 as *mut u64 as *mut _, &mut a4 as *mut u64 as *mut _, &mut a5 as *mut u64 as *mut _,
        &mut a6 as *mut u64 as *mut _, &mut a7 as *mut u64 as *mut _, &mut a8 as *mut u64 as *mut _,
        &mut a9 as *mut u64 as *mut _, &mut a10 as *mut u64 as *mut _,
        &mut a_hcn as *mut f32 as *mut _, &mut a_lrm as *mut i32 as *mut _, &mut a_hhc as *mut i32 as *mut _,
        &mut a_fl as *mut i32 as *mut _, &mut a_pr as *mut u64 as *mut _,
    ];
    let r = unsafe { sys::cuLaunchKernel(f, grid, 1, 1, 256, 1, 1, 0, stream, args.as_mut_ptr(), std::ptr::null_mut()) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "xq_hc_w4_prof launch ({r:?})");
    Ok(())
}
