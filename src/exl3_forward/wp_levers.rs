//! Lever code for the GDN ring / commit modes (WP12), the grouped expert entries and WP20 selection / launch,
//! the MoE A1B3 / coop / chain-pair tunables and the WP18 scorer plan with `Wp18Scratch`.
//! See `mod.rs` for the file map; `moe_wp20.rs` holds the FwdModel methods that consume WP20.

use super::*;

/// S-A3-o G1: GDN commit mode — 1 = rank-1 ring commit (default), 0 = the replay
/// (--exl3-gdn-ring=0, A/B), 2 = replay + ring-on-a-copy bitwise diff every layer
/// (--exl3-gdn-ring-check=1; eager only — forces the verify graph off).
pub(super) fn gdn_ring_mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        if crate::opts::var(crate::opt!("exl3-gdn-ring-check")).is_ok() { 2 }
        else if crate::opts::var(crate::opt!("exl3-gdn-ring")).as_deref() == Ok("0") { 0 }
        else { 1 }
    })
}

/// WP12: the commit mode verify_commit ACTUALLY runs (the ring commit is specialised to
/// kd = 128 and C <= GDN_RING_CMAX = 8 since WP23). ONE predicate for both sides: verify_kernels
/// skips the replay-only saves (a_save/b_save/qkv_post) exactly when this says the replay never runs.
/// WP23: a depth ceiling past the ring's extent no longer drops to the replay SILENTLY (the last
/// depth-7 test lost the ring commit that way): it says so, once.
pub(super) fn gdn_commit_mode(kd: usize, vd: usize) -> u8 {
    if kd == 128 && vd % 32 == 0 && MTP_MAX_K + 1 <= GDN_RING_CMAX { return gdn_ring_mode(); }
    if kd == 128 && vd % 32 == 0 {
        static WARN: std::sync::Once = std::sync::Once::new();
        WARN.call_once(|| eprintln!("[exl3] WARNING: MTP_MAX_K+1 = {} > {GDN_RING_CMAX} (xq_gdn_commit_ring extent) — \
                                     GDN commit falls back to the REPLAY (slower rounds)", MTP_MAX_K + 1));
    }
    0
}

/// WP12 parts (all bitwise, all default ON). --wp12-off=1|all restores every old path;
/// or a comma list of parts: step (register-resident GDN verify chunk + decode step ->
/// the old xq_gdn_step_chunk / xq_gdn_step), diet (the dead replay saves when the ring
/// commit runs), conv (nostore verify conv on the live ring -> the shadow copy + chunk),
/// ab (one a|b GEMV launch -> two). Diagnostic A/B + bisect only (AGENTS §1b).
pub(super) const WP12_STEP: u8 = 1;
pub(super) const WP12_DIET: u8 = 2;
pub(super) const WP12_CONV: u8 = 4;
pub(super) const WP12_AB: u8 = 8;
pub(super) fn wp12_on(part: u8) -> bool {
    let t = match part {
        WP12_STEP => &T_WP12_STEP,
        WP12_DIET => &T_WP12_DIET,
        WP12_CONV => &T_WP12_CONV,
        WP12_AB => &T_WP12_AB,
        _ => return false,
    };
    tune::get_cur(t) != 0
}
/// The --wp12-off part mask (None = unset), parsed and reported once — the legacy alias of the
/// four wp12.* registry entries (same parse, same WP12 lines as before).
fn wp12_env_mask() -> Option<u8> {
    static M: std::sync::OnceLock<Option<u8>> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let v = crate::opts::var(crate::opt!("wp12-off")).ok()?;
        let m = wp12_mask_of(&v);
        crate::dprintln!("WP12: parts on = {m:#x} (step 1, diet 2, conv 4, ab 8; --wp12-off={v})");
        Some(m)
    })
}
/// The --wp12-off value -> parts-on mask (the pre-registry parse, verbatim).
pub(crate) fn wp12_mask_of(v: &str) -> u8 {
    let mut m = WP12_STEP | WP12_DIET | WP12_CONV | WP12_AB;
    for p in v.split(',') {
        match p.trim() {
            "1" | "all" => m = 0,
            "step" => m &= !WP12_STEP,
            "diet" => m &= !WP12_DIET,
            "conv" => m &= !WP12_CONV,
            "ab" => m &= !WP12_AB,
            "" | "0" => {}
            other => crate::dprintln!("WP12: unknown --wp12-off part '{other}' ignored"),
        }
    }
    m
}
pub(crate) const T_WP12_STEP: tune::TunableDef = tune::TunableDef {
    slot: 20, id: "wp12.step", class: tune::Class::S, scope: tune::Scope::Width,
    fams: &[tune::Fam::Step, tune::Fam::Verify],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--wp12-xcheck (gdn step/chunk)",
    env: "--wp12-off", env_parse: || wp12_env_mask().map(|m| (m & WP12_STEP != 0) as i32), rev: 1, wp: "WP12",
};
pub(crate) const T_WP12_DIET: tune::TunableDef = tune::TunableDef {
    slot: 21, id: "wp12.diet", class: tune::Class::S, scope: tune::Scope::Width, fams: &[tune::Fam::Verify],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--wp12-xcheck",
    env: "--wp12-off", env_parse: || wp12_env_mask().map(|m| (m & WP12_DIET != 0) as i32), rev: 1, wp: "WP12",
};
pub(crate) const T_WP12_CONV: tune::TunableDef = tune::TunableDef {
    slot: 22, id: "wp12.conv", class: tune::Class::S, scope: tune::Scope::Width, fams: &[tune::Fam::Verify],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--wp12-xcheck (verify conv)",
    env: "--wp12-off", env_parse: || wp12_env_mask().map(|m| (m & WP12_CONV != 0) as i32), rev: 1, wp: "WP12",
};
pub(crate) const T_WP12_AB: tune::TunableDef = tune::TunableDef {
    slot: 23, id: "wp12.ab", class: tune::Class::S, scope: tune::Scope::Width,
    fams: &[tune::Fam::Step, tune::Fam::Verify],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--wp12-xcheck (a|b)",
    env: "--wp12-off", env_parse: || wp12_env_mask().map(|m| (m & WP12_AB != 0) as i32), rev: 1, wp: "WP12",
};

/// WP12 XCHECK (--wp12-xcheck=1): every WP12 site also runs the OLD path on twin
/// buffers from identical inputs and bit-diffs the outputs (eager only — forces the step
/// and verify graphs off; skipped while a stream capture is live).
pub(super) fn wp12_xcheck() -> bool {
    static X: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *X.get_or_init(|| crate::opts::var(crate::opt!("wp12-xcheck")).map_or(false, |v| !v.is_empty() && v != "0"))
}

/// WP12 XCHECK report: bitwise compare `new` vs `old` (f32 bit patterns, or f16 widened),
/// print mismatch count, first mismatch and max |diff| on every mismatch and every 360th
/// check (one counter across all sites).
pub(super) fn wp12_report(site: &str, new: &[f32], old: &[f32]) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static BAD: AtomicUsize = AtomicUsize::new(0);
    let n = new.len().min(old.len());
    let mut mism = 0usize;
    let mut first: Option<usize> = None;
    let mut maxd = 0f32;
    for i in 0..n {
        if new[i].to_bits() != old[i].to_bits() {
            mism += 1;
            if first.is_none() { first = Some(i); }
            let d = (new[i] - old[i]).abs();
            if d.is_nan() || d > maxd { maxd = if d.is_nan() { f32::INFINITY } else { d }; }
        }
    }
    let k = N.fetch_add(1, Ordering::SeqCst) + 1;
    if mism > 0 { BAD.fetch_add(1, Ordering::SeqCst); }
    if mism > 0 || k % 360 == 0 {
        let fi = first.map_or(String::from("-"), |i| format!("{i} (new {:e} old {:e})", new[i], old[i]));
        crate::dprintln!("WP12_XCHECK {site}: checks {k} bad {} | this: n {n} mism {mism} max|d| {maxd:e} first {fi}",
                 BAD.load(Ordering::SeqCst));
    }
}
pub(super) fn wp12_report_f16(site: &str, new: &[u16], old: &[u16]) {
    let w = |v: &[u16]| v.iter().map(|&b| f16::from_bits(b).to_f32()).collect::<Vec<f32>>();
    // f16 -> f32 widening is exact and injective except NaN payloads; compare raw bits too
    let raw_bad = new.iter().zip(old.iter()).filter(|(a, b)| a != b).count();
    if raw_bad > 0 { crate::dprintln!("WP12_XCHECK {site}: raw f16 mismatches {raw_bad}"); }
    wp12_report(site, &w(new), &w(old));
}

// TUNE T0: the MoE expert-path switches (grouped_fn / moe()) and the paired-chain switch.
pub(crate) const T_MOE_A1B3: tune::TunableDef = tune::TunableDef {
    slot: 2, id: "moe.a1b3", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-binv (grouped a1b3 vs xh)",
    env: "--exl3-a1b3", env_parse: || tune::env_on_unless_0(crate::opt!("exl3-a1b3")), rev: 1, wp: "S-A3-o",
};
pub(crate) const T_MOE_COOP: tune::TunableDef = tune::TunableDef {
    slot: 4, id: "moe.coop", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "--probe-exl3-binv (coop vs grouped)",
    // `--moe-coop` set and != "0" = on (the old per-call map(v != "0").unwrap_or(false))
    env: "--moe-coop", env_parse: || tune::env_on_unless_0(crate::opt!("moe-coop")), rev: 1, wp: "S-A3-h",
};
pub(crate) const T_CHAIN_PAIR: tune::TunableDef = tune::TunableDef {
    slot: 1, id: "chain.pair", class: tune::Class::S, scope: tune::Scope::Shape,
    fams: &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass, tune::Fam::DraftChain,
            tune::Fam::Reprime, tune::Fam::Util],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--exl3-chain-pair-check",
    env: "--exl3-chain-pair", env_parse: || tune::env_on_unless_0(crate::opt!("exl3-chain-pair")), rev: 1, wp: "S-A3-o",
};

/// CF-P1e (16-row class): a 10..16-row gate/up (K = hidden) whose A-once smem exceeds the 48 KiB static cap is
/// staged in K-chunks by the `xq_moe_gu_epi_c<N>` entries (bit-identical to `xq_moe_gu_epi`: same mainloop order).
/// Returns (entry, dynamic smem bytes) for the epilogue path, or None (rows <= 9 keep the existing entry; `--exl3-gu-kch 0`
/// or an unsupported shape keeps the legacy barrier body at those widths, as before).
pub(crate) fn gu_chunk(m: usize, k: usize, diet: bool, bits: usize) -> Option<(&'static str, u32)> {
    // P7 B3: K = 5 chunks are SHFL (diet-independent); b3/b4 keep requiring the diet twin.
    if (!matches!(bits, 2 | 5 | 6 | 7) && !diet) || m < 10 || m > 16 || a1_smem(m, k) > 0 || tune::get_cur(&T_MOE_A1B3) == 0 { return None; }
    let kch = crate::opts::var(crate::opt!("exl3-gu-kch")).ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(2);
    let name = match (bits, kch) {
        (3, 2) => "xq_moe_gu_epi_c2", (3, 4) => "xq_moe_gu_epi_c4",
        (4, 2) => "xq_moe_gu_epi_b4_c2", (4, 4) => "xq_moe_gu_epi_b4_c4",
        (2, 2) => "xq_moe_gu_epi_b2_c2", (2, 4) => "xq_moe_gu_epi_b2_c4",
        (5, 2) => "xq_moe_gu_epi_b5_c2", (5, 4) => "xq_moe_gu_epi_b5_c4",
        (6, 2) => "xq_moe_gu_epi_b6_c2", (6, 4) => "xq_moe_gu_epi_b6_c4",
        (7, 2) => "xq_moe_gu_epi_b7_c2", (7, 4) => "xq_moe_gu_epi_b7_c4",
        _ => return None,
    };
    if k % (kch * 16) != 0 { return None; }
    let smem = m * (k / kch + 8) * 2;
    if smem > 48 * 1024 { return None; }
    Some((name, smem as u32))
}

/// S-A3-o O3: the grouped expert entry. The A-once 3-bit body runs in its own kernel
/// (`xq_gemm_grouped_a1b3`, 3 CTAs/SM) — bitwise identical to xq_gemm_grouped_xh, which
/// keeps every other (m, K, bits). --exl3-a1b3=0 = A/B diagnostic (old entry).
pub(crate) fn grouped_fn(m: usize, k: usize, bits: usize) -> &'static str {
    // K8EXP: K = 8 experts have exactly one entry (A-once and barrier bodies inside; xq_gemm_grouped_xh has no
    // case 8 and is kept byte-identical), at every width and under every A/B flag.
    if bits == 8 { return "xq_gemm_grouped_b8"; }
    let off = tune::get_cur(&T_MOE_A1B3) == 0;
    if off || a1_smem(m, k) == 0 { return "xq_gemm_grouped_xh"; }
    match bits {
        2 => "xq_gemm_grouped_a1b2", // P7 B3 leg 2
        3 => "xq_gemm_grouped_a1b3",
        4 => "xq_gemm_grouped_a1b4",
        5 => "xq_gemm_grouped_a1b5", // P7 B3: SHFL a1 body, bitwise == xq_gemm_grouped_xh
        6 => "xq_gemm_grouped_a1b6", // P7 B3 leg 2
        7 => "xq_gemm_grouped_a1b7", // P7 B3 leg 2
        _ => "xq_gemm_grouped_xh",
    }
}

/// A5 WP20 (PLAN/SURPASS_PLAN_2026-09-26.md): 0 = off (--wp20-off=1 — the pre-WP20 expert
/// path, bit for bit, kernels untouched), 1 = rung 1 only (--wp20-epi=0: the word-diet body
/// `xq_gemm_grouped_a1b3_wd` in the old launch sequence), 2 = rungs 1 + 2 (default: the
/// epilogue kernels xq_moe_gu_epi / xq_moe_dn_epi_k<topk>). A mode-2 call that rung 2 cannot
/// take (logged, see FwdModel::wp20_select) still gets rung 1 wherever the a1b3 entry runs.
/// Diagnostics only — every mode is bitwise.
pub(super) fn wp20_mode() -> u8 {
    tune::get_cur(&T_WP20_MODE).clamp(0, 2) as u8
}
// TUNE T0 (p3c): WP20's rungs (every mode bitwise: --probe-exl3-binv EXL3-MOE-EPI + MOE_EPI_CHECK).
pub(crate) const T_WP20_MODE: tune::TunableDef = tune::TunableDef {
    slot: 38, id: "wp20.mode", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1, 2], default: 2, valid: tune::valid_any,
    xcheck: "--probe-exl3-binv EXL3-MOE-EPI + --exl3-moe-epi-check",
    env: "--wp20-off|--wp20-epi",
    // the old precedence exactly: OFF=1 -> 0, else EPI=0 -> 1, else 2 (None when neither is set)
    env_parse: || {
        let (off, epi) = (tune::env_str(crate::opt!("wp20-off")), tune::env_str(crate::opt!("wp20-epi")));
        if off.is_none() && epi.is_none() { return None; }
        Some(if off.as_deref() == Some("1") { 0 } else if epi.as_deref() == Some("0") { 1 } else { 2 })
    },
    rev: 1, wp: "WP20",
};
pub(crate) const T_WP20_DIET: tune::TunableDef = tune::TunableDef {
    slot: 39, id: "wp20.diet", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any,
    xcheck: "--probe-exl3-binv EXL3-MOE-EPI (word diet vs SHFL)",
    env: "--wp20-diet", env_parse: || tune::env_on_unless_0(crate::opt!("wp20-diet")), rev: 1, wp: "WP20",
};

/// --wp20-diet=0: rung 2's epilogue kernels on the old SHFL decode (isolates rung 2's A/B);
/// also keeps rung 1 off (no diet anywhere).
pub(super) fn wp20_diet_on() -> bool {
    tune::get_cur(&T_WP20_DIET) != 0
}

/// --exl3-moe-epi-check=1: the eager bitwise old-vs-WP20 diff (see moe_wp20_xcheck).
pub(super) fn wp20_check_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-moe-epi-check")).is_ok())
}

/// WP20-v1b: the top-k values the down epilogue is compiled for (kernels/exl3_bench.cu
/// WP20_DN_ENTRY: the combine's j loops are a compile-time TOPK template — AGENTS §4). Any
/// other top-k keeps the pre-WP20 launch sequence, with a loud WP20 FALLBACK line.
pub(super) const WP20_TOPKS: [usize; 2] = [8, 10];

/// The down-epilogue entry for (top-k, decode body), None = no compiled instance.
pub(crate) fn wp20_dn_fn(topk: usize, diet: bool) -> Option<&'static str> {
    match (topk, diet) {
        (8, true) => Some("xq_moe_dn_epi_k8"),
        (8, false) => Some("xq_moe_dn_epi_sh_k8"),
        (10, true) => Some("xq_moe_dn_epi_k10"),
        (10, false) => Some("xq_moe_dn_epi_sh_k10"),
        _ => None,
    }
}

/// The gate/up epilogue entry (top-k independent) for the decode body.
pub(crate) fn wp20_gu_fn(diet: bool) -> &'static str {
    if diet { "xq_moe_gu_epi" } else { "xq_moe_gu_epi_sh" }
}

/// WP20 path decisions are logged ONCE per distinct key (class, m, detail): the MoE block runs
/// per layer per step (and every draft pass), so every active path and every fallback reason
/// shows up exactly once per width — never silently, never per call. Classes: 0 mode line,
/// 1 rung 2 active, 2 rung 2 fallback (detail = reason code), 3 rung 1 active (detail = K),
/// 4 rung 1 fallback (detail = K), 5 rung 1 pre-empted (coop / fh), 6 xcheck.
pub(super) fn wp20_log_once(key: (u8, usize, usize), msg: impl FnOnce() -> String) {
    static SEEN: std::sync::Mutex<Vec<(u8, usize, usize)>> = std::sync::Mutex::new(Vec::new());
    let first = match SEEN.lock() {
        Ok(mut v) => if v.contains(&key) { false } else { v.push(key); true },
        Err(_) => true,
    };
    if first { crate::dprintln!("{}", msg()); }
}

/// Rung 1 in the old launch sequence: the diet twin behind the a1b3 entry's exact contract
/// when `rung1` (mode 1, or a mode-2 call rung 2 did not take; never the xcheck reference).
/// Logs which body each (m, K) actually runs, and why not when the twin cannot apply.
pub(super) fn wp20_grouped(name: &'static str, rung1: bool, m: usize, k: usize) -> &'static str {
    if !rung1 { return name; }
    if name == "xq_gemm_grouped_a1b4" {
        wp20_log_once((3, m, k), || format!(
            "WP20 rung 1 ACTIVE m={m} K={k}: xq_gemm_grouped_a1b4_wd (4-bit word diet) in the pre-WP20 launch sequence"));
        "xq_gemm_grouped_a1b4_wd"
    } else if name == "xq_gemm_grouped_a1b3" {
        wp20_log_once((3, m, k), || format!(
            "WP20 rung 1 ACTIVE m={m} K={k}: xq_gemm_grouped_a1b3_wd (word diet: 2 ring words/lane/k16 step, no SHFL) \
             in the pre-WP20 launch sequence"));
        "xq_gemm_grouped_a1b3_wd"
    } else {
        wp20_log_once((4, m, k), || format!(
            "WP20 FALLBACK rung 1 m={m} K={k}: {name} runs (the diet twin needs the a1b3 entry: 3-bit, \
             m*(K+8)*2 = {} B <= 48 KiB, --exl3-a1b3 != 0) — old SHFL decode for this GEMM", m * (k + 8) * 2));
        name
    }
}

/// The process-wide suh verification (see crate::exl3_bench::wp20_calibrate): the epilogue's
/// pinned suh sequence (WP20-v1c) checked bitwise against the JIT'd xq_had_suh_multi ONCE,
/// eagerly at FwdModel::load (mode 2, MoE model) or at the first non-capturing eligible call;
/// Some(seq id) = verified, None = mismatch / launch error (rung 2 then stays off, loudly).
pub(super) static WP20_CONV: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();

pub(super) fn wp20_calibrated(dev: &Arc<CudaDevice>) -> Option<u32> {
    *WP20_CONV.get_or_init(|| match crate::exl3_bench::wp20_calibrate(dev) {
        Ok(Some(c)) => {
            crate::dprintln!("WP20: suh sequence 0x{c:02x} VERIFIED (the epilogue's pinned FMUL/FFMA suh stage == the JIT'd \
                      xq_had_suh_multi, bitwise on realistic + f16 edge-case rows)");
            Some(c)
        }
        Ok(None) => {
            println!("WP20 FALLBACK: suh verification FAILED (the pinned sequence != the JIT'd xq_had_suh_multi) — \
                      MoE epilogues (rung 2) OFF for this process");
            None
        }
        Err(e) => {
            println!("WP20 FALLBACK: suh verification failed ({e:#}) — MoE epilogues (rung 2) OFF for this process");
            None
        }
    })
}

/// FwdModel::load: announce the WP20 plan for this model (mode, top-k instance, decode body) and
/// verify the pinned suh sequence eagerly, so the first graph capture can never be the first
/// eligible call.
pub(super) fn wp20_announce(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config) {
    if cfg.num_experts == 0 { return; }
    let topk = cfg.num_experts_per_tok;
    let diet = wp20_diet_on();
    match wp20_mode() {
        0 => crate::dprintln!("WP20: OFF (--wp20-off=1) — pre-WP20 expert path (SHFL decode, 5 glue launches per MoE layer)"),
        1 => crate::dprintln!("WP20: mode 1 (--wp20-epi=0) — rung 1 only (word-diet twin {}), no MoE epilogues",
                      if diet { "on" } else { "OFF: --wp20-diet=0" }),
        _ => {
            match wp20_dn_fn(topk, diet) {
                Some(dn) => {
                    let (gu, have_gu, have_dn) = (wp20_gu_fn(diet), xq_raw_fn(wp20_gu_fn(diet)).is_some(), xq_raw_fn(dn).is_some());
                    crate::dprintln!("WP20: mode 2 — rungs 1+2 planned for top-k {topk}: {gu} + {dn} (decode body {}){}",
                             if diet { "diet" } else { "SHFL: --wp20-diet=0" },
                             if have_gu && have_dn { String::new() }
                             else { format!(" — KERNEL MISSING (gu {have_gu}, dn {have_dn}; stale PTX?) -> rung 2 will fall back") });
                    let _ = wp20_calibrated(dev);
                }
                None => crate::dprintln!("WP20 FALLBACK: mode 2 requested but top-k {topk} has no down-epilogue instance \
                                  (compiled: {WP20_TOPKS:?}) — rung 2 OFF for this model; rung 1 still applies"),
            }
        }
    }
}

// (WP20's free `stream_capturing(CUstream)` duplicated WP11's `stream_capturing(&Launcher)`:
// p2 keeps WP11's; the WP20 call sites pass the launcher.)

/// A5 WP20: raw launch of a 256-thread MoE epilogue kernel (> 12 params: past the typed
/// launcher's cap), with the PDL twin under --exl3-pdl=1 exactly as xqlaunch! does.
pub(super) unsafe fn wp20_launch(stream: cudarc::driver::sys::CUstream, name: &str, grid: u32, smem: u32,
                      params: &mut [*mut std::ffi::c_void]) -> Result<()> {
    use cudarc::driver::sys;
    if pdl_on() {
        if let Some(f) = xq_raw_fn_smem(name, smem) {
            return launch_pss_raw(stream, f, (grid, 1, 1), (256, 1, 1), smem, params)
                .with_context(|| format!("launch {name} (pdl)"));
        }
    }
    let f = xq_raw_fn(name).ok_or_else(|| anyhow::anyhow!("{name} raw fn unavailable (stale PTX?)"))?;
    let r = sys::cuLaunchKernel(f, grid, 1, 1, 256, 1, 1, smem, stream, params.as_mut_ptr(), std::ptr::null_mut());
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch {name} ({r:?})");
    Ok(())
}


// ===========================================================================
// S-A3-f-b: CHUNKED PREFILL. Process C consecutive tokens of ONE slot in one
// sweep: weight ops at M=C rows (wide-M GEMM), recurrences looped inside the
// chunk kernels. Prefill sits OUTSIDE the batch-invariance contract (same rule
// as the NVFP4 line's prefill); the ladder gate measures drift vs the
// sequential decode-step path on real chunks.
// ===========================================================================

// ---------------------------------------------------------------------------
// WP18 rung A (PLAN/SURPASS_PLAN_2026-09-26.md): bounded prefill QSA selection, bitwise.
// The old path scored every (sparse row, visible block) of a chunk into a [c][max_pos/ratio] f32
// matrix (qsa_score_prefill_b; 0.54 GB at C = 2048 on a 262K window) and ran xq_qsa_topk_asc per
// row. WP18 walks each group of `rows` sparse rows over tiles of WP18_T blocks: xq_wp18_score
// fills a [rows][WP18_T] slab with the SAME score bits, xq_wp18_merge folds the tile into each
// row's carried top-K list under the same total order (key desc, block asc) and emits the
// ascending selection list on the row's last tile (kernel notes: kernels/exl3_bench.cu, 3c).
// --wp18-off=1 = the old path; --wp18-xcheck=1 = run both, diff lists + score bits
// (eager); --wp18-rows = slab rows per group (default 256, multiple of 32, 32..2048).
// ---------------------------------------------------------------------------
pub(crate) const WP18_T: usize = 8192;     // kernels/exl3_bench.cu WP18_T (slab row pitch)
pub(crate) const WP18_TM: usize = 32;      // WP18_TM (rows per scorer CTA)
pub(crate) const WP18_TN: usize = 64;      // WP18_TN (blocks per scorer CTA)
pub(crate) const WP18_MT: usize = 512;     // WP18_MT (merge threads)
pub(crate) const WP18_KMAX: usize = 512;   // WP18_KMAX (carried-list capacity)

fn wp18_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| wp14_xcheck(crate::opt!("wp18-off")))
}
pub(super) fn wp18_xcheck() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| wp14_xcheck(crate::opt!("wp18-xcheck")))
}
pub(crate) fn wp18_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let r = crate::opts::var(crate::opt!("wp18-rows")).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256);
        (r.clamp(WP18_TM, 2048) / WP18_TM) * WP18_TM
    })
}

/// One launch pair of the WP18 walk: row group [g0, g0+gn) of the sparse rows (first position
/// pos_first), tile kt. score_grid.0 == 0: no block of the tile is visible (only when the group's
/// widest row has nb == 0 — the merge still runs to emit tail-only lists).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Wp18Step {
    pub g0: usize,
    pub gn: usize,
    pub pos_first: usize,
    pub kt: usize,
    pub score_grid: (u32, u32),
}

/// The WP18 walk for `rows` consecutive sparse rows starting at position `pos_first0`: row groups
/// of `rg`, each over tiles 0..ceil(nb_max / WP18_T) in ascending order (the merge's contract).
pub(crate) fn wp18_plan(pos_first0: usize, rows: usize, ratio: usize, rg: usize) -> Vec<Wp18Step> {
    let mut v = Vec::new();
    let mut g0 = 0;
    while g0 < rows {
        let gn = rg.min(rows - g0);
        let pos_first = pos_first0 + g0;
        let nb_max = (pos_first + gn) / ratio;             // the group's last (widest) row
        let nt = nb_max.div_ceil(WP18_T).max(1);
        for kt in 0..nt {
            let tb = nb_max.saturating_sub(kt * WP18_T).min(WP18_T);
            let gx = tb.div_ceil(WP18_TN) as u32;
            v.push(Wp18Step { g0, gn, pos_first, kt, score_grid: (gx, if gx > 0 { gn.div_ceil(WP18_TM) as u32 } else { 0 }) });
        }
        g0 += gn;
    }
    v
}


impl Wp18Scratch {
    pub(super) fn new(m: &FwdModel, c: usize) -> Result<Option<Self>> {
        let cfg = &m.cfg;
        let ratio = cfg.indexer_compress_ratio;
        let topk = if ratio > 0 { cfg.indexer_budget / ratio } else { 0 };
        // shape contract of the kernels + the diagnostics that read the full matrix
        let why_not = if wp18_off() { Some("--wp18-off") }
            else if !cfg.has_indexer() { Some("no indexer") }
            else if cfg.indexer_head_dim != 128 || cfg.indexer_n_heads != 4 { Some("indexer shape != hd 128 x 4 heads") }
            else if ratio == 0 || topk == 0 || topk > WP18_KMAX { Some("topk outside 1..=512") }
            else if m.qsa_dbg.is_some() { Some("--qsa-dump-layer reads the full matrix") }
            else if crate::opts::var(crate::opt!("prefillx-qsa-keys")).is_ok() { Some("--prefillx-qsa-keys reads the full matrix") }
            else { None };
        let nblk_stride = (m.max_pos / ratio.max(1)).max(1);
        static ONCE: std::sync::Once = std::sync::Once::new();
        if let Some(w) = why_not {
            ONCE.call_once(|| crate::dprintln!("WP18: bounded prefill selection OFF ({w}) — [c][max_pos/ratio] score matrix"));
            return Ok(None);
        }
        let rows = wp18_rows().min(c.max(1));
        let dev = &m.dev;
        let x = if wp18_xcheck() {
            Some((dev.alloc_zeros::<i32>(c * (cfg.indexer_budget + ratio))?, dev.alloc_zeros::<i32>(c)?,
                  dev.alloc_zeros::<i32>(4)?))
        } else { None };
        ONCE.call_once(|| crate::dprintln!("WP18: bounded prefill selection ON — slab {rows} x {WP18_T} f32 ({:.1} MB) + carried \
                                    top-{topk} lists ({:.1} MB) replace the [c][{nblk_stride}] matrix ({:.1} MB at c = {c}){}",
                                   (rows * WP18_T * 4) as f64 / 1e6, (rows * topk * 8) as f64 / 1e6,
                                   (c * nblk_stride * 4) as f64 / 1e6,
                                   if x.is_some() { "; --wp18-xcheck: the old matrix is kept as the reference" } else { "" }));
        Ok(Some(Wp18Scratch {
            rows,
            slab: dev.alloc_zeros::<f32>(rows * WP18_T)?,
            ck: dev.alloc_zeros::<u32>(rows * topk)?,
            ci: dev.alloc_zeros::<i32>(rows * topk)?,
            x,
        }))
    }
}
