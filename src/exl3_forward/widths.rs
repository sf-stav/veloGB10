//! Which EXL3 bit widths each kernel family supports (wide trellis / tiles / MoE prefill / coop), the wide-arm
//! refusal message, and the PFX1 prefill thresholds (moe MT class, router rows8, tile router) with `ReconBufs`.
//! See `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------
// S-A3-f-g — reconstruct→cuBLASLt wide-M dense path (the reference implementation's M>=17 split).
// For wide M the fused trellis-GEMM runs at 120 GB/s (weight-bound, 58 tok/s
// prefill); instead decode the trellis ONCE into an f16 staging matrix (the
// decode front-end is the oracle-validated `exl3_decode_dump`) and run the
// vendor GEMM (cuBLASLt → nvjet SM121) on it. Input/output Hadamard kernels
// (had_suh / had_svh) are UNCHANGED — the staged matrix is the same plain
// W_dec the fused mma consumes, so the substitution differs only in the GEMM
// reduction order (the drift class pre-authorized by the brief §1).
// Gated by --exl3-recon-min. S-A3-f-d step 0b (OWNER APPROVED 2026-09-21):
// the serving default is now 17 — the reference implementation's VLLM_EXL3_RECONSTRUCT_MIN_ROWS —
// so the shipped server takes the fast reconstruct→Lt prefill path by default
// (bit-drift class pre-authorized at f-g; decode m <= 16 NEVER reaches this
// path). 0 restores the wide trellis kernel.
// ---------------------------------------------------------------------------

/// PACKPF: the fewest rows a segment of a packed pass may have. Below it the unpacked chunk takes
/// row-count-dependent arms (the m <= 16 decode-contract chains, the per-row kernels under
/// --exl3-recon-min) that the pass, at its own total row count, would not take.
pub fn pf_min_seg_rows() -> usize { recon_min_rows().max(17) }

pub(super) fn recon_min_rows() -> usize {
    crate::opts::var(crate::opt!("exl3-recon-min"))
        .ok().and_then(|v| v.parse().ok()).unwrap_or(17)
}

/// S-B9-REL-INTEG D1: the bit widths the WIDE trellis GEMM family decodes. Its kernel switches
/// (`exl3_hmma_gemm_wide`, `_wide_grouped`, `_wide_tiles`) carry cases 3, 4 and 5 and `default: break;` in BOTH
/// module (P7 B5 widened this family to 3..=7 — cases 6 and 7 joined; K=2 stays on the pf/B4 path),
/// so a 2- or 8-bit module launched there writes NOTHING and the request is answered 200 with garbage.
/// The reconstruct + cuBLASLt arm is bits-generic, which is why the default `--exl3-recon-min 17` stays
/// safe for any width.
pub(crate) fn wide_trellis_supports_bits(bits: usize) -> bool { (3..=8).contains(&bits) } // P7 B5: 6/7 join; PR#27-D2: 8 joins (K=2 stays pf/B4)

/// K35 (2026-10-31): the kernel-side WIDTH CODE. The expert / wide-tile kernels switch on `bits`: an
/// integer K arrives as K (2..=8); a fractional K = KA + 0.5 (k2 odd) arrives as EXL3_FRAC_BITS_BASE + KA
/// (17 / 18 / 19) so it can never alias an integer width — k2 = 7 passed as `k2 / 2` = 3 would take the
/// 3-bit bodies on a 56-word ring (the S-B9-REL-INTEG D1 silent-garbage class with a plausible-looking
/// number). Twin of `#define EXL3_FRAC_BITS(KA) (16 + (KA))` in kernels/exl3_bench.cu; `MoeLayer::bits`
/// carries this code, `MoeLayer::k2` the raw half-bit units. Every width predicate below takes the code.
pub(crate) const EXL3_FRAC_BITS_BASE: usize = 16;
pub(crate) fn exl3_kernel_bits(k2: usize) -> usize { if k2 % 2 == 1 { EXL3_FRAC_BITS_BASE + k2 / 2 } else { k2 / 2 } }
pub(crate) fn exl3_is_frac_bits(bits: usize) -> bool { (EXL3_FRAC_BITS_BASE + 1..=EXL3_FRAC_BITS_BASE + 3).contains(&bits) }
/// The width a human reads for a kernel bits code: "3.5" for 19, "4" for 4.
pub(crate) fn exl3_width_name(bits: usize) -> String {
    if exl3_is_frac_bits(bits) { format!("{}.5", bits - EXL3_FRAC_BITS_BASE) } else { format!("{bits}") }
}
/// The device-routed prefill MoE arm's tile kernel (`exl3_hmma_gemm_wide_tiles`) decodes the integer
/// wide set AND the three fractional widths (K35: `EXL3_WT_CASE_H(1..3)` = exl3_gemm_body_wide<KA, MT, true>).
/// The per-expert / grouped wide kernels of the host-routed arm carry the integer cases only.
pub(crate) fn wide_tiles_supports_bits(bits: usize) -> bool { wide_trellis_supports_bits(bits) || exl3_is_frac_bits(bits) }

/// LEG-B: rows per block in the fractional-k2 prefill bridge (the decode moe() width). 8 = the
/// widest width the 2026-10-30 1.66 probe-state rung verified EXACT (N = 1..=8, all lanes).
pub(crate) const FRAC_PF_BLK: usize = 8;

/// P7-1P5 W2 loader capability audit: widths with real decode bodies in this build.
/// K = 3..=6, 8 (k2 6, 8, 10, 12, 16). Fractional K (odd k2) lands with this branch's W3/W4
/// kernels; K=8 (k2 16) serves since PR#27-D2; K=7 (k2 14) stays probe-reference-only and refuses here —
/// a width without a body must never reach a `default: break` switch (S-B9-REL-INTEG D1).
pub(crate) fn quad_k2_supported(k2: usize) -> bool {
    matches!(k2, 3 | 4 | 6 | 8 | 10 | 12 | 14 | 16) // PR#27-D2: k2=16 (K=8); PCKG-ALL: k2=14 (K=7) — hmma/ks/ks_x2/fx all carry case 14
}

/// The pipelined prefill MoE expert kernels (`xq_moe_pf*`, `--pf-moe-kernel`) decode 2..7-bit trellises
/// (B4: K=5; B4 leg 2: K=2/6/7 through the same two-ring-word SHFL select).
/// The pipelined prefill MoE kernels (xq_moe_pf*, V 0): integer K 2..7 (P7 B4 leg 2) and, PFHALF (2026-11-06), the
/// fractional K = 1.5 / 2.5 / 3.5 (codes 17 / 18 / 19) on the HALF body. The pf2 smem-ring twin is integer-only: the
/// caller masks it off for a fractional layer.
pub(crate) fn moe_pf_kernel_supports_bits(bits: usize) -> bool { matches!(bits, 2 | 3 | 4 | 5 | 6 | 7) || exl3_is_frac_bits(bits) }

/// P7-B1: expert-class modules the EXPERT kernel family can decode: K = 1.5, 2, 2.5, 3, 3.5, 4, 5, 6
/// (k2 3, 4, 5, 6, 7, 8, 10, 12). Distinct from quad_k2_supported (the DENSE set): the two families
/// have different kernel bodies, so each width is admitted only where its body exists — a width
/// admitted without a body reaches a `default: break` switch that writes nothing (the S-B9-REL-INTEG
/// D1 garbage class). The loader refuses the rest loudly (exl3.rs catalog, `width_support_text`).
/// K35 (2026-10-31): k2 5 / 7 (K = 2.5 / 3.5) join — the fractional grouped body
/// (exl3_gemm_body<KA, HALF> / exl3_gemm_body_a1<KA, HALF> behind xq_gemm_grouped_xh at the
/// EXL3_FRAC_BITS code) and the fractional wide-tiles prefill body.
/// K8EXP (2026-11-07): k2 14 / 16 (K = 7 / 8) join. K = 7: the integer expert family already carried it
/// (xq_gemm_grouped_a1b7, xq_gemm_grouped_xh case 7, the WP20 b7 epilogues, gu_pk_b7, xq_moe_pf bits 7) and only
/// this gate was missing; K = 8: its own grouped entry xq_gemm_grouped_b8 (exl3_gemm_body_a1<8> /
/// exl3_gemm_body<8>; grouped_fn) for decode / verify / draft, the wide-tiles kernel (case 8, PR#27-D2) for prefill. The coop kernels (2..=6,
/// opt-in) fall back to the grouped chain for both (moe_coop_supports_bits). Mixed widths INSIDE one layer (the
/// SAGE-4.77 pack) are a separate refusal in the MoE loader. Gate: --probe-exl3-bitwidths section 7.
pub(crate) fn expert_k2_supported(k2: usize) -> bool { matches!(k2, 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 14 | 16) }

/// K8EXP: the widths `xq_moe_coop_a` / `xq_moe_coop_b` decode (`case 2..=6`, `default: break`). A wider (or
/// fractional) expert layer asked for `--moe-coop` is served by the grouped chain instead.
pub(crate) fn moe_coop_supports_bits(bits: usize) -> bool { (2..=6).contains(&bits) }
// PCKG-ALL: k2 4 (K=2 experts, the 1.66/2.05/2.18 class) admitted with the coop case-2
// body + the pre-existing pf bits==2 branch, dn_pk_b2_* pack kernels and gu_epi_b2
// epilogues. Gated on the 1.66 pack serve receipts before any wider claim.

/// The widths each module class serves, as the loader's refusal text names them (from the predicates
/// above, never a hand-typed list — roadmap ISSUES 183 was a stale hand-typed one).
pub(crate) fn width_support_text() -> String {
    let name = |k2: usize| if k2 % 2 == 1 { format!("{}.5", k2 / 2) } else { format!("{}", k2 / 2) };
    let list = |pred: &dyn Fn(usize) -> bool| (2..=16usize).filter(|k| pred(*k)).map(name).collect::<Vec<_>>().join(", ");
    let missing = |pred: &dyn Fn(usize) -> bool| (2..=16usize)
        .filter(|k| crate::exl3::k2_from_words(8 * k).is_ok() && !pred(*k)).map(name).collect::<Vec<_>>().join(", ");
    format!("routed/MTP experts serve K = {} (no expert kernel for K = {}); dense modules serve K = {} (no dense kernel for K = {})",
            list(&expert_k2_supported), missing(&expert_k2_supported), list(&quad_k2_supported), missing(&quad_k2_supported))
}

/// The refusal text of the dense wide arm (the one `--exl3-recon-min` selects).
pub(crate) fn wide_arm_bits_refusal(bits: usize, m: usize, k: i32, n: i32, recon_min: usize, have_recon_buffers: bool) -> String {
    let cause = if !have_recon_buffers {
        "this call site has no reconstruct buffers — they exist only where a reconstruct-capable arm is \
wired; serve these modules from a call site that carries them, or ship the pack in 3- to 7-bit"
    } else {
        "the call site HAS reconstruct buffers but the dispatch did not take them — a routing invariant \
broke; report this"
    };
    format!("EXL3 prefill chain of {m} rows on a {bits}-bit module (K = {k}, N = {n}): the wide trellis GEMM \
(exl3_hmma_gemm_wide) decodes 3-, 4- and 5-bit trellises only and would leave this module's output \
unwritten (HTTP 200 with garbage). Since F3 the dispatch sends every such module through the reconstruct + \
cuBLASLt arm whenever the call site carries reconstruct buffers, whatever --exl3-recon-min ({recon_min}) says — \
the option cannot cause this any more, and neither can an oversized plain step (the serve caps --max-batch at \
16 lanes). What can still reach this refusal: {cause}")
}

// ---------------------------------------------------------------------------
// PFX1 (PLAN/SURPASS_PLAN_2026-09-26.md WP06 items 2-6 + WP00 B3's MT-by-chunk-size rule,
// PLAN/PROBES_14_2026-09-26.md): prefill fixed costs + memory diet. Every part is BITWISE —
// greedy output, prefill state and the draft side (head KV rows actually read) are unchanged.
// Default ON. --pfx1-off=1|all = every part off (the p2 behaviour); or a comma list of parts
// (diagnostics / A-B / bisect only — AGENTS §1b):
//   mt      (a) MoE prefill expert-GEMM m-tiles per CTA by chunk rows      -> MT = 4 always
//   headkv  (b) reset_slot before a head-filled prefill skips the head-KV memset -> memset
//   yraw    (c) PrefillScratch chain_yraw = max(c*12288, vocab)            -> c * vocab
//   rows8   (d) xq_gemm_f16_f32_rows8 prefill router on grids rows32 cannot fill -> rows32
//   ltplans (e) background pre-population of the cuBLASLt prefill plans    -> lazy per new m
//   w16     (f) prefill warm-up width 16                                   -> [C, 512, 160, 32]
// --exl3-wide-mt=1|2|4|8 still overrides (a) (every chunk, every wide launch).
// ---------------------------------------------------------------------------
const PFX1_MT: u8 = 1;
pub(super) const PFX1_HEADKV: u8 = 2;
pub(super) const PFX1_YRAW: u8 = 4;
const PFX1_ROWS8: u8 = 8;
pub(super) const PFX1_LTPLANS: u8 = 16;
pub(super) const PFX1_W16: u8 = 32;
const PFX1_ALL: u8 = PFX1_MT | PFX1_HEADKV | PFX1_YRAW | PFX1_ROWS8 | PFX1_LTPLANS | PFX1_W16;

fn pfx1_mask() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let mut m = PFX1_ALL;
        if let Ok(v) = crate::opts::var(crate::opt!("pfx1-off")) {
            for p in v.split(',') {
                match p.trim() {
                    "1" | "all" => m = 0,
                    "mt" => m &= !PFX1_MT,
                    "headkv" => m &= !PFX1_HEADKV,
                    "yraw" => m &= !PFX1_YRAW,
                    "rows8" => m &= !PFX1_ROWS8,
                    "ltplans" => m &= !PFX1_LTPLANS,
                    "w16" => m &= !PFX1_W16,
                    "" | "0" => {}
                    other => crate::dprintln!("PFX1: unknown --pfx1-off part '{other}' ignored"),
                }
            }
        }
        m
    })
}
pub(super) fn pfx1_on(part: u8) -> bool { pfx1_mask() & part != 0 }

/// --exl3-wide-mt override of the wide-M trellis GEMM's m-tiles per CTA (1|2|4|8; any other
/// value is ignored, loudly — the kernels' MT switch has no default arm that computes).
fn wide_mt_override() -> Option<usize> {
    static O: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *O.get_or_init(|| {
        let v = crate::opts::var(crate::opt!("exl3-wide-mt")).ok()?;
        match v.trim().parse::<usize>() {
            Ok(mt @ (1 | 2 | 4 | 8)) => Some(mt),
            _ => { crate::dprintln!("--exl3-wide-mt='{v}' ignored (must be 1, 2, 4 or 8)"); None }
        }
    })
}

/// PFX1 (a) chunk-row bounds of the MT rule (WP00 B3 on .14, prefill-window deltas vs MT=4:
/// c=26 -15.5 ms and c=72 -22.9 ms at MT=1; c=160 -17.8 ms at MT=2; c=512 flat). The cuts sit
/// at the linear-interpolated crossovers (MT=1 vs 2 ~117 rows; MT=2 vs 4 between 160 and 512).
const PFX1_MT1_MAX: usize = 112;
const PFX1_MT4_MIN: usize = 384;

/// PFX1 (a): m-tiles per CTA (16*MT rows per super-tile) of the MoE prefill expert GEMMs for a
/// chunk of `c` rows. BITWISE across MT (b7c8b3e: each output row's mma sequence over K is
/// MT-independent; --probe-exl3-wide) — only tile counts / CTA shapes change.
pub(crate) fn pfx1_moe_mt(c: usize) -> usize {
    if let Some(mt) = wide_mt_override() { return mt; }
    if !pfx1_on(PFX1_MT) { return 4; }
    if c <= PFX1_MT1_MAX { 1 } else if c < PFX1_MT4_MIN { 2 } else { 4 }
}

/// Device SM count (48 on GB10; queried once, 48 if the query fails).
fn dev_sms() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        use cudarc::driver::sys;
        let mut dev_h: sys::CUdevice = 0;
        let mut sms: std::ffi::c_int = 0;
        let ok = unsafe { sys::cuDeviceGet(&mut dev_h, 0) } == sys::CUresult::CUDA_SUCCESS
            && unsafe { sys::cuDeviceGetAttribute(&mut sms,
                   sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_h) }
               == sys::CUresult::CUDA_SUCCESS;
        if ok && sms > 0 { sms as usize } else { 48 }
    })
}

/// PFX1 (d) row cut: rows8 for chunks of at most this many rows. rows8's cost grows ~linearly
/// with c (every 8-row CTA re-reads its 64-col router slab) while rows32 stays near-flat on its
/// sub-wave grid. GPU receipt (.13, --probe-exl3-binv ROUTER8 timing, ne 512, K 2560, DRAM-cold
/// weights; us/layer rows32 vs rows8): c=26 216.5 vs 34.6 | 72 209.1 vs 49.5 | 160 220.3 vs 86.4
/// | 512 234.4 vs 248.6 | 736 247.3 vs 354.0. Linear fits (rows8 ~23 + 0.45c, rows32 ~212 +
/// 0.04c) cross at c ~ 465-480; the cut sits at 384 (~14% predicted margin, clear of the noise
/// band). The old rule (the rows32 grid under two waves = c <= 736) lost up to 107 us/layer
/// (5.1 ms per 48-layer chunk) at c = 736. --pfx1-rows8-max=<c> overrides (0 = never rows8).
const PFX1_ROWS8_MAX: usize = 384;

/// --pfx1-rows8-max=<c>: Some(c) replaces the whole row rule (the 384 cut AND the two-wave
/// cap) — an A/B / tuning knob; the kernel's shape contract still applies. None = the default rule.
fn pfx1_rows8_max_override() -> Option<usize> {
    static O: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *O.get_or_init(|| {
        let v = crate::opts::var(crate::opt!("pfx1-rows8-max")).ok()?;
        match v.trim().parse::<usize>() {
            Ok(n) => Some(n),
            Err(_) => { crate::dprintln!("--pfx1-rows8-max='{v}' ignored (must be a row count)"); None }
        }
    })
}

/// PFX1 (d): take xq_gemm_f16_f32_rows8 (64 cols x 8 rows per CTA) instead of rows32 for chunks of
/// c <= PFX1_ROWS8_MAX rows whose rows32 grid, ceil(c/32) * (ne/128) CTAs, is under two waves of
/// SMs (<= 736 rows at ne = 512 on 48 SMs — looser than the row cut there). Both kernels are
/// bitwise to the reference router, so the cut moves time only. Shape contract of the kernel:
/// ne % 64 == 0, K % 256 == 0 (else rows32).
pub(crate) fn pfx1_router_rows8(c: usize, ne: usize, k: usize) -> bool {
    pfx1_on(PFX1_ROWS8) && ne % 64 == 0 && k % 256 == 0 && k > 0
        && match pfx1_rows8_max_override() {
            Some(max) => c <= max,
            None => c <= PFX1_ROWS8_MAX && c.div_ceil(32) * (ne / 128) < 2 * dev_sms(),
        }
}

/// TP-4R1: the register-tiled prefill router kernels (`xq_gemm_f16_f32_tile128` = 128x64 outputs per CTA, 8x8 per
/// thread; `xq_gemm_f16_f32_tile64` = 64x64, 8x4) — the same single ascending-k fp32 chain per (row, expert) as
/// rows32/rows8/xq_gemm_f16_f32 (BITWISE equal; gate: --probe-exl3-routergemm), at ~2x the rate of rows32 (rows32 is
/// smem-operand-bound at one smem word per FFMA; these keep an 8x8 / 8x4 output tile in registers). Returns the
/// kernel name and its rows-per-CTA for a launch of `m` rows, or None when the shape contract fails
/// (ne % 64 == 0, K % 16 == 0) and the caller keeps rows32. 64x64 under `ROUTER_TILE128_MIN` rows (more CTAs per
/// wave: measured faster at m = 512..1021), 128x64 from there up.
pub(crate) const ROUTER_TILE128_MIN: usize = 1280;
pub(crate) fn router_tile_kernel(m: usize, ne: usize, k: usize) -> Option<(&'static str, usize)> {
    if m == 0 || ne % 64 != 0 || k == 0 || k % 16 != 0 { return None; }
    Some(if m < ROUTER_TILE128_MIN { ("xq_gemm_f16_f32_tile64", 64) } else { ("xq_gemm_f16_f32_tile128", 128) })
}

/// TP-SP2: the prefill router GEMM (fp32 logits of `m` rows of `x` into `out`, raw device addresses)
/// with the kernel the served dispatch picks for a chunk of `c_class` rows (`rows_on` = the tunable
/// prefill.router_rows) — the moe_prefill launches, callable on a row block (row-offset pointers).
/// Every variant is the same single ascending-k chain per (row, expert), so a row's bits do not depend
/// on m or on the block's offset (gate: --probe-exl3-sppin's EXL3-SP-ROUTER section).
#[allow(clippy::too_many_arguments)]
pub(crate) fn router_rows_launch(l: &Launcher, out: u64, w: u64, x: u64, m: usize, c_class: usize, ne: usize,
                                 h: usize, rows_on: bool) -> Result<()> {
    if rows_on {
        if pfx1_router_rows8(c_class, ne, h) {
            xqlaunch!(l, "xq_gemm_f16_f32_rows8", ((ne as u32) / 64, m.div_ceil(8) as u32, 1), (64, 1, 1), 0,
                     (out, w, x, m as i32, ne as i32, h as i32))?;
        } else if let Some((kn, bm)) = router_tile_kernel(m, ne, h) {
            xqlaunch!(l, kn, ((ne as u32) / 64, m.div_ceil(bm) as u32, 1), (128, 1, 1), 0,
                     (out, w, x, m as i32, ne as i32, h as i32))?;
        } else {
            xqlaunch!(l, "xq_gemm_f16_f32_rows32", ((ne as u32) / 128, ((m + 31) / 32) as u32, 1), (128, 1, 1), 0,
                     (out, w, x, m as i32, ne as i32, h as i32))?;
        }
    } else {
        xqlaunch!(l, "xq_gemm_f16_f32", ((ne as u32) / 128, m as u32, 1), (128, 1, 1), 0,
                 (out, w, x, m as i32, ne as i32, h as i32))?;
    }
    Ok(())
}

/// TUNE T0c (p5e): PFX1 (d) per prefill chunk class (the rows8 twin is bitwise == rows32). The
/// alias --pfx1-off (1|all|<parts>) decides exactly as before whenever it is set. PFX1's other
/// parts stay env-only: (a) mt IS prefill.wide_mt's default (the rule); (b) headkv / (c) yraw are
/// memory policy at admit / load; (e) ltplans / (f) w16 are host policy (P).
pub(crate) const T_PFX1_ROWS8: tune::TunableDef = tune::TunableDef {
    slot: 82, id: "pfx1.rows8", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-binv EXL3-ROUTER8 (rows8 vs rows32)",
    env: "--pfx1-off", env_parse: || tune::env_str(crate::opt!("pfx1-off")).map(|_| (pfx1_mask() & PFX1_ROWS8 != 0) as i32),
    rev: 1, wp: "PFX1",
};

/// PFX1 (c): the widest N of any m = c prefill chain (attention q_proj + gate = 12288; the head
/// fc chains run m = 4c at N = 2560 = 10240c elements). The prefill lm_head is the only wider
/// chain and runs at m = 1 (N = vocab).
pub(super) const PF_CHAIN_NMAX: usize = 12288;

/// --exl3-head-fill=0 turns the chunked prefill's MTP head-KV fill off (diagnostic).
pub(super) fn head_fill_on() -> bool {
    crate::opts::var(crate::opt!("exl3-head-fill")).as_deref() != Ok("0")
}

/// PFX1 (b) gate knob --pfx1-headkv-poison=1: where the head-KV memset is skipped, fill the
/// slot's head KV with a NaN pattern instead (diagnostic only; costs the memset it replaces).
pub(super) fn pfx1_headkv_poison() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::opts::var(crate::opt!("pfx1-headkv-poison")).is_ok_and(|v| !v.is_empty() && v != "0");
        if on { crate::dprintln!("PFX1: --pfx1-headkv-poison — head KV NaN-poisoned at every fresh-prompt admit (gate)"); }
        on
    })
}

/// PFX1: one boot line with the live parts and the MT rule (FwdModel::load).
pub(super) fn pfx1_announce() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let m = pfx1_mask();
        let rule = match wide_mt_override() {
            Some(mt) => format!("MT={mt} for every chunk (--exl3-wide-mt override)"),
            None if pfx1_on(PFX1_MT) => format!("MT=1 for c <= {PFX1_MT1_MAX}, MT=2 for c < {PFX1_MT4_MIN}, MT=4 from {PFX1_MT4_MIN} rows"),
            None => "MT=4 for every chunk (PFX1 mt off)".to_string(),
        };
        let router = if !pfx1_on(PFX1_ROWS8) {
            "rows32 for every chunk (PFX1 rows8 off)".to_string()
        } else if let Some(max) = pfx1_rows8_max_override() {
            format!("rows8 when c <= {max} (--pfx1-rows8-max override)")
        } else {
            format!("rows8 when c <= {PFX1_ROWS8_MAX} and ceil(c/32)*(ne/128) < 2*{} SMs", dev_sms())
        };
        crate::dprintln!("PFX1: parts on = {m:#04x} (mt 1, headkv 2, yraw 4, rows8 8, ltplans 16, w16 32{}); \
                  MoE prefill expert GEMM: {rule}; router: {router}",
                 crate::opts::var(crate::opt!("pfx1-off")).map(|v| format!("; --pfx1-off={v}")).unwrap_or_default());
    });
}
