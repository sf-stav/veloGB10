//! Tunables of the prefill levers (wide MT, hc fuse, GDN scan, MoE prefill kernel / fold / pf2, router rows),
//! the moe_pf entry helpers and the exl3_chain_* launchers including `chain_wide_gemm` (the wide-arm guard
//! precedes its launch: AGENTS 1b-bis 8, graceful fallback). See `mod.rs` for the file map.

use super::*;

/// TUNE T0 (S24): the prefill wide-M chain's M-tiles per CTA (exl3_chain_impl m > 16, moe_prefill).
/// MT only regroups rows over CTAs — the per-row K-ascending mma sequence is unchanged (bitwise,
/// b7c8b3e; the B3 receipt's MT-by-chunk-size rule is the tuner's known-answer test).
/// p5e (PFX1 a): value 0 = the built-in rule — the B3 MT-by-chunk rule for the MoE expert GEMMs
/// (pfx1_moe_mt) and MT 4 for the dense wide chains; 1|2|4|8 = that MT for every wide launch of the
/// chunk class. The alias keeps PFX1's parse exactly (trimmed; only 1|2|4|8 count, else ignored).
pub(crate) const T_WIDE_MT: tune::TunableDef = tune::TunableDef {
    slot: 7, id: "prefill.wide_mt", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1, 2, 4, 8], default: 0, valid: tune::valid_any, xcheck: "--probe-exl3-wide (mt sweep)",
    env: "--exl3-wide-mt",
    env_parse: || tune::env_str(crate::opt!("exl3-wide-mt"))
        .and_then(|v| match v.trim().parse::<usize>() { Ok(mt @ (1 | 2 | 4 | 8)) => Some(mt as i32), _ => None }),
    rev: 2, wp: "S-A3-f-b / PFX1",
};

/// A5-P2 (PLAN/notes_2026-09-27/P2.md): the prefill hc elementwise chain fused and regridded —
/// hc_post's inject deferred into the next hc_pre's norm (xq_pf_hc_inj_norm, 128-thread CTAs) and
/// the mix as xq_pf_hc_mix_2560x4. Bitwise == xq_hc_inject + xq_hc_norm / xq_hc_mix4 (same leaf
/// chains + halving tree; --probe-exl3-binv EXL3-PF-HC). The hc GEMMs (cuBLASLt, fp16 weights)
/// are unchanged. Alias --pf-hc-fuse=1. Default 0 (OFF) until integrated.
pub(crate) const T_PF_HC_FUSE: tune::TunableDef = tune::TunableDef {
    slot: 95, id: "prefill.hc_fuse", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-binv EXL3-PF-HC (fused vs served chain)",
    env: "--pf-hc-fuse", env_parse: || tune::env_on_if_1(crate::opt!("pf-hc-fuse")), rev: 1, wp: "A5-P2",
};

/// A5-L1 (PLAN/notes_2026-09-27/L1.md): the chunkwise GDN prefill scan through xq_gdn_chunk_tc2 —
/// the served xq_gdn_chunk_tc (split 7) re-scheduled: the forward substitution as a 31-step
/// shuffle wavefront (each row keeps its exact ascending FFMA chain), U stores fused into the solve,
/// the log-g scan in-warp, next sub-chunk's global loads prefetched into registers. Bitwise
/// (--probe-exl3-binv EXL3-GDN-TC2). Alias --pf-gdn-scan=1. Default 0 (OFF) until integrated.
pub(crate) const T_PF_GDN_SCAN: tune::TunableDef = tune::TunableDef {
    slot: 96, id: "prefill.gdn_scan", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-binv EXL3-GDN-TC2 (tc2 vs served chunk_tc)",
    env: "--pf-gdn-scan", env_parse: || tune::env_on_if_1(crate::opt!("pf-gdn-scan")), rev: 1, wp: "A5-L1",
};

/// A5-P1 (PLAN/notes_2026-09-27/P1.md): the prefill MoE routed-expert GEMMs through the
/// pipelined per-MT twin xq_moe_pf_t{1,2,4,8} (cp.async double-buffered A, next-stage trellis
/// words in registers, per-MT registers/smem) instead of exl3_hmma_gemm_wide_tiles. Same per-warp
/// decode + K-ascending mma chain per output element => bitwise (--probe-exl3-binv EXL3-MOE-PF).
/// Alias --pf-moe-kernel=1. Default 0 (OFF) until integrated.
pub(crate) const T_PF_MOE_KERNEL: tune::TunableDef = tune::TunableDef {
    slot: 91, id: "prefill.moe_kernel", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-binv EXL3-MOE-PF (pf vs wide_tiles)",
    env: "--pf-moe-kernel", env_parse: || tune::env_on_if_1(crate::opt!("pf-moe-kernel")), rev: 1, wp: "A5-P1",
};
/// A5-P1 / WP19: the prefill MoE M-tile keyed on each expert's OWN routed-row count (cnt >= 128
/// -> MT 8, >= 33 -> 4, >= 17 -> 2, else 1; xq_moe_pf_count4, one launch per MT class) instead of
/// the chunk-rows rule (pfx1_moe_mt). MT only regroups rows (bitwise across MT, b7c8b3e). Applies
/// to either kernel (prefill.moe_kernel 0|1). Alias --pf-moe-mt-pe=1. Default 0 (OFF).
pub(crate) const T_PF_MOE_MT_PE: tune::TunableDef = tune::TunableDef {
    slot: 92, id: "prefill.moe_mt_per_expert", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-binv EXL3-MOE-PF (per-expert MT lists)",
    env: "--pf-moe-mt-pe", env_parse: || tune::env_on_if_1(crate::opt!("pf-moe-mt-pe")), rev: 1, wp: "A5-P1 / WP19",
};
/// A5-P1: the prefill MoE glue folded into the pipelined kernel's epilogues (needs
/// prefill.moe_kernel = 1): gate/up = xq_moe_pf_g* (gate|up tile pair per CTA; epilogue ==
/// xq_had_svh_rows -> xq_moe_gate_mul -> xq_had_suh_rows(down) => writes xhd), down = xq_moe_pf_s*
/// (epilogue == xq_had_svh_rows => writes yd). Bitwise (EXL3-MOE-PF fold rows). Alias
/// --pf-moe-fold=1. Default 0 (OFF).
pub(crate) const T_PF_MOE_FOLD: tune::TunableDef = tune::TunableDef {
    slot: 93, id: "prefill.moe_glue_fold", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-binv EXL3-MOE-PF (fold rows)",
    env: "--pf-moe-fold", env_parse: || tune::env_on_if_1(crate::opt!("pf-moe-fold")), rev: 1, wp: "A5-P1",
};
/// A5-L2 (PLAN/notes_2026-09-27/L2.md): the prefill MoE expert GEMMs through xq_moe_pf2_{t,s,g}{MT}
/// — the P1 kernel's per-warp work (same ring words, decode, A fragments, K-ascending m16n8k16
/// chain, f16 RN epilogues) with the trellis in a cp.async smem ring instead of registers: P
/// stages in flight (was 1), MT <= 2 t/s at 3 CTAs/SM (the freed registers), and (v2) the MT <= 2
/// gate|up fold as paired 256-thread CTAs (each warp decodes the gate and the up block). Bitwise
/// (--probe-exl3-binv EXL3-MOE-PF, pf2 rows). The value is an MT-class MASK: bit k = the MT 1<<k
/// launches (per-expert class k, or the chunk-rule MT) use pf2; 15 = every class. Needs
/// prefill.moe_kernel = 1. Alias --pf-moe-pf2=<mask 0..15>. Default 0 (OFF).
pub(crate) const T_PF_MOE_PF2: tune::TunableDef = tune::TunableDef {
    slot: 97, id: "prefill.moe_pf2", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], default: 3, valid: tune::valid_any,
    xcheck: "--probe-exl3-binv EXL3-MOE-PF (pf2 vs pf)",
    env: "--pf-moe-pf2", env_parse: || tune::env_str(crate::opt!("pf-moe-pf2")).map(|v| v.trim().parse::<i32>().ok().unwrap_or(0).clamp(0, 15)),
    rev: 1, wp: "A5-L2",
};
/// (KS, P) of each pf2 instance (mirror of the XQ_PF2 entry table in kernels/exl3_bench.cu).
pub(crate) fn moe_pf2_cfg(mt: usize) -> (usize, usize) {
    match mt { 1 | 2 => (4, 3), 4 => (4, 2), _ => (2, 2) }
}
/// Mirror of XqPf2Smem<MT, NW, KS, P, BITS>::halves * 2 (EXL3-MOE-PF checks it against
/// xq_moe_pf2_smem_query for bits 3 and 4).
pub(crate) fn moe_pf2_smem_bytes(mt: usize, nw: usize, bits: usize) -> u32 {
    let (ks, p) = moe_pf2_cfg(mt);
    let rows = mt * 16;
    let asl = rows * (ks * 16 + 8);
    let tw = ks * (nw / 8) * 8 * 8 * bits;
    let ring = (p + 1) * (asl + 2 * tw);
    let epi = (nw / 8) * rows.min(64) * 136;
    (ring.max(epi) * 2) as u32
}
/// A5-L2 v2: the paired gate|up entries xq_moe_pf2_h{1,2} (256 threads, 8 warps each decoding the
/// gate AND the up block, 2 CTAs/SM; KS 4, P 2, 2 trellis runs) — mirror of XqPf2Smem<MT, 8, 4, 2,
/// BITS, 2>::halves * 2.
pub(crate) fn moe_pf2h_smem_bytes(mt: usize, bits: usize) -> u32 {
    let (ks, p) = (4usize, 2usize);
    let rows = mt * 16;
    let asl = rows * (ks * 16 + 8);
    let tw = ks * 2 * 8 * 8 * bits;
    let ring = (p + 1) * (asl + 2 * tw);
    let epi = 2 * rows.min(64) * 136;
    (ring.max(epi) * 2) as u32
}
/// The L2 entry for (kind, MT).
pub(crate) fn moe_pf2_fn(kind: char, mt: usize) -> &'static str {
    match (kind, mt) {
        ('t', 1) => "xq_moe_pf2_t1", ('t', 2) => "xq_moe_pf2_t2", ('t', 4) => "xq_moe_pf2_t4", ('t', _) => "xq_moe_pf2_t8",
        ('s', 1) => "xq_moe_pf2_s1", ('s', 2) => "xq_moe_pf2_s2", ('s', 4) => "xq_moe_pf2_s4", ('s', _) => "xq_moe_pf2_s8",
        ('g', 1) => "xq_moe_pf2_g1", ('g', 2) => "xq_moe_pf2_g2", ('g', 4) => "xq_moe_pf2_g4", (_, _) => "xq_moe_pf2_g8",
    }
}
/// (entry, dynamic smem, threads) of one prefill MoE expert launch: the pf2 family when `mask`
/// holds the MT class and the instance fits the 48 KB no-opt-in budget (bits 3 always does), else
/// the P1 entry. pf2 family: t/s -> xq_moe_pf2_{t,s}{MT}; g -> the v2 paired xq_moe_pf2_h{MT} for
/// MT <= 2 (256 threads), xq_moe_pf2_g{MT} for MT 4 / 8 (512 threads).
pub(crate) fn moe_pf_pick(kind: char, mt: usize, mask: i32, bits: usize) -> (&'static str, u32, u32) {
    let nw = if kind == 'g' { 16 } else { 8 };
    let k = mt.trailing_zeros() as i32;
    if mask & (1 << k) != 0 {
        if kind == 'g' && mt <= 2 {
            let sm = moe_pf2h_smem_bytes(mt, bits);
            if sm <= 48 * 1024 { return (if mt == 1 { "xq_moe_pf2_h1" } else { "xq_moe_pf2_h2" }, sm, 256); }
        } else {
            let sm = moe_pf2_smem_bytes(mt, nw, bits);
            if sm <= 48 * 1024 { return (moe_pf2_fn(kind, mt), sm, (nw * 32) as u32); }
        }
    }
    (moe_pf_fn(kind, mt), moe_pf_smem_bytes(mt, nw), (nw * 32) as u32)
}
/// Mirror of XqPfSmem<MT, NW>::halves * 2 (kernels/exl3_bench.cu; EXL3-MOE-PF checks the mirror).
pub(crate) fn moe_pf_smem_bytes(mt: usize, nw: usize) -> u32 {
    let rows = mt * 16;
    let ks = if mt == 8 { 4 } else { 8 };
    let a = 2 * rows * (ks * 16 + 8);
    let t = (nw / 8) * rows.min(64) * 136;
    (a.max(t) * 2) as u32
}
/// The P1 entry for (kind, MT): kind 't' raw y, 's' + svh epilogue, 'g' gate|up fold.
pub(crate) fn moe_pf_fn(kind: char, mt: usize) -> &'static str {
    match (kind, mt) {
        ('t', 1) => "xq_moe_pf_t1", ('t', 2) => "xq_moe_pf_t2", ('t', 4) => "xq_moe_pf_t4", ('t', _) => "xq_moe_pf_t8",
        ('s', 1) => "xq_moe_pf_s1", ('s', 2) => "xq_moe_pf_s2", ('s', 4) => "xq_moe_pf_s4", ('s', _) => "xq_moe_pf_s8",
        ('g', 1) => "xq_moe_pf_g1", ('g', 2) => "xq_moe_pf_g2", ('g', 4) => "xq_moe_pf_g4", (_, _) => "xq_moe_pf_g8",
    }
}
/// WP19 per-class tile-list capacities (host bounds for the grid's y extent): class k = MT 1<<k.
pub(crate) fn moe_pf_class_caps(r: usize, ne: usize) -> [usize; 4] {
    [ne.min(r), ne.min(r / 17), 2 * ne.min(r / 33), r / 64 + 1]
}

// TUNE T0 (p3c): the remaining bitwise switches on the line (each alias parsed exactly as its old
// per-call / OnceLock read). The prefill twins are Chunk-scoped (the prefill unit is not built yet:
// registered, never searched); the graph switches are Global.
pub(crate) const T_ROUTER_FUSED: tune::TunableDef = tune::TunableDef {
    slot: 47, id: "router.fused", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--exl3-router-xcheck (fused vs launches)",
    env: "--exl3-router-fused", env_parse: || tune::env_on_unless_0(crate::opt!("exl3-router-fused")), rev: 1, wp: "S-A3-z",
};
pub(crate) const T_PF_ROUTER_ROWS: tune::TunableDef = tune::TunableDef {
    slot: 48, id: "prefill.router_rows", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-wide (rows32 twin)",
    env: "--exl3-router-rows", env_parse: || tune::env_on_unless_0(crate::opt!("exl3-router-rows")), rev: 1, wp: "S-A3-q",
};
pub(crate) const T_PF_MOE_GROUPED: tune::TunableDef = tune::TunableDef {
    slot: 49, id: "prefill.moe_grouped", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-prefillx (grouped vs per-expert)",
    env: "--exl3-moe-grouped", env_parse: || tune::env_on_unless_0(crate::opt!("exl3-moe-grouped")), rev: 1, wp: "S-A3-q",
};
pub(crate) const T_PF_CONV_PAR: tune::TunableDef = tune::TunableDef {
    slot: 50, id: "prefill.conv_par", class: tune::Class::S, scope: tune::Scope::Chunk, fams: &[tune::Fam::Prefill],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-prefillx (parallel conv vs serial)",
    // --exl3-conv-serial set (any value) = the serial loop (value 0)
    env: "--exl3-conv-serial", env_parse: || tune::env_str(crate::opt!("exl3-conv-serial")).map(|_| 0), rev: 1, wp: "A5-D17",
};
pub(crate) const T_REPRIME_GRAPH: tune::TunableDef = tune::TunableDef {
    slot: 43, id: "reprime.graph", class: tune::Class::S, scope: tune::Scope::Global, fams: &[tune::Fam::Reprime],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--reprime-xcheck (graphed vs eager device re-prime)",
    env: "--exl3-reprime-graph", env_parse: || tune::env_on_unless_0(crate::opt!("exl3-reprime-graph")), rev: 1, wp: "REPRIME",
};
pub(crate) const T_DRAFT_GRAPH: tune::TunableDef = tune::TunableDef {
    slot: 44, id: "draft.graph", class: tune::Class::S, scope: tune::Scope::Global,
    fams: &[tune::Fam::DraftPass, tune::Fam::DraftChain, tune::Fam::Util],
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "trivial: the same launches, captured or eager",
    env: "--exl3-draft-graph", env_parse: || tune::env_on_unless_0(crate::opt!("exl3-draft-graph")), rev: 1, wp: "S-A3-o",
};

/// Prefill-side chain that MAY take the reconstruct+cuBLASLt arm (rb present AND
/// --exl3-recon-min > 0 AND m >= min). Otherwise byte-identical to exl3_chain.
pub(super) fn exl3_chain_rb(l: &Launcher, q: &Quad, x: &CudaSlice<u16>, xh: &mut CudaSlice<u16>,
                 yraw: &mut CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize,
                 ws: &CudaSlice<f32>, rb: &ReconBufs) -> Result<()> {
    // PFX1 (c): every prefill chain but the qsa indexer and the m = 1 lm_head goes through here
    // (m = c, or 4c for the head fc chains)
    pf_chain_check_n(q, m, xh, yraw)?;
    exl3_chain_impl(l, q, x, xh, yraw, y, m, ws, Some(rb))
}

/// PFX1 (c): a prefill chain's staging must fit the PrefillScratch buffers (m*K into chain_xh,
/// m*N into chain_yraw) — a loud error instead of an out-of-bounds write if a future chain
/// outgrows the max(c*12288, vocab) chain_yraw. The m = 1 prefill lm_head uses this form.
pub(super) fn pf_chain_check(q: &Quad, m: usize, xh: &CudaSlice<u16>, yraw: &CudaSlice<u16>) -> Result<()> {
    anyhow::ensure!(m * q.k as usize <= xh.len() && m * q.n as usize <= yraw.len(),
                    "prefill chain m={m} K={} N={} exceeds chain_xh {} / chain_yraw {}",
                    q.k, q.n, xh.len(), yraw.len());
    Ok(())
}

/// EXL27B: exl3_chain_rb for a dense pack's MLP chains (a copy): their N (27B: 17408) is wider than
/// PF_CHAIN_NMAX, so only the capacity check applies (the dense chain_yraw is sized for it).
pub(super) fn exl3_chain_rb_dense(l: &Launcher, q: &Quad, x: &CudaSlice<u16>, xh: &mut CudaSlice<u16>,
                       yraw: &mut CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize,
                       ws: &CudaSlice<f32>, rb: &ReconBufs) -> Result<()> {
    pf_chain_check(q, m, xh, yraw)?;
    exl3_chain_impl(l, q, x, xh, yraw, y, m, ws, Some(rb))
}

/// PFX1 (c): pf_chain_check + N <= PF_CHAIN_NMAX — every m = c (or 4c) prefill chain site.
pub(super) fn pf_chain_check_n(q: &Quad, m: usize, xh: &CudaSlice<u16>, yraw: &CudaSlice<u16>) -> Result<()> {
    anyhow::ensure!(q.n as usize <= PF_CHAIN_NMAX,
                    "prefill chain N={} exceeds PF_CHAIN_NMAX={PF_CHAIN_NMAX} (chain_yraw diet)", q.n);
    pf_chain_check(q, m, xh, yraw)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn exl3_chain_impl(l: &Launcher, q: &Quad, x: &CudaSlice<u16>, xh: &mut CudaSlice<u16>,
                   yraw: &mut CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize,
                   ws: &CudaSlice<f32>, rb: Option<&ReconBufs>) -> Result<()> {
    // FP4L: an FP4-Local weight never reaches a trellis kernel (no Hadamards; y = x W^T directly).
    if let Some(fx) = q.fx.as_deref() { return fp4l_fwd::fx_chain(l, q, fx, x, y, m, ws, rb); }
    if m <= 16 && q.ks > 1 && dense::dense_on() {
        // W4/DENSE: the whole split-K chain (suh + ks GEMM + combine + svh) in ONE launch on the
        // LMH streaming body — bitwise this function's path below. Ok(false) = not taken, nothing
        // launched (shape / buffer / grid outside the kernel's class). Package default OFF (p5e;
        // opt-in --w4dense=1): then this block is skipped and the p3 path below runs.
        use dense::{dptr, DOut};
        let o = DOut { y: dptr(y), y_len: y.len(), yraw: dptr(yraw), yraw_len: yraw.len(), xh: dptr(xh), xh_len: xh.len() };
        if dense::dense_chains(l, &[q], &[o], (dptr(x), x.len()), (0, 0), m, &[(dptr(ws), ws.len())])? {
            return Ok(());
        }
    }
    xqlaunch!(l, "exl3_had_suh", (m as u32 * (q.k as u32 / 128), 1, 1), (32, 1, 1), 0, (x, &q.suh, &mut *xh, q.k))?;
    // W3/LMH: true when the persistent lm_head GEMM already applied the output Hadamard.
    let mut svh_done = false;
    if m <= 16 {
        // Decode/verify contract path (G-A3-5, batch invariance).
        // S-A3-f-f ITEM 2: chains with ks > 1 run the split-K body (grid
        // (N/128, ks), fp32 partials) + the fixed-order ascending combine. ks is a
        // pure function of (K, N): the reduction per row is IDENTICAL at every m,
        // so batch invariance holds; the order vs ks=1 differs (re-baseline
        // protocol, brief §5). ks == 1 (lm_head) keeps the ORIGINAL kernel,
        // byte-identical launch shape.
        if q.ks > 1 {
            let kb = (q.k as u32) / 16;
            anyhow::ensure!(kb % q.ks == 0, "chain ks {} does not divide kb {}", q.ks, kb);
            let need = q.ks as usize * m * q.n as usize;
            anyhow::ensure!(need <= ws.len(), "chain_ws too small: need {} have {}", need, ws.len());
            // w3-PSK: persistent grid (bitwise == exl3_hmma_gemm_ks for every G); 0 = old grid.
            // PSK is opt-in (--psk=1) and keeps the combine + svh tail (no WP21 fixup in its body).
            let g = psk_grid(l.dev, q.k, q.n, q.bits, q.ks, false);
            if g > 0 {
                // trailing &q.tr = the prefetch base (same address; see PSK_CFG_* note)
                xqlaunch!(l, "exl3_hmma_gemm_ks_p", (g, 1, 1), (256, 1, 1), 0,
                         (&q.tr, &*xh, ws, m as i32, q.k, q.n, psk_cfg(q.bits, q.ks), &q.tr))?;
                if psk_xcheck() { psk_xcheck_single(l, q, *xh.device_ptr() as u64, ws, m, g)?; }
            } else if wp21_on(q.ks) && wp21_cnt_ok(q) && wp21_fx_width_ok(q) {
                // WP21: split-K fixup epilogue — the last-arriving CTA per N-tile does the
                // ascending combine + H128 svh in-launch (exl3_ks_combine and exl3_had_svh
                // are not launched; bitwise the same y). --wp21-off=1 = the path below.
                wp21_occupancy_once(l);
                xqlaunch!(l, "exl3_hmma_gemm_ks_fx", ((q.n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
                         (&q.tr, &*xh, ws, &q.svh, &mut *y, &q.cnt, m as i32, q.k, q.n, q.k2))?;
                if wp21_xcheck() { wp21_check_single(l, q, xh, yraw, y, m, ws)?; }
                return Ok(());
            } else {
                xqlaunch!(l, "exl3_hmma_gemm_ks", ((q.n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
                         (&q.tr, &mut *xh, ws, m as i32, q.k, q.n, q.k2))?;
            }
            let mn = (m * q.n as usize) as u32;
            xqlaunch!(l, "exl3_ks_combine", ((mn + 255) / 256, 1, 1), (256, 1, 1), 0,
                     (ws, &mut *yraw, m as i32, q.n, q.ks as i32))?;
        } else if q.lmh && !lmh_off() && {
            // W3/LMH: the lm_head class (ks == 1) — persistent 1-CTA/SM GEMM, bitwise ==
            // exl3_hmma_gemm (+ exl3_had_svh when fused). Ok(false) = ineligible -> old kernel.
            let out_p = if lmh_had() { *y.device_ptr() } else { *yraw.device_ptr() };
            lmh_gemm_launch(l.dev, l.stream.stream, *q.tr.device_ptr(), *xh.device_ptr(), out_p,
                            *q.svh.device_ptr(), m, q.k as usize, q.n as usize, q.bits as usize,
                            lmh_nst(), lmh_had())?
        } {
            svh_done = lmh_had();
            if lmh_xcheck() && !stream_capturing(l) {
                let got: &CudaSlice<u16> = if svh_done { &*y } else { &*yraw };
                lmh_xcheck_run(l, q, &*xh, got, m, svh_done)?;
            }
        } else {
            xqlaunch!(l, "exl3_hmma_gemm", ((q.n as u32) / 128, 1, 1), (256, 1, 1), 0, (&q.tr, &mut *xh, &mut *yraw, m as i32, q.k, q.n, q.k2))?;
        }
    } else {
        chain_wide_gemm(l, q, xh, yraw, m, rb)?;
    }
    if !svh_done {
        xqlaunch!(l, "exl3_had_svh", (m as u32 * (q.n as u32 / 128), 1, 1), (32, 1, 1), 0, (&mut *yraw, &q.svh, &mut *y, q.n))?;
    }
    Ok(())
}

/// TP-B: the m > 16 chain up to y_raw (had_suh + the wide / recon GEMM), no output transform —
/// the row-parallel prefill partial (xtp::chain_rowpar).
pub(super) fn chain_wide_raw(l: &Launcher, q: &Quad, x: &CudaSlice<u16>, xh: &mut CudaSlice<u16>,
                  yraw: &mut CudaSlice<u16>, m: usize, rb: Option<&ReconBufs>) -> Result<()> {
    anyhow::ensure!(m > 16, "chain_wide_raw is the m > 16 arm");
    // FP4L: the raw output IS y (no output transform); the f16 partial lands in yraw.
    if let Some(fx) = q.fx.as_deref() {
        return match rb {
            Some(r) => { crate::metrics::fp4l_gemm(fx.kind(), m); fp4l_fwd::fx_wide(l, q, fx, *x.device_ptr() as u64, *yraw.device_ptr() as u64, m, r) }
            None => anyhow::bail!("FP4L: a row-parallel m > 16 chain reached chain_wide_raw without recon buffers"),
        };
    }
    xqlaunch!(l, "exl3_had_suh", (m as u32 * (q.k as u32 / 128), 1, 1), (32, 1, 1), 0, (x, &q.suh, &mut *xh, q.k))?;
    chain_wide_gemm(l, q, xh, yraw, m, rb)
}

/// F3: the once-per-boot note for a module the wide arm cannot serve (forced to the reconstruct arm).
static WIDE_FALLBACK_NOTE: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// The m > 16 GEMM of exl3_chain_impl (xh -> y_raw), factored out for chain_wide_raw.
fn chain_wide_gemm(l: &Launcher, q: &Quad, xh: &mut CudaSlice<u16>, yraw: &mut CudaSlice<u16>, m: usize,
                   rb: Option<&ReconBufs>) -> Result<()> {
    {
        // S-A3-f-g dispatch: reconstruct+cuBLASLt when the caller supplied buffers
        // AND --exl3-recon-min > 0 AND m >= the threshold; else the wide-M
        // trellis kernel (S-A3-f-b) — row-wise bitwise equal to the m16 body.
        // F3 (owner 2026-10-10): a module the wide kernel cannot decode (any width outside 3..=7
        // since P7 B5 — the 4.05 pack's 6-bit modules now take the wide arm) runs the reconstruct arm WHATEVER --exl3-recon-min
        // says — the option can no longer route such a module into a kernel that would leave its
        // output unwritten. Once per boot, one line names the option, the module width and what is
        // used instead; /metrics counts the served GEMMs (`velogb10_exl3_recon_forced_total`).
        let rmin = recon_min_rows();
        let recon = match rb {
            Some(r) if !wide_trellis_supports_bits(q.bits as usize) => {
                if WIDE_FALLBACK_NOTE.set(()).is_ok() {
                    crate::drprintln!("[exl3] --exl3-recon-min {rmin}: this pack has K={:.1} modules the wide trellis kernel \
cannot decode (K 3, 4, 5 only); those modules serve through the reconstruct + cuBLASLt arm, whatever the option says",
                                     q.k2 as f32 / 2.0);
                }
                crate::metrics::exl3_recon_forced(q.bits);
                Some(r)
            }
            Some(r) if rmin > 0 && m >= rmin => Some(r),
            _ => None,
        };
        // LANEWIDE-MID (--exl3-mid-gemm on): 17..=MID_MAX rows through the split-K trellis body with MT m-tiles per CTA
        // (decode once per k16 step for every row; per row bitwise the decode-class chain), then the same combine
        if let Some(r) = recon.as_ref().filter(|_| dres::mid_fits(q, m)) {
            if dres::mid_gemm(l, q, xh, yraw, m, r)? { return Ok(()); }
        }
        if let Some(r) = recon {
            let need = (q.k as usize) * (q.n as usize);
            anyhow::ensure!(need <= r.ws_len,
                "recon staging too small: need {need} have {}", r.ws_len);
            // LANEWIDE-DRES (--exl3-dense-resident): the matrix decoded once and kept, else today's per-pass decode
            let wdec = match dres::kept_ptr(l, q, r.rc)? {
                Some(p) => p,
                None => {
                    exl3_reconstruct(l, *q.tr.device_ptr() as u64, r.rc, r.ws, q.k, q.n, q.k2)?;
                    r.ws
                }
            };
            exl3_lt_matmul(l, wdec, *xh.device_ptr() as u64, *yraw.device_ptr() as u64,
                           q.k as usize, q.n as usize, m, false, r.ltws, r.ltws_len)?;
        } else {
            // S-B9-REL-INTEG D1: no 1-/2-bit arm exists here (8-bit joined in PR#27-D2): refuse, never launch a kernel
            // whose `default: break;` writes nothing.
            anyhow::ensure!(wide_trellis_supports_bits(q.bits as usize),
                            "{}", wide_arm_bits_refusal(q.bits as usize, m, q.k, q.n, rmin, rb.is_some()));
            // S-A3-f-b chunked prefill (M > 16): wide-M kernel — row-wise bitwise
            // equal to the m16 body (same decode, same fixed K-ascending mma
            // sequence per element); gated by --probe-exl3-wide before serving use.
            // Decode/verify (m <= 16) never reaches this arm; AGENTS §6 discipline.
            // (dense chains: only reached with --exl3-recon-min=0 or no recon buffers; the
            // PFX1 MT rule is calibrated on the MoE rows and does not apply here)
            // TUNE S24: prefill.wide_mt (0 = the built-in value 4 here; 1|2|4|8 = a fixed MT)
            let mt: usize = match tune::get(&T_WIDE_MT, &l.cx()) { v @ (1 | 2 | 4 | 8) => v as usize, _ => 4 };
            let msup = (m + 16 * mt - 1) / (16 * mt);
            xqlaunch!(l, "exl3_hmma_gemm_wide", (msup as u32, (q.n as u32) / 128, 1), (256, 1, 1), 0, (&q.tr, &mut *xh, &mut *yraw, m as i32, q.k, q.n, q.bits, mt as i32))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// EXL27C-DF2: --df2-trace (CLI flag, serve side parses): stage syncs + prints inside
/// df2_round, to localize async faults to draft / verify / staging / inject.
pub static DF2_TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
