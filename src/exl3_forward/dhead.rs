//! DHEAD (w2, 2026-09-26): draft lm_head diet — DRAFT-ONLY (verification is untouched, so greedy
//! output bytes cannot change; only which tokens get drafted can).
//!
//! The draft pass's pruned lm_head (EXL3 5-bit column slice of the shared head, 65,536 x 2560 =
//! 104.9 MB per pass) was 44% of a draft pass: had_suh + exl3_hmma_gemm (0.52 ms, ~85% of its
//! byte floor) + had_svh + argmax_rows + xq_rowmax_f16 + 2 x xq_copy1_i32. It runs up to 5x a round.
//! This module replaces that tail (m = 1 graphed draft passes only) with three launches — see the
//! "DHEAD" header in kernels/exl3_bench.cu for the device side:
//!   xq_dh_prep    xh (bitwise the old had_suh) + int8 screen x,
//!   xq_dh_screen  a B-bit (default 2) screen of the same rotated weights built at load: 41.9 MB,
//!   xq_dh_rescore exact EXL3 logits of the top candidate blocks -> argmax, dconf, d_dev, toks.
//! When the old argmax's 128-column block is a candidate (XCHECK reports the rate), the draft id and
//! the DDS confidence equal the old head's up to split-K fp32 re-association before the f16 yraw
//! rounding; otherwise the draft is the best in-candidate token.
//!
//! Escapes / knobs (diagnostics; the defaults are the served config):
//!   --dhead-off=1       the old head (no screen built, no memory spent)
//!   --dhead-xcheck=1    eager (draft graphs off) per-pass new-vs-old comparison + per-request line
//!   --dhead-bits=1|2|4  screen width (default 2)
//!   --dhead-t=<1..32>   candidate cap (default 16)
//!   --dhead-delta=<f>   candidate margin in screen-noise sigmas (default 4)
//!
//! DHEADP (w3, 2026-09-27): the same tail for PENALIZED drafts (the WP15 draft mirror), which
//! used to fall back to the old head. With monotone params (rep >= 1, pres >= 0, freq >= 0) a
//! penalty never raises a logit, so DHEAD's per-block screen upper bound UB_b = bmax + delta *
//! sigma * max|svh_b| also bounds the penalized logits. xq_dh_rescore_pa rescores DHEAD's candidate
//! set C1 exactly, penalizes it with the draft mirror's formula (xq_pen_tok) and window counts,
//! and takes the best penalized value L (a real value: a lower bound on the penalized max); every
//! block outside C1 with UB_b >= L is then rescored + penalized by xq_dh_rescore_pb before the
//! argmax. More than --dheadp-cap qualifying blocks (or a degenerate L, or non-monotone params
//! reaching the device) = every block is rescored: the old head's function, counted per reason.
//! Non-monotone requests are routed to the old head on the host (draft_tail / pen_draft_key).
//!   --dheadp-off=1      penalized drafts keep the old head (today's behaviour)
//!   --dheadp-cap=<n>    expansion cap in blocks (default 256; above it: all blocks). p5c:
//!                          32 -> 256 (cap sweep on .14, 2026-09-27, binary 27936955: over-cap
//!                          fallbacks R1 5.23% -> 0.08%, R2 11.58% -> 1.37%, R3 0.40% -> 0; hashes and
//!                          round/draft/accept counts identical at every cap; never slower)
//!   --dhead-xcheck=1    also compares every penalized pass with the old penalized head
//!
//! A5-L4 (spec.ratio_dhead, registry slot 98, N class, env --spec-ratio-dhead=1, default OFF):
//! UNPENALIZED real-q (--spec-sampling ratio) draft passes run DHEAD too — screen + exact rescore,
//! then xq_rq_draft_dh builds the proposal q' (request top-k -> tau -> top-p) over DHEAD's exactly
//! rescored candidate set and samples the draft from it (list row i = what the verify reads).
//! Tokens outside the candidate blocks get q' = 0. Speculative sampling is exact for any q' drawn
//! from and judged against the same list row, so the emitted DISTRIBUTION is unchanged; seeded
//! sampled BYTES change vs the full-slice q' (a different q' draws different drafts). Penalized
//! real-q passes keep the old full-slice tail. --dhead-xcheck=1 adds `RQDH_XCHECK` lines: the
//! full-slice q' recomputed on the same x — exact-q' rate, draft agreement, q' overlap
//! sum min(q'_full, q'_dhead) and the full q' mass inside the candidate blocks (coverage).

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// Candidate cap (== DH_TCAP in kernels/exl3_bench.cu).
const DH_TCAP: usize = 32;
/// Approximate block maxima per rescore thread (== DH_KPT): nblk <= 256 * DH_KPT.
const DH_KPT: usize = 8;
/// k16 rows per rescore K-split (== DH_RMAX).
const DH_RMAX: usize = 16;

/// --dhead-off=1: the old draft head (read once, at load).
pub(super) fn dhead_off() -> bool {
    crate::opts::var(crate::opt!("dhead-off")).map_or(false, |v| !v.is_empty() && v != "0")
}

/// --dheadp-off=1: penalized draft passes keep the old head (read once, at load).
/// TUNE T0c (p5e): dheadp.off / dheadp.cap are class D (draft-only: a penalized draft can differ,
/// the verified output cannot) and Load scope — never searched; a table's D section applies only with
/// --tune-draft on. The aliases keep their exact parses.
pub(super) fn dheadp_off() -> bool {
    tune::get_cur(&T_DHEADP_OFF) != 0
}
pub(crate) const T_DHEADP_OFF: tune::TunableDef = tune::TunableDef {
    slot: 83, id: "dheadp.off", class: tune::Class::D, scope: tune::Scope::Load,
    fams: &[tune::Fam::DraftPass, tune::Fam::DraftChain],
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "--dhead-xcheck + greedy hashes (D: draft-only)",
    env: "--dheadp-off",
    env_parse: || tune::env_str(crate::opt!("dheadp-off")).map(|v| (!v.is_empty() && v != "0") as i32),
    rev: 1, wp: "w3/DHEADP",
};
pub(crate) const T_DHEADP_CAP: tune::TunableDef = tune::TunableDef {
    slot: 84, id: "dheadp.cap", class: tune::Class::D, scope: tune::Scope::Load,
    fams: &[tune::Fam::DraftPass, tune::Fam::DraftChain],
    // expansion cap in blocks (p5c: 256; DHP_CAP_MAX = 2047 fits geomA)
    domain: &[32, 64, 128, 256, 512, 1024], default: 256, valid: tune::valid_any,
    xcheck: "--dhead-xcheck + greedy hashes (D: draft-only)",
    env: "--dheadp-cap",
    // env_or's parse (trimmed; unparsable = the default); a huge value still fails the build's cap check
    env_parse: || tune::env_str(crate::opt!("dheadp-cap")).and_then(|v| v.trim().parse::<usize>().ok())
        .map(|c| c.min(i32::MAX as usize) as i32),
    rev: 1, wp: "w3/DHEADP",
};
/// A5-L4: real-q (ratio) draft passes on DHEAD (see the module header). N class: distribution-exact
/// but seeded sampled bytes change — never tuned, owner consent to default. Load scope (read once
/// at DHead::build; the draft graph keys need no extra bit — the value is fixed per process).
pub(crate) const T_SPEC_RATIO_DHEAD: tune::TunableDef = tune::TunableDef {
    slot: 98, id: "spec.ratio_dhead", class: tune::Class::N, scope: tune::Scope::Load,
    fams: &[tune::Fam::DraftPass, tune::Fam::DraftChain],
    domain: &[0, 1], default: 1, valid: tune::valid_any,
    xcheck: "WP24 chi2 ladder (--probe-spec-sampling) + --wp24-xcheck / --dhead-xcheck RQDH lines",
    env: "--spec-ratio-dhead",
    env_parse: || tune::env_str(crate::opt!("spec-ratio-dhead")).map(|v| (!v.is_empty() && v != "0") as i32),
    rev: 1, wp: "A5-L4",
};

/// --dhead-xcheck=1: every eager draft pass also runs the old head on the same input and
/// compares (the draft-chain graphs are forced off so every pass is eager).
pub(super) fn dhead_xcheck() -> bool {
    static X: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *X.get_or_init(|| crate::opts::var(crate::opt!("dhead-xcheck")).map_or(false, |v| !v.is_empty() && v != "0"))
}

fn env_or<T: std::str::FromStr>(o: crate::opts::OptId, default: T) -> T {
    crate::opts::var(o).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// The screen's global uniform quantizer per width: (level spacing dq, MSE per weight). Fitted to
/// the mul1 codebook's marginal over its 65,536 states (mean -0.002, std 1.0003, range
/// [-3.45, 3.35]) — the decoded EXL3 weights are unit-variance by construction (the scale lives in
/// suh/svh), so one global quantizer serves every column and no group scales are stored.
/// Level c of L = 2^B is dq * (c - (L-1)/2); a weight w gets c = clamp(floor(w/dq + L/2), 0, L-1).
pub(super) fn screen_quant(b: u32) -> Option<(f32, f32)> {
    match b {
        1 => Some((1.616, 0.3471)),
        2 => Some((0.978, 0.1083)),
        4 => Some((0.311, 0.00928)),
        _ => None,
    }
}

/// Workspace layout in u32 words — mirrors dh_off_* in kernels/exl3_bench.cu:
/// (bmax, cnt, dbg, part, total).
pub(super) fn ws_layout(k: usize, nblk: usize, s: usize) -> (usize, usize, usize, usize, usize) {
    let bmax = ((k >> 2) + 4 + 3) & !3;
    let cnt = bmax + ((nblk + 3) & !3);
    let dbg = cnt + 4;
    let part = dbg + DH_TCAP + 4;
    (bmax, cnt, dbg, part, part + DH_TCAP * s * 128)
}

/// DHEADP workspace header (u32 words) — mirrors DHP_* in kernels/exl3_bench.cu.
const DHP_DBG: usize = 16;
const DHP_STAT: usize = 24;
const DHP_HDR: usize = 32;
const DHP_F_OVER: u32 = 1;
const DHP_F_NONMONO: u32 = 2;
const DHP_F_DEGEN: u32 = 4;
const DHP_F_UNPEN: u32 = 8;
const DHP_F_FB: u32 = 16;
/// Largest --dheadp-cap the pass geometry carries (11 bits of geomA).
const DHP_CAP_MAX: usize = 2047;

/// DHEADP workspace layout in u32 words: (cntc, c2, part, total) — mirrors dhp_nblk4 / DHP_HDR.
pub(super) fn pws_layout(nblk: usize, s: usize) -> (usize, usize, usize, usize) {
    let n4 = (nblk + 3) & !3;
    let cntc = DHP_HDR;
    let c2 = cntc + n4;
    let part = c2 + n4;
    (cntc, c2, part, part + nblk * s * 128)
}

/// The DHEADP bound in host arithmetic (the device's rule, for the unit tests and XCHECK): the
/// blocks outside C1 that pass B must rescore — UB_b = bmax[b] + max(delta * sigma * bsig[b], 0)
/// >= L, a NaN bound qualifying — or ALL blocks outside C1 when more than `cap` qualify, the
/// params are non-monotone, or L is degenerate (NaN / <= -65504). Returns (list ascending, flags).
#[allow(clippy::too_many_arguments)]
pub(super) fn expansion_set(bmax: &[f32], bsig: &[f32], in_c1: &[bool], sig: f32, delta: f32, l: f32,
                            cap: usize, monotone: bool) -> (Vec<usize>, u32) {
    let outside: Vec<usize> = (0..bmax.len()).filter(|&b| !in_c1[b]).collect();
    let qual: Vec<usize> = outside.iter().copied().filter(|&b| {
        let m = delta * sig * bsig[b];
        let ub = bmax[b] + if m > 0.0 { m } else { 0.0 }; // fmaxf(NaN, 0) = 0
        !(ub < l)
    }).collect();
    let degen = !(l > -65504.0);
    let over = qual.len() > cap;
    let fb = !monotone || degen || over;
    let fl = if !monotone { DHP_F_NONMONO } else { 0 } | if degen { DHP_F_DEGEN } else { 0 }
        | if over { DHP_F_OVER } else { 0 } | if fb { DHP_F_FB } else { 0 };
    (if fb { outside } else { qual }, fl)
}

/// Host mirror of the screen packing (xq_dh_build + xq_dh_prep): the byte offset in the permuted
/// int8 x array that holds k, and the (u32 word index within the column's k-group stream, bit
/// offset) that holds k's code. Used by the unit test that proves the dp4a pairing is k-for-k.
#[cfg(test)]
fn x_byte_of(k: usize, b: usize) -> usize {
    let (g_len, cpb) = (32 / b, 8 / b);
    let (g, kl) = (k / g_len, k % g_len);
    let (byte, j) = (kl / cpb, kl % cpb);
    (g * cpb + j) * 4 + byte
}
#[cfg(test)]
fn code_pos_of(k: usize, b: usize) -> (usize, u32) {
    let (g_len, cpb) = (32 / b, 8 / b);
    let (g, kl) = (k / g_len, k % g_len);
    let (byte, j) = (kl / cpb, kl % cpb);
    (g, (8 * byte + b * j) as u32)
}

pub(super) struct DHead {
    bits: u32,     // screen width B
    hbits: i32,    // trellis bits of the exact slice (rescore)
    k: usize,
    n: usize,
    nblk: usize,
    s: usize,      // rescore K-splits
    r: usize,      // k16 rows per split
    tmax: usize,
    delta: f32,
    dq: f32,
    mse: f32,
    codes: CudaSlice<u32>, // [nblk][K / (128/B)][128] x 16-B chunks
    bsig: CudaSlice<f32>,  // [nblk] max |svh| per block, then [nblk] = the max over blocks
    ws: CudaSlice<u32>,    // ws_layout
    xc_id: CudaSlice<i32>, // XCHECK: the old head's argmax / max logit
    xc_conf: CudaSlice<f32>,
    p: Option<DHeadP>,     // DHEADP (None = --dheadp-off=1 / build refused)
    rq: bool,              // A5-L4 spec.ratio_dhead: unpenalized real-q passes take DHEAD
    xc_list: CudaSlice<u32>, // RQDH XCHECK: the full-slice pass lists [(MTP_MAX_K + 1) * RQ_ROW]
    shard: Option<DhShard>,  // TP-H #3: the sharded screen (None = the replicated screen)
}

/// TP-H #3 (--tp-dh-shard, world 2): this rank screens `grid` of the `nblk` blocks (the codes stay
/// whole, indexed by the GLOBAL block) into `part` — a ZERO-PADDED [nblk] f32 exchange buffer whose
/// peer half is never written — and the doorbell adds the halves back into ws bmax (bit-exact: value + 0).
pub(super) struct DhShard {
    part: CudaSlice<f32>,
    nb_off: usize,
    grid: usize,
}

/// DHEADP runtime: the pass-A/B workspace + knobs + the per-request fallback accounting.
pub(super) struct DHeadP {
    pws: CudaSlice<u32>, // pws_layout
    cap: usize,          // --dheadp-cap
    g: usize,            // pass-B CTAs per K-split (one resident wave of the S x g grid)
    seen: AtomicBool,    // a round routed its penalized drafts here since the last report
    stat0: std::sync::Mutex<[u32; 8]>, // DHP_STAT at the last report
}

impl DHead {
    /// Build the screen from the exact pruned head `q` (lm_head_draft). Runs at load, outside any
    /// capture. Err = a refused shape/knob (the caller keeps the old head and says so).
    pub(super) fn build(dev: &Arc<CudaDevice>, stream: &CudaStream, q: &Quad) -> Result<DHead> {
        let t0 = std::time::Instant::now();
        let bits: u32 = env_or(crate::opt!("dhead-bits"), 2u32);
        let tmax: usize = env_or(crate::opt!("dhead-t"), 16usize);
        let delta: f32 = env_or(crate::opt!("dhead-delta"), 4.0f32);
        let (dq, mse) = screen_quant(bits)
            .ok_or_else(|| anyhow::anyhow!("--dhead-bits={bits} unsupported (1|2|4)"))?;
        let (k, n, hbits) = (q.k as usize, q.n as usize, q.bits);
        let nblk = n / 128;
        let kb = k / 16;
        anyhow::ensure!(k % 128 == 0 && k <= 4096, "K {k}: need K % 128 == 0 and K <= 4096");
        anyhow::ensure!(k % (128 / bits as usize) == 0, "K {k} not a multiple of the {bits}-bit chunk");
        anyhow::ensure!(n % 128 == 0 && nblk >= 1 && nblk <= 256 * DH_KPT, "N {n}: need N % 128 == 0, N <= {}", 128 * 256 * DH_KPT);
        anyhow::ensure!((2..=8).contains(&hbits), "head trellis bits {hbits} outside 2..=8");
        anyhow::ensure!((1..=DH_TCAP).contains(&tmax), "--dhead-t={tmax} outside 1..={DH_TCAP}");
        anyhow::ensure!(delta.is_finite() && delta >= 0.0, "--dhead-delta={delta}");
        let s = (1..=kb.min(255)).find(|&s| kb % s == 0 && kb / s <= DH_RMAX)
            .ok_or_else(|| anyhow::anyhow!("no K-split of {kb} k16 rows with <= {DH_RMAX} rows"))?;
        let r = kb / s;

        let nwords = n * k / (32 / bits as usize);
        let mut codes = dev.alloc_zeros::<u32>(nwords)?; // fully written by xq_dh_build
        let l = Launcher { dev, stream };
        xqlaunch!(l, "xq_dh_build", (((nwords + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&q.tr, &mut codes, k as i32, n as i32, hbits, bits as i32, 1.0f32 / dq, nwords as i64))?;
        dev.synchronize()?;

        let svh: Vec<u16> = dev.dtoh_sync_copy(&q.svh)?;
        anyhow::ensure!(svh.len() >= n, "svh len {} < N {n}", svh.len());
        let mut bs: Vec<f32> = (0..nblk)
            .map(|b| svh[b * 128..(b + 1) * 128].iter()
                .map(|&h| f16::from_bits(h).to_f32().abs())
                .fold(0f32, |a, v| if v > a { v } else { a }))
            .collect();
        let bmax_all = bs.iter().cloned().fold(0f32, |a, v| if v > a { v } else { a });
        bs.push(bmax_all);
        let bsig = dev.htod_sync_copy(&bs)?;
        let (_, _, _, _, total) = ws_layout(k, nblk, s);
        // a REAL zero (alloc_zeros does not zero, AGENTS §2.2): the rescore counter re-arms itself
        let ws = dev.htod_sync_copy(&vec![0u32; total])?;
        let xc_id = dev.htod_sync_copy(&[0i32])?;
        let xc_conf = dev.htod_sync_copy(&[0f32])?;
        dev.synchronize()?;
        let p = if dheadp_off() {
            println!("DHEADP: off (--dheadp-off=1) — penalized drafts keep the old draft head");
            None
        } else {
            match DHeadP::build(dev, nblk, s) {
                Ok(p) => Some(p),
                Err(e) => { println!("DHEADP: NOT built ({e:#}) — penalized drafts keep the old draft head"); None }
            }
        };
        let rq = tune::get_cur(&T_SPEC_RATIO_DHEAD) != 0; // A5-L4 (N, Load)
        let xc_list = dev.htod_sync_copy(&vec![0u32; (MTP_MAX_K + 1) * crate::wp24::RQ_ROW])?;
        if rq {
            println!("DHEAD-RQ: spec.ratio_dhead=1 — unpenalized real-q (ratio) draft passes run DHEAD + xq_rq_draft_dh \
                      (q' over the exactly rescored candidate blocks; distribution-exact, seeded sampled bytes change \
                      vs the full-slice q'; penalized real-q passes keep the full-slice head)");
        }
        println!("DHEAD: draft head = {bits}-bit screen ({:.1} MB/pass vs {:.1} MB EXL3 slice) + exact rescore \
                  of <= {tmax} candidate blocks (margin {delta} sigma, K-splits {s}); +{:.1} MB device, built in {:.0} ms \
                  (--dhead-off=1 = old head)",
                 (nwords * 4) as f64 / 1e6, (q.tr.len() * 2) as f64 / 1e6,
                 (nwords * 4 + total * 4 + bs.len() * 4) as f64 / 1e6, t0.elapsed().as_secs_f64() * 1e3);
        Ok(DHead { bits, hbits, k, n, nblk, s, r, tmax, delta, dq, mse, codes, bsig, ws, xc_id, xc_conf, p, rq, xc_list, shard: None })
    }

    /// TP-H #3: arm the sharded screen for rank `rank` of `world` (2; TP-4C: or 4 — each rank screens
    /// nblk / world blocks and the multi-round K2 reassembles the array, bitwise: one non-zero contributor
    /// per entry). Boot-time, outside any capture;
    /// every condition is a structural property of the model and the shipped env, so both ranks reach the
    /// same verdict (an Err fails the boot on both — never a one-sided degrade).
    pub(super) fn enable_shard(&mut self, dev: &Arc<CudaDevice>, rank: usize, world: usize) -> Result<()> {
        anyhow::ensure!((world == 2 || world == 4) && rank < world, "the sharded screen is world 2 or 4 only (rank {rank}/{world})");
        anyhow::ensure!(self.nblk % world == 0 && (self.nblk / world) >= 1, "nblk {} not divisible by world {world}", self.nblk);
        anyhow::ensure!(matches!(self.bits, 1 | 2 | 4), "screen width {} has no sharded screen kernel", self.bits);
        anyhow::ensure!((self.nblk * 4) % 16 == 0, "nblk {} floats are not a whole number of 16-byte vectors", self.nblk);
        // a REAL zero (alloc_zeros does not zero, AGENTS §2.2): the peer's half must stay 0.0 forever
        let part = dev.htod_sync_copy(&vec![0f32; self.nblk])?;
        dev.synchronize()?;
        let grid = self.nblk / world;
        self.shard = Some(DhShard { part, nb_off: rank * grid, grid });
        Ok(())
    }

    /// The sharded screen is armed (TP-H #3).
    pub(super) fn shard_live(&self) -> bool { self.shard.is_some() }

    /// (blocks screened per rank, first global block) of the armed shard — for the boot log.
    pub(super) fn shard_span(&self) -> Option<(usize, usize, usize)> {
        self.shard.as_ref().map(|s| (s.nb_off, s.nb_off + s.grid, self.nblk))
    }

    /// DHEADP is live (built, not escaped): monotone penalized drafts take dheadp_pass.
    pub(super) fn penalized_live(&self) -> bool {
        self.p.is_some()
    }

    /// A5-L4: unpenalized real-q passes take DHEAD (spec.ratio_dhead=1).
    pub(super) fn rq_live(&self) -> bool {
        self.rq
    }

    /// A round routed its penalized drafts to DHEADP (graphed or eager): the request report reads
    /// the device fallback counters only then.
    pub(super) fn mark_penalized_round(&self) {
        if let Some(p) = self.p.as_ref() { p.seen.store(true, Ordering::Relaxed); }
    }

    /// geom of pass i (xq_dh_rescore's layout; K < 2^16 so the pen kernels' extra fields fit above).
    fn geom(&self, i: usize) -> i64 {
        (i as i64) | ((self.hbits as i64) << 8) | ((self.s as i64) << 16)
            | ((self.tmax as i64) << 24) | ((self.k as i64) << 32)
    }
}

impl DHeadP {
    fn build(dev: &Arc<CudaDevice>, nblk: usize, s: usize) -> Result<DHeadP> {
        // p5c: default 256 (was 32) — see the --dheadp-cap note in the module header.
        let cap: usize = tune::get_cur(&T_DHEADP_CAP).max(0) as usize; // TUNE T0c: dheadp.cap (D)
        anyhow::ensure!(cap <= DHP_CAP_MAX, "--dheadp-cap={cap} > {DHP_CAP_MAX}");
        anyhow::ensure!(PEN_RING_LOG2 < 32, "penalty ring log2 {PEN_RING_LOG2} does not fit geomA");
        use cudarc::driver::sys;
        let sms = dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?.max(1) as usize;
        // pass B: 3 CTAs/SM (__launch_bounds__(256, 3)) -> one resident wave of s x g
        let g = ((3 * sms) / s.max(1)).clamp(1, nblk.max(1));
        let (_, _, _, total) = pws_layout(nblk, s);
        // REAL zeros (AGENTS §2.2): the split / done counters re-arm themselves from here on
        let pws = dev.htod_sync_copy(&vec![0u32; total])?;
        dev.synchronize()?;
        println!("DHEADP: penalized drafts (monotone params) = DHEAD screen + exact penalized rescore of C1, then every \
                  block with screen bound >= the best penalized C1 logit (cap {cap}, above it all blocks); pass-B grid \
                  {s} x {g}; +{:.1} MB device (--dheadp-off=1 = old head)", (total * 4) as f64 / 1e6);
        Ok(DHeadP { pws, cap, g, seen: AtomicBool::new(false), stat0: std::sync::Mutex::new([0; 8]) })
    }
}

/// XCHECK accumulators (since the last per-request report).
#[derive(Default)]
struct XStats {
    passes: u64,
    agree: u64,
    conf_eq: u64,
    captured: u64,
    cap_agree: u64,
    miss_agree: u64,
    conf_maxd: f32,
    ns_sum: u64,
    ns_max: usize,
    nq_sum: u64,
    trunc: u64,
    // approximate rank of the old argmax's block among the screen's block maxima:
    // 0, 1, 2-3, 4-7, 8-15, 16-31, 32+
    rank_hist: [u64; 7],
    printed: u32,
}

impl XStats {
    fn line(&self) -> String {
        let p = self.passes.max(1) as f64;
        format!("passes {} argmax-agree {} ({:.3}%) dconf-bitwise {} ({:.3}%) max|d dconf| {:e} | captured {} ({:.3}%, agree {}) \
                 missed {} (agree {}) | candidates mean {:.2} max {} qualified mean {:.1} truncated {} | old-block screen rank \
                 [0:{} 1:{} 2-3:{} 4-7:{} 8-15:{} 16-31:{} 32+:{}]",
                self.passes, self.agree, 100.0 * self.agree as f64 / p, self.conf_eq, 100.0 * self.conf_eq as f64 / p,
                self.conf_maxd, self.captured, 100.0 * self.captured as f64 / p, self.cap_agree,
                self.passes - self.captured, self.miss_agree, self.ns_sum as f64 / p, self.ns_max,
                self.nq_sum as f64 / p, self.trunc, self.rank_hist[0], self.rank_hist[1], self.rank_hist[2],
                self.rank_hist[3], self.rank_hist[4], self.rank_hist[5], self.rank_hist[6])
    }
}

static XSTATS: std::sync::Mutex<Option<XStats>> = std::sync::Mutex::new(None);

/// DHEADP XCHECK accumulators (penalized passes, since the last per-request report).
#[derive(Default)]
struct PStats {
    passes: u64,
    agree: u64,
    conf_eq: u64,
    in_c1: u64,     // the old argmax's block was a DHEAD candidate
    in_c2: u64,     // ... was caught by the bound expansion (or a fallback)
    cap_agree: u64,
    miss_agree: u64,
    conf_maxd: f32,
    expanded: u64,  // passes with a non-empty, under-cap expansion
    n2_sum: u64,
    n2_max: usize,
    qual_max: usize,
    fb_over: u64,
    fb_nonmono: u64,
    fb_degen: u64,
    unpen: u64,
    bound_bad: u64, // pass A's expansion list / flags != the host mirror (expansion_set)
    printed: u32,
}

impl PStats {
    fn line(&self) -> String {
        let p = self.passes.max(1) as f64;
        format!("passes {} argmax-agree {} ({:.3}%) dconf-bitwise {} ({:.3}%) max|d dconf| {:e} | old block rescored {} \
                 ({:.3}%: C1 {} + expansion {}, agree {}) missed {} (agree {}) | expanded {} (mean +{:.2} max +{} blocks, \
                 max qualifying {}) | fallbacks over-cap {} non-monotone {} degenerate {} | unpenalized-row {} | \
                 bound-mirror mismatches {}",
                self.passes, self.agree, 100.0 * self.agree as f64 / p, self.conf_eq, 100.0 * self.conf_eq as f64 / p,
                self.conf_maxd, self.in_c1 + self.in_c2, 100.0 * (self.in_c1 + self.in_c2) as f64 / p, self.in_c1,
                self.in_c2, self.cap_agree, self.passes - self.in_c1 - self.in_c2, self.miss_agree, self.expanded,
                self.n2_sum as f64 / self.expanded.max(1) as f64, self.n2_max, self.qual_max, self.fb_over,
                self.fb_nonmono, self.fb_degen, self.unpen, self.bound_bad)
    }
}

static PSTATS: std::sync::Mutex<Option<PStats>> = std::sync::Mutex::new(None);

/// RQDH XCHECK accumulators (A5-L4 real-q DHEAD passes vs the full-slice q', since the last report).
#[derive(Default)]
struct RqStats {
    passes: u64,
    exact: u64,          // same kept ids and |dq'| <= 1e-4
    pick_eq: u64,        // same sampled draft (same uniform)
    pick_eq_exact: u64,  // ... among the exact passes (must equal `exact`)
    overlap_sum: f64,    // sum over passes of sum_x min(q'_full, q'_dh)
    overlap_min: f64,
    mass_in_sum: f64,    // sum over passes of the full q' mass inside DHEAD's candidate blocks
    ns_sum: u64,
    hard: u64,           // a served id outside the candidates / served q' not summing to 1
    printed: u32,
}

impl RqStats {
    fn line(&self) -> String {
        let p = self.passes.max(1) as f64;
        format!("passes {} exact-q' {} ({:.3}%) draft-agree {} ({:.3}%; among exact {}/{}) | overlap mean {:.6} min {:.6} | \
                 full q' mass in candidates mean {:.6} | candidates mean {:.2} | hard {}",
                self.passes, self.exact, 100.0 * self.exact as f64 / p, self.pick_eq, 100.0 * self.pick_eq as f64 / p,
                self.pick_eq_exact, self.exact, self.overlap_sum / p, self.overlap_min, self.mass_in_sum / p,
                self.ns_sum as f64 / p, self.hard)
    }
}

static RQSTATS: std::sync::Mutex<Option<RqStats>> = std::sync::Mutex::new(None);

impl FwdModel {
    /// The DHEAD tail of one m = 1 draft pass i: launches only (graph-capturable — every per-round
    /// value lives in a device buffer; the pass index i is baked per pass graph exactly as the old
    /// xq_copy1_i32 / xq_rowmax_f16 launches baked it). Input: sc.x (the head's collapsed hidden,
    /// head_launches' lm_head input). Output: sc.argmax[0], sc.dconf[i], sc.d_dev[i], sc.toks[0] —
    /// the old tail's four writes.
    pub(super) fn dhead_pass(&self, l: &Launcher, sc: &mut Scratch, dh: &DHead, q: &Quad, i: usize) -> Result<()> {
        anyhow::ensure!(i < 256 && i < sc.dconf.len() && i < sc.d_dev.len(), "dhead_pass: pass {i}");
        self.dhead_screen(l, sc, dh, q)?;
        let geom: i64 = dh.geom(i);
        let smem = (dh.r * 8 * 32 * dh.hbits as usize) as u32; // [8 warps][R rows][2 * hbits x 16 B]
        xqlaunch!(l, "xq_dh_rescore", (dh.s as u32, dh.tmax as u32, 1), (256, 1, 1), smem,
                  (&q.tr, &sc.chain_xh, &q.svh, &dh.bsig, &dh.ws,
                   &mut sc.argmax, &mut sc.dconf, &mut sc.d_dev, &mut sc.toks,
                   geom, dh.n as i32, dh.delta))?;
        if dhead_xcheck() && !self.stream_capturing() {
            self.dhead_xcheck_pass(l, sc, dh, q, i)?;
        }
        Ok(())
    }

    /// A5-L4: one UNPENALIZED real-q (rq_mode 2) m = 1 draft pass i on DHEAD: the DHEAD tail (screen +
    /// exact rescore: argmax, dconf[i] = the in-candidate max), then xq_rq_draft_dh — q' over the
    /// rescored candidates, the draft sampled from it -> sc.argmax[0] / d_dev[i] / toks[0], list row i
    /// (what xq_sample_rows_rq reads). Launches only (graph-capturable, pass index baked per graph).
    pub(super) fn dhead_rq_pass(&self, l: &Launcher, sc: &mut Scratch, dh: &DHead, q: &Quad, i: usize) -> Result<()> {
        anyhow::ensure!(i <= MTP_MAX_K, "dhead_rq_pass: pass {i}");
        self.dhead_pass(l, sc, dh, q, i)?;
        xqlaunch!(l, "xq_rq_draft_dh", (1, 1, 1), (256, 1, 1), 0,
                  (&mut sc.argmax, &mut sc.d_dev, &mut sc.toks, &q.svh, &dh.ws, &mut sc.rq_list,
                   &sc.rq_dsamp, &sc.draft_meta, dh.geom(i), dh.n as i32))?;
        if dhead_xcheck() && !self.stream_capturing() {
            self.dhead_rq_xcheck_pass(l, sc, dh, q, i)?;
        }
        Ok(())
    }

    /// RQDH XCHECK (eager): the full-slice real-q pass on the same sc.x (exl3_chain -> xq_rq_draft
    /// mode 2 into PRIVATE buffers, same counter meta[3] + i — the served trajectory stays DHEAD's)
    /// vs the served list row i: exact-q' rate (same kept ids, |dq'| <= 1e-4), draft agreement (the
    /// same uniform: equal whenever the q' agree), overlap sum min(q'_full, q'_dh) (= 1 - TV) and
    /// the full q' mass on ids inside DHEAD's candidate blocks. Hard flags: a served list id
    /// outside the candidate blocks, or a served q' that does not sum to 1.
    fn dhead_rq_xcheck_pass(&self, l: &Launcher, sc: &mut Scratch, dh: &DHead, q: &Quad, i: usize) -> Result<()> {
        use crate::wp24::{RQ_K, RQ_ROW};
        self.dev.synchronize()?;
        let new_id = self.dev.dtoh_sync_copy(&sc.argmax)?[0];
        let lst = self.dev.dtoh_sync_copy(&sc.rq_list)?;
        let ws = self.dev.dtoh_sync_copy(&dh.ws)?;
        let v = dh.n;
        anyhow::ensure!(crate::wp24::draft_fits(v), "RQDH XCHECK: slice {v} outside the xq_rq_draft envelope");
        exl3_chain(l, q, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.logits, 1, &sc.chain_ws)?;
        xqlaunch!(l, "xq_rq_draft", (crate::wp24::draft_nb(v) as u32, 1, 1), (256, 1, 1), 0,
                  (&dh.xc_id, &sc.logits, v as i32, &mut sc.rq_part, &mut sc.rq_cnt,
                   &dh.xc_list, &sc.rq_dsamp, &sc.draft_meta, i as i32, 2i32))?;
        self.dev.synchronize()?;
        let full_id = self.dev.dtoh_sync_copy(&dh.xc_id)?[0];
        let fl = self.dev.dtoh_sync_copy(&dh.xc_list)?;
        let row = |b: &[u32]| -> (Vec<i32>, Vec<f32>) {
            let r = &b[i * RQ_ROW..(i + 1) * RQ_ROW];
            ((0..RQ_K).map(|j| r[j] as i32).collect(), (0..RQ_K).map(|j| f32::from_bits(r[64 + j])).collect())
        };
        let (ids_n, q_n) = row(&lst);
        let (ids_f, q_f) = row(&fl);
        let (_, _, o_dbg, _, _) = ws_layout(dh.k, dh.nblk, dh.s);
        let ns = (ws[o_dbg] as usize).min(DH_TCAP);
        let cands: Vec<usize> = ws[o_dbg + 1..o_dbg + 1 + ns].iter().map(|&c| c as usize).collect();
        let in_c = |id: i32| id >= 0 && cands.contains(&((id as usize) / 128));
        let kept = |ids: &[i32], qs: &[f32]| -> Vec<(i32, f64)> {
            (0..RQ_K).filter(|&j| qs[j] > 0.0).map(|j| (ids[j], qs[j] as f64)).collect()
        };
        let (kn, kf) = (kept(&ids_n, &q_n), kept(&ids_f, &q_f));
        let qn_of = |id: i32| kn.iter().find(|x| x.0 == id).map_or(0.0, |x| x.1);
        let overlap: f64 = kf.iter().map(|&(id, qf)| qf.min(qn_of(id))).sum();
        let mass_in: f64 = kf.iter().filter(|x| in_c(x.0)).map(|x| x.1).sum();
        let exact = kn.len() == kf.len() && kn.iter().zip(kf.iter()).all(|(a, b)| a.0 == b.0 && (a.1 - b.1).abs() <= 1e-4);
        let sum_n: f64 = kn.iter().map(|x| x.1).sum();
        let outside = kn.iter().filter(|x| !in_c(x.0)).count();
        let mut g = RQSTATS.lock().unwrap();
        let st = g.get_or_insert_with(RqStats::default);
        st.passes += 1;
        st.exact += exact as u64;
        st.pick_eq += (new_id == full_id) as u64;
        st.pick_eq_exact += (exact && new_id == full_id) as u64;
        st.overlap_sum += overlap;
        st.overlap_min = if st.passes == 1 { overlap } else { st.overlap_min.min(overlap) };
        st.mass_in_sum += mass_in;
        st.ns_sum += ns as u64;
        st.hard += (outside > 0 || (sum_n - 1.0).abs() > 1e-3) as u64;
        if (outside > 0 || (sum_n - 1.0).abs() > 1e-3 || (exact && new_id != full_id)) && st.printed < 32 {
            st.printed += 1;
            println!("RQDH_XCHECK pass {i}: HARD served draft {new_id} full {full_id} exact {exact} outside-ids {outside} \
                      sum q' {sum_n:.6} | served {:?} | full {:?}", &kn[..kn.len().min(6)], &kf[..kf.len().min(6)]);
        }
        if st.passes % 256 == 0 {
            println!("RQDH_XCHECK running: {}", st.line());
        }
        Ok(())
    }

    /// prep + screen (xh, the int8 screen x, the approximate block maxima): DHEAD's and DHEADP's
    /// first two launches.
    fn dhead_screen(&self, l: &Launcher, sc: &mut Scratch, dh: &DHead, q: &Quad) -> Result<()> {
        xqlaunch!(l, "xq_dh_prep", (1, 1, 1), ((32 * dh.k / 128) as u32, 1, 1), 0,
                  (&sc.x, &q.suh, &mut sc.chain_xh, &dh.ws, dh.k as i32, dh.bits as i32, dh.mse))?;
        if let Some(sh) = dh.shard.as_ref() {
            // TP-H #3: this rank's half of the blocks + the K1 fold, then K2 reassembles bmax
            let tp = self.tp.as_ref().context("the sharded DHEAD screen without a TP attachment")?;
            let arrive = tp.dec_arrive().context("the sharded DHEAD screen without the decode transport")?;
            let o_bmax = ws_layout(dh.k, dh.nblk, dh.s).0;
            let out = *dh.ws.device_ptr() as u64 + 4 * o_bmax as u64;
            xqlaunch_raw!(l, "xq_dh_screen_k1", (sh.grid as u32, 1, 1), (128, 1, 1), 0,
                          (tp.ctx_addr(), &dh.codes, &dh.ws, &q.svh, &sh.part, dh.k as i32, dh.bits as i32, dh.dq,
                           sh.nb_off as i32, arrive, (dh.nblk * 4) as u32))?;
            return tp.dh_bmax_k2(l, out, *sh.part.device_ptr() as u64, dh.nblk);
        }
        xqlaunch!(l, "xq_dh_screen", (dh.nblk as u32, 1, 1), (128, 1, 1), 0,
                  (&dh.codes, &dh.ws, &q.svh, dh.k as i32, dh.bits as i32, dh.dq))?;
        Ok(())
    }

    /// DHEADP: the tail of one PENALIZED m = 1 draft pass i (monotone params; launches only,
    /// graph-capturable). Input sc.x; penalty inputs = the draft mirror's (sc.pen row 0,
    /// sc.draft_meta, sc.d_dev[..i], sc.pen_hist). Output: sc.argmax[0], sc.dconf[i], sc.d_dev[i],
    /// sc.toks[0] — the old penalized tail's four writes. Four launches: prep, screen, pass A
    /// (C1 exact + penalized -> L -> the expansion list), pass B (the expansion, then the outputs).
    pub(super) fn dheadp_pass(&self, l: &Launcher, sc: &mut Scratch, dh: &DHead, q: &Quad, i: usize) -> Result<()> {
        let p = dh.p.as_ref().ok_or_else(|| anyhow::anyhow!("dheadp_pass: DHEADP not built"))?;
        anyhow::ensure!(i < 256 && i < sc.dconf.len() && i < sc.d_dev.len(), "dheadp_pass: pass {i}");
        self.dhead_screen(l, sc, dh, q)?;
        let ga: i64 = dh.geom(i) | ((PEN_RING_LOG2 as i64) << 48) | ((p.cap as i64) << 53);
        let gb: i64 = ((dh.n as u64) | ((dh.delta.to_bits() as u64) << 32)) as i64;
        let ring = dh.r * 8 * 32 * dh.hbits as usize;          // [8 warps][R rows][2 * hbits x 16 B]
        let smem_a = ring.max(2 * dh.tmax * 128 * 4) as u32;    // + pass A's C1 window counts
        xqlaunch!(l, "xq_dh_rescore_pa", (dh.s as u32, dh.tmax as u32, 1), (256, 1, 1), smem_a,
                  (&q.tr, &sc.chain_xh, &q.svh, &dh.bsig, &dh.ws, &p.pws,
                   &sc.pen, &sc.draft_meta, &sc.d_dev, &sc.pen_hist, ga, gb))?;
        xqlaunch!(l, "xq_dh_rescore_pb", (dh.s as u32, p.g as u32, 1), (256, 1, 1), ring as u32,
                  (&q.tr, &sc.chain_xh, &q.svh, &p.pws, &mut sc.d_dev, &sc.pen_hist,
                   &mut sc.argmax, &mut sc.dconf, &mut sc.toks, ga, gb))?;
        if dhead_xcheck() && !self.stream_capturing() {
            self.dheadp_xcheck_pass(l, sc, dh, q, i)?;
        }
        Ok(())
    }

    /// DHEADP XCHECK (eager): the OLD penalized head on the same sc.x (exl3_chain on the slice ->
    /// xq_pen_draft with the same params / window -> xq_argmax / xq_rowmax_f16 into private
    /// buffers; the served trajectory stays DHEADP's) vs DHEADP's id / dconf, plus the bound's
    /// bookkeeping: was the old argmax's block rescored (C1, or C2 via the expansion)?
    fn dheadp_xcheck_pass(&self, l: &Launcher, sc: &mut Scratch, dh: &DHead, q: &Quad, i: usize) -> Result<()> {
        let p = dh.p.as_ref().unwrap();
        self.dev.synchronize()?;
        let new_id = self.dev.dtoh_sync_copy(&sc.argmax)?[0];
        let new_conf = self.dev.dtoh_sync_copy(&sc.dconf)?[i];
        let ws = self.dev.dtoh_sync_copy(&dh.ws)?;
        let (_, c2_off, _, _) = pws_layout(dh.nblk, dh.s);
        let mut hdr = vec![0u32; c2_off + dh.nblk];
        self.dev.dtoh_sync_copy_into(&p.pws.slice(0..c2_off + dh.nblk), &mut hdr)?;
        let v = dh.n as i32;
        exl3_chain(l, q, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.logits, 1, &sc.chain_ws)?;
        xqlaunch!(l, "xq_pen_draft", (dh.n.div_ceil(PEN_SPAN) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.logits, v, &sc.pen, &sc.draft_meta, &sc.d_dev, i as i32,
                   &sc.pen_hist, PEN_RING_LOG2 as i32))?;
        xqlaunch!(l, "xq_argmax", (1, 1, 1), (1024, 1, 1), (1024 * 12) as u32,
                  (&dh.xc_id, 0i32, &sc.logits, 0i64, v))?;
        xqlaunch!(l, "xq_rowmax_f16", (1, 1, 1), (1024, 1, 1), 0,
                  (&dh.xc_conf, 0i64, &sc.logits, 0i64, v))?;
        self.dev.synchronize()?;
        let old_id = self.dev.dtoh_sync_copy(&dh.xc_id)?[0];
        let old_conf = self.dev.dtoh_sync_copy(&dh.xc_conf)?[0];
        let (o_bmax, _, o_dbg, _, _) = ws_layout(dh.k, dh.nblk, dh.s);
        let ns = (ws[o_dbg] as usize).min(DH_TCAP);
        let c1: Vec<usize> = ws[o_dbg + 1..o_dbg + 1 + ns].iter().map(|&c| c as usize).collect();
        let dg = &hdr[DHP_DBG..DHP_DBG + 8];
        let (n2, fl, lval, nqual) = (dg[1] as usize, dg[2], f32::from_bits(dg[3]), dg[5] as usize);
        let c2: Vec<usize> = hdr[c2_off..c2_off + n2.min(dh.nblk)].iter().map(|&c| c as usize).collect();
        // the bound's host mirror on the device's own inputs (bmax, bsig, sigma, C1, L, cap, params):
        // pass A's expansion list and flags must be exactly expansion_set's
        let pen = self.dev.dtoh_sync_copy(&sc.pen)?;
        let (pflag, rep, pres, freq) = (pen[0], f32::from_bits(pen[1]), f32::from_bits(pen[2]), f32::from_bits(pen[3]));
        let mono = !((pflag & 1 != 0 && !(rep >= 1.0)) || (pflag & 2 != 0 && !(pres >= 0.0 && freq >= 0.0)));
        let bound_ok = if pflag == 0 {
            fl == DHP_F_UNPEN && n2 == 0
        } else {
            let bm: Vec<f32> = ws[o_bmax..o_bmax + dh.nblk].iter().map(|&u| f32::from_bits(u)).collect();
            let bs = self.dev.dtoh_sync_copy(&dh.bsig)?;
            let sig = f32::from_bits(ws[dh.k / 4 + 2]);
            let mut in_c1v = vec![false; dh.nblk];
            for &c in &c1 { in_c1v[c] = true; }
            let (want, wfl) = expansion_set(&bm, &bs[..dh.nblk], &in_c1v, sig, dh.delta, lval, p.cap, mono);
            want == c2 && wfl == fl && (wfl & DHP_F_FB != 0 || nqual == want.len())
        };
        let ob = ((old_id.max(0) as usize) / 128).min(dh.nblk - 1);
        let in_c1 = c1.contains(&ob);
        let in_c2 = c2.contains(&ob);
        let agree = new_id == old_id;
        let conf_eq = new_conf.to_bits() == old_conf.to_bits();
        let mut g = PSTATS.lock().unwrap();
        let st = g.get_or_insert_with(PStats::default);
        st.passes += 1;
        st.agree += agree as u64;
        st.conf_eq += conf_eq as u64;
        st.in_c1 += in_c1 as u64;
        st.in_c2 += in_c2 as u64;
        st.cap_agree += ((in_c1 || in_c2) && agree) as u64;
        st.miss_agree += (!(in_c1 || in_c2) && agree) as u64;
        if in_c1 || in_c2 {
            let d = (new_conf - old_conf).abs();
            if d.is_nan() || d > st.conf_maxd { st.conf_maxd = if d.is_nan() { f32::INFINITY } else { d }; }
        }
        if fl & DHP_F_UNPEN != 0 {
            st.unpen += 1;
        } else if fl & DHP_F_FB != 0 {
            if fl & DHP_F_NONMONO != 0 { st.fb_nonmono += 1; }
            else if fl & DHP_F_DEGEN != 0 { st.fb_degen += 1; }
            else { st.fb_over += 1; }
        } else if n2 > 0 {
            st.expanded += 1;
            st.n2_sum += n2 as u64;
            st.n2_max = st.n2_max.max(n2);
        }
        st.qual_max = st.qual_max.max(nqual);
        st.bound_bad += (!bound_ok) as u64;
        if (!agree || ((in_c1 || in_c2) && !conf_eq) || !bound_ok) && st.printed < 32 {
            st.printed += 1;
            println!("DHEADP_XCHECK pass {i}: new id {new_id} conf {new_conf} | old id {old_id} conf {old_conf} | old block {ob} \
                      in C1 {in_c1} in C2 {in_c2} | L {lval} C1 {ns} C2 {n2} qualifying {nqual} flags {fl:#x} \
                      bound-mirror {} C2 {:?}", if bound_ok { "ok" } else { "MISMATCH" },
                     &c2[..c2.len().min(8)]);
        }
        if st.passes % 256 == 0 {
            println!("DHEADP_XCHECK running: {}", st.line());
        }
        Ok(())
    }

    /// XCHECK (eager): run the OLD head on the same sc.x (exl3_chain on the slice, then xq_argmax /
    /// xq_rowmax_f16 into private buffers — the new pass's argmax / dconf / d_dev / toks are left
    /// as the new head wrote them, so the served trajectory is the new head's) and compare.
    fn dhead_xcheck_pass(&self, l: &Launcher, sc: &mut Scratch, dh: &DHead, q: &Quad, i: usize) -> Result<()> {
        self.dev.synchronize()?;
        let new_id = self.dev.dtoh_sync_copy(&sc.argmax)?[0];
        let new_conf = self.dev.dtoh_sync_copy(&sc.dconf)?[i];
        let ws = self.dev.dtoh_sync_copy(&dh.ws)?;
        // chain_xh gets the identical xh; chain_yraw / logits are dead after a draft pass
        exl3_chain(l, q, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.logits, 1, &sc.chain_ws)?;
        let v = dh.n as i32;
        xqlaunch!(l, "xq_argmax", (1, 1, 1), (1024, 1, 1), (1024 * 12) as u32,
                  (&dh.xc_id, 0i32, &sc.logits, 0i64, v))?;
        xqlaunch!(l, "xq_rowmax_f16", (1, 1, 1), (1024, 1, 1), 0,
                  (&dh.xc_conf, 0i64, &sc.logits, 0i64, v))?;
        self.dev.synchronize()?;
        let old_id = self.dev.dtoh_sync_copy(&dh.xc_id)?[0];
        let old_conf = self.dev.dtoh_sync_copy(&dh.xc_conf)?[0];
        let (o_bmax, _, o_dbg, _, _) = ws_layout(dh.k, dh.nblk, dh.s);
        let ns = (ws[o_dbg] as usize).min(DH_TCAP);
        let cands: Vec<usize> = ws[o_dbg + 1..o_dbg + 1 + ns].iter().map(|&c| c as usize).collect();
        let nq = ws[o_dbg + DH_TCAP + 1] as usize;
        let ob = ((old_id.max(0) as usize) / 128).min(dh.nblk - 1);
        let bm = |b: usize| f32::from_bits(ws[o_bmax + b]);
        let rank = (0..dh.nblk).filter(|&b| bm(b) > bm(ob) || (bm(b) == bm(ob) && b < ob)).count();
        let captured = cands.contains(&ob);
        let agree = new_id == old_id;
        let conf_eq = new_conf.to_bits() == old_conf.to_bits();
        let mut g = XSTATS.lock().unwrap();
        let st = g.get_or_insert_with(XStats::default);
        st.passes += 1;
        st.agree += agree as u64;
        st.conf_eq += conf_eq as u64;
        st.captured += captured as u64;
        st.cap_agree += (captured && agree) as u64;
        st.miss_agree += (!captured && agree) as u64;
        if captured {
            let d = (new_conf - old_conf).abs();
            if d.is_nan() || d > st.conf_maxd { st.conf_maxd = if d.is_nan() { f32::INFINITY } else { d }; }
        }
        st.ns_sum += ns as u64;
        st.ns_max = st.ns_max.max(ns);
        st.nq_sum += nq as u64;
        st.trunc += (nq > ns) as u64;
        let bucket = match rank { 0 => 0, 1 => 1, 2..=3 => 2, 4..=7 => 3, 8..=15 => 4, 16..=31 => 5, _ => 6 };
        st.rank_hist[bucket] += 1;
        if (!agree || (captured && !conf_eq)) && st.printed < 32 {
            st.printed += 1;
            println!("DHEAD_XCHECK pass {i}: new id {new_id} conf {new_conf} | old id {old_id} conf {old_conf} | \
                      old block {ob} captured {captured} screen-rank {rank} candidates {ns} qualified {nq} {:?}",
                     &cands[..ns.min(8)]);
        }
        if st.passes % 256 == 0 {
            println!("DHEAD_XCHECK running: {}", st.line());
        }
        Ok(())
    }

    /// Per-request DHEAD lines; the EXL3 server calls this when a request finishes.
    /// (1) DHEADP: when the request's rounds routed penalized drafts to DHEADP, one line with the
    ///     device counters since the last report — penalized passes, expansions, and the FALLBACKS
    ///     (all-block rescore) per reason. Costs one sync + a 32-byte readback, only then.
    /// (2) --dhead-xcheck=1: the per-request agreement lines (unpenalized, penalized), then reset.
    pub fn dhead_request_report(&self) {
        let Some(dh) = self.dhead.as_ref() else { return };
        if let Some(p) = dh.p.as_ref() {
            if p.seen.swap(false, Ordering::Relaxed) {
                let mut now = [0u32; 8];
                match self.dev.dtoh_sync_copy_into(&p.pws.slice(DHP_STAT..DHP_STAT + 8), &mut now) {
                    Ok(()) => {
                        let mut s0 = p.stat0.lock().unwrap();
                        let d: Vec<u32> = (0..8).map(|j| now[j].wrapping_sub(s0[j])).collect();
                        *s0 = now;
                        if d[0] > 0 {
                            crate::rprintln!("DHEADP request: penalized draft passes {} | expanded {} (+{:.2} blocks/expansion) | \
                                      FALLBACKS {} (all-block rescore: over-cap {}, non-monotone {}, degenerate {}; {} blocks) | \
                                      unpenalized-row passes {}",
                                     d[0], d[1], d[2] as f64 / d[1].max(1) as f64, d[3] + d[4] + d[5], d[3], d[4], d[5],
                                     d[6], d[7]);
                        }
                    }
                    Err(e) => crate::rprintln!("DHEADP request: counter readback failed ({e})"),
                }
            }
        }
        if !dhead_xcheck() { return; }
        if let Some(st) = XSTATS.lock().unwrap().take() {
            if st.passes > 0 { crate::rprintln!("DHEAD_XCHECK request: {}", st.line()); }
        }
        if let Some(st) = PSTATS.lock().unwrap().take() {
            if st.passes > 0 { crate::rprintln!("DHEADP_XCHECK request: {}", st.line()); }
        }
        if let Some(st) = RQSTATS.lock().unwrap().take() {
            if st.passes > 0 { crate::rprintln!("RQDH_XCHECK request: {}", st.line()); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dp4a pairing in xq_dh_screen is k-for-k: for every k, the x byte xq_dh_prep writes and
    /// the code bits xq_dh_build writes land in the SAME byte lane of the SAME (group g, shift j)
    /// dp4a — and every (x byte, code slot) is used exactly once. Emulates the kernel's arithmetic
    /// on random codes/x and checks it against the plain sum over k.
    #[test]
    fn dhead_screen_packing_pairs_k_for_k() {
        let k = 2560usize;
        for &b in &[1usize, 2, 4] {
            let g_len = 32 / b;
            let cpb = 8 / b;
            let mask = ((1u32 << b) - 1) * 0x0101_0101;
            // deterministic pseudo-random x (int8) and codes (B-bit)
            let mut s = 0x9E37_79B9u32 ^ (b as u32);
            let mut rnd = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
            let xq: Vec<i8> = (0..k).map(|_| (rnd() % 255) as i32 as i8).map(|v| v.max(-127)).collect();
            let codes: Vec<u32> = (0..k).map(|_| rnd() & ((1 << b) - 1)).collect();
            // xq_dh_prep's permuted x bytes
            let mut xb = vec![0u8; k];
            let mut seen = vec![false; k];
            for kk in 0..k {
                let o = x_byte_of(kk, b);
                assert!(!seen[o], "x byte {o} written twice (b={b})");
                seen[o] = true;
                xb[o] = xq[kk] as u8;
            }
            // xq_dh_build's code words (one column): word g = the group's G codes
            let mut words = vec![0u32; k / g_len];
            let mut used = vec![0u32; k / g_len];
            for kk in 0..k {
                let (g, sh) = code_pos_of(kk, b);
                let m = ((1u32 << b) - 1) << sh;
                assert_eq!(used[g] & m, 0, "code slot reused (b={b})");
                used[g] |= m;
                words[g] |= codes[kk] << sh;
            }
            // the kernel: for word g, for j < cpb: dp4a((w >> B j) & mask, xword(g, j))
            let dp4a = |a: u32, x: u32| -> i32 {
                (0..4).map(|l| ((a >> (8 * l)) & 0xFF) as i8 as i32 * ((x >> (8 * l)) & 0xFF) as i8 as i32).sum()
            };
            let mut acc = 0i32;
            for g in 0..k / g_len {
                for j in 0..cpb {
                    let o = (g * cpb + j) * 4;
                    let xw = u32::from_le_bytes([xb[o], xb[o + 1], xb[o + 2], xb[o + 3]]);
                    acc += dp4a((words[g] >> (b * j)) & mask, xw);
                }
            }
            let want: i32 = (0..k).map(|kk| xq[kk] as i32 * codes[kk] as i32).sum();
            assert_eq!(acc, want, "b={b}");
        }
    }

    /// ws_layout mirrors dh_off_* (kernels/exl3_bench.cu) and never overlaps.
    #[test]
    fn dhead_ws_layout() {
        let (bm, cnt, dbg, part, total) = ws_layout(2560, 512, 10);
        assert_eq!(bm, ((2560 / 4 + 4) + 3) & !3);
        assert!(bm >= 2560 / 4 + 3);
        assert_eq!(cnt, bm + 512);
        assert_eq!(dbg, cnt + 4);
        assert_eq!(part, dbg + DH_TCAP + 4);
        assert_eq!(total, part + DH_TCAP * 10 * 128);
    }

    /// The fitted screen quantizers minimize (to the grid) the MSE over the mul1 codebook's marginal
    /// and report it: re-derive both from crate::exl3::mul1_codebook().
    #[test]
    fn dhead_screen_quant_matches_codebook() {
        let cb = crate::exl3::mul1_codebook();
        let w: Vec<f64> = cb.iter().map(|&h| f16::from_bits(h).to_f64()).collect();
        let mse = |b: u32, dq: f64| -> f64 {
            let l = (1u32 << b) as f64;
            w.iter().map(|&x| {
                let c = ((x / dq + l / 2.0).floor()).clamp(0.0, l - 1.0);
                let q = (c - (l - 1.0) / 2.0) * dq;
                (x - q) * (x - q)
            }).sum::<f64>() / w.len() as f64
        };
        for &b in &[1u32, 2, 4] {
            let (dq, m) = screen_quant(b).unwrap();
            let got = mse(b, dq as f64);
            assert!((got - m as f64).abs() < 0.02 * m as f64, "b={b}: mse {got} vs table {m}");
            // no spacing 3% either side does better than 1% (the table is at the optimum)
            for f in [0.97f64, 1.03] {
                assert!(mse(b, dq as f64 * f) > got * 0.99, "b={b}: dq*{f} beats the table");
            }
        }
    }

    // ---------------------------------------------------------------- DHEADP (w3) bound tests

    /// Host xq_pen_tok (kernels/exl3_bench.cu). `fused` picks the rounding of the two FMA-able
    /// steps (nvcc emits fma.rn for the rep blend; ptxas fuses the pres/freq mul+sub): the
    /// monotonicity must hold under either rounding, so the tests run both.
    #[allow(clippy::too_many_arguments)]
    fn pen_tok_host(v: f32, fm: u32, fs: u32, flag: u32, rep: f32, pres: f32, freq: f32, scale: f32,
                    fused: bool) -> f32 {
        let f = fm as f32 / scale;
        let mut v = v;
        if flag & 1 != 0 {
            let w = if v > 0.0 { v / rep } else { v * rep };
            let fr = f + 1e-30f32;
            let f1 = (1.0f32 - fr) + 1e-30f32;
            v = if fused { v.mul_add(f1, fr * w) } else { v * f1 + w * fr };
        }
        if flag & 2 != 0 {
            let x = fs as f32 / scale;
            if fused {
                v = (-freq).mul_add(x, v);
                v = (-f).mul_add(pres, v);
            } else {
                v -= freq * x;
                v -= f * pres;
            }
        }
        if v > 65504.0 { v = 65504.0 } else if v < -65504.0 { v = -65504.0 }
        f16::from_f32(v).to_f32()
    }

    fn pen_flag(rep: f32, pres: f32, freq: f32) -> u32 {
        (rep != 1.0) as u32 | (((pres != 0.0 || freq != 0.0) as u32) << 1)
    }

    /// THE bound's premise: with monotone params (rep >= 1, pres >= 0, freq >= 0) the penalized fp16
    /// logit never exceeds the unpenalized one for any v >= -65504 (v = -inf -> -65504), under both
    /// roundings. Full fp16 sweep for the served-looking params, strided sweep over a param grid.
    /// And the converse the host routing relies on: rep < 1 / pres < 0 / freq < 0 DO raise logits.
    #[test]
    fn dheadp_penalty_never_raises_a_logit() {
        let check = |v: f32, fm: u32, fs: u32, rep: f32, pres: f32, freq: f32, d: u32| {
            let flag = pen_flag(rep, pres, freq);
            let scale = if d > 0 { d as f32 } else { 1.0 };
            for fused in [false, true] {
                let y = pen_tok_host(v, fm, fs, flag, rep, pres, freq, scale, fused);
                if v == f32::NEG_INFINITY {
                    assert_eq!(y, -65504.0, "v=-inf rep {rep} pres {pres} freq {freq} fm {fm}/{d}");
                } else {
                    assert!(y <= v, "raised: v {v} -> {y} (rep {rep} pres {pres} freq {freq} fm {fm} fs {fs} D {d} fused {fused})");
                }
            }
        };
        let served = [(1.05f32, 0.0f32, 0.0f32), (1.0, 0.5, 0.3), (1.05, 0.5, 0.3), (1.000_000_1, 0.0, 0.0),
                      (1.0, 0.001, 0.0), (1.0, 0.0, 2.0)];
        for &(rep, pres, freq) in &served {
            for &(fm, fs, d) in &[(1u32, 1u32, 1024u32), (1024, 1024, 1024), (512, 5000, 1024), (1, 1, 0)] {
                for b in 0..=0xFFFFu16 {
                    let v = f16::from_bits(b).to_f32();
                    if !v.is_nan() { check(v, fm, fs, rep, pres, freq, d); }
                }
            }
        }
        let reps = [1.0f32, 1.000_000_1, 1.01, 1.05, 1.3, 3.0];
        let preses = [0.0f32, 0.001, 0.5, 2.0];
        let freqs = [0.0f32, 0.3, 2.0];
        for &rep in &reps {
            for &pres in &preses {
                for &freq in &freqs {
                    if pen_flag(rep, pres, freq) == 0 { continue; }
                    for &d in &[1024u32, 4064, 7] {
                        for &fm in &[1u32, d / 3 + 1, d - 1, d] {
                            for &fs in &[fm, fm * 3 + 7] {
                                for b in (0..=0xFFFFu32).step_by(61).chain([0x7BFF, 0xFBFF, 0x0001, 0x8001, 0x7C00, 0xFC00, 0x0000, 0x8000]) {
                                    let v = f16::from_bits(b as u16).to_f32();
                                    if !v.is_nan() { check(v, fm, fs, rep, pres, freq, d); }
                                }
                            }
                        }
                    }
                }
            }
        }
        // the non-monotone side (host-routed to the old head): each breaks the premise
        assert!(pen_tok_host(10.0, 1024, 1024, 1, 0.9, 0.0, 0.0, 1024.0, false) > 10.0);
        assert!(pen_tok_host(-10.0, 1024, 1024, 1, 0.9, 0.0, 0.0, 1024.0, false) > -10.0);
        assert!(pen_tok_host(10.0, 1024, 1024, 2, 1.0, -0.5, 0.0, 1024.0, false) > 10.0);
        assert!(pen_tok_host(10.0, 1024, 1024, 2, 1.0, 0.0, -0.3, 1024.0, false) > 10.0);
        let mono = |r, p, f| PenParams::new(r, p, f, 1024, 1024).map(|x| x.monotone());
        assert_eq!(mono(1.05, 0.0, 0.0), Some(true));
        assert_eq!(mono(1.0, 0.5, 0.3), Some(true));
        assert_eq!(mono(0.9, 0.0, 0.0), Some(false));
        assert_eq!(mono(1.05, -0.5, 0.0), Some(false));
        assert_eq!(mono(1.05, 0.0, -0.1), Some(false));
        assert_eq!(mono(1.0, 0.0, 0.0), None); // a no-op row is not penalized at all
    }

    /// (value desc by f32 compare, id asc) — xq_argmax's / xq_am_key's order.
    fn argmax_of(ids: impl Iterator<Item = usize>, v: &[f32]) -> (usize, f32) {
        let mut best = (0usize, f32::NEG_INFINITY);
        let mut first = true;
        for t in ids {
            if first || v[t] > best.1 || (v[t] == best.1 && t < best.0) { best = (t, v[t]); first = false; }
        }
        best
    }

    /// DHEADP's rule end to end on synthetic heads where the screen-noise model HOLDS (every
    /// approximate logit within delta * sigma * |svh| of the exact one): C1 = DHEAD's selection,
    /// L = the best penalized C1 logit, C2 = expansion_set(...). The argmax over C1 u C2 must equal
    /// the penalized argmax over ALL tokens — id and value — in every trial, including heavy
    /// penalties on the unpenalized leaders, near-ties (fp16 values) and the over-cap fallback.
    /// Non-vacuous: asserts that C1 alone was WRONG in some trials (the expansion was needed).
    #[test]
    fn dheadp_expansion_recovers_the_penalized_argmax() {
        let (nblk, tmax, cap, delta) = (64usize, 16usize, 32usize, 4.0f32);
        let n = nblk * 128;
        let mut s = 0x2545_F491u64;
        let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let uni = |lo: f32, hi: f32, r: &mut dyn FnMut() -> u64| lo + (hi - lo) * ((r() >> 40) as f32 / (1u64 << 24) as f32);
        let (mut needed, mut expanded, mut fallbacks, mut trials) = (0, 0, 0, 0);
        for trial in 0..3000 {
            trials += 1;
            let sig = uni(0.02, 0.4, &mut rnd);
            let bsig: Vec<f32> = (0..nblk).map(|_| uni(0.3, 2.0, &mut rnd)).collect();
            let bsmax = bsig.iter().cloned().fold(0f32, f32::max);
            // exact fp16 logits: a broad floor + a few peaks (the draft's likely continuations)
            let mut y: Vec<f32> = (0..n).map(|_| f16::from_f32(uni(-6.0, 6.0, &mut rnd)).to_f32()).collect();
            let npk = 1 + (rnd() % 6) as usize;
            let peaks: Vec<usize> = (0..npk).map(|_| (rnd() % n as u64) as usize).collect();
            let top = uni(8.0, 22.0, &mut rnd);
            for (j, &t) in peaks.iter().enumerate() {
                // near-ties on purpose: some peaks share the top value exactly
                let v = if j > 0 && rnd() % 3 == 0 { top } else { top - uni(0.0, 3.0, &mut rnd) };
                y[t] = f16::from_f32(v).to_f32();
            }
            // screen: a_j = y_j + e_j, |e_j| <= delta * sigma * |svh_j|, |svh_j| <= bsig[b]
            let mut bmax = vec![f32::NEG_INFINITY; nblk];
            for b in 0..nblk {
                for c in 0..128 {
                    let sv = bsig[b] * uni(0.2, 1.0, &mut rnd);
                    let a = y[b * 128 + c] + uni(-1.0, 1.0, &mut rnd) * delta * sig * sv;
                    if a > bmax[b] { bmax[b] = a; }
                }
            }
            // C1: DHEAD's selection (qualified, then the top tmax by (value desc, id asc))
            let gmax = bmax.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut qual: Vec<usize> = (0..nblk).filter(|&b| {
                let m = delta * sig * (bsig[b] + bsmax);
                bmax[b] + if m > 0.0 { m } else { 0.0 } >= gmax
            }).collect();
            qual.sort_by(|&a, &b| bmax[b].partial_cmp(&bmax[a]).unwrap().then(a.cmp(&b)));
            qual.truncate(tmax.min(DH_TCAP));
            let mut in_c1 = vec![false; nblk];
            for &b in &qual { in_c1[b] = true; }
            // penalties: monotone params; the window holds the leaders with high probability
            let rep = if rnd() % 4 == 0 { 1.0 } else { uni(1.0, 1.6, &mut rnd) };
            let pres = if rnd() % 3 == 0 { 0.0 } else { uni(0.0, 2.0, &mut rnd) };
            let freq = if rnd() % 3 == 0 { 0.0 } else { uni(0.0, 1.0, &mut rnd) };
            let flag = pen_flag(rep, pres, freq);
            let d = 1024u32;
            let mut pv = y.clone();
            if flag != 0 {
                let mut win: Vec<usize> = peaks.iter().copied().filter(|_| rnd() % 10 < 7).collect();
                win.extend((0..(rnd() % 200) as usize).map(|_| (rnd() % n as u64) as usize));
                win.sort_unstable();
                win.dedup();
                for &t in &win {
                    let fm = 1 + (rnd() % d as u64) as u32;
                    let fs = fm + (rnd() % 5000) as u32;
                    pv[t] = pen_tok_host(y[t], fm, fs, flag, rep, pres, freq, d as f32, trial % 2 == 0);
                }
            }
            let reference = argmax_of(0..n, &pv);
            let c1_tok = || qual.iter().flat_map(|&b| b * 128..b * 128 + 128);
            let c1_best = argmax_of(c1_tok(), &pv);
            let (c2, fl) = expansion_set(&bmax, &bsig, &in_c1, sig, delta, c1_best.1, cap, true);
            let got = argmax_of(c1_tok().chain(c2.iter().flat_map(|&b| b * 128..b * 128 + 128)), &pv);
            assert_eq!((got.0, got.1.to_bits()), (reference.0, reference.1.to_bits()),
                       "trial {trial}: DHEADP {got:?} vs full penalized argmax {reference:?} (C1 {qual:?}, C2 {c2:?}, flags {fl:#x})");
            if c1_best.0 != reference.0 { needed += 1; }
            if fl & DHP_F_FB != 0 { fallbacks += 1; } else if !c2.is_empty() { expanded += 1; }
        }
        println!("dheadp bound test: {trials} trials, expansion needed (C1 alone wrong) {needed}, \
                  expanded {expanded}, fallbacks {fallbacks}");
        assert!(needed > 0 && expanded > 0, "the synthetic heads never exercised the expansion");
    }

    /// expansion_set's flag / fallback edges: unpenalized-looking L, degenerate L, over-cap, and
    /// non-monotone params all behave as the kernel's pass A does.
    #[test]
    fn dheadp_expansion_edges() {
        let bmax = [5.0f32, 4.0, f32::NAN, -1.0, 3.0];
        let bsig = [1.0f32; 5];
        let in_c1 = [true, false, false, false, false];
        // UB = bmax + 4 * 0.1 * 1 = bmax + 0.4; L = 4.2 -> block 1 (4.4) and the NaN block qualify
        let (c2, fl) = expansion_set(&bmax, &bsig, &in_c1, 0.1, 4.0, 4.2, 32, true);
        assert_eq!((c2, fl), (vec![1, 2], 0));
        // the bound is inclusive: UB == L qualifies (a tie with a lower id could win)
        let (c2, _) = expansion_set(&bmax, &bsig, &in_c1, 0.1, 4.0, 4.4, 32, true);
        assert_eq!(c2, vec![1, 2]);
        // over the cap -> every block outside C1
        let (c2, fl) = expansion_set(&bmax, &bsig, &in_c1, 0.1, 4.0, 4.2, 1, true);
        assert_eq!((c2, fl), (vec![1, 2, 3, 4], DHP_F_OVER | DHP_F_FB));
        // degenerate L (NaN, the clamp floor) -> all
        let (c2, fl) = expansion_set(&bmax, &bsig, &in_c1, 0.1, 4.0, f32::NAN, 32, true);
        assert_eq!(c2, vec![1, 2, 3, 4]);
        assert_eq!(fl & (DHP_F_DEGEN | DHP_F_FB), DHP_F_DEGEN | DHP_F_FB);
        let (_, fl) = expansion_set(&bmax, &bsig, &in_c1, 0.1, 4.0, -65504.0, 32, true);
        assert_eq!(fl & DHP_F_DEGEN, DHP_F_DEGEN);
        // non-monotone params -> all
        let (c2, fl) = expansion_set(&bmax, &bsig, &in_c1, 0.1, 4.0, 4.2, 32, false);
        assert_eq!((c2, fl), (vec![1, 2, 3, 4], DHP_F_NONMONO | DHP_F_FB));
        // a NaN sigma margin counts as 0 (fmaxf(NaN, 0) = 0), as in the selection
        let (c2, _) = expansion_set(&bmax, &bsig, &in_c1, f32::NAN, 4.0, 4.2, 32, true);
        assert_eq!(c2, vec![2]);
    }

    /// pws_layout mirrors DHP_HDR / dhp_nblk4 (kernels/exl3_bench.cu); geomA's fields decode.
    #[test]
    fn dheadp_pws_layout_and_geom() {
        let (cntc, c2, part, total) = pws_layout(512, 10);
        assert_eq!((cntc, c2, part), (32, 32 + 512, 32 + 1024));
        assert_eq!(total, part + 512 * 10 * 128);
        assert_eq!(pws_layout(513, 10).1, 32 + 516);
        assert!(DHP_DBG + 8 <= DHP_STAT && DHP_STAT + 8 <= DHP_HDR);
        // geomA = pass_i | hbits << 8 | S << 16 | tmax << 24 | K << 32 | rlog2 << 48 | cap << 53
        let (i, hb, s, tm, k, rl, cap) = (6i64, 5i64, 10i64, 16i64, 4096i64, PEN_RING_LOG2 as i64, DHP_CAP_MAX as i64);
        let ga = i | (hb << 8) | (s << 16) | (tm << 24) | (k << 32) | (rl << 48) | (cap << 53);
        assert_eq!((ga & 0xFF, (ga >> 8) & 0xFF, (ga >> 16) & 0xFF, (ga >> 24) & 0xFF), (i, hb, s, tm));
        assert_eq!(((ga >> 32) & 0xFFFF, (ga >> 48) & 0x1F, (ga >> 53) & 0x7FF), (k, rl, cap));
        let gb = ((65536u64) | ((4.0f32.to_bits() as u64) << 32)) as i64;
        assert_eq!((gb & 0xFFFF_FFFF, f32::from_bits(((gb as u64) >> 32) as u32)), (65536, 4.0));
    }
}
