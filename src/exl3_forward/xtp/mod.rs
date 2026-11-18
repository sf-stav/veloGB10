//! TP-A — EXL3 tensor parallelism, T1 rungs 1–3 (PLAN/TP2_DESIGN_EXL3_2026-09-28.md §1, §2, §6).
//!
//!   rung 1  G-T1-0 slice oracle (`--probe-exl3-slice`): a trellis tensor sliced on 128-column /
//!           128-row boundaries (+ its suh / svh slices) reconstructs EXACTLY the matching slice of
//!           the full tensor, for every quantized tensor class of the pack; CPU (box-free) + one
//!           GPU check (the served decode + chain on sliced vs full Quads).
//!   rung 2  plumbing: `TpAttach` / `TpState` (the doorbell transport attached to an EXL3
//!           FwdModel), the ShardSpec of the routed experts (`EpShard`, load-time shape asserts).
//!   rung 3  MoE expert parallelism, everything else replicated: static deal, per-rank FP32
//!           partial (xq_moe_combine_ep), one FP32 all-reduce per MoE layer (K1/K2 mode 2), one
//!           f16 rounding. `--probe-exl3-xtp` = the TP=1 reference + Q1-style ulp floor; the
//!           TP=2 run (`--head --tp 2 --probe-exl3-xtp`) = G-T1-b rank identity + the cross-TP
//!           tolerance tier + greedy divergence index.
//!
//! File map (the former single `xtp.rs`; every item keeps its `crate::exl3_forward::xtp::` path via `pub use self::<sub>::*`;
//! this file: the imports, the vocab-parallel head methods (`impl FwdModel`, whose `pub(super)` text is pinned by source-text tests) and the module declarations):
//! - `ep.rs`: expert-parallel deal, EpShard, TrunkSplit and the slice helpers
//! - `ep_deal4.rs`: the world-4 `freq` deal table (loader, validator, tests)
//! - `tp_state.rs`: TpAttach / TpState / DecXport structs and `impl TpState` (attach, doorbells, sp/dec helpers)
//! - `tp_config.rs`: TP knobs (decode / prefill transport, vp / sp / overlap codes), world>2 refusals, rail preflight
//! - `chain_ep.rs`: `impl FwdModel` row-parallel chains and the EP combine-reduce
//! - `slice_probe.rs`: identity lines, the slice oracle (`--probe-exl3-slice`), MoE ulp perturbation
//! - `gates.rs`: XtpOpts, trace / time passes, the G-T1-c/d gates, verify_teacher
//! - `xtp_ref.rs`: `--probe-exl3-xtp` reference writer and the TP rank-identity run (`run_tp_xtp`)
//! - `spine.rs`: the TP-C graph spine audit
//! - `geometry_tests.rs`, `agree16b_tests.rs`: the former inline `#[cfg(test)]` modules, named after them.


use super::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering as AtOrd};

/// TP-H gate harness ONLY: while set, `vp_greedy` reports false so a reference decode runs the
/// replicated head (ground truth of a different class than the vocab-parallel tail). Never set on a
/// served path; the gates restore it on every exit.
static GATE_VP_OFF: AtomicBool = AtomicBool::new(false);

mod ep_deal4; // TP-4F-prep: the world-4 `freq` table (loader, validator, tests)
mod chain_ep;
mod ep;
mod gates;
mod slice_probe;
mod spine;
mod tp_config;
mod tp_state;
mod xtp_ref;
pub use self::ep::*;
pub use self::gates::*;
pub use self::slice_probe::*;
pub use self::spine::*;
pub use self::tp_config::*;
pub use self::tp_state::*;
pub use self::xtp_ref::*;

impl FwdModel {
    /// TP-H #4: the vocab-parallel greedy tail (shard GEMM -> local argmax keys -> one key exchange
    /// -> merged ids in sc.argmax). Only a greedy, unpenalized, non-ratio-rule step / verify round
    /// with <= 16 rows takes it; everything else runs the replicated head.
    pub(super) fn vp_greedy(&self, sc: &Scratch, m: usize) -> bool {
        // a logprobs step (sc.lp_live) needs full logits rows: it takes the sampled rows' tail
        self.vp_greedy_st(sc.samp_live || sc.lp_live, sc.pen_live, sc.rq_live, m)
    }

    /// `vp_greedy` for an explicit sampler state (S-B9-CF-PACK5: the packed graph FAMILIES are keyed
    /// and captured per state without mutating the packed scratch's shadows).
    pub(super) fn vp_greedy_st(&self, samp: bool, pen: bool, rq: bool, m: usize) -> bool {
        self.lm_head_vp.is_some() && (1..=16).contains(&m) && !samp && !pen && !rq
            && !GATE_VP_OFF.load(AtOrd::Relaxed) && !GATE_VP_GATHER.load(AtOrd::Relaxed)
    }

    /// TP-H2 (`--tp-vp-sampled`): the rows vp_greedy refuses (sampled / penalized / ratio-rule) take the
    /// shard GEMM + row all-gather tail (vp_gather_rows) and then the unchanged pen / argmax / sampler.
    pub(super) fn vp_gather(&self, sc: &Scratch, m: usize) -> bool {
        self.vp_gather_st(sc.samp_live || sc.lp_live, sc.pen_live, sc.rq_live, m)
    }

    /// `vp_gather` for an explicit sampler state (see `vp_greedy_st`).
    pub(super) fn vp_gather_st(&self, samp: bool, pen: bool, rq: bool, m: usize) -> bool {
        if self.lm_head_vp.is_none() || !(1..=16).contains(&m) || GATE_VP_OFF.load(AtOrd::Relaxed) { return false; }
        // the row all-gather has kernels at world 2 (TP-H2: peer rank ^ 1) and world 4 (TP-4H2: recursive
        // doubling) only — never live at another world, whatever the code says (resolve_vp_sampled resolves
        // it OFF / refuses there; this is the structural backstop)
        if self.tp.as_ref().map_or(true, |t| t.world != 2 && t.world != 4) { return false; }
        if GATE_VP_GATHER.load(AtOrd::Relaxed) { return true; }
        vp_sampled_on() && (samp || pen || rq)
    }

    /// Graph-key bits for the vp tails: bit 45 = the greedy key-exchange tail (TP-H #4), bit 46 = the
    /// TP-H2 row all-gather tail. Both are capture-time properties of the graph.
    pub(super) fn vp_key(&self, sc: &Scratch, m: usize) -> usize {
        self.vp_key_st(sc.samp_live || sc.lp_live, sc.pen_live, sc.rq_live, m)
    }

    /// `vp_key` for an explicit sampler state (see `vp_greedy_st`).
    pub(super) fn vp_key_st(&self, samp: bool, pen: bool, rq: bool, m: usize) -> usize {
        ((self.vp_greedy_st(samp, pen, rq, m) as usize) << 45) | ((self.vp_gather_st(samp, pen, rq, m) as usize) << 46)
    }

    /// TP-H2: the shard GEMM into sc.vp_rows [m][n_sh], then the row all-gather into sc.logits [m][V]
    /// (this rank's columns by K1, the peer's by K2). sc.logits rows 0..m are then the replicated
    /// head's rows bit for bit on both ranks. Launches only (graph-capturable; the epoch is derived on
    /// device, every launch argument is a per-width constant).
    pub(super) fn vp_gather_rows(&self, l: &Launcher, sc: &mut Scratch, m: usize) -> Result<()> {
        let vp = self.lm_head_vp.as_ref().context("vp_gather_rows without the lm_head shard")?;
        let tp = self.tp.as_ref().context("vp_gather_rows without a TP attachment")?;
        let d = tp.dec.as_ref().context("vp_gather_rows without the decode transport")?;
        anyhow::ensure!(tp.world == 2, "vp_gather_rows is the world-2 row gather (world 4 is vp_gather_rows_w4, TP-4H2; world {})", tp.world);
        let v = self.cfg.vocab_size;
        let n_sh = vp.n_sh as usize;
        anyhow::ensure!(n_sh * 2 == v && n_sh % 8 == 0 && v % 8 == 0 && (vp.base as usize) % 8 == 0,
                        "vp gather: shard {n_sh} / vocab {v} / base {} not 16 B aligned halves", vp.base);
        anyhow::ensure!(VP_GATHER_ROWS * n_sh * 2 + 8 <= crate::tp::TP_SLOT_BYTES,
                        "vp gather: {VP_GATHER_ROWS} rows x {n_sh} halves exceed one ring slot");
        anyhow::ensure!(m * n_sh <= sc.vp_rows.len() && m * v <= sc.logits.len(),
                        "vp gather: m {m} exceeds the scratch (vp_rows {}, logits {})", sc.vp_rows.len(), sc.logits.len());
        // the shard's fused-Hadamard logits (the TP-H #4 shard Quad; ks = 1 asserted at build)
        exl3_chain(l, &vp.q, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.vp_rows, m, &sc.chain_ws)?;
        let arrive = *d.arrive.device_ptr() as u64;
        let (own, peer) = (vp.base, (tp.rank ^ 1) * vp.n_sh);
        let ne = m.div_ceil(VP_GATHER_ROWS);
        let blocks = |nr: usize| ((nr * n_sh / 8).div_ceil(256 * 8) as u32).clamp(1, 32);
        for j in 0..ne {
            let (r0, nr) = (j * VP_GATHER_ROWS, VP_GATHER_ROWS.min(m - j * VP_GATHER_ROWS));
            xqlaunch_raw!(l, "xq_vp_gather_k1", (blocks(nr), 1, 1), (256, 1, 1), 0,
                          (tp.ctx, &sc.vp_rows, &mut sc.logits, v as i64, vp.n_sh, own, r0 as i32, nr as i32, arrive))?;
        }
        for j in 0..ne {
            let (r0, nr) = (j * VP_GATHER_ROWS, VP_GATHER_ROWS.min(m - j * VP_GATHER_ROWS));
            let lag = (ne - 1 - j) as u32;
            xqlaunch_raw!(l, "xq_vp_gather_k2", (blocks(nr), 1, 1), (256, 1, 1), 0,
                          (tp.ctx, &mut sc.logits, v as i64, vp.n_sh, peer, r0 as i32, nr as i32, lag,
                           (d.grecv as i32) << 3))?;
        }
        Ok(())
    }

    /// TP-4H2: the row all-gather tail at either world that has kernels. World 2 = `vp_gather_rows` (TP-H2,
    /// untouched: same launches, same arguments); world 4 = `vp_gather_rows_w4`. Both call sites (plain step
    /// and verify) go through here.
    pub(super) fn vp_gather_tail(&self, l: &Launcher, sc: &mut Scratch, m: usize) -> Result<()> {
        match self.tp.as_ref().map(|t| t.world) {
            Some(2) => self.vp_gather_rows(l, sc, m),
            Some(4) => self.vp_gather_rows_w4(l, sc, m),
            w => anyhow::bail!("vp gather: world {w:?} has no row-gather kernels (2 or 4)"),
        }
    }

    /// TP-4H2: the WORLD-4 row all-gather. The shard GEMM writes sc.vp_rows [m][n_sh] (n_sh = V / 4, this
    /// rank's columns [rank * n_sh, (rank + 1) * n_sh)); the launches of `tp_vpgather::gather4_plan` — per
    /// group of <= 4 rows K1(stage 0) K2(stage 0) K1(stage 1) K2(stage 1), every K2 lag 0 — then assemble
    /// all four shards into sc.logits [m][V] by recursive doubling (2 epochs per group; the epoch, the
    /// partner masks and every shard index are derived ON DEVICE from `c->epoch`, so every launch argument is
    /// a per-width constant and the sequence is graph-capturable). sc.logits rows 0..m are then the
    /// replicated head's rows bit for bit on all four ranks. Launches only; the plan (and the tests that pin
    /// it) live in src/tp_vpgather.rs.
    pub(super) fn vp_gather_rows_w4(&self, l: &Launcher, sc: &mut Scratch, m: usize) -> Result<()> {
        use crate::tp_vpgather as g;
        let vp = self.lm_head_vp.as_ref().context("vp_gather_rows_w4 without the lm_head shard")?;
        let tp = self.tp.as_ref().context("vp_gather_rows_w4 without a TP attachment")?;
        let d = tp.dec.as_ref().context("vp_gather_rows_w4 without the decode transport")?;
        anyhow::ensure!(tp.world as usize == g::WORLD && tp.rounds() == g::ROUNDS,
                        "the world-4 vocab-parallel row gather needs world {} (rounds {}); got world {} (rounds {})",
                        g::WORLD, g::ROUNDS, tp.world, tp.rounds());
        let v = self.cfg.vocab_size;
        let n_sh = vp.n_sh as usize;
        anyhow::ensure!(n_sh * g::WORLD == v && n_sh % 8 == 0 && v % 8 == 0 && (vp.base as usize) == (tp.rank as usize) * n_sh,
                        "vp gather (world 4): shard {n_sh} x 4 != vocab {v}, or not 16 B aligned, or base {} != rank {} x shard",
                        vp.base, tp.rank);
        anyhow::ensure!(g::slot_fits(g::GROUP_ROWS, n_sh, crate::tp::TP_SLOT_BYTES),
                        "vp gather (world 4): {} rows x 2 shards x {n_sh} halves exceed one ring slot", g::GROUP_ROWS);
        anyhow::ensure!((1..=g::MAX_ROWS).contains(&m) && m * n_sh <= sc.vp_rows.len() && m * v <= sc.logits.len(),
                        "vp gather (world 4): m {m} outside 1..={} or exceeds the scratch (vp_rows {}, logits {})",
                        g::MAX_ROWS, sc.vp_rows.len(), sc.logits.len());
        // the shard's fused-Hadamard logits (the TP-H #4 shard Quad; ks = 1 asserted at build)
        exl3_chain(l, &vp.q, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.vp_rows, m, &sc.chain_ws)?;
        let arrive = *d.arrive.device_ptr() as u64;
        for la in g::gather4_plan(m, n_sh) {
            match la.half {
                g::Half::K1 => xqlaunch_raw!(l, "xq_vp_gather4_k1", (la.blocks, 1, 1), (256, 1, 1), 0,
                                             (tp.ctx, &sc.vp_rows, &mut sc.logits, v as i64, vp.n_sh, la.r0 as i32,
                                              la.nr as i32, la.stage as i32, arrive))?,
                g::Half::K2 => xqlaunch_raw!(l, "xq_vp_gather4_k2", (la.blocks, 1, 1), (256, 1, 1), 0,
                                             (tp.ctx, &mut sc.logits, v as i64, vp.n_sh, la.r0 as i32, la.nr as i32,
                                              la.stage as i32, la.lag, (d.grecv as i32) << 3))?,
            }
        }
        Ok(())
    }

    pub(super) fn vp_greedy_tail(&self, l: &Launcher, sc: &mut Scratch, m: usize) -> Result<()> {
        let vp = self.lm_head_vp.as_ref().context("vp_greedy_tail without the lm_head shard")?;
        let tp = self.tp.as_ref().context("vp_greedy_tail without a TP attachment")?;
        let d = tp.dec.as_ref().context("vp_greedy_tail without the decode transport")?;
        // the shard's fused-Hadamard logits land in the first m * n_sh entries of sc.logits (the
        // full-row buffer is idle on this path)
        exl3_chain(l, &vp.q, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.logits, m, &sc.chain_ws)?;
        let nbytes = 8 * ((m + 1) & !1);
        let arrive = *d.arrive.device_ptr() as u64;
        let nb = wp09_am_nb(m);
        anyhow::ensure!(m <= sc.am_cnt.len(), "vp tail: m {m} > scratch width {}", sc.am_cnt.len());
        xqlaunch_raw!(l, "xq_argmax_rows_vp", (nb as u32, m as u32, 1), (256, 1, 1), 0,
                 (tp.ctx, &mut sc.vp_keys, &sc.logits, vp.n_sh as i64, vp.n_sh as i32, vp.base,
                  &mut sc.am_part, &mut sc.am_cnt, arrive, nbytes as u32))?;
        // TP-4C: `rounds` key exchanges (1 at world 2 = the single TP-H #4 launch). Exchange 0's K1 is the
        // fold of xq_argmax_rows_vp; later rounds republish the in-place merged keys with the plain K1
        // (byte-agnostic copy of nbytes; the keys buffer is 8-byte keys, `nbytes / 4` floats' worth) and every
        // round's xq_tp_wait_keys merges the partner's keys into sc.vp_keys (max = the total order: value
        // desc, lowest global id) and rewrites sc.argmax from them — the last round's ids are the global argmax.
        for x in crate::tp_xport::dec_exchanges(tp.rounds()) {
            if !x.folded_k1 {
                let kp = *sc.vp_keys.device_ptr() as u64;
                crate::tp_xport::k1_serial(&tp.kern, l.stream, tp.ctx, kp, nbytes / 4)?;
            }
            xqlaunch_raw!(l, "xq_tp_wait_keys", (1u32, 1, 1), (32, 1, 1), 0,
                     (tp.ctx, &mut sc.vp_keys, &mut sc.argmax, m as i32, nbytes as i32, (d.grecv as i32) << 3))?;
        }
        Ok(())
    }

    pub fn tp_rank(&self) -> Option<(i32, i32)> { self.tp.as_ref().map(|t| (t.rank, t.world)) }

    /// TP-A rung 3: turn this rank's routed experts (+ the replicated shared expert on rank 0)
    /// into the full-rank moe_out: FP32 partial -> all-reduce -> one f16 rounding.
    pub(super) fn ep_combine_reduce(&self, l: &Launcher, sc: &mut Scratch, moe: &MoeLayer, ep: &EpShard,
                                    m: usize) -> Result<()> {
        let tp = self.tp.as_ref().context("an expert-parallel MoE layer without a TP attachment")?;
        anyhow::ensure!(ep.rank == tp.rank as usize && ep.world == tp.world as usize,
            "EP shard rank {}/{} != attached TP rank {}/{}", ep.rank, ep.world, tp.rank, tp.world);
        let h = self.cfg.hidden_size;
        let topk = self.cfg.num_experts_per_tok;
        let n = m * h;
        anyhow::ensure!(n <= tp.part.len(), "EP combine: m={m} exceeds the partial buffer ({TP_MAX_ROWS} rows)");
        let part = *tp.part.device_ptr() as u64;
        // FP4L stored-bf16 gate: the channels moe_inner's xf_sg_gate wrote (moe.sg = their constants)
        let gx = self.gate_x(sc, moe);
        if let Some(d) = tp.dec_ok(n) {
            // TP-G: combine + K1 in one launch, then K2 writes moe_out (f16) directly
            let arrive = *d.arrive.device_ptr() as u64;
            xqlaunch_raw!(l, "xq_moe_combine_ep_k1l", (m as u32, 1, 1), (1024, 1, 1), 4096,
                     (tp.dec_ctx(), part, &sc.yd, &sc.ysh, &sc.ids, &sc.wts, &sc.slotmap, &moe.sg, gx,
                      topk as i32, h as i32, m as i32, (ep.shared_split || ep.rank == 0) as i32, arrive))?;
            let mo = *sc.moe_out.device_ptr() as u64;
            return tp.dec_sum_finish(l, d, mo, part, n);
        }
        xqlaunch!(l, "xq_moe_combine_ep", (m as u32, 1, 1), (1024, 1, 1), 4096,
                 (part, &sc.yd, &sc.ysh, &sc.ids, &sc.wts, &sc.slotmap, &moe.sg, gx,
                  topk as i32, h as i32, m as i32, (ep.shared_split || ep.rank == 0) as i32))?;
        tp.all_reduce_f32(l, part, n)?;
        xqlaunch!(l, "xq_cvt_f32_f16", ((n as u32).div_ceil(256), 1, 1), (256, 1, 1), 0,
                 (&mut sc.moe_out, part, n as i64))?;
        Ok(())
    }
}

#[cfg(test)]
mod geometry_tests;
/// S-B9-CF-AGREE16B (W6): the harness passes compare the peer's word. The sites need a live link, so they
/// are pinned in the source; the comparison itself (`exl3_serve::tp_agree_eq` -> `tp_agree_decide`) has its
/// own exhaustive tests.
#[cfg(test)]
mod agree16b_tests;
