//! DSV4.1 TP (leg FW): DSpark speculative decoding at TP, on leg DS's GPU round (band 90: the vocab passes, DeepSeek's
//! acceptance rule and residual, the device-side lane state machine) and leg TPC's transport.
//!
//! Per round of the lanes `0..nl` (every lane of a round drafts the same k; inactive lanes are skipped by the kernels):
//! 1. **draft** (k > 0): the block `[tok, noise x 4]` of every lane at positions wend + 1 .. wend + 5 runs through the
//!    3 DSpark stages (layers 40-42, split like the trunk: heads / o_groups / inner width / hc_fn K slice, one reduce
//!    per sublayer), the tied head on this rank's vocabulary rows, the rank-256 Markov chain (one vocab exchange per
//!    draft) and the confidence head ([`TpSpec::draft`]);
//! 2. **verify**: one TP forward call with `[tok, d1..dk]` per lane (k + 1 <= 6 rows per lane, <= 24 rows), the taps of
//!    layers 37-39 written for every row and `main_proj`'s K-split partial riding layer 39's attention reduce; the
//!    target logits of every row on this rank's vocabulary shard; leg DS's `Round::verify` (statistics, accept,
//!    residual, finish) with the records all-gathered byte-exactly over TPC's fp32 all-reduce ([`TpXchg`]);
//! 3. **main rows**: `main_x = main_norm(round(sum of the partials))` of every verify row, each stage's window keys
//!    into the lane's 134-slot ring at the row's position (rejected rows land in slots the next round overwrites
//!    before a window reads them);
//! 4. **commit** (`vbase = wend + 1`), the lane states and emitted tokens to the host, each lane rolled back to its
//!    accepted length (`TpModel::rollback`).
//! The draft length comes from DS's policy (`--dsv41-draft-len auto|0..5`), priced from the round costs measured on
//! rank 0 at run time (cached per binary + pack + TP size) and the acceptance by position; rank 0 decides and
//! broadcasts k, so every rank runs the same shapes. Greedy rows take the exact argmax (lossless: the verify rows are
//! the decode rows' batch-invariant twins), sampled rows DeepSeek's rule.

use super::comm::{Dsv41Comm, TpComm};
use super::fwd::{HeadWant, TapReq, TpLayer, TpModel};
use super::frame::MIX_STRIDE;
use super::tpk::TpfKernels;
use super::wload::RankSrc;
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::fwd::dev::{bf16, f32a, Buf, Gpu};
use crate::dsv41::fwd::glue;
use crate::dsv41::fwd::layer::{fmt_of, HC_DIM};
use crate::dsv41::fwd::receipts;
use crate::dsv41::gpu::dspark::mcache::{self, MarkovCache};
use crate::dsv41::gpu::dspark::policy::{self, Policy};
use crate::dsv41::gpu::dspark::vocab::{Round, Rows, Shard};
use crate::dsv41::gpu::dspark::xchg::VocabExchange;
use crate::dsv41::gpu::dspark::{Kv3Args, KvArgs, Lane, SetupArgs, BLOCK, IDS_ROWS, OUT_TOK, WIDE_RPL};
use crate::dsv41::gpu::{attn, mixers as km};
use crate::dsv41::weights::exl3::dense_dec::{self, Call as DdCall};
use crate::dsv41::weights::{Accum, BackendKind, DenseWeight, DevPtr, Scratch};
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaStream;

const SLOT_Q: usize = 512;
const SLOT_S: usize = 16;
const RING: usize = 134;
pub(super) const RING_SLOTS: usize = RING;
pub(super) const RING_Q_BYTES: usize = RING * SLOT_Q;
pub(super) const RING_S_BYTES: usize = RING * SLOT_S;
/// Prompt-tail rows the drafter's rings are seeded from after a prefill (the window).
pub const TAP_ROWS: usize = 128;
/// Rows a verify carries per lane at most (k + 1 with k <= 5).
pub const RPL: usize = BLOCK + 1;
/// Leg PS2: the boot calibration's timed sweeps over every verify shape (at least / at most), the time after which it stops
/// at the minimum, and the timed repetitions of each draft step.
pub const CAL_REPS_MIN: usize = 4;
pub const CAL_REPS_MAX: usize = 8;
pub const CAL_BUDGET: std::time::Duration = std::time::Duration::from_secs(15);
pub const CAL_DRAFT_REPS: usize = 4;

fn now_unix() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) }

/// The byte-exact all-gather leg DS's round needs (`VocabExchange`), over TPC's fp32 sum all-reduce: each 32-bit word
/// travels as two exact floats (its 16-bit halves) in this rank's slot of a zeroed [world] frame, so any int / NaN /
/// denormal bit pattern of the 64-B records survives (x + 0 on 0..65535 is exact).
pub struct TpXchg<'a> { comm: &'a TpComm, k: TpfKernels, cs: &'a CudaStream, frame: Buf, max_bytes: usize }

impl<'a> TpXchg<'a> {
    pub fn new(g: &Gpu, comm: &'a TpComm, cs: &'a CudaStream, max_bytes: usize) -> Result<TpXchg<'a>> {
        Ok(TpXchg { comm, k: TpfKernels::load(&g.dev)?, cs, frame: g.buf(comm.world() * 2 * max_bytes)?, max_bytes })
    }
}

impl VocabExchange for TpXchg<'_> {
    fn ranks(&self) -> usize { self.comm.world() }
    fn rank(&self) -> usize { self.comm.rank() }
    fn max_bytes(&self) -> usize { self.max_bytes }
    fn all_gather(&self, g: &Gpu, send: u64, bytes: usize, recv: u64) -> Result<()> {
        ensure!(bytes % 16 == 0 && bytes <= self.max_bytes, "vocab exchange: {bytes} B (16-B multiple, <= {})", self.max_bytes);
        let world = self.comm.world();
        if world == 1 { if send != recv { g.d2d(recv, send, bytes)?; } return Ok(()); }
        let nw = bytes / 4;
        self.k.xenc(self.cs, send, self.frame.ptr(), nw, self.comm.rank(), world)?;
        self.comm.all_reduce_raw(self.cs, self.frame.ptr(), world * 2 * nw)?;
        self.k.xdec(self.cs, self.frame.ptr(), recv, nw, world)
    }
}

/// Per-lane acceptance bookkeeping on the host (identical on every rank).
#[derive(Clone, Copy, Debug, Default)]
pub struct LaneStats { pub rounds: u64, pub drafted: u64, pub accepted: u64, pub emitted: u64 }

/// Leg P7.4: deferred round-cost observation from the previous priced round so Rank 0's exact `(wall_ms, gap_ms)` `f32` bits
/// ride the next `Op::Round` header and keep `Policy` / `RoundCost` bit-identical across every rank with zero mid-round broadcasts.
#[derive(Clone, Copy, Debug)]
pub struct PendingRoundObs {
    pub na: usize,
    pub total_c: usize,
    pub k_draft: usize,
    pub rows: usize,
    pub k_req: usize,
    pub priced_ms: Option<f64>,
    pub ncopy: usize,
    pub wall_ms: f32,
    pub gap_ms: f32,
}

/// The drafter's state and the round's buffers on one rank.
pub struct TpSpec<'a> {
    pub x: TpXchg<'a>,
    pub round: Round,
    pub sh: Shard,
    pub cap: usize,
    ring_q: Buf,
    ring_s: Buf,
    blk_q: Buf,
    blk_s: Buf,
    eids: Buf,
    bpos: Buf,
    bwend: Buf,
    blane: Buf,
    pre: Buf,
    /// [cap][5][vlocal] f32: the drafter's logits (+ the Markov bias row by row)
    pub dlog: Buf,
    cin: Buf,
    xn: Buf,
    /// [cap][5] f32 raw confidence logits
    pub conf: Buf,
    /// [cap][6][vlocal] f32: the verify rows' target logits (lane l at l * (k + 1) rows)
    pub tlog: Buf,
    taps: Buf,
    mrow: Buf,
    mpos: Buf,
    mlane: Buf,
    freq: Buf,
    /// the lanes' round states as the host last read them
    pub host: Vec<Lane>,
    pub stats: Vec<LaneStats>,
    pub policy: Policy,
    /// round timings on this rank (ms): the last draft step, the last verify
    pub t_draft: f64,
    pub t_verify: f64,
    /// leg PS2: when the last round ended (the host gap to the next round is priced by the online model)
    pub t_end: Option<std::time::Instant>,
    /// leg SE: the request penalties (None = never configured)
    pub pen: Option<super::pen::Pen>,
    /// leg SE2: per-lane top-k / top-p, the exchange its candidates travel on, and the cut arrays ([cap][RPL] f32 / i32)
    pub cutcfg: Vec<super::cut::CutCfg>,
    pub cx: TpXchg<'a>,
    pub cut_bytes: usize,
    pub cut_send: Buf,
    pub cut_recv: Buf,
    pub cut_a: Buf,
    pub cut_id: Buf,
    /// leg SQ1: the verify's test flags (0 in serving; the sq1 part's negative control sets `TEST_SWAP_RULE` for one round)
    pub test_flags: i32,
    /// leg PS3: the drafts each lane offers to the next round (`alloc::next_held`: all after a round alone or a fresh
    /// request, its last shared count + 2 after a shared round), and the head's rows of a round whose lanes verify
    /// different counts (one launch over the call's rows, then each lane's rows to its `RPL`-row slot of `tlog`)
    pub held: Vec<usize>,
    tlog_c: Buf,
    /// leg PS3: the served-path sampled-exactness probe (`--dsv41-spec-pit-log`, debug; None = off)
    pub pit: Option<super::pit::Pit>,
    /// leg PH1: the Markov bias cache of frequent tokens (None = off: `--dsv41-markov-cache 0`, speculation off, or not attached)
    pub mc: Option<MarkovCache>,
    /// leg PH1: drafted rounds since the device counters were last read
    mc_rounds: u64,
    /// leg PH1: the drafter never runs in this process (the serve loop's policy is off: `--mtp off` / `--dsv41-draft-len 0`):
    /// the verify writes no DSpark taps, `main_proj` rides no reduce, and no window keys go into the drafter's rings
    pub drafter_off: bool,
    /// leg CD1: copy drafting (`--dsv41-copy-draft on`; None = off)
    pub copy: Option<super::copy::CopyDraft>,
    /// leg CD1: the head's permission for the next round's copy proposals (the op carries it; true unless the A/B switch is on)
    pub copy_ok: bool,
    /// leg P6-1: rank-agreed 1-lane round costs `[t_0..t_5]` in device memory (`early_t[0] <= 0.0` disables early-stop)
    early_t: Buf,
    early_t_host: [f64; 6],
    /// leg P7.4: when true (TP serve), every rank holds bit-identical `Policy` / `RoundCost` state and computes `cs` locally
    pub lockstep_cs: bool,
    pub pending_obs: Option<PendingRoundObs>,
    /// leg P7COPY: lazy per-lane `WideCopyGuard` + `COPYPF1` `FrontierCkpt` state
    pub cpv: super::cpv::CpvState,
}

impl TpModel<'_> {
    /// Load the DSpark stages (layers 40-42) as this rank holds them (split like the trunk layers) and `main_proj`.
    pub fn load_stages(&mut self, src: &RankSrc) -> Result<()> {
        let cfg = self.cfg;
        ensure!(self.layers.len() == cfg.layers, "the DSpark stages are loaded already");
        ensure!(cfg.dspark_stages == 3 && cfg.dspark_block == BLOCK, "DSpark: {} stages of block {} (this build drafts 3 stages, block {BLOCK})", cfg.dspark_stages, cfg.dspark_block);
        for s in 0..cfg.dspark_stages {
            let l = cfg.layers + s;
            ensure!(src.has_layer(l), "the rank's source holds no DSpark stage {s} (layer {l})");
            let w = src.load_layer(self.f.g, l).with_context(|| format!("DSpark stage {s}"))?;
            self.weights_bytes += w.bytes;
            let tl = TpLayer::new(w, cfg, &self.rg, self.world)?;
            crate::dsv41::weights::exl3::moe::prepare_moe(&tl.w.moe.as_ref().context("DSpark stage without routed experts")?.0)?;
            crate::dsv41::weights::exl3::moe::prepare_moe(&tl.shared)?;
            self.layers.push(Some(tl));
        }
        self.main_proj = Some(self.layers[cfg.layers].as_ref().unwrap().w.lin(C::DsMainProj)?.clone());
        // leg MEM1: the stages' calls (<= 32 rows) may need more scratch than the trunk's; grow it once, before any graph
        // holds its address
        let need = super::mem::scratch_need(&self.layers, Some(&self.glob.head), cfg.layers, self.max_rows)?;
        if need > self.scr_buf.bytes {
            ensure!(self.graphs.as_ref().map_or(true, |g| g.len() == 0), "the scratch must grow ({} -> {need} B) after graphs captured its address", self.scr_buf.bytes);
            let g = self.f.g;
            g.sync()?;
            self.scr_buf = g.buf(need)?;
            self.f.scr = crate::dsv41::fwd::glue::Scr { ptr: self.scr_buf.ptr(), bytes: self.scr_buf.bytes };
        }
        Ok(())
    }

    /// A broadcast of up to 16 small exact values from rank 0 (others' inputs ignored): one 16-float all-reduce.
    pub fn bcast_small(&mut self, v: &[f32]) -> Result<Vec<f32>> {
        ensure!(v.len() <= 16, "bcast_small: {} values", v.len());
        let t = self.tiny.ptr();
        self.f.g.memset(t, 64, 0)?;
        if self.rank == 0 { self.f.g.h2d_async(t, v)?; }
        self.reduce(t, 16)?;
        let out = self.f.g.down::<f32>(t, v.len())?;
        self.comm.check("a broadcast")?;
        Ok(out)
    }

    /// One DSpark stage `s` (0..3) on the block rows of `nl` lanes: stream `xs` [rows][hc][dim] with pre-mix `pre`
    /// (stride `ps`) -> (output stream, FFN record). The trunk's TP split: hc_fn K slice riding each sublayer's reduce,
    /// q / attention / wo_a over the rank's heads and groups, wo_b and the experts as FP32 partials.
    #[allow(clippy::too_many_arguments)]
    fn stage(&mut self, sp: &TpSpec, s: usize, xs: u64, pre: u64, ps: usize, rows: usize, nl: usize) -> Result<(u64, u64)> {
        let cfg = self.cfg;
        let l = cfg.layers + s;
        let (d, hd, nh) = (cfg.hidden, cfg.head_dim, self.rg.nh);
        let (fa, fb) = (self.frames[0].ptr(), self.frames[1].ptr());
        self.mix_partial(self.layers[l].as_ref().unwrap().mix_a, xs, rows, fa + (rows * d * 4) as u64)?;
        let an = self.layers[l].as_ref().unwrap().w.param(C::AttnNorm)?.clone();
        let ax = self.f.hc_pre_norm(xs, pre, ps, &an, rows)?;
        let qr = if rows <= 32 {
            self.q_lora(l, ax, rows)?
        } else {
            let qr = self.f.b.take(rows * cfg.q_lora_rank * 2)?;
            for r0 in (0..rows).step_by(32) {
                let rn = (rows - r0).min(32);
                let qri = self.q_lora(l, ax + (r0 * d * 2) as u64, rn)?;
                self.f.g.d2d(qr + (r0 * cfg.q_lora_rank * 2) as u64, qri, rn * cfg.q_lora_rank * 2)?;
            }
            qr
        };
        let q = self.f.b.take(rows * nh * hd * 2)?;
        let wq = self.layers[l].as_ref().unwrap().wq_b.clone();
        for r0 in (0..rows).step_by(32) {
            let rn = (rows - r0).min(32);
            glue::linear(self.f.g, "tp.ds.attn.wq_b (heads)", &wq,
                         &bf16(qr + (r0 * cfg.q_lora_rank * 2) as u64, rn, cfg.q_lora_rank, cfg.q_lora_rank),
                         &{ let mut a = bf16(q + (r0 * nh * hd * 2) as u64, rn, nh * hd, nh * hd); a.cols = nh * hd; a },
                         Accum::Overwrite, &self.f.scr)?;
        }
        glue::rope_rows(self.f.g, q, nh * hd, sp.bpos.ptr(), rows, nh, hd, false, &self.f.rope.q.plain)?;
        // the block's own keys (replicated: every rank attends with them)
        let raw = self.f.b.take(rows * SLOT_Q * 2)?;
        for r0 in (0..rows).step_by(32) {
            let rn = (rows - r0).min(32);
            let lw = &self.layers[l].as_ref().unwrap().w;
            self.f.lin_rows("tp.ds.blk_wkv", lw, C::AttnWkv, None, ax + (r0 * d * 2) as u64, d, d, rn, raw + (r0 * SLOT_Q * 2) as u64, SLOT_Q, false)?;
        }
        let norm = self.layers[l].as_ref().unwrap().w.param(C::AttnKvNorm)?;
        self.f.g.launch1("d41_ds_kv_rows", (rows as u32, 1, 1), (128, 1, 1), 0,
                         &KvArgs { kv_raw: raw, norm_w: norm.ptr.0, inv_freq: sp.freq.ptr(), pos: sp.bpos.ptr(), chunk_q: sp.blk_q.ptr(), chunk_s: sp.blk_s.ptr(),
                                   rows: rows as i32, norm_fmt: fmt_of(norm)?, eps: cfg.rms_eps as f32, ..Default::default() })?;
        let lanes: Vec<attn::LaneSrc> = (0..nl).map(|ln| {
            let (rq, rs) = sp.ring(s, ln);
            attn::LaneSrc { swa_q: DevPtr(rq), swa_s: DevPtr(rs), slots: RING, lin: None, kv: None,
                            ext: Some((DevPtr(sp.blk_q.ptr() + (ln * BLOCK * SLOT_Q) as u64), DevPtr(sp.blk_s.ptr() + (ln * BLOCK * SLOT_S) as u64), BLOCK)) }
        }).collect();
        let o = self.f.b.take(rows * nh * hd * 2)?;
        let err = self.f.b.take(4)?;
        self.f.g.memset(err, 4, 0)?;
        let call = attn::AttnCall { rows, heads: nh, q: DevPtr(q), q_stride: nh * hd, sink: DevPtr(self.layers[l].as_ref().unwrap().sink), pos: DevPtr(sp.bpos.ptr()),
                                    win_end: DevPtr(sp.bwend.ptr()), lane: DevPtr(sp.blane.ptr()), sel: DevPtr(0), sel_stride: 0, sel_cap: 0, n_sel: DevPtr(0),
                                    lanes, out: DevPtr(o), out_stride: nh * hd, out_q: None, out_s: None, epi: attn::EPI_ROPE | attn::EPI_AQ,
                                    rope_freq: self.f.rope.ka.plain, err: DevPtr(err) };
        attn::launch_prefill(self.f.g.l(), &call)?;
        receipts::kernel("tp.ds.attn (ring window + block keys, heads)", "d41_attn_prefill", rows);
        for r0 in (0..rows).step_by(32) {
            let rn = (rows - r0).min(32);
            self.wo_partial(l, o + (r0 * nh * hd * 2) as u64, rn, fa + (r0 * d * 4) as u64)?;
        }
        self.reduce_act(fa, rows * (d + MIX_STRIDE))?;
        let rec_a = self.mix_finish(l, true, fa + (rows * d * 4) as u64, rows)?;
        let ao = self.f.b.take(rows * d * 2)?;
        glue::add_round(self.f.g, fa, 0, ao, rows, d)?;
        let hm = self.f.hc_post(ao, xs, rec_a, rows)?;
        self.mix_partial(self.layers[l].as_ref().unwrap().mix_f, hm, rows, fb + (rows * d * 4) as u64)?;
        let fnm = self.layers[l].as_ref().unwrap().w.param(C::FfnNorm)?.clone();
        let fin = self.f.hc_pre_norm(hm, rec_a, 24, &fnm, rows)?;
        let (rid, rw) = self.f.router(&self.layers[l].as_ref().unwrap().w, fin, rows)?;
        for r0 in (0..rows).step_by(32) {
            let rn = (rows - r0).min(32);
            self.ffn_core(l, fin + (r0 * d * 2) as u64, rn, rid + (r0 * cfg.top_k * 4) as u64, rw + (r0 * cfg.top_k * 4) as u64, fb + (r0 * d * 4) as u64)?;
        }
        self.reduce_act(fb, rows * (d + MIX_STRIDE))?;
        let rec_f = self.mix_finish(l, false, fb + (rows * d * 4) as u64, rows)?;
        let fo = self.f.b.take(rows * d * 2)?;
        glue::add_round(self.f.g, fb, 0, fo, rows, d)?;
        let xo = self.f.hc_post(fo, hm, rec_f, rows)?;
        Ok((xo, rec_f))
    }
}

impl<'a> TpSpec<'a> {
    /// The round's state for `cap` lanes on this rank (`m` holds the stages: `load_stages` first).
    pub fn new(m: &TpModel, comm: &'a TpComm, cs: &'a CudaStream, cap: usize) -> Result<TpSpec<'a>> {
        let g = m.f.g;
        let cfg = m.cfg;
        ensure!(m.layers.len() == cfg.total_layers(), "TpSpec needs the DSpark stages (TpModel::load_stages)");
        ensure!(cap >= 1 && cap <= m.lanes.len(), "{cap} spec lanes > the model's {} lanes", m.lanes.len());
        let v = comm.vocab();
        let sh = Shard { v0: v.start, vlocal: v.len() };
        let (d, r) = (cfg.hidden, cfg.dspark_markov_rank);
        let n = cap * BLOCK;
        let round = Round::new(g, cap, WIDE_RPL, sh.nblk(), comm.world())?;
        let x = TpXchg::new(g, comm, cs, round.rows_max * crate::dsv41::gpu::dspark::vocab::REC)?;
        let cut_bytes = super::cut::MAX_CUT_ROWS.min(cap * WIDE_RPL) * super::cut::C2 * 8;
        let cx = TpXchg::new(g, comm, cs, cut_bytes)?;
        let freq = g.buf(128)?;
        let tab = crate::dsv41::rope::table_for(cfg, crate::dsv41::rope::Class::Plain)?;
        g.h2d(freq.ptr(), crate::dsv41::gpu::dspark::vocab::as_bytes(&tab))?;
        let tap_rows = (cap * TAP_ROWS).max(cap * WIDE_RPL);
        let mut policy = policy::from_flags(policy::Costs::default(), comm.world())?;
        if cap > 4 && comm.world() >= 4 && crate::opts::lookup("dsv41-spec-max-rows").and_then(crate::opts::get).is_none() {
            policy.max_rows = policy::ROUND_ROWS_MAX;
            policy.max_rows_asked = policy::ROUND_ROWS_MAX;
        }
        crate::dsv41::gpu::dspark::metrics::set_rule((policy.mode == policy::Mode::Auto).then_some(policy.rule.name()));
        if policy.decides_after_draft() {
            crate::dsv41::gpu::dspark::alloc::set_mode(policy.alloc.name(), policy.max_rows_asked, policy.max_rows);
            crate::dsv41::gpu::dspark::alloc::set_stop(policy.stop.name());
            crate::dsv41::gpu::dspark::alloc::set_gate_jump(policy.gate_jump_ms);
        }
        let pit = super::pit::Pit::from_flags(g, comm.rank())?;
        let tlog_c_bytes = if pit.is_some() { cap * RPL * sh.vlocal * 4 } else { 16 };
        Ok(TpSpec {
            x, round, sh, cap,
            ring_q: g.buf(3 * cap * RING * SLOT_Q)?, ring_s: g.buf(3 * cap * RING * SLOT_S)?,
            blk_q: g.buf(n * SLOT_Q)?, blk_s: g.buf(n * SLOT_S)?,
            eids: g.buf(n * 4)?, bpos: g.buf(n * 4)?, bwend: g.buf(n * 4)?, blane: g.buf(n * 4)?, pre: g.buf(n * 16)?,
            dlog: g.buf(n * sh.vlocal * 4)?, cin: g.buf(n * (d + r) * 2)?, xn: g.buf(n * d * 2)?, conf: g.buf(n * 4)?,
            tlog: g.buf(cap * WIDE_RPL * sh.vlocal * 4)?,
            taps: g.buf(tap_rows * cfg.dspark_targets.len() * d * 2)?, mrow: g.buf(tap_rows * d * 4)?,
            mpos: g.buf(2 * tap_rows * 4)?, mlane: g.buf(16)?, freq,
            host: vec![Lane::default(); cap], stats: vec![LaneStats::default(); cap], policy, t_draft: 0.0, t_verify: 0.0, t_end: None, pen: None,
            cutcfg: vec![super::cut::CutCfg::default(); cap], cx, cut_bytes, cut_send: g.buf(cut_bytes)?, cut_recv: g.buf(comm.world() * cut_bytes)?,
            cut_a: g.buf(cap * WIDE_RPL * 4)?, cut_id: g.buf(cap * WIDE_RPL * 4)?, test_flags: 0, mc: None, mc_rounds: 0, drafter_off: false,
            copy: super::copy::from_flags()?.map(|c| super::copy::CopyDraft::new_world(cap, c, comm.world())), copy_ok: true,
            held: vec![BLOCK; cap], tlog_c: g.buf(if pit.is_some() { (cap * WIDE_RPL).max(super::fwd::HEAD_MAX) * sh.vlocal * 4 } else { 16 })?, pit,
            early_t: g.buf(6 * 8)?, early_t_host: [0.0; 6], lockstep_cs: false, pending_obs: None, cpv: super::cpv::CpvState::default(),
        })
    }

    /// Device bytes of the round's buffers, the exchange frame and the penalty counts (leg MEM1's ledger).
    pub fn dev_bytes(&self) -> u64 {
        let own: u64 = [&self.ring_q, &self.ring_s, &self.blk_q, &self.blk_s, &self.eids, &self.bpos, &self.bwend, &self.blane, &self.pre, &self.dlog, &self.cin, &self.xn,
                        &self.conf, &self.tlog, &self.taps, &self.mrow, &self.mpos, &self.mlane, &self.freq, &self.tlog_c, &self.early_t].iter().map(|b| b.bytes as u64).sum();
        let cut: u64 = [&self.cut_send, &self.cut_recv, &self.cut_a, &self.cut_id].iter().map(|b| b.bytes as u64).sum();
        own + cut + self.round.dev_bytes() + self.x.frame.bytes as u64 + self.pen.as_ref().map_or(0, |p| p.dev_bytes())
            + self.mc.as_ref().map_or(0, |c| c.dev_bytes()) + self.cpv.dev_bytes()
    }

    pub(super) fn ring_q_ptr(&self) -> u64 { self.ring_q.ptr() }
    pub(super) fn ring_s_ptr(&self) -> u64 { self.ring_s.ptr() }
    pub(super) fn mrow_ptr(&self) -> u64 { self.mrow.ptr() }

    fn ring(&self, stage: usize, lane: usize) -> (u64, u64) {
        (self.ring_q.ptr() + ((stage * self.cap + lane) * RING * SLOT_Q) as u64, self.ring_s.ptr() + ((stage * self.cap + lane) * RING * SLOT_S) as u64)
    }

    /// Where a prefill writes the prompt tail's taps (lane 0 slot; per-lane callers use [`Self::taps_lane`]).
    pub fn taps_dst(&self, m: &TpModel) -> (u64, usize) { self.taps_lane(m, 0) }

    /// Leg P7.3: dedicated 128-row DSpark taps slot for `lane` so multiple prefilled lanes hold their tails concurrently.
    pub fn taps_lane(&self, m: &TpModel, lane: usize) -> (u64, usize) {
        let ld = m.cfg.dspark_targets.len() * m.cfg.hidden;
        (self.taps.ptr() + (lane * TAP_ROWS * ld * 2) as u64, ld)
    }

    /// The taps request of a verify call of `rows` rows (every row tapped, `main_proj` riding layer 39's reduce).
    pub fn verify_taps(&self, m: &TpModel, rows: usize) -> TapReq {
        TapReq { buf: self.taps.ptr(), ld: m.cfg.dspark_targets.len() * m.cfg.hidden, row0: 0, n: rows, dst0: 0, rider: true }
    }
    /// The taps request of a prefill chunk `[c0, c1)` of a `p`-row prompt (its rows in the last `TAP_ROWS`), or None.
    pub fn prefill_taps(&self, m: &TpModel, p: usize, c0: usize, c1: usize) -> Option<TapReq> {
        let t0 = p.saturating_sub(TAP_ROWS);
        let lo = c0.max(t0);
        if lo >= c1 { return None; }
        Some(TapReq { buf: self.taps.ptr(), ld: m.cfg.dspark_targets.len() * m.cfg.hidden, row0: lo - c0, n: c1 - lo, dst0: lo - t0, rider: false })
    }

    /// Clear lane `lane`'s rings and round state (a new request).
    pub fn reset_lane(&mut self, g: &Gpu, lane: usize) -> Result<()> {
        for s in 0..3 {
            let (q, sc) = self.ring(s, lane);
            g.memset(q, RING * SLOT_Q, 0)?;
            g.memset(sc, RING * SLOT_S, 0)?;
        }
        self.host[lane] = Lane::default();
        self.stats[lane] = LaneStats::default();
        self.held[lane] = BLOCK;
        if let Some(c) = self.copy.as_mut() { c.reset_lane(lane); }
        self.cpv.reset_lane(lane);
        Ok(())
    }

    /// Each stage's window keys of `rows` main rows from the REDUCED `main_proj` sums `sums` (f32 [rows][dim]):
    /// main_x = main_norm(bf16(sums)), then grouped `main_wkv` (PD1 when `rows <= 32`) + fused 3-stage `d41_ds_kv_rows3`
    /// into the lane's ring at the row's position.
    pub fn main_rows(&mut self, m: &mut TpModel, sums: u64, rows: usize, pos: &[i32], lanes: &[i32]) -> Result<()> {
        let n_small = if rows <= 32 { rows } else { 0 };
        self.main_rows_split(m, sums, rows, n_small, pos, lanes)
    }

    fn run_main_wkv_range(&self, m: &mut TpModel, mx: u64, raw_all: u64, total_rows: usize, r0: usize, rn: usize) -> Result<()> {
        if rn == 0 { return Ok(()); }
        let d = m.cfg.hidden;
        let x = bf16(mx + (r0 * d * 2) as u64, rn, d, d);
        if m.pd1.group && (1..=32).contains(&rn) && m.f.wide_min == 0 {
            let w0 = m.layers[m.cfg.layers].as_ref().unwrap().w.lin(C::AttnWkv)?;
            let w1 = m.layers[m.cfg.layers + 1].as_ref().unwrap().w.lin(C::AttnWkv)?;
            let w2 = m.layers[m.cfg.layers + 2].as_ref().unwrap().w.lin(C::AttnWkv)?;
            if [w0, w1, w2].iter().all(|w| w.1 == BackendKind::Exl3 && dense_dec::eligible(&w.0, rn))
                && dense_dec::scratch_bytes_group(&[&w0.0, &w1.0, &w2.0], rn).map_or(false, |b| b <= m.f.scr.bytes)
            {
                let (w0c, w1c, w2c) = (w0.0.clone(), w1.0.clone(), w2.0.clone());
                let ys: [crate::dsv41::weights::Act; 3] = std::array::from_fn(|s| {
                    bf16(raw_all + ((s * total_rows + r0) * SLOT_Q * 2) as u64, rn, SLOT_Q, SLOT_Q)
                });
                let calls = [
                    DdCall::new(&w0c, &x, &ys[0], Accum::Overwrite),
                    DdCall::new(&w1c, &x, &ys[1], Accum::Overwrite),
                    DdCall::new(&w2c, &x, &ys[2], Accum::Overwrite),
                ];
                let scr = Scratch { ptr: DevPtr(m.f.scr.ptr), bytes: m.f.scr.bytes };
                dense_dec::run_group(&m.f.g.l(), &calls, scr)?;
                receipts::kernel("tp.ds.main_wkv (PD1: 3 DSpark stages grouped)", "d41x_dd_prep+d41x_dd_a*_k* (grouped)", rn);
                return Ok(());
            }
        }
        for s in 0..3 {
            let w = &m.layers[m.cfg.layers + s].as_ref().unwrap().w;
            let dst = raw_all + ((s * total_rows + r0) * SLOT_Q * 2) as u64;
            m.f.lin_rows("tp.ds.main_wkv", w, C::AttnWkv, None, mx + (r0 * d * 2) as u64, d, d, rn, dst, SLOT_Q, false)?;
        }
        Ok(())
    }

    /// `main_rows` where rows `0..n_small` use the `<= 32` decode class (chunked by 32) and rows `n_small..rows` (`== 0` or `>= 33`)
    /// use the wide class, preserving 100% bit-identity with solo `start_lane` while sharing one `d41_rmsnorm` and one `d41_ds_kv_rows3`.
    fn main_rows_split(&mut self, m: &mut TpModel, sums: u64, rows: usize, n_small: usize, pos: &[i32], lanes: &[i32]) -> Result<()> {
        if rows == 0 { return Ok(()); }
        let cfg = m.cfg;
        let (d, eps) = (cfg.hidden, cfg.rms_eps as f32);
        let mxb = m.f.b.take(rows * d * 2)?;
        glue::add_round(m.f.g, sums, 0, mxb, rows, d)?;
        let w0 = &m.layers[cfg.layers].as_ref().unwrap().w;
        let nw = w0.param(C::DsMainNorm)?;
        let mx = m.f.b.take(rows * d * 2)?;
        let a = km::RmsNormArgs { x: mxb, x_ld: d as i64, w: nw.ptr.0, y: mx, y_ld: d as i64, w_fmt: fmt_of(nw)?, dim: d as i32, rows: rows as i32, eps, inv_dim: 1.0 / d as f32 };
        m.f.g.launch1("d41_rmsnorm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        ensure!(rows * 2 * 4 <= self.mpos.bytes, "main_rows: {rows} rows exceed mpos buffer ({} B)", self.mpos.bytes);
        let mut packed = Vec::with_capacity(rows * 2);
        packed.extend_from_slice(&pos[..rows]);
        packed.extend_from_slice(&lanes[..rows]);
        m.f.g.h2d_async(self.mpos.ptr(), &packed)?;
        let pos_ptr = self.mpos.ptr();
        let lane_ptr = pos_ptr + (rows * 4) as u64;
        let raw_all = m.f.b.take(3 * rows * SLOT_Q * 2)?;
        let mut r0 = 0usize;
        while r0 < n_small {
            let rn = (n_small - r0).min(32);
            self.run_main_wkv_range(m, mx, raw_all, rows, r0, rn)?;
            r0 += rn;
        }
        if n_small < rows {
            self.run_main_wkv_range(m, mx, raw_all, rows, n_small, rows - n_small)?;
        }
        let mut norm_w = [0u64; 3];
        let mut norm_fmt = [0u32; 3];
        for s in 0..3 {
            let norm = m.layers[cfg.layers + s].as_ref().unwrap().w.param(C::AttnKvNorm)?;
            norm_w[s] = norm.ptr.0;
            norm_fmt[s] = fmt_of(norm)?;
        }
        let (rq, rs) = self.ring(0, 0);
        let k3 = Kv3Args {
            kv_raw: raw_all, norm_w, inv_freq: self.freq.ptr(), pos: pos_ptr, lane: lane_ptr,
            ring_q: rq, ring_s: rs, ring_lq: (RING * SLOT_Q) as i64, ring_ls: (RING * SLOT_S) as i64,
            rows: rows as i32, cap: self.cap as i32, norm_fmt, eps,
        };
        m.f.g.launch1("d41_ds_kv_rows3", (rows as u32, 3, 1), (128, 1, 1), 0, &k3)?;
        receipts::kernel("tp.ds.main_rows (main_proj rider -> rings)", "d41_rmsnorm+d41_ds_kv_rows3", rows);
        Ok(())
    }

    /// After lane `lane`'s prefill (the last call left the prompt's final stream; the taps of its last rows are in the
    /// taps buffer): seed the drafter's rings from the prompt tail and draw the first token with DS's sampler (a k = 0
    /// verify of the last prompt row). `temp` 0 = greedy. Returns the first token.
    pub fn start_lane(&mut self, m: &mut TpModel, lane: usize, p: usize, last_row: usize, temp: f32, seed: u64) -> Result<u32> {
        self.start_lane_ex(m, lane, p, last_row, None, temp, seed, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start_lane_ex(&mut self, m: &mut TpModel, lane: usize, p: usize, last_row: usize, saved_row: Option<(u64, u64)>, temp: f32, seed: u64, suppress_eos: bool) -> Result<u32> {
        let g = m.f.g;
        let (d, vl) = (m.cfg.hidden, self.sh.vlocal);
        ensure!(p >= 1, "an empty prompt");
        if let Some((hp, pp)) = saved_row {
            let prev = m.last;
            m.last = Some((hp, pp));
            let hr = m.head_logits(0, 1, self.tlog.ptr() + (lane * vl * 4) as u64);
            m.last = prev;
            hr?;
        } else {
            m.head_logits(last_row, 1, self.tlog.ptr() + (lane * vl * 4) as u64)?;
        }
        // the drafter's rings from the prompt tail (main_proj K-split partial, its own reduce here); none when the drafter
        // never runs (leg PH1: every rank has the same `drafter_off`, so the reduce is skipped on all of them)
        if !self.drafter_off {
            let nt = p.min(TAP_ROWS);
            let (tap_ptr, tap_ld) = self.taps_lane(m, lane);
            m.main_proj_partial(tap_ptr, tap_ld, nt, self.mrow.ptr())?;
            m.reduce(self.mrow.ptr(), nt * d)?;
            let pos: Vec<i32> = ((p - nt)..p).map(|x| x as i32).collect();
            self.main_rows(m, self.mrow.ptr(), nt, &pos, &vec![lane as i32; nt])?;
        }
        // the first token: a k = 0 verify of the last prompt row, only this lane active
        let saved: Vec<i32> = self.host.iter().map(|l| l.active).collect();
        for l in self.host.iter_mut() { l.active = 0; l.trow0_p1 = 0; }
        let tok = *m.lanes[lane].hist.last().context("lane without tokens")?;
        self.host[lane] = Lane {
            seed, inv_t: if temp > 0.0 { 1.0 / temp } else { 0.0 }, vbase: p as i32 - 1, k: 0, wend: p as i32 - 2,
            tok: tok as i32, active: 1, suppress_eos: i32::from(suppress_eos), trow0_p1: 0, ..Default::default()
        };
        self.held[lane] = BLOCK;   // leg PS3: a fresh request drafts its full depth (theirs: `openDepth`)
        let nl = lane + 1;
        self.round.set_lanes(g, &self.host[..nl])?;
        let tcut = self.target_cut(g, nl, 1, vl)?;
        self.round.verify_rpl(g, &self.x, self.sh, Rows { ptr: self.tlog.ptr(), lane_stride: vl, row_stride: vl }, Rows { ptr: self.dlog.ptr(), lane_stride: BLOCK * vl, row_stride: vl },
                              nl, 1, tcut, None, 0, 0)?;   // leg PH1: a k = 0 verify (rpl = 1), no residual pass
        self.round.commit(g, nl)?;
        let (lanes, out) = self.round.get_lanes_and_out(g, nl)?;
        self.host[lane] = lanes[lane];
        self.host[lane].suppress_eos = 0;
        self.host[lane].trow0_p1 = 0;
        for (i, a) in saved.iter().enumerate() { if i != lane { self.host[i].active = *a; } }
        self.host[lane].active = 1;
        m.comm.check("the first token")?;
        Ok(out[lane][0] as u32)
    }

    /// Leg P7.3: batched first-token draw and drafter ring seeding across `items` (`(lane, prompt_len, h_ptr, pre_ptr, temp, seed, suppress_eos)`).
    /// Shares one `lm_head` launch, one `main_proj` RDMA reduce, one `d41_ds_kv_rows3`, and one `verify_k(0)` exchange across all started lanes,
    /// while preserving 100% bit-identity with per-lane [`Self::start_lane_ex`].
    pub fn start_lanes(&mut self, m: &mut TpModel, items: &[(usize, usize, u64, u64, f32, u64, bool)]) -> Result<Vec<u32>> {
        ensure!(!items.is_empty() && items.len() <= self.cap, "start_lanes: {} lanes (cap {})", items.len(), self.cap);
        if items.len() == 1 {
            let (lane, p, hp, pp, temp, seed, sup) = items[0];
            return Ok(vec![self.start_lane_ex(m, lane, p, 0, Some((hp, pp)), temp, seed, sup)?]);
        }
        let g = m.f.g;
        let (d, hc, vl) = (m.cfg.hidden, m.cfg.hc, self.sh.vlocal);
        let b_n = items.len();
        let mark = m.f.b.off;
        let hb = m.f.b.take(b_n * hc * d * 2)?;
        let pb = m.f.b.take(b_n * 24 * 4)?;
        for (i, &(_, p, hp, pp, _, _, _)) in items.iter().enumerate() {
            ensure!(p >= 1, "an empty prompt");
            g.d2d(hb + (i * hc * d * 2) as u64, hp, hc * d * 2)?;
            g.d2d(pb + (i * 24 * 4) as u64, pp, 24 * 4)?;
        }
        let prev_last = m.last;
        m.last = Some((hb, pb));
        let hr = m.head_logits(0, b_n, self.tlog.ptr());
        m.last = prev_last;
        hr?;
        super::metrics::head_launches(1, b_n);
        m.f.b.off = mark;

        if !self.drafter_off {
            let ld = m.cfg.dspark_targets.len() * d;
            let mut small: Vec<(usize, usize)> = Vec::new();
            let mut wide: Vec<(usize, usize)> = Vec::new();
            for &(lane, p, ..) in items {
                let nt = p.min(TAP_ROWS);
                if nt <= 32 { small.push((lane, p)); } else { wide.push((lane, p)); }
            }
            let n_small: usize = small.iter().map(|&(_, p)| p.min(TAP_ROWS)).sum();
            let n_wide: usize = wide.iter().map(|&(_, p)| p.min(TAP_ROWS)).sum();
            let total_nt = n_small + n_wide;
            let mark_t = m.f.b.off;
            let taps_cat = m.f.b.take(total_nt * ld * 2)?;
            let mut pos = Vec::with_capacity(total_nt);
            let mut lns = Vec::with_capacity(total_nt);
            let mut dst_r = 0usize;
            for &(lane, p) in small.iter().chain(wide.iter()) {
                let nt = p.min(TAP_ROWS);
                let (src_p, _) = self.taps_lane(m, lane);
                g.d2d(taps_cat + (dst_r * ld * 2) as u64, src_p, nt * ld * 2)?;
                for x in (p - nt)..p { pos.push(x as i32); lns.push(lane as i32); }
                dst_r += nt;
            }
            let mut r0 = 0usize;
            while r0 < n_small {
                let rn = (n_small - r0).min(32);
                m.main_proj_partial(taps_cat + (r0 * ld * 2) as u64, ld, rn, self.mrow.ptr() + (r0 * d * 4) as u64)?;
                r0 += rn;
            }
            if n_wide > 0 {
                m.main_proj_partial(taps_cat + (n_small * ld * 2) as u64, ld, n_wide, self.mrow.ptr() + (n_small * d * 4) as u64)?;
            }
            m.reduce(self.mrow.ptr(), total_nt * d)?;
            self.main_rows_split(m, self.mrow.ptr(), total_nt, n_small, &pos, &lns)?;
            m.f.b.off = mark_t;
        }

        let saved: Vec<i32> = self.host.iter().map(|l| l.active).collect();
        for l in self.host.iter_mut() { l.active = 0; l.trow0_p1 = 0; }
        let mut nl = 0usize;
        for (i, &(lane, p, _, _, temp, seed, sup)) in items.iter().enumerate() {
            let tok = *m.lanes[lane].hist.last().context("lane without tokens")?;
            self.host[lane] = Lane {
                seed, inv_t: if temp > 0.0 { 1.0 / temp } else { 0.0 }, vbase: p as i32 - 1, k: 0, wend: p as i32 - 2,
                tok: tok as i32, active: 1, suppress_eos: i32::from(sup), trow0_p1: (i + 1) as i32, ..Default::default()
            };
            self.held[lane] = BLOCK;
            nl = nl.max(lane + 1);
        }
        self.round.set_lanes(g, &self.host[..nl])?;
        let tcut = self.target_cut(g, nl, 1, vl)?;
        self.round.verify_rpl(g, &self.x, self.sh, Rows { ptr: self.tlog.ptr(), lane_stride: vl, row_stride: vl },
                              Rows { ptr: self.dlog.ptr(), lane_stride: BLOCK * vl, row_stride: vl }, nl, 1, tcut, None, 0, 0)?;
        self.round.commit(g, nl)?;
        let (lanes, out) = self.round.get_lanes_and_out(g, nl)?;
        for (i, a) in saved.iter().enumerate() { self.host[i].active = *a; }
        let mut toks = Vec::with_capacity(b_n);
        for &(lane, ..) in items {
            self.host[lane] = lanes[lane];
            self.host[lane].active = 1;
            self.host[lane].suppress_eos = 0;
            self.host[lane].trow0_p1 = 0;
            toks.push(out[lane][0] as u32);
        }
        m.comm.check("batched first tokens")?;
        Ok(toks)
    }

    /// The drafts of lanes 0..nl (their round states are on the device): the 3 stages on the blocks, the head, the
    /// Markov chain of `k` drafts (each a vocab exchange), the confidence head. Leaves the drafts in the ids table.
    pub fn draft(&mut self, m: &mut TpModel, nl: usize, k: usize) -> Result<()> {
        // a CUDA graph per (lanes, k) when graphs are on: every per-round value is read from device memory (the lanes'
        // round state, the ids table), the outputs land in persistent buffers
        let key = super::graph::Key { segs: vec![(usize::MAX, nl)], big: vec![], head: vec![k], taps: false };
        let mut ran = false;
        if m.graphs.is_some() {
            ran = m.graph_run(key, None, &mut |mm: &mut TpModel| self.draft_body(mm, nl, k))?;
        }
        if !ran { self.draft_body(m, nl, k)?; }
        Ok(())
    }

    fn draft_body(&mut self, m: &mut TpModel, nl: usize, k: usize) -> Result<()> {
        let g = m.f.g;
        let cfg = m.cfg;
        let (d, r, vl) = (cfg.hidden, cfg.dspark_markov_rank, self.sh.vlocal);
        let rows = nl * BLOCK;
        g.launch1("d41_ds_setup", ((rows as u32).div_ceil(32), 1, 1), (32, 1, 1), 0,
                  &SetupArgs { lanes: self.round.lanes.ptr(), eids: self.eids.ptr(), bpos: self.bpos.ptr(), bwend: self.bwend.ptr(), blane: self.blane.ptr(),
                               ids: self.round.ids.ptr(), pre: self.pre.ptr(), nlanes: nl as i32, noise: cfg.dspark_noise_token as i32 })?;
        m.swap_bump();
        let mut xs = m.embed(self.eids.ptr(), rows)?;
        let (mut pre, mut ps) = (self.pre.ptr(), 4usize);
        for s in 0..3 {
            m.swap_bump();
            let (xo, rec) = m.stage(self, s, xs, pre, ps, rows, nl)?;
            xs = xo;
            pre = rec;
            ps = 24;
        }
        // forward_head: hc_pre with the last stage's FFN pre-mix (un-normed -> the confidence input), the stage-2 norm
        let ld = d + r;
        let w2 = &m.layers[cfg.layers + 2].as_ref().unwrap().w;
        let nw = w2.param(C::DsNorm)?;
        let a = km::HcPreNormArgs { x: xs, x_ld: HC_DIM as i64, pre, pre_stride: 24, w: nw.ptr.0, y_norm: self.xn.ptr(), yn_ld: d as i64,
                                    y_raw: self.cin.ptr(), yr_ld: ld as i64, w_fmt: fmt_of(nw)?, dim: d as i32, rows: rows as i32, eps: cfg.rms_eps as f32, inv_dim: 1.0 / d as f32 };
        g.launch1("d41_hc_pre_norm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        // the tied head on this rank's vocabulary rows (chunked by 32 rows when nl > 6 so dense_dec serves every call)
        for r0 in (0..rows).step_by(32) {
            let rn = (rows - r0).min(32);
            let mut ya = f32a(self.dlog.ptr() + (r0 * vl * 4) as u64, rn, vl, vl);
            ya.cols = vl;
            glue::linear(g, "tp.ds.head (vocabulary rows, fp32)", &m.glob.head,
                         &bf16(self.xn.ptr() + (r0 * d * 2) as u64, rn, d, d), &ya, Accum::Overwrite, &m.f.scr)?;
        }
        // the Markov chain: logits[l][i] += markov_head(markov_embed[ids[i]]); ids[i + 1] = a draw (exchange)
        let me: (DenseWeight, BackendKind) = w2.lin(C::DsMarkovEmbed)?.clone();
        let mh: (DenseWeight, BackendKind) = w2.lin(C::DsMarkovHead)?.clone();
        let cw: (DenseWeight, BackendKind) = w2.lin(C::DsConfidence)?.clone();
        let meb = glue::backend(me.1);
        let cw_fast = mcache::conf_weight(&cw, ld);
        let early_on = cw_fast.is_some()
            && mcache::early_stop_on()
            && self.policy.decides_after_draft()
            && self.policy.stop != policy::Stop::Scan
            && self.policy.gate_jump_ms == 0.0;
        let lookahead_m = if self.policy.stop == policy::Stop::Lookahead { policy::LOOKAHEAD_M } else { 0.0 };
        for i in 0..k {
            let out = bf16(self.cin.ptr() + ((i * ld + d) * 2) as u64, nl, r, BLOCK * ld);
            meb.gather(&g.l(), &me.0, DevPtr(self.round.ids.ptr() + (i * nl * 4) as u64), nl, &out)?;
            if let Some((cw_ptr, cw_code)) = cw_fast {
                let ca = crate::dsv41::gpu::dspark::ConfArgs {
                    w: cw_ptr, x: self.cin.ptr() + (i * ld * 2) as u64, y: self.conf.ptr() + (i * 4) as u64,
                    wcode: cw_code, k: ld as i32, ldx: (BLOCK * ld) as i32, ldy: BLOCK as i32, rows: nl as i32, pad: 0,
                };
                g.launch1("d41_ds_conf", (nl as u32, 1, 1), (32, 1, 1), 0, &ca)?;
            }
            let xa = bf16(self.cin.ptr() + ((i * ld + d) * 2) as u64, nl, r, BLOCK * ld);
            let mut ya = f32a(self.dlog.ptr() + (i * vl * 4) as u64, nl, vl, BLOCK * vl);
            ya.cols = vl;
            let ids_i = self.round.ids.ptr() + (i * nl * 4) as u64;
            match &self.mc {
                // leg PH1 + P6-1/P6-3: cached bias rows + dynamic miss writeback + 1-lane per-step confidence early-stop
                Some(mc) if nl <= mcache::MAX_LANES => {
                    let adm = mcache::StepAdmit {
                        conf: self.conf.ptr(), early_t: self.early_t.ptr(),
                        conf_offset: self.policy.conf_offset, lookahead_m,
                        step: i, k_max: k, conf_ld: BLOCK, early_on,
                    };
                    mc.step_admit(g, xa.ptr.0, xa.ld, nl, ya.ptr.0, ya.ld, ids_i, self.round.lanes.ptr(), adm)?;
                }
                _ => { glue::linear(g, "tp.ds.markov_head (vocabulary rows)", &mh, &xa, &ya, Accum::Add, &m.f.scr)?; }
            }
            self.round.draft_pick(g, &self.x, self.sh, Rows { ptr: self.dlog.ptr(), lane_stride: BLOCK * vl, row_stride: vl }, nl, i, 1, None)?;
        }
        receipts::kernel("tp.ds.markov_chain (exchange per draft)",
                         if self.mc.is_some() && nl <= mcache::MAX_LANES { "gather+d41_ds_mc_admit+d41_ds_mbias+d41_ds_vpart/vrank/xchg/vmerge" } else { "gather+markov_head+d41_ds_vpart/vrank/xchg/vmerge" }, nl);
        if cw_fast.is_some() {
            if k > 0 { receipts::count("tp.ds.confidence", cw.1, "d41_ds_conf", k * nl, crate::dsv41::weights::RowClass::of(nl.min(32))); }
        } else {
            for r0 in (0..rows).step_by(32) {
                let rn = (rows - r0).min(32);
                glue::linear(g, "tp.ds.confidence", &cw, &bf16(self.cin.ptr() + (r0 * ld * 2) as u64, rn, ld, ld),
                             &f32a(self.conf.ptr() + (r0 * 4) as u64, rn, 1, 1), Accum::Overwrite, &m.f.scr)?;
            }
        }
        Ok(())
    }

    fn one_lane_prices(&self, depth: usize) -> [f64; 6] {
        let mut out = [f64::INFINITY; 6];
        let cmax = depth.min(BLOCK).min(self.policy.max_rows.saturating_sub(1));
        let o = self.policy.overhead_ms(1);
        for c in 0..=cmax {
            let Some(t) = self.policy.model.as_ref().and_then(|m| m.price(1, c))
                .or_else(|| self.policy.costs.round_ms_d(1, c, depth).map(|x| x + o)) else { return [0.0; 6] };
            if !t.is_finite() || t <= 0.0 { return [0.0; 6]; }
            out[c] = t + if c > 0 { self.policy.gate_jump_ms } else { 0.0 };
        }
        out
    }

    fn sync_early_t_from_f32(&mut self, g: &Gpu, ts: &[f32]) -> Result<()> {
        if ts.len() < 6 { return Ok(()); }
        let mut next = [0.0f64; 6];
        for i in 0..6 {
            let v = ts[i];
            next[i] = if v.is_infinite() && v.is_sign_positive() { f64::INFINITY } else if v > 0.0 { v as f64 } else { 0.0 };
        }
        if next.iter().zip(&self.early_t_host).any(|(a, b)| a.to_bits() != b.to_bits()) {
            self.early_t_host = next;
            g.h2d_async(self.early_t.ptr(), &self.early_t_host)?;
        }
        Ok(())
    }

    /// One round of the active lanes among 0..nl with `k` drafts each (k = 0: a plain decode step through the same
    /// verify). When the policy decides after the draft (`--dsv41-spec-rule conf`, leg PS1), `k` is the draft step's depth
    /// and rank 0 picks the drafts every lane verifies from the drafter's confidences (broadcast to every rank).
    /// Returns the tokens each lane emitted (empty for an inactive lane).
    pub fn round(&mut self, m: &mut TpModel, nl: usize, k: usize) -> Result<Vec<Vec<u32>>> {
        let g = m.f.g;
        let vl = self.sh.vlocal;
        ensure!(nl >= 1 && nl <= self.cap && k <= BLOCK, "round: {nl} lanes, k {k}");
        ensure!(k == 0 || !self.drafter_off, "round: {k} drafts asked, but this process runs without the drafter (speculation off at boot)");
        let active: Vec<usize> = (0..nl).filter(|&l| self.host[l].active != 0).collect();
        ensure!(!active.is_empty(), "a round without an active lane");
        if self.copy.as_ref().map_or(false, |c| c.cfg.prefill) && !self.drafter_off && self.copy_ok && self.pit.is_none() {
            if let Some((rem, mut stretch_em)) = super::cpv::try_stretches(self, m, nl, &active)? {
                if rem.is_empty() { return Ok(stretch_em); }
                for &l in &active { if !rem.contains(&l) { self.host[l].active = 0; } }
                let saved_pf = self.copy.as_mut().map(|c| { let v = c.cfg.prefill; c.cfg.prefill = false; v });
                let rest = self.round(m, nl, k);
                if let (Some(c), Some(v)) = (self.copy.as_mut(), saved_pf) { c.cfg.prefill = v; }
                for &l in &active { if !rem.contains(&l) { self.host[l].active = 1; } }
                let rest = rest?;
                for &l in &rem { stretch_em[l] = rest[l].clone(); }
                return Ok(stretch_em);
            }
        }
        let g = m.f.g;
        super::hostprof::mark(super::hostprof::M_ENTRY);
        // legs CD1 + P7COPY: deterministic lockstep CopySpan proposer on every TP rank (zero RDMA broadcasts);
        // a lane with a proposal verifies its copied tokens instead of drafts, and all-copy rounds skip `draft()`.
        let t0 = std::time::Instant::now();
        let mut copies: Vec<Vec<u32>> = if self.copy.is_some() && !self.drafter_off && self.copy_ok { self.copy_propose(m, nl, &active) } else { vec![Vec::new(); nl] };
        if self.round.stop_ids[0] >= 0 {
            for &l in &active {
                let sup = self.host[l].suppress_eos.max(0) as usize;
                if sup > 0 && !copies[l].is_empty() {
                    if let Some(cut_i) = copies[l].iter().take(sup).position(|&t| self.round.stop_ids.contains(&(t as i32))) {
                        copies[l].truncate(cut_i);
                    }
                }
            }
        }
        let dact: Vec<usize> = active.iter().copied().filter(|&l| copies[l].is_empty()).collect();
        let ncopy = active.len() - dact.len();
        let k_all = k;
        let k = if dact.is_empty() {
            0
        } else if dact.len() == 1 && ncopy == 0 && self.policy.decides_after_draft() && self.policy.alloc == policy::Alloc::Lanes {
            k.min(self.held[dact[0]].max(1))
        } else {
            k
        };
        if ncopy > 0 && m.rank == 0 {
            use std::sync::atomic::Ordering::Relaxed as R;
            super::copy::ROUNDS.fetch_add(1, R);
            if dact.is_empty() { super::copy::ALL_COPY_ROUNDS.fetch_add(1, R); if k_all > 0 { super::copy::DRAFT_SKIPPED.fetch_add(1, R); } }
        }
        for l in 0..nl {
            self.host[l].copy = 0;
            self.host[l].trow0_p1 = 0;
            self.host[l].k = if self.host[l].active == 0 { 0 } else if !copies[l].is_empty() { copies[l].len() as i32 } else { k as i32 };
            if !copies[l].is_empty() && self.host[l].active != 0 { self.host[l].copy = 1; }
        }
        self.round.set_lanes(g, &self.host[..nl])?;
        let pend_warm: Vec<(usize, u32)> = active.iter().map(|&l| (l, self.host[l].tok as u32)).collect();
        let lockstep = self.lockstep_cs || m.world == 1;
        let want_conf = k > 0 && self.policy.decides_after_draft() && (lockstep || m.rank == 0);
        let stop_ptr = if want_conf && nl <= mcache::MAX_LANES { self.mc.as_ref().map_or(0, |mc| mc.stopped_ptr()) } else { 0 };
        let (mut ids, conf_host, stopped_raw) = if k > 0 {
            self.draft(m, nl, k)?;
            let _ = m.engram_warm_pending(&pend_warm);
            g.down3::<i32, f32, i32>(
                self.round.ids.ptr(), IDS_ROWS * nl,
                self.conf.ptr(), if want_conf { nl * BLOCK } else { 0 },
                stop_ptr, usize::from(stop_ptr != 0),
            )?
        } else if ncopy > 0 {
            let _ = m.engram_warm_pending(&pend_warm);
            (vec![0i32; IDS_ROWS * nl], Vec::new(), Vec::new())
        } else {
            let _ = m.engram_warm_pending(&pend_warm);
            (Vec::new(), Vec::new(), Vec::new())
        };
        let stopped_from: Option<usize> = stopped_raw.first().copied().filter(|&s| s > 0).map(|s| (s - 1) as usize);
        if ncopy > 0 {
            let max_copy = copies.iter().map(|v| v.len()).max().unwrap_or(0);
            let need_ids = (max_copy + 1).max(IDS_ROWS) * nl;
            if ids.len() < need_ids { ids.resize(need_ids, 0); }
            for &l in &active {
                ids[l] = self.host[l].tok;
                for (j, &t) in copies[l].iter().enumerate() { ids[(j + 1) * nl + l] = t as i32; }
            }
            g.h2d_async(self.round.ids.ptr(), &ids)?;
        }
        // leg PS1 / PS3 / P7.4: in lockstep mode (`TpServe`), every rank holds bit-identical `Policy` / `RoundCost` state and
        // computes `cs` locally from the `conf_host` + `stopped_raw` already downloaded in `down3` (zero mid-round `bcast_small`).
        let k_draft = k;
        let _ = k_all;
        let na_all = active.len();
        // the draft-count rule prices and cuts the DRAFTING lanes only (a copy lane verifies its whole proposal, unpriced:
        // the baseline recipe's lone-copy rule)
        let (act_all, active) = (active, dact.clone());
        let na = active.len();
        let mut choice: Option<policy::Choice> = None;
        let mut lchoice: Option<policy::LaneChoice> = None;
        let mut expected = 0.0f64;
        let per_lane = k > 0 && self.policy.decides_after_draft() && self.policy.alloc == policy::Alloc::Lanes;
        let mut cs: Vec<usize> = vec![k; na];
        let mut need_lanes_upload = false;
        if k > 0 && self.policy.decides_after_draft() {
            if per_lane && na >= 2 {
                if lockstep {
                    let zl: Vec<Vec<f32>> = active.iter().map(|&l| conf_host[l * BLOCK..l * BLOCK + k].to_vec()).collect();
                    let held: Vec<usize> = active.iter().map(|&l| self.held[l]).collect();
                    lchoice = self.policy.choose_lanes(&zl, k, &held);
                    if lchoice.is_none() && m.rank == 0 { crate::dsv41::gpu::dspark::alloc::NO_PRICE.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                    cs = lchoice.as_ref().map_or_else(|| vec![k; na], |c| c.c.iter().map(|&x| x.min(k)).collect());
                } else {
                    let mine: Vec<f32> = if m.rank == 0 {
                        let zl: Vec<Vec<f32>> = active.iter().map(|&l| conf_host[l * BLOCK..l * BLOCK + k].to_vec()).collect();
                        let held: Vec<usize> = active.iter().map(|&l| self.held[l]).collect();
                        lchoice = self.policy.choose_lanes(&zl, k, &held);
                        if lchoice.is_none() { crate::dsv41::gpu::dspark::alloc::NO_PRICE.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                        lchoice.as_ref().map_or_else(|| vec![k as f32; na], |c| c.c.iter().map(|&x| x as f32).collect())
                    } else { vec![0.0; na] };
                    cs = m.bcast_small(&mine)?.iter().map(|&v| (v as usize).min(k)).collect();
                }
                expected = lchoice.as_ref().map_or(0.0, |c| c.expected);
            } else {
                // a lane alone (or `shared`): PS1's one c; in `lanes` mode a lane alone right after a shared round offers
                // its last count + 2 (theirs: the drafts held for its next round)
                let mut depth = if per_lane { k.min(self.held[active[0]]) } else { k };
                if let Some(s0) = stopped_from {
                    depth = depth.min(s0);
                    if m.rank == 0 { mcache::EARLY_SKIP_STEPS.fetch_add(k.saturating_sub(s0) as u64, std::sync::atomic::Ordering::Relaxed); }
                }
                if lockstep {
                    let zl: Vec<Vec<f32>> = active.iter().map(|&l| conf_host[l * BLOCK..l * BLOCK + depth].to_vec()).collect();
                    choice = self.policy.choose_conf(&zl, depth);
                    let c = choice.as_ref().map_or(depth, |c| c.c).min(depth);
                    if na == 1 && ncopy == 0 {
                        let et = self.one_lane_prices(BLOCK);
                        let et_f32 = [et[0] as f32, et[1] as f32, et[2] as f32, et[3] as f32, et[4] as f32, et[5] as f32];
                        self.sync_early_t_from_f32(g, &et_f32)?;
                    }
                    cs = vec![c; na];
                } else {
                    let send: [f32; 7] = if m.rank == 0 {
                        let zl: Vec<Vec<f32>> = active.iter().map(|&l| conf_host[l * BLOCK..l * BLOCK + depth].to_vec()).collect();
                        choice = self.policy.choose_conf(&zl, depth);
                        let mine = choice.as_ref().map_or(depth, |c| c.c);
                        let et = if na == 1 && ncopy == 0 { self.one_lane_prices(BLOCK) } else { self.early_t_host };
                        [mine as f32, et[0] as f32, et[1] as f32, et[2] as f32, et[3] as f32, et[4] as f32, et[5] as f32]
                    } else { [0.0; 7] };
                    let got = m.bcast_small(&send)?;
                    let c = (got[0] as usize).min(depth);
                    if na == 1 && ncopy == 0 { self.sync_early_t_from_f32(g, &got[1..7])?; }
                    cs = vec![c; na];
                }
                expected = choice.as_ref().map_or(0.0, |c| c.expected);
            }
            if cs.iter().any(|&c| c != k) {
                for (i, &l) in active.iter().enumerate() { self.host[l].k = cs[i] as i32; }
                need_lanes_upload = true;
            }
        } else if k > 0 && self.policy.mode == policy::Mode::Auto {
            expected = na as f64 * policy::expected_tokens(&self.policy.hz, k);
        }
        // back to every active lane: a copy lane verifies its whole proposal; guard total verify rows <= HEAD_MAX
        let cs_d = cs;
        let (active, na) = (act_all, na_all);
        let mut cs: Vec<usize> = active.iter().map(|&l| if copies[l].is_empty() { cs_d[dact.iter().position(|&x| x == l).unwrap()] } else { copies[l].len() }).collect();
        if active.len() > 1 {
            let mut total_rows: usize = active.len() + cs.iter().sum::<usize>();
            let mut trimmed = false;
            while total_rows > super::fwd::HEAD_MAX {
                let Some((idx, _)) = cs.iter().enumerate().filter(|&(_, &c)| c > 0).max_by_key(|&(i, &c)| (c, std::cmp::Reverse(i))) else { break };
                cs[idx] -= 1;
                total_rows -= 1;
                trimmed = true;
            }
            if trimmed {
                for (i, &l) in active.iter().enumerate() { self.host[l].k = cs[i] as i32; }
                need_lanes_upload = true;
            }
        }
        let kres = active.iter().zip(&cs).filter(|(&l, _)| copies[l].is_empty()).map(|(_, &c)| c).max().unwrap_or(0);   // the residual pass: only a drafting lane needs it
        let kmax = cs.iter().copied().max().unwrap_or(0);
        let even = cs.iter().all(|&c| c == kmax);
        // leg P7.4: zero-copy row-indexed target logits whenever lanes are uneven or non-contiguous (eliminates 4x d2d head_scatter)
        let packed_tlog = self.pit.is_none() && !(even && active.iter().copied().eq(0..active.len()));
        if packed_tlog {
            let mut r0 = 0usize;
            for (i, &l) in active.iter().enumerate() {
                self.host[l].trow0_p1 = (r0 + 1) as i32;
                r0 += cs[i] + 1;
            }
        }
        if need_lanes_upload || packed_tlog {
            self.round.set_lanes(g, &self.host[..nl])?;
        }
        let t1 = std::time::Instant::now();
        super::hostprof::mark(super::hostprof::M_DRAFTED);
        // the verify call: [tok, d1..d_c] per active lane (c = that lane's count)
        let toks: Vec<Vec<u32>> = active.iter().enumerate().map(|(i, &l)| {
            let mut v = vec![self.host[l].tok as u32];
            for j in 1..=cs[i] { v.push(ids[j * nl + l] as u32); }
            v
        }).collect();
        if kmax > 0 {
            let blocks: Vec<(usize, u32, &[u32])> = active.iter().zip(&toks).map(|(&l, t)| (l, t[0], &t[1..])).collect();
            let _ = m.engram_warm_drafts(&blocks);
        }
        for &l in &active {
            ensure!(m.lanes[l].pos == self.host[l].vbase as usize, "lane {l}: model position {} != the round's verify base {}", m.lanes[l].pos, self.host[l].vbase);
        }
        let rows: usize = toks.iter().map(|t| t.len()).sum();
        let parts: Vec<(usize, &[u32])> = active.iter().zip(&toks).map(|(&l, t)| (l, t.as_slice())).collect();
        let wide_lanes = self.cpv.save_wide_verify(m, self.ring_q.ptr(), self.ring_s.ptr(), self.cap, &parts)?;
        m.taps = if self.drafter_off { None } else { Some(self.verify_taps(m, rows)) };
        let r = m.forward_segs(&parts, HeadWant::None, None);
        m.taps = None;
        r?;
        let rpl_slot = if kmax + 1 <= RPL { RPL } else { WIDE_RPL };
        let rpl_v = if even { kmax + 1 } else { rpl_slot };
        let lane_stride = if packed_tlog { vl } else { rpl_v * vl };
        if packed_tlog {
            m.head_logits(0, rows, self.tlog.ptr())?;
            super::metrics::head_launches(1, m.segs.len());
        } else if even {
            m.head_logits_lanes(kmax + 1, self.tlog.ptr())?;
        } else {
            self.head_scatter_rpl(m, rows, rpl_slot)?;
        }
        if let Some(p) = self.pen.as_mut() {
            if p.any() {
                for (i, &l) in active.iter().enumerate() {
                    let row_ptr = if packed_tlog {
                        self.tlog.ptr() + ((self.host[l].trow0_p1 as usize - 1) * vl * 4) as u64
                    } else {
                        self.tlog.ptr() + (l * lane_stride * 4) as u64
                    };
                    p.apply(g, self.x.cs, l, row_ptr, cs[i], &toks[i][1..])?;
                }
            }
        }
        let tcut = self.target_cut(g, nl, rpl_v, lane_stride)?;
        // leg PH1: at kmax = 0 (no lane verifies a draft this round) the residual pass is skipped. Merge INT12: with PS3's
        // per-lane counts the skip keys on the round's LARGEST count, so a round where any lane verifies a draft runs the
        // residual pass for every lane, exactly as `verify` did (each lane's own count is in its `Lane::k`)
        self.round.verify_rpl(g, &self.x, self.sh, Rows { ptr: self.tlog.ptr(), lane_stride, row_stride: vl },
                              Rows { ptr: self.dlog.ptr(), lane_stride: BLOCK * vl, row_stride: vl }, nl, rpl_v, tcut, None, self.test_flags, kres)?;
        // the drafter's keys of every verify row (the main_proj sums rode layer 39's attention reduce: frame A's extra);
        // nothing reads them when the drafter never runs (leg PH1)
        if !self.drafter_off {
            let (mut pos, mut lns) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
            for sg in &m.segs { for i in 0..sg.n { pos.push((sg.p0 + i) as i32); lns.push(sg.lane as i32); } }
            let sums = m.frames[0].ptr() + (rows * (m.cfg.hidden + MIX_STRIDE) * 4) as u64;
            self.main_rows(m, sums, rows, &pos, &lns)?;
        }
        self.round.commit(g, nl)?;
        super::hostprof::mark(super::hostprof::M_ENQUEUED);
        let (lanes, out) = self.round.get_lanes_and_out(g, nl)?;
        let t2 = std::time::Instant::now();
        super::hostprof::mark(super::hostprof::M_SYNCED);
        m.comm.check("a spec round")?;
        if self.pit.is_some() {
            // leg PS3: every sampled token against its own verify row (debug; after the round's timing)
            let toks = super::pit::tokens(&active, &self.host, &lanes, &out);
            let z = if m.rank == 0 && k > 0 {
                if !conf_host.is_empty() { conf_host } else { g.down::<f32>(self.conf.ptr(), nl * BLOCK)? }
            } else { Vec::new() };
            // leg PS4: per lane, whether the inexact scan would have chosen another count (computed from the same confidences,
            // nothing served changes) and whether the exact decision was one draft from flipping (rank 0 decided: None elsewhere)
            let (diff, sens): (Option<Vec<bool>>, Option<bool>) = match (&choice, &lchoice) {
                _ if ncopy > 0 => (None, None),
                (Some(c), _) => (Some(vec![c.scan_differs; na]), Some(c.sensitive)),
                (None, Some(c)) => (Some(c.scan_c.iter().zip(&c.exact_c).map(|(a, b)| a != b).collect()), Some(c.sensitive)),
                _ => (None, None),
            };
            let ctx: Vec<super::pit::Ctx> = active.iter().enumerate()
                .map(|(i, &l)| super::pit::Ctx { lane: l, c: cs[i], n_emit: lanes[l].n_emit.max(0) as usize,
                                                  z: if z.is_empty() { Vec::new() } else { z[l * BLOCK..l * BLOCK + k].to_vec() },
                                                  scan_diff: diff.as_ref().map(|d| d[i]), sensitive: sens }).collect();
            let mut pit = self.pit.take().unwrap();
            let r = pit.record(m, &self.round, self.tlog.ptr(), lane_stride, vl, self.sh.v0, tcut, &toks, &ctx);
            self.pit = Some(pit);
            r?;
        }
        let mut em = vec![Vec::new(); nl];
        let mut emitted = 0u64;
        for (i, &l) in active.iter().enumerate() {
            let ln = lanes[l];
            let c = cs[i];
            let n = ln.n_emit.max(0) as usize;
            ensure!(n >= 1 && n <= c + 1 && n <= OUT_TOK, "lane {l}: {n} tokens emitted by a round verifying {c} drafts");
            em[l] = out[l][..n].iter().map(|&t| t as u32).collect();
            emitted += n as u64;
            if wide_lanes.contains(&l) {
                self.cpv.rollback_wide_verify(m, self.ring_q.ptr(), self.ring_s.ptr(), self.cap, l, ln.wend as usize + 1)?;
            } else {
                m.rollback(l, ln.wend as usize + 1)?;
            }
            let st = &mut self.stats[l];
            st.rounds += 1; st.drafted += c as u64; st.accepted += ln.n_acc.max(0) as u64; st.emitted += n as u64;
            if copies[l].is_empty() {
                if k > 0 { self.policy.record(c, ln.n_acc.max(0) as usize); }
                // leg PS3: what this lane offers to its next round
                if per_lane { self.held[l] = crate::dsv41::gpu::dspark::alloc::next_held(BLOCK, dact.len(), c); }
            } else {
                self.held[l] = BLOCK;
            }
        }
        crate::dsv41::gpu::dspark::metrics::record_lanes(&lanes);
        if m.rank == 0 && k_draft > 0 { self.markov_receipts(g, &ids, nl, &dact, k_draft, stopped_from)?; }
        self.host[..nl].copy_from_slice(&lanes);
        for l in 0..nl { self.host[l].suppress_eos = 0; self.host[l].trow0_p1 = 0; }
        self.t_draft = (t1 - t0).as_secs_f64() * 1e3;
        self.t_verify = (t2 - t1).as_secs_f64() * 1e3;
        if let Some(cd) = self.copy.as_mut() {
            if !self.drafter_off {
                cd.finish_round(m.world, &active, &copies, &cs, &lanes, &em, self.t_draft + self.t_verify, k_draft > 0, rows, &self.early_t_host, m.rank == 0);
            }
        }
        let total_c: usize = cs.iter().sum();
        let gap = self.t_end.map(|e| t0.saturating_duration_since(e).as_secs_f64() * 1e3);
        self.t_end = Some(t2);
        // leg CD1: a round with a copy lane (or without its draft pass) is not the shape the cost model prices: not observed
        let priced = if ncopy > 0 { None } else { choice.as_ref().map(|c| c.ms).or(lchoice.as_ref().map(|c| c.ms)) };
        let obs_rec = PendingRoundObs {
            na, total_c, k_draft, rows, k_req: k, priced_ms: priced, ncopy,
            wall_ms: (self.t_draft + self.t_verify) as f32,
            gap_ms: gap.map_or(-1.0, |g| g as f32),
        };
        if self.lockstep_cs && m.world > 1 {
            self.pending_obs = Some(obs_rec);
        } else {
            self.commit_round_obs(obs_rec, self.t_draft + self.t_verify, gap);
        }
        if m.rank == 0 && k_draft > 0 && ncopy == 0 {
            use crate::dsv41::gpu::dspark::metrics as dm;
            dm::round_wall(self.t_draft + self.t_verify);
            if self.policy.mode == policy::Mode::Auto {
                // the one-c histogram counts the rounds whose lanes share a count; leg PS3's per-lane receipts count every round
                if even { dm::decision(kmax, expected, emitted); } else { dm::decision_uneven(expected, emitted); }
                if self.policy.decides_after_draft() {
                    crate::dsv41::gpu::dspark::alloc::record(&cs, expected, emitted, rows, lchoice.as_ref().map_or(false, |c| c.scan_differs) || choice.as_ref().map_or(false, |c| c.scan_differs),
                                                             lchoice.as_ref().map_or(0, |c| c.capped));
                }
                dm::PROBES.store(self.policy.probes, std::sync::atomic::Ordering::Relaxed);
                if self.policy.decisions % 16 == 1 { self.publish_costs(); }
            }
        }
        super::hostprof::round_end(m.rank, rows);
        Ok(em)
    }

    fn commit_round_obs(&mut self, o: PendingRoundObs, wall_ms: f64, gap: Option<f64>) {
        if let Some(ms) = o.priced_ms {
            if let Some(model) = self.policy.model.as_mut() {
                model.observe_d(o.na, o.total_c, wall_ms);
                if let Some(g) = gap { model.observe_gap(g); }
            } else {
                let table = ms - self.policy.overhead_ms(o.na);
                self.policy.observe_overhead(o.na, wall_ms, table);
            }
        }
        if !self.policy.decides_after_draft() && o.ncopy == 0 {
            if o.k_draft > 0 { self.policy.costs.observe_draft(o.na, o.k_draft, wall_ms * 0.15); }
            self.policy.costs.observe_verify(o.rows, wall_ms);
            self.policy.costs.observe_vk(o.na, o.k_req, wall_ms);
        }
    }

    pub fn peek_rank0_obs_bits(&self) -> Option<(u32, u32)> {
        self.pending_obs.map(|o| (o.wall_ms.to_bits(), o.gap_ms.to_bits()))
    }

    pub fn apply_pending_obs(&mut self, rank0_bits: Option<(u32, u32)>) {
        let Some(o) = self.pending_obs.take() else { return };
        let Some((wb, gb)) = rank0_bits else { return };
        let wall = f32::from_bits(wb) as f64;
        let gf = f32::from_bits(gb);
        let gap = (gf >= 0.0 && gf.is_finite()).then_some(gf as f64);
        self.commit_round_obs(o, wall, gap);
    }

    pub fn sync_pending_obs(&mut self, m: &mut TpModel) -> Result<()> {
        if !self.lockstep_cs || m.world <= 1 { return Ok(()); }
        let send = if m.rank == 0 {
            if let Some((wb, gb)) = self.peek_rank0_obs_bits() {
                [1.0, (wb & 0xFFFF) as f32, (wb >> 16) as f32, (gb & 0xFFFF) as f32, (gb >> 16) as f32]
            } else { [0.0; 5] }
        } else { [0.0; 5] };
        let got = m.bcast_small(&send)?;
        let bits = (got[0] != 0.0).then(|| (
            (got[1] as u32) | ((got[2] as u32) << 16),
            (got[3] as u32) | ((got[4] as u32) << 16),
        ));
        self.apply_pending_obs(bits);
        Ok(())
    }

    /// Leg CD1, the head: whether the next round may take copy proposals. Always true, except under the measurement switch
    /// `--dsv41-copy-ab-period-s P` (> 0): true during even P-second periods of the head's clock, false during odd ones, so one
    /// server alternates copy drafting ON and OFF in adjacent time slices (a paired A/B on the same box, thermal state and
    /// boot). Also counts the rounds served under each permission (`velogb10_dsv41_copy_ab_*`).
    pub fn copy_permission(&mut self) -> bool {
        let Some(cd) = self.copy.as_ref() else { return true };
        let ok = if cd.ab_period_s > 0.0 {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
            ((t / cd.ab_period_s) as u64) % 2 == 0
        } else { true };
        if cd.ab_period_s > 0.0 {
            use std::sync::atomic::Ordering::Relaxed as R;
            if ok { super::copy::AB_ON_ROUNDS.fetch_add(1, R); } else { super::copy::AB_OFF_ROUNDS.fetch_add(1, R); }
        }
        ok
    }

    /// Legs CD1 + P7COPY: deterministic lockstep `CopySpan` + `COPYLANES` proposer run identically on every TP rank (zero RDMA).
    fn copy_propose(&mut self, m: &TpModel, nl: usize, active: &[usize]) -> Vec<Vec<u32>> {
        let pen_act: Vec<bool> = (0..nl).map(|l| self.pen.as_ref().map_or(false, |p| p.active(l))).collect();
        let Some(cd) = self.copy.as_mut() else { return vec![Vec::new(); nl] };
        cd.propose_round(nl, active, &m.lanes, &self.host, &pen_act, m.cfg.eos, self.policy.max_rows, &self.early_t_host, m.rank == 0)
    }

    /// Leg PS3 + P7COPY: the head's logits of a round whose lanes verify different counts: ONE head launch over the call's
    /// `rows` rows (`rows <= 32`), then each lane's rows copied to its `rpl`-row slot of `tlog` on the compute stream.
    fn head_scatter_rpl(&mut self, m: &mut TpModel, rows: usize, rpl: usize) -> Result<()> {
        let vl = self.sh.vlocal;
        ensure!(rows <= 32 && rows * vl * 4 <= self.tlog_c.bytes && rpl <= WIDE_RPL, "head scatter: {rows} rows, rpl {rpl}");
        m.head_logits(0, rows, self.tlog_c.ptr())?;
        let segs = m.segs.clone();
        let plan = scatter_plan_rpl(&segs.iter().map(|sg| (sg.lane, sg.row0, sg.n)).collect::<Vec<_>>(), self.cap, rpl)?;
        for &(dst_row, src_row, n) in &plan {
            m.f.g.d2d(self.tlog.ptr() + (dst_row * vl * 4) as u64, self.tlog_c.ptr() + (src_row * vl * 4) as u64, n * vl * 4)?;
        }
        super::metrics::head_launches(1, segs.len());
        Ok(())
    }

    /// Leg PH1: the Markov bias cache, after `new` and before any draft graph is captured. The list (`--dsv41-markov-tokens`,
    /// read on rank 0) and its size (`--dsv41-markov-cache`) are broadcast from rank 0, every rank fills its vocabulary
    /// rows and checks them bit for bit against the uncached step over every cached token. Any failure leaves the cache
    /// off with one line (the drafts are the same either way; only the step time differs). Returns the boot line.
    pub fn attach_markov_cache(&mut self, m: &mut TpModel) -> Result<String> {
        self.mc = None;
        let off = |why: String| -> Result<String> { mcache::set_state(Err(why.clone())); Ok(format!("[dsv41 markov cache] off: {why}")) };
        if self.drafter_off { return off("speculation is off (the drafter never runs)".into()); }
        let mut k = mcache::k_from_flags()?;
        let mut dyn_k = mcache::dyn_k_from_flags(m.world)?;
        let k_explicit = crate::opts::lookup("dsv41-markov-cache").and_then(crate::opts::get).is_some_and(|v| v != "unset");
        let dyn_explicit = crate::opts::lookup("dsv41-markov-dyn-cache").and_then(crate::opts::get).is_some_and(|v| v != "unset" && v != "auto");
        let mut ram_capped = false;
        // leg P7.6: keep Rank 0 free RAM well above the 5.0 GB watchdog floor (>= 7.5 GB headroom before 512+512 default)
        if m.rank == 0 && (!k_explicit || !dyn_explicit) {
            if let Some(free) = super::mem::avail() {
                if free < 7_500_000_000 {
                    if !k_explicit && k > 256 { k = 256; ram_capped = true; }
                    if !dyn_explicit && dyn_k > 256 { dyn_k = 256; ram_capped = true; }
                }
            }
        }
        let kd = m.bcast_small(&[k as f32, dyn_k as f32, u8::from(ram_capped) as f32])?;
        k = kd[0] as usize;
        dyn_k = kd[1] as usize;
        ram_capped = kd[2] != 0.0;
        // rank 0 resolves the list; every rank takes part in the same broadcasts
        let (mine, mut label) = if m.rank == 0 {
            match mcache::list_from_flags() {
                Ok((ids, label)) => (mcache::first_distinct(&ids, k, m.cfg.vocab), label),
                Err(e) => (Vec::new(), format!("the token list could not be read: {e:#}")),
            }
        } else { (Vec::new(), String::new()) };
        if ram_capped && m.rank == 0 { label.push_str("; capped to 256+256 under tight free RAM"); }
        let n = m.bcast_small(&[mine.len() as f32])?[0] as usize;
        let mut toks = Vec::with_capacity(n);
        for c in (0..n).step_by(16) {
            let part: Vec<f32> = (c..(c + 16).min(n)).map(|i| mine.get(i).copied().unwrap_or(0) as f32).collect();
            toks.extend(m.bcast_small(&part)?.into_iter().map(|x| x as u32));
        }
        if k == 0 { return off("--dsv41-markov-cache 0".into()); }
        if n == 0 { return off(if m.rank == 0 { label } else { "rank 0 sent an empty token list".into() }); }
        let cfg = m.cfg;
        let w2 = &m.layers[cfg.layers + 2].as_ref().context("the DSpark stages are not loaded")?.w;
        let (me, mh) = (w2.lin(C::DsMarkovEmbed)?.clone(), w2.lin(C::DsMarkovHead)?.clone());
        if let Some(why) = mcache::unsupported(&me, &mh) { return off(why); }
        let mc = match MarkovCache::build_with_dyn(m.f.g, &me, &mh, toks, dyn_k, cfg.vocab) { Ok(c) => c, Err(e) => return off(format!("fill failed: {e:#}")) };
        let (groups, bad) = mc.selfcheck(m.f.g, &mh, &m.f.scr)?;
        let mb = (mc.total_k() * mc.nn * 4) as f64 / 1048576.0;
        let label = if m.rank == 0 { label } else { "from rank 0".into() };
        if bad > 0 {
            return off(format!("the load-time check found {bad} of {groups} token groups where the cached row differs from the computed step (a defect: report it)"));
        }
        mcache::DYN_CAP.store(mc.dyn_cap as u64, std::sync::atomic::Ordering::Relaxed);
        let line = format!("[dsv41 markov cache] rank {}: {} static + {} dynamic tokens ({label}) x {} vocabulary rows = {mb:.1} MB, filled in {:.1} ms by the step's own kernel; \
                            bit check over every cached token (the plain GEMV, the computed step and the cached step on the same logits, 1 and 4 lanes): \
                            {groups} groups identical. A draft step whose every active lane's input is cached skips the Markov head read; misses write back into the dynamic ring",
                           m.rank, mc.k(), mc.dyn_cap, mc.nn, mc.fill_ms);
        mcache::set_state(Ok((mc.k(), mb, mc.fill_ms, groups, bad, label)));
        self.mc = Some(mc);
        Ok(line)
    }

    /// Leg PH1 + P6-3, rank 0 after a drafted round: mirror dynamic admissions + step outcomes from the round's ids table,
    /// and every 64 rounds read the kernel's device counters.
    fn markov_receipts(&mut self, g: &Gpu, ids: &[i32], nl: usize, active: &[usize], k: usize, stopped_from: Option<usize>) -> Result<()> {
        let Some(mc) = self.mc.as_mut() else { return Ok(()) };
        if nl > mcache::MAX_LANES { return Ok(()); }
        let steps: Vec<Vec<i32>> = (0..k).map(|i| active.iter().map(|&l| ids[i * nl + l]).collect()).collect();
        mcache::record_steps(mc, &steps, stopped_from);
        self.mc_rounds += 1;
        if self.mc_rounds % 64 == 1 {
            let [h, c, df, es] = mc.device_counts_all(g)?;
            mcache::DEV_HIT.store(h, std::sync::atomic::Ordering::Relaxed);
            mcache::DEV_MISS.store(c, std::sync::atomic::Ordering::Relaxed);
            mcache::DEV_DYN_FILLS.store(df, std::sync::atomic::Ordering::Relaxed);
            mcache::DEV_EARLY_SKIP.store(es, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// `/metrics`: the round-cost table the policy prices (every lane count up to the spec lanes, c = 0..5) and the
    /// conf rule's learned overhead; with leg PS2's online model, its priced table, where it came from and how old it is.
    pub fn publish_costs(&self) {
        let p = &self.policy;
        if let Some(model) = p.model.as_ref() {
            crate::dsv41::gpu::dspark::metrics::publish_costs(model.rows(), Vec::new());
            crate::dsv41::gpu::dspark::metrics::publish_model(model);
            return;
        }
        let mut t = Vec::new();
        for l in 1..=self.cap {
            for c in 0..=BLOCK {
                let depth = if p.decides_after_draft() { BLOCK } else { c };
                if let Some(ms) = p.costs.round_ms_d(l, c, depth) { t.push((l, c, ms)); }
            }
        }
        let o: Vec<(usize, f64)> = if p.decides_after_draft() { (1..=self.cap).map(|l| (l, p.overhead_ms(l))).collect() } else { Vec::new() };
        crate::dsv41::gpu::dspark::metrics::publish_costs(t, o);
    }

    /// The round costs the draft-length policy prices (DS's policy, `--dsv41-draft-len auto`): read from the cache
    /// (`<binary dir>/dsv41_spec_calib/<pack>.json`, stamped with the kernel build, the binary, the pack, the TP size, the lane
    /// count, the node list and the set Spmd flags) or measured here: verify calls of 1..6 rows per lane at every lane count
    /// 1..=nl, draft steps of k = 1..5. Rank 0 decides (cache or measure) and its timings price the policy; every rank runs the
    /// same calls. The lanes 0..nl must be started (prompts prefilled). Returns a one-line receipt.
    ///
    /// Leg PS2: the verify shapes are measured in INTERLEAVED sweeps (every shape once per sweep, so heat or a clock change
    /// during the calibration moves every shape alike instead of tilting the table), one warm-up sweep (graph capture) then
    /// `CAL_REPS_MIN`..`CAL_REPS_MAX` timed sweeps: sweeps stop after the minimum once `CAL_BUDGET` has passed (the boot-time
    /// bound; P1M's first boot timed 4 iterations per shape). Every repetition is kept: the conf rule's online model
    /// (`roundcost`) starts from their fit (fixed + per-row cost, strictly increasing) and then follows the served rounds.
    pub fn calibrate(&mut self, m: &mut TpModel, nl: usize, pack: &std::path::Path) -> Result<String> {
        let stamp = policy::calib_stamp_with_dual(&policy::pack_fingerprint(pack), m.world, nl, m.comm.dual_on());
        let path = policy::calib_path(pack);
        let cached = if m.rank == 0 { path.as_ref().and_then(|p| policy::read_calib(p, &stamp)) } else { None };
        let hit = m.bcast_small(&[if cached.is_some() { 1.0 } else { 0.0 }])?[0] == 1.0;
        if hit {
            if let Some(c) = cached { self.policy.costs = c; }
            let model = self.sync_calib_from_rank0(m, nl, crate::dsv41::gpu::dspark::roundcost::Source::Cached)?;
            if m.rank == 0 { self.publish_costs(); }
            return Ok(format!("draft-length costs from the cache {} (measured {} s ago; {} verify shapes){model}", path.map(|p| p.display().to_string()).unwrap_or_default(),
                              now_unix().saturating_sub(self.policy.costs.measured_unix), self.policy.costs.vk.len()));
        }
        let g = m.f.g;
        let t_cal = std::time::Instant::now();
        let mut c = policy::Costs { measured_unix: now_unix(), ..Default::default() };
        let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.total_cmp(b)); v[v.len() / 2] };
        // leg PS1 + P6-4: 1..=nl when nl <= 4, representative [1, 2, 4, nl] when nl > 4, capped to HEAD_MAX rows
        let cal_lanes: Vec<usize> = if nl <= 4 { (1..=nl).collect() } else { let mut v = vec![1, 2, 4, nl]; v.sort_unstable(); v.dedup(); v };
        let mut reps: std::collections::BTreeMap<(usize, usize), Vec<f64>> = std::collections::BTreeMap::new();
        let mut sweeps = 0usize;
        for sweep in 0..=CAL_REPS_MAX {
            for &ln in &cal_lanes {
                let r_max = if ln == 1 { RPL } else { RPL.min(super::fwd::HEAD_MAX / ln).max(1) };
                for r in 1..=r_max {
                    let toks: Vec<Vec<u32>> = (0..ln).map(|l| (0..r).map(|i| 1000 + ((sweep * 7 + i + l * 13) % 50000) as u32).collect()).collect();
                    let parts: Vec<(usize, &[u32])> = (0..ln).map(|l| (l, toks[l].as_slice())).collect();
                    g.sync()?;
                    let t = std::time::Instant::now();
                    m.taps = Some(self.verify_taps(m, ln * r));
                    let res = m.forward_segs(&parts, HeadWant::None, None);
                    m.taps = None;
                    res?;
                    m.head_logits_lanes(r, self.tlog.ptr())?;   // leg PF1: priced as the round runs it
                    g.sync()?;
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    for l in 0..ln { m.rollback(l, m.lanes[l].pos - r)?; }
                    if sweep > 0 { reps.entry((ln, r - 1)).or_default().push(ms); }
                }
            }
            if sweep > 0 { sweeps = sweep; }
            // every rank stops after the same sweep: rank 0's clock decides, one broadcast per sweep
            let stop = sweep >= CAL_REPS_MIN && t_cal.elapsed() >= CAL_BUDGET;
            if m.bcast_small(&[if stop { 1.0 } else { 0.0 }])?[0] == 1.0 { break; }
        }
        for (&(ln, k), v) in &reps {
            let ms = med(&mut v.clone());
            c.vk.insert((ln, k), ms);
            if nl > 4 || ln == 1 || ln == nl { c.verify.insert(ln * (k + 1), ms); }
        }
        c.vk_reps = reps;
        for &ln in &cal_lanes {
            self.round.set_lanes(g, &self.host[..ln])?;
            for k in 1..=BLOCK {
                let mut v = Vec::new();
                for it in 0..=CAL_DRAFT_REPS {
                    g.sync()?;
                    let t = std::time::Instant::now();
                    self.draft(m, ln, k)?;
                    g.sync()?;
                    if it > 0 { v.push(t.elapsed().as_secs_f64() * 1e3); }
                }
                c.draft.insert((ln, k), med(&mut v));
            }
        }
        if let Some(mc) = self.mc.as_mut() { mc.reset_dyn(g)?; }
        m.comm.check("the draft-length calibration")?;
        if m.rank == 0 { if let Some(p) = &path { let _ = policy::write_calib(p, &stamp, &c); } }
        let line = format!("draft-length costs measured in {:.1} s ({sweeps} interleaved sweeps; rank {} timings): verify (lanes x drafts, median) {} ms; draft {} ms",
                           t_cal.elapsed().as_secs_f64(), m.rank,
                           c.vk.iter().map(|((l, k), ms)| format!("{l}x{k}:{ms:.1}")).collect::<Vec<_>>().join(" "),
                           c.draft.iter().map(|((l, k), ms)| format!("{l}x{k}:{ms:.1}")).collect::<Vec<_>>().join(" "));
        self.policy.costs = c;
        let model = self.sync_calib_from_rank0(m, nl, crate::dsv41::gpu::dspark::roundcost::Source::Measured)?;
        if m.rank == 0 { self.publish_costs(); }
        Ok(format!("{line}{model}"))
    }

    /// Leg P7.4: broadcast Rank 0's calibrated `Costs` + fitted `RoundCost` prior once at boot so every rank constructs
    /// bit-identical `Policy` / `RoundCost` / `early_t` state and never needs a mid-round `bcast_small`.
    fn sync_calib_from_rank0(&mut self, m: &mut TpModel, nl: usize, source: crate::dsv41::gpu::dspark::roundcost::Source) -> Result<String> {
        let pack_u64 = |w: &mut Vec<u64>, x: u64| w.push(x);
        let mut words = Vec::new();
        if m.rank == 0 {
            let c = &self.policy.costs;
            pack_u64(&mut words, c.measured_unix);
            pack_u64(&mut words, c.host_ms.to_bits());
            pack_u64(&mut words, c.verify.len() as u64);
            for (&r, &ms) in &c.verify { pack_u64(&mut words, r as u64); pack_u64(&mut words, ms.to_bits()); }
            pack_u64(&mut words, c.vk.len() as u64);
            for (&(l, k), &ms) in &c.vk { pack_u64(&mut words, ((l as u64) << 32) | (k as u64)); pack_u64(&mut words, ms.to_bits()); }
            pack_u64(&mut words, c.draft.len() as u64);
            for (&(l, k), &ms) in &c.draft { pack_u64(&mut words, ((l as u64) << 32) | (k as u64)); pack_u64(&mut words, ms.to_bits()); }
            let prior = if self.policy.decides_after_draft() { c.prior(nl, BLOCK, self.policy.overhead0) } else { None };
            match prior {
                Some(pr) => {
                    pack_u64(&mut words, pr.len() as u64);
                    for row in pr { for v in row { pack_u64(&mut words, v.to_bits()); } }
                }
                None => pack_u64(&mut words, 0),
            }
        }
        let nw = m.bcast_small(&[words.len() as f32])?[0] as usize;
        let mut got_w = Vec::with_capacity(nw);
        for c0 in (0..nw).step_by(4) {
            let n = (nw - c0).min(4);
            let mut send = vec![0.0f32; n * 4];
            if m.rank == 0 {
                for i in 0..n {
                    let u = words[c0 + i];
                    send[4 * i] = (u & 0xFFFF) as f32;
                    send[4 * i + 1] = ((u >> 16) & 0xFFFF) as f32;
                    send[4 * i + 2] = ((u >> 32) & 0xFFFF) as f32;
                    send[4 * i + 3] = ((u >> 48) & 0xFFFF) as f32;
                }
            }
            let r = m.bcast_small(&send)?;
            for i in 0..n {
                got_w.push((r[4 * i] as u64) | ((r[4 * i + 1] as u64) << 16) | ((r[4 * i + 2] as u64) << 32) | ((r[4 * i + 3] as u64) << 48));
            }
        }
        let mut it = got_w.into_iter();
        let measured_unix = it.next().unwrap_or(0);
        let host_ms = f64::from_bits(it.next().unwrap_or(0));
        let mut synced = policy::Costs { measured_unix, host_ms, vk_reps: std::mem::take(&mut self.policy.costs.vk_reps), ..Default::default() };
        let nv = it.next().unwrap_or(0) as usize;
        for _ in 0..nv { let r = it.next().unwrap_or(0) as usize; synced.verify.insert(r, f64::from_bits(it.next().unwrap_or(0))); }
        let nvk = it.next().unwrap_or(0) as usize;
        for _ in 0..nvk { let lk = it.next().unwrap_or(0); synced.vk.insert(((lk >> 32) as usize, (lk & 0xFFFF_FFFF) as usize), f64::from_bits(it.next().unwrap_or(0))); }
        let nd = it.next().unwrap_or(0) as usize;
        for _ in 0..nd { let lk = it.next().unwrap_or(0); synced.draft.insert(((lk >> 32) as usize, (lk & 0xFFFF_FFFF) as usize), f64::from_bits(it.next().unwrap_or(0))); }
        self.policy.costs = synced;
        let np = it.next().unwrap_or(0) as usize;
        let tail = if np > 0 {
            let mut pr = Vec::with_capacity(np);
            for _ in 0..np {
                let row: [f64; crate::dsv41::gpu::dspark::roundcost::NC] = std::array::from_fn(|_| f64::from_bits(it.next().unwrap_or(0)));
                pr.push(row);
            }
            let desc = pr.iter().enumerate().map(|(l, t)| format!("{}: {}", l + 1, t.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(" "))).collect::<Vec<_>>().join("; ");
            self.policy.model = Some(crate::dsv41::gpu::dspark::roundcost::RoundCost::new(pr, source, measured_unix));
            format!("; conf rule prices start from the fitted table (fixed + per-row cost, + {:.1} ms bookkeeping until served rounds replace it) {desc} ms", self.policy.overhead0)
        } else {
            self.build_model(nl, source)
        };
        let et = self.one_lane_prices(BLOCK);
        let et_f32 = [et[0] as f32, et[1] as f32, et[2] as f32, et[3] as f32, et[4] as f32, et[5] as f32];
        self.sync_early_t_from_f32(m.f.g, &et_f32)?;
        Ok(tail)
    }

    /// Leg PS2: the conf rule's online model from the boot costs (prior = fitted boot table + `--dsv41-spec-overhead-ms`);
    /// returns the receipt's tail (the fitted prior at every lane count) or "" when the rule does not use it.
    fn build_model(&mut self, nl: usize, source: crate::dsv41::gpu::dspark::roundcost::Source) -> String {
        if !self.policy.decides_after_draft() { return String::new(); }
        let Some(prior) = self.policy.costs.prior(nl, BLOCK, self.policy.overhead0) else { return String::new() };
        let tail = prior.iter().enumerate().map(|(l, t)| format!("{}: {}", l + 1, t.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(" "))).collect::<Vec<_>>().join("; ");
        self.policy.model = Some(crate::dsv41::gpu::dspark::roundcost::RoundCost::new(prior, source, self.policy.costs.measured_unix));
        format!("; conf rule prices start from the fitted table (fixed + per-row cost, + {:.1} ms bookkeeping until served rounds replace it) {tail} ms",
                self.policy.overhead0)
    }

    /// The draft length of the next round: rank 0's policy decides, every rank takes its value. The conf rule drafts
    /// `MAX_K` on every rank (its verify count is decided after the draft, in `round`): no broadcast needed here.
    pub fn choose_k(&mut self, m: &mut TpModel, nl: usize) -> Result<usize> {
        if self.policy.decides_after_draft() { return Ok(self.policy.choose(nl)); }
        // leg PH1: a pinned length (or speculation off) is the same on every rank from the flags: no broadcast
        if matches!(self.policy.mode, policy::Mode::Off | policy::Mode::Fixed(_)) { return Ok(self.policy.choose(nl)); }
        let k = self.policy.choose(nl);
        let v = m.bcast_small(&[k as f32])?;
        Ok(v[0] as usize)
    }
}

/// Leg PS3: the copies of an uneven round's head rows, (destination row in `tlog`, source row of the call, rows) per
/// segment (lane, call row0, rows): lane l's rows land at l x RPL .. (the verify reads row (l, j) at l x RPL + j). Refuses
/// a segment over RPL rows, a lane outside the round's capacity, a lane twice, or a call whose rows are not consecutive.
pub fn scatter_plan(segs: &[(usize, usize, usize)], cap: usize) -> Result<Vec<(usize, usize, usize)>> {
    scatter_plan_rpl(segs, cap, RPL)
}

pub fn scatter_plan_rpl(segs: &[(usize, usize, usize)], cap: usize, rpl: usize) -> Result<Vec<(usize, usize, usize)>> {
    let mut next = 0usize;
    let mut seen = vec![false; cap];
    let mut v = Vec::with_capacity(segs.len());
    for &(lane, row0, n) in segs {
        ensure!(lane < cap && !seen[lane], "head scatter: lane {lane} (cap {cap}, twice?)");
        ensure!(n >= 1 && n <= rpl && row0 == next, "head scatter: lane {lane} rows {row0}..{} (expected from {next}, at most {rpl})", row0 + n);
        seen[lane] = true;
        next += n;
        v.push((lane * rpl, row0, n));
    }
    Ok(v)
}

#[cfg(test)]
mod ps3_tests {
    use super::*;

    /// Greedy byte identity of uneven rounds rests on (a) every lane's target rows reaching the verify unchanged and at the
    /// right place, and (b) the head rows being bit-identical whatever the call holds (a GPU property: G4 + the head probe
    /// in the pair plan). (a) on CPU: every uneven split of 2-4 lanes x 1..6 rows: each verify read (l, j) at l x RPL + j
    /// is fed by call row row0(l) + j, destinations never overlap, and every call row is copied exactly once.
    #[test]
    fn dsv41_ps3_uneven_head_rows_reach_their_lane_slots() {
        let mut checked = 0;
        for lanes in 2..=4usize {
            let combos = RPL.pow(lanes as u32);
            for code in 0..combos {
                let rows: Vec<usize> = (0..lanes).map(|i| 1 + (code / RPL.pow(i as u32)) % RPL).collect();
                if rows.iter().sum::<usize>() > 24 { continue; }
                // active lanes in slot order, some slots skipped (an inactive lane between active ones)
                let slots: Vec<usize> = if lanes < 4 && code % 2 == 1 { (0..lanes).map(|i| i + 1).collect() } else { (0..lanes).collect() };
                let mut segs = Vec::new();
                let mut r0 = 0;
                for (i, &l) in slots.iter().enumerate() { segs.push((l, r0, rows[i])); r0 += rows[i]; }
                let plan = scatter_plan(&segs, 4).unwrap();
                let mut src_of = std::collections::HashMap::new();
                let mut copied = vec![0; r0];
                for &(d, sr, n) in &plan { for j in 0..n { assert!(src_of.insert(d + j, sr + j).is_none(), "overlap"); copied[sr + j] += 1; } }
                assert!(copied.iter().all(|&c| c == 1));
                for &(l, row0, n) in &segs { for j in 0..n { assert_eq!(src_of[&(l * RPL + j)], row0 + j); } }
                checked += 1;
            }
        }
        assert!(checked > 1000, "{checked}");
        assert!(scatter_plan(&[(0, 0, 7)], 4).is_err());
        assert!(scatter_plan(&[(0, 0, 2), (0, 2, 2)], 4).is_err());
        assert!(scatter_plan(&[(0, 0, 2), (1, 3, 2)], 4).is_err());
        assert!(scatter_plan(&[(4, 0, 2)], 4).is_err());
    }
}

