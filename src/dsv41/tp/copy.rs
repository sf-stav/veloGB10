//! Leg CD1 + P7COPY (`COPY-A` / `COPY-B`): lockstep `CopySpan` + `YieldGate` + `COPYLANES` + optional `CopyRecord`
//! copy drafting for the DSV4.1 TP speculation round (`--dsv41-copy-draft`).
//!
//! Every TP rank holds the identical committed lane history (`TpModel::lanes[l].hist`), the identical pending token
//! (`TpSpec::host[l].tok`), the identical round outcomes, and the identical rank-agreed `early_t_host` 1-lane cost
//! vector. Because `crate::copy_span::CopySpan` uses a deterministic fixed-seed Murmur3 multi-length n-gram table
//! plus an `O(1)` sticky source cursor and integer fixed-point `YieldGate`, **every TP rank runs the exact same
//! proposer locally with ZERO RDMA broadcasts** (and folds a proposal digest into `Guard::agree("serve round")` as a
//! zero-cost fail-fast invariant).
//!
//! Exactness contract (`91_ds_vocab.cuh` `d41_ds_accept` `L.copy` branch, unchanged across widths `1..=15` and
//! `COPYPF1` stretches `32 / 160 / 512`):
//! * greedy lanes: a copied token is kept iff it equals the target's exact argmax (`tr.aid`), lowest id on ties;
//! * sampled lanes: a copied token is kept iff it equals the target's own keyed Gumbel draw (`tr.gid`, key
//!   `(seed, pos * 3)`), and the first mismatch emits `tr.gid` itself — distribution-exact AND bit-identical to
//!   plain decode for the same seed, with no residual pass needed.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::copy_lanes::{self, Verdict, Work};
use crate::copy_prefill;
use crate::copy_record::{CopyRecord, Identity, Options as RecordOptions};
use crate::copy_span::{CopySpan, CopySpanCfg, Proposal, Source};
use crate::dsv41::gpu::dspark::{Lane as DsLane, BLOCK, WIDE_RPL};

/// Default single-lane decode copy width cap (`WIDEVERIFY1`: 15 copied tokens = 16 verify rows).
pub const MAX_COPY: usize = WIDE_RPL - 1;
/// Legacy CD1 / penalty-active per-lane copy cap (`BLOCK` = 5 copied tokens = 6 verify rows).
pub const LEGACY_MAX_COPY: usize = BLOCK;
/// Minimum copied tokens worth proposing (1 copied token alone never amortizes skipping DSpark or widening verify).
pub const MIN_COPY: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CopyCfg {
    /// minimum backward n-gram match length (`--dsv41-copy-min-match`, default 12)
    pub min_match: usize,
    /// maximum copied tokens in a single-lane decode verify (`--dsv41-copy-max-w`, default 15)
    pub max_w: usize,
    /// initial search proposal cap before cursor ramp (`--dsv41-copy-start-w`, default 4 -> 8 -> 15)
    pub start_w: usize,
    /// neural-vs-copy cost-ratio `YieldGate` on search starts (`--dsv41-copy-yield-gate`, default true)
    pub yield_gate: bool,
    /// optional cost-ratio override (x256; 0 = `CopySpan` analytical cost-ratio rule `ec*(4+en) >= en*(3+W)`)
    pub yield_ratio_x256: i64,
    /// `COPYLANES` 1.15x multi-user batch-rate pricing (`--dsv41-copy-lanes`, default true)
    pub copy_lanes: bool,
    /// cross-request `CopyRecord` fallback (`--dsv41-copy-record`, default false)
    pub copy_record: bool,
    /// opt-in `COPYPF1` 32/160/512-row prefill copy stretch (`--dsv41-copy-prefill`, default false)
    pub prefill: bool,
    /// max rows of a solo `COPYPF1` stretch (`--dsv41-copy-prefill-max-rows`, default 512; multi-lane <= 160)
    pub prefill_max_rows: usize,
    /// `ECHOPRICE` online stretch pricing (`--dsv41-copy-prefill-price`, default true)
    pub prefill_price: bool,
    /// legacy CD1 zero-accept streak threshold (`--dsv41-copy-zero-windows`, default 4)
    pub zero_windows: u32,
    /// legacy CD1 silence duration (`--dsv41-copy-silence`, default 16)
    pub silence: u32,
    /// legacy CD1 running mean threshold (`--dsv41-copy-min-yield`, default 0.0 = off)
    pub min_yield: f64,
}

impl Default for CopyCfg {
    fn default() -> Self {
        CopyCfg {
            min_match: 12,
            max_w: MAX_COPY,
            start_w: 4,
            yield_gate: true,
            yield_ratio_x256: 0,
            copy_lanes: true,
            copy_record: false,
            prefill: false,
            prefill_max_rows: 512,
            prefill_price: true,
            zero_windows: 4,
            silence: 16,
            min_yield: 0.0,
        }
    }
}

impl CopyCfg {
    pub fn span_cfg(&self) -> CopySpanCfg {
        let mm = self.min_match.max(4);
        let sw = self.start_w.max(1).min(self.max_w.max(1));
        let mut sc = CopySpanCfg::new(mm).with_start_w(sw);
        sc.max_n = mm.max(32);
        if self.yield_gate {
            sc = sc.with_yield_gate(self.max_w.max(MIN_COPY)).with_yield_ratio(self.yield_ratio_x256.max(0));
        }
        sc
    }
}

pub struct LaneCopy {
    pub span: CopySpan,
    pub work: Work,
    pub cpv_rung: usize,
    pub pending: Option<Proposal>,
    pub synced: bool,
    zero_streak: u32,
    silent_for: u32,
    ewma: Option<f64>,
    judged: u32,
    was_suspended: bool,
}

impl Default for LaneCopy {
    fn default() -> Self { Self::new(&CopyCfg::default()) }
}

impl LaneCopy {
    pub fn new(cfg: &CopyCfg) -> Self {
        LaneCopy {
            span: CopySpan::new(&[], cfg.span_cfg()),
            work: Work::default(),
            cpv_rung: 0,
            pending: None,
            synced: false,
            zero_streak: 0,
            silent_for: 0,
            ewma: None,
            judged: 0,
            was_suspended: false,
        }
    }

    pub fn reset(&mut self) {
        self.span.reset(&[]);
        self.work = Work::default();
        self.cpv_rung = 0;
        self.pending = None;
        self.synced = false;
        self.zero_streak = 0;
        self.silent_for = 0;
        self.ewma = None;
        self.judged = 0;
        self.was_suspended = false;
    }

    pub fn reset_with_cfg(&mut self, cfg: &CopyCfg) { *self = Self::new(cfg); }

    /// Ensure `self.span` reflects `hist + [pending_tok]` without rebuilding when already aligned in lockstep.
    pub fn sync_lane(&mut self, hist: &[u32], pending_tok: u32, cfg: &CopyCfg) {
        let want_cfg = cfg.span_cfg();
        if self.span.cfg() != want_cfg {
            // Preserve live context/cursor when only start_w/yield_ratio changed, else rebuild:
            let cur_ctx = self.span.ctx().to_vec();
            self.span = CopySpan::new(&cur_ctx, want_cfg);
        }
        let want_len = hist.len() + 1;
        let aligned = self.synced
            && self.span.ctx_len() == want_len
            && self.span.ctx().last() == Some(&pending_tok)
            && (hist.is_empty() || self.span.ctx().get(hist.len() - 1) == hist.last());
        if aligned { return; }
        let mut full = Vec::with_capacity(want_len);
        full.extend_from_slice(hist);
        full.push(pending_tok);
        self.span.reset(&full);
        self.work = Work::default();
        self.cpv_rung = 0;
        self.pending = None;
        self.synced = true;
        self.was_suspended = false;
    }

    /// Direct context proposal helper (preserves leg CD1's standalone unit-test API while running `CopySpan`).
    pub fn propose(&mut self, ctx: &[u32], cfg: &CopyCfg, room: usize) -> Vec<u32> {
        if self.silent_for > 0 { self.silent_for -= 1; return Vec::new(); }
        let cap = cfg.max_w.min(room);
        if ctx.len() < cfg.min_match.max(4) + 1 || cap < MIN_COPY { return Vec::new(); }
        let want_cfg = cfg.span_cfg();
        if self.span.cfg() != want_cfg {
            self.span = CopySpan::new(&[], want_cfg);
            self.synced = false;
        }
        let stale = !self.synced
            || self.span.ctx_len() > ctx.len()
            || (self.span.ctx_len() > 0 && self.span.ctx().last() != ctx.get(self.span.ctx_len() - 1));
        if stale {
            self.span.reset(ctx);
            self.synced = true;
        } else if self.span.ctx_len() < ctx.len() {
            let add = &ctx[self.span.ctx_len()..];
            self.span.commit_follow(add);
        }
        match self.span.propose(cap) {
            Some(p) if p.tokens.len() >= MIN_COPY => {
                let toks = p.tokens.clone();
                self.pending = Some(p);
                toks
            }
            _ => {
                self.pending = None;
                Vec::new()
            }
        }
    }

    /// Standalone outcome helper (legacy zero-window / min-yield gate + `YieldGate` suspension counter).
    pub fn judged(&mut self, accepted: usize, cfg: &CopyCfg) {
        self.judged += 1;
        self.zero_streak = if accepted == 0 { self.zero_streak + 1 } else { 0 };
        let e = self.ewma.map_or(accepted as f64, |e| 0.7 * e + 0.3 * accepted as f64);
        self.ewma = Some(e);
        let low = cfg.min_yield > 0.0 && self.judged >= 3 && e < cfg.min_yield;
        if (!cfg.yield_gate && self.zero_streak >= cfg.zero_windows) || low {
            self.silent_for = cfg.silence;
            self.zero_streak = 0;
            self.ewma = None;
            self.judged = 0;
            SILENCED.fetch_add(1, Relaxed);
        }
    }

    pub fn note_gate_transition(&mut self, is_head: bool) {
        let susp = self.span.gate().suspended();
        if susp && !self.was_suspended && is_head {
            SILENCED.fetch_add(1, Relaxed);
        }
        self.was_suspended = susp;
    }
}

/// Server-wide copy drafting state (replicated deterministically on every TP rank).
pub struct CopyDraft {
    pub cfg: CopyCfg,
    pub lanes: Vec<LaneCopy>,
    pub ab_period_s: f64,
    pub lane_price: copy_lanes::Price,
    pub cpv_price: copy_prefill::Price,
    pub record: Option<CopyRecord>,
    pub last_digest: u32,
}

impl CopyDraft {
    pub fn new(cap: usize, cfg: CopyCfg) -> CopyDraft { Self::new_world(cap, cfg, 1) }

    pub fn new_world(cap: usize, cfg: CopyCfg, world: usize) -> CopyDraft {
        let record = if cfg.copy_record {
            let ident = Identity { digest: *b"dsv41-p7copy-session-record-v1.1", vocab: 200_000 };
            let cap_mb = crate::opts::lookup("copy-span-record-mb")
                .and_then(crate::opts::get)
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(64)
                .max(8);
            let dir = if world == 1 {
                crate::opts::lookup("copy-span-record-dir")
                    .and_then(crate::opts::get)
                    .filter(|s| !s.is_empty() && s != "unset")
                    .map(std::path::PathBuf::from)
            } else {
                None // At TP > 1 keep CopyRecord purely session-replicated across ranks so disk state never diverges.
            };
            let opts = RecordOptions { dir, cap_bytes: cap_mb << 20, read_only: false };
            Some(CopyRecord::open(ident, &opts).0)
        } else {
            None
        };
        CopyDraft {
            lanes: (0..cap).map(|_| LaneCopy::new(&cfg)).collect(),
            cfg,
            ab_period_s: ab_period(),
            lane_price: copy_lanes::Price::default(),
            cpv_price: copy_prefill::Price::default(),
            record,
            last_digest: 0,
        }
    }

    pub fn reset_lane(&mut self, lane: usize) {
        if let Some(lc) = self.lanes.get_mut(lane) {
            if let Some(rec) = self.record.as_mut() { lc.span.flush_record(rec); }
            lc.reset_with_cfg(&self.cfg);
        }
    }

    pub fn invalidate_lane(&mut self, lane: usize) {
        if let Some(lc) = self.lanes.get_mut(lane) {
            if let Some(rec) = self.record.as_mut() { lc.span.flush_record(rec); }
            lc.synced = false;
            lc.pending = None;
        }
    }

    pub fn finish_request(&mut self, lane: usize, hist: &[u32]) {
        if let Some(lc) = self.lanes.get_mut(lane) {
            if let Some(rec) = self.record.as_mut() {
                lc.span.flush_record(rec);
                if hist.len() > self.cfg.min_match {
                    let _ = rec.add_prompt(hist);
                }
            }
            lc.reset_with_cfg(&self.cfg);
        }
    }

    /// Delivered batch rate (tok/ms) summed over active lanes that have completed at least one round.
    pub fn batch_rate(&self, active: &[usize]) -> f64 {
        active.iter().filter_map(|&l| self.lanes.get(l).and_then(|lc| lc.work.rate())).sum()
    }

    /// Update `CopyCfg::yield_ratio_x256` from rank-agreed `early_t_host` (`256 * t_copy_start / t_neural`).
    pub fn refresh_yield_ratio(&mut self, early_t: &[f64; 6]) {
        if !self.cfg.yield_gate || early_t[0] <= 0.0 || !early_t[5].is_finite() { return; }
        let t_neural = if early_t[3].is_finite() && early_t[3] > 0.0 { early_t[3] } else { early_t[5] };
        let t_copy = det_round_ms(self.cfg.start_w + 1, false, early_t).max(1.0);
        // YieldGate::ratio_allows checks `ec * 256 >= en * ratio` => `ec / t_copy >= en / t_neural` => `ratio = 256 * t_copy / t_neural`!
        let ratio = ((256.0 * t_copy / t_neural.max(1.0)).round() as i64).clamp(64, 1024);
        self.cfg.yield_ratio_x256 = ratio;
    }

    /// Deterministic zero-RDMA proposer run identically on every TP rank.
    #[allow(clippy::too_many_arguments)]
    pub fn propose_round(&mut self, nl: usize, active: &[usize], m_lanes: &[super::fwd::Lane], host: &[DsLane],
                         pen_active: &[bool], eos: u32, max_rows: usize, early_t: &[f64; 6], is_head: bool) -> Vec<Vec<u32>> {
        let t_start = is_head.then(std::time::Instant::now);
        self.refresh_yield_ratio(early_t);
        let na = active.len();
        let batch_rate = self.batch_rate(active);
        let mut out = vec![Vec::new(); nl];

        // 1. Sync contexts & peek candidate offers per active lane.
        let mut cand: Vec<(usize, usize, bool)> = Vec::new(); // (lane, w_cap, cursor)
        for &l in active {
            let pending_tok = host[l].tok as u32;
            let lc = &mut self.lanes[l];
            lc.pending = None;
            lc.sync_lane(&m_lanes[l].hist, pending_tok, &self.cfg);
            if lc.silent_for > 0 { lc.silent_for -= 1; continue; }
            let seq_room = m_lanes[l].max_seq.saturating_sub(m_lanes[l].pos + 3);
            let pen_cap = if pen_active.get(l).copied().unwrap_or(false) { LEGACY_MAX_COPY } else { self.cfg.max_w };
            let w_cap = self.cfg.max_w.min(pen_cap).min(seq_room);
            if w_cap < MIN_COPY { continue; }

            let peek = lc.span.peek(w_cap).filter(|&(len, _)| len >= MIN_COPY);
            if self.cfg.copy_lanes && na >= 2 {
                let facts = copy_lanes::Facts {
                    spec_on: true,
                    depth: w_cap,
                    peek,
                    wide: peek.map_or(false, |(len, _)| len > LEGACY_MAX_COPY),
                    lane_rate: lc.work.rate(),
                    lane_n: lc.work.n,
                    batch: batch_rate,
                    priced: true,
                    stretch: false,
                };
                match copy_lanes::decide(&self.lane_price, &facts) {
                    Verdict::Take { .. } => {
                        if let Some((_, cur)) = peek { cand.push((l, w_cap, cur)); }
                    }
                    Verdict::Skip(copy_lanes::Skip::PricedOut) => {
                        if is_head { LANES_PRICED_OUT.fetch_add(1, Relaxed); }
                    }
                    Verdict::Skip(_) => {
                        if peek.is_none() && self.record.is_some() && batch_rate <= 0.0 {
                            cand.push((l, w_cap.min(self.cfg.start_w), false));
                        }
                    }
                }
            } else if let Some((_, cur)) = peek {
                cand.push((l, w_cap, cur));
            } else if self.record.is_some() {
                cand.push((l, w_cap.min(self.cfg.start_w), false));
            }
        }

        // 2. Multi-lane row budgeting: keep co-scheduled drafting lanes inside the <= 16-row sweet spot when mixed,
        //    and distribute verify rows evenly when all active lanes copy.
        let c_lanes = cand.len();
        let d_lanes = na.saturating_sub(c_lanes);
        let row_budget = max_rows.max(WIDE_RPL).min(super::fwd::HEAD_MAX);
        for (l, w_cap, is_cursor) in cand {
            let eff_w = if na == 1 {
                w_cap
            } else if d_lanes == 0 {
                (row_budget / c_lanes.max(1)).saturating_sub(1).clamp(MIN_COPY, w_cap)
            } else if is_cursor && self.cfg.copy_lanes {
                let slack16 = WIDE_RPL.saturating_sub(4 * d_lanes);
                let per_copy = (slack16 / c_lanes.max(1)).saturating_sub(1).max(LEGACY_MAX_COPY);
                per_copy.min(w_cap)
            } else {
                w_cap.min(LEGACY_MAX_COPY)
            };
            if eff_w < MIN_COPY { continue; }
            let lc = &mut self.lanes[l];
            if let Some(mut prop) = lc.span.propose_with(eff_w, self.record.as_mut()) {
                trim_before_eos(&mut prop.tokens, eos);
                if prop.tokens.len() >= MIN_COPY {
                    if is_head {
                        match prop.source {
                            Source::Cursor => { CURSOR_PROPOSALS.fetch_add(1, Relaxed); }
                            Source::Search { .. } => { SEARCH_PROPOSALS.fetch_add(1, Relaxed); }
                            Source::Record { .. } => { RECORD_PROPOSALS.fetch_add(1, Relaxed); }
                        }
                        if prop.tokens.len() < w_cap { NARROW_PROPOSALS.fetch_add(1, Relaxed); }
                    }
                    out[l] = prop.tokens.clone();
                    lc.pending = Some(prop);
                }
            }
        }

        // 3. Deterministic proposal digest folded into `Guard::agree("serve round")`.
        let mut dg = 0x811C_9DC5u32;
        for (l, v) in out.iter().enumerate() {
            dg = fold_u32(dg, (l as u32) ^ ((v.len() as u32) << 8));
            for &t in v { dg = fold_u32(dg, t); }
        }
        self.last_digest = dg;

        if let Some(t) = t_start {
            PROPOSE_US.fetch_add(t.elapsed().as_micros() as u64, Relaxed);
            PROPOSE_CALLS.fetch_add(na as u64, Relaxed);
        }
        out
    }

    /// Commit round outcomes into `CopySpan`, `YieldGate`, `COPYLANES` `Price`/`Work`, and `COPYPF1` `Price` identically on all ranks.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_round(&mut self, world: usize, active: &[usize], copies: &[Vec<u32>], cs: &[usize],
                        out_lanes: &[DsLane], emitted: &[Vec<u32>], wall_ms: f64, ran_draft: bool, rows: usize,
                        early_t: &[f64; 6], is_head: bool) {
        let round_ms = if world == 1 && wall_ms > 0.0 { wall_ms } else { det_round_ms(rows, ran_draft, early_t) };
        let share_ms = (round_ms / active.len().max(1) as f64).max(0.05);
        let solo = active.len() == 1;
        for (i, &l) in active.iter().enumerate() {
            let em = &emitted[l];
            let c = cs[i];
            let acc = out_lanes[l].n_acc.max(0) as usize;
            let lc = &mut self.lanes[l];
            if !copies[l].is_empty() {
                lc.work.note(em.len(), share_ms);
                self.lane_price.work.note(em.len(), share_ms);
                let (was_cursor, offered) = match lc.pending.take() {
                    Some(mut p) => {
                        if p.tokens.len() > c { p.tokens.truncate(c); p.max_w = p.max_w.min(c); }
                        let cur = matches!(p.source, Source::Cursor);
                        let off = p.tokens.len();
                        lc.span.commit(Some(&p), acc, em);
                        (cur, off)
                    }
                    None => {
                        lc.span.commit_follow(em);
                        (false, c)
                    }
                };
                if acc < offered { lc.cpv_rung = 0; }
                lc.judged(acc, &self.cfg);
                lc.note_gate_transition(is_head);
                if solo && offered >= MIN_COPY {
                    self.lane_price.note_copy_round(round_ms, offered, acc, was_cursor, offered > LEGACY_MAX_COPY);
                    if acc == offered {
                        self.cpv_price.note_copy_round(round_ms, offered);
                    }
                }
                if is_head {
                    LANE_ROUNDS.fetch_add(1, Relaxed);
                    PROPOSED.fetch_add(c as u64, Relaxed);
                    ACCEPTED.fetch_add(acc as u64, Relaxed);
                    if c > LEGACY_MAX_COPY { WIDE_ROUNDS.fetch_add(1, Relaxed); }
                }
            } else {
                lc.pending = None;
                lc.cpv_rung = 0;
                if ran_draft { lc.span.note_neural_round(em.len()); }
                lc.span.commit_follow(em);
                lc.work.note(em.len(), share_ms);
            }
        }
    }
}

pub(super) fn trim_before_eos(tokens: &mut Vec<u32>, eos: u32) {
    if let Some(pos) = tokens.iter().position(|&t| t == eos) {
        tokens.truncate(pos);
    }
}

pub fn fold_u32(h: u32, v: u32) -> u32 { h.wrapping_mul(0x0100_0193) ^ v }

/// Rank-deterministic round latency (ms) derived from `early_t_host` (broadcast to every rank at boot calibration).
pub fn det_round_ms(rows: usize, ran_draft: bool, early_t: &[f64; 6]) -> f64 {
    let (fixed, slope, draft_ms) = if early_t[0] > 0.0 {
        let sl = if early_t[5].is_finite() && early_t[5] > early_t[0] {
            ((early_t[5] - early_t[0]) / 5.0).max(0.4)
        } else {
            2.4
        };
        ((early_t[0] - 0.5).max(1.0), sl, 3.3)
    } else {
        (24.0, 2.5, 3.5)
    };
    let cliff = if rows > WIDE_RPL { 4.5 } else { 0.0 };
    fixed + slope * rows.saturating_sub(1) as f64 + cliff + if ran_draft { draft_ms } else { 0.0 }
}

/// `--dsv41-copy-ab-period-s` (a measurement switch, 0 = off).
pub fn ab_period() -> f64 {
    crate::opts::lookup("dsv41-copy-ab-period-s").and_then(crate::opts::get).and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(0.0).max(0.0)
}

/// True only when `--dsv41-copy-draft on` was explicitly requested (used for the `--mtp off` boot notice).
pub fn explicitly_on() -> bool {
    crate::opts::lookup("dsv41-copy-draft").and_then(crate::opts::get).as_deref() == Some("on")
}

/// `--dsv41-copy-draft`: Some(config) when copy drafting is on (`auto` / `on`; `off` disables).
pub fn from_flags() -> anyhow::Result<Option<CopyCfg>> {
    let get = |f: &str| crate::opts::lookup(f).and_then(crate::opts::get);
    let on = match get("dsv41-copy-draft").as_deref().unwrap_or("auto") {
        "on" | "auto" => true,
        "off" => false,
        x => anyhow::bail!("--dsv41-copy-draft {x} (off | on | auto)"),
    };
    if !on { return Ok(None); }
    let mut c = CopyCfg::default();
    let num = |f: &str| -> anyhow::Result<Option<f64>> {
        get(f).map(|v| v.trim().parse::<f64>().map_err(|e| anyhow::anyhow!("--{f} {v}: {e}"))).transpose()
    };
    if let Some(v) = num("dsv41-copy-min-match")? { c.min_match = (v as usize).clamp(4, 64); }
    if let Some(v) = num("dsv41-copy-max-w")? { c.max_w = (v as usize).clamp(1, MAX_COPY); }
    if let Some(v) = num("dsv41-copy-start-w")? { c.start_w = (v as usize).clamp(1, MAX_COPY); }
    c.start_w = c.start_w.min(c.max_w);
    if let Some(v) = get("dsv41-copy-yield-gate") { c.yield_gate = v != "off"; }
    if let Some(v) = get("dsv41-copy-lanes") { c.copy_lanes = v != "off"; }
    if let Some(v) = get("dsv41-copy-record") { c.copy_record = v == "on"; }
    if let Some(v) = get("dsv41-copy-prefill") { c.prefill = matches!(v.as_str(), "on" | "auto"); }
    if let Some(v) = num("dsv41-copy-prefill-max-rows")? { c.prefill_max_rows = (v as usize).clamp(32, 4096); }
    if let Some(v) = get("dsv41-copy-prefill-price") { c.prefill_price = v != "off"; }
    if let Some(v) = num("dsv41-copy-zero-windows")? { c.zero_windows = (v as u32).max(1); }
    if let Some(v) = num("dsv41-copy-silence")? { c.silence = v as u32; }
    if let Some(v) = num("dsv41-copy-min-yield")? { c.min_yield = v.max(0.0); }
    Ok(Some(c))
}

// ------------------------------------------------------------------------------------------------ counters (rank 0)

pub static ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static LANE_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static PROPOSED: AtomicU64 = AtomicU64::new(0);
pub static ACCEPTED: AtomicU64 = AtomicU64::new(0);
pub static ALL_COPY_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static DRAFT_SKIPPED: AtomicU64 = AtomicU64::new(0);
pub static SILENCED: AtomicU64 = AtomicU64::new(0);
pub static PROPOSE_US: AtomicU64 = AtomicU64::new(0);
pub static PROPOSE_CALLS: AtomicU64 = AtomicU64::new(0);
pub static AB_ON_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static AB_OFF_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static CURSOR_PROPOSALS: AtomicU64 = AtomicU64::new(0);
pub static SEARCH_PROPOSALS: AtomicU64 = AtomicU64::new(0);
pub static NARROW_PROPOSALS: AtomicU64 = AtomicU64::new(0);
pub static WIDE_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static LANES_PRICED_OUT: AtomicU64 = AtomicU64::new(0);
pub static RECORD_PROPOSALS: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_STEPS: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_OFFERED: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_ACCEPTED: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_RERUN_ROWS: AtomicU64 = AtomicU64::new(0);
pub static ENABLED: AtomicU64 = AtomicU64::new(0);
pub static MAX_W_GAUGE: AtomicU64 = AtomicU64::new(0);

pub fn set_enabled(cfg: Option<&CopyCfg>) {
    ENABLED.store(cfg.map_or(0, |c| 1 + c.min_match as u64), Relaxed);
    MAX_W_GAUGE.store(cfg.map_or(0, |c| c.max_w as u64), Relaxed);
}

pub fn render(o: &mut String) {
    let en = ENABLED.load(Relaxed);
    if en == 0 { return; }
    let g = |o: &mut String, name: &str, help: &str, v: u64| o.push_str(&format!("# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}\n"));
    let c = |o: &mut String, name: &str, help: &str, v: u64| o.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}\n"));
    g(o, "velogb10_dsv41_copy_draft_min_match", "Copy drafting (--dsv41-copy-draft on) is active; the backward match a proposal needs (tokens).", en - 1);
    g(o, "velogb10_dsv41_copy_max_w", "Maximum copied tokens verified per single-lane decode round (--dsv41-copy-max-w).", MAX_W_GAUGE.load(Relaxed));
    c(o, "velogb10_dsv41_copy_rounds_total", "Speculation rounds in which at least one lane verified a copy proposal instead of DSpark drafts.", ROUNDS.load(Relaxed));
    c(o, "velogb10_dsv41_copy_lane_rounds_total", "Lane-rounds that verified a copy proposal.", LANE_ROUNDS.load(Relaxed));
    c(o, "velogb10_dsv41_copy_proposed_total", "Copied tokens verified (proposal rows, the pending row excluded).", PROPOSED.load(Relaxed));
    c(o, "velogb10_dsv41_copy_accepted_total", "Copied tokens kept by the acceptance rule (greedy: equal to the target's argmax; sampled: equal to the target's own keyed draw).", ACCEPTED.load(Relaxed));
    c(o, "velogb10_dsv41_copy_all_copy_rounds_total", "Rounds in which every active lane took a copy proposal.", ALL_COPY_ROUNDS.load(Relaxed));
    c(o, "velogb10_dsv41_copy_draft_pass_skipped_total", "DSpark draft passes skipped because every active lane of the round took a copy proposal.", DRAFT_SKIPPED.load(Relaxed));
    c(o, "velogb10_dsv41_copy_silenced_total", "Times the yield gate silenced a lane's copy proposals.", SILENCED.load(Relaxed));
    c(o, "velogb10_dsv41_copy_cursor_proposals_total", "Copy proposals served from a locked O(1) CopySpan cursor (zero search).", CURSOR_PROPOSALS.load(Relaxed));
    c(o, "velogb10_dsv41_copy_search_proposals_total", "Copy proposals served from a multi-length n-gram search.", SEARCH_PROPOSALS.load(Relaxed));
    c(o, "velogb10_dsv41_copy_narrow_proposals_total", "Copy proposals narrowed by the start-width ramp (4 -> 8 -> 15).", NARROW_PROPOSALS.load(Relaxed));
    c(o, "velogb10_dsv41_copy_wide_rounds_total", "Decode copy lane-rounds verifying > 5 copied tokens (7..=16 verify rows, WIDEVERIFY1).", WIDE_ROUNDS.load(Relaxed));
    c(o, "velogb10_dsv41_copy_lanes_priced_out_total", "Multi-lane copy offers declined by COPYLANES 1.15x batch-rate pricing.", LANES_PRICED_OUT.load(Relaxed));
    c(o, "velogb10_dsv41_copy_record_proposals_total", "Cross-request CopyRecord fallback proposals (--dsv41-copy-record on).", RECORD_PROPOSALS.load(Relaxed));
    c(o, "velogb10_dsv41_copy_prefill_steps_total", "Prefill-width copy stretches executed (--dsv41-copy-prefill on, COPYPF1).", PREFILL_STEPS.load(Relaxed));
    c(o, "velogb10_dsv41_copy_prefill_offered_total", "Copied tokens offered across prefill-width stretches.", PREFILL_OFFERED.load(Relaxed));
    c(o, "velogb10_dsv41_copy_prefill_accepted_total", "Copied tokens accepted across prefill-width stretches.", PREFILL_ACCEPTED.load(Relaxed));
    c(o, "velogb10_dsv41_copy_prefill_rerun_rows_total", "Rows re-advanced after an early mismatch inside a prefill-width stretch.", PREFILL_RERUN_ROWS.load(Relaxed));
    o.push_str("# HELP velogb10_dsv41_copy_propose_seconds_total Host time rank 0 spent building proposals (index extension + lookup), on the round's critical path.\n# TYPE velogb10_dsv41_copy_propose_seconds_total counter\n");
    o.push_str(&format!("velogb10_dsv41_copy_propose_seconds_total {:.6}\n", PROPOSE_US.load(Relaxed) as f64 * 1e-6));
    if AB_ON_ROUNDS.load(Relaxed) + AB_OFF_ROUNDS.load(Relaxed) > 0 {
        c(o, "velogb10_dsv41_copy_ab_on_rounds_total", "MEASUREMENT SWITCH (--dsv41-copy-ab-period-s): rounds served while copy proposals were permitted.", AB_ON_ROUNDS.load(Relaxed));
        c(o, "velogb10_dsv41_copy_ab_off_rounds_total", "MEASUREMENT SWITCH (--dsv41-copy-ab-period-s): rounds served while copy proposals were not permitted (the baseline arm).", AB_OFF_ROUNDS.load(Relaxed));
    }
    c(o, "velogb10_dsv41_copy_propose_calls_total", "Proposal lookups (one per active lane per round while copy drafting is on).", PROPOSE_CALLS.load(Relaxed));
}

// ------------------------------------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::dsv41::gpu::dspark::vocab::{counter, gumbel, key};

    fn cfg(min_match: usize) -> CopyCfg {
        CopyCfg { min_match, start_w: MAX_COPY, yield_gate: false, ..Default::default() }
    }

    #[test]
    fn dsv41_cd1_proposes_the_continuation_of_the_longest_recent_match() {
        let phrase: Vec<u32> = (100..120).collect();
        let mut ctx: Vec<u32> = vec![1, 2, 3];
        ctx.extend(&phrase);
        ctx.extend([7, 8, 9]);
        ctx.extend(&phrase[..14]);
        let mut lc = LaneCopy::new(&cfg(12));
        let p = lc.propose(&ctx, &cfg(12), 5);
        assert_eq!(p, phrase[14..19].to_vec());
        assert!(LaneCopy::new(&cfg(15)).propose(&ctx, &cfg(15), 5).is_empty());
        assert_eq!(LaneCopy::new(&cfg(12)).propose(&ctx, &cfg(12), 3), phrase[14..17].to_vec());
        assert!(LaneCopy::new(&cfg(12)).propose(&ctx, &cfg(12), 1).is_empty());
    }

    #[test]
    fn dsv41_cd1_the_index_follows_a_growing_context_and_rebuilds_on_a_changed_history() {
        let phrase: Vec<u32> = (500..530).collect();
        let mut ctx: Vec<u32> = phrase.clone();
        ctx.extend([1, 2, 3, 4]);
        let c = cfg(12);
        let mut lc = LaneCopy::new(&c);
        let mut served = Vec::new();
        for i in 0..14 { ctx.push(phrase[i]); if i >= 11 { served.push(lc.propose(&ctx, &c, 5)); } }
        assert_eq!(served[0], phrase[12..17].to_vec());
        assert_eq!(served[2], phrase[14..19].to_vec());
        let other: Vec<u32> = (900..960).collect();
        assert!(lc.propose(&other, &c, 5).is_empty());
        assert_eq!(lc.propose(&ctx, &c, 5), phrase[14..19].to_vec());
    }

    #[test]
    fn dsv41_cd1_the_yield_gate_silences_a_lane_after_zero_accept_windows_and_lets_it_back() {
        let c = CopyCfg { zero_windows: 4, silence: 3, min_match: 12, min_yield: 0.0, yield_gate: false, start_w: 5, ..Default::default() };
        let phrase: Vec<u32> = (100..140).collect();
        let mut ctx = phrase.clone();
        ctx.extend([1, 2, 3]);
        ctx.extend(&phrase[..20]);
        let mut lc = LaneCopy::new(&c);
        assert!(!lc.propose(&ctx, &c, 5).is_empty());
        for _ in 0..3 { lc.judged(0, &c); assert!(lc.silent_for == 0); }
        lc.judged(0, &c);
        assert_eq!(lc.silent_for, 3);
        for _ in 0..3 { assert!(lc.propose(&ctx, &c, 5).is_empty()); }
        assert!(!lc.propose(&ctx, &c, 5).is_empty());
        for _ in 0..3 { lc.judged(0, &c); }
        lc.judged(2, &c);
        lc.judged(0, &c);
        assert_eq!(lc.silent_for, 0);
        let c2 = CopyCfg { min_yield: 1.5, ..c };
        let mut lc2 = LaneCopy::new(&c2);
        lc2.judged(1, &c2); lc2.judged(1, &c2);
        assert_eq!(lc2.silent_for, 0);
        lc2.judged(1, &c2);
        assert_eq!(lc2.silent_for, 3);
    }

    #[test]
    fn dsv41_p7copy_cursor_locks_on_repetitive_jsonl_and_ramps_4_8_15() {
        let c = CopyCfg::default();
        let header: Vec<u32> = (1000..1020).collect();
        let payload1: Vec<u32> = (2000..2060).collect();
        let payload2: Vec<u32> = (3000..3060).collect();
        let mut ctx = vec![42u32];
        ctx.extend(&header);
        ctx.extend(&payload1);
        ctx.push(77);
        ctx.extend(&header);
        ctx.extend(&payload2);
        ctx.push(88);
        let mut unique1 = vec![42u32];
        unique1.extend(&header[..15]);
        ctx.extend(&unique1);

        let mut lc = LaneCopy::new(&c);
        lc.span.reset(&ctx);

        // Round 1: Search hit -> capped by start_w = 4 (`[1015..1019]`), bonus = 1019 (continues source!)
        let p1 = lc.span.propose(15).expect("search proposal");
        assert!(matches!(p1.source, Source::Search { .. }));
        assert_eq!(p1.tokens, vec![1015, 1016, 1017, 1018]);
        lc.span.commit(Some(&p1), 4, &[1015, 1016, 1017, 1018, 1019]);
        assert!(lc.span.cursor().is_some(), "cursor must lock when bonus continues source");

        // Round 2: Cursor hit -> stays on record #1 (`payload1 = 2000..`) and doubles width to 8!
        let p2 = lc.span.propose(15).expect("cursor proposal");
        assert_eq!(p2.source, Source::Cursor);
        assert_eq!(p2.tokens, (2000..2008).collect::<Vec<u32>>());
        let mut em2 = p2.tokens.clone();
        em2.push(2008);
        lc.span.commit(Some(&p2), 8, &em2);

        // Round 3: Cursor hit -> doubles again, clamped to max_w = 15 (`16` verify rows)!
        let p3 = lc.span.propose(15).expect("wide cursor proposal");
        assert_eq!(p3.source, Source::Cursor);
        assert_eq!(p3.tokens.len(), 15);
        assert_eq!(p3.tokens, (2009..2024).collect::<Vec<u32>>());

        // Partial mismatch at token 5 drops cursor!
        let mut em3: Vec<u32> = (2009..2014).collect();
        em3.push(9999);
        lc.span.commit(Some(&p3), 5, &em3);
        assert!(lc.span.cursor().is_none());
    }

    #[test]
    fn dsv41_p7copy_cost_ratio_yield_gate_suspends_paraphrase_and_probes_back() {
        let c = CopyCfg { yield_ratio_x256: 220, ..CopyCfg::default() };
        let block: Vec<u32> = (500..560).collect();
        let mut ctx = block.clone();
        ctx.push(999);
        ctx.extend(&block[..16]);

        let mut lc = LaneCopy::new(&c);
        lc.span.reset(&ctx);
        // Calibrate neural draft yield (3 tokens/round across 4 neural rounds)
        for i in 0..4 {
            lc.span.note_neural_round(3);
            lc.span.commit_follow(&[7000 + i * 3, 7001 + i * 3, 7002 + i * 3]);
        }
        // Feed matching prefix again, but model paraphrases right after the match (`accepted = 0`):
        lc.span.commit_follow(&block[..16]);
        for round in 0..3 {
            if let Some(p) = lc.span.propose(15) {
                lc.span.commit(Some(&p), 0, &[8000 + round]);
                lc.span.commit_follow(&block[..16]);
            }
        }
        assert!(lc.span.gate().suspended(), "cost-ratio YieldGate must suspend failing copy starts");

        // While suspended, 15 of 16 matching rounds are suppressed; the 16th probes and recovers on high yield!
        let mut suppressed = 0usize;
        let mut probed = false;
        for step in 0..20 {
            if let Some(p) = lc.span.propose(15) {
                probed = true;
                let mut em = p.tokens.clone();
                em.push(block[16 + p.tokens.len()]);
                lc.span.commit(Some(&p), p.tokens.len(), &em);
                // Continue cursor with 15-wide accepts to lift `ec` above the neural bar:
                for _ in 0..2 {
                    if let Some(pc) = lc.span.propose(15) {
                        let next_b = block[(16 + 5 + 15).min(block.len() - 1)];
                        let mut emc = pc.tokens.clone();
                        emc.push(next_b);
                        lc.span.commit(Some(&pc), pc.tokens.len(), &emc);
                    }
                }
                break;
            } else {
                suppressed += 1;
                lc.span.commit_follow(&[9000 + step as u32]);
                lc.span.commit_follow(&block[..16]);
            }
        }
        assert!(suppressed >= 10 && probed, "expected suspended rounds before 1-in-16 probe (suppressed={suppressed})");
    }

    #[test]
    fn dsv41_p7copy_copylanes_prices_out_weak_copy_and_four_ranks_agree_zero_rdma() {
        let c = CopyCfg { copy_record: true, ..CopyCfg::default() };
        let early_t = [18.0, 20.4, 22.8, 25.2, 27.6, 30.0];
        let mut ranks: Vec<CopyDraft> = (0..4).map(|_| CopyDraft::new_world(4, c, 4)).collect();

        let phrase: Vec<u32> = (100..220).collect();
        let mut hist0 = phrase.clone();
        hist0.push(999);
        hist0.extend(&phrase[..15]);
        let hist1: Vec<u32> = (5000..5080).collect();

        let make_lanes = |h0: &[u32], h1: &[u32]| -> Vec<crate::dsv41::tp::fwd::Lane> {
            vec![h0, h1].into_iter().map(|h| crate::dsv41::tp::fwd::Lane {
                pos: h.len(), win: Vec::new(), src: HashMap::new(),
                hash: crate::dsv41::engram::hash::NgramHashState::new(),
                cap_rows: 1024, max_seq: 4096, hist: h.to_vec(), temp: 0.0, seed: 7,
                kind: crate::dsv41::ced::schedule::StateKind::Full,
                vl_types: Vec::new(), vl_row_ops: Vec::new(), vl_aligner: None,
            }).collect()
        };

        let mut m_lanes = make_lanes(&hist0, &hist1);
        let mut host = vec![DsLane::default(); 2];
        host[0].active = 1; host[0].tok = phrase[15] as i32;
        host[1].active = 1; host[1].tok = 6000;
        let active = [0usize, 1];
        let pen = [false, false];

        for step in 0..40u32 {
            let props: Vec<Vec<Vec<u32>>> = ranks.iter_mut().enumerate().map(|(r, cd)| {
                cd.propose_round(2, &active, &m_lanes, &host, &pen, 1, 24, &early_t, r == 0)
            }).collect();
            for r in 1..4 {
                assert_eq!(props[0], props[r], "rank {r} diverged on step {step}");
                assert_eq!(ranks[0].last_digest, ranks[r].last_digest, "digest mismatch on step {step}");
            }
            let c0 = props[0][0].len();
            let c1 = if props[0][1].is_empty() { 3 } else { props[0][1].len() };
            let cs = [c0, c1];
            let em0: Vec<u32> = if c0 > 0 && step < 10 {
                let mut v = props[0][0].clone();
                let next_idx = (16 + m_lanes[0].hist.len() - hist0.len() + c0) % phrase.len();
                v.push(phrase[next_idx]);
                v
            } else {
                vec![7000 + step]
            };
            let em1 = vec![6001 + step * 3, 6002 + step * 3, 6003 + step * 3];
            let mut out_lanes = host.clone();
            out_lanes[0].n_acc = if c0 > 0 && step < 10 { c0 as i32 } else { 0 };
            out_lanes[1].n_acc = 2;
            let emitted = vec![em0.clone(), em1.clone()];
            let rows = 2 + c0 + c1;
            for (r, cd) in ranks.iter_mut().enumerate() {
                cd.finish_round(4, &active, &props[r], &cs, &out_lanes, &emitted, 35.0 + r as f64 * 0.7, true, rows, &early_t, r == 0);
            }
            m_lanes[0].hist.push(host[0].tok as u32);
            m_lanes[0].hist.extend_from_slice(&em0[..em0.len() - 1]);
            m_lanes[0].pos = m_lanes[0].hist.len();
            host[0].tok = *em0.last().unwrap() as i32;

            m_lanes[1].hist.push(host[1].tok as u32);
            m_lanes[1].hist.extend_from_slice(&em1[..em1.len() - 1]);
            m_lanes[1].pos = m_lanes[1].hist.len();
            host[1].tok = *em1.last().unwrap() as i32;
        }
        // Finish request on all 4 ranks and confirm cross-request CopyRecord proposes identically on a new turn!
        for cd in &mut ranks { cd.finish_request(0, &m_lanes[0].hist); }
        m_lanes[0].hist = phrase[..15].to_vec();
        m_lanes[0].pos = 15;
        host[0].tok = phrase[15] as i32;
        let rec_props: Vec<Vec<Vec<u32>>> = ranks.iter_mut().enumerate().map(|(r, cd)| {
            cd.propose_round(1, &[0], &m_lanes[..1], &host[..1], &[false], 1, 16, &early_t, r == 0)
        }).collect();
        assert!(!rec_props[0][0].is_empty(), "CopyRecord must propose from previous completed turn");
        for r in 1..4 { assert_eq!(rec_props[0], rec_props[r]); }
    }

    // ---- exactness of the acceptance rule across widths 1..=15 (`WIDEVERIFY1`) and stretches 32, 160 (`COPYPF1`) -----------

    const V: usize = 6;
    fn q_of(prev: usize) -> [f64; V] {
        let mut w = [0f64; V];
        for (i, x) in w.iter_mut().enumerate() { *x = 1.0 + ((prev * 7 + i * 3) % 5) as f64 + if i == (prev + 1) % V { 4.0 } else { 0.0 }; }
        let s: f64 = w.iter().sum();
        w.map(|x| x / s)
    }
    fn draw(dist: &[f64; V], k: u64) -> usize {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for (id, &w) in dist.iter().enumerate() { if w > 0.0 { let s = w.ln() + gumbel(k, id as u32); if s > best.0 { best = (s, id); } } }
        best.1
    }
    fn plain(seed: u64, x0: usize, pos0: i64, n: usize) -> Vec<usize> {
        let (mut prev, mut out) = (x0, Vec::new());
        for i in 0..n { let t = draw(&q_of(prev), key(seed, counter(pos0 + 1 + i as i64, 0))); out.push(t); prev = t; }
        out
    }
    fn proposal(prev: usize, round: u32, len: usize) -> Vec<usize> {
        let mut v = Vec::new();
        let mut p = prev;
        for j in 0..len { p = (p * 5 + round as usize * 3 + j * 2 + 1) % V; v.push(p); }
        v
    }
    fn copy_round(seed: u64, prev0: usize, pos: i64, prop: &[usize]) -> Vec<usize> {
        let (mut prev, mut out) = (prev0, Vec::new());
        for (n, &d) in prop.iter().enumerate() {
            let g = draw(&q_of(prev), key(seed, counter(pos + 1 + n as i64, 0)));
            if g == d { out.push(d); prev = d; } else { out.push(g); return out; }
        }
        out.push(draw(&q_of(prev), key(seed, counter(pos + 1 + prop.len() as i64, 0))));
        out
    }
    fn biased_round(seed: u64, prev0: usize, pos: i64, prop: &[usize]) -> Vec<usize> {
        let (mut prev, mut out) = (prev0, Vec::new());
        for (n, &d) in prop.iter().enumerate() {
            let tp = pos + 1 + n as i64;
            let q = q_of(prev);
            if crate::dsv41::gpu::dspark::vocab::uniform(seed, counter(tp, 2)) < q[d] { out.push(d); prev = d; } else { out.push(draw(&q, key(seed, counter(tp, 1)))); return out; }
        }
        out.push(draw(&q_of(prev), key(seed, counter(pos + 1 + prop.len() as i64, 0))));
        out
    }

    #[test]
    fn dsv41_cd1_copy_rounds_emit_exactly_the_plain_decode_tokens_of_the_same_seed() {
        const WIDTHS: [usize; 17] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 159];
        let mut wide_accepts = 0usize;
        let mut stretch_accepts = 0usize;
        for t in 0..4000u64 {
            let seed = crate::dsv41::gpu::dspark::vocab::mix(t + 17);
            let (x0, pos0, want) = (t as usize % V, 100i64, 200usize);
            let reference = plain(seed, x0, pos0, want);
            let (mut got, mut prev, mut pos, mut round) = (Vec::new(), x0, pos0, 0u32);
            while got.len() < want {
                let len = WIDTHS[(round as usize + t as usize) % WIDTHS.len()];
                let prop = match round % 3 {
                    0 => proposal(prev, round, len),
                    1 => plain(seed, prev, pos, len),
                    _ => {
                        let mut p = plain(seed, prev, pos, len);
                        let flip = (t as usize + round as usize) % len;
                        p[flip] = (p[flip] + 1) % V;
                        p
                    }
                };
                let em = copy_round(seed, prev, pos, &prop);
                if len > 5 && em.len() > 6 { wide_accepts += 1; }
                if len >= 31 && em.len() > 16 { stretch_accepts += 1; }
                pos += em.len() as i64;
                prev = *em.last().unwrap();
                got.extend(em);
                round += 1;
            }
            assert_eq!(got[..want], reference[..], "seed {t}: wide/stretch copy rounds differ from plain decode");
        }
        assert!(wide_accepts > 1000 && stretch_accepts > 1000, "wide={wide_accepts}, stretch={stretch_accepts}");
    }

    #[test]
    fn dsv41_cd1_copy_acceptance_gives_the_target_distribution_chi_square() {
        const WIDTHS: [usize; 17] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 159];
        let x0 = 2usize;
        let n = 400_000u64;
        let run = |biased: bool, salt: u64| -> f64 {
            let mut h: HashMap<i32, usize> = HashMap::new();
            for t in 0..n {
                let seed = crate::dsv41::gpu::dspark::vocab::mix(t ^ (salt << 40));
                let mut out: Vec<usize> = Vec::new();
                let (mut prev, mut pos, mut round) = (x0, 100i64, 0u32);
                while out.len() < 2 {
                    let len = WIDTHS[(t as usize + round as usize) % WIDTHS.len()];
                    let prop = proposal(prev, round, len);
                    let em = if biased { biased_round(seed, prev, pos, &prop) } else { copy_round(seed, prev, pos, &prop) };
                    pos += em.len() as i64; prev = *em.last().unwrap(); out.extend(em); round += 1;
                }
                *h.entry((out[0] * V + out[1]) as i32).or_default() += 1;
            }
            let probs: Vec<(u32, f64)> = (0..V * V).map(|i| (i as u32, q_of(x0)[i / V] * q_of(i / V)[i % V])).collect();
            crate::dsv41::gpu::dspark::chisq::one_sample(&h, &probs).2
        };
        let p_exact = run(false, 1);
        assert!(p_exact > 1e-3, "wide/stretch copy acceptance is not exact: chi-square p = {p_exact}");
        let p_biased = run(true, 2);
        assert!(p_biased < 1e-6, "the control (reads the draw it decides on) was not rejected: p = {p_biased}");
        println!("dsv41_p7copy chi-square (widths 1..=15, 32, 160): exact arm p = {p_exact:.4}, control p = {p_biased:.3e} (N = {n})");
    }
}
