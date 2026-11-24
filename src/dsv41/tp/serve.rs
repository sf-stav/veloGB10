//! DSV4.1 TP (leg SE): the serving program. The head runs the lane loop (`serve::engine`) and the HTTP server; every
//! rank holds the same `TpServe` (TP trunk + DSpark stages + prefix states) and executes the same ops in the same order.
//! The head decides, then broadcasts a 16-float header (op + small integers, each an exact float: ids and positions
//! below 2^24, 32-bit values as two 16-bit halves) over one fp32 all-reduce; the nodes run `follow`, which reads the
//! header, takes the payload (token ids: `TpModel::bcast_tokens`) and runs the SAME `exec`. Every op ends with the
//! per-step agree guard (AGENTS §2.10). While the head has nothing to do it sends `Nop` every 250 ms, so a node never
//! waits in a collective longer than the transport watchdog allows; after `--dsv41-idle-park-ms` without a request it sends
//! `Park` instead and every rank leaves the collectives until the head's next op (leg HT1, `idle.rs`).
//!
//! Ops: `Prefill` (CE's exact CED prefill of the lane's new tokens; a lane holding a restored state resumes), `Start`
//! (seed the drafter from the prompt tail and draw the first token with DS's sampler), `Round` (one DSpark round over
//! the active lanes: draft, verify, DeepSeek's acceptance rule), `Release`, `Snap` / `Restore` / `Free` (the prefix
//! pool's device states, by id), `Truncate` (roll a lane back <= 6 rows), `Score` (an exact every-layer prefill and the
//! per-token log-probs, GA's G1), `Nop`, `Park` (leg HT1: the nodes block on the head's control connection until its next
//! op), `Shutdown`.
//!
//! The one sampler is DS's (D-DSV41-5): every token of a request, the first included, is drawn inside a round (`Start`
//! is a k = 0 round of the prompt's last row; a round with k = 0 is plain decoding), on TPC's byte-exact exchange.

use super::comm::{Dsv41Comm, TpComm};
use super::ced::TpSnap;
use super::fwd::{HeadWant, TpModel};
use super::prog::Guard;
use super::spec::TpSpec;
use super::wload::RankSrc;
use crate::dsv41::ced::schedule::{self, Mode as CedMode, StateKind};
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::fwd::dev::Gpu;
use crate::dsv41::gpu::dspark::Lane as DsLane;
use crate::tokenizer::QwenTokenizer;
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::CudaDevice;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The program name the nodes dispatch on (`exl3_mode`).
pub const SERVE_MODE: &str = "dsv41-tp-serve";

/// One command, as the head broadcasts it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Nop,
    Shutdown,
    /// prefill `lane` with the broadcast tokens appended after the lane's own `pos` tokens; `taps`: write the DSpark taps;
    /// `total` = the prompt's full length (one prompt may take several calls: leg SE2, so other lanes decode between them);
    /// `first` = the first call of this prompt; `full_at` (leg CE2) = the lowest mid-prompt position whose state must be full
    /// (the serve shell snapshots there: `Schedule::new_with`), 0 = none; every call of one prompt carries the same value;
    /// `ahead` (leg PP1) = how many of the broadcast tokens, at the end, are NOT this call's rows but the next call's (their
    /// Engram rows are read while this call computes)
    Prefill { lane: usize, taps: bool, total: u32, first: bool, full_at: u32, ahead: u32 },
    /// seed the drafter of `lane` (just prefilled) and draw its first token (`suppress_eos`: mask model stop_ids on row 0)
    Start { lane: usize, temp: f32, seed: u64, pen: [f32; 3], suppress_eos: bool },
    /// one round over lanes `0..nl`; bit l of `mask` = lane l is active; byte l of `sup_mask` = rows `0..n` masking stop_ids;
    /// `obs` = Rank 0's `(wall_ms, gap_ms)` `f32` bits from the previous priced round (leg P7.4 lockstep `cs`)
    Round { nl: usize, mask: u32, copy: bool, sup_mask: u64, obs: Option<(u32, u32)> },
    /// leg SE2 (`thinking_token_budget`): the last round's chain lost its last `drop` tokens (>= 1, the pending one included);
    /// the lane's pending token becomes `tok` (the lane rolls back `drop - 1` rows)
    Force { lane: usize, drop: usize, tok: u32 },
    /// leg SE2: the lane's top-k / top-p for the next `Start` (sent only when the request has one)
    Cut { lane: usize, top_k: u32, top_p: f32 },
    Release { lane: usize },
    Snap { lane: usize, id: u32 },
    Restore { lane: usize, id: u32 },
    Free { id: u32 },
    Truncate { lane: usize, pos: u32 },
    /// exact prefill of the broadcast tokens, per-token log-probs
    Score { lane: usize },
    /// leg HT1: every rank parks (the nodes block on the head's control connection until its wake byte)
    Park,
    /// leg VIS1: set per-token VL metadata (`token_types` over `0..len`, packed 4 bytes per `u32` word) on `lane`
    SetVl { lane: usize, len: u32 },
    /// leg P7.3: batched multi-lane prefill of `count` lanes (`mask` bits in ascending lane order; `taps_mask` writes per-lane DSpark taps)
    PrefillBatch { count: usize, mask: u32, taps_mask: u32 },
    /// leg P7.3: batched multi-lane first-token start of `count` lanes (`mask` bits in ascending lane order; `sup_mask` masks EOS on row 0)
    StartBatch { count: usize, mask: u32, sup_mask: u32 },
}

fn halves(v: u32) -> [f32; 2] { [(v & 0xFFFF) as f32, (v >> 16) as f32] }
fn join(a: f32, b: f32) -> u32 { (a as u32) | ((b as u32) << 16) }

pub fn pack_vl_types(types: &[i8]) -> Vec<u32> {
    let mut out = vec![0u32; types.len().div_ceil(4)];
    for (i, &t) in types.iter().enumerate() {
        let b = ((t as i16) + 1).clamp(0, 255) as u32;
        out[i / 4] |= b << ((i % 4) * 8);
    }
    out
}

pub fn unpack_vl_types(words: &[u32], len: usize) -> Result<Vec<i8>> {
    ensure!(words.len() == len.div_ceil(4), "set_vl: expected {} words for {len} token types, got {}", len.div_ceil(4), words.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let b = ((words[i / 4] >> ((i % 4) * 8)) & 0xFF) as i16 - 1;
        ensure!((-1..=3).contains(&b), "set_vl: invalid token type {b} at {i}");
        out.push(b as i8);
    }
    Ok(out)
}

impl Op {
    pub fn encode(&self) -> [f32; 16] {
        let mut h = [0f32; 16];
        let (code, a): (f32, Vec<f32>) = match *self {
            Op::Nop => (0.0, vec![]),
            Op::Shutdown => (1.0, vec![]),
            Op::Prefill { lane, taps, total, first, full_at, ahead } => {
                let (t, f, a) = (halves(total), halves(full_at), halves(ahead));
                (2.0, vec![lane as f32, taps as u8 as f32, t[0], t[1], first as u8 as f32, f[0], f[1], a[0], a[1]])
            }
            Op::Start { lane, temp, seed, pen, suppress_eos } => {
                let t = halves(temp.to_bits());
                let s = [seed & 0xFFFF, (seed >> 16) & 0xFFFF, (seed >> 32) & 0xFFFF, seed >> 48];
                let mut v = vec![lane as f32, t[0], t[1], s[0] as f32, s[1] as f32, s[2] as f32, s[3] as f32];
                for x in pen { v.extend_from_slice(&halves(x.to_bits())); }
                v.push(suppress_eos as u8 as f32);
                (3.0, v)
            }
            Op::Round { nl, mask, copy, sup_mask, obs } => {
                let sm_lo = halves(sup_mask as u32);
                let sm_hi = halves((sup_mask >> 32) as u32);
                let (ho, w, g) = match obs {
                    Some((wb, gb)) => (1.0, halves(wb), halves(gb)),
                    None => (0.0, [0.0; 2], [0.0; 2]),
                };
                (4.0, vec![nl as f32, mask as f32, copy as u8 as f32, sm_lo[0], sm_lo[1], ho, w[0], w[1], g[0], g[1], sm_hi[0], sm_hi[1]])
            }
            Op::Release { lane } => (5.0, vec![lane as f32]),
            Op::Snap { lane, id } => { let i = halves(id); (6.0, vec![lane as f32, i[0], i[1]]) }
            Op::Restore { lane, id } => { let i = halves(id); (7.0, vec![lane as f32, i[0], i[1]]) }
            Op::Free { id } => { let i = halves(id); (8.0, vec![i[0], i[1]]) }
            Op::Truncate { lane, pos } => { let p = halves(pos); (9.0, vec![lane as f32, p[0], p[1]]) }
            Op::Score { lane } => (10.0, vec![lane as f32]),
            Op::Force { lane, drop, tok } => { let t = halves(tok); (11.0, vec![lane as f32, drop as f32, t[0], t[1]]) }
            Op::Cut { lane, top_k, top_p } => { let k = halves(top_k); let p = halves(top_p.to_bits()); (12.0, vec![lane as f32, k[0], k[1], p[0], p[1]]) }
            Op::Park => (13.0, vec![]),
            Op::SetVl { lane, len } => { let l = halves(len); (14.0, vec![lane as f32, l[0], l[1]]) }
            Op::PrefillBatch { count, mask, taps_mask } => (15.0, vec![count as f32, mask as f32, taps_mask as f32]),
            Op::StartBatch { count, mask, sup_mask } => (16.0, vec![count as f32, mask as f32, sup_mask as f32]),
        };
        h[0] = code;
        for (i, v) in a.iter().enumerate() { h[1 + i] = *v; }
        h
    }

    pub fn decode(h: &[f32]) -> Result<Op> {
        ensure!(h.len() >= 8, "short op header");
        let u = |i: usize| h[i] as usize;
        Ok(match h[0] as u32 {
            0 => Op::Nop,
            1 => Op::Shutdown,
            2 => Op::Prefill { lane: u(1), taps: h[2] != 0.0, total: join(h[3], h[4]), first: h[5] != 0.0, full_at: join(h[6], h[7]),
                               ahead: if h.len() >= 10 { join(h[8], h[9]) } else { 0 } },
            3 => Op::Start { lane: u(1), temp: f32::from_bits(join(h[2], h[3])),
                             seed: (h[4] as u64) | ((h[5] as u64) << 16) | ((h[6] as u64) << 32) | ((h[7] as u64) << 48),
                             pen: [f32::from_bits(join(h[8], h[9])), f32::from_bits(join(h[10], h[11])), f32::from_bits(join(h[12], h[13]))],
                             suppress_eos: h.len() >= 15 && h[14] != 0.0 },
            4 => Op::Round { nl: u(1), mask: h[2] as u32, copy: h[3] != 0.0,
                             sup_mask: (if h.len() >= 6 { join(h[4], h[5]) as u64 } else { 0 })
                                     | (if h.len() >= 13 { (join(h[11], h[12]) as u64) << 32 } else { 0 }),
                             obs: if h.len() >= 11 && h[6] != 0.0 { Some((join(h[7], h[8]), join(h[9], h[10]))) } else { None } },
            5 => Op::Release { lane: u(1) },
            6 => Op::Snap { lane: u(1), id: join(h[2], h[3]) },
            7 => Op::Restore { lane: u(1), id: join(h[2], h[3]) },
            8 => Op::Free { id: join(h[1], h[2]) },
            9 => Op::Truncate { lane: u(1), pos: join(h[2], h[3]) },
            10 => Op::Score { lane: u(1) },
            11 => Op::Force { lane: u(1), drop: u(2), tok: join(h[3], h[4]) },
            12 => Op::Cut { lane: u(1), top_k: join(h[2], h[3]), top_p: f32::from_bits(join(h[4], h[5])) },
            13 => Op::Park,
            14 => Op::SetVl { lane: u(1), len: join(h[2], h[3]) },
            15 => Op::PrefillBatch { count: u(1), mask: h[2] as u32, taps_mask: h[3] as u32 },
            16 => Op::StartBatch { count: u(1), mask: h[2] as u32, sup_mask: h[3] as u32 },
            c => bail!("unknown op code {c} in the serve header (head and node run different builds?)"),
        })
    }

    pub fn name(&self) -> &'static str {
        match self { Op::Nop => "nop", Op::Shutdown => "shutdown", Op::Prefill { .. } => "prefill", Op::Start { .. } => "start", Op::Round { .. } => "round",
                     Op::Release { .. } => "release", Op::Snap { .. } => "snap", Op::Restore { .. } => "restore", Op::Free { .. } => "free",
                     Op::Truncate { .. } => "truncate", Op::Score { .. } => "score", Op::Force { .. } => "force", Op::Cut { .. } => "cut",
                     Op::Park => "park", Op::SetVl { .. } => "set_vl", Op::PrefillBatch { .. } => "prefill_batch", Op::StartBatch { .. } => "start_batch" }
    }
}

/// What an op produced on this rank (identical on every rank except `Scores`, which the head collects).
#[derive(Debug, Default)]
pub struct Done {
    pub tok: u32,
    pub chains: Vec<Vec<u32>>,
    pub scores: Vec<f32>,
    pub snap_pos: usize,
    pub snap_bytes: u64,
    pub snap_full: bool,
    /// leg CE3: the state carries the prompt tail (`TpSnap::tail`)
    pub snap_tail: bool,
    /// per-lane model positions after the op
    pub pos: Vec<usize>,
}

/// What the serve program needs from the options (the head sets the registry before the TpConfig snapshot).
#[derive(Clone, Debug)]
pub struct ServeOpts { pub lanes: usize, pub max_seq: usize, pub chunk: usize, pub cache_mib: usize, pub tail_merge: usize }

fn opt(name: &str) -> Option<String> { crate::opts::lookup(name).and_then(crate::opts::get).filter(|s| s != "unset") }

impl ServeOpts {
    pub fn from_registry() -> ServeOpts { Self::from_registry_world(0) }
    pub fn from_registry_world(world: usize) -> ServeOpts {
        let num = |n: &str, d: usize| opt(n).and_then(|v| v.parse::<usize>().ok()).unwrap_or(d);
        let def_cache = crate::dsv41::load::fit::default_engram_cache_mib(world) as usize;
        ServeOpts { lanes: num("dsv41-tp-lanes", 4).clamp(1, super::fwd::MAX_LANES), max_seq: num("max-seq-len", 16384), chunk: num("dsv41-tp-chunk", 1024).max(1),
                    cache_mib: num("engram-row-cache-mib", def_cache), tail_merge: num("dsv41-prefill-tail-merge", 512) }
    }
}

/// The TP model + DSpark + prefix states of one rank.
pub struct TpServe<'a> {
    pub m: TpModel<'a>,
    pub sp: TpSpec<'a>,
    pub guard: Guard,
    snaps: HashMap<u32, TpSnap>,
    last_row: Vec<usize>,
    /// the (position, state kind) a lane's running prefill started from (None = no prefill is split over calls right now)
    pf_origin: Vec<Option<(usize, StateKind)>>,
    mode: CedMode,
    chunk: usize,
    pub rank: usize,
    pub rounds: u64,
    pub graphs: bool,
    /// leg MEM1: the last trim (device pool + host heap) and the last ledger refresh on this rank
    trim_at: Option<std::time::Instant>,
    ledger_at: Option<std::time::Instant>,
    /// leg HT1: the idle tier (off until `run` hands it the control connection)
    pub idle: super::idle::Idle,
    /// leg CE3 / P7.3: per lane, the prompt length whose tail the lane's DSpark taps slot holds right now
    tap_owner: Vec<Option<usize>>,
    /// leg CE3: per lane, the prompt-end taps of the tail state it was restored from: (that state's position, the rows)
    lane_tail: Vec<Option<(usize, crate::dsv41::fwd::dev::Buf)>>,
    /// leg P7.3: per-lane saved final prompt stream `(h, pre)` row so batched / deferred `Start` survives intermediate ops
    start_h: crate::dsv41::fwd::dev::Buf,
    start_pre: crate::dsv41::fwd::dev::Buf,
    has_start_row: Vec<bool>,
}

/// Leg CE3: can a prefill of a `total`-row prompt resuming at `at` get the DSpark taps it needs? The last call writes the taps
/// of every row it computes; rows of the prompt's last `TAP_ROWS` below `at` must come from the restored state, which has
/// them only as a tail state taken at `tail_pos >= at` with `total >= tail_pos` (then they are the same positions and tokens).
pub fn tail_resume_ok(total: usize, at: usize, tail_pos: Option<usize>) -> bool {
    let t0 = total.saturating_sub(super::spec::TAP_ROWS);
    t0 >= at || tail_pos.map_or(false, |ps| total >= ps && at <= ps)
}

/// The rows of one scoring call (leg MEM1, G1): the every-layer forward plans the indexer in 8-row blocks, and a call
/// holds at most `fwd::MAX_ROW_BLOCKS` of them (the merge tickets and layer 20's persistent candidate union).
pub fn score_rows(chunk: usize) -> usize { chunk.clamp(1, 8 * super::fwd::MAX_ROW_BLOCKS) }

/// FNV-style fold of tokens for the agree guard.
fn fold(h: u32, t: u32) -> u32 { h.wrapping_mul(0x0100_0193) ^ t }

/// Leg PP1: a prefill broadcast = the call's rows then `ahead` look-ahead tokens (the next call's); at least one row is the call's.
pub fn split_ahead(mut got: Vec<u32>, ahead: usize) -> Result<(Vec<u32>, Vec<u32>)> {
    ensure!(ahead < got.len(), "prefill: {ahead} look-ahead tokens of {} broadcast (no rows for the call)", got.len());
    let next = got.split_off(got.len() - ahead);
    Ok((got, next))
}

/// The ops that are a unit of GPU work for the thermal governor (`crate::thermal::pace`): a prefill call, a lane start, a
/// round, a score. The bookkeeping ops (and the idle `Nop` heartbeat) never pace.
pub fn op_paces(op: &Op) -> bool {
    matches!(op, Op::Prefill { .. } | Op::PrefillBatch { .. } | Op::Start { .. } | Op::StartBatch { .. } | Op::Round { .. } | Op::Score { .. })
}

/// NIGHT1's thermal governor before an op of GPU work, on every rank (each rank reads its own box; the hottest box sets the
/// group's pace at the next collective). Timing only: below --thermal-soft-c it returns at once.
pub fn op_begin(op: &Op) { if op_paces(op) { crate::thermal::pace(); } }

/// Leg PS2: the op's GPU work is over (every paced op ends with a read-back on every rank). The governor then bills only
/// the op itself: the idle time before the next request is not GPU work (before PS2 a gap of up to 2 s after the last op
/// was billed as work and a hot box slept up to ~1x that gap again before the request's first prefill: the "park
/// transition" TTFT cliff of P1M, which was not the park at all).
pub fn op_end(op: &Op) { if op_paces(op) { crate::thermal::work_done(); } }

impl<'a> TpServe<'a> {
    pub fn new(m: TpModel<'a>, mut sp: TpSpec<'a>, opts: &ServeOpts, mode: CedMode) -> TpServe<'a> {
        let world = m.world;
        let rank = m.rank;
        let n = m.lanes.len();
        let graphs = m.graphs.is_some();
        sp.lockstep_cs = true;
        let start_h = m.f.g.buf((n * m.cfg.hc * m.cfg.hidden * 2).max(256)).expect("TpServe start_h buffer");
        let start_pre = m.f.g.buf((n * 24 * 4).max(256)).expect("TpServe start_pre buffer");
        TpServe { m, sp, guard: Guard::new(world), snaps: HashMap::new(), last_row: vec![0; n], pf_origin: vec![None; n], mode, chunk: opts.chunk, rank, rounds: 0, graphs,
                  trim_at: None, ledger_at: None, idle: super::idle::Idle::off(), tap_owner: vec![None; n], lane_tail: (0..n).map(|_| None).collect(),
                  start_h, start_pre, has_start_row: vec![false; n] }
    }

    fn start_row_ptrs(&self, lane: usize) -> (u64, u64) {
        let hb = self.m.cfg.hc * self.m.cfg.hidden * 2;
        (self.start_h.ptr() + (lane * hb) as u64, self.start_pre.ptr() + (lane * 24 * 4) as u64)
    }

    fn save_start_row(&mut self, lane: usize, row: usize) -> Result<()> {
        let (h, pre) = self.m.last.context("prefill left no final stream")?;
        let hb = self.m.cfg.hc * self.m.cfg.hidden * 2;
        let (dh, dp) = self.start_row_ptrs(lane);
        self.m.f.g.d2d(dh, h + (row * hb) as u64, hb)?;
        self.m.f.g.d2d(dp, pre + (row * 24 * 4) as u64, 24 * 4)?;
        self.has_start_row[lane] = true;
        Ok(())
    }

    /// Device bytes of the kept prefix states on this rank.
    pub fn snap_bytes(&self) -> u64 { self.snaps.values().map(|s| s.bytes as u64).sum() }

    /// Leg MEM1: give freed memory back to the box (device pool + host heap) when `gap` passed since the last trim, and
    /// refresh the ledger's moving parts (at most once a second).
    fn upkeep(&mut self, trim_gap: Option<std::time::Duration>) {
        let now = std::time::Instant::now();
        if let Some(gap) = trim_gap {
            if self.trim_at.map_or(true, |t| now.duration_since(t) >= gap) { super::mem::trim(&self.m.f.g.dev); self.trim_at = Some(now); }
        }
        if self.ledger_at.map_or(true, |t| now.duration_since(t) >= std::time::Duration::from_secs(1)) {
            let sb = self.snap_bytes();
            super::mem::refresh(&self.m.f.g.dev, &self.m, sb);
            self.ledger_at = Some(now);
        }
    }

    /// `--dsv41-engram-dump` (GA's G6): the hash ids of the lane's committed token history, one JSON line per request.
    fn dump_engram(&self, lane: usize) {
        let Some(path) = opt("dsv41-engram-dump") else { return };
        let ln = &self.m.lanes[lane];
        let (Some(hs), true) = (self.m.hasher.as_ref(), !ln.hist.is_empty()) else { return };
        let per = hs.ids_per_token();
        let mut ids = vec![0u32; ln.hist.len() * per];
        let mut st = crate::dsv41::engram::hash::NgramHashState::new();
        let (live_vec, dead): (Option<Vec<bool>>, Vec<usize>) = if ln.vl_types.is_empty() {
            (None, Vec::new())
        } else {
            let lv: Vec<bool> = (0..ln.hist.len()).map(|p| ln.vl_types.get(p).copied().unwrap_or(-1) < 0).collect();
            let dd: Vec<usize> = lv.iter().enumerate().filter_map(|(i, &b)| (!b).then_some(i)).collect();
            (Some(lv), dd)
        };
        if hs.hash_chunk(&mut st, &ln.hist, live_vec.as_deref(), &mut ids).is_err() { return; }
        use base64::Engine;
        let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let line = serde_json::json!({"name": format!("req-{}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)), "tokens": ln.hist, "dead": dead,
                                      "ids_u32_le_b64": base64::engine::general_purpose::STANDARD.encode(bytes)});
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) { let _ = writeln!(f, "{line}"); }
    }

    fn positions(&self) -> Vec<usize> { self.m.lanes.iter().map(|l| l.pos).collect() }

    /// Run one op on this rank. `toks` is rank 0's payload for the ops that carry tokens (ignored elsewhere).
    pub fn exec(&mut self, op: Op, toks: Option<&[u32]>) -> Result<Done> {
        let mut d = Done::default();
        // NIGHT1's thermal governor, once per unit of GPU work on EVERY rank (each rank reads its own box; the hottest box
        // sets the group's pace at the next collective). Timing only: below --thermal-soft-c it returns at once.
        op_begin(&op);
        if op_paces(&op) {
            // leg HT1: a parked Engram reader takes 0.6-1 ms to wake; start that now, under this op's first GPU work (two atomic
            // loads per reader when they are awake, as they always are inside a request)
            if let Some(io) = self.m.eio.as_ref() { io.wake(); }
        }
        match op {
            Op::Nop => { self.upkeep(Some(std::time::Duration::from_secs(120))); }
            Op::Park => { self.upkeep(Some(std::time::Duration::from_secs(120))); self.idle.mark_parked(); }
            Op::Shutdown => {}
            Op::SetVl { lane, len } => {
                let words = self.m.bcast_tokens(if self.rank == 0 { Some(toks.context("the head must give packed vl_types")?.to_vec()) } else { None })?;
                let types = unpack_vl_types(&words, len as usize)?;
                let mut h = len;
                for &w in &words { h = fold(h, w); }
                self.m.lanes[lane].vl_types = types;
                self.m.lanes[lane].vl_row_ops.clear();
                self.m.lanes[lane].vl_aligner = None;
                self.guard.agree("serve set_vl", h)?;
            }
            Op::Prefill { lane, taps, total, first, full_at, ahead } => {
                let got = self.m.bcast_tokens(if self.rank == 0 { Some(toks.context("the head must give the tokens")?.to_vec()) } else { None })?;
                // leg PP1: the last `ahead` tokens are the next call's (read ahead, not prefilled here)
                let (new, next) = split_ahead(got, ahead as usize)?;
                self.m.pf_ahead = next;
                let mut full = self.m.lanes[lane].hist.clone();
                full.extend_from_slice(&new);
                let total = (total as usize).max(full.len());
                if first { self.pf_origin[lane] = Some((self.m.lanes[lane].pos, self.m.lanes[lane].kind)); }
                // a first call of a whole prompt computes exactly what the single-call prefill did (same schedule, same chunks)
                let origin = if first { None } else { Some(self.pf_origin[lane].context("a prefill continues without its first call")?) };
                let td = if taps && !self.sp.drafter_off { Some(self.sp.taps_lane(&self.m, lane)) } else { None };
                let full_at = (full_at > 0).then_some(full_at as usize);
                self.tap_owner[lane] = None;
                self.has_start_row[lane] = false;
                if full.len() == total && taps {
                    // leg CE3: a prompt resumed less than TAP_ROWS rows before its end takes the taps below the resume point from
                    // the tail state it was restored from (the same positions and tokens: `tail_resume_ok`)
                    let at = origin.map_or(self.m.lanes[lane].pos, |o| o.0);
                    let t0 = total.saturating_sub(super::spec::TAP_ROWS);
                    if t0 < at {
                        let tp = self.lane_tail[lane].as_ref().map(|t| t.0);
                        ensure!(tail_resume_ok(total, at, tp), "lane {lane}: a {total}-row prompt resumed at {at} needs the taps of rows {t0}..{at}, which its state does not carry");
                        let (ps, buf) = self.lane_tail[lane].as_ref().unwrap();
                        let (tb, ld) = self.sp.taps_lane(&self.m, lane);
                        let rb = ld * 2;
                        let sb = ps.saturating_sub(super::spec::TAP_ROWS);
                        self.m.f.g.d2d(tb, buf.ptr() + ((t0 - sb) * rb) as u64, (at - t0) * rb)?;
                    }
                }
                let r = self.m.prefill_part(lane, &full, total, origin, full_at, self.chunk, self.mode, td, false)?;
                if full.len() == total {
                    self.last_row[lane] = r.last_row;
                    self.save_start_row(lane, r.last_row)?;
                    self.pf_origin[lane] = None;
                    self.lane_tail[lane] = None;
                    if taps { self.tap_owner[lane] = Some(total); }
                }
                self.guard.agree("serve prefill", fold(full.len() as u32, *full.last().unwrap()))?;
            }
            Op::PrefillBatch { count, mask, taps_mask } => {
                let got = self.m.bcast_tokens(if self.rank == 0 { Some(toks.context("the head must give the batched tokens")?.to_vec()) } else { None })?;
                let lanes_in: Vec<usize> = (0..self.m.lanes.len()).filter(|&l| ((mask >> l) & 1) != 0).collect();
                ensure!(count >= 2 && lanes_in.len() == count && got.len() > count, "prefill_batch: count={count} mask={mask:#x} payload={}", got.len());
                let lens: Vec<usize> = got[..count].iter().map(|&x| x as usize).collect();
                let sum_new: usize = lens.iter().sum();
                ensure!(count + sum_new == got.len(), "prefill_batch: header lens sum {sum_new} + {count} != {}", got.len());
                let mut cur = count;
                let mut fulls: Vec<Vec<u32>> = Vec::with_capacity(count);
                for (&lane, &n_new) in lanes_in.iter().zip(&lens) {
                    ensure!(n_new >= 1, "prefill_batch: empty segment on lane {lane}");
                    let mut full = self.m.lanes[lane].hist.clone();
                    let at = self.m.lanes[lane].pos;
                    full.extend_from_slice(&got[cur..cur + n_new]);
                    cur += n_new;
                    let total = full.len();
                    self.tap_owner[lane] = None;
                    self.has_start_row[lane] = false;
                    if ((taps_mask >> lane) & 1) != 0 && !self.sp.drafter_off {
                        let t0 = total.saturating_sub(super::spec::TAP_ROWS);
                        if t0 < at {
                            let tp = self.lane_tail[lane].as_ref().map(|t| t.0);
                            ensure!(tail_resume_ok(total, at, tp), "lane {lane}: a {total}-row prompt resumed at {at} needs the taps of rows {t0}..{at}");
                            let (ps, buf) = self.lane_tail[lane].as_ref().unwrap();
                            let (tb, ld) = self.sp.taps_lane(&self.m, lane);
                            let rb = ld * 2;
                            let sb = ps.saturating_sub(super::spec::TAP_ROWS);
                            self.m.f.g.d2d(tb, buf.ptr() + ((t0 - sb) * rb) as u64, (at - t0) * rb)?;
                        }
                    }
                    fulls.push(full);
                }
                let eff_taps = if self.sp.drafter_off { 0 } else { taps_mask };
                let refs: Vec<(usize, &[u32])> = lanes_in.iter().copied().zip(fulls.iter().map(|v| v.as_slice())).collect();
                self.m.prefill_batch_ced(&self.sp, &refs, self.mode, eff_taps, self.start_h.ptr(), self.start_pre.ptr())?;
                let mut h = count as u32;
                for (&lane, full) in lanes_in.iter().zip(&fulls) {
                    self.has_start_row[lane] = true;
                    self.pf_origin[lane] = None;
                    self.lane_tail[lane] = None;
                    if ((taps_mask >> lane) & 1) != 0 { self.tap_owner[lane] = Some(full.len()); }
                    h = fold(fold(h, full.len() as u32), *full.last().unwrap());
                }
                self.guard.agree("serve prefill_batch", h)?;
            }
            Op::Start { lane, temp, seed, pen, suppress_eos } => {
                if let Some(p) = self.sp.pen.as_mut() { p.set(lane, Some(pen)); }
                let p = self.m.lanes[lane].pos;
                let last = self.last_row[lane];
                let saved = self.has_start_row[lane].then(|| self.start_row_ptrs(lane));
                let t = self.sp.start_lane_ex(&mut self.m, lane, p, last, saved, temp, seed, suppress_eos)?;
                self.has_start_row[lane] = false;
                self.guard.agree("serve start", t)?;
                if let Some(p) = self.sp.pen.as_mut() { p.add(lane, &[t]); }
                d.tok = t;
            }
            Op::StartBatch { count, mask, sup_mask } => {
                let words = self.m.bcast_tokens(if self.rank == 0 { Some(toks.context("the head must give start_batch params")?.to_vec()) } else { None })?;
                let lanes_in: Vec<usize> = (0..self.m.lanes.len()).filter(|&l| ((mask >> l) & 1) != 0).collect();
                ensure!(count >= 2 && lanes_in.len() == count && words.len() == count * 8, "start_batch: count={count} mask={mask:#x} words={}", words.len());
                let mut items = Vec::with_capacity(count);
                for (i, &lane) in lanes_in.iter().enumerate() {
                    ensure!(self.has_start_row[lane], "start_batch: lane {lane} has no saved final prompt row");
                    let w = &words[i * 8..(i + 1) * 8];
                    let temp = f32::from_bits(w[0]);
                    let seed = (w[1] as u64) | ((w[2] as u64) << 32);
                    let pen = [f32::from_bits(w[3]), f32::from_bits(w[4]), f32::from_bits(w[5])];
                    let top_k = w[6] as usize;
                    let top_p = f32::from_bits(w[7]);
                    if let Some(p) = self.sp.pen.as_mut() { p.set(lane, Some(pen)); }
                    self.sp.cutcfg[lane] = super::cut::CutCfg { top_k, top_p };
                    let (hp, pp) = self.start_row_ptrs(lane);
                    let sup = ((sup_mask >> lane) & 1) != 0;
                    items.push((lane, self.m.lanes[lane].pos, hp, pp, temp, seed, sup));
                }
                let out_toks = self.sp.start_lanes(&mut self.m, &items)?;
                let mut h = count as u32;
                for (&lane, &t) in lanes_in.iter().zip(&out_toks) {
                    self.has_start_row[lane] = false;
                    if let Some(p) = self.sp.pen.as_mut() { p.add(lane, &[t]); }
                    h = fold(h, t);
                }
                self.guard.agree("serve start_batch", h)?;
                d.chains = out_toks.into_iter().map(|t| vec![t]).collect();
            }
            Op::Round { nl, mask, copy, sup_mask, obs } => {
                self.sp.apply_pending_obs(obs);
                self.sp.copy_ok = copy;   // leg CD1: the head's permission for this round's copy proposals (always true unless the A/B switch alternates it)
                ensure!(nl >= 1 && nl <= self.sp.cap && mask != 0, "round: nl {nl} mask {mask:#b}");
                for l in 0..nl {
                    self.sp.host[l].active = ((mask >> l) & 1) as i32;
                    self.sp.host[l].suppress_eos = ((sup_mask >> (8 * l)) & 0xFF) as i32;
                }
                self.tap_owner.fill(None); // the verify writes the taps buffer
                let k = self.sp.choose_k(&mut self.m, nl)?;
                let em = self.sp.round(&mut self.m, nl, k)?;
                let mut h = fold(k as u32, self.sp.copy.as_ref().map_or(0, |c| c.last_digest));
                for l in 0..nl {
                    if ((mask >> l) & 1) != 0 {
                        h = fold(h, self.sp.host[l].k as u32);
                    }
                }
                for e in &em { for &t in e { h = fold(h, t); } h = fold(h, 0xFFFF_FFFF); }
                self.guard.agree("serve round", h)?;
                if let Some(p) = self.sp.pen.as_mut() { for (l, e) in em.iter().enumerate() { if (mask >> l) & 1 == 1 { p.add(l, e); } } }
                self.rounds += 1;
                d.chains = em;
            }
            Op::Release { lane } => {
                if self.rank == 0 { self.dump_engram(lane); }
                if let Some(c) = self.sp.copy.as_mut() { c.finish_request(lane, &self.m.lanes[lane].hist); }
                self.sp.cpv.reset_lane(lane);
                self.m.reset_lane_n(lane)?;
                self.sp.host[lane] = DsLane::default();
                self.last_row[lane] = 0;
                self.pf_origin[lane] = None;
                self.lane_tail[lane] = None;
                self.tap_owner[lane] = None;
                self.has_start_row[lane] = false;
                self.sp.cutcfg[lane] = super::cut::CutCfg::default();
                if let Some(p) = self.sp.pen.as_mut() { p.set(lane, None); }
            }
            Op::Snap { lane, id } => {
                let mut s = self.m.snapshot(lane)?;
                if self.tap_owner[lane] == Some(s.pos) && s.kind == StateKind::Full {
                    // leg CE3: taken right after the prompt's last call: keep its tail taps with it
                    let (tb, ld) = self.sp.taps_lane(&self.m, lane);
                    let n = s.pos.min(super::spec::TAP_ROWS) * ld * 2;
                    let buf = self.m.f.g.buf(n)?;
                    self.m.f.g.d2d(buf.ptr(), tb, n)?;
                    s.bytes += n;
                    s.tail = Some(buf);
                }
                d.snap_pos = s.pos;
                d.snap_bytes = s.bytes as u64;
                d.snap_full = s.kind == StateKind::Full;
                d.snap_tail = s.tail.is_some();
                self.snaps.insert(id, s);
                self.upkeep(None);
            }
            Op::Restore { lane, id } => {
                let s = self.snaps.get(&id).with_context(|| format!("restore: no state {id} on rank {}", self.rank))?;
                self.m.restore(lane, s)?;
                self.lane_tail[lane] = match &s.tail {
                    // a copy of its own: the pool may drop the state before the lane's prefill runs
                    Some(t) => { let b = self.m.f.g.buf(t.bytes)?; self.m.f.g.d2d(b.ptr(), t.ptr(), t.bytes)?; Some((s.pos, b)) }
                    None => None,
                };
                self.tap_owner[lane] = None;
                self.has_start_row[lane] = false;
                self.sp.host[lane] = DsLane::default();
                if let Some(c) = self.sp.copy.as_mut() { c.invalidate_lane(lane); }
                self.sp.cpv.reset_lane(lane);
                self.pf_origin[lane] = None;
            }
            Op::Cut { lane, top_k, top_p } => { self.sp.cutcfg[lane] = super::cut::CutCfg { top_k: top_k as usize, top_p }; }
            Op::Force { lane, drop, tok } => {
                ensure!((1..=4096).contains(&drop), "force: dropping {drop} tokens");
                let pos = self.m.lanes[lane].pos;
                let h = &mut self.sp.host[lane];
                ensure!(h.vbase as usize == pos && pos + 1 >= drop, "force: lane {lane} holds {pos} positions, its round state is at {} (drop {drop})", h.vbase);
                let np = pos + 1 - drop;
                if pos - np <= self.m.cfg.backtrack() {
                    self.m.rollback(lane, np)?;
                } else {
                    let mut cpv = std::mem::take(&mut self.sp.cpv);
                    let r = cpv.force_rollback(&mut self.m, &mut self.sp, lane, np);
                    self.sp.cpv = cpv;
                    r?;
                }
                if let Some(c) = self.sp.copy.as_mut() { c.invalidate_lane(lane); }
                let h = &mut self.sp.host[lane];
                h.vbase = np as i32;
                h.wend = np as i32 - 1;
                h.tok = tok as i32;
                if let Some(p) = self.sp.pen.as_mut() { p.add(lane, &[tok]); }
                self.guard.agree("serve force", fold(drop as u32, tok))?;
                d.tok = tok;
            }
            Op::Free { id } => { self.snaps.remove(&id); self.upkeep(Some(std::time::Duration::from_secs(1))); }
            Op::Truncate { lane, pos } => {
                self.m.rollback(lane, pos as usize)?;
                if let Some(c) = self.sp.copy.as_mut() { c.invalidate_lane(lane); }
                self.sp.cpv.reset_lane(lane);
            }
            Op::Score { lane } => {
                let ids = self.m.bcast_tokens(if self.rank == 0 { Some(toks.context("the head must give the tokens")?.to_vec()) } else { None })?;
                ensure!(ids.len() >= 2, "score: needs at least two tokens");
                self.tap_owner.fill(None);
                self.has_start_row[lane] = false;
                self.lane_tail[lane] = None;
                self.m.reset_lane_n(lane)?;
                self.sp.host[lane] = DsLane::default();
                let n = ids.len();
                let c = score_rows(self.chunk);
                let mut r0 = 0;
                let mut lp = Vec::with_capacity(n - 1);
                let r = (|| -> Result<()> {
                    while r0 < n {
                        let w = (n - r0).min(c);
                        let probe: Vec<i32> = (0..w).map(|i| if r0 + i + 1 < n { ids[r0 + i + 1] as i32 } else { -1 }).collect();
                        let outs = self.m.forward_segs(&[(lane, &ids[r0..r0 + w])], HeadWant::All, Some(&probe))?;
                        let mut h = 0u32;
                        for (i, o) in outs.iter().enumerate() { if r0 + i + 1 < n { lp.push(o.probe_lp); } h = fold(h, o.greedy as u32); }
                        self.guard.agree("serve score", h)?;
                        r0 += w;
                    }
                    Ok(())
                })();
                self.m.reset_lane_n(lane)?;
                r?;
                d.scores = lp;
            }
        }
        op_end(&op);
        d.pos = self.positions();
        Ok(d)
    }

    /// The node side: follow the head's ops until `Shutdown`.
    pub fn follow(&mut self, say: &mut dyn FnMut(String)) -> Result<()> {
        let mut n = 0u64;
        let (mut parks, mut parked) = (0u64, std::time::Duration::ZERO);
        loop {
            let h = self.m.bcast_small(&[0.0; 16])?;
            let op = Op::decode(&h)?;
            if op == Op::Shutdown { say(format!("rank {}: the head said shutdown after {n} ops", self.rank)); return Ok(()); }
            self.exec(op, None).with_context(|| format!("rank {}: op {} ({n})", self.rank, op.name()))?;
            n += 1;
            if op == Op::Park {
                // leg HT1: out of every collective until the head's next op (it writes the wake byte first)
                let t = self.idle.wait_for_wake().with_context(|| format!("rank {}: parked after {n} ops", self.rank))?;
                if let Some(io) = self.m.eio.as_ref() { io.wake(); }
                parks += 1;
                parked += t;
                if parks.is_power_of_two() {
                    say(format!("rank {}: idle tier: {parks} park(s), {:.1} s parked so far", self.rank, parked.as_secs_f64()));
                }
            }
        }
    }

    /// The head side of one op: broadcast the header, then run it.
    pub fn head_op(&mut self, op: Op, toks: Option<&[u32]>) -> Result<Done> {
        self.m.bcast_small(&op.encode())?;
        self.exec(op, toks)
    }
}

/// The lane prompts of the boot warm-up.
const WARM: [&str; 4] = [
    "Explain how a refrigerator moves heat out of its interior, step by step.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List ten practical tips for writing maintainable Rust code, with a sentence on each.",
    "What were the main causes of the fall of the Western Roman Empire? Answer in a few paragraphs.",
];

/// Everything one rank builds before serving: the model, the DSpark stages, the draft-length policy's costs (cached).
pub fn boot_rank<'a>(g: &'a Gpu, cs: &'a cudarc::driver::CudaStream, cfg: &'a Dsv41Config, comm: &'a TpComm, src: &RankSrc, model_dir: &Path, engram_dir: &Path,
                     so: &ServeOpts, say: &mut dyn FnMut(String)) -> Result<TpServe<'a>> {
    let rank = comm.rank;
    let t0 = std::time::Instant::now();
    // NIGHT1: resolve the thermal governor now (its one boot line; `exec` paces every op of GPU work on this rank)
    crate::thermal::start();
    let mut m = TpModel::new(g, cs, cfg, comm, src, engram_dir, model_dir, so.chunk.max(super::fwd::HEAD_MAX) + so.tail_merge, so.max_seq, so.cache_mib, so.lanes, say)?;
    m.tail_merge = so.tail_merge;
    if opt("dsv41-tp-graphs").as_deref() != Some("off") {
        m.graphs = Some(super::graph::Graphs::default());
        say(format!("rank {rank}: decode / verify calls and DSpark draft steps run as CUDA graphs (captured per shape, replayed at every context)"));
    }
    m.load_stages(src)?;
    say(format!("rank {rank}: DSpark stages 40-42 + main_proj loaded; weights {:.2} GiB, {:.0} s", m.weights_bytes as f64 / 1073741824.0, t0.elapsed().as_secs_f64()));
    m.barrier(0x5345)?;
    // INT12: the hc side stream only when every rank holds hc_fn whole (else off on every rank, one line, counted): after the
    // boot barrier, so no rank waits in a collective while another still loads
    m.pa1_agree()?;
    let mut sp = TpSpec::new(&m, comm, cs, m.lanes.len())?;
    sp.pen = Some(super::pen::Pen::new(g, m.lanes.len(), cfg.vocab, comm.vocab().start, comm.vocab().len())?);
    // leg P7.2: broadcast the model's stop token IDs (up to 4) from Rank 0 so every rank masks them identically on suppressed rows
    {
        let mut sids = [-1i32; 4];
        if rank == 0 {
            let mut found = crate::dsv41::chat::special::stop_token_ids(model_dir);
            if found.is_empty() { found.push(cfg.eos); }
            for (i, &id) in found.iter().take(4).enumerate() { sids[i] = id as i32; }
        }
        let mut send = [0.0f32; 8];
        if rank == 0 {
            for i in 0..4 {
                let p = halves(sids[i] as u32);
                send[2 * i] = p[0];
                send[2 * i + 1] = p[1];
            }
        }
        let got = m.bcast_small(&send)?;
        for i in 0..4 { sp.round.stop_ids[i] = join(got[2 * i], got[2 * i + 1]) as i32; }
    }
    // leg PH1: with speculation off at boot (`--mtp off` / `--dsv41-draft-len 0`, the same flags on every rank) the drafter
    // never runs: no DSpark taps, no main_proj rider, no ring keys; otherwise the Markov bias cache
    sp.drafter_off = sp.policy.mode == crate::dsv41::gpu::dspark::policy::Mode::Off;
    // leg CD1 / OPT1: copy drafting needs the drafter's rings and speculation round; silent off under `auto`, one line when explicitly asked
    if sp.copy.is_some() && sp.drafter_off {
        sp.copy = None;
        if super::copy::explicitly_on() {
            say("[dsv41 copy] --dsv41-copy-draft on asked, but speculation is off (--mtp off / --dsv41-draft-len 0): copy drafting is OFF (it rides the speculation round)".to_string());
        }
    }
    if let Some(c) = sp.copy.as_ref() {
        super::copy::set_enabled(Some(&c.cfg));
        say(format!("[dsv41 copy] copy drafting ON (--dsv41-copy-draft, legs CD1/P7COPY): lockstep CopySpan (min match {}, start ramp {} -> max_w {} copied tokens = {} verify rows, zero RDMA broadcast, WideCopyGuard O(1) rollback; yield_gate={}, copy_lanes={}, copy_record={}, copy_prefill={}); \
kept iff equal to the target's own draw (greedy: argmax; sampled: keyed draw, distribution-exact and bit-identical to plain decode); /metrics velogb10_dsv41_copy_*",
                   c.cfg.min_match, c.cfg.start_w, c.cfg.max_w, c.cfg.max_w + 1,
                   if c.cfg.yield_gate { "on" } else { "off" }, if c.cfg.copy_lanes { "on" } else { "off" },
                   if c.cfg.copy_record { "on" } else { "off" }, if c.cfg.prefill { format!("on(max={})", c.cfg.prefill_max_rows) } else { "off".into() }));
    }
    if let Some(c) = sp.copy.as_ref() {
        if c.ab_period_s > 0.0 {
            say(format!("[dsv41 copy] !! MEASUREMENT SWITCH --dsv41-copy-ab-period-s {}: copy proposals are permitted only in even {}-second periods of the head's clock (a paired A/B inside one boot); /metrics velogb10_dsv41_copy_ab_on_rounds_total / _off_rounds_total. Not a serving configuration", c.ab_period_s, c.ab_period_s));
        }
    }
    let line = sp.attach_markov_cache(&mut m)?;
    if rank == 0 || !line.contains(" off: ") { say(line); }
    if let Some(l) = crate::dsv41::gpu::dspark::metrics::announce("dspark (TP serve, leg SE)", &format!("{:?}", sp.policy.mode)) {
        say(l);
        say(crate::dsv41::gpu::dspark::policy::boot_line(&sp.policy));
        if sp.policy.decides_after_draft() && sp.policy.gate_jump_ms > 0.0 {
            say(format!("[dsv41 spec] !! GATE ONLY --dsv41-spec-gate-jump-ms {}: every round with a draft is PRICED {} ms dearer (a synthetic concave cost table, so the exact rule and the scan part often for the PIT gate); not natural traffic, speculation pays less", sp.policy.gate_jump_ms, sp.policy.gate_jump_ms));
        }
        if sp.policy.decides_after_draft() && sp.policy.stop == crate::dsv41::gpu::dspark::policy::Stop::Scan {
            say("[dsv41 spec] !! --dsv41-spec-inexact-scan on: the draft count is the best over all counts, which reads each draft's own draw: sampled output is NOT exactly the model's distribution (a measurement build only; /metrics velogb10_dsv41_spec_inexact_scan 1)".to_string());
        }
    }
    let mode = schedule::mode_from_flags()?;
    let mut s = TpServe::new(m, sp, so, mode);
    s.guard.agree("serve model loaded", 0x5345_0001)?;
    super::mem::phase("model + stages + lanes");
    warm(&mut s, model_dir, say)?;
    // leg MEM1: the loader's and the warm-up's freed memory back to the box, then the ledger of what this rank holds
    super::mem::trim(&g.dev);
    s.trim_at = Some(std::time::Instant::now());
    super::mem::phase("warm-up + trim");
    let (line, detail) = super::mem::ledger(&g.dev, &s.m, s.sp.dev_bytes(), s.snap_bytes(), rank);
    say(line);
    crate::dprintln!("{detail}");
    Ok(s)
}

/// Warm-up before the first request: start every lane on a short prompt, measure the draft-length costs (cached per
/// binary + pack + TP size; the policy needs them), run a few rounds so the verify / draft graphs are captured, then
/// release the lanes. Every rank runs the same calls.
fn warm(s: &mut TpServe, model_dir: &Path, say: &mut dyn FnMut(String)) -> Result<()> {
    let t = std::time::Instant::now();
    let tok = QwenTokenizer::from_file(&model_dir.join("tokenizer.json").to_string_lossy()).context("tokenizer.json")?;
    let nl = s.m.lanes.len();
    for l in 0..nl {
        let ids = super::fwprog::encode(&tok, WARM[l % WARM.len()], true)?;
        s.m.reset_lane_n(l)?;
        let td = if s.sp.drafter_off { None } else { Some(s.sp.taps_lane(&s.m, l)) };
        let r = s.m.prefill(l, &ids, s.chunk, s.mode, td, false)?;
        s.last_row[l] = r.last_row;
        let p = s.m.lanes[l].pos;
        let t0 = s.sp.start_lane(&mut s.m, l, p, r.last_row, 1.0, 17 + l as u64)?;
        s.guard.agree("warm start", t0)?;
    }
    if matches!(s.sp.policy.mode, crate::dsv41::gpu::dspark::policy::Mode::Auto) {
        // leg HT1: the costs feed the draft-length policy: always timed with spinning waits (--dsv41-round-wait hybrid could
        // wake a cheap call late and price it high: the trunk's depth trap, CPUIDLE)
        let line = { let _spin = crate::host_wait::SpinScope::new(); s.sp.calibrate(&mut s.m, nl, model_dir)? };
        s.guard.agree("warm calibration", 0x5CA1)?;
        say(format!("rank {}: {line}", s.rank));
    }
    let clk_probe = if s.rank == 0 {
        std::thread::Builder::new().name("gb10-clk".into()).spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(40));
            crate::exl3_tune::run_timeout(
                "nvidia-smi",
                &["--query-gpu=name,clocks.current.graphics", "--format=csv,noheader,nounits"],
                std::time::Duration::from_millis(1200),
            )
        }).ok()
    } else {
        None
    };
    let copy_off = s.sp.copy.take();   // leg CD1: the warm-up's rounds are not copy traffic
    for _ in 0..8 {
        let k = s.sp.choose_k(&mut s.m, nl)?;
        let em = s.sp.round(&mut s.m, nl, k)?;
        s.sp.sync_pending_obs(&mut s.m)?;
        let mut h = k as u32;
        for l in 0..nl { h = fold(h, s.sp.host[l].k as u32); }
        h = em.iter().flatten().fold(h, |a, &x| fold(a, x));
        s.guard.agree("warm round", h)?;
    }
    s.sp.pending_obs = None;
    s.sp.t_end = None;
    s.sp.copy = copy_off;
    for l in 0..nl { s.m.reset_lane_n(l)?; s.sp.host[l] = DsLane::default(); s.last_row[l] = 0; s.has_start_row[l] = false; s.tap_owner[l] = None; }
    s.m.settle()?;
    if let Some(w) = clk_probe.and_then(|j| j.join().ok()).flatten().and_then(|csv| gb10_unlocked_clock_warn(&csv)) {
        crate::boot_report::warn(&w);
    }
    say(format!("rank {}: warm-up done in {:.1} s ({nl} lanes started, costs priced, graphs captured for the shapes the rounds used)", s.rank, t.elapsed().as_secs_f64()));
    Ok(())
}

struct BatchCandRows {
    row0: usize,
    n: usize,
    bm: crate::dsv41::fwd::dev::Buf,
    pos: crate::dsv41::fwd::dev::Buf,
    nbw_cap: usize,
    nb_cap: usize,
    n_keys_cap: usize,
    ratio: u32,
}

impl TpModel<'_> {
    /// Leg P7.3: multi-lane CED prefill for `>= 2` text-only prompts whose uncached tokens fit one arena pass (`<= max_rows`).
    /// Shares all 40 layers' dense / MoE / Engram weight reads and all 80 RDMA all-reduces in one pass while executing window,
    /// compressor, indexer scoring, sparse attention sources, and DSpark tail taps per lane.
    pub fn prefill_batch_ced(
        &mut self,
        sp: &TpSpec,
        items: &[(usize, &[u32])],
        mode: CedMode,
        taps_mask: u32,
        start_h: u64,
        start_pre: u64,
    ) -> Result<()> {
        use super::frame::MIX_STRIDE;
        use super::fwd::Seg;
        use crate::dsv41::census::classify::TensorClass as C;
        use crate::dsv41::ced::schedule::{n_of, Schedule};
        use crate::dsv41::fwd::glue;

        struct BSeg { lane: usize, k0: usize, p: usize, sched: Schedule, hrow0: usize }

        let nb = items.len();
        ensure!(nb >= 2, "prefill_batch_ced requires at least 2 lanes");
        ensure!(self.layers.len() >= self.cfg.layers && self.layers[..self.cfg.layers].iter().all(|x| x.is_some()), "the prefill needs all trunk layers");
        let mut bsegs = Vec::with_capacity(nb);
        let mut init_segs = Vec::with_capacity(nb);
        let mut all_new = Vec::new();
        for &(lane, toks) in items {
            let k = self.lanes[lane].pos;
            let p = toks.len();
            ensure!(k < p, "lane {lane} holds {k} positions of {p}");
            ensure!(self.lanes[lane].hist[..] == toks[..k], "lane {lane}: prefix mismatch");
            ensure!(self.lanes[lane].vl_types.is_empty(), "lane {lane}: prefill_batch_ced is text-only");
            let sched = if k == 0 {
                Schedule::new_with(self.cfg, p, mode, None)?
            } else {
                Schedule::resumed_with(self.cfg, p, mode, k, self.lanes[lane].kind, None)?
            };
            init_segs.push(Seg { lane, row0: all_new.len(), n: p - k, p0: k });
            all_new.extend_from_slice(&toks[k..p]);
            bsegs.push(BSeg { lane, k0: k, p, sched, hrow0: k });
        }
        ensure!(all_new.len() <= self.max_rows, "prefill_batch_ced: {} rows > max_rows {}", all_new.len(), self.max_rows);

        let wm = std::mem::replace(&mut self.f.wide_min, super::ced::WIDE_MIN);
        let r = (|| -> Result<u64> {
            let cfg = self.cfg;
            let (d, hc) = (cfg.hidden, cfg.hc);
            let (fa, fb) = (self.frames[0].ptr(), self.frames[1].ptr());
            let rows0 = all_new.len();
            self.swap_bump();
            self.segs = init_segs.clone();
            self.upload_rows(&all_new)?;
            let mut cur_q_segs = init_segs;
            let erows = self.engram_rows(&all_new)?;
            let mut h = self.embed(self.ids_dev, rows0)?;
            let mut pre = self.f.b.take(rows0 * 24 * 4)?;
            self.k.fill_pre(self.cs, pre, 24, rows0, hc)?;
            let mut kv_next: Option<u64> = None;
            let mut sel_of: HashMap<usize, (u64, Vec<Seg>)> = HashMap::new();
            let mut cands_by_lane: Vec<Option<BatchCandRows>> = (0..self.lanes.len()).map(|_| None).collect();
            let mut layer_rows = 0u64;
            let mut last_q0: Vec<Option<usize>> = vec![None; nb];

            for l in 0..cfg.layers {
                if l > 0 && l % crate::thermal::PASS_EVERY_LAYERS == 0 {
                    let g = self.f.g;
                    crate::thermal::pace_pass(self.gm, || g.sync())?;
                }
                self.swap_bump();
                let crs: Vec<_> = bsegs.iter().map(|bs| bs.sched.chunk(l, bs.k0, bs.p)).collect();
                let na_total: usize = crs.iter().map(|cr| n_of(cr.a)).sum();
                if na_total == 0 { break; }
                for (bs, cr) in bsegs.iter().zip(&crs) {
                    if n_of(cr.a) > 0 { ensure!(cr.a.0 >= bs.hrow0, "layer {l}: lane {} needs rows from {}, stream starts at {}", bs.lane, cr.a.0, bs.hrow0); }
                }
                if let Some(kvb) = kv_next.take() {
                    ensure!(bsegs.iter().all(|bs| bs.hrow0 == bs.k0), "an Engram layer inside a decoder cone");
                    let h_total: usize = bsegs.iter().map(|bs| bs.p - bs.hrow0).sum();
                    h = self.engram_gate(l, h, kvb, h_total)?;
                }
                if let Some(ti) = cfg.dspark_targets.iter().position(|&x| x == l) {
                    let mut hr = 0usize;
                    for bs in &bsegs {
                        let hlen = bs.p - bs.hrow0;
                        if hlen > 0 && ((taps_mask >> bs.lane) & 1) != 0 {
                            let t0 = bs.p.saturating_sub(super::spec::TAP_ROWS);
                            let lo = t0.max(bs.hrow0);
                            if lo < bs.p {
                                let (tb, tld) = sp.taps_lane(self, bs.lane);
                                self.k.tap(self.cs, h + (hr * hc * d * 2) as u64, hc, d, lo - bs.hrow0, bs.p - lo, tb, lo - t0, tld, ti * d)?;
                            }
                        }
                        hr += hlen;
                    }
                }
                let a_full = bsegs.iter().zip(&crs).all(|(bs, cr)| n_of(cr.a) == bs.p - bs.hrow0 && (n_of(cr.a) == 0 || cr.a.0 == bs.hrow0));
                let (ha, prea) = if a_full {
                    (h, pre)
                } else {
                    let ha = self.f.b.take(na_total * hc * d * 2)?;
                    let prea = self.f.b.take(na_total * 24 * 4)?;
                    let (mut hr, mut ar) = (0usize, 0usize);
                    for (bs, cr) in bsegs.iter().zip(&crs) {
                        let hlen = bs.p - bs.hrow0;
                        let na = n_of(cr.a);
                        if na > 0 {
                            let off = hr + (cr.a.0 - bs.hrow0);
                            self.f.g.d2d(ha + (ar * hc * d * 2) as u64, h + (off * hc * d * 2) as u64, na * hc * d * 2)?;
                            self.f.g.d2d(prea + (ar * 24 * 4) as u64, pre + (off * 24 * 4) as u64, na * 24 * 4)?;
                            ar += na;
                        }
                        hr += hlen;
                    }
                    (ha, prea)
                };
                let ratio = cfg.ratio(l);
                let compressed = ratio > 0;
                let an = self.layers[l].as_ref().unwrap().w.param(C::AttnNorm)?.clone();
                let ain = self.f.hc_pre_norm(ha, prea, 24, &an, na_total)?;
                let mut a_offs = Vec::with_capacity(nb);
                {
                    let mut ar = 0usize;
                    for cr in &crs { a_offs.push(ar); ar += n_of(cr.a); }
                }

                // Compressor (`cr.kv`)
                let nkv_total: usize = crs.iter().map(|cr| cr.kv.map_or(0, n_of)).sum();
                if nkv_total > 0 {
                    let kv_full = crs.iter().all(|cr| cr.kv.map_or(0, n_of) == n_of(cr.a) && cr.kv.map_or(true, |kv| n_of(kv) == 0 || kv.0 == cr.a.0));
                    let mut kv_segs = Vec::with_capacity(nb);
                    let akv = if kv_full {
                        let mut r0 = 0usize;
                        for (bs, cr) in bsegs.iter().zip(&crs) {
                            if let Some(kv) = cr.kv {
                                let nkv = n_of(kv);
                                if nkv > 0 { kv_segs.push(Seg { lane: bs.lane, row0: r0, n: nkv, p0: kv.0 }); r0 += nkv; }
                            }
                        }
                        ain
                    } else {
                        let akv = self.f.b.take(nkv_total * d * 2)?;
                        let mut r0 = 0usize;
                        for ((bs, cr), &ao) in bsegs.iter().zip(&crs).zip(&a_offs) {
                            if let Some(kv) = cr.kv {
                                let nkv = n_of(kv);
                                if nkv > 0 {
                                    let src_r = ao + (kv.0 - cr.a.0);
                                    self.f.g.d2d(akv + (r0 * d * 2) as u64, ain + (src_r * d * 2) as u64, nkv * d * 2)?;
                                    kv_segs.push(Seg { lane: bs.lane, row0: r0, n: nkv, p0: kv.0 });
                                    r0 += nkv;
                                }
                            }
                        }
                        akv
                    };
                    self.segs = kv_segs;
                    self.compressor(l, akv, nkv_total)?;
                }

                // Window (`cr.w`)
                let nw_total: usize = crs.iter().map(|cr| n_of(cr.w)).sum();
                if nw_total > 0 {
                    let w_full = crs.iter().all(|cr| n_of(cr.w) == n_of(cr.a) && (n_of(cr.w) == 0 || cr.w.0 == cr.a.0));
                    let mut w_segs = Vec::with_capacity(nb);
                    let aw = if w_full {
                        let mut r0 = 0usize;
                        for (bs, cr) in bsegs.iter().zip(&crs) {
                            let nw = n_of(cr.w);
                            if nw > 0 { w_segs.push(Seg { lane: bs.lane, row0: r0, n: nw, p0: cr.w.0 }); r0 += nw; }
                        }
                        ain
                    } else {
                        let aw = self.f.b.take(nw_total * d * 2)?;
                        let mut r0 = 0usize;
                        for ((bs, cr), &ao) in bsegs.iter().zip(&crs).zip(&a_offs) {
                            let nw = n_of(cr.w);
                            if nw > 0 {
                                let src_r = ao + (cr.w.0 - cr.a.0);
                                self.f.g.d2d(aw + (r0 * d * 2) as u64, ain + (src_r * d * 2) as u64, nw * d * 2)?;
                                w_segs.push(Seg { lane: bs.lane, row0: r0, n: nw, p0: cr.w.0 });
                                r0 += nw;
                            }
                        }
                        aw
                    };
                    self.segs = w_segs;
                    self.window(l, aw, nw_total, compressed)?;
                }

                // Queries (`cr.q`)
                let nq_total: usize = crs.iter().map(|cr| n_of(cr.q)).sum();
                if nq_total == 0 { break; }
                layer_rows += nq_total as u64;
                let mut q_segs = Vec::with_capacity(nb);
                {
                    let mut r0 = 0usize;
                    for (bs, cr) in bsegs.iter().zip(&crs) {
                        let nq = n_of(cr.q);
                        if nq > 0 { q_segs.push(Seg { lane: bs.lane, row0: r0, n: nq, p0: cr.q.0 }); r0 += nq; }
                    }
                }
                if q_segs != cur_q_segs {
                    let mut pos_v = Vec::with_capacity(nq_total);
                    let mut lane_v = Vec::with_capacity(nq_total);
                    for (si, sg) in q_segs.iter().enumerate() {
                        for i in 0..sg.n { pos_v.push((sg.p0 + i) as i32); lane_v.push(si as i32); }
                    }
                    self.f.g.h2d_async(self.pos_buf.ptr(), &pos_v)?;
                    if q_segs.len() > 1 { self.f.g.h2d_async(self.lane_buf.ptr(), &lane_v)?; }
                    cur_q_segs = q_segs.clone();
                }
                self.segs = q_segs.clone();
                let posq = self.pos_buf.ptr();

                let hq_full = bsegs.iter().zip(&crs).all(|(bs, cr)| n_of(cr.q) == bs.p - bs.hrow0 && (n_of(cr.q) == 0 || cr.q.0 == bs.hrow0));
                let hq = if hq_full {
                    h
                } else {
                    let hq = self.f.b.take(nq_total * hc * d * 2)?;
                    let (mut hr, mut qr0) = (0usize, 0usize);
                    for (bs, cr) in bsegs.iter().zip(&crs) {
                        let hlen = bs.p - bs.hrow0;
                        let nq = n_of(cr.q);
                        if nq > 0 {
                            let off = hr + (cr.q.0 - bs.hrow0);
                            self.f.g.d2d(hq + (qr0 * hc * d * 2) as u64, h + (off * hc * d * 2) as u64, nq * hc * d * 2)?;
                            qr0 += nq;
                        }
                        hr += hlen;
                    }
                    hq
                };
                let qa_full = crs.iter().all(|cr| n_of(cr.q) == n_of(cr.a) && (n_of(cr.q) == 0 || cr.q.0 == cr.a.0));
                let qa = if qa_full {
                    ain
                } else {
                    let qa = self.f.b.take(nq_total * d * 2)?;
                    let mut qr0 = 0usize;
                    for (cr, &ao) in crs.iter().zip(&a_offs) {
                        let nq = n_of(cr.q);
                        if nq > 0 {
                            let src_r = ao + (cr.q.0 - cr.a.0);
                            self.f.g.d2d(qa + (qr0 * d * 2) as u64, ain + (src_r * d * 2) as u64, nq * d * 2)?;
                            qr0 += nq;
                        }
                    }
                    qa
                };

                self.mix_partial(self.layers[l].as_ref().unwrap().mix_a, hq, nq_total, fa + (nq_total * d * 4) as u64)?;
                let qr = self.q_lora(l, qa, nq_total)?;
                let mut sel = None;
                let mut src = None;
                if compressed {
                    let s = cfg.kv_source_of(l).context("compressed layer without a KV source")?;
                    src = Some(s);
                    let isrc = cfg.index_source_of(l).context("no index source")?;
                    if isrc == l {
                        let keep = self.indexer_prefill_batch(l, s, qr, qa, posq, &q_segs, &mut cands_by_lane)?;
                        sel_of.insert(l, (keep, q_segs.clone()));
                    }
                    let (sl_base, ref s_segs) = *sel_of.get(&isrc).with_context(|| format!("layer {l}: index source {isrc} missing"))?;
                    if *s_segs == q_segs {
                        sel = Some((sl_base, 512usize, 512usize));
                    } else {
                        let spk = self.f.b.take(nq_total * 512 * 4)?;
                        for sg in &q_segs {
                            let ssg = s_segs.iter().find(|x| x.lane == sg.lane).with_context(|| format!("layer {l}: lane {} missing in index source {isrc}", sg.lane))?;
                            ensure!(sg.p0 >= ssg.p0 && sg.p0 + sg.n <= ssg.p0 + ssg.n, "layer {l}: query range outside index source");
                            let src_r = ssg.row0 + (sg.p0 - ssg.p0);
                            self.f.g.d2d(spk + (sg.row0 * 512 * 4) as u64, sl_base + (src_r * 512 * 4) as u64, sg.n * 512 * 4)?;
                        }
                        sel = Some((spk, 512usize, 512usize));
                    }
                }
                let win_lo = crs.iter().filter(|cr| n_of(cr.q) > 0).map(|cr| cr.win_lo).next().unwrap_or(0);
                self.win_lo = win_lo;
                self.attn_core(l, qr, posq, nq_total, src, sel, fa)?;
                self.win_lo = 0;
                self.reduce_act(fa, nq_total * (d + MIX_STRIDE))?;
                let rec_a = self.mix_finish(l, true, fa + (nq_total * d * 4) as u64, nq_total)?;
                let ao = self.f.b.take(nq_total * d * 2)?;
                glue::add_round(self.f.g, fa, 0, ao, nq_total, d)?;
                let xm = self.f.hc_post(ao, hq, rec_a, nq_total)?;
                let engram_next = cfg.engram_layers.contains(&(l + 1));
                let extra = if engram_next { (hc + 1) * d } else { 0 };
                self.mix_partial(self.layers[l].as_ref().unwrap().mix_f, xm, nq_total, fb + (nq_total * d * 4) as u64)?;
                let fnm = self.layers[l].as_ref().unwrap().w.param(C::FfnNorm)?.clone();
                let fin = self.f.hc_pre_norm(xm, rec_a, 24, &fnm, nq_total)?;
                let (rid, rw) = self.f.router(&self.layers[l].as_ref().unwrap().w, fin, nq_total)?;
                self.ffn_core(l, fin, nq_total, rid, rw, fb)?;
                if engram_next {
                    ensure!(q_segs.len() == nb && bsegs.iter().zip(&q_segs).all(|(bs, sg)| sg.p0 == bs.k0 && sg.n == bs.p - bs.k0), "an Engram layer inside a decoder cone");
                    let (e, stride) = erows.context("an Engram layer without the Engram runtime")?;
                    let li = cfg.engram_layers.iter().position(|&x| x == l + 1).unwrap();
                    self.engram_partial(l + 1, e + (li * stride) as u64, nq_total, fb + (nq_total * (d + MIX_STRIDE) * 4) as u64)?;
                }
                self.reduce_act(fb, nq_total * (d + MIX_STRIDE + extra))?;
                let rec_f = self.mix_finish(l, false, fb + (nq_total * d * 4) as u64, nq_total)?;
                let fo = self.f.b.take(nq_total * d * 2)?;
                glue::add_round(self.f.g, fb, 0, fo, nq_total, d)?;
                h = self.f.hc_post(fo, xm, rec_f, nq_total)?;
                if engram_next {
                    let kvb = self.f.b.take(nq_total * extra * 2)?;
                    glue::add_round(self.f.g, fb + (nq_total * (d + MIX_STRIDE) * 4) as u64, 0, kvb, nq_total, extra)?;
                    kv_next = Some(kvb);
                }
                pre = rec_f;
                for (i, (bs, cr)) in bsegs.iter_mut().zip(&crs).enumerate() {
                    let nq = n_of(cr.q);
                    bs.hrow0 = if nq > 0 { cr.q.0 } else { bs.p };
                    if l + 1 == cfg.layers { last_q0[i] = (nq > 0).then_some(cr.q.0); }
                }
            }

            self.engram_check()?;
            let mut q_off = 0usize;
            for (i, (bs, &(_, toks))) in bsegs.iter().zip(items).enumerate() {
                let q0 = last_q0[i].context("a batched prefill lane did not reach the last layer")?;
                let nq = bs.p - q0;
                let rf = q_off + (nq - 1);
                let hb = hc * d * 2;
                self.f.g.d2d(start_h + (bs.lane * hb) as u64, h + (rf * hb) as u64, hb)?;
                self.f.g.d2d(start_pre + (bs.lane * 24 * 4) as u64, pre + (rf * 24 * 4) as u64, 24 * 4)?;
                q_off += nq;

                let ln = &mut self.lanes[bs.lane];
                ln.pos = bs.p;
                ln.hist.extend_from_slice(&toks[bs.k0..bs.p]);
                ln.kind = bs.sched.state_kind_at(bs.p);
                ln.vl_aligner = None;
                ln.vl_row_ops.clear();
            }
            self.last = None;
            self.comm.check("after a batched prefill")?;
            Ok(layer_rows)
        })();
        self.f.wide_min = wm;
        self.win_lo = 0;
        let lr = r?;
        let exact: u64 = bsegs.iter().map(|bs| ((bs.p - bs.k0) * self.cfg.layers) as u64).sum();
        super::metrics::ced_rows(lr, exact);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn indexer_prefill_batch(
        &mut self,
        l: usize,
        s: usize,
        qr: u64,
        x: u64,
        pos: u64,
        q_segs: &[super::fwd::Seg],
        cands_by_lane: &mut [Option<BatchCandRows>],
    ) -> Result<u64> {
        use crate::dsv41::census::classify::TensorClass as C;
        use crate::dsv41::gpu::indexer as ki;

        let cfg = self.cfg;
        let (d, ql) = (cfg.hidden, cfg.q_lora_rank);
        let cand_out = cfg.candidate_source == Some(l);
        let cand_in = cfg.uses_candidates(l);
        let nq_total: usize = q_segs.iter().map(|sg| sg.n).sum();
        let w = &self.layers[l].as_ref().unwrap().w;
        let iq = self.f.b.take(nq_total * ki::HEADS * ki::HEAD_DIM * 2)?;
        self.f.lin_rows("tp.prefill.idx.wq_b", w, C::IdxWqB, None, qr, ql, ql, nq_total, iq, ki::HEADS * ki::HEAD_DIM, false)?;
        let wp = self.f.b.take(nq_total * ki::HEADS * 2)?;
        let w = &self.layers[l].as_ref().unwrap().w;
        self.f.lin_rows("tp.prefill.idx.weights_proj", w, C::IdxWeightsProj, None, x, d, d, nq_total, wp, ki::HEADS, false)?;
        let keep = {
            let b = self.sel_bufs.get(&l).context("index source without a selection buffer")?;
            ensure!(b.bytes >= nq_total * 512 * 4, "layer {l}: selection buffer {} B < {nq_total} rows", b.bytes);
            b.ptr()
        };

        for sg in q_segs {
            let (lane, r_off, nq, q0) = (sg.lane, sg.row0, sg.n, sg.p0);
            let st = self.lanes[lane].src.get(&s).context("index source state")?;
            let ratio = st.ratio;
            let keys_at_end = (q0 + nq) / ratio.max(1) as usize;
            ensure!(keys_at_end <= st.n_groups, "layer {l}: lane {lane} index keys up to {keys_at_end} needed, source holds {}", st.n_groups);
            let (kq, ks) = (st.iq.ptr(), st.is.ptr());
            let r = 8usize;
            let rbs: Vec<(i32, i32)> = (0..nq).step_by(r).map(|i| (i as i32, (nq - i).min(r) as i32)).collect();
            let n_rb = rbs.len();
            let cap = keys_at_end.max(1);
            let ucap = r * ki::CAND_BLOCKS;
            let sz = ki::sizes(n_rb, r, 1, nq, cap, ucap);
            let rbv: Vec<u64> = rbs.iter().flat_map(|&(r0, n)| [kq, ks, (r0 as u32 as u64) | ((n as u32 as u64) << 32), 0]).collect();
            let rbd = self.f.b.take(rbv.len() * 8)?;
            self.f.g.h2d_async(rbd, &rbv)?;
            let sel = keep + (r_off * 512 * 4) as u64;
            let cnt = self.f.b.take(nq * 4)?;
            let part = self.f.b.take(sz.part_u64.max(1) * 8)?;
            let pcnt = self.f.b.take(sz.pcnt_i32.max(1) * 4)?;
            let bmax = if cand_out { self.f.b.take(sz.bmax_u16.max(1) * 2)? } else { 0 };
            let cand = self.f.b.take(sz.cand_i32.max(1) * 4)?;
            let ccnt = self.f.b.take(nq * 4)?;
            let err = self.f.b.take(4)?;
            self.f.g.memset(err, 4, 0)?;
            let pos_s = pos + (r_off * 4) as u64;
            let mut a = ki::IdxArgs::default();
            a.iq = iq + (r_off * ki::HEADS * ki::HEAD_DIM * 2) as u64;
            a.w = wp + (r_off * ki::HEADS * 2) as u64;
            a.pos = pos_s; a.rbs = rbd; a.sel = sel; a.sel_cnt = cnt; a.part = part; a.pcnt = pcnt;
            a.bmax = bmax; a.cand = cand; a.cand_cnt = ccnt; a.err = err; a.tick = 0;
            a.n_rb = n_rb as i32; a.rows_total = nq as i32; a.iq_ld = (ki::HEADS * ki::HEAD_DIM) as i32; a.w_ld = ki::HEADS as i32; a.in_f32 = 0;
            a.ratio = ratio as i32; a.write_bmax = cand_out as i32;
            a.n_keys_cap = cap as i32; a.nb_cap = cap.div_ceil(ki::CAND_BLOCK) as i32; a.nbw_cap = (a.nb_cap as usize).div_ceil(32) as i32;
            a.g = 1; a.min_tiles = 8; a.dbg_cap = 0; a.ucap = ucap as i32; a.r = r as i32; a.inv_freq = self.f.rope.ki;
            let path = if cand_in {
                let c = cands_by_lane[lane].as_ref().with_context(|| format!("layer {l}: lane {lane} has no candidate rows"))?;
                ensure!(q0 >= c.row0 && q0 + nq <= c.row0 + c.n, "layer {l}: lane {lane} rows outside candidate rows");
                let off = q0 - c.row0;
                let uni = self.f.b.take(n_rb * ucap * 4)?;
                let ucnt = self.f.b.take(n_rb * 4)?;
                let mut u = a;
                u.pos = c.pos.ptr() + (off * 4) as u64;
                u.bm = c.bm.ptr() + (off * c.nbw_cap * 4) as u64;
                u.n_keys_cap = c.n_keys_cap as i32; u.nb_cap = c.nb_cap as i32; u.nbw_cap = c.nbw_cap as i32; u.ratio = c.ratio as i32;
                u.uni = uni; u.ucnt = ucnt;
                ki::union(&self.f.g.l(), &u)?;
                a.uni = uni; a.ucnt = ucnt; a.mode = 1;
                a.n_keys_cap = c.n_keys_cap as i32; a.nb_cap = c.nb_cap as i32; a.nbw_cap = c.nbw_cap as i32;
                ki::score(&self.f.g.l(), &a)?;
                "tp.prefill.idx (union of own rows + mode 1)"
            } else if cand_out {
                let bm = self.f.g.buf(sz.bm_u32.max(1) * 4)?;
                let pb = self.f.g.buf(nq.max(1) * 4)?;
                self.f.g.d2d(pb.ptr(), pos_s, nq * 4)?;
                a.bm = bm.ptr();
                a.uni = self.f.b.take(4)?; a.ucnt = self.f.b.take(n_rb.max(1) * 4)?;
                ki::score(&self.f.g.l(), &a)?;
                ki::cand(&self.f.g.l(), &a)?;
                cands_by_lane[lane] = Some(BatchCandRows { row0: q0, n: nq, bm, pos: pb, nbw_cap: a.nbw_cap as usize, nb_cap: a.nb_cap as usize, n_keys_cap: cap, ratio });
                "tp.prefill.idx (scores + candidate blocks)"
            } else {
                a.bm = self.f.b.take(4)?; a.uni = self.f.b.take(4)?; a.ucnt = self.f.b.take(n_rb.max(1) * 4)?;
                ki::score(&self.f.g.l(), &a)?;
                "tp.prefill.idx (prefill shape)"
            };
            crate::dsv41::fwd::receipts::kernel(path, "d41_idx_score_r8", nq);
            if self.f.checks {
                let e = self.f.g.down::<i32>(err, 1)?[0];
                ensure!(e == 0, "indexer layer {l}: kernel error word {e}");
            }
        }
        Ok(keep)
    }
}

/// Leg OPT1: parse `nvidia-smi --query-gpu=name,clocks.current.graphics --format=csv,noheader,nounits` sampled during
/// GPU warm-up; warn when a GB10 boosts past 2,250 MHz (factory-unlocked clocks trigger GB10 package/LPDDR5X power throttling).
pub fn gb10_unlocked_clock_warn(csv: &str) -> Option<String> {
    for line in csv.lines() {
        let (name, mhz_str) = line.rsplit_once(',')?;
        if !name.contains("GB10") { continue; }
        let mhz: u32 = mhz_str.trim().parse().ok()?;
        if mhz > 2250 {
            return Some(format!(
                "GB10 SM clock boosted to {mhz} MHz (> 2250 MHz factory-unlocked): GB10 package/LPDDR5X power throttling can cut sustained decode throughput by 17-32%; lock clocks with 'sudo nvidia-smi -lgc 0,2200' (or '0,1800' in warm rooms)"
            ));
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------------------------------
// The head's `Forward` for the lane loop
// ---------------------------------------------------------------------------------------------------------------------

/// Leg CE2: the `full_at` of a prefill resumed at `from` of a `total`-token prompt that snapshots at `at` (the shell's plan): the
/// lowest snapshot strictly inside the prompt, so every mid-prompt snapshot of the plan is a full state (`Schedule::new_with`).
pub fn full_at_for(from: usize, total: usize, at: &[usize]) -> Option<usize> { at.iter().copied().filter(|&k| k > from && k < total).min() }

/// Pending prefix-state ids the pool dropped (a `Snapshot`'s handle pushes its id here when it is dropped).
type FreeQ = Arc<Mutex<Vec<u32>>>;

struct SnapHandle { id: u32, q: FreeQ }
impl Drop for SnapHandle { fn drop(&mut self) { if let Ok(mut q) = self.q.lock() { q.push(self.id); } } }

/// The lane loop's forward on the head: every call is an op broadcast to the nodes and run here.
pub struct TpForward<'a, 'b> {
    pub s: &'b mut TpServe<'a>,
    vocab: usize,
    max_seq: usize,
    free_q: FreeQ,
    next_id: u32,
    dead: bool,
    cfg_min_suffix: usize,
    /// per lane: the prompt length of the prefill in progress, and whether its first call is still to come
    pf_total: Vec<Option<usize>>,
    pf_first: Vec<bool>,
    /// per lane (leg CE2): the lowest mid-prompt snapshot position of the prefill in progress (its state must be full)
    pf_full_at: Vec<Option<usize>>,
    /// leg PP1: the tokens of the lane's next prefill call (`Forward::prefill_lookahead`), sent with the current call
    pf_ahead: Vec<Vec<u32>>,
    /// leg VIS1: per lane's in-progress VL inputs `(token_types, images)` on the head, retained until prefill finishes for cold fallback
    pf_vl: Vec<Option<(Vec<i8>, Vec<crate::dsv41::vision::preproc::PreparedImage>)>>,
    /// leg MEM1: the prefix pool's allowance from the box's free memory (created on the lane loop's first ask)
    gov: Option<super::mem::Governor>,
}

impl<'a, 'b> TpForward<'a, 'b> {
    pub fn new(s: &'b mut TpServe<'a>, vocab: usize, max_seq: usize) -> Self {
        let n = s.m.lanes.len();
        TpForward { s, vocab, max_seq, free_q: Arc::new(Mutex::new(Vec::new())), next_id: 1, dead: false, cfg_min_suffix: crate::dsv41::ced::schedule::MIN_KEEP,
                    pf_total: vec![None; n], pf_first: vec![true; n], pf_full_at: vec![None; n], pf_ahead: vec![Vec::new(); n],
                    pf_vl: (0..n).map(|_| None).collect(), gov: None }
    }

    /// Encode any uncached images (`img.end > cached`) on Rank 0's GPU and populate `vl_aligner` + `vl_row_ops`.
    fn encode_vl_on_head(&mut self, lane: usize, cached: usize) -> Result<()> {
        let Some((ref types, ref imgs)) = self.pf_vl[lane] else { return Ok(()) };
        let uncached: Vec<&crate::dsv41::vision::preproc::PreparedImage> = imgs.iter().filter(|im| im.end > cached).collect();
        if uncached.is_empty() {
            let (ops, _) = crate::dsv41::vision::tower::build_vl_row_ops(types, &[]);
            self.s.m.lanes[lane].vl_aligner = None;
            self.s.m.lanes[lane].vl_row_ops = ops;
            return Ok(());
        }
        let vis = self.s.m.vision.as_ref().context("rank 0: vision weights are not loaded")?;
        let total_align_rows: usize = uncached.iter().map(|im| im.n_aligner_tokens()).sum();
        let uncached_span_tokens: usize = uncached.iter().map(|im| im.end - im.start).sum();
        let align_buf = self.s.m.f.g.buf((total_align_rows * vis.hidden * 2).max(256))?;
        let t0 = std::time::Instant::now();
        let mut row_off = 0usize;
        let mut encoded_spans = Vec::with_capacity(uncached.len());
        for im in &uncached {
            let dst = align_buf.ptr() + (row_off * vis.hidden * 2) as u64;
            vis.encode_image(self.s.m.f.g, &mut self.s.m.f.b, &self.s.m.f.scr, im, dst)?;
            row_off += im.n_aligner_tokens();
            encoded_spans.push((im.start, im.end));
        }
        self.s.m.f.g.sync()?;
        let us = t0.elapsed().as_micros() as u64;
        m::add(&m::VIS_IMAGES, uncached.len() as u64);
        m::add(&m::VIS_TOKENS, uncached_span_tokens as u64);
        m::add(&m::VIS_ENCODE_US, us);
        let (ops, count) = crate::dsv41::vision::tower::build_vl_row_ops(types, &encoded_spans);
        ensure!(count == total_align_rows, "vl_row_ops aligner count {count} != encoded rows {total_align_rows}");
        self.s.m.lanes[lane].vl_aligner = Some(align_buf);
        self.s.m.lanes[lane].vl_row_ops = ops;
        Ok(())
    }

    /// The pool's allowance at the end of the boot and the line that says it (None = the asked budget fits).
    pub fn pool_line(&mut self, budget: u64) -> Option<String> {
        let snap_max = super::mem::snap_bytes(self.s.m.cfg, self.max_seq);
        let g = self.gov.get_or_insert_with(|| super::mem::Governor::new(budget, snap_max));
        let free = super::mem::avail()?;
        if g.floor == 0 { return None; }
        let a = super::mem::pool_allowance(0, free, g.floor, g.reserve, budget);
        if a >= budget { return None; }
        let gb = |b: u64| b as f64 / 1e9;
        Some(format!("prefix pool: {:.1} GB of the asked {:.1} GB for now: this box has {:.1} GB free, and the pool keeps {:.1} GB above the {:.1} GB memory-watchdog floor (one 1.5 GB reserve + one state of the longest context); it grows as memory frees up and gives states back before the floor (velogb10_dsv41_mem_*)",
                     gb(a), gb(budget), gb(free), gb(g.reserve), gb(g.floor)))
    }

    fn flush_frees(&mut self) -> Result<()> { self.flush_frees_inner(false).map(|_| ()) }

    fn flush_frees_inner(&mut self, trim_tight: bool) -> Result<usize> {
        let ids: Vec<u32> = match self.free_q.lock() { Ok(mut q) => std::mem::take(&mut *q), Err(_) => Vec::new() };
        let n = ids.len();
        for id in ids { self.op(Op::Free { id }, None)?; }
        if n > 0 && trim_tight {
            let tight = self.gov.as_ref().map_or(true, |g| g.floor > 0 && g.last_allowance.map_or(true, |a| a < g.max));
            if tight { self.s.upkeep(Some(std::time::Duration::ZERO)); }
        }
        Ok(n)
    }

    /// Run an op; any error leaves the ranks in different states, so the forward is dead from then on.
    fn op(&mut self, op: Op, toks: Option<&[u32]>) -> Result<Done> {
        if self.dead { bail!("the TP forward stopped after an earlier failure (the ranks may disagree): restart the server"); }
        if op != Op::Nop && op != Op::Park {
            // leg HT1: a parked server wakes its nodes before the op's header (one byte each), then runs as always
            if let Err(e) = self.s.idle.begin_op() {
                self.dead = true;
                m::inc(&m::TP_FATAL);
                crate::reprintln!("\n*** FATAL: dsv41 TP: {e:#}\n*** the server answers 503 from now on; restart it ***\n");
                return Err(e);
            }
            if let Some(io) = self.s.m.eio.as_ref() { io.wake(); }
        }
        let r = self.s.head_op(op, toks);
        self.s.idle.op_done();
        match r {
            Ok(d) => Ok(d),
            Err(e) => {
                self.dead = true;
                eprintln!("\n*** FATAL: dsv41 TP op {} failed on the head: {e:#}\n*** the ranks can no longer be trusted to agree; the server answers 503 from now on ***\n", op.name());
                Err(e)
            }
        }
    }
}

use crate::dsv41::serve::forward::{Forward, RowOut, Snapshot, StepRow};
use crate::dsv41::serve::sampler::Sampling;
use crate::dsv41::serve::metrics as m;

impl Forward for TpForward<'_, '_> {
    fn vocab(&self) -> usize { self.vocab }
    fn lanes(&self) -> usize { self.s.m.lanes.len() }
    fn max_seq_len(&self) -> usize { self.max_seq }
    fn name(&self) -> &'static str { "dsv41-tp" }
    fn rounds(&self) -> bool { true }
    fn fatal(&self) -> bool { self.dead }
    fn mid_prompt_keep(&self) -> usize { self.cfg_min_suffix }
    fn begin_prefill(&mut self, lane: usize, total: usize) { self.pf_total[lane] = Some(total); self.pf_first[lane] = true; self.pf_full_at[lane] = None; self.pf_vl[lane] = None; }
    fn prefill_snapshots(&mut self, lane: usize, from: usize, total: usize, at: &[usize]) { self.pf_full_at[lane] = full_at_for(from, total, at); }

    fn set_vl_inputs(&mut self, lane: usize, cached: usize, token_types: Vec<i8>, images: Vec<crate::dsv41::vision::preproc::PreparedImage>) -> Result<()> {
        if token_types.is_empty() { return Ok(()); }
        self.flush_frees()?;
        m::inc(&m::VIS_REQS);
        let packed = pack_vl_types(&token_types);
        let len = token_types.len() as u32;
        self.pf_vl[lane] = Some((token_types, images));
        self.op(Op::SetVl { lane, len }, Some(&packed))?;
        self.encode_vl_on_head(lane, cached)
    }

    /// Leg CE2: the exact check the lane driver applies on a resumed prefill (`Schedule::resumed`), from the head's copy of
    /// the state (every rank holds the same), BEFORE the shell commits the lane to it.
    fn can_resume(&self, snap: &Snapshot, at: usize, prompt_len: usize) -> Result<()> {
        let h = snap.data.downcast_ref::<SnapHandle>().context("can_resume: a foreign snapshot")?;
        let st = self.s.snaps.get(&h.id).with_context(|| format!("can_resume: no state {} on the head", h.id))?;
        ensure!(at <= st.pos && st.pos - at <= self.s.m.cfg.backtrack(), "a state at {} cannot be rolled back to {at}", st.pos);
        schedule::resume_check(self.s.m.cfg, self.s.mode, prompt_len, at, st.kind)?;
        ensure!(tail_resume_ok(prompt_len, at, st.tail.as_ref().map(|_| st.pos)),
                "a {prompt_len}-token prompt resumed at {at} needs its last rows' taps, which the state at {} does not carry", st.pos);
        Ok(())
    }
    /// Leg CE3: the TP CED forward's mid-prompt states are encoder-only: the shell snapshots cold prompts only at call ends below
    /// the decoder cone (free), and keeps the prompt end (full, with its tail).
    fn encoder_cone(&self) -> Option<usize> { Some(schedule::snapshot_label(self.s.m.cfg, self.s.mode, StateKind::Encoder)) }
    fn prefill_tail(&self) -> usize { super::spec::TAP_ROWS }
    fn prefill_lookahead(&mut self, lane: usize, next: &[u32]) {
        if let Some(a) = self.pf_ahead.get_mut(lane) { a.clear(); a.extend_from_slice(next); }
    }
    fn force(&mut self, lane: usize, drop: usize, tok: u32) -> Result<()> { self.op(Op::Force { lane, drop, tok }, None).map(|_| ()) }
    fn lane_pos(&self, lane: usize) -> Option<usize> { self.s.m.lanes.get(lane).map(|l| l.pos) }

    fn pool_allowance(&mut self, pool_bytes: u64, budget: u64) -> Option<u64> {
        if self.dead { return None; }
        let snap_max = super::mem::snap_bytes(self.s.m.cfg, self.max_seq);
        let g = self.gov.get_or_insert_with(|| super::mem::Governor::new(budget, snap_max));
        g.max = budget;
        g.poll(pool_bytes)
    }

    fn state_bytes(&self, pos: usize) -> Option<u64> { Some(super::mem::snap_bytes(self.s.m.cfg, pos)) }
    fn has_tail_snaps(&self) -> bool { true }
    fn state_bytes_for(&self, pos: usize, is_tail: bool) -> Option<u64> {
        let tail = if is_tail { (pos.min(super::spec::TAP_ROWS) * self.s.m.cfg.dspark_targets.len() * self.s.m.cfg.hidden * 2) as u64 } else { 0 };
        Some(super::mem::snap_bytes(self.s.m.cfg, pos) + tail)
    }
    fn flush_frees(&mut self) -> Result<()> { self.flush_frees_inner(true).map(|_| ()) }

    fn prefill(&mut self, lane: usize, from: usize, tokens: &[u32]) -> Result<RowOut> {
        self.flush_frees()?;
        ensure!(self.s.m.lanes[lane].pos == from, "lane {lane} holds {} positions, asked to resume at {from}", self.s.m.lanes[lane].pos);
        let total = self.pf_total[lane].unwrap_or(from + tokens.len()).max(from + tokens.len());
        let first = std::mem::replace(&mut self.pf_first[lane], false);
        // the same value on every call of this prompt (it is part of the schedule every call rebuilds)
        let full_at = self.pf_full_at[lane].unwrap_or(0) as u32;
        let ln = &self.s.m.lanes[lane];
        if first && from > 0 {
            // leg CE2, defence in depth: a lane resumed from a state this prompt cannot resume exactly is NOT sent to the ranks (the
            // refusal there would stop the forward for good); the lane is reset and the prompt's known rows are prefilled cold
            let tail_ps = self.s.lane_tail[lane].as_ref().map(|t| t.0);
            let chk = schedule::resume_check(self.s.m.cfg, self.s.mode, total, ln.pos, ln.kind)
                .and_then(|_| if tail_resume_ok(total, ln.pos, tail_ps) { Ok(()) } else { Err(anyhow::anyhow!("the prompt's last rows' taps are not in the state")) });
            if let Err(e) = chk {
                let mut all = ln.hist.clone();
                all.extend_from_slice(tokens);
                m::inc(&m::TP_RESUME_COLD);
                crate::reprintln!("dsv41: lane {lane}: the restored prefix state ({:?} at {from}) cannot resume this {total}-token prompt ({e:#}); prefilling it cold instead (velogb10_dsv41_tp_resume_cold_fallback_total)", ln.kind);
                self.op(Op::Release { lane }, None)?;
                if let Some((ref types, _)) = self.pf_vl[lane] {
                    let packed = pack_vl_types(types);
                    let len = types.len() as u32;
                    self.op(Op::SetVl { lane, len }, Some(&packed))?;
                    self.encode_vl_on_head(lane, 0)?;
                }
                self.op(Op::Prefill { lane, taps: true, total: total as u32, first: true, full_at, ahead: 0 }, Some(&all))?;
                if from + tokens.len() >= total { self.pf_total[lane] = None; self.pf_first[lane] = true; self.pf_full_at[lane] = None; self.pf_vl[lane] = None; }
                m::inc(&m::TP_PREFILLS);
                m::add(&m::TP_PREFILL_TOKENS, all.len() as u64);
                return Ok(RowOut::default());
            }
        }
        // leg PP1: the next call's tokens ride this call (the ranks read their Engram rows while this call computes)
        let ahead = std::mem::take(&mut self.pf_ahead[lane]);
        let ahead = if from + tokens.len() < total && !ahead.is_empty() { ahead } else { Vec::new() };
        if ahead.is_empty() {
            self.op(Op::Prefill { lane, taps: true, total: total as u32, first, full_at, ahead: 0 }, Some(tokens))?;
        } else {
            let mut all = Vec::with_capacity(tokens.len() + ahead.len());
            all.extend_from_slice(tokens);
            all.extend_from_slice(&ahead);
            self.op(Op::Prefill { lane, taps: true, total: total as u32, first, full_at, ahead: ahead.len() as u32 }, Some(&all))?;
        }
        if from + tokens.len() >= total { self.pf_total[lane] = None; self.pf_first[lane] = true; self.pf_full_at[lane] = None; self.pf_vl[lane] = None; }
        m::inc(&m::TP_PREFILLS);
        m::add(&m::TP_PREFILL_TOKENS, tokens.len() as u64);
        Ok(RowOut::default())
    }

    fn prefill_batch(&mut self, items: &[crate::dsv41::serve::forward::PrefillItem<'_>]) -> Result<Vec<RowOut>> {
        if items.len() < 2 || matches!(self.s.mode, CedMode::Replay { .. }) {
            return items.iter().map(|it| self.prefill(it.lane, it.from, it.tokens)).collect();
        }
        self.flush_frees()?;
        // Verify every lane is eligible for clean single-call multi-lane CED prefill; fall back to solo `prefill` if any needs cold fallback or VL/mid-snap
        let mut total_new = 0usize;
        let mut mask = 0u32;
        for it in items {
            let lane = it.lane;
            if self.pf_vl[lane].is_some() || self.pf_full_at[lane].is_some() || it.tokens.is_empty() {
                return items.iter().map(|x| self.prefill(x.lane, x.from, x.tokens)).collect();
            }
            if self.s.m.lanes[lane].pos != it.from || (mask & (1 << lane)) != 0 {
                return items.iter().map(|x| self.prefill(x.lane, x.from, x.tokens)).collect();
            }
            let total = it.from + it.tokens.len();
            if self.pf_total[lane].unwrap_or(total) > total {
                return items.iter().map(|x| self.prefill(x.lane, x.from, x.tokens)).collect();
            }
            if it.from > 0 {
                let ln = &self.s.m.lanes[lane];
                let tail_ps = self.s.lane_tail[lane].as_ref().map(|t| t.0);
                if schedule::resume_check(self.s.m.cfg, self.s.mode, total, ln.pos, ln.kind).is_err()
                    || !tail_resume_ok(total, ln.pos, tail_ps)
                {
                    return items.iter().map(|x| self.prefill(x.lane, x.from, x.tokens)).collect();
                }
            }
            mask |= 1 << lane;
            total_new += it.tokens.len();
        }
        if total_new > self.s.m.max_rows {
            return items.iter().map(|x| self.prefill(x.lane, x.from, x.tokens)).collect();
        }
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by_key(|&i| items[i].lane);
        let mut payload = Vec::with_capacity(items.len() + total_new);
        for &idx in &order { payload.push(items[idx].tokens.len() as u32); }
        for &idx in &order { payload.extend_from_slice(items[idx].tokens); }
        let taps_mask = if self.s.sp.drafter_off { 0 } else { mask };
        self.op(Op::PrefillBatch { count: items.len(), mask, taps_mask }, Some(&payload))?;
        for it in items {
            let lane = it.lane;
            self.pf_total[lane] = None;
            self.pf_first[lane] = true;
            self.pf_full_at[lane] = None;
            self.pf_vl[lane] = None;
            self.pf_ahead[lane].clear();
        }
        m::inc(&m::TP_BATCH_PREFILLS);
        m::inc(&m::TP_PREFILLS);
        m::add(&m::TP_PREFILL_TOKENS, total_new as u64);
        Ok(vec![RowOut::default(); items.len()])
    }

    fn step(&mut self, _rows: &[StepRow]) -> Result<Vec<RowOut>> { bail!("the TP forward serves in rounds (Forward::round), not per-token steps") }

    fn start(&mut self, lane: usize, s: &Sampling, seed: u64) -> Result<u32> {
        self.start_ex(lane, s, seed, false)
    }

    fn start_ex(&mut self, lane: usize, s: &Sampling, seed: u64, suppress_eos: bool) -> Result<u32> {
        let temp = if s.greedy() { 0.0 } else { s.temperature };
        let pen = [s.presence_penalty, s.frequency_penalty, s.repetition_penalty];
        if !s.greedy() && (s.top_p < 1.0 || s.top_k > 0) {
            self.op(Op::Cut { lane, top_k: s.top_k.min(u32::MAX as usize) as u32, top_p: s.top_p }, None)?;
        }
        Ok(self.op(Op::Start { lane, temp, seed, pen, suppress_eos }, None)?.tok)
    }

    fn start_lanes(&mut self, items: &[crate::dsv41::serve::forward::StartItem<'_>]) -> Result<Vec<u32>> {
        if items.len() < 2 {
            return items.iter().map(|it| self.start_ex(it.lane, it.sampling, it.seed, it.suppress_eos)).collect();
        }
        if items.iter().any(|it| !self.s.has_start_row[it.lane]) {
            return items.iter().map(|it| self.start_ex(it.lane, it.sampling, it.seed, it.suppress_eos)).collect();
        }
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by_key(|&i| items[i].lane);
        let mut mask = 0u32;
        let mut sup_mask = 0u32;
        let mut words = Vec::with_capacity(items.len() * 8);
        for &idx in &order {
            let it = &items[idx];
            let s = it.sampling;
            let temp = if s.greedy() { 0.0 } else { s.temperature };
            let (top_k, top_p) = if s.greedy() { (0u32, 1.0f32) } else { (s.top_k.min(u32::MAX as usize) as u32, s.top_p) };
            mask |= 1 << it.lane;
            if it.suppress_eos { sup_mask |= 1 << it.lane; }
            words.extend_from_slice(&[
                temp.to_bits(),
                it.seed as u32,
                (it.seed >> 32) as u32,
                s.presence_penalty.to_bits(),
                s.frequency_penalty.to_bits(),
                s.repetition_penalty.to_bits(),
                top_k,
                top_p.to_bits(),
            ]);
        }
        let d = self.op(Op::StartBatch { count: items.len(), mask, sup_mask }, Some(&words))?;
        m::inc(&m::TP_BATCH_STARTS);
        let mut out = vec![0u32; items.len()];
        for (pos_in_batch, &orig_idx) in order.iter().enumerate() {
            out[orig_idx] = d.chains[pos_in_batch][0];
        }
        Ok(out)
    }

    fn round(&mut self, lanes: &[usize]) -> Result<Vec<Vec<u32>>> {
        let ls: Vec<(usize, usize)> = lanes.iter().map(|&l| (l, 0)).collect();
        self.round_ex(&ls)
    }

    fn round_ex(&mut self, lanes_sup: &[(usize, usize)]) -> Result<Vec<Vec<u32>>> {
        self.flush_frees()?;
        let nl = lanes_sup.iter().map(|&(l, _)| l).max().map_or(0, |m| m + 1);
        let mask = lanes_sup.iter().fold(0u32, |a, &(l, _)| a | (1 << l));
        let sup_mask = lanes_sup.iter().fold(0u64, |a, &(l, sup)| a | ((sup.min(255) as u64) << (8 * l)));
        let copy = self.s.sp.copy_permission();
        let obs = self.s.sp.peek_rank0_obs_bits();
        let d = self.op(Op::Round { nl, mask, copy, sup_mask, obs }, None)?;
        Ok(lanes_sup.iter().map(|&(l, _)| d.chains.get(l).cloned().unwrap_or_default()).collect())
    }

    fn snapshot(&mut self, lane: usize, pos: usize) -> Result<Snapshot> {
        self.flush_frees()?;
        ensure!(self.s.m.lanes[lane].pos == pos, "snapshot at {pos}: lane {lane} holds {} positions", self.s.m.lanes[lane].pos);
        let id = self.next_id;
        self.next_id += 1;
        let d = self.op(Op::Snap { lane, id }, None)?;
        // leg CE2: the label the state really needs (an encoder-only state must leave its whole decoder cone, 2,541 rows in CED;
        // the old `MIN_KEEP` let the pool hand a mid-prompt state to a prompt the lane driver then refused)
        let kind = if d.snap_full { StateKind::Full } else { StateKind::Encoder };
        let min_suffix = schedule::snapshot_label(self.s.m.cfg, self.s.mode, kind);
        Ok(Snapshot { pos: d.snap_pos, bytes: d.snap_bytes, data: Box::new(SnapHandle { id, q: self.free_q.clone() }), min_suffix, tail: d.snap_tail })
    }

    fn restore(&mut self, lane: usize, snap: &Snapshot) -> Result<()> {
        let h = snap.data.downcast_ref::<SnapHandle>().context("restore: a foreign snapshot")?;
        self.op(Op::Restore { lane, id: h.id }, None)?;
        Ok(())
    }

    fn truncate(&mut self, lane: usize, to: usize) -> Result<()> {
        let pos = self.s.m.lanes.get(lane).map_or(0, |l| l.pos);
        ensure!(to <= pos && pos - to <= self.s.m.cfg.backtrack(),
                "truncate: lane {lane} at {pos} -> {to} exceeds backtrack {}", self.s.m.cfg.backtrack());
        self.op(Op::Truncate { lane, pos: to as u32 }, None).map(|_| ())
    }

    fn release(&mut self, lane: usize) {
        self.pf_vl[lane] = None;
        if self.dead { return; }
        let _ = self.flush_frees();
        let _ = self.op(Op::Release { lane }, None);
    }

    fn score(&mut self, lane: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        self.flush_frees()?;
        Ok(self.op(Op::Score { lane }, Some(tokens))?.scores)
    }

    fn idle(&mut self) -> Result<()> {
        if self.dead { return Ok(()); }
        self.flush_frees()?;
        match self.s.idle.tick() {
            super::idle::Tick::Parked => {
                // leg HT1: parked: no heartbeat (nobody is in a collective); the head's own ledger keeps moving, and a node that left
                // is noticed now rather than by the next request
                self.s.upkeep(None);
                if let Some(why) = self.s.idle.node_gone() {
                    self.dead = true;
                    m::inc(&m::TP_FATAL);
                    crate::reprintln!("\n*** FATAL: dsv41 TP: {why}\n*** the server answers 503 from now on; restart it ***\n");
                }
                Ok(())
            }
            super::idle::Tick::Park => self.op(Op::Park, None).map(|_| ()),
            super::idle::Tick::Heartbeat => { m::inc(&m::TP_NOPS); self.op(Op::Nop, None).map(|_| ()) }
        }
    }

    fn shutdown(&mut self) {
        if self.dead { return; }
        if self.s.idle.wake_nodes().is_err() { return; }
        let _ = self.s.m.bcast_small(&Op::Shutdown.encode());
    }
}

/// The program of one rank: build the model, then the head runs `head_main` (HTTP + lane loop) and a node follows.
#[allow(clippy::too_many_arguments)]
pub fn run(dev: &Arc<CudaDevice>, comm: &TpComm, src: &RankSrc, model_dir: &Path, engram_dir: &Path, so: &ServeOpts, say: &mut dyn FnMut(String),
           link: super::idle::Link, head_main: Option<&mut dyn FnMut(&mut TpServe, &mut dyn FnMut(String)) -> Result<()>>) -> Result<()> {
    let cfg: Dsv41Config = src.cfg.clone();
    super::mem::phase("start");
    crate::dsv41::gpu::load_module(dev)?;
    let cs = crate::gpu::fork_blocking_stream(dev);
    let g = Gpu { dev: dev.clone(), stream: cs.stream as u64 };
    let mut s = boot_rank(&g, &cs, &cfg, comm, src, model_dir, engram_dir, so, say)?;
    // leg HT1: the idle tier on the session's control connection (the head decides when; a node only obeys `Park`)
    let head = matches!(link, super::idle::Link::Head(_));
    s.idle = super::idle::Idle::new(link, super::idle::park_after_from_registry());
    if head { s.idle.register_for_prewake(); say(s.idle.boot_line()); }
    match head_main {
        Some(f) => f(&mut s, say),
        None => s.follow(say),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_headers_round_trip_exactly() {
        let ops = [Op::Nop, Op::Shutdown, Op::Prefill { lane: 3, taps: true, total: 262_143, first: true, full_at: 261_888, ahead: 2_176 }, Op::Prefill { lane: 0, taps: false, total: 9, first: false, full_at: 0, ahead: 0 },
                   Op::Force { lane: 2, drop: 5, tok: 128_821 }, Op::Cut { lane: 1, top_k: 40, top_p: 0.9 }, Op::Cut { lane: 3, top_k: 0, top_p: 1.0 },
                   Op::Start { lane: 2, temp: 0.7, seed: 0xDEAD_BEEF_0123_4567, pen: [0.5, -0.25, 1.1], suppress_eos: true },
                   Op::Start { lane: 0, temp: 0.0, seed: u64::MAX, pen: [0.0, 0.0, 1.0], suppress_eos: false },
                   Op::Round { nl: 4, mask: 0b1011, copy: true, sup_mask: 0x6043, obs: Some((42.5f32.to_bits(), 0.25f32.to_bits())) },
                   Op::Round { nl: 8, mask: 0xFF, copy: false, sup_mask: 0x1006_0504_0302_0106, obs: None },
                   Op::PrefillBatch { count: 4, mask: 0b1111, taps_mask: 0b1111 },
                   Op::StartBatch { count: 4, mask: 0b1111, sup_mask: 0b0101 },
                   Op::Release { lane: 1 }, Op::Snap { lane: 1, id: 0xFFFF_FFFE }, Op::Restore { lane: 3, id: 70_000 },
                   Op::Free { id: 123_456_789 }, Op::Truncate { lane: 2, pos: 262_143 }, Op::Score { lane: 1 }, Op::Park,
                   Op::SetVl { lane: 2, len: 262_143 }];
        for op in ops {
            let h = op.encode();
            assert!(h.iter().all(|v| *v >= 0.0 && *v < 16_777_216.0 && v.fract() == 0.0), "{op:?}: every word is an exact float");
            assert_eq!(Op::decode(&h).unwrap(), op);
        }
        assert!(Op::decode(&[99.0; 16]).is_err());
    }

    #[test]
    fn dsv41_pp1_prefill_broadcast_splits_the_call_rows_from_the_look_ahead() {
        assert_eq!(split_ahead(vec![1, 2, 3, 4, 5], 2).unwrap(), (vec![1, 2, 3], vec![4, 5]));
        assert_eq!(split_ahead(vec![1, 2, 3], 0).unwrap(), (vec![1, 2, 3], vec![]));
        assert!(split_ahead(vec![1, 2], 2).is_err(), "a call keeps at least one row");
        assert!(split_ahead(vec![], 0).is_err());
        // the look-ahead count rides the header exactly (a 2,048-row chunk plus the 128-row tail fits in the halves)
        let op = Op::Prefill { lane: 1, taps: true, total: 131_072, first: false, full_at: 0, ahead: 2_176 };
        assert_eq!(Op::decode(&op.encode()).unwrap(), op);
        assert_eq!(crate::dsv41::serve::engine::lookahead_end(2048, 32_092, 2048, 128), 4224);
        assert_eq!(crate::dsv41::serve::engine::lookahead_end(30_720, 32_092, 2048, 128), 32_092);
        assert_eq!(crate::dsv41::serve::engine::lookahead_end(32_092, 32_092, 2048, 128), 32_092, "nothing after the prompt");
    }

    #[test]
    fn dsv41_ce2_full_at_is_the_lowest_snapshot_strictly_inside_the_prompt() {
        assert_eq!(full_at_for(0, 5286, &[4736, 4864, 4992, 5120, 5286]), Some(4736));
        assert_eq!(full_at_for(5120, 5286, &[5286]), None);
        assert_eq!(full_at_for(4800, 5286, &[4736, 4864, 5286]), Some(4864));
        assert_eq!(full_at_for(0, 100, &[]), None);
    }

    #[test]
    fn the_thermal_governor_paces_the_ops_that_run_gpu_work_and_never_the_heartbeat() {
        // every rank calls crate::thermal::pace() at the top of exec for these ops (NIGHT1 parity with the other families)
        for op in [Op::Prefill { lane: 0, taps: true, total: 9, first: true, full_at: 0, ahead: 0 },
                   Op::PrefillBatch { count: 2, mask: 0b11, taps_mask: 0b11 },
                   Op::Start { lane: 1, temp: 1.0, seed: 3, pen: [0.0, 0.0, 1.0], suppress_eos: false },
                   Op::StartBatch { count: 2, mask: 0b11, sup_mask: 0 },
                   Op::Round { nl: 4, mask: 0b1111, copy: false, sup_mask: 0, obs: None }, Op::Score { lane: 2 }] {
            assert!(op_paces(&op), "{op:?} is GPU work");
        }
        for op in [Op::Nop, Op::Park, Op::Shutdown, Op::Release { lane: 0 }, Op::Snap { lane: 0, id: 1 }, Op::Restore { lane: 0, id: 1 }, Op::Free { id: 1 },
                   Op::Truncate { lane: 0, pos: 3 }, Op::Cut { lane: 0, top_k: 40, top_p: 0.9 }, Op::Force { lane: 0, drop: 1, tok: 5 },
                   Op::SetVl { lane: 0, len: 512 }] {
            assert!(!op_paces(&op), "{op:?} is bookkeeping: an idle server's Nop heartbeat must never sleep");
        }
        let sample_types = vec![-1i8, 0, 1, 1, 2, 3, -1];
        assert_eq!(unpack_vl_types(&pack_vl_types(&sample_types), sample_types.len()).unwrap(), sample_types);
        assert!(crate::thermal::PASS_EVERY_LAYERS >= 1 && crate::thermal::PASS_EVERY_LAYERS < 40, "the in-pass checkpoints fall inside the 40-layer trunk");
    }

    #[test]
    fn dsv41_opt1_gb10_unlocked_clock_warn_fires_only_above_2250_mhz_on_gb10() {
        assert!(gb10_unlocked_clock_warn("NVIDIA GB10, 2190\n").is_none());
        assert!(gb10_unlocked_clock_warn("NVIDIA GB10, 2200\n").is_none());
        assert!(gb10_unlocked_clock_warn("NVIDIA H100, 2500\n").is_none());
        let w = gb10_unlocked_clock_warn("NVIDIA GB10, 2418\n").expect("warns on factory-unlocked GB10 boost");
        assert!(w.contains("2418 MHz") && w.contains("nvidia-smi -lgc 0,2200"), "{w}");
    }
}
