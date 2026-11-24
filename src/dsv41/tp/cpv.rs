//! Leg P7COPY (`COPY-A` `WideCopyGuard` + `COPY-B` `COPYPF1` / `ECHOPRICE`):
//! 1. **O(1) Wide Decode Copy Rollback (`WideCopyGuard`, `7..=16` verify rows):**
//!    When a lane verifies `c` in `6..=15` copied tokens (`7..=16` rows in `TpModel::forward_segs`), `d41_kc_stash`
//!    writes only the last `7` rows into the ratio-2 KV-source frontier rings (`L2, L8, L14`), and `d41_kc_swa_write`
//!    + `d41_ds_kv_rows` write up to `16` slots into the 134-slot trunk & DSpark SWA rings. Saving a tiny `~3.1 MB`
//!    per-lane `FrontierCkpt` (`< 0.015 ms` D2D on GB10) before the call plus stashing the `<= 16` ratio-2 `(kv, score)`
//!    rows (`96 KB`) lets any rejection (`0..c` accepted) restore the rejected ring slots and exact 7-row frontier
//!    in **O(1) (`< 0.02 ms`) with ZERO re-prefill or re-decode** — 100% bitwise identical to `--mtp off`!
//! 2. **Prefill-Width Copy Stretch (`COPYPF1` + `ECHOPRICE`, `32 / 160 / 512` rows, `--dsv41-copy-prefill on`):**
//!    When a lane holds an established live `CopySpan` cursor (`>= 16` remaining tokens, text-only, no penalties)
//!    and `copy_prefill::priced_pick` (+ `1.15x` multi-lane `stretch_pays`) says the stretch pays:
//!    * saves the `~3.1 MB` `FrontierCkpt`,
//!    * runs ONE single-chunk `CedMode::Exact` pass over `[bonus, c_1 .. c_N]` (`N` up to `31 / 159 / 511`),
//!    * judges rows in `<= 16`-row groups via `TpModel::head_logits` + `TpSpec::target_cut` + `Round::stats`
//!      (DS's exact `91_ds_vocab.cuh` sampler), stopping at the first group containing a mismatch `pick != c_{a+1}`:
//!      - **full hit (`a == N`):** zero rollback, zero re-run! Seeds DSpark rings from `sp.taps` and emits `N + 1` tokens;
//!      - **mismatch (`a < N`):** restores `FrontierCkpt` (`< 0.015 ms`) and advances `[bonus, c_1 .. c_a]` (`a + 1` rows:
//!        `forward_segs` when `<= 16`, `CedMode::Exact` prefill when `> 16`), leaving zero stray state from `a+1..N`.

use std::sync::atomic::Ordering::Relaxed;

use anyhow::{ensure, Context, Result};

use super::ced::WIDE_MIN;
use super::comm::Dsv41Comm;
use super::copy::{self, fold_u32, trim_before_eos};
use super::frame::MIX_STRIDE;
use super::fwd::{HeadWant, TpModel};
use super::spec::{TpSpec, RING_Q_BYTES, RING_SLOTS, RING_S_BYTES, TAP_ROWS};
use crate::copy_lanes;
use crate::copy_prefill::{self, Decision, Facts};
use crate::dsv41::ced::schedule::{Mode as CedMode, StateKind};
use crate::dsv41::engram::hash::NgramHashState;
use crate::dsv41::fwd::dev::{Buf, Gpu};
use crate::dsv41::gpu::dspark::vocab::Rows;
use crate::dsv41::gpu::dspark::{BLOCK, WIDE_RPL};

fn a256(n: usize) -> usize { (n + 255) & !255 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LastKind {
    None,
    WideVerify,
    Stretch,
}

/// Reusable `~3.1 MB` device + host frontier checkpoint for one lane (allocated lazily on first wide/stretch round).
struct LaneCkpt {
    mem: Buf,
    win_offs: Vec<(usize, usize, usize, usize)>,
    ds_offs: [(usize, usize); 3],
    front_offs: Vec<(usize, usize, usize, usize)>,
    last_kind: LastKind,
    p0: usize,
    rows_ran: usize,
    kind: StateKind,
    n_groups: Vec<(usize, usize)>,
    hash: NgramHashState,
    committed_toks: Vec<u32>,
}

impl LaneCkpt {
    fn alloc(m: &TpModel, lane: usize) -> Result<Self> {
        let ln = &m.lanes[lane];
        let mut off = 0usize;
        let mut take = |bytes: usize| { let o = off; off += a256(bytes); o };
        let win_offs: Vec<(usize, usize, usize, usize)> = ln.win.iter()
            .map(|w| (take(w.ring_q.bytes), w.ring_q.bytes, take(w.ring_s.bytes), w.ring_s.bytes))
            .collect();
        let ds_offs = [
            (take(RING_Q_BYTES), take(RING_S_BYTES)),
            (take(RING_Q_BYTES), take(RING_S_BYTES)),
            (take(RING_Q_BYTES), take(RING_S_BYTES)),
        ];
        let mut keys: Vec<usize> = ln.src.keys().copied().collect();
        keys.sort_unstable();
        let mut front_offs = Vec::new();
        for s in keys {
            if let Some((fk, fs)) = &ln.src[&s].front {
                debug_assert_eq!(fk.bytes, fs.bytes);
                front_offs.push((s, take(fk.bytes), take(fs.bytes), fk.bytes));
            }
        }
        let mem = m.f.g.buf(off.max(256))?;
        Ok(LaneCkpt {
            mem, win_offs, ds_offs, front_offs,
            last_kind: LastKind::None, p0: 0, rows_ran: 0, kind: StateKind::Full,
            n_groups: Vec::new(), hash: NgramHashState::new(), committed_toks: Vec::new(),
        })
    }

    fn capture(&mut self, m: &TpModel, ds_rq: u64, ds_rs: u64, ds_cap: usize, lane: usize, rows: usize, lkind: LastKind) -> Result<()> {
        let g = m.f.g;
        let ln = &m.lanes[lane];
        let base = self.mem.ptr();
        for (w, &(qo, qb, so, sb)) in ln.win.iter().zip(&self.win_offs) {
            g.d2d(base + qo as u64, w.ring_q.ptr(), qb)?;
            g.d2d(base + so as u64, w.ring_s.ptr(), sb)?;
        }
        for s in 0..3 {
            let (rq, rs) = ds_ring_ptrs(ds_rq, ds_rs, ds_cap, s, lane);
            let (qo, so) = self.ds_offs[s];
            g.d2d(base + qo as u64, rq, RING_Q_BYTES)?;
            g.d2d(base + so as u64, rs, RING_S_BYTES)?;
        }
        for &(sl, fko, fso, fb) in &self.front_offs {
            let (fk, fs) = ln.src[&sl].front.as_ref().context("missing ratio>=2 frontier")?;
            g.d2d(base + fko as u64, fk.ptr(), fb)?;
            g.d2d(base + fso as u64, fs.ptr(), fb)?;
        }
        self.n_groups.clear();
        for (&sl, st) in &ln.src { self.n_groups.push((sl, st.n_groups)); }
        self.p0 = ln.pos;
        self.rows_ran = rows;
        self.kind = ln.kind;
        self.hash = ln.hash;
        self.last_kind = lkind;
        self.committed_toks.clear();
        Ok(())
    }

    fn restore_to_p0(&self, m: &mut TpModel, ds_rq: u64, ds_rs: u64, ds_cap: usize, lane: usize) -> Result<()> {
        let g = m.f.g;
        let ln = &mut m.lanes[lane];
        let base = self.mem.ptr();
        for (w, &(qo, qb, so, sb)) in ln.win.iter().zip(&self.win_offs) {
            g.d2d(w.ring_q.ptr(), base + qo as u64, qb)?;
            g.d2d(w.ring_s.ptr(), base + so as u64, sb)?;
        }
        for s in 0..3 {
            let (rq, rs) = ds_ring_ptrs(ds_rq, ds_rs, ds_cap, s, lane);
            let (qo, so) = self.ds_offs[s];
            g.d2d(rq, base + qo as u64, RING_Q_BYTES)?;
            g.d2d(rs, base + so as u64, RING_S_BYTES)?;
        }
        for &(sl, fko, fso, fb) in &self.front_offs {
            let (fk, fs) = ln.src[&sl].front.as_ref().context("missing ratio>=2 frontier")?;
            g.d2d(fk.ptr(), base + fko as u64, fb)?;
            g.d2d(fs.ptr(), base + fso as u64, fb)?;
        }
        for &(sl, ng) in &self.n_groups {
            if let Some(st) = ln.src.get_mut(&sl) { st.n_groups = ng; }
        }
        ln.pos = self.p0;
        ln.kind = self.kind;
        ln.hist.truncate(self.p0);
        if ln.vl_types.len() > self.p0 { ln.vl_types.truncate(self.p0); }
        ln.hash = self.hash;
        Ok(())
    }

    fn rollback_wide(&self, m: &mut TpModel, ds_rq: u64, ds_rs: u64, ds_cap: usize, lane: usize, new_pos: usize) -> Result<()> {
        let (p0, rows) = (self.p0, self.rows_ran);
        ensure!(self.last_kind == LastKind::WideVerify && new_pos >= p0 && new_pos <= p0 + rows && rows <= WIDE_RPL,
                "wide rollback: lane {lane} new_pos {new_pos} outside {p0}..={}", p0 + rows);
        if new_pos == p0 + rows { return Ok(()); }
        let n_emit = new_pos - p0;
        let rej = rows - n_emit;
        let g = m.f.g;
        let hd = m.cfg.head_dim;
        let slots = m.cfg.swa_slots();
        let base = self.mem.ptr();

        for (w, &(qo, _, so, _)) in m.lanes[lane].win.iter().zip(&self.win_offs) {
            restore_ring_slice(g, w.ring_q.ptr(), base + qo as u64, slots, hd, new_pos, rej)?;
            restore_ring_slice(g, w.ring_s.ptr(), base + so as u64, slots, hd / 32, new_pos, rej)?;
        }
        for s in 0..3 {
            let (rq, rs) = ds_ring_ptrs(ds_rq, ds_rs, ds_cap, s, lane);
            let (qo, so) = self.ds_offs[s];
            restore_ring_slice(g, rq, base + qo as u64, RING_SLOTS, RING_Q_BYTES / RING_SLOTS, new_pos, rej)?;
            restore_ring_slice(g, rs, base + so as u64, RING_SLOTS, RING_S_BYTES / RING_SLOTS, new_pos, rej)?;
        }

        for st in m.lanes[lane].src.values_mut() {
            st.n_groups = new_pos / st.ratio.max(1) as usize;
        }
        for &(sl, fko, fso, fb) in &self.front_offs {
            let (fk_ptr, fs_ptr, ratio) = {
                let st = &m.lanes[lane].src[&sl];
                let (fk, fs) = st.front.as_ref().context("missing ratio>=2 frontier")?;
                (fk.ptr(), fs.ptr(), st.ratio)
            };
            g.d2d(fk_ptr, base + fko as u64, fb)?;
            g.d2d(fs_ptr, base + fso as u64, fb)?;
            if n_emit > 0 {
                let rr = m.cfg.frontier_rows(ratio);
                let row_b = hd * 4;
                let (sk, ss) = m.wide_stash.get(&(lane, sl)).context("missing wide_stash for ratio>=2 KV source")?;
                for i in n_emit.saturating_sub(rr)..n_emit {
                    let slot = (p0 + i) % rr;
                    g.d2d(fk_ptr + (slot * row_b) as u64, sk.ptr() + (i * row_b) as u64, row_b)?;
                    g.d2d(fs_ptr + (slot * row_b) as u64, ss.ptr() + (i * row_b) as u64, row_b)?;
                }
            }
        }

        let ln = &mut m.lanes[lane];
        ln.pos = new_pos;
        ln.kind = StateKind::Full;
        ln.hist.truncate(new_pos);
        if ln.vl_types.len() > new_pos { ln.vl_types.truncate(new_pos); }
        m.rebuild_lane_hash(lane)
    }
}

fn ds_ring_ptrs(ds_rq: u64, ds_rs: u64, ds_cap: usize, stage: usize, lane: usize) -> (u64, u64) {
    let idx = stage * ds_cap + lane;
    (ds_rq + (idx * RING_Q_BYTES) as u64, ds_rs + (idx * RING_S_BYTES) as u64)
}

fn restore_ring_slice(g: &Gpu, dst: u64, src: u64, slots: usize, elem_bytes: usize, start_pos: usize, count: usize) -> Result<()> {
    if count == 0 { return Ok(()); }
    debug_assert!(count <= slots);
    let s0 = start_pos % slots;
    let first = count.min(slots - s0);
    let off0 = (s0 * elem_bytes) as u64;
    g.d2d(dst + off0, src + off0, first * elem_bytes)?;
    if count > first {
        g.d2d(dst, src, (count - first) * elem_bytes)?;
    }
    Ok(())
}

/// Per-`TpSpec` state for `WideCopyGuard` (`7..=16` rows) and `COPYPF1` (`32..=512` rows).
#[derive(Default)]
pub struct CpvState {
    ckpts: Vec<Option<LaneCkpt>>,
}

impl CpvState {
    pub fn dev_bytes(&self) -> u64 {
        self.ckpts.iter().flatten().map(|c| c.mem.bytes as u64).sum()
    }

    pub fn reset_lane(&mut self, lane: usize) {
        if let Some(Some(c)) = self.ckpts.get_mut(lane) {
            c.last_kind = LastKind::None;
            c.committed_toks.clear();
        }
    }

    fn lane_mut(&mut self, m: &TpModel, lane: usize) -> Result<&mut LaneCkpt> {
        if self.ckpts.len() <= lane { self.ckpts.resize_with(lane + 1, || None); }
        if self.ckpts[lane].is_none() { self.ckpts[lane] = Some(LaneCkpt::alloc(m, lane)?); }
        Ok(self.ckpts[lane].as_mut().unwrap())
    }

    pub fn save_wide_verify(&mut self, m: &TpModel, ds_rq: u64, ds_rs: u64, ds_cap: usize, parts: &[(usize, &[u32])]) -> Result<Vec<usize>> {
        let bt = m.cfg.backtrack();
        let mut wide = Vec::new();
        for &(lane, toks) in parts {
            if toks.len() > bt {
                ensure!(toks.len() <= WIDE_RPL, "wide copy verify: lane {lane} has {} rows (> {WIDE_RPL})", toks.len());
                let c = self.lane_mut(m, lane)?;
                c.capture(m, ds_rq, ds_rs, ds_cap, lane, toks.len(), LastKind::WideVerify)?;
                c.committed_toks.extend_from_slice(toks);
                wide.push(lane);
            } else if let Some(Some(c)) = self.ckpts.get_mut(lane) {
                c.last_kind = LastKind::None;
                c.committed_toks.clear();
            }
        }
        Ok(wide)
    }

    pub fn rollback_wide_verify(&mut self, m: &mut TpModel, ds_rq: u64, ds_rs: u64, ds_cap: usize, lane: usize, new_pos: usize) -> Result<()> {
        let c = self.ckpts.get_mut(lane).and_then(|x| x.as_mut()).context("missing WideCopyGuard for wide verify lane")?;
        c.rollback_wide(m, ds_rq, ds_rs, ds_cap, lane, new_pos)?;
        c.committed_toks.truncate(new_pos - c.p0);
        c.rows_ran = new_pos - c.p0;
        Ok(())
    }

    pub fn force_rollback(&mut self, m: &mut TpModel, sp: &mut TpSpec, lane: usize, new_pos: usize) -> Result<()> {
        let (ds_rq, ds_rs, ds_cap) = (sp.ring_q_ptr(), sp.ring_s_ptr(), sp.cap);
        let c = self.ckpts.get_mut(lane).and_then(|x| x.as_mut())
            .with_context(|| format!("force rollback to {new_pos}: lane {lane} has no wide/stretch checkpoint"))?;
        match c.last_kind {
            LastKind::WideVerify => {
                c.rollback_wide(m, ds_rq, ds_rs, ds_cap, lane, new_pos)?;
                c.committed_toks.truncate(new_pos - c.p0);
                c.rows_ran = new_pos - c.p0;
                Ok(())
            }
            LastKind::Stretch => {
                let p0 = c.p0;
                ensure!(new_pos >= p0 && new_pos <= p0 + c.committed_toks.len(),
                        "stretch force rollback: lane {lane} new_pos {new_pos} outside {p0}..={}", p0 + c.committed_toks.len());
                let keep = c.committed_toks[..new_pos - p0].to_vec();
                c.restore_to_p0(m, ds_rq, ds_rs, ds_cap, lane)?;
                c.committed_toks = keep.clone();
                if !keep.is_empty() {
                    advance_prefix(m, sp, lane, p0, &keep)?;
                }
                Ok(())
            }
            LastKind::None => anyhow::bail!("force rollback: lane {lane} last round was not wide/stretch"),
        }
    }
}

fn advance_prefix(m: &mut TpModel, sp: &mut TpSpec, lane: usize, p0: usize, keep: &[u32]) -> Result<()> {
    let rerun = keep.len();
    if rerun == 0 { return Ok(()); }
    if rerun <= WIDE_RPL {
        m.taps = if sp.drafter_off { None } else { Some(sp.verify_taps(m, rerun)) };
        let r = m.forward_segs(&[(lane, keep)], HeadWant::None, None);
        m.taps = None;
        r?;
        if !sp.drafter_off {
            let pos_v: Vec<i32> = (p0..p0 + rerun).map(|x| x as i32).collect();
            let sums = m.frames[0].ptr() + (rerun * (m.cfg.hidden + MIX_STRIDE) * 4) as u64;
            sp.main_rows(m, sums, rerun, &pos_v, &vec![lane as i32; rerun])?;
        }
    } else {
        let mut full = Vec::with_capacity(p0 + rerun);
        full.extend_from_slice(&m.lanes[lane].hist);
        full.extend_from_slice(keep);
        let td = if sp.drafter_off { None } else { Some(sp.taps_dst(m)) };
        m.prefill_part(lane, &full, full.len(), Some((p0, StateKind::Full)), None, rerun, CedMode::Exact, td, false)?;
        if !sp.drafter_off {
            seed_dspark_from_taps(m, sp, lane, p0 + rerun, rerun)?;
        }
    }
    Ok(())
}

fn seed_dspark_from_taps(m: &mut TpModel, sp: &mut TpSpec, lane: usize, p_end: usize, rows: usize) -> Result<()> {
    let nt = rows.min(TAP_ROWS);
    let (tb, tap_ld) = sp.taps_dst(m);
    let tap_off = p_end.min(TAP_ROWS) - nt;
    let tap_ptr = tb + (tap_off * tap_ld * 2) as u64;
    let d = m.cfg.hidden;
    let mrow = sp.mrow_ptr();
    m.main_proj_partial(tap_ptr, tap_ld, nt, mrow)?;
    m.reduce(mrow, nt * d)?;
    let pos_v: Vec<i32> = ((p_end - nt)..p_end).map(|x| x as i32).collect();
    sp.main_rows(m, mrow, nt, &pos_v, &vec![lane as i32; nt])
}

/// Check active lanes for qualifying `COPYPF1` stretches (`32 / 160 / 512` rows).
pub fn try_stretches(sp: &mut TpSpec, m: &mut TpModel, nl: usize, active: &[usize]) -> Result<Option<(Vec<usize>, Vec<Vec<u32>>)>> {
    if m.layers.len() < m.cfg.layers || m.layers[..m.cfg.layers].iter().any(|x| x.is_none()) {
        return Ok(None);
    }
    let na = active.len();
    let mut plan: Vec<(usize, usize, usize)> = Vec::new(); // (lane, rung, offered)
    {
        let Some(cd) = sp.copy.as_mut() else { return Ok(None) };
        if !cd.cfg.prefill { return Ok(None); }
        let batch_rate = cd.batch_rate(active);
        let cfg_cp = cd.cfg;
        for &lane in active {
            let pending_tok = sp.host[lane].tok as u32;
            let lc = &mut cd.lanes[lane];
            lc.sync_lane(&m.lanes[lane].hist, pending_tok, &cfg_cp);
            if lc.span.cursor().is_none() || lc.span.cursor_left() < copy_prefill::CPV_MIN_N { continue; }
            if !m.lanes[lane].vl_types.is_empty() { continue; }
            if sp.pen.as_ref().map_or(false, |p| p.active(lane)) { continue; }
            if sp.host[lane].suppress_eos > 0 { continue; }
            let seq_room = m.lanes[lane].max_seq.saturating_sub(m.lanes[lane].pos + 3);
            if seq_room < copy_prefill::CPV_MIN_N { continue; }
            let eos_room = cursor_room_before_eos(&lc.span, m.cfg.eos);
            let avail = lc.span.cursor_left().min(seq_room).min(eos_room);
            if avail < copy_prefill::CPV_MIN_N { continue; }

            let eff_max = if na >= 2 { cfg_cp.prefill_max_rows.min(160) } else { cfg_cp.prefill_max_rows };
            let top = copy_prefill::schedule_top(m.max_rows, eff_max);
            if top < WIDE_MIN.saturating_sub(1) { continue; }
            let want = copy_prefill::expected_want(lc.span.cursor_run(), lc.span.copy_len_hist());
            let offer_fn = |g: usize| copy_prefill::offered_len(g, top, m.max_rows, avail, seq_room);
            let (rung, pays) = if cfg_cp.prefill_price {
                copy_prefill::priced_pick(&cd.cpv_price, lc.cpv_rung, want, offer_fn)
            } else {
                (lc.cpv_rung, true)
            };
            let offered = offer_fn(rung);
            let facts = Facts {
                on: true,
                solo: true,
                plain_step: false,
                cursor_live: true,
                in_prefill: false,
                penalized: false,
                logprobs: false,
                image: false,
                tap_ok: true,
                scratch_ok: true,
                n: offered,
                run: lc.span.cursor_run(),
                min_run: 0,
                pays,
            };
            if let Decision::Take(n_take) = copy_prefill::decide(&facts) {
                if na >= 2 && cfg_cp.copy_lanes && batch_rate > 0.0 && cd.cpv_price.stretch_ms(rung).is_some() {
                    let e = (want.min(n_take) + 1) as f64;
                    if !copy_lanes::stretch_pays(e, cd.cpv_price.stretch_ms(rung), batch_rate) {
                        if m.rank == 0 { copy::LANES_PRICED_OUT.fetch_add(1, Relaxed); }
                        continue;
                    }
                }
                plan.push((lane, rung, n_take));
            }
        }
    }
    if plan.is_empty() { return Ok(None); }

    let mut em = vec![Vec::new(); nl];
    let mut ran_lanes = Vec::new();
    for &(lane, rung, offered) in &plan {
        if let Some(em_lane) = run_one_stretch(sp, m, lane, rung, offered)? {
            em[lane] = em_lane;
            ran_lanes.push(lane);
        }
    }
    if ran_lanes.is_empty() { return Ok(None); }
    let rem: Vec<usize> = active.iter().copied().filter(|l| !ran_lanes.contains(l)).collect();
    Ok(Some((rem, em)))
}

fn cursor_room_before_eos(span: &crate::copy_span::CopySpan, eos: u32) -> usize {
    let left = span.cursor_left();
    if left == 0 || span.cursor_in_record() { return left; }
    let Some(cur) = span.cursor() else { return 0 };
    let ctx = span.ctx();
    if cur >= ctx.len() { return 0; }
    ctx[cur..].iter().position(|&t| t == eos).unwrap_or(left)
}

fn run_one_stretch(sp: &mut TpSpec, m: &mut TpModel, lane: usize, rung: usize, offered: usize) -> Result<Option<Vec<u32>>> {
    let eos = m.cfg.eos;
    let mut prop = match sp.copy.as_mut().context("copy off")?.lanes[lane].span.propose_span(offered) {
        Some(p) => p,
        None => return Ok(None),
    };
    trim_before_eos(&mut prop.tokens, eos);
    if prop.tokens.len() < copy_prefill::CPV_MIN_N {
        return Ok(None);
    }
    prop.max_w = prop.tokens.len();
    let (em_lane, a, stretch_ms) = run_stretch_core(sp, m, lane, &prop.tokens)?;
    let n = prop.tokens.len();
    let bonus_choice = *em_lane.last().unwrap();
    if let Some(cd) = sp.copy.as_mut() {
        let lc = &mut cd.lanes[lane];
        lc.span.commit(Some(&prop), a, &em_lane);
        lc.cpv_rung = copy_prefill::next_rung(rung, a == n);
        lc.work.note(a + 1, stretch_ms);
        cd.lane_price.work.note(a + 1, stretch_ms);
        lc.judged(a, &cd.cfg);
        lc.note_gate_transition(m.rank == 0);
        cd.cpv_price.note_stretch(rung, stretch_ms);
        let mut dg = fold_u32(cd.last_digest, 0xC09F_F100 ^ (lane as u32));
        dg = fold_u32(dg, (n as u32) ^ ((a as u32) << 16));
        dg = fold_u32(dg, bonus_choice);
        cd.last_digest = dg;
    }
    Ok(Some(em_lane))
}

/// Core `COPYPF1` stretch execution over `prop_tokens` (`[c_1 .. c_N]`): returns `(emitted_tokens, accepted_count, stretch_ms)`.
fn run_stretch_core(sp: &mut TpSpec, m: &mut TpModel, lane: usize, prop_tokens: &[u32]) -> Result<(Vec<u32>, usize, f64)> {
    let p0 = m.lanes[lane].pos;
    ensure!(p0 == sp.host[lane].vbase as usize, "cpv stretch: lane {lane} pos {p0} != vbase {}", sp.host[lane].vbase);
    let bonus = sp.host[lane].tok as u32;
    let n = prop_tokens.len();
    let rows = n + 1;
    ensure!(n >= copy_prefill::CPV_MIN_N && rows <= m.max_rows, "cpv stretch: {rows} rows (max_rows {})", m.max_rows);

    let t0 = std::time::Instant::now();
    let (ds_rq, ds_rs, ds_cap) = (sp.ring_q_ptr(), sp.ring_s_ptr(), sp.cap);
    {
        let ckpt = sp.cpv.lane_mut(m, lane)?;
        ckpt.capture(m, ds_rq, ds_rs, ds_cap, lane, rows, LastKind::Stretch)?;
    }

    let mut full = Vec::with_capacity(p0 + rows);
    full.extend_from_slice(&m.lanes[lane].hist);
    full.push(bonus);
    full.extend_from_slice(prop_tokens);
    let td = if sp.drafter_off { None } else { Some(sp.taps_dst(m)) };
    m.prefill_part(lane, &full, full.len(), Some((p0, StateKind::Full)), None, rows, CedMode::Exact, td, false)?;

    let vl = sp.sh.vlocal;
    let saved_act: Vec<i32> = sp.host.iter().map(|l| l.active).collect();
    let orig_lane = sp.host[lane];
    for l in &mut sp.host { l.active = 0; }
    sp.host[lane].active = 1;
    let greedy = orig_lane.inv_t == 0.0;
    let nl_judge = lane + 1;

    let mut a = n;
    let mut bonus_choice = 0u32;
    let judge_res = (|| -> Result<()> {
        let mut r0 = 0usize;
        while r0 < rows {
            let m_grp = (rows - r0).min(WIDE_RPL);
            let dst = sp.tlog.ptr() + (lane * WIDE_RPL * vl * 4) as u64;
            m.head_logits(r0, m_grp, dst)?;
            sp.host[lane].vbase = (p0 + r0) as i32;
            sp.host[lane].k = (m_grp - 1) as i32;
            sp.host[lane].copy = 1;
            sp.round.set_lanes(m.f.g, &sp.host[..nl_judge])?;
            let tcut = sp.target_cut(m.f.g, nl_judge, m_grp, WIDE_RPL * vl)?;
            let sres = sp.round.stats(m.f.g, &sp.x, sp.sh, Rows { ptr: sp.tlog.ptr(), lane_stride: WIDE_RPL * vl, row_stride: vl }, nl_judge, m_grp, tcut)?;
            let recs = sp.round.recs(m.f.g, sres, nl_judge * m_grp)?;
            for j in 0..m_grp {
                let rec = &recs[lane * m_grp + j];
                let pick = (if greedy { rec.aid } else { rec.gid }) as u32;
                let idx = r0 + j;
                if idx < n {
                    if pick != prop_tokens[idx] {
                        a = idx;
                        bonus_choice = pick;
                        return Ok(());
                    }
                } else {
                    bonus_choice = pick;
                }
            }
            r0 += m_grp;
        }
        Ok(())
    })();
    for (i, act) in saved_act.into_iter().enumerate() { sp.host[i].active = act; }
    sp.host[lane] = orig_lane;
    judge_res?;

    if a == n {
        if !sp.drafter_off {
            seed_dspark_from_taps(m, sp, lane, p0 + rows, rows)?;
        }
    } else {
        {
            let ckpt = sp.cpv.ckpts[lane].as_ref().unwrap();
            ckpt.restore_to_p0(m, ds_rq, ds_rs, ds_cap, lane)?;
        }
        let keep = &full[p0..p0 + a + 1];
        advance_prefix(m, sp, lane, p0, keep)?;
    }
    if let Some(Some(ckpt)) = sp.cpv.ckpts.get_mut(lane) {
        ckpt.committed_toks.clear();
        ckpt.committed_toks.extend_from_slice(&full[p0..p0 + a + 1]);
    }

    m.f.g.sync()?;
    m.comm.check("a copy prefill stretch")?;
    let mut stretch_ms = t0.elapsed().as_secs_f64() * 1e3;
    if m.world > 1 {
        let send = if m.rank == 0 { [stretch_ms as f32, a as f32, bonus_choice as f32] } else { [0.0; 3] };
        let got = m.bcast_small(&send)?;
        stretch_ms = (got[0] as f64).max(0.05);
        ensure!(got[1] as usize == a && got[2] as u32 == bonus_choice,
                "rank {}: cpv stretch disagreement: local (a={a}, tok={bonus_choice}) vs head (a={}, tok={})",
                m.rank, got[1] as usize, got[2] as u32);
    }

    let mut ln = orig_lane;
    ln.vbase = (p0 + a + 1) as i32;
    ln.wend = (p0 + a) as i32;
    ln.tok = bonus_choice as i32;
    ln.k = n as i32;
    ln.n_acc = a as i32;
    ln.n_emit = (a + 1) as i32;
    ln.need_res = 0;
    ln.zmass = 0;
    ln.copy = 1;
    ln.rounds = ln.rounds.wrapping_add(1);
    ln.drafted = ln.drafted.wrapping_add(n as u32);
    ln.accepted = ln.accepted.wrapping_add(a as u32);
    sp.host[lane] = ln;

    let st_lane = &mut sp.stats[lane];
    st_lane.rounds += 1;
    st_lane.drafted += n as u64;
    st_lane.accepted += a as u64;
    st_lane.emitted += (a + 1) as u64;
    sp.held[lane] = BLOCK;

    let mut em_lane = Vec::with_capacity(a + 1);
    em_lane.extend_from_slice(&prop_tokens[..a]);
    em_lane.push(bonus_choice);

    if m.rank == 0 {
        copy::ROUNDS.fetch_add(1, Relaxed);
        copy::LANE_ROUNDS.fetch_add(1, Relaxed);
        copy::PROPOSED.fetch_add(n as u64, Relaxed);
        copy::ACCEPTED.fetch_add(a as u64, Relaxed);
        copy::ALL_COPY_ROUNDS.fetch_add(1, Relaxed);
        copy::DRAFT_SKIPPED.fetch_add(1, Relaxed);
        copy::PREFILL_STEPS.fetch_add(1, Relaxed);
        copy::PREFILL_OFFERED.fetch_add(n as u64, Relaxed);
        copy::PREFILL_ACCEPTED.fetch_add(a as u64, Relaxed);
        if a < n { copy::PREFILL_RERUN_ROWS.fetch_add((a + 1) as u64, Relaxed); }
    }
    Ok((em_lane, a, stretch_ms))
}

#[cfg(test)]
pub(super) fn run_stretch_for_test(sp: &mut TpSpec, m: &mut TpModel, lane: usize, rung: usize, prop_tokens: &[u32]) -> Result<Vec<u32>> {
    let (em, a, ms) = run_stretch_core(sp, m, lane, prop_tokens)?;
    if let Some(cd) = sp.copy.as_mut() {
        cd.lanes[lane].cpv_rung = copy_prefill::next_rung(rung, a == prop_tokens.len());
        cd.cpv_price.note_stretch(rung, ms);
    }
    Ok(em)
}

#[cfg(test)]
pub(super) fn seed_dspark_for_test(m: &mut TpModel, sp: &mut TpSpec, lane: usize, p_end: usize, rows: usize) -> Result<()> {
    seed_dspark_from_taps(m, sp, lane, p_end, rows)
}

#[cfg(test)]
pub(super) fn advance_prefix_for_test(m: &mut TpModel, sp: &mut TpSpec, lane: usize, p0: usize, keep: &[u32]) -> Result<()> {
    advance_prefix(m, sp, lane, p0, keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_p7copy_cursor_room_before_eos_stops_cleanly_at_eos() {
        let mut ctx: Vec<u32> = (100..130).collect();
        ctx.push(1); // EOS = 1 at offset +30 (`120..130` = 10 tokens before EOS)
        ctx.extend(131..150);
        ctx.push(999);
        ctx.extend(100..115);
        let mut cs = crate::copy_span::CopySpan::new(&ctx, crate::copy_span::CopySpanCfg::new(12).with_start_w(4));
        let p = cs.propose(15).unwrap();
        assert_eq!(p.tokens.len(), 4);
        let em_exact: Vec<u32> = (115..120).collect();
        cs.commit(Some(&p), 4, &em_exact);
        assert!(cs.cursor().is_some());
        assert_eq!(cursor_room_before_eos(&cs, 1), 10);
        assert!(cursor_room_before_eos(&cs, 99999) > 25);
    }
}
