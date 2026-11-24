//! CE: one lane's per-layer prefill state on the device, and its prefix snapshots.
//!
//! Per layer: the SWA window (KC's ring of 134 slots, FP8 + E8M0/32, slot = position % 134, plus the current
//! chunk's linear rows, which the attention reads for positions >= `lin_base`), and for a KV source its caches
//! (main KV 288-B records, index-K planes, the ratio-2 compressor's frontier ring of r + 5 f32 rows). No Engram
//! state: the n-gram windows are rebuilt from the token ids (`EngramHasher::state_at`).
//!
//! **Ring order inside a chunk** (X-DI §2: "store only after the cache load"): the chunk's window rows go to the
//! linear planes only; the attention reads history from the ring and this chunk from the planes; THEN
//! [`LayerLane::commit_ring`] copies the chunk's last 134 window rows into the ring. Writing the ring first would
//! overwrite history the chunk's own queries read (a chunk of >= 8 rows overwrites slots of positions it needs).
//!
//! A snapshot copies every buffer a later prefill or decode reads, so a restored lane is bit-identical to the
//! lane that was snapshotted (gated: prefix resume == cold).

use super::schedule::StateKind;
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::fwd::dev::{Buf, Gpu};
use crate::dsv41::fwd::layer::{SrcState, Window};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// One layer's state.
pub struct LayerLane {
    pub layer: usize,
    pub win: Window,
    pub src: Option<SrcState>,
}

/// Bytes of one SWA row in the two planes.
const SWA_Q: usize = 512;
const SWA_S: usize = 16;

impl LayerLane {
    /// `lin_cap` = the most window rows one chunk writes; `max_seq` sizes a KV source's caches.
    pub fn new(g: &Gpu, cfg: &Dsv41Config, layer: usize, lin_cap: usize, max_seq: usize) -> Result<LayerLane> {
        let (hd, slots) = (cfg.head_dim, cfg.swa_slots());
        ensure!(hd == SWA_Q, "head_dim {hd}: the SWA planes are built for 512");
        let win = Window { ring_q: g.buf(slots * hd)?, ring_s: g.buf(slots * hd / 32)?, lin_q: g.buf(lin_cap.max(1) * hd)?, lin_s: g.buf(lin_cap.max(1) * hd / 32)?,
                           lin_base: i64::MAX, rows: lin_cap };
        let src = if cfg.kv_sources.contains(&layer) && cfg.ratio(layer) > 0 {
            let ratio = cfg.ratio(layer);
            let cap = max_seq / ratio as usize + 1;
            let rr = cfg.frontier_rows(ratio);
            Some(SrcState { layer, ratio, n_groups: 0, kv_main: g.buf(cap * 288)?, iq: g.buf(cap * 64)?, is: g.buf(cap * 4)?,
                            front: if ratio >= 2 { Some((g.buf(rr * hd * 4)?, g.buf(rr * hd * 4)?)) } else { None }, latent: g.buf(256)? })
        } else { None };
        Ok(LayerLane { layer, win, src })
    }

    /// The last min(slots, n) of the `n` linear window rows written at `lin_base` go to their ring slots.
    pub fn commit_ring(&mut self, g: &Gpu, n: usize) -> Result<()> {
        let slots = (self.win.ring_q.bytes / SWA_Q) as i64;
        let take = (n as i64).min(slots);
        let base = self.win.lin_base;
        ensure!(base >= 0, "commit_ring with no linear rows");
        let mut j = n as i64 - take; // first linear row to commit
        while j < n as i64 {
            let pos = base + j;
            let slot = pos % slots;
            let run = (slots - slot).min(n as i64 - j); // contiguous slots until the wrap or the end
            g.d2d(self.win.ring_q.ptr() + (slot as usize * SWA_Q) as u64, self.win.lin_q.ptr() + (j as usize * SWA_Q) as u64, run as usize * SWA_Q)?;
            g.d2d(self.win.ring_s.ptr() + (slot as usize * SWA_S) as u64, self.win.lin_s.ptr() + (j as usize * SWA_S) as u64, run as usize * SWA_S)?;
            j += run;
        }
        // later calls of this lane see only the ring until they write their own linear rows
        self.win.lin_base = i64::MAX;
        Ok(())
    }

    /// Every state byte, for "nothing else is read" checks: fill with a pattern no computation produces.
    pub fn poison(&mut self, g: &Gpu, v: u8) -> Result<()> {
        for b in self.bufs() { g.memset(b.ptr(), b.bytes, v)?; }
        if let Some(s) = &mut self.src { s.n_groups = 0; }
        self.win.lin_base = i64::MAX;
        Ok(())
    }

    fn bufs(&self) -> Vec<&Buf> {
        let mut v = vec![&self.win.ring_q, &self.win.ring_s];
        if let Some(s) = &self.src {
            v.extend([&s.kv_main, &s.iq, &s.is]);
            if let Some((a, b)) = &s.front { v.extend([a, b]); }
        }
        v
    }
}

/// A lane's layers (only the layers a run holds: the slice's, or a TP rank's).
#[derive(Default)]
pub struct Lane { pub layers: BTreeMap<usize, LayerLane>, /** positions [0, pos) are in the state */ pub pos: usize }

impl Lane {
    pub fn get(&self, l: usize) -> Result<&LayerLane> { self.layers.get(&l).with_context(|| format!("lane holds no state for layer {l}")) }
    pub fn get_mut(&mut self, l: usize) -> Result<&mut LayerLane> { self.layers.get_mut(&l).with_context(|| format!("lane holds no state for layer {l}")) }
    pub fn src(&self, l: usize) -> Result<&SrcState> { self.get(l)?.src.as_ref().with_context(|| format!("layer {l} is not a KV source")) }
    pub fn poison(&mut self, g: &Gpu, v: u8) -> Result<()> {
        for ll in self.layers.values_mut() { ll.poison(g, v)?; }
        self.pos = 0;
        Ok(())
    }
}

/// A prefix state: device copies of every layer's ring and KV-source caches at `pos`.
pub struct LaneSnap {
    pub pos: usize,
    pub kind: StateKind,
    layers: Vec<LayerSnap>,
    pub bytes: u64,
}

struct LayerSnap { layer: usize, ring_q: Buf, ring_s: Buf, src: Option<SrcSnap> }
struct SrcSnap { n_groups: usize, kv_main: Buf, iq: Buf, is: Buf, front: Option<(Buf, Buf)> }

fn copy_of(g: &Gpu, b: &Buf, bytes: usize) -> Result<Buf> {
    let n = bytes.min(b.bytes);
    let c = g.buf(n.max(1))?;
    g.d2d(c.ptr(), b.ptr(), n)?;
    Ok(c)
}

impl LaneSnap {
    /// Hand the state to the serve shell's prefix pool (`serve::prefix::PrefixPool`): `min_suffix` from the schedule
    /// that took it (`Schedule::min_suffix(kind)`: 134 for a full state, the decoder cone for an encoder-only one).
    pub fn into_serve(self, min_suffix: usize) -> crate::dsv41::serve::forward::Snapshot {
        crate::dsv41::serve::forward::Snapshot { pos: self.pos, bytes: self.bytes, min_suffix, data: Box::new(self), tail: false }
    }
    /// The device state inside a pool entry (None: another forward's snapshot).
    pub fn of_serve(s: &crate::dsv41::serve::forward::Snapshot) -> Option<&LaneSnap> { s.data.downcast_ref::<LaneSnap>() }

    /// Copy the lane's state at `lane.pos` (the caller says what kind it is: `Schedule::state_kind_at`).
    pub fn take(g: &Gpu, lane: &Lane, kind: StateKind) -> Result<LaneSnap> {
        let mut layers = Vec::new();
        let mut bytes = 0u64;
        for (&l, ll) in &lane.layers {
            ensure!(ll.win.lin_base == i64::MAX, "layer {l}: snapshot inside a chunk (uncommitted linear rows)");
            let ring_q = copy_of(g, &ll.win.ring_q, ll.win.ring_q.bytes)?;
            let ring_s = copy_of(g, &ll.win.ring_s, ll.win.ring_s.bytes)?;
            bytes += (ring_q.bytes + ring_s.bytes) as u64;
            let src = match &ll.src {
                Some(s) => {
                    let n = s.n_groups;
                    let front = match &s.front { Some((a, b)) => Some((copy_of(g, a, a.bytes)?, copy_of(g, b, b.bytes)?)), None => None };
                    let ss = SrcSnap { n_groups: n, kv_main: copy_of(g, &s.kv_main, n * 288)?, iq: copy_of(g, &s.iq, n * 64)?, is: copy_of(g, &s.is, n * 4)?, front };
                    bytes += (n * (288 + 64 + 4)) as u64 + ss.front.as_ref().map_or(0, |(a, b)| (a.bytes + b.bytes) as u64);
                    Some(ss)
                }
                None => None,
            };
            layers.push(LayerSnap { layer: l, ring_q, ring_s, src });
        }
        Ok(LaneSnap { pos: lane.pos, kind, layers, bytes })
    }

    /// Make `lane` hold exactly this state (every layer of the snapshot; the lane may hold garbage before).
    pub fn restore(&self, g: &Gpu, lane: &mut Lane) -> Result<()> {
        for s in &self.layers {
            let ll = lane.get_mut(s.layer)?;
            g.d2d(ll.win.ring_q.ptr(), s.ring_q.ptr(), s.ring_q.bytes.min(ll.win.ring_q.bytes))?;
            g.d2d(ll.win.ring_s.ptr(), s.ring_s.ptr(), s.ring_s.bytes.min(ll.win.ring_s.bytes))?;
            ll.win.lin_base = i64::MAX;
            match (&s.src, &mut ll.src) {
                (Some(ss), Some(d)) => {
                    let n = ss.n_groups;
                    ensure!(n * 288 <= d.kv_main.bytes, "layer {}: snapshot of {n} groups > the lane's caches", s.layer);
                    g.d2d(d.kv_main.ptr(), ss.kv_main.ptr(), n * 288)?;
                    g.d2d(d.iq.ptr(), ss.iq.ptr(), n * 64)?;
                    g.d2d(d.is.ptr(), ss.is.ptr(), n * 4)?;
                    if let (Some((a, b)), Some((da, db))) = (&ss.front, &d.front) {
                        g.d2d(da.ptr(), a.ptr(), a.bytes.min(da.bytes))?;
                        g.d2d(db.ptr(), b.ptr(), b.bytes.min(db.bytes))?;
                    }
                    d.n_groups = n;
                }
                (None, None) => {}
                _ => anyhow::bail!("layer {}: snapshot and lane disagree on the KV source", s.layer),
            }
        }
        lane.pos = self.pos;
        Ok(())
    }
}
