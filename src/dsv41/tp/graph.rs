//! DSV4.1 TP (leg FW): CUDA graphs of the decode / verify calls (and the DSpark draft step), replayed at every context.
//!
//! A call is captured once per KEY: its lane segments (lane, rows), the index layers' context class (PK1's decode plan
//! cuts the keys differently above 16K visible keys), the head rows and whether the DSpark taps ride along. Everything a
//! captured call reads per round comes from device memory: the positions, ids and lanes of the rows (uploaded before the
//! replay), the Engram rows (EIO's step runs eagerly before the replay into its fixed buffer), the head's row parameters.
//! In graph mode (`TpModel::gm`) the forward takes the graph-safe pieces: the window keys and the compressor with the
//! positions read on the device (`44_kv_graph.cuh`: KC's arithmetic line for line), the attention's KV bound and the
//! indexer's caps at the cache capacity (bounds only: the selections never reach past the visible groups), the
//! row-block tables prepared before the capture, the decode rows read from the ring only. The arenas, frames, caches
//! and scratch are persistent, so the captured addresses stay valid; the host restores the arena state after a replay.

use super::fwd::TpModel;
use crate::dsv41::fwd::dev::Buf;
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use std::collections::HashMap;

/// The tables a captured call reads: per index layer its row-block table, per layer the rows' ring pointers.
pub struct GTab {
    rbs: HashMap<usize, (Buf, Vec<u64>)>,
    /// u64 [layers][2][rows] (q planes, then s planes)
    rings: Buf,
    rows: usize,
}

impl GTab {
    pub fn rbs_of(&self, l: usize) -> Option<(u64, Vec<u64>)> { self.rbs.get(&l).map(|(b, v)| (b.ptr(), v.clone())) }
    pub fn ring_tab(&self, l: usize, rows: usize) -> (u64, u64) {
        let base = self.rings.ptr() + (l * 2 * self.rows * 8) as u64;
        let _ = rows;
        (base, base + (self.rows * 8) as u64)
    }
}

/// One captured call (the template graph is destroyed once instantiated: the exec does not need it).
struct Entry {
    exec: sys::CUgraphExec,
    _tab: Option<GTab>,
    /// the final stream / record of the call (the head's input) and the arena state after it
    last: (u64, u64),
    arena: (u64, usize),
    replays: u64,
    /// the replay clock at the last use (the least recently used entry is destroyed above `mem::GRAPH_CAP`)
    used: u64,
    /// the device memory the instantiation took (`cuMemGetInfo` before / after; 0 when the box moved the other way)
    bytes: u64,
    /// leg PD1: what one execution of the captured call does (counted on /metrics at every replay)
    pd1: super::pd1::CallStats,
}

/// Captured graphs alive in this process (`velogb10_dsv41_mem_graphs_live`).
static LIVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn live() -> u64 { LIVE.load(std::sync::atomic::Ordering::Relaxed) }

fn free_mem() -> u64 {
    let (mut f, mut t) = (0usize, 0usize);
    if unsafe { sys::cuMemGetInfo_v2(&mut f, &mut t) } == sys::CUresult::CUDA_SUCCESS { f as u64 } else { 0 }
}

/// The key of a captured call.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Key { pub segs: Vec<(usize, usize)>, pub big: Vec<bool>, pub head: Vec<usize>, pub taps: bool }

/// Leg PG1: graph keys independent of how a call's rows are split between lanes (`--dsv41-graph-keys rows`, the default).
/// A decode / verify call is captured once per (lane slots = the highest lane in the call + 1, total rows, context class
/// [, above PK1's 16K-key plan: its rows per block and row-block count]) and reads its segmentation from device memory:
/// `seg` = i32 [lanes][2] (row0, rows) per lane (0 rows = the lane has no row in this call), `lid` = i32 [rows] the lane of
/// each row, the window ring table and the indexer's row-block table rewritten before every replay. `split` = the keys of
/// leg FW (one graph per (lane, rows) vector): kept for the A/B.
pub struct RowKeys { pub on: bool, pub seg: Buf, pub lid: Buf, pub live: bool }

/// Marker slots of a rows key (never a lane index).
pub const K_SLOTS: usize = usize::MAX - 1;
pub const K_ROWS: usize = usize::MAX - 2;
pub const K_R: usize = usize::MAX - 3;
pub const K_NRB: usize = usize::MAX - 4;

/// The rows-key parts of a call: `segs` = (lane, rows) per segment in call order, `big` = the context class per index
/// source, `plan` = (rows per block, row blocks) of PK1's decode plan when any source is above its row-key threshold.
pub fn rows_key(segs: &[(usize, usize)], plan: Option<(usize, usize)>, lanes: usize) -> Vec<(usize, usize)> {
    let slots = slot_class(segs.iter().map(|s| s.0 + 1).max().unwrap_or(0), lanes);
    let rows: usize = segs.iter().map(|s| s.1).sum();
    let mut k = vec![(K_SLOTS, slots), (K_ROWS, rows)];
    if let Some((r, n)) = plan { k.push((K_R, r)); k.push((K_NRB, n)); }
    k
}

/// The lane slots a rows-keyed call runs its per-lane steps over: the highest lane of the call + 1, rounded up to a power
/// of two (1, 2, 4, ...) and capped at the model's lanes: a lane alone on lane 0 runs one slot (no empty launches), and
/// the key space stays small (4 lanes: 1, 2 or 4 slots).
pub fn slot_class(highest_plus_one: usize, lanes: usize) -> usize { highest_plus_one.max(1).next_power_of_two().min(lanes.max(1)) }

/// The segment table of a call for `lanes` lane slots: (row0, rows) per lane, 0 rows for a lane not in the call; and the
/// lane of every row.
pub fn seg_table(segs: &[(usize, usize, usize)], lanes: usize) -> (Vec<i32>, Vec<i32>) {
    let mut t = vec![0i32; 2 * lanes];
    let mut lid = Vec::new();
    for &(lane, row0, n) in segs {
        t[2 * lane] = row0 as i32;
        t[2 * lane + 1] = n as i32;
        lid.extend(std::iter::repeat(lane as i32).take(n));
    }
    (t, lid)
}

/// The captured calls of one model.
#[derive(Default)]
pub struct Graphs { map: HashMap<Key, Entry>, seen: std::collections::HashSet<Key>, pub captures: u64, pub replays: u64, pub failed: Option<String>,
                    clock: u64, pub evictions: u64 }

impl Drop for Graphs {
    fn drop(&mut self) {
        for (_, e) in self.map.drain() {
            unsafe { sys::cuGraphExecDestroy(e.exec); }
            LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

fn ok(r: sys::CUresult, what: &str) -> Result<()> { ensure!(r == sys::CUresult::CUDA_SUCCESS, "{what}: {r:?}"); Ok(()) }

impl Graphs {
    pub fn len(&self) -> usize { self.map.len() }
    /// The device memory the live instantiations took (as measured at capture).
    pub fn bytes(&self) -> u64 { self.map.values().map(|e| e.bytes).sum() }
    /// Destroy the least recently used entries until `room` more fit under `mem::GRAPH_CAP` (its key stays seen: a later call
    /// of that shape is captured again at once, its lazy tables exist).
    fn make_room(&mut self, room: usize) {
        while self.map.len() + room > super::mem::GRAPH_CAP {
            let Some(k) = self.map.iter().min_by_key(|(_, e)| e.used).map(|(k, _)| k.clone()) else { break };
            if let Some(e) = self.map.remove(&k) {
                unsafe { sys::cuGraphExecDestroy(e.exec); }
                LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                self.evictions += 1;
                super::metrics::graph_evicted();
            }
        }
    }
    pub fn line(&self) -> String {
        format!("graphs: {} decode / verify call shapes captured ({} destroyed above {}), {} replays{}", self.map.len(), self.evictions, super::mem::GRAPH_CAP, self.replays,
                self.failed.as_ref().map(|f| format!(" (capture failed once, eager since: {f})")).unwrap_or_default())
    }
}

impl TpModel<'_> {
    /// The context class of the call's index layers (the plan's shape changes at PK1's PLAN_ROW_KEYS).
    fn idx_big(&self) -> Vec<bool> {
        let mut v = Vec::new();
        for &s in &self.cfg.index_sources {
            let kv = match self.cfg.kv_source_of(s) { Some(k) => k, None => continue };
            let n = self.segs.iter().map(|sg| self.lanes[sg.lane].src.get(&kv).map_or(0, |x| x.n_groups)).max().unwrap_or(0);
            v.push(n > crate::dsv41::gpu::indexer::PLAN_ROW_KEYS);
        }
        v
    }

    /// The tables of the call in flight (row-block tables for every index layer, ring pointers for every layer).
    fn graph_tables(&mut self, rows: usize) -> Result<GTab> {
        let (rbv, t) = self.table_data(rows)?;
        let g = self.f.g;
        let mut rbs = HashMap::new();
        for (l, v) in rbv {
            let b = g.buf(v.len() * 8)?;
            g.h2d(b.ptr(), crate::dsv41::fwd::layer::bytemuck_u64(&v))?;
            rbs.insert(l, (b, v));
        }
        let rings = g.buf(t.len() * 8)?;
        g.h2d(rings.ptr(), crate::dsv41::fwd::layer::bytemuck_u64(&t))?;
        Ok(GTab { rbs, rings, rows })
    }

    /// Leg PG1: rewrite a rows-keyed graph's tables with the call in flight (the same sizes: the key fixes them), on the
    /// compute stream before the replay.
    fn refresh_tables(&self, tab: &GTab, rows: usize) -> Result<()> {
        ensure!(tab.rows == rows, "graph tables of {} rows replayed for {rows}", tab.rows);
        let (rbv, t) = self.table_data(rows)?;
        let g = self.f.g;
        for (l, v) in rbv {
            let (b, old) = tab.rbs.get(&l).context("a row-block table missing from the captured call")?;
            ensure!(old.len() == v.len(), "layer {l}: {} row-block words for a graph captured with {}", v.len(), old.len());
            g.h2d_async(b.ptr(), &v)?;
        }
        g.h2d_async(tab.rings.ptr(), &t)?;
        Ok(())
    }

    /// The tables' contents for the call in flight: per index layer its row-block table, and the rings table.
    fn table_data(&self, rows: usize) -> Result<(Vec<(usize, Vec<u64>)>, Vec<u64>)> {
        use crate::dsv41::gpu::indexer as ki;
        let mut rbs = Vec::new();
        for &l in &self.cfg.index_sources {
            let s = self.cfg.kv_source_of(l).context("index source without a KV source")?;
            let mut cap = 1usize;
            for sg in &self.segs { cap = cap.max(self.lanes[sg.lane].src.get(&s).context("index source state")?.n_groups); }
            let p = ki::decode_plan(&self.segs.iter().map(|sg| (sg.row0, sg.n)).collect::<Vec<_>>(), cap);
            let mut v = Vec::new();
            for &(r0, n) in &p.rbs {
                let sg = self.segs.iter().find(|sg| (r0 as usize) >= sg.row0 && (r0 as usize) < sg.row0 + sg.n).context("row block outside the call")?;
                let st = self.lanes[sg.lane].src.get(&s).context("index source state")?;
                v.extend_from_slice(&[st.iq.ptr(), st.is.ptr(), (r0 as u32 as u64) | ((n as u32 as u64) << 32), 0]);
            }
            rbs.push((l, v));
        }
        let nl = self.cfg.layers;
        let mut t = vec![0u64; nl * 2 * rows];
        for l in 0..nl {
            for sg in &self.segs {
                let w = &self.lanes[sg.lane].win[l];
                for i in 0..sg.n {
                    t[l * 2 * rows + sg.row0 + i] = w.ring_q.ptr();
                    t[l * 2 * rows + rows + sg.row0 + i] = w.ring_s.ptr();
                }
            }
        }
        Ok((rbs, t))
    }

    /// Leg PG1: the decode plan's (rows per block, row blocks) of the call when an index source is above PK1's row-key
    /// threshold (those shapes follow the split), else None.
    fn big_plan(&self) -> Option<(usize, usize)> {
        if !self.idx_big().iter().any(|&b| b) { return None; }
        let p = crate::dsv41::gpu::indexer::decode_plan(&self.segs.iter().map(|sg| (sg.row0, sg.n)).collect::<Vec<_>>(), usize::MAX);
        Some((p.r, p.rbs.len()))
    }

    /// Replay (capturing first if new) the decode / verify call laid out by `plan_call` (rows uploaded, Engram rows
    /// written, the head's row parameters on the device). `body` runs the call's kernels (it is called once, under
    /// capture). Returns false when graphs are off or the capture failed (the caller then runs the call eagerly).
    pub fn graph_call(&mut self, rows: usize, head: &[usize], body: &mut dyn FnMut(&mut Self) -> Result<()>) -> Result<bool> {
        if self.graphs.as_ref().map_or(true, |g| g.failed.is_some()) { return Ok(false); }
        let rk = self.rk.on && head.is_empty() && self.segs.iter().all(|s| s.n <= super::fwd::graph_seg_max(self.cfg) && s.n <= 8);
        let segs = if rk { rows_key(&self.segs.iter().map(|s| (s.lane, s.n)).collect::<Vec<_>>(), self.big_plan(), self.lanes.len()) }
                   else { self.segs.iter().map(|s| (s.lane, s.n)).collect() };
        let key = Key { segs, big: self.idx_big(), head: head.to_vec(), taps: self.taps.is_some() };
        self.rk.live = rk;
        let r = self.graph_run(key, Some(rows), body);
        self.rk.live = false;
        r
    }

    /// Replay (capturing first if new) the work of `body` under `key`: `rows` = Some for a forward call (graph mode,
    /// prepared tables, the final stream and arena state restored after a replay), None for other work whose inputs and
    /// outputs live in persistent buffers (the DSpark draft step).
    pub fn graph_run(&mut self, key: Key, rows: Option<usize>, body: &mut dyn FnMut(&mut Self) -> Result<()>) -> Result<bool> {
        if self.graphs.as_ref().map_or(true, |g| g.failed.is_some()) { return Ok(false); }
        let st = self.f.g.stream as sys::CUstream;
        if !self.graphs.as_ref().unwrap().map.contains_key(&key) {
            // the first call of a shape runs eagerly: it builds whatever the pieces create lazily on first use (expert
            // tables, module functions), which a capture must not do; the second call is captured
            if self.graphs.as_mut().unwrap().seen.insert(key.clone()) {
                super::metrics::graph_shapes(self.graphs.as_ref().unwrap().seen.len());
                return Ok(false);
            }
            let tab = match rows { Some(r) => Some(self.graph_tables(r)?), None => None };
            self.f.g.sync()?;
            self.gtab = tab;
            self.gm = rows.is_some();
            let mut graph: sys::CUgraph = std::ptr::null_mut();
            let r0 = unsafe { sys::cuStreamBeginCapture_v2(st, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED) };
            let res = if r0 != sys::CUresult::CUDA_SUCCESS { Err(anyhow::anyhow!("cuStreamBeginCapture: {r0:?}")) } else { body(self) };
            let r1 = unsafe { sys::cuStreamEndCapture(st, &mut graph) };
            self.gm = false;
            let tab = self.gtab.take();
            if let Err(e) = res.and_then(|_| ok(r1, "cuStreamEndCapture")) {
                if !graph.is_null() { unsafe { sys::cuGraphDestroy(graph); } }
                let msg = format!("{e:#}");
                self.graphs.as_mut().unwrap().failed = Some(msg.clone());
                anyhow::bail!("dsv41 TP graph capture failed ({msg}); the call cannot be re-run eagerly after a partial capture");
            }
            self.graphs.as_mut().unwrap().make_room(1);
            let f0 = free_mem();
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            let r = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
            unsafe { sys::cuGraphDestroy(graph); }
            ok(r, "cuGraphInstantiate")?;
            let bytes = f0.saturating_sub(free_mem());
            let last = if rows.is_some() { self.last.context("the captured call left no final stream")? } else { (0, 0) };
            let arena = (self.f.b.base.ptr(), self.f.b.off);
            let pd1 = if rows.is_some() { self.pd1.stats } else { Default::default() };
            let gr = self.graphs.as_mut().unwrap();
            gr.map.insert(key.clone(), Entry { exec, _tab: tab, last, arena, replays: 0, used: 0, bytes, pd1 });
            LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            gr.captures += 1;
            super::metrics::graph_capture();
        }
        // leg PG1: a rows-keyed call replays with this call's segmentation (its tables rewritten before the launch)
        if self.rk.live {
            let tab = self.graphs.as_mut().unwrap().map.get_mut(&key).unwrap()._tab.take();
            let r = match (&tab, rows) { (Some(t), Some(n)) => self.refresh_tables(t, n), _ => Ok(()) };
            self.graphs.as_mut().unwrap().map.get_mut(&key).unwrap()._tab = tab;
            r?;
        }
        let gr = self.graphs.as_mut().unwrap();
        gr.clock += 1;
        let clock = gr.clock;
        let e = gr.map.get_mut(&key).unwrap();
        e.used = clock;
        ok(unsafe { sys::cuGraphLaunch(e.exec, st) }, "cuGraphLaunch")?;
        e.replays += 1;
        gr.replays += 1;
        super::metrics::graph_replays(1);
        super::pd1::add(&e.pd1);
        let (last, arena) = (e.last, e.arena);
        if rows.is_none() { return Ok(true); }
        self.last = Some(last);
        // the host's arena state as after the captured call (head_logits and the like allocate after it)
        if self.f.b.base.ptr() != arena.0 { self.swap_arenas_only(); }
        ensure!(self.f.b.base.ptr() == arena.0, "graph replay: arena state");
        self.f.b.off = arena.1;
        Ok(true)
    }
}

#[cfg(test)]
mod pg1_tests {
    use super::*;

    /// Every split of a call over up to 4 lanes (any active subset, 1..6 rows a lane, the caps 16 and 24) maps to a key
    /// that depends only on (lane-slot class, total rows); the segment table and the row -> lane map carry the split.
    /// Distinct keys a 4-lane server can meet: 6 + 12 + 16 = 34 at the 16-row cap, 6 + 12 + 24 = 42 at 24 (+ the drafter's
    /// 4 per-lane-count graphs: 38 / 46 of the 64-graph cache); the split keys met 2,074 / 2,400 vectors.
    #[test]
    fn dsv41_pg1_keys_follow_total_rows_not_the_split() {
        for (cap, want_rows, want_split) in [(16usize, 34usize, 2074usize), (24, 42, 2400)] {
            let mut rows_keys = std::collections::HashSet::new();
            let mut split_keys = std::collections::HashSet::new();
            for mask in 1u32..16 {
                let lanes: Vec<usize> = (0..4).filter(|l| mask >> l & 1 == 1).collect();
                let combos = 6usize.pow(lanes.len() as u32);
                for code in 0..combos {
                    let n: Vec<usize> = (0..lanes.len()).map(|i| 1 + (code / 6usize.pow(i as u32)) % 6).collect();
                    let total: usize = n.iter().sum();
                    if total > cap { continue; }
                    let segs: Vec<(usize, usize)> = lanes.iter().copied().zip(n.iter().copied()).collect();
                    let k = rows_key(&segs, None, 4);
                    let slots = slot_class(*lanes.last().unwrap() + 1, 4);
                    assert_eq!(k, vec![(K_SLOTS, slots), (K_ROWS, total)]);
                    rows_keys.insert(k);
                    split_keys.insert(segs.clone());
                    // the data the graph reads: each row's lane, each lane's (row0, rows)
                    let mut r0 = 0;
                    let s3: Vec<(usize, usize, usize)> = segs.iter().map(|&(l, c)| { let x = (l, r0, c); r0 += c; x }).collect();
                    let (t, lid) = seg_table(&s3, 4);
                    assert_eq!(lid.len(), total);
                    for &(l, row0, c) in &s3 {
                        assert_eq!((t[2 * l] as usize, t[2 * l + 1] as usize), (row0, c));
                        assert!(lid[row0..row0 + c].iter().all(|&x| x as usize == l));
                    }
                    for l in 0..4 { if !lanes.contains(&l) { assert_eq!(t[2 * l + 1], 0, "an absent lane has no rows"); } }
                }
            }
            assert_eq!(rows_keys.len(), want_rows, "rows keys at cap {cap}");
            assert_eq!(split_keys.len(), want_split, "split vectors at cap {cap}");
            assert!(rows_keys.len() + 4 <= super::super::mem::GRAPH_CAP - 16, "rows keys + drafter graphs leave room in the cache");
        }
        // a lane alone on lane 0 runs one slot; lanes 0 + 1: two; lanes up to 2 or 3: four (a 4-lane model)
        assert_eq!((slot_class(1, 4), slot_class(2, 4), slot_class(3, 4), slot_class(4, 4), slot_class(3, 3)), (1, 2, 4, 4, 3));
        // above PK1's 16K-key plan the decode plan's shape joins the key
        assert_eq!(rows_key(&[(0, 3), (1, 2)], Some((4, 2)), 4), vec![(K_SLOTS, 2), (K_ROWS, 5), (K_R, 4), (K_NRB, 2)]);
    }
}

