//! Engram runtime, part 2 (leg EIO): [`EngramIo`], the per-rank row runtime.
//!
//! What one rank does per step (decode round or prefill chunk):
//! 1. The caller hashes the step's fed tokens with ENGa's `EngramHasher` (`peek` for a verify block,
//!    `hash_chunk` for a prompt chunk) and calls [`EngramIo::submit_decode`] / [`EngramIo::submit_prefill`]
//!    with the ids (`[t][layer][column]`, 48 per token) and the liveness mask.
//!
//! Channels: a decode round is split by Engram layer, one channel each (`ch_decode(l)`): the first Engram
//! layer's rows are needed one layer into the verify (they feed the layer-0 FFN reduce), the second's about
//! 13 layers later. Each decode channel publishes on its own, the first layer's reads go out at high
//! priority (`io::ReadJob::hi`), and the round launches one bounded wait per layer, so only half of a
//! round's rows sit on the critical path. The prefill channel (`ch_prefill`) carries both layers per chunk.
//! 2. The planner (the caller's thread) maps every (Engram layer, token, owned column) to its record in the
//!    rank's head files, looks it up in the pinned row cache ([`super::cache::RowCache`]) and hands the misses
//!    to the reader pool ([`super::io::ReaderPool`]). It writes the step's slot table straight into the
//!    channel's descriptor ring in device-mapped memory.
//! 3. When the last row of the step has landed (the reader thread that completes it, or the planner when
//!    everything was a hit), the step is PUBLISHED: one release store of the channel's sequence word.
//!    Steps of a channel publish in order.
//! 4. On the GPU, `d41_engram_step_decode` (launched by the round, graph-safe) waits for that word with a
//!    bounded in-stream wait, counts a stall if it had to wait, and decodes the rows to bf16 in the rank's
//!    `wkv` K-slice layout (and/or copies the raw records for the gather plug-in). Its last block bumps the
//!    channel's done word, which retires the step's cache references.
//!
//! Warm requests ([`EngramIo::warm`]) read rows into the cache without a GPU consumer: the draft-token
//! prefetch (the sampled token's and the drafts' rows, read while the drafter runs) and the prompt
//! read-ahead (the next prefill chunk's rows, read while the current chunk runs). A later `submit` of the
//! same tokens finds them resident, or waits on the reads already in flight (never reads a row twice).
//!
//! OF's verdict plugs in at the output: `MODE_DECODE` gives the rank's columns as the K-slice input of a
//! partial `wkv` (T0 §3.6: the partial rides the layer-0 / layer-13 FFN reduce); `MODE_RECORDS` gives the raw
//! 264-B records of the rank's columns for an exchange, and `d41_engram_records_decode` decodes gathered
//! records on the row owner (T0 §7 item 8, the prefill exchange).
//!
//! Invariants: every submitted step of a channel is consumed by exactly one `launch` (a round that is
//! abandoned calls [`EngramIo::reset_channel`] with the stream idle); a slot is never recycled while a step
//! that names it is unretired; a failed read fails its steps (status `ERR_HOST`), the GPU then writes no row.

use super::cache::{key_of, RowCache};
use super::io::{DoneFn, HeadFile, ReadJob, ReaderConfig, ReaderPool};
use super::io_metrics as m;
use super::layout::EngramLayout;
use super::manifest::EngramIoError;
use crate::dsv41::gpu::engram as ge;
use anyhow::{bail, Context, Result};
use cudarc::driver::CudaDevice;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The decode channel of Engram layer index `l` (0 = model layer 1, 1 = model layer 14); the prefill channel
/// is `EngramIo::ch_prefill()` (= the number of Engram layers).
pub const fn ch_decode(l: usize) -> usize { l }
/// Descriptor-ring depth of a decode channel (steps the host may run ahead of the GPU).
pub const DECODE_RING: usize = 8;
pub const PREFILL_RING: usize = 3;
/// Status written into a descriptor before its step is published (never seen by the GPU: the sequence
/// word only moves past a step after its final status is stored).
const STATUS_PENDING: u32 = 0xFFFF_FFFF;

#[derive(Clone, Debug)]
pub struct EngramIoConfig {
    /// directory of `engram_head{h}.{bin,manifest.json}`
    pub dir: PathBuf,
    /// the Engram heads this rank serves (sorted, unique): `EngramLayout::heads_for_rank`, or all for `replica`
    pub heads: Vec<usize>,
    /// pinned row-cache size; 0 = no reuse (every row read from disk; slots only stage in-flight rows)
    pub cache_mib: usize,
    pub readers: usize,
    /// io_uring entries per reader (the most reads one reader keeps in flight)
    pub ring_entries: u32,
    /// read alignment (4096 = ENGb's decision; 512 works on these NVMe)
    pub align: u64,
    /// a reader parks after this long without work (it spins before that)
    pub park_after: Duration,
    /// most fed positions in one decode step (lanes x rows per lane)
    pub decode_max_tok: usize,
    /// most positions in one prefill chunk (0 = no prefill channel)
    pub prefill_max_tok: usize,
    /// the GPU's bounded wait for a step's rows (fail-stop beyond it)
    pub wait_timeout: Duration,
    /// NVMe keep-alive period while the readers are awake (zero = off); see `io::ReaderConfig::keepalive`
    pub keepalive: Duration,
}

impl EngramIoConfig {
    pub fn new(dir: PathBuf, heads: Vec<usize>) -> Self {
        EngramIoConfig { dir, heads, cache_mib: 2048, readers: 2, ring_entries: 256, align: 4096, park_after: Duration::from_millis(250),
                         decode_max_tok: 24, prefill_max_tok: 0, wait_timeout: Duration::from_secs(2), keepalive: Duration::from_millis(2) }
    }
}

/// What `submit` did (for the caller's accounting and the bench).
#[derive(Clone, Copy, Debug)]
pub struct StepTicket {
    pub channel: usize,
    pub seq: u64,
    pub n_tok: usize,
    pub rows: usize,
    pub hits: usize,
    pub misses: usize,
    /// rows that hit a slot still being read (by a warm request or an earlier step)
    pub joined: usize,
    pub dead: usize,
}

struct LoadState {
    waiters: Vec<u64>,
    failed: bool,
}
struct StepState {
    chan: Option<usize>,
    seq: u64,
    pending: u32,
    status: u32,
    t_submit: Instant,
}

/// Raw addresses the publisher needs (mapped memory owned by `Chan`, alive as long as `EngramIo`).
#[derive(Clone, Copy)]
struct ChanPtrs {
    ctrl: *mut u8,
    ring: *mut u8,
    stride: usize,
    depth: usize,
    /// metrics kind: 0 = decode, 1 = prefill
    kind: usize,
    /// the first Engram layer's decode channel (its publish latency is the critical one)
    critical: bool,
}

struct Shared {
    loading: HashMap<u32, LoadState>,
    steps: HashMap<u64, StepState>,
    order: Vec<VecDeque<u64>>,
    failed_slots: Vec<u32>,
    warm_done: Vec<u64>,
    ptrs: Vec<ChanPtrs>,
    /// host submit -> publish ns of every published step, per channel (the bench drains it)
    lat_ns: Vec<Vec<u64>>,
}
// The raw pointers address mapped memory owned by `EngramIo`, which outlives the reader threads.
unsafe impl Send for Shared {}

impl Shared {
    fn finish_slot(&mut self, slot: u32, err: bool) {
        let Some(ls) = self.loading.get_mut(&slot) else { return };
        let waiters = std::mem::take(&mut ls.waiters);
        if err {
            ls.failed = true;
            self.failed_slots.push(slot);
        } else {
            self.loading.remove(&slot);
        }
        for sid in waiters {
            if let Some(st) = self.steps.get_mut(&sid) {
                st.pending -= 1;
                if err { st.status = ge::ERR_HOST; }
                if st.pending == 0 && st.chan.is_none() {
                    self.steps.remove(&sid);
                    self.warm_done.push(sid);
                }
            }
        }
    }

    /// Publish the ready prefix of a channel's queue (in sequence order).
    fn publish(&mut self, ch: usize) {
        let p = self.ptrs[ch];
        while let Some(&sid) = self.order[ch].front() {
            let st = &self.steps[&sid];
            if st.pending > 0 { break; }
            let desc = unsafe { p.ring.add((st.seq % p.depth as u64) as usize * p.stride) };
            unsafe {
                std::ptr::write_volatile(desc.add(20) as *mut u32, st.status);
                (*AtomicU64::from_ptr(p.ctrl.add(ge::CTRL_OFF_PUB) as *mut u64)).store(st.seq + 1, Ordering::Release);
            }
            let ns = st.t_submit.elapsed().as_nanos() as u64;
            self.lat_ns[ch].push(ns);
            if p.critical { m::observe_decode_latency_us(ns / 1000); }
            m::PUBLISHED[p.kind].fetch_add(1, Ordering::Relaxed);
            self.steps.remove(&sid);
            self.order[ch].pop_front();
        }
    }
}

struct Chan {
    ctrl: ge::MappedHost,
    ring_mem: ge::MappedHost,
    depth: usize,
    stride: usize,
    max_tok: usize,
    /// the Engram layers this channel carries: l0 .. l0 + nl
    l0: usize,
    nl: usize,
    /// 0 = decode, 1 = prefill
    kind: usize,
    dev: ge::ChannelDev,
    next_seq: u64,
    /// published or pending steps not yet retired: (seq, slots it references)
    inflight: VecDeque<(u64, Vec<u32>)>,
    /// GPU counters already folded into the metrics
    seen_stalls: u64,
    seen_stall_ns: u64,
    seen_timeouts: u64,
}

impl Chan {
    fn ctrl_word(&self, off: usize) -> u64 {
        unsafe { (*AtomicU64::from_ptr(self.ctrl.host_ptr().add(off) as *mut u64)).load(Ordering::Acquire) }
    }
    fn ctrl(&self) -> ge::Ctrl {
        unsafe { std::ptr::read_volatile(self.ctrl.host_ptr() as *const ge::Ctrl) }
    }
}

/// The per-rank Engram row runtime.
pub struct EngramIo {
    /// FIRST field: dropped first (fields drop in declaration order), so the reader threads have stopped and
    /// drained their in-flight reads before the slot memory they write into is freed.
    readers: ReaderPool,
    dev: Arc<CudaDevice>,
    layout: Arc<EngramLayout>,
    files: Vec<HeadFile>,
    /// output column order: (g, index into `files`), ascending column `c = g * n_heads + head`
    own: Vec<(usize, usize)>,
    n_layers: usize,
    n_g: usize,
    n_heads: usize,
    n_cols: usize,
    ids_per_token: usize,
    slots: ge::MappedHost,
    cache: RowCache,
    shared: Arc<Mutex<Shared>>,
    chans: Vec<Chan>,
    next_step: u64,
    warm_live: HashMap<u64, Vec<u32>>,
    pub cfg: EngramIoConfig,
    jobs: Vec<ReadJob>,
    hits: Vec<u32>,
    step_slots: Vec<u32>,
}

impl EngramIo {
    /// Open the rank's head files (each checked against the layout), allocate the pinned cache and the
    /// channels, start the readers. The gpu_dsv41 module must be loadable on `dev`.
    pub fn open(dev: &Arc<CudaDevice>, layout: Arc<EngramLayout>, cfg: EngramIoConfig) -> Result<Self> {
        crate::dsv41::gpu::load_module(dev)?;
        let mut heads = cfg.heads.clone();
        heads.sort_unstable();
        heads.dedup();
        if heads.is_empty() || *heads.last().unwrap() >= layout.n_heads() {
            bail!("engram: heads {:?} outside 0..{}", cfg.heads, layout.n_heads());
        }
        let files: Vec<HeadFile> = heads.iter().map(|&h| HeadFile::open(&cfg.dir, h, &layout)).collect::<Result<_>>()?;
        let (n_layers, n_g, n_heads) = (layout.n_layers(), layout.n_ngram_cols(), layout.n_heads());
        let mut own: Vec<(usize, usize)> = (0..n_g).flat_map(|g| (0..files.len()).map(move |fi| (g, fi))).collect();
        own.sort_by_key(|&(g, fi)| g * n_heads + files[fi].head);
        let n_own = own.len();
        // slots: the cache, but never fewer than the rows of a full decode ring + three prefill chunks in flight
        let need = (DECODE_RING + 1) * n_layers * cfg.decode_max_tok * n_own + 3 * n_layers * cfg.prefill_max_tok * n_own + 1024;
        let want = cfg.cache_mib * 1024 * 1024 / ge::SLOT_STRIDE;
        let n_slots = want.max(need);
        if n_slots >= u32::MAX as usize - 1 { bail!("engram: {n_slots} cache slots exceed the u32 slot index"); }
        if cfg.cache_mib > 0 && want < need {
            crate::boot_report::notice(&format!(
                "engram: --engram-row-cache-mib {} holds {want} rows; using {need} ({} MiB), the rows of the steps that can be in flight",
                cfg.cache_mib, need * ge::SLOT_STRIDE >> 20));
        }
        let slots = ge::MappedHost::new(dev, n_slots * ge::SLOT_STRIDE).context("engram row cache (pinned, mapped)")?;
        let cache = RowCache::new(n_slots as u32, cfg.cache_mib > 0);
        let mut chans = Vec::new();
        // one decode channel per Engram layer, then the prefill channel (both layers) if configured
        let mut plan: Vec<(usize, usize, usize, usize, usize)> = (0..n_layers).map(|l| (cfg.decode_max_tok, DECODE_RING, l, 1, 0)).collect();
        if cfg.prefill_max_tok > 0 { plan.push((cfg.prefill_max_tok, PREFILL_RING, 0, n_layers, 1)); }
        for (max_tok, depth, l0, nl, kind) in plan {
            let stride = ge::desc_stride(nl * max_tok * n_own);
            chans.push(Chan {
                ctrl: ge::MappedHost::new(dev, ge::CTRL_BYTES)?,
                ring_mem: ge::MappedHost::new(dev, stride * depth)?,
                depth,
                stride,
                max_tok,
                l0,
                nl,
                kind,
                dev: ge::ChannelDev::new(dev)?,
                next_seq: 0,
                inflight: VecDeque::new(),
                seen_stalls: 0,
                seen_stall_ns: 0,
                seen_timeouts: 0,
            });
        }
        let ptrs: Vec<ChanPtrs> = chans.iter().map(|c| ChanPtrs { ctrl: c.ctrl.host_ptr(), ring: c.ring_mem.host_ptr(), stride: c.stride, depth: c.depth,
                                                                   kind: c.kind, critical: c.kind == 0 && c.l0 == 0 }).collect();
        let nch = chans.len();
        let shared = Arc::new(Mutex::new(Shared { loading: HashMap::new(), steps: HashMap::new(), order: vec![VecDeque::new(); nch],
                                                  failed_slots: Vec::new(), warm_done: Vec::new(), ptrs, lat_ns: vec![Vec::new(); nch] }));
        let sh2 = shared.clone();
        let done: Arc<DoneFn> = Arc::new(move |batch: &[(u32, Option<EngramIoError>)]| {
            let mut sh = sh2.lock().unwrap();
            let mut touched = vec![false; sh.order.len()];
            for (slot, e) in batch {
                if let Some(e) = e { crate::reprintln!("[dsv41] engram: a row read failed (slot {slot}), its round fails: {e}"); }
                if let Some(ls) = sh.loading.get(slot) {
                    for sid in &ls.waiters {
                        if let Some(c) = sh.steps.get(sid).and_then(|s| s.chan) { touched[c] = true; }
                    }
                }
                sh.finish_slot(*slot, e.is_some());
            }
            for (c, t) in touched.iter().enumerate() {
                if *t { sh.publish(c); }
            }
        });
        let ka = (!cfg.keepalive.is_zero()).then(|| {
            let f = &files[0];
            let pages = std::fs::metadata(&f.path).map(|md| md.len() / 4096).unwrap_or(0);
            (f.fd(), pages, cfg.keepalive)
        });
        let readers = ReaderPool::start(ReaderConfig { readers: cfg.readers, ring_entries: cfg.ring_entries, align: cfg.align, park_after: cfg.park_after,
                                                       keepalive: ka }, done)?;
        m::ACTIVE.store(true, Ordering::Relaxed);
        m::CACHE_SLOTS.store(n_slots as u64, Ordering::Relaxed);
        m::READERS.store(cfg.readers as u64, Ordering::Relaxed);
        let ids_per_token = layout.ids_per_token();
        let n_cols = layout.n_cols();
        Ok(EngramIo { readers, dev: dev.clone(), layout, files, own, n_layers, n_g, n_heads, n_cols, ids_per_token, slots, cache, shared, chans,
                      next_step: 0, warm_live: HashMap::new(), cfg, jobs: Vec::new(), hits: Vec::new(), step_slots: Vec::new() })
    }

    pub fn layout(&self) -> &Arc<EngramLayout> { &self.layout }
    /// Owned columns per Engram layer (the K-slice width of `wkv` / 256).
    pub fn n_own(&self) -> usize { self.own.len() }
    /// The owned columns in output order (global column index `c = g * n_heads + head`).
    pub fn own_cols(&self) -> Vec<usize> { self.own.iter().map(|&(g, fi)| g * self.n_heads + self.files[fi].head).collect() }
    pub fn heads(&self) -> Vec<usize> { self.files.iter().map(|f| f.head).collect() }
    pub fn n_slots(&self) -> u32 { self.cache.n_slots() }
    /// (pinned mapped bytes: the row cache + the channels' rings and control blocks, host bytes of the cache's slot tables
    /// and index) (leg MEM1's ledger).
    pub fn mem_bytes(&self) -> (u64, u64) {
        let pinned = self.slots.len() + self.chans.iter().map(|c| c.ctrl.len() + c.ring_mem.len()).sum::<usize>();
        (pinned as u64, self.cache.host_bytes())
    }
    pub fn channel_max_tok(&self, ch: usize) -> usize { self.chans[ch].max_tok }
    /// Engram layers of channel `ch` (its output holds `[layers][max_tok][n_own][256]`).
    pub fn channel_layers(&self, ch: usize) -> usize { self.chans[ch].nl }
    /// The prefill channel (after the per-layer decode channels).
    pub fn ch_prefill(&self) -> usize { self.n_layers }
    /// Engram layers (= decode channels).
    pub fn n_layers(&self) -> usize { self.n_layers }
    pub fn cache_stats(&self) -> super::cache::CacheStats { self.cache.stats }
    pub fn reader_totals(&self) -> [u64; 8] { self.readers.totals() }
    /// Wake parked readers (a request arrived; the first step should not pay a reader's wake-up).
    pub fn wake(&self) { self.readers.wake(); }
    /// Leg HT1: the readers' (park entries, seconds parked, wake-ups).
    pub fn reader_idle(&self) -> (u64, f64, u64) { self.readers.idle_totals() }

    /// Host bytes of a row slot (for the bench / identity checks).
    pub fn slot_bytes(&self, slot: u32) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.slots.host_ptr().add(slot as usize * ge::SLOT_STRIDE), ge::REC_BYTES) }
    }

    /// The record of (Engram layer `l`, owned column `j`) for hash id `id`: (file index, record index).
    #[inline]
    fn record_of(&self, l: usize, j: usize, id: u32) -> Result<(usize, u64)> {
        let (g, fi) = self.own[j];
        let s = &self.files[fi].sections[l * self.n_g + g];
        let row = (id as u64).wrapping_sub(s.source_row_start);
        if row >= s.rows {
            bail!("engram: hash id {id} (layer idx {l}, g {g}, head {}) outside its column [{}, +{})", self.files[fi].head, s.source_row_start, s.rows);
        }
        Ok((fi, s.first_record + row))
    }
    #[inline]
    fn id_at(&self, ids: &[u32], t: usize, l: usize, j: usize) -> u32 {
        let (g, fi) = self.own[j];
        ids[t * self.ids_per_token + l * self.n_cols + g * self.n_heads + self.files[fi].head]
    }

    /// Absolute file offset and head of the record a CPU path must read for (t, l, j) — the same mapping
    /// `submit` uses, exposed for the identity check.
    pub fn locate(&self, ids: &[u32], t: usize, l: usize, j: usize) -> Result<(std::path::PathBuf, u64)> {
        let (fi, rec) = self.record_of(l, j, self.id_at(ids, t, l, j))?;
        Ok((self.files[fi].path.clone(), self.files[fi].record_offset(rec)))
    }

    fn alloc_slot(&mut self, key: u64) -> Result<u32> {
        if let Some(s) = self.cache.alloc(key) { return Ok(s); }
        m::CACHE_FULL_WAITS.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();
        loop {
            self.housekeep();
            if let Some(s) = self.cache.alloc(key) { return Ok(s); }
            if t0.elapsed() > self.cfg.wait_timeout + Duration::from_secs(1) {
                bail!("engram: every row-cache slot is held by an unretired step for {:?} (cache too small for the steps in flight, or the GPU stopped consuming)", t0.elapsed());
            }
            std::hint::spin_loop();
        }
    }

    /// Retire consumed steps, forget failed slots, release finished warm requests; fold GPU counters.
    pub fn housekeep(&mut self) {
        for c in self.chans.iter_mut() {
            let done = unsafe { (*AtomicU64::from_ptr(c.ctrl.host_ptr().add(ge::CTRL_OFF_DONE) as *mut u64)).load(Ordering::Acquire) };
            while c.inflight.front().is_some_and(|(seq, _)| *seq < done) {
                let (_, sl) = c.inflight.pop_front().unwrap();
                for s in sl { if s != ge::ZERO_SLOT { self.cache.release(s); } }
            }
            let k = c.ctrl();
            m::STALLS.fetch_add(k.stalls.saturating_sub(c.seen_stalls), Ordering::Relaxed);
            m::STALL_NS.fetch_add(k.stall_ns.saturating_sub(c.seen_stall_ns), Ordering::Relaxed);
            m::TIMEOUTS.fetch_add(k.timeouts.saturating_sub(c.seen_timeouts), Ordering::Relaxed);
            (c.seen_stalls, c.seen_stall_ns, c.seen_timeouts) = (k.stalls, k.stall_ns, k.timeouts);
        }
        let (failed, warm) = {
            let mut sh = self.shared.lock().unwrap();
            let f = std::mem::take(&mut sh.failed_slots);
            for s in &f { sh.loading.remove(s); }
            (f, std::mem::take(&mut sh.warm_done))
        };
        for s in failed { self.cache.forget(s); }
        for sid in warm {
            if let Some(sl) = self.warm_live.remove(&sid) {
                for s in sl { self.cache.release(s); }
            }
        }
        let st = self.cache.stats;
        m::EVICTIONS.store(st.evictions, Ordering::Relaxed);
        m::CACHE_OCCUPIED.store(self.cache.occupied() as u64, Ordering::Relaxed);
        let t = self.readers.totals();
        m::READS.store(t[0], Ordering::Relaxed);
        m::READ_BYTES.store(t[1], Ordering::Relaxed);
        m::READ_RETRIES.store(t[2], Ordering::Relaxed);
        m::READ_ERRORS.store(t[3], Ordering::Relaxed);
        m::SLOW_READS.store(t[6], Ordering::Relaxed);
        m::MAX_READ_US.store(t[7] / 1000, Ordering::Relaxed);
    }

    /// Submit one decode round: `ids` = `n_tok * 48` hash ids of the fed positions of every lane
    /// (`[t][layer][column]`, from `EngramHasher::peek`), `live[t] == false` for DEAD positions (no read,
    /// zero row). One step per Engram layer, the first layer's reads at high priority. Returns once the reads
    /// are queued; each layer's step publishes itself when its rows land.
    pub fn submit_decode(&mut self, ids: &[u32], live: Option<&[bool]>) -> Result<Vec<StepTicket>> {
        let mut v = Vec::with_capacity(self.n_layers);
        for l in 0..self.n_layers { v.push(self.submit_ch(ch_decode(l), ids, live, l == 0)?); }
        m::STEPS[0].fetch_add(1, Ordering::Relaxed);
        Ok(v)
    }

    /// Submit one prefill chunk (both Engram layers in one step, normal priority).
    pub fn submit_prefill(&mut self, ids: &[u32], live: Option<&[bool]>) -> Result<StepTicket> {
        let ch = self.ch_prefill();
        if ch >= self.chans.len() { bail!("engram: no prefill channel (prefill_max_tok = 0)"); }
        let t = self.submit_ch(ch, ids, live, false)?;
        m::STEPS[1].fetch_add(1, Ordering::Relaxed);
        Ok(t)
    }

    fn submit_ch(&mut self, ch: usize, ids: &[u32], live: Option<&[bool]>, hi: bool) -> Result<StepTicket> {
        if ch >= self.chans.len() { bail!("engram: channel {ch} not configured"); }
        if ids.is_empty() || ids.len() % self.ids_per_token != 0 { bail!("engram: {} ids is not a whole number of tokens x {}", ids.len(), self.ids_per_token); }
        let n_tok = ids.len() / self.ids_per_token;
        if n_tok > self.chans[ch].max_tok { bail!("engram: step of {n_tok} positions exceeds the channel's {}", self.chans[ch].max_tok); }
        if let Some(lv) = live { if lv.len() != n_tok { bail!("engram: live mask has {} entries for {n_tok} positions", lv.len()); } }
        let t_submit = Instant::now();
        self.housekeep();
        let seq = self.chans[ch].next_seq;
        // the ring slot of `seq` must have been consumed by the GPU (seq - depth)
        let depth = self.chans[ch].depth as u64;
        if self.chans[ch].ctrl_word(ge::CTRL_OFF_DONE) + depth <= seq {
            m::RING_WAITS.fetch_add(1, Ordering::Relaxed);
            let t0 = Instant::now();
            while self.chans[ch].ctrl_word(ge::CTRL_OFF_DONE) + depth <= seq {
                if t0.elapsed() > self.cfg.wait_timeout + Duration::from_secs(1) {
                    bail!("engram: the GPU did not consume channel {ch} step {} within {:?} (submit without a matching launch?)", seq - depth, t0.elapsed());
                }
                std::hint::spin_loop();
            }
            self.housekeep();
        }
        let (n_own, l0, nl) = (self.own.len(), self.chans[ch].l0, self.chans[ch].nl);
        let rows = nl * n_tok * n_own;
        let (desc, stride) = {
            let c = &self.chans[ch];
            (unsafe { c.ring_mem.host_ptr().add((seq % depth) as usize * c.stride) }, c.stride)
        };
        debug_assert!(ge::DESC_HEAD_BYTES + 4 * rows <= stride);
        unsafe {
            std::ptr::write_volatile(desc as *mut ge::DescHead, ge::DescHead { seq, n_tok: n_tok as u32, n_own: n_own as u32, n_layers: nl as u32, status: STATUS_PENDING, pad: [0; 2] });
        }
        let table = unsafe { std::slice::from_raw_parts_mut(desc.add(ge::DESC_HEAD_BYTES) as *mut u32, rows) };
        let mut jobs = std::mem::take(&mut self.jobs);
        let mut hits = std::mem::take(&mut self.hits);
        let mut step_slots = std::mem::take(&mut self.step_slots);
        jobs.clear();
        hits.clear();
        step_slots.clear();
        let (mut n_hit, mut n_dead) = (0usize, 0usize);
        let mut r = 0usize;
        let res: Result<()> = (|| {
            for l in l0..l0 + nl {
                for t in 0..n_tok {
                    let dead = live.is_some_and(|lv| !lv[t]);
                    for j in 0..n_own {
                        if dead {
                            table[r] = ge::ZERO_SLOT;
                            n_dead += 1;
                            r += 1;
                            continue;
                        }
                        let (fi, rec) = self.record_of(l, j, self.id_at(ids, t, l, j))?;
                        let key = key_of(self.files[fi].head, rec);
                        let slot = match self.cache.lookup(key) {
                            Some(s) => { hits.push(s); n_hit += 1; s }
                            None => {
                                let s = self.alloc_slot(key)?;
                                let dst = unsafe { self.slots.host_ptr().add(s as usize * ge::SLOT_STRIDE) };
                                jobs.push(ReadJob { fd: self.files[fi].fd(), rec_off: self.files[fi].record_offset(rec), dst, slot: s, hi });
                                s
                            }
                        };
                        self.cache.add_ref(slot);
                        step_slots.push(slot);
                        table[r] = slot;
                        r += 1;
                    }
                }
            }
            Ok(())
        })();
        if let Err(e) = res {
            // nothing was registered: undo the references, leave the descriptor unpublished
            for &s in &step_slots { self.cache.release(s); }
            for j in &jobs { self.cache.forget(j.slot); }
            self.jobs = jobs;
            self.hits = hits;
            self.step_slots = step_slots;
            return Err(e);
        }
        let sid = self.next_step;
        self.next_step += 1;
        let mut joined = 0usize;
        {
            let mut sh = self.shared.lock().unwrap();
            let mut st = StepState { chan: Some(ch), seq, pending: 0, status: 0, t_submit };
            for jb in &jobs {
                sh.loading.insert(jb.slot, LoadState { waiters: vec![sid], failed: false });
                st.pending += 1;
            }
            for s in &hits {
                if let Some(ls) = sh.loading.get_mut(s) {
                    if ls.failed { st.status = ge::ERR_HOST; } else { ls.waiters.push(sid); st.pending += 1; joined += 1; }
                }
            }
            sh.steps.insert(sid, st);
            sh.order[ch].push_back(sid);
            sh.publish(ch);
        }
        self.readers.dispatch(&jobs);
        let misses = jobs.len();
        self.chans[ch].inflight.push_back((seq, step_slots.clone()));
        self.chans[ch].next_seq += 1;
        m::ROWS_HIT.fetch_add(n_hit as u64, Ordering::Relaxed);
        m::ROWS_MISS.fetch_add(misses as u64, Ordering::Relaxed);
        m::ROWS_DEAD.fetch_add(n_dead as u64, Ordering::Relaxed);
        self.jobs = jobs;
        self.hits = hits;
        self.step_slots = step_slots;
        Ok(StepTicket { channel: ch, seq, n_tok, rows, hits: n_hit, misses, joined, dead: n_dead })
    }

    /// Read the rows of `ids` into the cache ahead of their step: the draft-token prefetch (`decode = true`:
    /// the first Engram layer's rows go out at high priority) and the prompt read-ahead (`decode = false`).
    /// Rows already resident or in flight are skipped. Returns the number of reads issued. A no-op without a
    /// cache (`--engram-row-cache-mib 0`).
    pub fn warm(&mut self, ids: &[u32], live: Option<&[bool]>, decode: bool) -> Result<usize> {
        if !self.cache.indexed() { return Ok(0); }
        if ids.len() % self.ids_per_token != 0 { bail!("engram warm: {} ids is not a whole number of tokens", ids.len()); }
        let n_tok = ids.len() / self.ids_per_token;
        self.housekeep();
        let mut jobs = std::mem::take(&mut self.jobs);
        jobs.clear();
        let mut slots = Vec::new();
        let mut resident = 0u64;
        for l in 0..self.n_layers {
            for t in 0..n_tok {
                if live.is_some_and(|lv| !lv[t]) { continue; }
                for j in 0..self.own.len() {
                    let (fi, rec) = self.record_of(l, j, self.id_at(ids, t, l, j))?;
                    let key = key_of(self.files[fi].head, rec);
                    if self.cache.lookup(key).is_some() { resident += 1; continue; }
                    let s = self.alloc_slot(key)?;
                    self.cache.add_ref(s);
                    slots.push(s);
                    let dst = unsafe { self.slots.host_ptr().add(s as usize * ge::SLOT_STRIDE) };
                    jobs.push(ReadJob { fd: self.files[fi].fd(), rec_off: self.files[fi].record_offset(rec), dst, slot: s, hi: decode && l == 0 });
                }
            }
        }
        m::WARM_RESIDENT.fetch_add(resident, Ordering::Relaxed);
        let n = jobs.len();
        if n > 0 {
            let sid = self.next_step;
            self.next_step += 1;
            {
                let mut sh = self.shared.lock().unwrap();
                for jb in &jobs { sh.loading.insert(jb.slot, LoadState { waiters: vec![sid], failed: false }); }
                sh.steps.insert(sid, StepState { chan: None, seq: 0, pending: n as u32, status: 0, t_submit: Instant::now() });
            }
            self.warm_live.insert(sid, slots);
            self.readers.dispatch(&jobs);
            m::WARM_READS.fetch_add(n as u64, Ordering::Relaxed);
        }
        self.jobs = jobs;
        Ok(n)
    }

    /// Steps of channel `ch` published so far / consumed by the GPU so far.
    pub fn published(&self, ch: usize) -> u64 { self.chans[ch].ctrl_word(ge::CTRL_OFF_PUB) }
    pub fn consumed(&self, ch: usize) -> u64 { self.chans[ch].ctrl_word(ge::CTRL_OFF_DONE) }
    pub fn submitted(&self, ch: usize) -> u64 { self.chans[ch].next_seq }
    /// The channel's GPU-side counters (stalls, waits, errors).
    pub fn gpu_ctrl(&self, ch: usize) -> ge::Ctrl { self.chans[ch].ctrl() }
    /// Warm requests still reading.
    pub fn warm_pending(&self) -> usize { self.warm_live.len() }
    /// Drain the host submit -> publish latencies (ns) recorded so far for channel `ch`.
    pub fn take_latencies(&self, ch: usize) -> Vec<u64> { std::mem::take(&mut self.shared.lock().unwrap().lat_ns[ch]) }

    /// The launch geometry of channel `ch` into `out_bf16` (`[channel layers][max_tok][n_own][256]` bf16, may
    /// be 0 without `MODE_DECODE`) and `out_rec` (`[channel layers][max_tok][n_own][264]` bytes, 0 without
    /// `MODE_RECORDS`). A decode channel carries one Engram layer: the round launches `ch_decode(0)` before
    /// the first Engram layer's `wkv` partial and `ch_decode(1)` before the second's.
    pub fn step_launch(&self, ch: usize, out_bf16: u64, out_rec: u64, mode: u32) -> ge::StepLaunch {
        let c = &self.chans[ch];
        use cudarc::driver::DevicePtr;
        ge::StepLaunch {
            ctrl_dptr: c.ctrl.dev_ptr(),
            dev_words: *c.dev.words.device_ptr(),
            desc_dptr: c.ring_mem.dev_ptr(),
            desc_stride: c.stride as u32,
            ring: c.depth as u32,
            cache_dptr: self.slots.dev_ptr(),
            out_bf16,
            out_rec,
            max_tok: c.max_tok as u32,
            max_own: self.own.len() as u32,
            layers: c.nl as u32,
            mode,
            timeout_ns: self.cfg.wait_timeout.as_nanos() as u64,
            blocks: 0,
        }
    }

    /// Enqueue the GPU side of channel `ch`'s next step (bounded wait + decode) on the device stream.
    pub fn launch(&self, ch: usize, out_bf16: u64, out_rec: u64, mode: u32) -> Result<()> {
        ge::launch_step_decode(&self.dev, &self.step_launch(ch, out_bf16, out_rec, mode))
    }

    /// After an abandoned round: the device stream must be idle. Every submitted-but-unconsumed step of the
    /// channel is dropped (its reads finish into the cache), and the GPU counter jumps to the next submit.
    pub fn reset_channel(&mut self, ch: usize) -> Result<()> {
        self.dev.synchronize()?;
        // wait for the queued steps to publish (their reads land) so no reader still writes a dropped slot table
        let t0 = Instant::now();
        while self.published(ch) < self.chans[ch].next_seq {
            if t0.elapsed() > self.cfg.wait_timeout { bail!("engram: channel {ch} steps did not publish within {:?}", t0.elapsed()); }
            std::hint::spin_loop();
        }
        let next = self.chans[ch].next_seq;
        self.chans[ch].dev.reset_to(&self.dev, next)?;
        unsafe { (*AtomicU64::from_ptr(self.chans[ch].ctrl.host_ptr().add(ge::CTRL_OFF_DONE) as *mut u64)).store(next, Ordering::Release) };
        self.housekeep();
        Ok(())
    }

    /// Clear a channel's sticky GPU error (after the host has failed the affected request).
    pub fn clear_error(&self, ch: usize) {
        let p = unsafe { self.chans[ch].ctrl.host_ptr().add(std::mem::offset_of!(ge::Ctrl, err)) } as *mut u32;
        unsafe { (*AtomicU32::from_ptr(p)).store(0, Ordering::Release) };
    }
}

impl Drop for EngramIo {
    fn drop(&mut self) {
        // a decode kernel may still read the mapped slots / descriptors: let the stream finish first (a kernel
        // waiting for a step that will never publish ends at its bounded wait)
        let _ = self.dev.synchronize();
    }
}
