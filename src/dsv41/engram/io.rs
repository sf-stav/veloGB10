//! Engram runtime, part 1 (leg EIO): the rank's head files and the reader pool.
//!
//! * [`HeadFile`]: one packed head file (ENGb's format) opened `O_DIRECT`, checked at open against the ONE
//!   layout: its manifest's geometry and sections must equal ENGa's `EngramLayout::rows_for_head` for that
//!   head (a pack whose bucket layout drifted from the hash would serve wrong rows; it is refused with the
//!   field that differs).
//! * [`ReaderPool`]: N reader threads, each with its own `io_uring` (ENGb's raw wrapper) and its own staging
//!   buffers. A job is "read the aligned span covering record R and copy its 264 bytes to slot pointer P".
//!   Readers spin on their queue while the engine is active and on the completion queue while reads are in
//!   flight (a sleeping reader costs 0.6-1 ms to wake on GB10, measured; a blocked `io_uring_enter` adds
//!   the same tail), and park after `park_after` without work. The completion callback runs on the reader
//!   thread with a batch of (slot, error) pairs.
//!
//! Errors keep ENGb's classification: EINTR / EAGAIN retry (up to [`MAX_RETRIES`]); any other errno or a
//! short read is a failed row (the runtime fails the step: the GPU never decodes an unwritten slot).

use super::layout::{EngramLayout, HeadSection};
use super::manifest::{EngramIoError, Geometry, HeadManifest, PackedHead, ReadSpan, RecordLayout, RECORD_BYTES};
use super::uring::{AlignedBuf, Ring};
use anyhow::{bail, Result};
use std::collections::VecDeque;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const MAX_RETRIES: u32 = 5;
/// Staging bytes per in-flight read: a 264-B record crosses at most one 4 KiB boundary.
const STAGE_BYTES: usize = 8192;

/// One opened head pack.
pub struct HeadFile {
    pub head: usize,
    pub path: PathBuf,
    file: std::fs::File,
    pub layout: RecordLayout,
    pub header_bytes: u64,
    /// layer-major, then g (`layer_idx * n_g + g`): from the layout, proven equal to the manifest at open
    pub sections: Vec<HeadSection>,
    pub total_records: u64,
}

impl HeadFile {
    pub fn open(dir: &Path, head: usize, lay: &EngramLayout) -> Result<Self> {
        let ph = PackedHead::open(dir, head).map_err(|e| anyhow::anyhow!("{e}"))?;
        check_manifest_against_layout(&ph.manifest, lay, head)?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&ph.path)
            .map_err(|e| anyhow::anyhow!("engram: open {} O_DIRECT: {e}", ph.path.display()))?;
        let rows = lay.rows_for_head(head);
        Ok(HeadFile {
            head,
            path: ph.path.clone(),
            file,
            layout: ph.manifest.layout,
            header_bytes: ph.manifest.header_bytes,
            sections: rows.sections,
            total_records: rows.total_records,
        })
    }
    pub fn fd(&self) -> i32 { self.file.as_raw_fd() }
    /// Absolute file offset of record `rec`.
    #[inline]
    pub fn record_offset(&self, rec: u64) -> u64 { self.header_bytes + self.layout.data_offset(rec) }
}

/// The manifest of `head` must describe exactly the rows the layout assigns to that head (geometry and
/// every section). This is where the packer's and the hash's layouts meet at run time.
pub fn check_manifest_against_layout(m: &HeadManifest, lay: &EngramLayout, head: usize) -> Result<()> {
    let geo = Geometry::from_layout(lay);
    if m.geometry != geo {
        bail!("engram head {head}: the pack's bucket geometry differs from this model's Engram layout (packed for another config?)");
    }
    let rows = lay.rows_for_head(head);
    if m.total_records != rows.total_records || m.sections.len() != rows.sections.len() {
        bail!("engram head {head}: the pack holds {} records in {} sections, the layout wants {} in {}", m.total_records,
              m.sections.len(), rows.total_records, rows.sections.len());
    }
    for (a, b) in m.sections.iter().zip(&rows.sections) {
        if (a.layer_idx, a.layer_id, a.g, a.source_row_start, a.rows, a.first_record)
            != (b.layer_idx, b.layer_id, b.g, b.source_row_start, b.rows, b.first_record)
        {
            bail!("engram head {head}: pack section {a:?} differs from the layout's {b:?}");
        }
    }
    Ok(())
}

/// One read: the record at absolute offset `rec_off` of `fd`, copied to `dst` (a cache slot), reported as `slot`.
/// `hi` = on a decode round's critical path (the first Engram layer's rows, needed one layer into the
/// verify): the pool submits low-priority reads only while no high-priority read is queued or in flight
/// anywhere in the pool, and keeps at most [`LO_INFLIGHT_CAP`] of them in flight per reader, so a decode
/// round's critical rows never queue behind a prompt read-ahead or the second layer's rows.
#[derive(Clone, Copy, Debug)]
pub struct ReadJob {
    pub fd: i32,
    pub rec_off: u64,
    pub dst: *mut u8,
    pub slot: u32,
    pub hi: bool,
}

/// Low-priority reads one reader keeps in flight (a high-priority read queues behind at most this many).
pub const LO_INFLIGHT_CAP: usize = 32;
// `dst` points into the mapped slot array; each job owns its slot until it reports completion.
unsafe impl Send for ReadJob {}

/// Completion callback: a batch of (slot, None = ok / Some(error)).
pub type DoneFn = dyn Fn(&[(u32, Option<EngramIoError>)]) + Send + Sync;

/// Per-reader counters (relaxed atomics; the runtime sums them for the metrics).
#[derive(Default)]
pub struct ReaderStats {
    pub reads: AtomicU64,
    pub bytes: AtomicU64,
    pub retries: AtomicU64,
    pub errors: AtomicU64,
    pub parks: AtomicU64,
    pub max_inflight: AtomicU64,
    /// reads that took longer than [`SLOW_READ`] from submit to completion, and the longest one (ns)
    pub slow: AtomicU64,
    pub max_read_ns: AtomicU64,
    /// leg HT1: ns spent parked, and wake-ups back to the awake (spinning) tier by work or `wake()`
    pub parked_ns: AtomicU64,
    pub wakes: AtomicU64,
}

/// Leg HT1: the longest single sleep of a parked reader (a safety net; work, `wake()` and shutdown unpark it at once).
pub const PARK_SLICE: Duration = Duration::from_millis(200);

/// A read slower than this counts in `ReaderStats::slow` (the NVMe tail that can make a round wait).
pub const SLOW_READ: Duration = Duration::from_micros(500);

struct WorkerQ {
    q: Mutex<Vec<ReadJob>>,
    has: AtomicBool,
    /// leg HT1: `ReaderPool::wake` asked the reader to come back to the awake tier without work (a request is starting)
    poke: AtomicBool,
    parked: AtomicBool,
    stop: AtomicBool,
    thread: OnceLock<std::thread::Thread>,
    stats: ReaderStats,
}

pub struct ReaderPool {
    /// high-priority jobs queued or in flight in the whole pool
    hi_pending: Arc<AtomicU64>,
    ws: Vec<Arc<WorkerQ>>,
    handles: Vec<std::thread::JoinHandle<()>>,
    next: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ReaderConfig {
    pub readers: usize,
    pub ring_entries: u32,
    pub align: u64,
    pub park_after: Duration,
    /// NVMe keep-alive while the readers are awake: reader 0 reads one random 4 KiB page of (fd, pages) when
    /// nothing was read for this long (None = off). Between 12 ms decode rounds the drive otherwise drops into
    /// a power state whose exit costs ~20-60 us per step and up to ~1 ms at the tail (measured on .14:
    /// 72-row steps p50 300 -> 234 us, max 1165 -> 377 us with a 2 ms keep-alive; DeepSeek-V4.1 readers
    /// elsewhere saw an 11 ms PS3 exit after 100 ms idle).
    pub keepalive: Option<(i32, u64, Duration)>,
}

impl ReaderPool {
    pub fn start(cfg: ReaderConfig, done: Arc<DoneFn>) -> Result<Self> {
        if cfg.readers == 0 || !(cfg.align == 512 || cfg.align == 4096) {
            bail!("engram readers: {} readers, align {} (want >= 1 reader, align 512 or 4096)", cfg.readers, cfg.align);
        }
        let mut ws = Vec::new();
        let mut handles = Vec::new();
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let hi_pending = Arc::new(AtomicU64::new(0));
        for i in 0..cfg.readers {
            let w = Arc::new(WorkerQ { q: Mutex::new(Vec::new()), has: AtomicBool::new(false), poke: AtomicBool::new(false), parked: AtomicBool::new(false),
                                       stop: AtomicBool::new(false), thread: OnceLock::new(), stats: ReaderStats::default() });
            let (w2, d2, tx2, hp2) = (w.clone(), done.clone(), tx.clone(), hi_pending.clone());
            let mut c = cfg;
            if i > 0 { c.keepalive = None; }
            let h = std::thread::Builder::new()
                .name(format!("engram-rd{i}"))
                .spawn(move || reader_main(w2, d2, c, hp2, tx2))
                .map_err(|e| anyhow::anyhow!("engram: spawn reader {i}: {e}"))?;
            let _ = w.thread.set(h.thread().clone());
            ws.push(w);
            handles.push(h);
        }
        drop(tx);
        for _ in 0..cfg.readers {
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => bail!("engram reader start: {e}"),
                Err(_) => bail!("engram reader exited during start"),
            }
        }
        Ok(ReaderPool { hi_pending, ws, handles, next: 0 })
    }

    pub fn readers(&self) -> usize { self.ws.len() }

    /// Hand `jobs` to the readers: contiguous shares, round-robin start, one queue lock per reader.
    pub fn dispatch(&mut self, jobs: &[ReadJob]) {
        if jobs.is_empty() { return; }
        let hi = jobs.iter().filter(|j| j.hi).count() as u64;
        if hi > 0 { self.hi_pending.fetch_add(hi, Ordering::SeqCst); }
        let n = self.ws.len();
        let used = n.min(jobs.len());
        let per = jobs.len().div_ceil(used);
        for (k, chunk) in jobs.chunks(per).enumerate() {
            let w = &self.ws[(self.next + k) % n];
            w.q.lock().unwrap().extend_from_slice(chunk);
            // SeqCst pairs with the reader's parked-store / has-load: one of the two sides sees the other
            w.has.store(true, Ordering::SeqCst);
            if w.parked.load(Ordering::SeqCst) {
                if let Some(t) = w.thread.get() { t.unpark(); }
            }
        }
        self.next = (self.next + used) % n;
    }

    /// Wake parked readers ahead of work (a request arrived; the first decode step should not pay the wake). A woken
    /// reader spins for `park_after` again before it parks; an awake reader is untouched (two atomic loads per reader).
    pub fn wake(&self) {
        for w in &self.ws {
            // SeqCst pairs with the reader's parked-store / poke-load, as in `dispatch`
            if w.parked.load(Ordering::SeqCst) {
                w.poke.store(true, Ordering::SeqCst);
                if let Some(t) = w.thread.get() { t.unpark(); }
            }
        }
    }

    /// Leg HT1: (park entries, seconds parked, wake-ups back to the awake tier) summed over the readers.
    pub fn idle_totals(&self) -> (u64, f64, u64) {
        let mut t = (0u64, 0u64, 0u64);
        for w in &self.ws {
            t.0 += w.stats.parks.load(Ordering::Relaxed);
            t.1 += w.stats.parked_ns.load(Ordering::Relaxed);
            t.2 += w.stats.wakes.load(Ordering::Relaxed);
        }
        (t.0, t.1 as f64 / 1e9, t.2)
    }

    /// (reads, bytes, retries, errors, parks, max in flight on one reader, reads slower than SLOW_READ, longest read ns)
    pub fn totals(&self) -> [u64; 8] {
        let mut t = [0u64; 8];
        for w in &self.ws {
            let s = &w.stats;
            t[0] += s.reads.load(Ordering::Relaxed);
            t[1] += s.bytes.load(Ordering::Relaxed);
            t[2] += s.retries.load(Ordering::Relaxed);
            t[3] += s.errors.load(Ordering::Relaxed);
            t[4] += s.parks.load(Ordering::Relaxed);
            t[5] = t[5].max(s.max_inflight.load(Ordering::Relaxed));
            t[6] += s.slow.load(Ordering::Relaxed);
            t[7] = t[7].max(s.max_read_ns.load(Ordering::Relaxed));
        }
        t
    }
}

impl Drop for ReaderPool {
    fn drop(&mut self) {
        for w in &self.ws {
            w.stop.store(true, Ordering::Release);
            if let Some(t) = w.thread.get() { t.unpark(); }
        }
        for h in self.handles.drain(..) { let _ = h.join(); }
    }
}

struct InFlight {
    job: ReadJob,
    span: ReadSpan,
    retries: u32,
    t: Instant,
}

/// user_data of a keep-alive read (never a staging index)
const KEEPALIVE_UD: u64 = u64::MAX;

fn reader_main(w: Arc<WorkerQ>, done: Arc<DoneFn>, cfg: ReaderConfig, hi_pending: Arc<AtomicU64>, ready: std::sync::mpsc::Sender<Result<(), String>>) {
    // a TP rank's launch thread is pinned to one core and spawned threads inherit its mask: a spinning reader
    // must not share that core (cpu_affinity::pin_worker re-pins to the worker mask; a no-op at TP1)
    crate::cpu_affinity::pin_worker();
    let mut ring = match Ring::new(cfg.ring_entries.next_power_of_two().max(8)) {
        Ok(r) => r,
        Err(e) => { let _ = ready.send(Err(format!("io_uring setup: {e}"))); return; }
    };
    let cap = ring.entries() as usize;
    let mut stage = AlignedBuf::new(cap * STAGE_BYTES);
    let stage_ptr = stage.as_mut_ptr();
    let mut free: Vec<usize> = (0..cap).rev().collect();
    let mut inflight: Vec<Option<InFlight>> = (0..cap).map(|_| None).collect();
    let mut backlog: VecDeque<(ReadJob, u32)> = VecDeque::new();
    let mut lo_backlog: VecDeque<(ReadJob, u32)> = VecDeque::new();
    let mut lo_out = 0usize;
    let mut batch: Vec<(u32, Option<EngramIoError>)> = Vec::with_capacity(cap);
    let mut outstanding = 0usize;
    let mut last_work = Instant::now();
    let mut last_reap = Instant::now();
    let mut last_read = Instant::now();
    let mut ka_buf = AlignedBuf::new(4096);
    let mut ka_inflight = false;
    let mut ka_rng = 0x9E37_79B9_7F4A_7C15u64 ^ (std::process::id() as u64);
    // leg HT1: true from the first park until work (or a `wake()`) arrives. While asleep a timeout or a spurious unpark
    // goes straight back to sleep (before HT1 it spun `park_after` again after every 200 ms timeout: ~55 % of a core per
    // idle reader, TP4B D1) and the NVMe keep-alive is not sent (it exists for the gaps between decode rounds)
    let mut asleep = false;
    let _ = ready.send(Ok(()));
    drop(ready);
    loop {
        let stopping = w.stop.load(Ordering::Acquire);
        if stopping {
            // never leave a read in flight into the staging buffers this thread is about to free
            if outstanding == 0 && !ka_inflight { break; }
            let dropped = backlog.iter().filter(|(j, _)| j.hi).count() as u64;
            if dropped > 0 { hi_pending.fetch_sub(dropped, Ordering::SeqCst); }
            backlog.clear();
            lo_backlog.clear();
        }
        if !stopping && w.has.swap(false, Ordering::AcqRel) {
            if asleep { asleep = false; w.stats.wakes.fetch_add(1, Ordering::Relaxed); super::io_metrics::READER_WAKES.fetch_add(1, Ordering::Relaxed); }
            let mut q = w.q.lock().unwrap();
            for j in q.drain(..) {
                if j.hi { backlog.push_back((j, 0u32)); } else { lo_backlog.push_back((j, 0u32)); }
            }
        }
        // low-priority reads move to the submit queue only while the pool has no high-priority work
        while backlog.is_empty() && !lo_backlog.is_empty() && lo_out < LO_INFLIGHT_CAP && hi_pending.load(Ordering::SeqCst) == 0 {
            backlog.push_back(lo_backlog.pop_front().unwrap());
            lo_out += 1;
        }
        let mut pushed = 0;
        while !backlog.is_empty() && !free.is_empty() {
            let (job, retries) = backlog.pop_front().unwrap();
            let span = ReadSpan::for_record(job.rec_off, cfg.align);
            let idx = free.pop().unwrap();
            let buf = unsafe { stage_ptr.add(idx * STAGE_BYTES) };
            if !ring.push_read(job.fd, buf, span.aligned_len as u32, span.aligned_offset, idx as u64) {
                free.push(idx);
                backlog.push_front((job, retries));
                break;
            }
            inflight[idx] = Some(InFlight { job, span, retries, t: Instant::now() });
            pushed += 1;
        }
        if pushed > 0 { last_read = Instant::now(); }
        if let Some((fd, pages, every)) = cfg.keepalive {
            if !ka_inflight && !stopping && !asleep && pages > 0 && last_read.elapsed() >= every {
                ka_rng ^= ka_rng << 13;
                ka_rng ^= ka_rng >> 7;
                ka_rng ^= ka_rng << 17;
                if ring.push_read(fd, ka_buf.as_mut_ptr(), 4096, (ka_rng % pages) * 4096, KEEPALIVE_UD) {
                    ka_inflight = true;
                    last_read = Instant::now();
                    let _ = ring.submit_and_wait(0);
                }
            }
        }
        if pushed > 0 {
            outstanding += pushed;
            w.stats.max_inflight.fetch_max(outstanding as u64, Ordering::Relaxed);
            if let Err(e) = ring.submit_and_wait(0) {
                // the submission itself failed: fail every read just pushed, keep the reader alive
                crate::reprintln!("[dsv41] engram reader: io_uring submit failed, retrying: {e}");
            }
            last_reap = Instant::now();
        }
        let mut reaped = 0;
        while let Some((ud, res)) = ring.pop() {
            reaped += 1;
            if ud == KEEPALIVE_UD { ka_inflight = false; continue; }
            let idx = ud as usize;
            let Some(f) = inflight[idx].take() else { continue };
            outstanding -= 1;
            free.push(idx);
            let need = f.span.intra + RECORD_BYTES;
            let finished = !(((res == -libc::EINTR) || (res == -libc::EAGAIN)) && f.retries < MAX_RETRIES);
            if finished {
                if f.job.hi { hi_pending.fetch_sub(1, Ordering::SeqCst); } else { lo_out = lo_out.saturating_sub(1); }
            }
            if res >= need as i32 {
                let ns = f.t.elapsed().as_nanos() as u64;
                if ns > SLOW_READ.as_nanos() as u64 { w.stats.slow.fetch_add(1, Ordering::Relaxed); }
                w.stats.max_read_ns.fetch_max(ns, Ordering::Relaxed);
                unsafe { std::ptr::copy_nonoverlapping(stage_ptr.add(idx * STAGE_BYTES + f.span.intra), f.job.dst, RECORD_BYTES) };
                w.stats.reads.fetch_add(1, Ordering::Relaxed);
                w.stats.bytes.fetch_add(f.span.aligned_len as u64, Ordering::Relaxed);
                batch.push((f.job.slot, None));
            } else if (res == -libc::EINTR || res == -libc::EAGAIN) && f.retries < MAX_RETRIES {
                w.stats.retries.fetch_add(1, Ordering::Relaxed);
                backlog.push_front((f.job, f.retries + 1));
            } else {
                w.stats.errors.fetch_add(1, Ordering::Relaxed);
                let e = if res < 0 {
                    EngramIoError::Os { op: "io_uring read", errno: -res, file: format!("fd {}", f.job.fd), offset: f.span.aligned_offset }
                } else {
                    EngramIoError::Short { file: format!("fd {}", f.job.fd), offset: f.span.aligned_offset, got: res as usize, want: need }
                };
                batch.push((f.job.slot, Some(e)));
            }
        }
        if !batch.is_empty() {
            done(&batch);
            batch.clear();
        }
        let now = Instant::now();
        if reaped > 0 { last_reap = now; }
        if outstanding > 0 || ka_inflight {
            // completions normally appear without a syscall; flush deferred completion work every 20 us
            if now.duration_since(last_reap) > Duration::from_micros(20) {
                let _ = ring.flush_events();
                last_reap = now;
            }
            if outstanding > 0 { last_work = now; }
            std::hint::spin_loop();
        } else if backlog.is_empty() && lo_backlog.is_empty() {
            if asleep || now.duration_since(last_work) > cfg.park_after {
                w.parked.store(true, Ordering::SeqCst);
                if !w.has.load(Ordering::SeqCst) && !w.poke.load(Ordering::SeqCst) && !w.stop.load(Ordering::Acquire) {
                    if !asleep {
                        asleep = true;
                        w.stats.parks.fetch_add(1, Ordering::Relaxed);
                        super::io_metrics::READER_PARKS.fetch_add(1, Ordering::Relaxed);
                    }
                    let t0 = Instant::now();
                    std::thread::park_timeout(PARK_SLICE);
                    let ns = t0.elapsed().as_nanos() as u64;
                    w.stats.parked_ns.fetch_add(ns, Ordering::Relaxed);
                    super::io_metrics::READER_PARKED_NS.fetch_add(ns, Ordering::Relaxed);
                }
                w.parked.store(false, Ordering::Release);
                // back to the awake tier only for work (picked up at the top of the loop) or a `wake()`; a timeout or a spurious
                // unpark parks again on the next pass
                if w.poke.swap(false, Ordering::AcqRel) {
                    if asleep { asleep = false; w.stats.wakes.fetch_add(1, Ordering::Relaxed); super::io_metrics::READER_WAKES.fetch_add(1, Ordering::Relaxed); }
                    last_work = Instant::now();
                } else if w.has.load(Ordering::Acquire) {
                    last_work = Instant::now();
                }
            } else {
                std::hint::spin_loop();
            }
        } else {
            last_work = now;
        }
    }
    drop(stage);
    drop(ka_buf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn idle_pool(readers: usize, park_after: Duration, done: Arc<DoneFn>) -> Option<ReaderPool> {
        match ReaderPool::start(ReaderConfig { readers, ring_entries: 16, align: 4096, park_after, keepalive: None }, done) {
            Ok(p) => Some(p),
            Err(e) => { eprintln!("io_uring unavailable ({e}); SKIP"); None }
        }
    }
    fn per_reader(p: &ReaderPool) -> Vec<(u64, u64, u64)> {
        p.ws.iter().map(|w| (w.stats.parks.load(Ordering::Relaxed), w.stats.parked_ns.load(Ordering::Relaxed), w.stats.wakes.load(Ordering::Relaxed))).collect()
    }
    /// One temp file of 64 KiB opened O_DIRECT (None = the temp fs has no O_DIRECT: SKIP).
    fn direct_file(tag: &str) -> Option<(PathBuf, std::fs::File)> {
        let dir = std::env::temp_dir().join(format!("ht1_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.bin");
        std::fs::File::create(&p).unwrap().write_all(&vec![7u8; 65536]).unwrap();
        match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&p) {
            Ok(f) => Some((dir, f)),
            Err(e) => { eprintln!("O_DIRECT unavailable on the temp fs ({e}); SKIP"); std::fs::remove_dir_all(&dir).ok(); None }
        }
    }

    /// Leg HT1 (TP4B D1): an idle reader parks ONCE and stays parked across its 200 ms timeouts. Before HT1 it spun
    /// `park_after` again after every timeout (~55 % of a core per idle reader at the 250 ms default); with a 150 ms
    /// `park_after` and 1.2 s of idle that bug shows as >= 3 parks per reader and < 0.6 s parked.
    #[test]
    fn an_idle_reader_parks_once_and_stays_parked() {
        let done: Arc<DoneFn> = Arc::new(|_b: &[(u32, Option<EngramIoError>)]| {});
        let Some(pool) = idle_pool(2, Duration::from_millis(150), done) else { return };
        std::thread::sleep(Duration::from_millis(1200));
        for (i, (parks, ns, wakes)) in per_reader(&pool).into_iter().enumerate() {
            assert_eq!(parks, 1, "reader {i} parked {parks} times in 1.2 s of idle (it must stay parked)");
            assert!(ns >= 750_000_000, "reader {i} parked only {:.2} s of 1.2 s (it spun after a timeout)", ns as f64 / 1e9);
            assert_eq!(wakes, 0, "reader {i}: nothing woke it");
        }
        let (p, s, _) = pool.idle_totals();
        assert!(p == 2 && s >= 1.5, "totals {p} parks, {s:.2} s");
    }

    /// A parked reader wakes on work, every time: dispatches raced against the reader's park (no lost wake-up: each read
    /// completes within the bound, never after a 200 ms timeout), and a pool dropped while its readers are parked joins at once.
    #[test]
    fn a_parked_reader_wakes_on_work_with_no_lost_wake_up_and_shuts_down_while_parked() {
        let Some((dir, f)) = direct_file("wake") else { return };
        let got = Arc::new(AtomicU64::new(0));
        let g2 = got.clone();
        let done: Arc<DoneFn> = Arc::new(move |b: &[(u32, Option<EngramIoError>)]| { assert!(b.iter().all(|(_, e)| e.is_none())); g2.fetch_add(b.len() as u64, Ordering::SeqCst); });
        // park_after 2 ms: most dispatches below land around the moment the reader parks
        let Some(mut pool) = idle_pool(1, Duration::from_millis(2), done) else { std::fs::remove_dir_all(&dir).ok(); return };
        let mut slot = vec![0u8; 272];
        let mut slowest = Duration::ZERO;
        for i in 0..120u64 {
            let want = got.load(Ordering::SeqCst) + 1;
            let t0 = Instant::now();
            pool.dispatch(&[ReadJob { fd: f.as_raw_fd(), rec_off: (i * 517) % 60000, dst: slot.as_mut_ptr(), slot: 0, hi: i % 2 == 0 }]);
            while got.load(Ordering::SeqCst) < want {
                assert!(t0.elapsed() < Duration::from_secs(5), "read {i} never completed (lost wake-up)");
                std::thread::yield_now();
            }
            slowest = slowest.max(t0.elapsed());
            // sleep across the park point: 0, 1, 2, 3 ms
            std::thread::sleep(Duration::from_micros((i % 4) * 1000));
        }
        assert!(slowest < Duration::from_millis(150), "a read waited {slowest:?}: a dispatch missed the parked reader and waited for its 200 ms timeout");
        let (parks, _, wakes) = pool.idle_totals();
        assert!(parks >= 1 && wakes >= 1, "the reader parked and was woken by work ({parks} parks, {wakes} wakes)");
        std::thread::sleep(Duration::from_millis(50));
        assert!(pool.ws[0].parked.load(Ordering::SeqCst), "idle again: parked");
        let t0 = Instant::now();
        drop(pool);
        assert!(t0.elapsed() < Duration::from_millis(100), "dropping a pool whose reader is parked took {:?}", t0.elapsed());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `wake()` (a request is starting) brings a parked reader back to the awake tier without work: it counts a wake, spins
    /// `park_after` again and then parks a second time; an awake reader is left alone.
    #[test]
    fn wake_without_work_returns_the_reader_to_the_awake_tier_once() {
        let done: Arc<DoneFn> = Arc::new(|_b: &[(u32, Option<EngramIoError>)]| {});
        let Some(pool) = idle_pool(1, Duration::from_millis(40), done) else { return };
        std::thread::sleep(Duration::from_millis(120));
        assert_eq!(per_reader(&pool)[0].0, 1);
        pool.wake();
        std::thread::sleep(Duration::from_millis(15));
        assert!(!pool.ws[0].parked.load(Ordering::SeqCst), "woken: spinning its park_after");
        assert_eq!(per_reader(&pool)[0].2, 1, "one wake counted");
        std::thread::sleep(Duration::from_millis(150));
        let (parks, _, wakes) = per_reader(&pool)[0];
        assert_eq!((parks, wakes), (2, 1), "parked again after park_after, nothing else woke it");
    }

    /// The pool reads records at arbitrary (unaligned) offsets of a real file through O_DIRECT and
    /// reports each slot once.
    #[test]
    fn pool_reads_records_at_unaligned_offsets() {
        let dir = std::env::temp_dir().join(format!("eio_pool_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.bin");
        let data: Vec<u8> = (0..65536u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        std::fs::File::create(&p).unwrap().write_all(&data).unwrap();
        let f = match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&p) {
            Ok(f) => f,
            Err(e) => { eprintln!("O_DIRECT unavailable on the temp fs ({e}); SKIP"); std::fs::remove_dir_all(&dir).ok(); return; }
        };
        let n = 64usize;
        let mut slots = vec![0u8; n * 272];
        let got = Arc::new(Mutex::new(Vec::<(u32, bool)>::new()));
        let g2 = got.clone();
        let done: Arc<DoneFn> = Arc::new(move |b: &[(u32, Option<EngramIoError>)]| {
            g2.lock().unwrap().extend(b.iter().map(|(s, e)| (*s, e.is_none())));
        });
        let mut pool = match ReaderPool::start(ReaderConfig { readers: 3, ring_entries: 16, align: 4096, park_after: Duration::from_millis(5),
                                                               keepalive: Some((f.as_raw_fd(), 16, Duration::from_micros(200))) }, done) {
            Ok(p) => p,
            Err(e) => { eprintln!("io_uring unavailable ({e}); SKIP"); std::fs::remove_dir_all(&dir).ok(); return; }
        };
        // offsets that straddle 4 KiB pages and the last page
        let offs: Vec<u64> = (0..n as u64).map(|i| (i * 1013 + 3900 * (i % 3)) % (65536 - 264)).collect();
        let jobs: Vec<ReadJob> = (0..n).map(|i| ReadJob { fd: f.as_raw_fd(), rec_off: offs[i], dst: unsafe { slots.as_mut_ptr().add(i * 272) }, slot: i as u32, hi: i % 2 == 0 }).collect();
        pool.dispatch(&jobs);
        let t0 = Instant::now();
        while got.lock().unwrap().len() < n {
            assert!(t0.elapsed() < Duration::from_secs(10), "reads did not complete");
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(pool);
        let mut g = got.lock().unwrap().clone();
        g.sort();
        assert_eq!(g, (0..n as u32).map(|s| (s, true)).collect::<Vec<_>>());
        for i in 0..n {
            assert_eq!(&slots[i * 272..i * 272 + 264], &data[offs[i] as usize..offs[i] as usize + 264], "slot {i}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
