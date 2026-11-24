//! KVPOOL (fork, 2026-11): a shared physical pool behind the per-lane position planes.
//!
//! The EXL3 engine allocates every position plane (trunk KV, raw/pooled indexer keys, the draft
//! head's KV and keys) as one `[width][segs][max_pos][row]` slab, so `--max-batch N` pins N full
//! windows of memory whether or not a lane holds any tokens. This module keeps the SAME virtual
//! layout — every kernel, launch argument and captured CUDA graph is untouched — but backs it with
//! CUDA virtual memory: the address range is reserved up front and 2 MiB physical pages from ONE
//! fixed pool are mapped under the rows a lane actually uses, and handed back when the lane resets.
//!
//! Design rules (measured on GB10, see docs in the fork):
//! * The physical pool is created ONCE at boot. `cuMemCreate` under memory pressure takes 100-400 ms
//!   per 32 MiB and can fail (CUDA sees MemFree, not page cache); moving an existing page between
//!   addresses is ~0.06 ms and never allocates.
//! * Addresses never move: graphs captured before a mapping replay correctly after it, and after an
//!   unmap+remap with different physical pages.
//! * Touching an unmapped page is a fatal device fault, not an error return. Callers map AHEAD of the
//!   highest row a round can touch (`FwdModel::vmm_ensure`), and `Mode::Full` maps everything at boot
//!   so the plumbing can be proven bit-exact before on-demand mapping is switched on.
//!
//! Off by default: `--kv-pool full|lazy`, pool size `--kv-pool-gb` (lazy; default = everything). One box only: on a TP
//! rank the pool falls back to the stock allocation with one notice line (`mode_from_opts`).
//!
//! Contributed by @AbsolomBane (veloGB10 #43); design notes in docs/KV_POOL.md.

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::sys as cu;

/// Physical page size: the GB10 VMM allocation granularity (verified at `new`).
pub const PAGE: usize = 2 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Every page of every plane mapped at reserve time. Same memory as the slab allocation; proves
    /// the VMM plumbing changes nothing.
    Full,
    /// Pages mapped on demand from the fixed pool.
    Lazy,
}

/// `--kv-pool off|full|lazy` as asked on the command line (off = None), whatever this process can serve.
pub fn requested_mode() -> Option<Mode> {
    match crate::opts::var(crate::opt!("kv-pool")).ok().as_deref() {
        Some("full") => Some(Mode::Full),
        Some("lazy") => Some(Mode::Lazy),
        _ => None,
    }
}

/// The mode this process runs: the asked mode, except on a TP rank (head or node), where the pool falls back to the
/// stock allocation (AGENTS 1b-bis(8): boot anyway, one notice line, a /metrics gauge). Why: a full pool ends a lane
/// on the rank that ran out, the ranks hold differently sized planes (head-sharded KV), and the idle-lane eviction is
/// single-box only, so the lane sets could diverge between ranks. The TP groundwork (dry runs, the head's capacity
/// gate, the lane-set agree) stays in, inert until the pool runs under TP.
pub fn mode_from_opts() -> Option<Mode> {
    if TP_ROLE.load(Relaxed) { return None; }
    requested_mode()
}

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
/// Set once a process knows it is a TP rank (the builder and the TP load both say so, before any plane exists).
static TP_ROLE: AtomicBool = AtomicBool::new(false);
/// What backs this process's position planes (0 = stock allocation, 1 = full pool, 2 = lazy pool): set by the EXL3 load.
static ENGAGED: AtomicU64 = AtomicU64::new(0);
/// Why an asked-for pool is not running: 0 = it runs or was not asked, 1 = TP rank, 2 = this model is not served by
/// pool-backed planes (the pool is an EXL3 position-plane feature).
static FALLBACK: AtomicU64 = AtomicU64::new(0);

/// This process is a TP rank: an asked-for pool falls back to the stock allocation.
pub fn note_tp_role() { TP_ROLE.store(true, Relaxed); }
/// The EXL3 load built the pool behind its position planes.
pub fn note_engaged(m: Mode) { ENGAGED.store(if m == Mode::Lazy { 2 } else { 1 }, Relaxed); }
/// 0 = stock, 1 = full, 2 = lazy: what backs the planes now (the `velogb10_kv_vmm_mode` gauge).
pub fn engaged() -> u64 { ENGAGED.load(Relaxed) }
/// 0 none, 1 tp, 2 model (the `velogb10_kv_vmm_fallback` gauge's reason).
pub fn fallback_reason() -> u64 { FALLBACK.load(Relaxed) }

/// The bind-time line of an asked-for pool that is not running (None = nothing to say). Pure, for the tests.
pub fn fallback_line(asked: Option<Mode>, engaged: u64, tp: bool) -> Option<(u64, String)> {
    let m = asked?;
    if engaged != 0 { return None; }
    let name = if m == Mode::Lazy { "lazy" } else { "full" };
    Some(if tp {
        (1, format!("--kv-pool {name} asked on a TP rank: served with the standard per-lane KV allocation (the shared pool \
                     runs on one box only so far: a full pool would end a lane on one rank only); /metrics velogb10_kv_vmm_fallback"))
    } else {
        (2, format!("--kv-pool {name} asked, but this model is not served by the EXL3 position planes the pool backs: served \
                     with its standard KV allocation; /metrics velogb10_kv_vmm_fallback"))
    })
}

/// Called once when the HTTP API binds (every server): the notice and the gauge for an asked-for pool that is not
/// running here. Never refuses (AGENTS 1b-bis(8)).
pub fn bind_notice() {
    if let Some((why, line)) = fallback_line(requested_mode(), engaged(), TP_ROLE.load(Relaxed)) {
        FALLBACK.store(why, Relaxed);
        crate::boot_report::notice(&line);
    }
}

// Live pool counters for /metrics (velogb10_kv_vmm_*): pool / mapped / reserved bytes, idle lanes evicted, requests
// ended (finish=length) because the pool could not grow them, prompts that waited for room.
pub static POOL_BYTES: AtomicU64 = AtomicU64::new(0);
pub static MAPPED_BYTES: AtomicU64 = AtomicU64::new(0);
pub static RESERVED_BYTES: AtomicU64 = AtomicU64::new(0);
pub static EVICTIONS: AtomicU64 = AtomicU64::new(0);
pub static LENGTH_ENDS: AtomicU64 = AtomicU64::new(0);
pub static WAITS: AtomicU64 = AtomicU64::new(0);

/// `--kv-pool-gb` (decimal GB, like the boot fit message). None = size the pool to cover every plane.
pub fn pool_bytes_from_opts() -> Option<usize> {
    crate::opts::var(crate::opt!("kv-pool-gb")).ok().and_then(|v| v.parse::<u64>().ok()).filter(|g| *g > 0).map(|g| g as f64)
        .map(|g| (g * 1e9) as usize)
}

struct Plane {
    base: u64,
    /// Reserved bytes (a multiple of PAGE; >= the plane's logical length).
    bytes: usize,
    /// Fresh pages are zero-filled (the indexer key planes: `qsa_pool_update_b` reads the whole
    /// `ratio`-row block, rows past the lane's position included, and the stock boot uploads zeros).
    zero: bool,
    /// Per page: index into `KvVmm::handles` + 1, 0 = unmapped.
    pages: Vec<u32>,
}

/// KVPOOL (v2 dry-run): page bounds exactly as `ensure` walks them (every page the range touches) and as
/// `drop_range` frees them (only pages wholly inside the range). Pure arithmetic, shared by the real calls
/// and the dry-run queries below, so the head's capacity gate and the apply path cannot disagree.
pub fn ensure_page_bounds(off: usize, len: usize) -> (usize, usize) {
    (off / PAGE, (off + len).div_ceil(PAGE))
}
pub fn drop_page_bounds(off: usize, len: usize) -> (usize, usize) {
    (off.div_ceil(PAGE), (off + len) / PAGE)
}
/// Pages in `pages[p0..p1]` that are unmapped (`want_mapped == false`) or mapped (`true`).
pub fn count_pages(pages: &[u32], (p0, p1): (usize, usize), want_mapped: bool) -> usize {
    let p1 = p1.min(pages.len());
    if p0 >= p1 { return 0; }
    pages[p0..p1].iter().filter(|&&h| (h != 0) == want_mapped).count()
}

/// TPPOOL (design v3 step 4): unmapped pages in the UNION of several page spans (`(plane pages, (p0, p1))`). A page
/// two spans share (same plane, same index) counts once. Pure, shared by `pages_needed_union` and its tests.
pub fn count_union_unmapped(spans: &[(&[u32], (usize, usize))]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for &(pages, (p0, p1)) in spans {
        for pg in p0..p1.min(pages.len()) {
            if pages[pg] == 0 { seen.insert((pages.as_ptr() as usize, pg)); }
        }
    }
    seen.len()
}

pub struct KvVmm {
    mode: Mode,
    dev_id: i32,
    planes: Vec<Plane>,
    handles: Vec<cu::CUmemGenericAllocationHandle>,
    free: Vec<u32>,
    /// Lazy with no explicit pool size: grow the pool to cover each reserved plane (boot only).
    cover_all: bool,
    sealed: bool,
    pub stat_maps: u64,
    pub stat_unmaps: u64,
}

// Raw CUDA handles and addresses; every use is under the model's Mutex with the context bound.
unsafe impl Send for KvVmm {}

fn ck(r: cu::CUresult, what: &str) -> Result<()> {
    if r == cu::CUresult::CUDA_SUCCESS { Ok(()) } else { bail!("kv-vmm: {what} failed: {r:?}") }
}

impl KvVmm {
    /// `pool_bytes`: the physical pool for `Mode::Lazy` (None = grow to cover every plane reserved before
    /// `seal`). `Mode::Full` ignores it (each plane brings its own pages).
    pub fn new(mode: Mode, dev_ordinal: usize, pool_bytes: Option<usize>) -> Result<Self> {
        let mut v = KvVmm { mode, dev_id: dev_ordinal as i32, planes: Vec::new(), handles: Vec::new(), free: Vec::new(),
                            cover_all: mode == Mode::Full || pool_bytes.is_none(), sealed: false, stat_maps: 0, stat_unmaps: 0 };
        let prop = v.prop();
        let mut gran = 0usize;
        ck(unsafe { cu::cuMemGetAllocationGranularity(&mut gran, &prop,
                    cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM) }, "cuMemGetAllocationGranularity")?;
        ensure!(gran > 0 && PAGE % gran == 0, "kv-vmm: device granularity {gran} does not divide the {PAGE}-byte page");
        if mode == Mode::Lazy {
            if let Some(b) = pool_bytes { v.grow(b.div_ceil(PAGE))?; }
        }
        Ok(v)
    }

    fn prop(&self) -> cu::CUmemAllocationProp {
        let mut p: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        p.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        p.location.id = self.dev_id;
        p
    }

    fn grow(&mut self, pages: usize) -> Result<()> {
        let prop = self.prop();
        for _ in 0..pages {
            let mut h: cu::CUmemGenericAllocationHandle = 0;
            ck(unsafe { cu::cuMemCreate(&mut h, PAGE, &prop, 0) }, "cuMemCreate (kv pool page)")
                .with_context(|| format!("after {} pages ({:.2} GB)", self.handles.len(), (self.handles.len() * PAGE) as f64 / 1e9))?;
            self.free.push(self.handles.len() as u32);
            self.handles.push(h);
        }
        Ok(())
    }

    /// Reserve the address range of one plane of `bytes` logical bytes. Returns its base address.
    pub fn reserve(&mut self, bytes: usize, zero: bool) -> Result<u64> {
        ensure!(!self.sealed, "kv-vmm: reserve after seal");
        ensure!(bytes > 0, "kv-vmm: empty plane");
        let rbytes = bytes.div_ceil(PAGE) * PAGE;
        let mut base: cu::CUdeviceptr = 0;
        ck(unsafe { cu::cuMemAddressReserve(&mut base, rbytes, PAGE, 0, 0) }, "cuMemAddressReserve")?;
        let npages = rbytes / PAGE;
        self.planes.push(Plane { base: base as u64, bytes: rbytes, zero, pages: vec![0; npages] });
        if self.cover_all { self.grow(npages)?; }
        if self.mode == Mode::Full { self.ensure(base as u64, rbytes)?; }
        Ok(base as u64)
    }

    /// No more planes: the pool is final.
    pub fn seal(&mut self) { self.sealed = true; }

    fn plane_of(&self, at: u64, len: usize) -> Result<usize> {
        for (i, p) in self.planes.iter().enumerate() {
            if at >= p.base && at + len as u64 <= p.base + p.bytes as u64 { return Ok(i); }
        }
        bail!("kv-vmm: range {at:#x}+{len} is not inside a reserved plane")
    }

    /// KVPOOL (v2): how many pages `ensure(at, len)` would map right now (0 = nothing to do). Dry run: changes nothing.
    pub fn pages_needed(&self, at: u64, len: usize) -> Result<usize> {
        if len == 0 { return Ok(0); }
        let pi = self.plane_of(at, len)?;
        let base = self.planes[pi].base;
        Ok(count_pages(&self.planes[pi].pages, ensure_page_bounds((at - base) as usize, len), false))
    }
    /// KVPOOL (v2): how many mapped pages `drop_range(at, len)` would return to the pool. Dry run: changes nothing.
    /// Full mode maps everything and never drops, so it frees 0.
    pub fn pages_freed(&self, at: u64, len: usize) -> Result<usize> {
        if self.mode == Mode::Full || len == 0 { return Ok(0); }
        let pi = self.plane_of(at, len)?;
        let base = self.planes[pi].base;
        Ok(count_pages(&self.planes[pi].pages, drop_page_bounds((at - base) as usize, len), true))
    }
    /// TPPOOL (design v3 step 4): how many pages a SEQUENCE of `ensure` calls over `ranges` would map right now.
    /// The union, not the sum: two neighbouring ranges (two planes of one lane, or two lanes) can share their
    /// boundary page, which the first call maps and the second then finds mapped. Dry run: changes nothing.
    pub fn pages_needed_union(&self, ranges: &[(u64, usize)]) -> Result<usize> {
        let mut spans: Vec<(&[u32], (usize, usize))> = Vec::with_capacity(ranges.len());
        for &(at, len) in ranges {
            if len == 0 { continue; }
            let pi = self.plane_of(at, len)?;
            spans.push((&self.planes[pi].pages, ensure_page_bounds((at - self.planes[pi].base) as usize, len)));
        }
        Ok(count_union_unmapped(&spans))
    }
    /// Make every page under `[at, at+len)` mapped. Returns the pages newly mapped (0 = nothing to do).
    /// Err (nothing half-done that matters: pages mapped so far stay mapped) when the pool is exhausted.
    pub fn ensure(&mut self, at: u64, len: usize) -> Result<usize> {
        if len == 0 { return Ok(0); }
        let pi = self.plane_of(at, len)?;
        let (base, zero) = (self.planes[pi].base, self.planes[pi].zero);
        let (p0, p1) = ensure_page_bounds((at - base) as usize, len);
        let mut acc: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        acc.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        acc.location.id = self.dev_id;
        acc.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        let mut n = 0usize;
        for pg in p0..p1 {
            if self.planes[pi].pages[pg] != 0 { continue; }
            let Some(hi) = self.free.pop() else {
                bail!("kv-vmm: pool exhausted ({} pages = {:.2} GB all mapped)", self.handles.len(), (self.handles.len() * PAGE) as f64 / 1e9)
            };
            let addr = (base + (pg * PAGE) as u64) as cu::CUdeviceptr;
            let r = unsafe { cu::cuMemMap(addr, PAGE, 0, self.handles[hi as usize], 0) };
            if r != cu::CUresult::CUDA_SUCCESS { self.free.push(hi); ck(r, "cuMemMap")?; }
            // Ownership is published the moment the mapping exists: a later access/zero failure must leave the page
            // registered (drop_range and Drop unmap it, a retry does not map over it) — the error still propagates.
            self.planes[pi].pages[pg] = hi + 1;
            self.stat_maps += 1;
            ck(unsafe { cu::cuMemSetAccess(addr, PAGE, &acc, 1) }, "cuMemSetAccess")?;
            if zero { ck(unsafe { cu::cuMemsetD8_v2(addr, 0, PAGE) }, "cuMemsetD8 (fresh key page)")?; }
            n += 1;
        }
        MAPPED_BYTES.store(self.mapped_bytes() as u64, Relaxed);
        Ok(n)
    }

    /// Unmap every page that lies WHOLLY inside `[at, at+len)` and return it to the pool. Pages that
    /// straddle the range's edges belong partly to a neighbour (the next segment or lane) and stay.
    /// The caller has synchronized the device: no queued kernel may still reference these rows.
    pub fn drop_range(&mut self, at: u64, len: usize) -> Result<usize> {
        if self.mode == Mode::Full || len == 0 { return Ok(0); }
        let pi = self.plane_of(at, len)?;
        let base = self.planes[pi].base;
        let (p0, p1) = drop_page_bounds((at - base) as usize, len);
        let mut n = 0usize;
        for pg in p0..p1 {
            let h = self.planes[pi].pages[pg];
            if h == 0 { continue; }
            ck(unsafe { cu::cuMemUnmap((base + (pg * PAGE) as u64) as cu::CUdeviceptr, PAGE) }, "cuMemUnmap")?;
            self.planes[pi].pages[pg] = 0;
            self.free.push(h - 1);
            n += 1;
        }
        self.stat_unmaps += n as u64;
        MAPPED_BYTES.store(self.mapped_bytes() as u64, Relaxed);
        Ok(n)
    }

    pub fn mode(&self) -> Mode { self.mode }
    pub fn pool_pages(&self) -> usize { self.handles.len() }
    pub fn free_pages(&self) -> usize { self.free.len() }
    pub fn mapped_bytes(&self) -> usize { (self.handles.len() - self.free.len()) * PAGE }
    pub fn pool_bytes(&self) -> usize { self.handles.len() * PAGE }
    pub fn reserved_bytes(&self) -> usize { self.planes.iter().map(|p| p.bytes).sum() }
}

impl Drop for KvVmm {
    fn drop(&mut self) {
        // Best effort at teardown (the slices over these planes were leaked by the model's Drop).
        unsafe {
            for p in &self.planes {
                for (pg, h) in p.pages.iter().enumerate() {
                    if *h != 0 { let _ = cu::cuMemUnmap((p.base + (pg * PAGE) as u64) as cu::CUdeviceptr, PAGE); }
                }
                let _ = cu::cuMemAddressFree(p.base as cu::CUdeviceptr, p.bytes);
            }
            for h in &self.handles { let _ = cu::cuMemRelease(*h); }
        }
    }
}

/// True for the error `KvVmm::ensure` returns when every page of the pool is mapped (the caller may free a lane and retry).
pub fn is_exhausted(e: &anyhow::Error) -> bool { format!("{e:#}").contains("kv-vmm: pool exhausted") }

/// Rows of a position plane to keep mapped for a lane that will touch rows `< rows`: rounded up to a
/// grid so the fast path (an atomic compare) absorbs almost every call, capped at the plane.
pub fn rows_grid(rows: usize, cap: usize) -> usize {
    const GRID: usize = 4096;
    (rows.div_ceil(GRID) * GRID).min(cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rows_grid_rounds_up_and_caps() {
        assert_eq!(rows_grid(0, 1000), 0);
        assert_eq!(rows_grid(1, 100_000), 4096);
        assert_eq!(rows_grid(4096, 100_000), 4096);
        assert_eq!(rows_grid(4097, 100_000), 8192);
        assert_eq!(rows_grid(99_999, 100_000), 100_000);
    }
    /// The pool is opt-in (default off = today's allocation byte-for-byte); on a TP rank it falls back (never a refusal)
    /// with one line naming why.
    #[test]
    fn kv_pool_is_off_by_default_and_falls_back_under_tp() {
        use crate::opts::{Scope, Ty};
        let o = crate::opt!("kv-pool").def();
        assert_eq!((o.def, o.ty, o.scope), ("off", Ty::Choice(&["off", "full", "lazy"]), Scope::Spmd));
        assert_eq!(crate::opt!("kv-pool-gb").def().def, "unset");
        assert_eq!(crate::opt!("fit-floor-gb").def().def, "20");
        assert_eq!(fallback_line(None, 0, true), None, "not asked: nothing to say");
        assert_eq!(fallback_line(Some(Mode::Lazy), 2, false), None, "running: nothing to say");
        let (why, line) = fallback_line(Some(Mode::Lazy), 0, true).unwrap();
        assert_eq!(why, 1);
        assert!(line.starts_with("--kv-pool lazy asked on a TP rank: served with the standard per-lane KV allocation"), "{line}");
        let (why, line) = fallback_line(Some(Mode::Full), 0, false).unwrap();
        assert_eq!(why, 2);
        assert!(line.contains("not served by the EXL3 position planes"), "{line}");
    }
    /// The startup checks no longer refuse --kv-pool with a TP role (AGENTS 1b-bis(8)); the fallback is decided where
    /// the role is known (builder / TP load) and said when the API binds.
    #[test]
    fn no_startup_refusal_and_the_bind_notice_is_wired() {
        let sse = include_str!("server/sse.rs");
        assert!(!sse.contains("tp_topology"), "the startup refusal is gone");
        let srv = include_str!("server/mod.rs");
        assert!(srv.contains("crate::kv_vmm::bind_notice()"), "http_listen says why an asked-for pool is not running");
        let load = include_str!("exl3_forward/load.rs");
        assert!(load.contains("crate::kv_vmm::note_tp_role()"), "a TP load turns the pool off before any plane exists");
        let b = include_str!("exl3_serve/builder.rs");
        assert!(b.contains("crate::kv_vmm::note_tp_role()"), "the TP builder turns the pool off before the fit plan");
    }
}

#[cfg(test)]
mod dry_run_tests {
    use super::*;
    #[test]
    fn ensure_bounds_cover_every_touched_page() {
        assert_eq!(ensure_page_bounds(0, 1), (0, 1));
        assert_eq!(ensure_page_bounds(PAGE - 1, 2), (0, 2));          // straddles a boundary: both pages
        assert_eq!(ensure_page_bounds(PAGE, PAGE), (1, 2));
        assert_eq!(ensure_page_bounds(3 * PAGE, 0), (3, 3));          // empty
    }
    #[test]
    fn drop_bounds_free_only_wholly_covered_pages() {
        assert_eq!(drop_page_bounds(0, PAGE), (0, 1));
        assert_eq!(drop_page_bounds(1, PAGE), (1, 1));                 // straddles both edges: nothing wholly inside
        assert_eq!(drop_page_bounds(PAGE, 2 * PAGE), (1, 3));
        assert_eq!(drop_page_bounds(PAGE - 1, PAGE + 2), (1, 2));      // covers page 1 wholly
    }
    #[test]
    fn union_counts_a_shared_boundary_page_once_and_skips_mapped_pages() {
        // plane A: pages 0..8, page 3 already mapped; plane B: an equal-looking but distinct plane
        let mut a = vec![0u32; 8];
        a[3] = 1;
        let b = vec![0u32; 8];
        // two ranges on A that share page 2 (one ends inside it, the next starts inside it)
        let r1 = ensure_page_bounds(0, 2 * PAGE + 10);           // pages 0..3
        let r2 = ensure_page_bounds(2 * PAGE + 20, 3 * PAGE);    // pages 2..6
        assert_eq!(r1, (0, 3));
        assert_eq!(r2, (2, 6));
        // sum per range would be 3 + 3 = 6 (page 2 twice); the union is pages {0,1,2,4,5} = 5 (page 3 is mapped)
        assert_eq!(count_pages(&a, r1, false) + count_pages(&a, r2, false), 6);
        assert_eq!(count_union_unmapped(&[(&a, r1), (&a, r2)]), 5);
        // the same page index on ANOTHER plane is another page
        assert_eq!(count_union_unmapped(&[(&a, r1), (&b, r1)]), 6);
        // empty and out-of-range spans add nothing
        assert_eq!(count_union_unmapped(&[(&a, (4, 4)), (&a, (9, 12))]), 0);
        assert_eq!(count_union_unmapped(&[]), 0);
    }

    #[test]
    fn boundary_page_is_counted_once_in_a_union_not_per_lane() {
        // lane A: [0, PAGE+1) needs pages 0,1; lane B: [PAGE+1, 2*PAGE+5) needs pages 1,2 (page 1 is shared)
        let pages = [0u32, 0, 0, 0];
        let a = count_pages(&pages, ensure_page_bounds(0, PAGE + 1), false);
        let b = count_pages(&pages, ensure_page_bounds(PAGE + 1, PAGE + 4), false);
        let union = count_pages(&pages, (0, 3), false);
        assert_eq!((a, b, union), (2, 2, 3));   // per-lane sum double-counts page 1: never sum ensure queries across lanes
    }
    #[test]
    fn count_pages_respects_mapped_and_unmapped() {
        let pages = [0u32, 7, 0, 3, 0];
        assert_eq!(count_pages(&pages, (0, 5), false), 3);
        assert_eq!(count_pages(&pages, (0, 5), true), 2);
        assert_eq!(count_pages(&pages, (2, 2), false), 0);            // empty range
        assert_eq!(count_pages(&pages, (3, 99), true), 1);            // clamped to the plane
    }
}
