//! The device-buffer `Pool` (best-fit reuse, byte cap), the tripwire / census counters and raw-function lookups.
//! A pool trusts the allocation, never the caller's claim (AGENTS.md section 4); `alloc_zeros` does not zero (section 2.2). Map: `mod.rs`.

use super::*;

pub(super) fn state_stride(cfg: &Config) -> usize { cfg.max_position_embeddings }

/// Reusable device-buffer pool.
///
/// TWO PROPERTIES IT DID NOT HAVE, AND AN OOM THAT FOLLOWED.
///
/// It used to key buffers on their EXACT length and never free anything: `release` just pushed onto a
/// list, and `get(n)` only reused a buffer of length exactly `n`. Every prefill activation is sized by
/// the PROMPT LENGTH, so **every distinct prompt length ever seen leaked a full set of buffers,
/// permanently**. That is invisible in a benchmark (which reuses a handful of lengths) and fatal in real
/// use: a tool-eval suite sending hundreds of different prompt lengths exhausted GPU memory and the
/// allocation panicked — killing the scheduler task, after which every request returned zero tokens.
///
/// So:
///   * **Best-fit reuse.** Hand back the smallest cached buffer that is big enough (and not absurdly
///     oversized). Callers only ever touch the first `n` elements — the extra capacity is inert. This
///     is what collapses "one bucket per prompt length" into a handful of buckets.
///   * **A hard cap on cached bytes.** Past the cap, a released buffer is DROPPED (freed) rather than
///     hoarded. The pool is an optimisation; it must not be able to consume the device.
///
/// Reused buffers are NOT re-zeroed — only fresh allocations are (cudarc's `alloc_zeros` is
/// `cuMemAllocAsync` and does not actually zero). Every kernel that reads a pool buffer must therefore
/// write it first, which was already the contract.
pub struct Pool {
    dev: Arc<CudaDevice>,
    /// (buffer, BUCKET capacity) — never the caller's exact `n`. See `bucket`.
    free: Vec<(S, usize)>,
    free_bf16: Vec<(B, usize)>,
    cached_bytes: usize,
    cap_bytes: usize,
    /// E13 tripwire: set around cuStreamBeginCapture/EndCapture (capture_verify_graph). An alloc in
    /// this window is a capture-illegal NULL-stream op — `tripwire_fire` (always-on, see
    /// PLAN/TRIPWIRE_SPEC.md §3.3) makes a bucket the eager warmup failed to cover LOUD in EVERY
    /// build, not just debug ones. Until 2026-09-13 this was a `debug_assert!`, i.e. DEAD in every
    /// deployed (release) binary — the spec's "wrong thing #1".
    pub(super) capturing: std::cell::Cell<bool>,
}

// ---------------------------------------------------------------------------------------------
// The permanent OOB / alloc tripwire (PLAN/TRIPWIRE_SPEC.md).
//
// The class it exists for: a kernel that writes a PADDED footprint (the pf8 lane writes
// `tp64 = ceil(n/64)*64` rows) into a buffer the caller sized for the UNPADDED `n`. The write
// lands inside the pool's power-of-two bucket slack, so there is no fault, no assert, and the
// bytes are later handed to somebody else — the client sees a silently truncated stream while
// the head log holds nothing at all. Three production incidents (74a4611, ccaa234, af362ca).
//
// Three postures (spec §3.3):
//   off     — the shipped default. Counters still move; nothing is printed.
//   observe — production diagnosis: one rate-limited line per violation + a counter.
//   assert  — probes/tests: panic with site + numbers + remedy.
// The assert posture is DOWNGRADED to observe when the process is a TP rank (main.rs): an
// asymmetric abort is worse than the bug (the FIX #4 report: a one-sided --fp8-prefill=0
// killed the node and tripped the TP watchdog).
// ---------------------------------------------------------------------------------------------

/// Tripwire posture. `Off` still COUNTS (the check is a branch, not a print).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tripwire { Off = 0, Observe = 1, Assert = 2 }

static TRIPWIRE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static TW_FIRES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CENSUS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn tripwire_set(t: Tripwire) { TRIPWIRE.store(t as u8, std::sync::atomic::Ordering::Relaxed); }
pub fn tripwire_get() -> Tripwire {
    match TRIPWIRE.load(std::sync::atomic::Ordering::Relaxed) {
        2 => Tripwire::Assert, 1 => Tripwire::Observe, _ => Tripwire::Off,
    }
}
/// How many violations the tripwire has observed this process. The selftest REQUIRES this to move;
/// a check that has never fired is unverified (spec §3.5.2, the negative-control discipline).
pub fn tripwire_fires() -> u64 { TW_FIRES.load(std::sync::atomic::Ordering::Relaxed) }

pub fn census_set(on: bool) { CENSUS.store(on, std::sync::atomic::Ordering::Relaxed); }
pub fn census_on() -> bool { CENSUS.load(std::sync::atomic::Ordering::Relaxed) }

/// Report one observed violation: count it, print it (rate-limited), and panic at the assert tier.
/// Host-only and allocation-scoped — never per launch, never on the decode path.
pub fn tripwire_fire(what: &str, detail: &str) {
    let n = TW_FIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let t = tripwire_get();
    if t != Tripwire::Off || n <= 5 || n % 200 == 0 {
        eprintln!("[tripwire #{n}] {what} — {detail}");
    }
    if t == Tripwire::Assert {
        panic!("tripwire (assert posture): {what} — {detail}");
    }
}

/// One census record (spec §3.2). Opt-in, one line per prefill-path allocation; NEVER enabled for
/// a timing run (it prints per allocation).
fn census_line(dtype: &str, site: &'static str, req: usize, footprint: usize, cap: usize) {
    if !census_on() { return; }
    let pad = footprint.saturating_sub(req);
    eprintln!("[pool-census] dtype={dtype} site={site} req={req} footprint={footprint} \
               pad={pad} bucket={cap} slack={} elems{}",
              cap.saturating_sub(footprint), if pad > 0 { " — PADDED" } else { "" });
}

/// The pf8 prefill lane pads its token count to a 64 multiple. ONE definition (spec §2: the
/// hand-inlined copy at the batch-mixer site was a second, divergent copy of a numeric rule the
/// three OOB incidents were all about).
pub fn pf8_pad_pub(n: usize) -> usize { if n >= 256 { n.div_ceil(64) * 64 } else { n } }

/// Scheduler gate 4 (PLAN/SCHEDULER_2LANE_WORKDOC.md §5): record one prefill window's width and
/// the (n, tp64, bucket, pad-danger) tuple the pf8 lane presents for it. The gate is a SUBSET
/// claim over the multiset of `n` values the two schedulers present for the same prompt mix, so
/// the line is `n` + everything derived from `n` — a diff of two such streams is the artifact.
///
/// `row` is a representative sink row width, used only to make the pool-luck arithmetic visible:
/// the incident fires when `tp64 * row` overflows `bucket_up(n * row)` (REGRESSION_REPORT §5.1:
/// n ∈ (384, 409.6] for row = 5120).
pub fn pf_census_window(w0: usize, w1: usize, row: usize) {
    if !census_on() { return; }
    let n = w1 - w0;
    let tp64 = pf8_pad_pub(n);
    let b_unpadded = bucket_up(n * row);
    let b_padded = bucket_up(tp64 * row);
    let danger = tp64 * row > b_unpadded;
    eprintln!("[pool-census] pfwin w0={w0} w1={w1} n={n} tp64={tp64} \
               bucket_n={b_unpadded} bucket_tp64={b_padded} danger={}",
              if danger { "UNPADDED-WOULD-OOB" } else { "ok" });
}

/// Past this many bytes of CACHED (idle) buffers, released buffers are freed rather than hoarded.
/// Generous next to a 6 GB model + KV on a 128 GB box, while making unbounded growth impossible.
const POOL_CAP_BYTES: usize = 3 << 30;


/// What a request for `n` elements NEEDS: the next power-of-two bucket (min 256).
///
/// Bucketing is what makes reuse possible at all. Buffers used to be keyed on the caller's EXACT
/// length, and every prefill activation is sized by the PROMPT LENGTH — so each distinct prompt length
/// got its own permanent bucket and nothing was ever reused. Costs <2x slack, buys bounded growth.
fn bucket_up(n: usize) -> usize { n.max(256).next_power_of_two() }

/// The GB10 opt-in cap on dynamic shared memory per block: 99 KiB = 101,376 B (the 100 KiB/SM
/// carve-out with 1 KiB of reserved smem). TRIPWIRE_SPEC §3.4 — this number was cited in prose at
/// two load sites and ASSERTED nowhere; `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` is an
/// upper bound, not a check, so a budget past the cap fails the call and one BELOW the kernel's
/// real need overflows into illegal shared memory (the documented 81,312 → illegal-smem burn).
pub const SMEM_OPTIN_CAP: i32 = 99 * 1024;

/// Log and check one raw-PTX dynamic-smem budget (TRIPWIRE_SPEC §3.4): the boot log must record
/// the budget (so a re-derived number can be compared), and the cap must be a checked constant.
fn smem_budget(tag: &str, bytes: i32) {
    if crate::boot_report::debug() {
        eprintln!("[{tag}] dynamic smem budget {bytes} B / cap {SMEM_OPTIN_CAP} B ({} KiB)",
                  bytes / 1024);
    }
    if bytes > SMEM_OPTIN_CAP {
        tripwire_fire("dynamic smem budget exceeds the GB10 opt-in cap",
            &format!("{tag}: {bytes} B > {SMEM_OPTIN_CAP} B — cuFuncSetAttribute would fail; \
                      re-derive the section sizes (prefer an expression to a literal)"));
    }
}

/// P4 B3: one-time raw-module load of gpu_batch.ptx, fetch gdn_chunk_prefill_b, and opt in
/// to ~86KB dynamic smem (cuFuncSetAttribute). None disables the chunked path gracefully.
pub(super) fn gdn_chunk_load_raw_fn() -> Option<cudarc::driver::sys::CUfunction> {
    use cudarc::driver::sys;
    let ptx = crate::ptx_embed::GPU_BATCH;
    // NULL-terminated PTX (the 2026-08-11 lesson; see the E9 loader comment)
    let ptx_c = std::ffi::CString::new(ptx).ok()?;
    let mut module: sys::CUmodule = std::ptr::null_mut();
    let r = unsafe {
        sys::cuModuleLoadDataEx(&mut module, ptx_c.as_ptr() as *const std::ffi::c_void,
                                0, std::ptr::null_mut(), std::ptr::null_mut())
    };
    if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[b3] module load failed ({r:?})"); return None; }
    // --gdn-oldstage=1: diagnostic twin with the pre-P0 in-kernel staging (see gpu_batch.cu) —
    // pairs with --gdn-xcheck to attribute chunked-vs-seq divergence to the P0 staging rework.
    let kname = if crate::opts::var(crate::opt!("gdn-oldstage")).is_ok() {
        eprintln!("[b3] DIAGNOSTIC: gdn_chunk_prefill_b_oldstage selected (--gdn-oldstage)");
        "gdn_chunk_prefill_b_oldstage"
    } else { "gdn_chunk_prefill_b" };
    let cname = std::ffi::CString::new(kname).ok()?;
    let mut f: sys::CUfunction = std::ptr::null_mut();
    let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
    if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[b3] get {kname} failed ({r:?})"); return None; }
    // Exact budget at the actual dims (kd=128, VC=64): ~86KB — under the ~99KB GB10 opt-in cap.
    // (Do NOT size for kd=256 worst case: 148KB > cap and the attribute call fails.)
    let bytes: i32 = (128 * 64 + 32*132*2 + 32*64*2 + 32*32*2 + 32 + 33) as i32 * 4;
    smem_budget("b3", bytes);
    let r = unsafe {
        sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes)
    };
    if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[b3] smem opt-in failed ({r:?})"); return None; }
    Some(f)
}

/// --gdn-chunk2 (2026-08-26): raw fetch of `gdn_chunk_tc_b` — the tensor-core chunked GDN
/// scan (probe: 9.4 ms/layer @8K vs 198 sequential = 21x; o/S rel-L2 ~2.2e-2 = bf16-operand
/// envelope). Same opt-in pattern; smem budget at kd=128/VC=64/C=32: ~81.3KB (under the ~99KB cap).
pub(super) fn gdn_chunk_tc_load_raw_fn() -> Option<cudarc::driver::sys::CUfunction> {
    use cudarc::driver::sys;
    let ptx = crate::ptx_embed::GPU_BATCH;
    let ptx_c = std::ffi::CString::new(ptx).ok()?;
    let mut module: sys::CUmodule = std::ptr::null_mut();
    let r = unsafe {
        sys::cuModuleLoadDataEx(&mut module, ptx_c.as_ptr() as *const std::ffi::c_void,
                                0, std::ptr::null_mut(), std::ptr::null_mut())
    };
    if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[gtc] module load failed ({r:?})"); return None; }
    let cname = std::ffi::CString::new("gdn_chunk_tc_b").ok()?;
    let mut f: sys::CUfunction = std::ptr::null_mut();
    let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
    if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[gtc] get gdn_chunk_tc_b failed ({r:?})"); return None; }
    // bf16 tiles: (32*136*2 + 128*40 + 32*72*3 + 32*40 + 128*72)*2 + f32
    // (32*32 + 32*64*2 + 32 + 33)*4 = 83,204 B (the probe's runtime figure — a hardcoded
    // 81,312 was 1.9KB short and overflowed Sb into illegal smem)
    let bytes: i32 = 83204;
    smem_budget("gtc", bytes);
    let r = unsafe {
        sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes)
    };
    if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[gtc] smem opt-in failed ({r:?})"); return None; }
    Some(f)
}

/// F8 P2c' — one-time raw-module fetch of the SPLIT chunked GDN pair
/// (`gdn_chunk_states_b` + `gdn_chunk_wuo_b`). Same pattern as `gdn_chunk_tc_load_raw_fn`.
/// Returns (states_fn, wuo_fn).
#[allow(clippy::type_complexity)]
fn gdn_split_load_raw_fns() -> Option<(cudarc::driver::sys::CUfunction, cudarc::driver::sys::CUfunction)> {
    use cudarc::driver::sys;
    let ptx = crate::ptx_embed::GPU_BATCH;
    let ptx_c = std::ffi::CString::new(ptx).ok()?;
    let mut module: sys::CUmodule = std::ptr::null_mut();
    let r = unsafe {
        sys::cuModuleLoadDataEx(&mut module, ptx_c.as_ptr() as *const std::ffi::c_void,
                                0, std::ptr::null_mut(), std::ptr::null_mut())
    };
    if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[gsc] module load failed ({r:?})"); return None; }
    // H: Sb 128*72*2 + Kt 128*40*2 + Vb 32*72*2 + f32 (32+32+33)*4 = 33,668 + slack.
    let h_smem: i32 = 34816;
    // WUO: the fused kernel's exact section layout (83204 B).
    let wuo_smem: i32 = 83204;
    let mut out = [std::ptr::null_mut(); 2];
    for (i, name) in ["gdn_chunk_states_b", "gdn_chunk_wuo_b"].iter().enumerate() {
        let cname = std::ffi::CString::new(*name).ok()?;
        let mut f: sys::CUfunction = std::ptr::null_mut();
        let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
        if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[gsc] get {name} failed ({r:?})"); return None; }
        let bytes: i32 = if i == 0 { h_smem } else { wuo_smem };
        smem_budget(if i == 0 { "gsc:states" } else { "gsc:wuo" }, bytes);
        let r = unsafe {
            sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes)
        };
        if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[gsc] {name} smem opt-in failed ({r:?})"); return None; }
        out[i] = f;
    }
    Some((out[0], out[1]))
}

/// Process-wide handles for the split pair (module loads once).
#[allow(clippy::type_complexity)]
pub(super) fn gdn_split_fns() -> Option<(cudarc::driver::sys::CUfunction, cudarc::driver::sys::CUfunction)> {
    #[derive(Clone, Copy)]
    struct FnH(cudarc::driver::sys::CUfunction);
    unsafe impl Send for FnH {}
    unsafe impl Sync for FnH {}
    static V: std::sync::OnceLock<Option<(FnH, FnH)>> = std::sync::OnceLock::new();
    V.get_or_init(|| gdn_split_load_raw_fns().map(|(a, b)| (FnH(a), FnH(b))))
        .as_ref()
        .map(|(a, b)| (a.0, b.0))
}

/// One-time raw-module fetch of `attn_prefill_fa_b` (the tensor-core fused flash-attention
/// prefill kernel) + the 64 KiB dynamic-smem opt-in (cuFuncSetAttribute — same pattern as
/// `gdn_chunk_load_raw_fn`). None falls the caller back to the scalar/oracle kernel gracefully.
pub(super) fn fa_tc_raw_fn() -> Option<cudarc::driver::sys::CUfunction> {
    // Newtype: a CUfunction is an opaque copied handle — sharing it across threads is sound.
    #[derive(Clone, Copy)]
    struct FnH(cudarc::driver::sys::CUfunction);
    unsafe impl Send for FnH {}
    unsafe impl Sync for FnH {}
    static V: std::sync::OnceLock<Option<FnH>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        use cudarc::driver::sys;
        let ptx = crate::ptx_embed::GPU_BATCH;
        let ptx_c = std::ffi::CString::new(ptx).ok()?;
        let mut module: sys::CUmodule = std::ptr::null_mut();
        let r = unsafe {
            sys::cuModuleLoadDataEx(&mut module, ptx_c.as_ptr() as *const std::ffi::c_void,
                                    0, std::ptr::null_mut(), std::ptr::null_mut())
        };
        if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[fa] module load failed ({r:?})"); return None; }
        let cname = std::ffi::CString::new("attn_prefill_fa_b").ok()?;
        let mut f: sys::CUfunction = std::ptr::null_mut();
        let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
        if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[fa] get attn_prefill_fa_b failed ({r:?})"); return None; }
        // 4 x 32 keys x 256 dims x 2 B = 65536 (hd=256; the attribute covers the max we request)
        smem_budget("fa", 65536);
        let r = unsafe {
            sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536)
        };
        if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[fa] smem opt-in failed ({r:?})"); return None; }
        Some(FnH(f))
    }).map(|h| h.0)
}

/// Cached env knob for the P4 B2 fast-prefill path (per-GEMM-launch hot path — no allocs).
pub(super) fn pf4_rm_nocache() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("pf-rm-nocache")).is_ok())
}

/// What a buffer of TRUE capacity `cap` can SERVE: the largest power-of-two bucket that fits inside it
/// (0 = too small to pool at all).
///
/// `release` must use this, never `bucket_up(n)` on the caller's `n`. Not every buffer handed to the
/// pool was allocated BY the pool: `embed_batch` allocates the prefill residual at EXACTLY `h*n`, and
/// `admit()` releases it. Filing that under `bucket_up(h*n)` claims it can serve up to 2x its real
/// size — so the pool later handed a 1033-column residual (4,231,168 elems) to a request that needed
/// 8,388,608 and wrote 16 MB into a 8 MB allocation. That surfaced as `CUDA_ERROR_INVALID_VALUE` on an
/// innocent memcpy, hours into a benchmark. The pool must trust the ALLOCATION, not the caller's claim.
fn bucket_down(cap: usize) -> usize {
    if cap < 256 { return 0; }
    1usize << (usize::BITS - 1 - cap.leading_zeros()) as usize
}

impl Pool {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        Pool { dev, free: vec![], free_bf16: vec![], cached_bytes: 0, cap_bytes: POOL_CAP_BYTES,
               capturing: std::cell::Cell::new(false) }
    }

    pub fn get(&mut self, n: usize) -> S {
        let b = bucket_up(n);
        if let Some(pos) = self.free.iter().position(|x| x.1 == b) {
            let (buf, _) = self.free.swap_remove(pos);
            self.cached_bytes -= buf.len() * 4;
            assert!(buf.len() >= n, "pool handed out {} f32 for a request of {n}", buf.len());
            return buf;
        }
        // E13 tripwire: a MISS here during graph capture is a capture-illegal NULL-stream alloc —
        // the eager warmup failed to cover this bucket (see capture_verify_graph). ALWAYS-ON
        // (TRIPWIRE_SPEC §3.3): a `debug_assert!` here was dead code in every deployed binary.
        if self.capturing.get() {
            tripwire_fire("pool f32 alloc during CUDA-graph capture",
                &format!("bucket={b} elems — the eager warmup did not cover this bucket"));
        }
        // cudarc's alloc_zeros is cuMemAllocAsync — it does NOT zero. Zero fresh allocations explicitly
        // so any pool buffer handed out is deterministic (a stale read would otherwise pick up GPU
        // garbage that varies run-to-run). Reused buffers are NOT re-zeroed: every kernel that reads a
        // pool buffer must write it first, which was already the contract.
        let mut buf = self.alloc_f32(b);
        self.dev.memset_zeros(&mut buf).unwrap();
        self.dev.synchronize().unwrap(); // ensure zeroing is visible to the compute stream
        buf
    }

    /// TRIPWIRE_SPEC §3.2/§3.3 — a pool hand-out that DECLARES the owning kernel's write footprint.
    ///
    /// `req` is what the pool must hand out (the SAME contract as the two-argument form, so a
    /// converted site allocates exactly what it did before); `footprint` is what the owning kernel
    /// will actually WRITE. The check is `capacity >= footprint`, and it is DELIBERATELY detection,
    /// not prevention: the spec's own claim is that a tripwire "does not fix the bug class, it
    /// shortens the time-to-detection" — silently allocating the padded size here would make the
    /// check vacuous for the very caller it exists to catch. Host-known numbers only: no sync, no
    /// readback (rule 16/17 reject a device-side check on this path by design).
    ///
    /// NEVER use on the decode path: the decode pool sites keep the two-argument form.
    pub fn get_fp(&mut self, site: &'static str, req: usize, footprint: usize) -> S {
        let buf = self.get(req);
        census_line("f32", site, req, footprint, buf.len());
        if footprint > buf.len() {
            tripwire_fire("declared write footprint exceeds the handed-out capacity",
                &format!("site={site} req={req} footprint={footprint} cap={} \
                          — UNDER-SIZED CALLER (the pf8 pad class); size the request to the \
                          padded width (pf8_pad(n))", buf.len()));
        }
        buf
    }

    /// Allocate, and if the device is out of memory, DROP THE CACHE AND TRY AGAIN before giving up.
    ///
    /// An OOM here used to panic on a tokio worker, which killed the scheduler task — after which the
    /// server kept accepting requests and returning ZERO TOKENS forever. A zombie is worse than a crash.
    /// The pool is only an optimisation: under pressure it should surrender its idle buffers, not the
    /// process.
    fn alloc_f32(&mut self, n: usize) -> S {
        match self.dev.alloc_zeros::<f32>(n) {
            Ok(b) => b,
            Err(_) => {
                let freed = self.cached_bytes;
                self.free.clear();
                self.free_bf16.clear();
                self.cached_bytes = 0;
                let _ = self.dev.synchronize();
                eprintln!("[pool] OUT OF MEMORY allocating {} f32 — dropped {:.1} MB of cached buffers, retrying",
                          n, freed as f64 / 1e6);
                self.dev.alloc_zeros::<f32>(n).expect("alloc f32 after dropping the pool cache")
            }
        }
    }

    fn alloc_bf16(&mut self, n: usize) -> B {
        match self.dev.alloc_zeros::<half::bf16>(n) {
            Ok(b) => b,
            Err(_) => {
                let freed = self.cached_bytes;
                self.free.clear();
                self.free_bf16.clear();
                self.cached_bytes = 0;
                let _ = self.dev.synchronize();
                eprintln!("[pool] OUT OF MEMORY allocating {} bf16 — dropped {:.1} MB of cached buffers, retrying",
                          n, freed as f64 / 1e6);
                self.dev.alloc_zeros::<half::bf16>(n).expect("alloc bf16 after dropping the pool cache")
            }
        }
    }

    /// Return a buffer to the pool. ALWAYS caches — never frees here.
    ///
    /// Freeing at release is a use-after-free waiting to happen: `release` is called IMMEDIATELY after
    /// launching the kernels that read the buffer, long before they run. Dropping it there hands the
    /// memory back to CUDA while queued work still points at it, and the next allocation reuses the
    /// address — which surfaces as a baffling `CUDA_ERROR_INVALID_VALUE` on some later, innocent memcpy.
    /// (I did exactly this and it blew up on the first run.) Eviction happens in `trim`, which syncs.
    pub fn release(&mut self, s: S, _n: usize) {
        // TRIPWIRE_SPEC §2 asked for a "s.len() power-of-two >= 256" sanity check here. That
        // premise is WRONG, and this session measured it: the check fired 27,253 times on ONE
        // 512-token leg. Non-power-of-two capacities are normal and documented — `embed_batch`
        // allocates the prefill residual at exactly `h*n`, and the decode/verify path releases
        // `h*batch` views (len 20,480 = 5120*4 on the 27B). `bucket_down` exists precisely to file
        // those under the largest bucket that FITS, and lengths < 256 are deliberately DROPPED.
        // A length check cannot see the hazard the spec was reaching for (a release of a buffer that
        // is still in flight — a use-after-free class); the pool's own accounting invariant can, and
        // `trim` checks it. What IS worth recording is how much slack the pool is asked to round.
        let b = bucket_down(s.len());   // TRUE capacity — see bucket_down
        if b != s.len() && census_on() {
            eprintln!("[pool-census] release-rounddown f32 true_cap={} bucket={} slack={} elems",
                      s.len(), b, s.len() - b);
        }
        if b == 0 { return; }
        self.cached_bytes += s.len() * 4;
        self.free.push((s, b));
    }

    pub fn get_bf16(&mut self, n: usize) -> B {
        let b = bucket_up(n);
        if let Some(pos) = self.free_bf16.iter().position(|x| x.1 == b) {
            let (buf, _) = self.free_bf16.swap_remove(pos);
            self.cached_bytes -= buf.len() * 2;
            assert!(buf.len() >= n, "pool handed out {} bf16 for a request of {n}", buf.len());
            return buf;
        }
        // E13 tripwire: a MISS here during graph capture is a capture-illegal NULL-stream alloc —
        // the eager warmup failed to cover this bucket (see capture_verify_graph). ALWAYS-ON
        // (TRIPWIRE_SPEC §3.3): a `debug_assert!` here was dead code in every deployed binary.
        if self.capturing.get() {
            tripwire_fire("pool bf16 alloc during CUDA-graph capture",
                &format!("bucket={b} elems — the eager warmup did not cover this bucket"));
        }
        let mut buf = self.alloc_bf16(b);
        self.dev.memset_zeros(&mut buf).unwrap();
        self.dev.synchronize().unwrap();
        buf
    }

    /// bf16 twin of [`Pool::get_fp`] — the census/footprint form for the PREFILL path.
    pub fn get_bf16_fp(&mut self, site: &'static str, req: usize, footprint: usize) -> B {
        let buf = self.get_bf16(req);
        census_line("bf16", site, req, footprint, buf.len());
        if footprint > buf.len() {
            tripwire_fire("declared write footprint exceeds the handed-out capacity",
                &format!("site={site} req={req} footprint={footprint} cap={} \
                          — UNDER-SIZED CALLER (the pf8 pad class); size the request to the \
                          padded width (pf8_pad(n))", buf.len()));
        }
        buf
    }

    pub fn release_bf16(&mut self, s: B, _n: usize) {
        // See `release` — the spec's power-of-two premise is wrong (27,253 legitimate non-pow2
        // releases in one 512-token leg); the pool rounds down and files by TRUE capacity.
        let b = bucket_down(s.len());   // TRUE capacity — see bucket_down
        if b != s.len() && census_on() {
            eprintln!("[pool-census] release-rounddown bf16 true_cap={} bucket={} slack={} elems",
                      s.len(), b, s.len() - b);
        }
        if b == 0 { return; }
        self.cached_bytes += s.len() * 2;
        self.free_bf16.push((s, b));
    }

    /// Bytes currently idle in the pool.
    pub fn cached_bytes(&self) -> usize { self.cached_bytes }

    /// Evict idle buffers down to the cap. Call ONLY where it is safe to free device memory — this
    /// SYNCHRONIZES first, because a cached buffer may still be the target of kernels that were launched
    /// before it was released and have not run yet.
    ///
    /// This is what keeps the pool bounded. Without it, every distinct prompt length permanently
    /// retained its own set of prefill activations (90–600 MB each on 9B) and a tool-eval suite with
    /// hundreds of prompt lengths exhausted the GPU.
    pub fn trim(&mut self) {
        // Pool invariant (TRIPWIRE_SPEC §2): `cached_bytes` must equal the bytes actually idle in
        // the two freelists. Nothing checked this before; a mis-sized release or a dropped bucket
        // would make the pool's own cap accounting lie forever. One pass over a short list, once
        // per request — never on the decode step.
        let tally = self.free.iter().map(|x| x.0.len() * 4).sum::<usize>()
                  + self.free_bf16.iter().map(|x| x.0.len() * 2).sum::<usize>();
        if tally != self.cached_bytes {
            tripwire_fire("pool accounting drift (cached_bytes != freelist)",
                &format!("cached_bytes={} freelist={} (f32 {} bufs, bf16 {} bufs)",
                         self.cached_bytes, tally, self.free.len(), self.free_bf16.len()));
            self.cached_bytes = tally;   // self-heal so the cap keeps working
        }
        if self.cached_bytes <= self.cap_bytes { return; }
        let before = self.cached_bytes;
        self.dev.synchronize().unwrap();      // no queued kernel may still reference what we free
        // Drop the biggest buffers first: they are the ones a rare long prompt left behind.
        self.free.sort_by_key(|x| x.0.len());
        self.free_bf16.sort_by_key(|x| x.0.len());
        while self.cached_bytes > self.cap_bytes {
            let f32_big = self.free.last().map(|x| x.0.len() * 4).unwrap_or(0);
            let bf_big = self.free_bf16.last().map(|x| x.0.len() * 2).unwrap_or(0);
            if f32_big == 0 && bf_big == 0 { break; }
            if f32_big >= bf_big {
                self.free.pop();                          // dropped here -> device memory freed
                self.cached_bytes -= f32_big;
            } else {
                self.free_bf16.pop();
                self.cached_bytes -= bf_big;
            }
        }
        eprintln!("[pool] trimmed {:.1} GB -> {:.1} GB of idle buffers",
                  before as f64 / 1e9, self.cached_bytes as f64 / 1e9);
    }
}

// ---------------------------------------------------------------------------------------------
// TRIPWIRE_SPEC §3.5.2 — the SELFTEST (the negative control).
//
// "A tripwire that has never fired in a test is unverified." This mode runs FOUR legs against the
// real Pool on the real device and exits 0 only if every leg holds:
//
//   A  the historical incident reproduced: req = 468*8704, footprint = 512*8704 (ccaa234 FIX #3's
//      arithmetic, the shape that produced `pf8 lane OOB: out has 4194304 elems < tp64*outn =
//      384*12288`) — the footprint check MUST fire.
//   B  the mechanism that made it SILENT, demonstrated on device: a 0xAA fill of the whole bucket,
//      then a 0x00 write of the FOOTPRINT (which is larger than the request but inside the bucket),
//      then a readback proving [req, footprint) was overwritten while [footprint, cap) was not.
//      Nothing faults; that is exactly why three production incidents shipped.
//   C  zero false positives: a correctly padded site (req == footprint == pf8_pad(n)*width) must
//      NOT fire.
//   D  the capture-window check is LIVE in a release build (it used to be a `debug_assert!`, i.e.
//      compiled out of every deployed binary — spec "wrong thing #1").
// ---------------------------------------------------------------------------------------------

/// Run the tripwire selftest. Returns the process exit code (0 = all legs hold).
pub fn tripwire_selftest() -> i32 {
    use cudarc::driver::sys;
    let dev = CudaDevice::new(0).expect("selftest: CUDA device 0");
    let stream = fork_blocking_stream(&dev);
    let mut pool = Pool::new(dev.clone());
    let tier = tripwire_get();
    eprintln!("[tripwire-selftest] release_build={} tier={:?} fires_at_entry={}",
              !cfg!(debug_assertions), tier, tripwire_fires());
    let mut ok = true;

    // ---- Leg A: the historical incident, through the real check path -------------------------
    // At the assert tier the check PANICS by design; catch it and require the panic.
    let (req_a, fp_a) = (468usize * 8704, 512usize * 8704);
    let f0 = tripwire_fires();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = pool.get_bf16_fp("selftest.incident_ccaa234", req_a, fp_a);
    })).is_err();
    let f1 = tripwire_fires();
    let leg_a = f1 == f0 + 1 && (tier != Tripwire::Assert || panicked);
    eprintln!("[tripwire-selftest] A incident(req={req_a} footprint={fp_a} pool_bucket={}) \
               fired={} panicked={} -> {}",
              bucket_up(req_a), f1 - f0, panicked, if leg_a { "PASS" } else { "FAIL" });
    ok &= leg_a;

    // ---- Leg B: the SILENT case (pool luck), demonstrated live on device ----------------------
    // The dangerous case (leg A) is the one where the bucket does NOT cover the footprint — there
    // the write is genuinely out of bounds, so this leg deliberately does NOT perform it (it would
    // corrupt the async allocator's heap; that is what the three production incidents did).
    // What is demonstrated here is the OTHER half of the class, the one that made the bug silent:
    // a write past the REQUEST but inside the BUCKET is invisible to every check the engine has.
    let req_b: usize = 1_000_000;               // bucket_up => 1,048,576
    let fp_b: usize = 1_040_000;                // > req_b, <= bucket => no fire, no fault
    let cap_b = bucket_up(req_b);
    let buf_b = pool.get_bf16_fp("selftest.slack", req_b, req_b);
    let dptr_b = *buf_b.device_ptr();
    unsafe {
        sys::cuMemsetD8Async(dptr_b as sys::CUdeviceptr, 0xAA, cap_b * 2, stream.stream);
        sys::cuMemsetD8Async(dptr_b as sys::CUdeviceptr, 0x00, fp_b * 2, stream.stream);
    }
    let mut inside: u16 = 0xFFFF;   // index fp_b-1: inside the declared footprint, PAST the request
    let mut past: u16 = 0xFFFF;     // index cap_b-1: past the footprint, inside the bucket
    unsafe {
        sys::cuMemcpyDtoHAsync_v2(&mut inside as *mut u16 as *mut std::ffi::c_void,
            (dptr_b + ((fp_b - 1) * 2) as u64) as sys::CUdeviceptr, 2, stream.stream);
        sys::cuMemcpyDtoHAsync_v2(&mut past as *mut u16 as *mut std::ffi::c_void,
            (dptr_b + ((cap_b - 1) * 2) as u64) as sys::CUdeviceptr, 2, stream.stream);
    }
    dev.synchronize().expect("selftest sync");
    let f_b = tripwire_fires();
    let leg_b = inside == 0x0000 && past == 0xAAAA && cap_b > req_b && fp_b <= cap_b
                && f_b == f1;   // leg A's fire is the last one; THIS allocation must stay SILENT
    eprintln!("[tripwire-selftest] B silent-slack req={req_b} bucket={cap_b} footprint={fp_b} \
               [fp-1]={inside:#06x} (expect 0x0000 — written PAST the request, no check fired) \
               [cap-1]={past:#06x} (expect 0xaaaa — untouched) extra_fires={} -> {}",
              f_b - f1, if leg_b { "PASS" } else { "FAIL" });
    ok &= leg_b;

    // ---- Leg C: zero false positives on a correctly padded site ------------------------------
    let (h, n) = (5120usize, 468usize);
    let f2 = tripwire_fires();
    let _ = pool.get_bf16_fp("selftest.clean_padded", h * n, h * pf8_pad_pub(n));
    let leg_c = tripwire_fires() == f2;
    eprintln!("[tripwire-selftest] C clean padded site (n={n} tp64={}) fired={} -> {}",
              pf8_pad_pub(n), tripwire_fires() - f2, if leg_c { "PASS" } else { "FAIL" });
    ok &= leg_c;

    // ---- Leg D: the capture-window check is live in THIS build --------------------------------
    let f3 = tripwire_fires();
    pool.capturing.set(true);
    let d_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = pool.get_bf16(1 << 26 | 1);     // a bucket the selftest has certainly not cached
    })).is_err();
    pool.capturing.set(false);
    let leg_d = tripwire_fires() == f3 + 1 && (tier != Tripwire::Assert || d_panicked);
    eprintln!("[tripwire-selftest] D capture-window check live fired={} panicked={} \
               (release build: this was a DEAD debug_assert!) -> {}",
              tripwire_fires() - f3, d_panicked, if leg_d { "PASS" } else { "FAIL" });
    ok &= leg_d;

    eprintln!("[tripwire-selftest] {} legs=4 fires_total={} tier={:?}",
              if ok { "PASS" } else { "FAIL" }, tripwire_fires(), tier);
    if ok { 0 } else { 1 }
}
