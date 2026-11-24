//! Replay plumbing for the SR / LI units: event helpers, `run_unit`, write-set regions and the graph-node hash / digest used to prove a replay left no trace.
//! Part of the `--autotune` core (see `mod.rs` for the file map).

use super::*;

pub(super) fn ev_create() -> Result<sys::CUevent> {
    let mut e: sys::CUevent = std::ptr::null_mut();
    let r = unsafe { sys::cuEventCreate(&mut e, 0) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventCreate {r:?}");
    Ok(e)
}
pub(super) fn ev_record(e: sys::CUevent, s: sys::CUstream) -> Result<()> {
    let r = unsafe { sys::cuEventRecord(e, s) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventRecord {r:?}");
    Ok(())
}
pub(super) fn ev_elapsed_ms(a: sys::CUevent, b: sys::CUevent) -> Result<f64> {
    let mut ms: f32 = 0.0;
    let r = unsafe { sys::cuEventElapsedTime(&mut ms, a, b) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventElapsedTime {r:?}");
    Ok(ms as f64)
}

/// Balanced LI schedule (§5.3, v2): consecutive blocks of `arms` steps are the successive rows of a
/// Williams design (tune::williams_row) — every block is a permutation, every arm takes every
/// in-block position equally often and follows every other arm equally often (carry-over balanced)
/// over each period. Deterministic; the start row comes from `seed`. Returns (arm, in-block position).
pub(super) struct LiSched {
    arms: usize,
    row: Vec<usize>,
    r: usize,
    j: usize,
}
impl LiSched {
    pub(super) fn new(arms: usize, seed: u64) -> LiSched {
        let arms = arms.max(1);
        LiSched { arms, row: Vec::new(), r: (seed as usize) % tune::williams_period(arms), j: 0 }
    }
    pub(super) fn next(&mut self) -> (usize, u8) {
        if self.j >= self.row.len() {
            self.row = tune::williams_row(self.arms, self.r);
            self.r += 1;
            self.j = 0;
        }
        let (a, p) = (self.row[self.j], self.j as u8);
        self.j += 1;
        (a, p)
    }
}

// ---------------------------------------------------------------------------------------------
// Units + write sets
// ---------------------------------------------------------------------------------------------

/// Tap restore (the served mtp_round's first launch): taps_keep row `tap_row` of `slot`'s window
/// (WP03 tap_base) -> sc.resid row 0.
pub(super) fn tap_restore(l: &Launcher, model: &FwdModel, sc: &mut Scratch, slot: usize, tap_row: usize) -> Result<()> {
    if tap_row == usize::MAX { return Ok(()); }
    let he = model.cfg.hc_count.max(1) * model.cfg.hidden_size;
    let tb = model.tap_base(sc, slot)?;
    xqlaunch!(l, "xq_copy_f32", ((((he as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
              (&mut sc.resid, &sc.taps_keep, he as i64, 0i64, (tb + tap_row * he) as i64))
}

/// A unit's launches (graph-capturable: launches only). The selftest spin node (§5.7 test 2) closes
/// the unit when the arm's overlay sets it.
pub(super) fn run_unit(model: &FwdModel, l: &Launcher, sc: &mut Scratch, head: Option<&DraftHead>, key: UnitKey) -> Result<()> {
    let cx = match key {
        UnitKey::Verify { m, slot, qsa, .. } => {
            let _tc = tune::enter(Ctx::fam(Fam::Verify, m).regime(qsa));
            let he = model.cfg.hc_count.max(1) * model.cfg.hidden_size;
            // verify_block minus verify_commit (the live state never moves in a replay)
            model.verify_shadow(l, sc, slot)?;
            model.verify_kernels(l, sc, m, qsa, slot, 0)?;
            // taps rows 0..m-1 into THIS slot's window (WP03 tap_base, as verify_block)
            let tb = model.tap_base(sc, slot)?;
            xqlaunch!(l, "xq_copy_f32", (((((he * m) as i64) + 255) / 256).max(1) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut sc.taps_keep, &sc.resid, (he * m) as i64, tb as i64, 0i64))?;
            xqlaunch!(l, "xq_accept", (1, 1, 1), (32, 1, 1), 0,
                      (&sc.d_dev, &sc.argmax, &mut sc.accept_out, &mut sc.acc2, (m - 1) as i32, slot as i32))?;
            tune::current()
        }
        UnitKey::Draft { s, qsa, .. } => {
            let head = head.context("draft unit without a head")?;
            let _tc = tune::enter(Ctx::fam(Fam::DraftChain, 1).regime(qsa));
            model.draft_block(l, sc, head, s, qsa)?;
            tune::current()
        }
        UnitKey::Step { m, qsa, .. } => {
            model.forward_step_kernels(l, sc, m, None, false, qsa, 0)?; // a tuned unit is a one-regime step (md = 0)
            Ctx::fam(Fam::Step, m).regime(qsa)
        }
    };
    let ns = tune::get(&tune::T_SELFTEST_SPIN, &cx);
    if ns > 0 {
        xqlaunch!(l, "xq_spin_ns", (1, 1, 1), (32, 1, 1), 0, (ns as i64, 0i32))?;
    }
    Ok(())
}

/// One digest region: `words` 32-bit words at device address `ptr` (4-byte aligned).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Reg {
    pub(super) ptr: u64,
    pub(super) words: u64,
}

pub(super) fn reg(ptr: u64, bytes: usize) -> Result<Reg> {
    anyhow::ensure!(ptr % 4 == 0, "digest region not 4-byte aligned ({ptr:#x})");
    Ok(Reg { ptr, words: (bytes / 4) as u64 }) // a trailing partial word (never on this model) is skipped
}

pub(super) fn dptr<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }

// ---------------------------------------------------------------------------------------------
// §4.8b inertness: the node-parameter hash of a captured graph
// ---------------------------------------------------------------------------------------------

/// CUDA_KERNEL_NODE_PARAMS_v2 (the v1 struct is its 56-byte prefix).
#[repr(C)]
struct KParamsV2 {
    func: sys::CUfunction,
    gx: u32, gy: u32, gz: u32,
    bx: u32, by: u32, bz: u32,
    smem: u32,
    params: *mut *mut std::ffi::c_void,
    extra: *mut *mut std::ffi::c_void,
    kern: *mut std::ffi::c_void,
    ctx: *mut std::ffi::c_void,
}
type GetKParamsFn = unsafe extern "C" fn(sys::CUgraphNode, *mut KParamsV2) -> sys::CUresult;
type ParamInfoFn = unsafe extern "C" fn(sys::CUfunction, usize, *mut usize, *mut usize) -> sys::CUresult;

/// Driver entry points resolved at run time (dlsym): never a link-time dependency, so a driver
/// without them only degrades the inertness check (argument bytes unhashed), never the binary.
fn driver_sym(name: &str) -> Option<usize> {
    let c = std::ffi::CString::new(name).ok()?;
    let p = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c.as_ptr()) };
    if p.is_null() { None } else { Some(p as usize) }
}
fn kparams_fn() -> Option<GetKParamsFn> {
    static F: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    F.get_or_init(|| driver_sym("cuGraphKernelNodeGetParams_v2").or_else(|| driver_sym("cuGraphKernelNodeGetParams")))
        .map(|p| unsafe { std::mem::transmute::<usize, GetKParamsFn>(p) })
}
fn param_info_fn() -> Option<ParamInfoFn> {
    static F: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    F.get_or_init(|| driver_sym("cuFuncGetParamInfo"))
        .map(|p| unsafe { std::mem::transmute::<usize, ParamInfoFn>(p) })
}

/// FNV-1a 64 over a captured graph's structure: per node in creation order its type, and for a
/// kernel node the function handle, grid, block, dynamic smem and the BYTES of every argument
/// (cuFuncGetParamInfo sizes); then the edge list as (from, to) node indices. Returns
/// (hash, kernel nodes, all arguments hashed). Two arms whose graphs hash equal launch the same
/// kernels with the same geometry and arguments in the same topology: the knob is INERT there.
pub(super) fn graph_hash(graph: sys::CUgraph) -> Result<(u64, usize, bool)> {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |b: &[u8]| { for &x in b { h ^= x as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); } };
    let mut n: usize = 0;
    unsafe {
        let r = sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n);
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuGraphGetNodes {r:?}");
    }
    let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); n];
    if n > 0 {
        let mut n2 = n;
        let r = unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n2) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuGraphGetNodes {r:?}");
        nodes.truncate(n2);
    }
    let (kp, pi) = (kparams_fn(), param_info_fn());
    let mut kernels = 0usize;
    let mut all_args = kp.is_some() && pi.is_some();
    for &nd in &nodes {
        let mut t = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        if unsafe { sys::cuGraphNodeGetType(nd, &mut t) } != sys::CUresult::CUDA_SUCCESS { all_args = false; continue; }
        feed(&(t as u32).to_le_bytes());
        if t != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL { continue; }
        kernels += 1;
        let Some(get) = kp else { continue };
        let mut p: KParamsV2 = unsafe { std::mem::zeroed() };
        if unsafe { get(nd, &mut p) } != sys::CUresult::CUDA_SUCCESS { all_args = false; continue; }
        feed(&(p.func as usize as u64).to_le_bytes());
        for d in [p.gx, p.gy, p.gz, p.bx, p.by, p.bz, p.smem] { feed(&d.to_le_bytes()); }
        // the node priority (WP27 prio raises it): an attribute, not a kernel parameter
        let mut av: sys::CUkernelNodeAttrValue = unsafe { std::mem::zeroed() };
        if unsafe { sys::cuGraphKernelNodeGetAttribute(nd, sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PRIORITY, &mut av) }
            == sys::CUresult::CUDA_SUCCESS {
            feed(&unsafe { av.priority }.to_le_bytes());
        } else {
            all_args = false;
        }
        match (pi, p.params.is_null(), p.func.is_null()) {
            (Some(info), false, false) => {
                for i in 0..256usize {
                    let (mut off, mut size) = (0usize, 0usize);
                    if unsafe { info(p.func, i, &mut off, &mut size) } != sys::CUresult::CUDA_SUCCESS { break; }
                    let a = unsafe { *p.params.add(i) } as *const u8;
                    if a.is_null() || size == 0 || size > 4096 { all_args = false; break; }
                    feed(unsafe { std::slice::from_raw_parts(a, size) });
                }
            }
            _ => all_args = false,
        }
    }
    let mut e: usize = 0;
    if unsafe { sys::cuGraphGetEdges(graph, std::ptr::null_mut(), std::ptr::null_mut(), &mut e) } != sys::CUresult::CUDA_SUCCESS {
        all_args = false; // e.g. programmatic (PDL) edges: LOSSY_QUERY — the topology is unknown
    } else if e > 0 {
        let (mut from, mut to) = (vec![std::ptr::null_mut(); e], vec![std::ptr::null_mut(); e]);
        let mut e2 = e;
        if unsafe { sys::cuGraphGetEdges(graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut e2) } != sys::CUresult::CUDA_SUCCESS {
            all_args = false;
        } else {
            let idx: HashMap<usize, usize> = nodes.iter().enumerate().map(|(i, &nd)| (nd as usize, i)).collect();
            let mut pairs: Vec<(usize, usize)> = (0..e2.min(e))
                .map(|j| (idx.get(&(from[j] as usize)).copied().unwrap_or(usize::MAX),
                          idx.get(&(to[j] as usize)).copied().unwrap_or(usize::MAX)))
                .collect();
            pairs.sort_unstable();
            for (a, b) in pairs { feed(&(a as u64).to_le_bytes()); feed(&(b as u64).to_le_bytes()); }
        }
    }
    Ok((if all_args { h } else { 0 }, kernels, all_args))
}

/// K/V rows p..p+n-1 of `slot` in a K1 row-strided cache ([W][K|V][nkv][max_pos] rows of rb bytes).
pub(super) fn kv_rows(model: &FwdModel, base: u64, slot: usize, p: usize, n: usize, out: &mut Vec<Reg>) -> Result<()> {
    let cfg = &model.cfg;
    let rb = kv_rowbytes(model.kv_fmt, cfg.head_dim);
    let nkv = cfg.num_kv_heads;
    for kv in 0..2usize {
        for kh in 0..nkv {
            let row0 = ((slot * 2 + kv) * nkv + kh) * model.max_pos + p;
            out.push(reg(base + (row0 * rb) as u64, n * rb)?);
        }
    }
    Ok(())
}

/// verify_dry(m') write set at position p (C, F-extras) — see the module doc.
pub(super) fn verify_regions(model: &FwdModel, sc: &Scratch, m: usize, p: usize, slot: usize) -> Result<(Vec<Reg>, Vec<Reg>)> {
    let cfg = &model.cfg;
    let v = cfg.vocab_size;
    let he = cfg.hc_count.max(1) * cfg.hidden_size;
    anyhow::ensure!(p + m <= model.max_pos, "verify regions past the window");
    let tb = model.tap_base(sc, slot)?;
    let mut c = vec![
        reg(dptr(&sc.logits), m * v * 2)?,
        reg(dptr(&sc.argmax), m * 4)?,
        reg(dptr(&sc.taps_keep) + (tb * 4) as u64, m * he * 4)?,
    ];
    let mut f = vec![reg(dptr(&sc.accept_out), (m + 2).min(sc.accept_out.len()) * 4)?];
    let hdx = cfg.indexer_head_dim;
    let ratio = cfg.indexer_compress_ratio.max(1);
    for (att_i, kc) in model.k_cache.iter().enumerate() {
        kv_rows(model, dptr(kc), slot, p, m, &mut c)?;
        if let Some(kp) = model.qsa_keys_p(att_i) {
            c.push(reg(kp + ((slot * model.max_pos + p) * hdx * 2) as u64, m * hdx * 2)?);
        }
        if let Some(pp) = model.qsa_pool_p(att_i) {
            let nbs = (model.max_pos / ratio).max(1);
            let (b0, b1) = (p / ratio, ((p + m - 1) / ratio).min(nbs - 1));
            if b0 <= b1 {
                f.push(reg(pp + ((slot * nbs + b0) * hdx * 2) as u64, (b1 - b0 + 1) * hdx * 2)?);
            }
        }
    }
    Ok((c, f))
}

/// draft(s) write set at position p (C, F-extras) — see the module doc.
pub(super) fn draft_regions(model: &FwdModel, sc: &Scratch, head: &DraftHead, s: usize, p: usize, slot: usize)
    -> Result<(Vec<Reg>, Vec<Reg>)> {
    let cfg = &model.cfg;
    let he = cfg.hc_count.max(1) * cfg.hidden_size;
    anyhow::ensure!(p + s <= model.max_pos, "draft regions past the window");
    let mut c = vec![reg(dptr(&sc.d_dev), s * 4)?, reg(dptr(&sc.dconf), s * 4)?];
    kv_rows(model, dptr(&head.kv), slot, p, s, &mut c)?;
    let hdx = cfg.indexer_head_dim;
    if let Some(kp) = model.qsa_keys_p(HEAD_ATT) {
        c.push(reg(kp + ((slot * model.max_pos + p) * hdx * 2) as u64, s * hdx * 2)?);
    }
    let vd = model.lm_head_draft.as_ref().map(|q| q.n as usize).unwrap_or(model.lm_head.n as usize);
    let mut f = vec![reg(dptr(&sc.resid), he * 4)?, reg(dptr(&sc.logits), (vd & !1) * 2)?];
    if let Some(pp) = model.qsa_pool_p(HEAD_ATT) {
        let ratio = cfg.indexer_compress_ratio.max(1);
        let nbs = (model.max_pos / ratio).max(1);
        let (b0, b1) = (p / ratio, ((p + s - 1) / ratio).min(nbs - 1));
        if b0 <= b1 {
            f.push(reg(pp + ((slot * nbs + b0) * hdx * 2) as u64, (b1 - b0 + 1) * hdx * 2)?);
        }
    }
    Ok((c, f))
}

/// Zero `slot`'s positional planes over positions [0, n): trunk + head K/V rows, indexer raw keys,
/// pooled blocks (compute-stream memsets), so every self-test pass starts from the same state.
pub(super) fn zero_positional(model: &FwdModel, slot: usize, n: usize) -> Result<()> {
    anyhow::ensure!(!model.vmm_lazy(), "KVPOOL: {} needs --kv-pool off or full (lazy maps rows on demand)", "autotune self-test");
    let l = Launcher { dev: &model.dev, stream: &model.stream };
    let n = n.min(model.max_pos);
    if n == 0 { return Ok(()); }
    let g = |w: usize| ((((w + 255) / 256).max(1)) as u32, 1, 1);
    let cfg = &model.cfg;
    let rw = kv_rowbytes(model.kv_fmt, cfg.head_dim) / 4;
    let mut kvs: Vec<u64> = model.k_cache.iter().map(dptr).collect();
    if let Some(h) = model.mtp.as_ref() { kvs.push(dptr(&h.kv)); }
    for &base in &kvs {
        for kv in 0..2usize {
            for kh in 0..cfg.num_kv_heads {
                let start = (((slot * 2 + kv) * cfg.num_kv_heads + kh) * model.max_pos * rw) as i64;
                xqlaunch!(l, "xq_memset_f32", g(n * rw), (256, 1, 1), 0, (base, start, (n * rw) as i64))?;
            }
        }
    }
    let hdx = cfg.indexer_head_dim;
    if cfg.has_indexer() && hdx % 2 == 0 {
        let ratio = cfg.indexer_compress_ratio.max(1);
        let nbs = (model.max_pos / ratio).max(1);
        let nblk = (n / ratio + 1).min(nbs);
        let hw = hdx / 2;
        let mut keys: Vec<u64> = model.qsa_keys.iter().map(dptr).collect();
        let mut pools: Vec<u64> = model.qsa_pool.iter().map(dptr).collect();
        if let Some(h) = model.mtp.as_ref() {
            if let Some(k) = h.qsa_keys.as_ref() { keys.push(dptr(k)); }
            if let Some(k) = h.qsa_pool.as_ref() { pools.push(dptr(k)); }
        }
        for &base in &keys {
            xqlaunch!(l, "xq_memset_f32", g(n * hw), (256, 1, 1), 0,
                      (base, (slot * model.max_pos * hw) as i64, (n * hw) as i64))?;
        }
        for &base in &pools {
            xqlaunch!(l, "xq_memset_f32", g(nblk * hw), (256, 1, 1), 0,
                      (base, (slot * nbs * hw) as i64, (nblk * hw) as i64))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Self-test driver (§5.7): gb10_inference --autotune --selftest --model-dir <EXL3 pack>
// ---------------------------------------------------------------------------------------------
