//! W4/DENSE (2026-09-27): the dense split-K chains on the W3/LMH streaming design — host side.
//! The device side, the bit-identity contract and the PSK-v1 post-mortem are the "W4/DENSE"
//! header in kernels/exl3_bench.cu.
//!
//! ONE launch (exl3_dks_b{B}_s{NST}) per chain — or per GROUP of same-input chains — replaces
//! exl3_had_suh -> exl3_hmma_gemm_ks[_p] -> exl3_ks_combine -> exl3_had_svh. The four levers:
//!   (a) the body: LMH's cp.async.cg ring (L2 evict_first, NST-1 slots in flight across item
//!       boundaries, decode from smem), 1 CTA/SM x 8 warps, items = the old (tile, K-slab) CTAs;
//!   (b) the split-K fixup: the last arrival per N-tile does the ascending combine + H128 svh;
//!   (c) the input Hadamard (suh) folded into the per-CTA A prologue (pinned WP20 sequence);
//!   (d) same-input chains share a launch: attention q + k|v + indexer qk, the shared-expert
//!       gate|up pair, and the shared-expert down with xq_silu_mul folded into its prologue.
//! Every output is BITWISE the old chain's (XCHECK below, binv section EXL3-DENSE).
//!
//! p5e DEFAULT = OFF (the p3 chain dispatch). Bitwise, but SLOWER IN-ROUND (p5c on .13, 2026-09-27
//! 03:17-05:02 UTC, hashes exact and every XCHECK zero): W4 DENSE costs +3.0..+4.6 ms/round on every
//! class; nsys exl3_dks chains 16.6 vs 12.4 ms per verify. With DENSE off (and MOE off), HC + SMALL +
//! QSA + DHEADP on ran 51.8 / 47.85 / 38.85 ms/round vs p3 53.6 / 48.97 / 39.57 (ansic/python/prose).
//!
//! Switch + knobs (diagnostics; the knobs below only act while the package is ON):
//!   (unset)                   OFF (default): the p3 dispatch — exl3_chain_impl / exl3_chain_pair /
//!                             wp22r1 with the old (N/128, ks) grids (PSK default OFF; --psk=1 opts
//!                             in to PSK v1, --psk-off=1 wins over it), xq_silu_mul + the down chain
//!   --w4dense=1            opt in to the one-launch dense split-K chains
//!   --w4dense-off=1        escape, unchanged; wins over --w4dense=1
//!   --w4dense-fix=0        lever (b) off: the kernel stores partials; exl3_ks_combine + exl3_had_svh run
//!   --w4dense-suh=0        lever (c) off: exl3_had_suh stages xh, the kernel copies it
//!                             (this also turns MULTI and SILU off: they need the fused input)
//!   --w4dense-multi=0      lever (d) off for attention: q, k|v and the indexer launch separately
//!   --w4dense-silu=0       the shared-expert down keeps xq_silu_mul + its own chain launch
//!   --w4dense-nst=4|6|8    ring depth (default 6)
//!   --w4dense-g=<n> | "<sig>=<G>,..."   grid override; sig = the members' KxN joined by '+'
//!                             (e.g. "2560x10240=48,2560x12288+2560x512+2560x512+2560x640=40");
//!                             an override the smem / A-entry budget cannot hold is ignored (logged)
//!   --w4dense-gsat=<n>     the G picker's DRAM-saturation CTA count (default 36)
//!   --w4dense-xcheck=1     eager (--exl3-no-graph=1): every DENSE launch is re-run through the
//!                             old four-kernel chain on twin buffers; partials, y and the tile
//!                             counters are bit-checked (DENSE_XCHECK lines)

use super::*;

/// k16 steps per ring slot (== DKS_KSTEP).
pub(crate) const DKS_KSTEP: usize = 4;
/// Members per launch (== DKS_MAXMEM).
pub(crate) const DKS_MAXMEM: usize = 4;
/// A entries per CTA (== DKS_MAXENT; the kernel traps past it).
pub(crate) const DKS_MAXENT: usize = 24;
/// Job flags (== DKS_F_*).
pub(crate) const DKS_F_SUH: i32 = 1;
pub(crate) const DKS_F_FIX: i32 = 2;
pub(crate) const DKS_F_SILU: i32 = 4;
/// Static smem of the entries (ptxas: 656 B) rounded up: the dynamic budget is the cap minus this.
const DKS_STATIC_SMEM: usize = 1024;

// ---------------------------------------------------------------------------
// knobs
// ---------------------------------------------------------------------------
fn env_is(o: crate::opts::OptId, v: &str) -> bool { crate::opts::var(o).as_deref() == Ok(v) }

/// --w4dense opt-in: set to anything but "", "0" or "off".
fn opt_in() -> bool {
    crate::opts::var(crate::opt!("w4dense")).is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off" | "OFF"))
}
/// --w4dense-off=1 escape (unchanged semantics: exactly "1"). Wins over the opt-in.
fn escape() -> bool { env_is(crate::opt!("w4dense-off"), "1") }

/// Package switch. p5e: DEFAULT OFF (the p3 dispatch); --w4dense=1 opts in, --w4dense-off=1
/// wins over it. TUNE T0c (p5e): w4dense.on and its levers are registry entries per width (every
/// value bitwise), so a full tune re-evaluates the package in-round, per (family, m) — the
/// measurement its standalone estimate lacked; the aliases keep their exact parses and still win.
pub(crate) fn dense_on() -> bool {
    tune::get_cur(&T_W4DENSE_ON) != 0
}
fn fix_on() -> bool {
    tune::get_cur(&T_W4DENSE_FIX) != 0
}
fn suh_on() -> bool {
    tune::get_cur(&T_W4DENSE_SUH) != 0
}
/// Lever (d) for attention (q + k|v + indexer in one launch). Needs the fused input (SUH).
pub(super) fn multi_on() -> bool {
    dense_on() && suh_on() && tune::get_cur(&T_W4DENSE_MULTI) != 0
}
/// The shared-expert down with xq_silu_mul in its prologue. Needs the fused input (SUH).
pub(super) fn silu_on() -> bool {
    dense_on() && suh_on() && tune::get_cur(&T_W4DENSE_SILU) != 0
}
pub(crate) fn dense_nst() -> usize {
    nst_of(tune::get_cur(&T_W4DENSE_NST))
}
fn nst_of(v: i32) -> usize { match v { 4 => 4, 8 => 8, _ => 6 } }
fn gsat() -> usize {
    tune::get_cur(&T_W4DENSE_GSAT).max(1) as usize
}

// ---------------------------------------------------------------------------
// TUNE T0c (p5e): the registry entries (class S, per width; the grid override --w4dense-g stays
// an env-only diagnostic — gsat is its searched proxy through the G picker)
// ---------------------------------------------------------------------------
const DENSE_XCHECK: &str = "--probe-exl3-binv EXL3-DENSE + --w4dense-xcheck (old four-kernel chain, eager)";
const fn dense_def(slot: u16, id: &'static str, domain: &'static [i32], default: i32, env: &'static str,
                   env_parse: fn() -> Option<i32>) -> tune::TunableDef {
    tune::TunableDef {
        slot, id, class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_CHAIN, domain, default,
        valid: tune::valid_any, xcheck: DENSE_XCHECK, env, env_parse, rev: 1, wp: "W4/DENSE",
    }
}
/// env "<name> != "0"" (the levers' old `!env_is(name, "0")`), None when unset.
fn lever_env(o: crate::opts::OptId) -> Option<i32> { tune::env_str(o).map(|v| (v != "0") as i32) }
pub(crate) const T_W4DENSE_ON: tune::TunableDef = dense_def(65, "w4dense.on", &[0, 1], 0, "--w4dense|--w4dense-off",
    // either alias set = the old rule exactly (opt-in unless the escape); neither = the default (off)
    || (tune::env_str(crate::opt!("w4dense")).is_some() || tune::env_str(crate::opt!("w4dense-off")).is_some())
        .then(|| (opt_in() && !escape()) as i32));
pub(crate) const T_W4DENSE_FIX: tune::TunableDef = dense_def(66, "w4dense.fix", &[0, 1], 1, "--w4dense-fix",
    || lever_env(crate::opt!("w4dense-fix")));
pub(crate) const T_W4DENSE_SUH: tune::TunableDef = dense_def(67, "w4dense.suh", &[0, 1], 1, "--w4dense-suh",
    || lever_env(crate::opt!("w4dense-suh")));
pub(crate) const T_W4DENSE_MULTI: tune::TunableDef = dense_def(68, "w4dense.multi", &[0, 1], 1, "--w4dense-multi",
    || lever_env(crate::opt!("w4dense-multi")));
pub(crate) const T_W4DENSE_SILU: tune::TunableDef = dense_def(69, "w4dense.silu", &[0, 1], 1, "--w4dense-silu",
    || lever_env(crate::opt!("w4dense-silu")));
pub(crate) const T_W4DENSE_NST: tune::TunableDef = dense_def(70, "w4dense.nst", &[4, 6, 8], 6, "--w4dense-nst",
    || tune::env_str(crate::opt!("w4dense-nst")).map(|v| match v.as_str() { "4" => 4, "8" => 8, _ => 6 }));
pub(crate) const T_W4DENSE_GSAT: tune::TunableDef = dense_def(71, "w4dense.gsat", &[24, 36, 48], 36, "--w4dense-gsat",
    || tune::env_str(crate::opt!("w4dense-gsat")).map(|v| v.trim().parse::<usize>().ok().filter(|&v| v >= 1).unwrap_or(36)
        .min(i32::MAX as usize) as i32));
fn xcheck_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("w4dense-xcheck")).is_ok_and(|v| !v.is_empty() && v != "0"))
}
/// --w4dense-g parse: (global G, per-signature [(sig, G)]). Malformed entries are ignored (once).
fn g_env() -> &'static (Option<usize>, Vec<(String, usize)>) {
    static V: std::sync::OnceLock<(Option<usize>, Vec<(String, usize)>)> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let Ok(raw) = crate::opts::var(crate::opt!("w4dense-g")) else { return (None, Vec::new()) };
        if let Ok(g) = raw.trim().parse::<usize>() { return (Some(g.max(1)), Vec::new()); }
        let mut v = Vec::new();
        for ent in raw.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            match ent.split_once('=').and_then(|(s, g)| g.trim().parse::<usize>().ok().map(|g| (s.trim().to_string(), g.max(1)))) {
                Some(e) => v.push(e),
                None => println!("W4/DENSE: ignoring malformed --w4dense-g entry {ent:?} (want <sig>=<G>)"),
            }
        }
        (None, v)
    })
}

// ---------------------------------------------------------------------------
// the launch parameter (mirror of DksMem / DksJob, kernels/exl3_bench.cu)
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub(crate) struct DksMemH {
    pub tr: u64,
    pub suh: u64,
    pub svh: u64,
    pub xh: u64,
    pub ws: u64,
    pub y: u64,
    pub cnt: u64,
    pub n: i32,
    pub ks: i32,
    pub off: i32,
    pub rsv: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub(crate) struct DksJobH {
    pub mb: [DksMemH; DKS_MAXMEM],
    pub x: u64,
    pub x2: u64,
    pub nmem: i32,
    pub m: i32,
    pub k: i32,
    pub n_items: i32,
    pub flags: i32,
    pub rsv: i32,
}
const _: () = assert!(std::mem::size_of::<DksMemH>() == 72, "DksMemH must mirror DksMem (72 B)");
const _: () = assert!(std::mem::size_of::<DksJobH>() == 328, "DksJobH must mirror DksJob (328 B)");

// ---------------------------------------------------------------------------
// schedule mirror (pure; unit-tested below)
// ---------------------------------------------------------------------------
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let t = a % b; a = b; b = t; }
    a
}

/// A member's shape is eligible for the kernel (its tripwires): N and K multiples of 128, ks > 1
/// divides K/16 and the slab (K/16/ks k16 steps) is a whole number of ring slots.
pub(crate) fn dks_member_ok(k: usize, n: usize, ks: usize) -> bool {
    let kb = k / 16;
    k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 && ks > 1 && kb % ks == 0 && (kb / ks) % DKS_KSTEP == 0
}

/// Per-launch schedule summary over every CTA of a G-wide grid at width m.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DksPlan {
    pub n_items: usize,
    /// ring slots of the busiest CTA / of all CTAs
    pub max_slots: usize,
    pub total_slots: usize,
    /// A-table halves and entries of the most demanding CTA
    pub max_a_halves: usize,
    pub max_entries: usize,
}

/// Mirror of exl3_dks_body's prologue (s_mt / s_ent) for every CTA: which items a CTA runs
/// (g = cta + i*G), how many ring slots that is, and its A table (one entry per distinct
/// (member, slab); a member's slabs repeat with period ks / gcd(G, ks)). `mem` = (N, ks).
pub(crate) fn dks_plan(mem: &[(usize, usize)], kb: usize, g: usize, m: usize) -> DksPlan {
    let mut offs = Vec::with_capacity(mem.len() + 1);
    let mut acc = 0usize;
    for &(n, ks) in mem {
        offs.push(acc);
        acc += (n / 128) * ks;
    }
    offs.push(acc);
    let mut p = DksPlan { n_items: acc, ..Default::default() };
    if g == 0 { return p; }
    for cta in 0..g.min(acc) {
        let (mut slots, mut a, mut ents) = (0usize, 0usize, 0usize);
        for (j, &(_, ks)) in mem.iter().enumerate() {
            let (off, end) = (offs[j], offs[j + 1]);
            let g0 = off + (cta as i64 - off as i64).rem_euclid(g as i64) as usize;
            let cnt = if g0 < end { (end - 1 - g0) / g + 1 } else { 0 };
            let e = (ks / gcd(g, ks)).min(cnt);
            let w = (kb / ks) * 16;
            slots += cnt * (kb / ks / DKS_KSTEP);
            a += e * ((m * (w + 8) + 7) & !7);
            ents += e;
        }
        p.max_slots = p.max_slots.max(slots);
        p.total_slots += slots;
        p.max_a_halves = p.max_a_halves.max(a);
        p.max_entries = p.max_entries.max(ents);
    }
    p
}

/// Dynamic smem of one launch (== the kernel's NST*STB + A-table bytes).
pub(crate) fn dks_smem_bytes(bits: usize, nst: usize, a_halves: usize) -> usize {
    nst * DKS_KSTEP * 8 * 32 * bits + a_halves * 2
}

/// Largest dynamic smem a launch may take.
fn smem_cap() -> usize { crate::gpu::SMEM_OPTIN_CAP as usize - DKS_STATIC_SMEM }

/// Width class of a launch: the grid is chosen per class (M <= 8: the served verify / draft /
/// re-prime widths; 9..16) so a class's worst-case A table always fits.
pub(crate) fn mclass(m: usize) -> usize { if m <= 8 { 8 } else { 16 } }

/// A plan the kernel can run at every width of class `mcls`.
fn plan_feasible(mem: &[(usize, usize)], kb: usize, bits: usize, nst: usize, g: usize, mcls: usize) -> Option<DksPlan> {
    let p = dks_plan(mem, kb, g, mcls);
    (g >= 1 && p.max_entries <= DKS_MAXENT && dks_smem_bytes(bits, nst, p.max_a_halves) <= smem_cap()).then_some(p)
}

/// The built-in grid rule. Model: a CTA streams at most R bytes/s, DRAM at B, so a launch of G CTAs
/// takes ~ max_slots(G) / min(R, B/G) = max_slots(G) * max(G, B/R) (in slot-times of a B/R-share);
/// B/R = `gsat` CTAs (LMH: 48 CTAs at 5.1 GB/s each were DRAM-bound). Minimise it over G in
/// 1..=min(SMs, items) (1 CTA/SM), ties to the larger G, subject to the smem / A-entry budget of
/// the width class. G never changes a bit of the output (the items, their partition and each
/// item's reduction order are fixed): a pure schedule knob.
pub(crate) fn dks_pick_g(mem: &[(usize, usize)], kb: usize, bits: usize, nst: usize, sms: usize,
                         gsat: usize, mcls: usize) -> Option<(usize, DksPlan)> {
    let items = dks_plan(mem, kb, 1, 1).n_items;
    let mut best: Option<(usize, usize, DksPlan)> = None;
    for g in 1..=sms.min(items) {
        let Some(p) = plan_feasible(mem, kb, bits, nst, g, mcls) else { continue };
        let cost = p.max_slots * g.max(gsat);
        if best.as_ref().map_or(true, |b| cost <= b.0) { best = Some((cost, g, p)); }
    }
    best.map(|(_, g, p)| (g, p))
}

fn sig_of(k: usize, mem: &[(usize, usize)]) -> String {
    mem.iter().map(|&(n, _)| format!("{k}x{n}")).collect::<Vec<_>>().join("+")
}

/// G for one job signature at width m (cached per width class). None = no feasible grid (the
/// old path runs). The ring depth and the picker's saturation count are the current context's.
fn dks_grid(dev: &Arc<CudaDevice>, k: usize, bits: usize, mem: &[(usize, usize)], m: usize) -> Option<usize> {
    dks_grid_at(dev, k, bits, mem, m, dense_nst(), gsat())
}
/// dks_grid for an explicit (ring depth, saturation count) — the load-time prewarm walks every
/// value a lookup can return (TUNE T0c). Cached per (K, bits, nst, m class, gsat, members).
fn dks_grid_at(dev: &Arc<CudaDevice>, k: usize, bits: usize, mem: &[(usize, usize)], m: usize, nst: usize,
               gsat: usize) -> Option<usize> {
    type Key = (usize, usize, usize, usize, usize, Vec<(usize, usize)>);
    static C: std::sync::Mutex<Vec<(Key, Option<usize>)>> = std::sync::Mutex::new(Vec::new());
    let (nst, mcls) = (entry_nst(bits, nst), mclass(m));
    let key: Key = (k, bits, nst, mcls, gsat, mem.to_vec());
    if let Ok(c) = C.lock() {
        if let Some((_, g)) = c.iter().find(|e| e.0 == key) { return *g; }
    }
    let kb = k / 16;
    let sms = lmh_sm_count(dev) as usize;
    let sig = sig_of(k, mem);
    let env = g_env();
    let want = env.1.iter().find(|e| e.0 == sig).map(|e| e.1).or(env.0);
    let g = match want {
        Some(w) => {
            let w = w.min(dks_plan(mem, kb, 1, 1).n_items.max(1));
            if plan_feasible(mem, kb, bits, nst, w, mcls).is_some() { Some(w) } else {
                println!("W4/DENSE: --w4dense-g={w} for {sig} (m <= {mcls}) exceeds the smem / A-entry budget — built-in rule");
                dks_pick_g(mem, kb, bits, nst, sms, gsat, mcls).map(|x| x.0)
            }
        }
        None => dks_pick_g(mem, kb, bits, nst, sms, gsat, mcls).map(|x| x.0),
    };
    if let Ok(mut c) = C.lock() { c.push((key, g)); }
    g
}

// ---------------------------------------------------------------------------
// entries + launch
// ---------------------------------------------------------------------------
fn dks_entry(bits: usize, nst: usize) -> Option<&'static str> {
    Some(match (bits, nst) {
        (5, 4) => "exl3_dks_b5_s4",
        (5, 6) => "exl3_dks_b5_s6",
        (5, 8) => "exl3_dks_b5_s8",
        (4, 4) => "exl3_dks_b4_s4",
        (4, 6) => "exl3_dks_b4_s6",
        (4, 8) => "exl3_dks_b4_s8",
        (3, _) => "exl3_dks_b3_s6",
        _ => return None,
    })
}
fn entry_nst(bits: usize, nst: usize) -> usize { if bits == 3 { 6 } else { nst } }

/// Raw handle of one entry (the xq-raw module), opted in to `smem` dynamic bytes and checked to be
/// resident (>= 1 CTA/SM at 256 threads). Cached; the opt-in is raised when a later call needs
/// more (the load-time prewarm opts every entry in to the model's maximum first).
fn dks_fn(name: &'static str, smem: u32) -> Option<cudarc::driver::sys::CUfunction> {
    use cudarc::driver::sys;
    #[derive(Clone, Copy)]
    struct FnH(sys::CUfunction);
    unsafe impl Send for FnH {}
    unsafe impl Sync for FnH {}
    static C: std::sync::Mutex<Vec<(&'static str, FnH, u32)>> = std::sync::Mutex::new(Vec::new());
    let mut c = C.lock().ok()?;
    if let Some(e) = c.iter().find(|e| e.0 == name) {
        if smem <= e.2 { return Some(e.1 .0); }
    }
    let f = xq_raw_fn(name)?;
    if smem as usize > smem_cap() {
        eprintln!("[w4dense] {name}: {smem} B over the dynamic budget {} B", smem_cap());
        return None;
    }
    let r = unsafe { sys::cuFuncSetAttribute(f,
        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32) };
    if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[w4dense] {name}: smem opt-in {smem} B failed ({r:?})"); return None; }
    let mut nb: i32 = 0;
    let r = unsafe { sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut nb, f, 256, smem as usize) };
    if r != sys::CUresult::CUDA_SUCCESS || nb < 1 {
        eprintln!("[w4dense] {name}: not resident at {smem} B smem (occupancy {nb}, {r:?})");
        return None;
    }
    let (mut regs, mut lmem) = (0i32, 0i32);
    unsafe {
        let _ = sys::cuFuncGetAttribute(&mut regs, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS, f);
        let _ = sys::cuFuncGetAttribute(&mut lmem, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);
    }
    if c.iter().all(|e| e.0 != name) {
        println!("W4/DENSE: {name} ready (dynamic smem {smem} B, {nb} CTA/SM, JIT {regs} regs, local {lmem} B)");
    }
    c.retain(|e| e.0 != name);
    c.push((name, FnH(f), smem));
    Some(f)
}

/// Raw launch of one job on `stream` with G CTAs x 256 threads. Ok(false) = no usable entry at this
/// width (nothing launched). PDL-aware (--exl3-pdl=1), graph-capturable (a plain stream launch;
/// the 328-B parameter is copied into the node at capture).
pub(crate) fn dks_launch(stream: cudarc::driver::sys::CUstream, job: &DksJobH, bits: usize, g: usize,
                         nst: usize) -> Result<bool> {
    use cudarc::driver::sys;
    let nst = entry_nst(bits, nst);
    let Some(name) = dks_entry(bits, nst) else { return Ok(false) };
    let nm = job.nmem.clamp(0, DKS_MAXMEM as i32) as usize;
    let mem: Vec<(usize, usize)> = job.mb[..nm].iter().map(|b| (b.n as usize, b.ks as usize)).collect();
    let p = dks_plan(&mem, job.k as usize / 16, g, job.m as usize);
    if nm == 0 || g == 0 || p.max_entries > DKS_MAXENT { return Ok(false); }
    let smem = dks_smem_bytes(bits, nst, p.max_a_halves);
    let Some(f) = dks_fn(name, smem as u32) else { return Ok(false) };
    let mut jb = *job;
    let mut params: [*mut std::ffi::c_void; 1] = [&mut jb as *mut DksJobH as *mut std::ffi::c_void];
    if pdl_on() {
        unsafe { launch_pss_raw(stream, f, (g as u32, 1, 1), (256, 1, 1), smem as u32, &mut params) }
            .with_context(|| format!("launch {name} (pdl)"))?;
    } else {
        let r = unsafe {
            sys::cuLaunchKernel(f, g as u32, 1, 1, 256, 1, 1, smem as u32, stream,
                                params.as_mut_ptr(), std::ptr::null_mut())
        };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch {name} failed: {r:?}");
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------
/// One member's buffers (device addresses + element capacities). `y` is the chain output; `yraw`
/// (FIX=0 tail) and `xh` (SUH=0 staging) are the chain's own scratch — 0 where the caller has none.
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct DOut {
    pub y: u64,
    pub y_len: usize,
    pub yraw: u64,
    pub yraw_len: usize,
    pub xh: u64,
    pub xh_len: usize,
}

pub(super) fn dptr<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }

/// Build the job for `qs` over input x (m rows) — None when a shape, buffer or grid does not fit
/// (then nothing may be launched: the caller runs the old path). `ws`: one buffer carved in member
/// order, or one per member.
#[allow(clippy::too_many_arguments)]
fn build(l: &Launcher, qs: &[&Quad], outs: &[DOut], x: (u64, usize), x2: (u64, usize), m: usize,
         ws: &[(u64, usize)], flags: i32) -> Option<(DksJobH, usize)> {
    let n_mem = qs.len();
    if n_mem == 0 || n_mem > DKS_MAXMEM || outs.len() != n_mem || m == 0 || m > 16 { return None; }
    if !(ws.len() == 1 || ws.len() == n_mem) { return None; }
    let (k, bits) = (qs[0].k as usize, qs[0].bits as usize);
    if !(3..=5).contains(&bits) || x.0 == 0 || x.1 < m * k { return None; }
    let (suh, fix, silu) = (flags & DKS_F_SUH != 0, flags & DKS_F_FIX != 0, flags & DKS_F_SILU != 0);
    if silu && (!suh || x2.0 == 0 || x2.1 < m * k) { return None; }
    let mut job = DksJobH { x: x.0, x2: if silu { x2.0 } else { 0 }, nmem: n_mem as i32, m: m as i32, k: k as i32,
                            flags, ..Default::default() };
    let mut mem = Vec::with_capacity(n_mem);
    let (mut items, mut carve) = (0usize, 0usize);
    for (j, (q, o)) in qs.iter().zip(outs.iter()).enumerate() {
        let (n, ks) = (q.n as usize, q.ks as usize);
        if q.k as usize != k || q.bits as usize != bits || !dks_member_ok(k, n, ks) { return None; }
        if o.y == 0 || o.y_len < m * n { return None; }
        if !fix && (o.yraw == 0 || o.yraw_len < m * n) { return None; }
        if !suh && (o.xh == 0 || o.xh_len < m * k || outs[..j].iter().any(|p| p.xh == o.xh)) { return None; }
        if fix && q.cnt.len() * 128 < n { return None; }
        let need = ks * m * n;
        let wsp = if ws.len() == 1 {
            if carve + need > ws[0].1 { return None; }
            let p = ws[0].0 + (carve * 4) as u64;
            carve += need;
            p
        } else {
            if need > ws[j].1 { return None; }
            ws[j].0
        };
        job.mb[j] = DksMemH { tr: dptr(&q.tr), suh: dptr(&q.suh), svh: dptr(&q.svh), xh: if suh { 0 } else { o.xh },
                              ws: wsp, y: o.y, cnt: dptr(&q.cnt), n: n as i32, ks: ks as i32, off: items as i32, rsv: 0 };
        items += (n / 128) * ks;
        mem.push((n, ks));
    }
    job.n_items = items as i32;
    let g = dks_grid(l.dev, k, bits, &mem, m)?;
    Some((job, g))
}

/// Run `qs` — chains reading the SAME input x (m <= 16 rows) — as ONE DENSE launch:
/// y_j = had_svh(combine(ks_gemm(had_suh(x)))) of chain j, bitwise. `x2` != 0: the input is
/// f16(silu(x) * x2) (the shared-expert down; xq_silu_mul folded in). Ok(false) = not taken and
/// NOTHING launched (the caller runs its old path).
#[allow(clippy::too_many_arguments)]
pub(super) fn dense_chains(l: &Launcher, qs: &[&Quad], outs: &[DOut], x: (u64, usize), x2: (u64, usize),
                           m: usize, ws: &[(u64, usize)]) -> Result<bool> {
    if !dense_on() { return Ok(false); }
    let mut flags = 0;
    if suh_on() { flags |= DKS_F_SUH; }
    if fix_on() { flags |= DKS_F_FIX; }
    if x2.0 != 0 { flags |= DKS_F_SILU; }
    let Some((job, g)) = build(l, qs, outs, x, x2, m, ws, flags) else { return Ok(false) };
    let k = job.k;
    if flags & DKS_F_SUH == 0 {
        // lever (c) off: exl3_had_suh stages every member's xh (the old chain's first kernel).
        for (q, o) in qs.iter().zip(outs.iter()) {
            xqlaunch!(l, "exl3_had_suh", (m as u32 * (k as u32 / 128), 1, 1), (32, 1, 1), 0, (x.0, &q.suh, o.xh, k))?;
        }
    }
    if !dks_launch(l.stream.stream, &job, q_bits(qs), g, dense_nst())? { return Ok(false); }
    if flags & DKS_F_FIX == 0 {
        // lever (b) off: the old tail, member by member (a shared yraw is safe: stream order).
        for (j, (q, o)) in qs.iter().zip(outs.iter()).enumerate() {
            let mn = (m * q.n as usize) as u32;
            xqlaunch!(l, "exl3_ks_combine", ((mn + 255) / 256, 1, 1), (256, 1, 1), 0,
                     (job.mb[j].ws, o.yraw, m as i32, q.n, q.ks as i32))?;
            xqlaunch!(l, "exl3_had_svh", (m as u32 * (q.n as u32 / 128), 1, 1), (32, 1, 1), 0,
                     (o.yraw, &q.svh, o.y, q.n))?;
        }
    }
    if xcheck_on() { xcheck(l, qs, &job, x, x2, m, g)?; }
    Ok(true)
}

fn q_bits(qs: &[&Quad]) -> usize { qs[0].bits as usize }

// ---------------------------------------------------------------------------
// XCHECK (eager): the old four-kernel chain on twin buffers, bit-compared
// ---------------------------------------------------------------------------
fn dtoh_raw<T: Copy + Default>(p: u64, n: usize) -> Result<Vec<T>> {
    use cudarc::driver::sys;
    let mut v = vec![T::default(); n];
    if n > 0 {
        let r = unsafe { sys::cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut std::ffi::c_void, p, n * std::mem::size_of::<T>()) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "DENSE_XCHECK dtoh failed: {r:?}");
    }
    Ok(v)
}

#[allow(clippy::too_many_arguments)]
fn xcheck(l: &Launcher, qs: &[&Quad], job: &DksJobH, x: (u64, usize), x2: (u64, usize), m: usize, g: usize) -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static BAD: AtomicUsize = AtomicUsize::new(0);
    if stream_capturing(l) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| println!("DENSE_XCHECK: skipped inside graph capture (run eager: --exl3-no-graph=1)"));
        return Ok(());
    }
    let k = job.k as usize;
    l.dev.synchronize()?;
    // the old chain's input: x, or xq_silu_mul(x, x2) (its in-place output, here a twin)
    let xs = l.dev.alloc_zeros::<u16>(m * k)?;
    let x_ref = if job.flags & DKS_F_SILU != 0 {
        xqlaunch!(l, "xq_silu_mul", (((m * k) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0, (&xs, x.0, x2.0, (m * k) as i64))?;
        dptr(&xs)
    } else { x.0 };
    let mut lines = Vec::new();
    let mut fail = false;
    for (j, q) in qs.iter().enumerate() {
        let (n, ks) = (q.n as usize, q.ks as usize);
        let xh_t = l.dev.alloc_zeros::<u16>(m * k)?;
        let ws_t = l.dev.alloc_zeros::<f32>(ks * m * n)?;
        let yr_t = l.dev.alloc_zeros::<u16>(m * n)?;
        let y_t = l.dev.alloc_zeros::<u16>(m * n)?;
        xqlaunch!(l, "exl3_had_suh", (m as u32 * (k as u32 / 128), 1, 1), (32, 1, 1), 0, (x_ref, &q.suh, &xh_t, q.k))?;
        xqlaunch!(l, "exl3_hmma_gemm_ks", ((n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
                 (&q.tr, &xh_t, &ws_t, m as i32, q.k, q.n, q.bits))?;
        xqlaunch!(l, "exl3_ks_combine", (((m * n) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (&ws_t, &yr_t, m as i32, q.n, q.ks as i32))?;
        xqlaunch!(l, "exl3_had_svh", ((m * n / 128) as u32, 1, 1), (32, 1, 1), 0, (&yr_t, &q.svh, &y_t, q.n))?;
        l.dev.synchronize()?;
        let p_new: Vec<f32> = dtoh_raw(job.mb[j].ws, ks * m * n)?;
        let p_old = l.dev.dtoh_sync_copy(&ws_t)?;
        let y_new: Vec<u16> = dtoh_raw(job.mb[j].y, m * n)?;
        let y_old = l.dev.dtoh_sync_copy(&y_t)?;
        let pm = p_new.iter().zip(p_old.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let ym = y_new.iter().zip(y_old.iter()).filter(|(a, b)| a != b).count();
        let live = y_old.iter().filter(|v| **v != 0).count();
        let cnt_nz = if job.flags & DKS_F_FIX != 0 { l.dev.dtoh_sync_copy(&q.cnt)?.iter().filter(|&&c| c != 0).count() } else { 0 };
        if pm > 0 || ym > 0 || cnt_nz > 0 || live == 0 { fail = true; }
        let fy = y_new.iter().zip(y_old.iter()).position(|(a, b)| a != b);
        lines.push(format!("[{j}] K{k}xN{n} ks{ks} partials mism {pm} y mism {ym} first {fy:?} y_nonzero {live} cnt_nonzero {cnt_nz}"));
    }
    let c = N.fetch_add(1, Ordering::SeqCst) + 1;
    if fail { BAD.fetch_add(1, Ordering::SeqCst); }
    if fail || c == 1 || c % 500 == 0 {
        println!("DENSE_XCHECK: checks {c} bad {} | this: m {m} G {g} flags {} members {} | {}",
                 BAD.load(Ordering::SeqCst), job.flags, qs.len(), lines.join(" | "));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// probe entry (--probe-exl3-binv, src/exl3_bench.rs): a job from raw device buffers
// ---------------------------------------------------------------------------
/// One member for the binv probe (raw device addresses; `cnt` zeroed, >= N/128 words).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProbeMem { pub tr: u64, pub suh: u64, pub svh: u64, pub xh: u64, pub ws: u64, pub y: u64, pub cnt: u64,
                             pub n: usize, pub ks: usize }

/// The binv probe's launch: exactly the served kernel + launcher (dks_launch), with an explicit G,
/// ring depth and flags. Ok(false) = the launcher refused (no entry / smem): a probe FAIL.
#[allow(clippy::too_many_arguments)]
pub(crate) fn probe_launch(stream: cudarc::driver::sys::CUstream, mems: &[ProbeMem], x: u64, x2: u64,
                           m: usize, k: usize, bits: usize, g: usize, nst: usize, flags: i32) -> Result<bool> {
    anyhow::ensure!(!mems.is_empty() && mems.len() <= DKS_MAXMEM, "probe: {} members", mems.len());
    let mut job = DksJobH { x, x2, nmem: mems.len() as i32, m: m as i32, k: k as i32, flags, ..Default::default() };
    let mut items = 0usize;
    for (j, p) in mems.iter().enumerate() {
        anyhow::ensure!(dks_member_ok(k, p.n, p.ks), "probe: member K{k}xN{} ks {} not eligible", p.n, p.ks);
        job.mb[j] = DksMemH { tr: p.tr, suh: p.suh, svh: p.svh, xh: p.xh, ws: p.ws, y: p.y, cnt: p.cnt,
                              n: p.n as i32, ks: p.ks as i32, off: items as i32, rsv: 0 };
        items += (p.n / 128) * p.ks;
    }
    job.n_items = items as i32;
    dks_launch(stream, &job, bits, g, nst)
}

/// A (G, ring depth, width) the launcher accepts for this member list (A entries + smem budget) —
/// the probe sweeps only these (the served picker / override never choose another).
pub(crate) fn dks_feasible(mem: &[(usize, usize)], k: usize, bits: usize, nst: usize, g: usize, m: usize) -> bool {
    let p = dks_plan(mem, k / 16, g, m);
    g >= 1 && p.max_entries <= DKS_MAXENT && dks_smem_bytes(bits, entry_nst(bits, nst), p.max_a_halves) <= smem_cap()
}

/// The served grid for a probe member list at width m (the built-in rule / --w4dense-g, as dks_grid).
pub(crate) fn probe_grid(dev: &Arc<CudaDevice>, k: usize, bits: usize, mem: &[(usize, usize)], m: usize) -> Option<usize> {
    dks_grid(dev, k, bits, mem, m)
}

// ---------------------------------------------------------------------------
// load-time plan + prewarm (feature marker)
// ---------------------------------------------------------------------------
/// Log the grid plan of every DENSE job this model runs and opt the entries in to the largest smem
/// any of them needs (outside any graph capture). The feature marker is the first line.
pub(super) fn log_and_prewarm(dev: &Arc<CudaDevice>, layers: &[LayerDev], mtp: Option<&DraftHead>) {
    use std::collections::BTreeSet;
    let on_now = dense_on();
    if !on_now {
        let why = if escape() {
            if opt_in() { "--w4dense-off=1 overrides --w4dense=1" } else { "--w4dense-off=1" }
        } else { "default" };
        let knobs: Vec<String> = [crate::opt!("w4dense-fix"), crate::opt!("w4dense-suh"), crate::opt!("w4dense-multi"), crate::opt!("w4dense-silu"),
                                crate::opt!("w4dense-nst"), crate::opt!("w4dense-g"), crate::opt!("w4dense-gsat"), crate::opt!("w4dense-xcheck")]
            .into_iter().filter(|k| crate::opts::var_os(*k).is_some()).map(|k| k.to_string()).collect();
        println!("W4/DENSE: OFF ({why}) — the p3 chain dispatch serves (old (N/128, ks) grids; PSK v1 only with \
                  --psk=1 — see the EXL3-FWD PSK line); --w4dense=1 = one-launch dense split-K chains{}",
                 if knobs.is_empty() { String::new() } else { format!(" ({} ignored while OFF)", knobs.join("/")) });
        // TUNE T0c: a table value or the --autotune overlay can still turn the package on at some
        // width — then every entry it can take is opted in here (outside any capture), silently
        if !tune::reachable(&T_W4DENSE_ON).contains(&1) { return; }
    }
    let nst = dense_nst();
    if on_now {
        println!("W4/DENSE: dense split-K chains ON (opt-in --w4dense=1) — ring nst {nst}, fixup {}, fused suh {}, \
                  attn multi {}, silu-down {}, gsat {}, xcheck {} (default OFF = the p3 dispatch; --w4dense-off=1 wins)",
                 fix_on() as u8, suh_on() as u8, multi_on() as u8, silu_on() as u8, gsat(), xcheck_on() as u8);
    }
    // the attention groups exist when multi is on now or reachable (lists with them are a superset)
    let multi_any = multi_on() || (tune::reachable(&T_W4DENSE_MULTI).contains(&1)
        && tune::reachable(&T_W4DENSE_SUH).contains(&1));
    // every member list this model launches (singles, pairs, attention groups)
    fn attn_lists<'a>(a: &'a AttnLayer, multi: bool, out: &mut Vec<Vec<&'a Quad>>) {
        for q in [&a.q_proj, &a.k_proj, &a.v_proj, &a.o_proj] { out.push(vec![q]); }
        out.push(vec![&a.k_proj, &a.v_proj]);
        if let Some(ix) = &a.idx { out.push(vec![&ix.qk]); }
        if multi {
            out.push(vec![&a.q_proj, &a.k_proj, &a.v_proj]);
            if let Some(ix) = &a.idx {
                out.push(vec![&a.q_proj, &a.k_proj, &a.v_proj, &ix.qk]);
                out.push(vec![&a.k_proj, &a.v_proj, &ix.qk]);
            }
        }
    }
    let multi_now = multi_on();
    // every member list this model launches at a multi setting
    let lists_at = |multi: bool| -> Vec<Vec<&Quad>> {
        let mut lists: Vec<Vec<&Quad>> = Vec::new();
        for l in layers {
            match &l.mixer {
                Mixer::Gdn(g) => for q in [&g.in_qkv, &g.in_z, &g.out_proj] { lists.push(vec![q]); },
                Mixer::Attn(a) => attn_lists(a, multi, &mut lists),
            }
        }
        if let Some(h) = mtp {
            attn_lists(&h.attn, multi, &mut lists);
            lists.push(vec![&h.fc_hidden]);
            lists.push(vec![&h.fc_embed]);
        }
        for moe in layers.iter().map(|l| &l.moe).chain(mtp.map(|h| &h.moe)) {
            lists.push(vec![&moe.sh_gate, &moe.sh_up]);
            for q in [&moe.sh_gate, &moe.sh_up, &moe.sh_down] { lists.push(vec![q]); }
        }
        lists
    };
    // job signatures: (K, bits, [(N, ks)])
    let jobs_of = |lists: &[Vec<&Quad>]| -> BTreeSet<(usize, usize, Vec<(usize, usize)>)> {
        let mut jobs = BTreeSet::new();
        for qs in lists {
            let (k, bits) = (qs[0].k as usize, qs[0].bits as usize);
            if qs.iter().all(|q| q.k as usize == k && q.bits as usize == bits && dks_member_ok(k, q.n as usize, q.ks as usize)) {
                jobs.insert((k, bits, qs.iter().map(|q| (q.n as usize, q.ks as usize)).collect()));
            }
        }
        jobs
    };
    if on_now {
        // the served plan at the current values (the boot receipt, unchanged)
        let jobs = jobs_of(&lists_at(multi_now));
        let mut need: Vec<(usize, usize)> = Vec::new(); // (bits, max smem)
        for (k, bits, mem) in &jobs {
            let kb = k / 16;
            let sig = sig_of(*k, mem);
            let ne = entry_nst(*bits, nst);
            let mut cells = Vec::new();
            for mcls in [8usize, 16] {
                match dks_grid(dev, *k, *bits, mem, mcls) {
                    Some(g) => {
                        let p = dks_plan(mem, kb, g, mcls);
                        let s = dks_smem_bytes(*bits, ne, p.max_a_halves);
                        let bal = p.total_slots as f64 / (g as f64 * p.max_slots.max(1) as f64);
                        cells.push(format!("m<={mcls}: G {g:>2} ({} slots/CTA max, balance {bal:.3}, smem {s} B, {} A entries)",
                                           p.max_slots, p.max_entries));
                        match need.iter_mut().find(|e| e.0 == *bits) {
                            Some(e) => e.1 = e.1.max(s),
                            None => need.push((*bits, s)),
                        }
                    }
                    None => cells.push(format!("m<={mcls}: no feasible grid (p4c path)")),
                }
            }
            println!("  DENSE b{bits} {sig:<44} ks {:<11} items {:>4} | {}",
                     mem.iter().map(|x| x.1.to_string()).collect::<Vec<_>>().join("+"),
                     dks_plan(mem, kb, 1, 1).n_items, cells.join(" | "));
        }
        for (bits, smem) in need {
            let name = dks_entry(bits, entry_nst(bits, nst));
            if !name.is_some_and(|nm| dks_fn(nm, smem as u32).is_some()) {
                println!("W4/DENSE: no usable entry for bits {bits} nst {nst} at {smem} B — those chains keep the p4c path");
            }
        }
    }
    // TUNE T0c: every (ring depth, saturation count, job) a lookup can select — the default, a table
    // value or the --autotune overlay — is planned and its entry opted in to the largest smem it can
    // need, here, outside any graph capture (a no-op beyond the lines above when nothing else is
    // reachable: the default posture opts in exactly what it did before)
    let nsts: Vec<usize> = {
        let mut v: Vec<usize> = tune::reachable(&T_W4DENSE_NST).into_iter().map(nst_of).collect();
        v.push(nst);
        v.sort_unstable();
        v.dedup();
        v
    };
    let gsats: Vec<usize> = {
        let mut v: Vec<usize> = tune::reachable(&T_W4DENSE_GSAT).into_iter().map(|g| g.max(1) as usize).collect();
        v.push(gsat());
        v.sort_unstable();
        v.dedup();
        v
    };
    let jobs = jobs_of(&lists_at(multi_any));
    let mut need: Vec<(&'static str, usize)> = Vec::new(); // (entry, max smem)
    for &nst_v in &nsts {
        for &gs in &gsats {
            for (k, bits, mem) in &jobs {
                let ne = entry_nst(*bits, nst_v);
                let Some(name) = dks_entry(*bits, ne) else { continue };
                for mcls in [8usize, 16] {
                    if let Some(g) = dks_grid_at(dev, *k, *bits, mem, mcls, nst_v, gs) {
                        let s = dks_smem_bytes(*bits, ne, dks_plan(mem, k / 16, g, mcls).max_a_halves);
                        match need.iter_mut().find(|e| e.0 == name) {
                            Some(e) => e.1 = e.1.max(s),
                            None => need.push((name, s)),
                        }
                    }
                }
            }
        }
    }
    for (name, smem) in need {
        let _ = dks_fn(name, smem as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force walk of the kernel's item sequence (cta + i*G) against dks_plan: every item runs
    /// exactly once, per-CTA slots/entries match, the A entry an item selects holds its slab.
    fn walk(mem: &[(usize, usize)], kb: usize, g: usize, m: usize) -> DksPlan {
        let mut offs = vec![0usize];
        for &(n, ks) in mem { offs.push(offs.last().unwrap() + (n / 128) * ks); }
        let n_items = *offs.last().unwrap();
        let mut seen = vec![0u32; n_items];
        let mut p = DksPlan { n_items, ..Default::default() };
        for cta in 0..g.min(n_items) {
            let mut slots = 0usize;
            // entries this CTA builds: (member, slab) in (member, e) order
            let mut ents: Vec<(usize, usize)> = Vec::new();
            let mut a = 0usize;
            for (j, &(_, ks)) in mem.iter().enumerate() {
                let (off, end) = (offs[j], offs[j + 1]);
                let g0 = off + (cta as i64 - off as i64).rem_euclid(g as i64) as usize;
                let cnt = if g0 < end { (end - 1 - g0) / g + 1 } else { 0 };
                let per = ks / gcd(g, ks);
                for e in 0..per.min(cnt) {
                    ents.push((j, (g0 - off + e * g) % ks));
                    a += (m * ((kb / ks) * 16 + 8) + 7) & !7;
                }
            }
            let mut gi = cta;
            while gi < n_items {
                seen[gi] += 1;
                let j = (0..mem.len()).rev().find(|&j| gi >= offs[j]).unwrap();
                let (off, ks) = (offs[j], mem[j].1);
                let slab = (gi - off) % ks;
                // the kernel's entry pick: ebase_j + ((g - g0_j) / G) % per_j
                let g0 = off + (cta as i64 - off as i64).rem_euclid(g as i64) as usize;
                let per = ks / gcd(g, ks);
                let ebase = ents.iter().position(|e| e.0 == j).unwrap();
                let e = ((gi - g0) / g) % per;
                assert_eq!(ents[ebase + e], (j, slab), "cta {cta} item {gi}: A entry holds the wrong slab");
                slots += kb / ks / DKS_KSTEP;
                gi += g;
            }
            p.max_slots = p.max_slots.max(slots);
            p.total_slots += slots;
            p.max_a_halves = p.max_a_halves.max(a);
            p.max_entries = p.max_entries.max(ents.len());
        }
        assert!(seen.iter().all(|&c| c == 1), "every item exactly once (G {g})");
        p
    }

    #[test]
    fn plan_matches_the_kernel_walk() {
        // the served Qwen3.8-Flash-Next chain shapes (5-bit), singles, pairs and the attention group
        let jobs: Vec<(usize, Vec<(usize, usize)>)> = vec![
            (2560, vec![(10240, 2)]), (2560, vec![(6144, 5)]), (6144, vec![(2560, 8)]), (2560, vec![(12288, 2)]),
            (2560, vec![(640, 40), (640, 40)]), (640, vec![(2560, 10)]), (2560, vec![(512, 40), (512, 40)]),
            (2560, vec![(640, 40)]), (2560, vec![(2560, 10)]),
            (2560, vec![(12288, 2), (512, 40), (512, 40), (640, 40)]),
            (2560, vec![(512, 40), (512, 40), (640, 40)]),
        ];
        for (k, mem) in &jobs {
            for &(n, ks) in mem { assert!(dks_member_ok(*k, n, ks), "K{k} N{n} ks{ks}"); }
            for g in [1usize, 2, 3, 5, 7, 8, 13, 36, 40, 47, 48, 64, 97] {
                for m in [1usize, 5, 8, 16] {
                    assert_eq!(dks_plan(mem, k / 16, g, m), walk(mem, k / 16, g, m), "K{k} {mem:?} G{g} m{m}");
                }
            }
        }
    }

    #[test]
    fn chain_ks_shapes_are_eligible() {
        // every served chain shape's own chain_ks must pass the kernel's slab rule
        for (k, n) in [(2560, 10240), (2560, 6144), (6144, 2560), (2560, 12288), (2560, 640), (640, 2560),
                       (2560, 512), (2560, 2560)] {
            let ks = chain_ks(k as i32, n as i32) as usize;
            assert!(dks_member_ok(k, n, ks), "K{k} N{n} ks{ks}");
        }
    }

    #[test]
    fn picker_fills_and_fits() {
        // the built-in rule on the served jobs (5-bit, nst 6, 48 SMs, gsat 36): (K, members, G at
        // m <= 8, G at m <= 16) — feasible, never above the SM count, perfectly balanced where the
        // item count allows it.
        let cases: Vec<(usize, Vec<(usize, usize)>, usize, usize)> = vec![
            (2560, vec![(10240, 2)], 40, 40),       // gdn in_proj_qkv: 160 items -> 4/CTA
            (2560, vec![(6144, 5)], 48, 40),        // gdn in_proj_z: 240 -> 5/CTA (m <= 16: A of 5 slabs too big)
            (6144, vec![(2560, 8)], 40, 40),        // gdn out / attn o: 160 -> 4/CTA
            (2560, vec![(12288, 2)], 48, 48),       // attn q: 192 -> 4/CTA
            (2560, vec![(640, 40), (640, 40)], 40, 40),  // shared gate|up
            (640, vec![(2560, 10)], 40, 40),        // shared down
            (2560, vec![(512, 40), (512, 40)], 40, 40),  // attn k|v
            (2560, vec![(12288, 2), (512, 40), (512, 40), (640, 40)], 48, 48), // attn q + k|v + indexer (12 A entries, 97 KB at m 16)
            (2560, vec![(512, 40), (512, 40), (640, 40)], 40, 40),             // re-prime k|v + indexer
        ];
        for (k, mem, w8, w16) in &cases {
            for (mcls, want) in [(8usize, *w8), (16, *w16)] {
                let (g, p) = dks_pick_g(mem, k / 16, 5, 6, 48, 36, mcls).expect("feasible");
                assert_eq!(g, want, "K{k} {mem:?} m<={mcls}: G");
                assert!(p.max_entries <= DKS_MAXENT);
                assert!(dks_smem_bytes(5, 6, p.max_a_halves) <= smem_cap(), "K{k} {mem:?}: smem");
            }
        }
        // ring depth 8 still fits every single chain / pair at its own m <= 16 grid
        for (k, mem, _, _) in cases.iter().take(7) {
            assert!(dks_pick_g(mem, k / 16, 5, 8, 48, 36, 16).is_some(), "K{k} {mem:?}: nst 8 infeasible");
        }
    }

    #[test]
    fn wide_same_shape_pairs_have_no_grid_at_m16() {
        // The binv EXL3-DENSE pair cells the probe must SKIP (p5c .13 run: 'no served grid' bail):
        // two K2560 x N10240 (ks 2) or N12288 (ks 2) members at widths 9..16 need >= one 41,216-B A
        // entry per member (m 16: 16 x (1,280 + 8) halves) + the 30,720-B nst-6 ring = 113,152 B >
        // the 100,352-B budget for every G, so dks_grid is None and the served pair falls back to
        // the old x2 chain. At widths <= 8 the same pairs are placeable.
        for n in [10240usize, 12288] {
            let mem = [(n, 2usize), (n, 2usize)];
            assert!(dks_pick_g(&mem, 2560 / 16, 5, 6, 48, 36, 16).is_none(), "N{n} pair m<=16: expected no grid");
            assert!(dks_pick_g(&mem, 2560 / 16, 5, 6, 48, 36, 8).is_some(), "N{n} pair m<=8: expected a grid");
            let p = dks_plan(&mem, 2560 / 16, 48, 16);
            assert!(p.max_a_halves * 2 >= 2 * 41_216 && dks_smem_bytes(5, 6, p.max_a_halves) > smem_cap());
        }
    }

    #[test]
    fn job_layout_offsets() {
        // DksJob field offsets (kernels/exl3_bench.cu): mb[i] at 72*i, x at 288, nmem at 304, flags at 320
        let j = DksJobH::default();
        let base = &j as *const DksJobH as usize;
        assert_eq!(&j.mb[1] as *const DksMemH as usize - base, 72);
        assert_eq!(&j.mb[0].n as *const i32 as usize - base, 56);
        assert_eq!(&j.x as *const u64 as usize - base, 288);
        assert_eq!(&j.nmem as *const i32 as usize - base, 304);
        assert_eq!(&j.flags as *const i32 as usize - base, 320);
    }
}
