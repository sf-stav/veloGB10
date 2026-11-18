//! Tuning context and the frozen process table: `Ctx`, the thread-local scope guard, `Key`, `get` (resolution order: override flag, tuner overlay,
//! frozen table, default) and the `--autotune` overlay. See `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------------------------
// Context + thread-local family scope
// ---------------------------------------------------------------------------------------------

/// Launch context of a lookup. `m` = the family's width (verify width, chunk rows, 1 for a step or
/// draft pass); regime 0 = dense, 1 = sparse (QSA); shape (k, n, bits) set by the chain helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ctx {
    pub fam: Fam,
    pub m: u16,
    pub regime: u8,
    pub k: i32,
    pub n: i32,
    pub bits: u8,
    pub chunk_class: u8,
}

impl Ctx {
    pub const UTIL: Ctx = Ctx { fam: Fam::Util, m: 0, regime: 0, k: 0, n: 0, bits: 0, chunk_class: 0 };
    pub fn fam(f: Fam, m: usize) -> Ctx { Ctx { fam: f, m: m.min(u16::MAX as usize) as u16, ..Ctx::UTIL } }
    pub fn prefill(c: usize) -> Ctx { Ctx { chunk_class: chunk_class(c), ..Ctx::fam(Fam::Prefill, c) } }
    pub fn regime(self, sparse: bool) -> Ctx { Ctx { regime: sparse as u8, ..self } }
    pub fn shape(self, k: i32, n: i32, bits: i32) -> Ctx { Ctx { k, n, bits: bits.clamp(0, 255) as u8, ..self } }
}

/// Key::C class of a Chunk-scope lookup outside every prefill scope (no table decision names it).
pub const NO_CHUNK_CLASS: u8 = u8::MAX;

/// Prefill chunk-size classes: <=32, <=64, <=128, <=256, <=512, <=1024, larger.
pub fn chunk_class(c: usize) -> u8 {
    match c { 0..=32 => 0, 33..=64 => 1, 65..=128 => 2, 129..=256 => 3, 257..=512 => 4, 513..=1024 => 5, _ => 6 }
}

thread_local! {
    static CUR: RefCell<Vec<Ctx>> = RefCell::new(Vec::with_capacity(8));
}

/// RAII family scope: launches (and knob lookups) inside run with this context. Not Send.
pub struct ScopeGuard(std::marker::PhantomData<*const ()>);

/// Enter a family scope for the life of the returned guard (bind it: `let _tc = tune::enter(..)`).
pub fn enter(cx: Ctx) -> ScopeGuard {
    CUR.with(|c| c.borrow_mut().push(cx));
    ScopeGuard(std::marker::PhantomData)
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        CUR.with(|c| { c.borrow_mut().pop(); });
    }
}

/// The innermost active family scope (Ctx::UTIL outside every scope).
pub fn current() -> Ctx {
    CUR.with(|c| c.borrow().last().copied().unwrap_or(Ctx::UTIL))
}

// ---------------------------------------------------------------------------------------------
// Frozen state + lookups
// ---------------------------------------------------------------------------------------------

/// Scope key of a table value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Key {
    G,
    F(u8),
    S(i32, i32, u8),
    W(u8, u16, u8),
    C(u8, u8),
}

pub fn key_of(scope: Scope, cx: &Ctx) -> Key {
    match scope {
        Scope::Global | Scope::Load => Key::G,
        Scope::Fam => Key::F(cx.fam as u8),
        Scope::Shape => Key::S(cx.k, cx.n, cx.bits),
        Scope::Width => Key::W(cx.fam as u8, cx.m, cx.regime),
        // T0c: a Chunk knob read outside a prefill scope (a head fill, a Util launch) never picks up
        // a chunk-class decision (the default applies): only Prefill keys name a class
        Scope::Chunk => Key::C(if cx.fam == Fam::Prefill { cx.chunk_class } else { NO_CHUNK_CLASS }, cx.regime),
    }
}

pub(super) struct Entry {
    pub(super) env: Option<i32>,
    /// resolved value when the table holds no decision for this entry
    pub(super) fast: Option<i32>,
    pub(super) dec: HashMap<Key, i32>,
}

/// How the process's table was resolved (the boot line).
#[derive(Clone, Debug)]
pub struct BootStatus {
    /// "OK" | "OFF" | "FALLBACK" | "DEFAULTS"
    pub state: &'static str,
    pub reason: String,
    pub table_path: Option<String>,
    pub table_sha8: Option<String>,
    pub decisions_s: usize,
    pub decisions_d: usize,
    pub ignored: Vec<String>,
    pub profile_mhz: u32,
    pub tp: u32,
}

pub(super) struct Frozen {
    pub(super) entries: Vec<Entry>,
    pub(super) status: BootStatus,
}

pub(super) static FROZEN: OnceLock<Frozen> = OnceLock::new();
static EARLY: AtomicU64 = AtomicU64::new(0);
static ENV: OnceLock<Vec<Option<i32>>> = OnceLock::new();
static CONSULT: [[AtomicU64; NFAM]; REGISTRY.len()] =
    [const { [const { AtomicU64::new(0) }; NFAM] }; REGISTRY.len()];

pub(super) fn env_cache() -> &'static [Option<i32>] {
    ENV.get_or_init(|| REGISTRY.iter().map(|t| (t.env_parse)()).collect())
}

/// The value of `t` in context `cx` (O(1); see the module doc for the resolution order).
#[inline]
pub fn get(t: &TunableDef, cx: &Ctx) -> i32 {
    let s = t.slot as usize;
    if TUNER_ARMED.load(Ordering::Relaxed) {
        if let Some(row) = CONSULT.get(s) {
            row[cx.fam as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
    let Some(f) = FROZEN.get() else {
        EARLY.fetch_add(1, Ordering::Relaxed);
        return env_cache().get(s).copied().flatten().unwrap_or(t.default);
    };
    let Some(e) = f.entries.get(s) else { return t.default; };
    if let Some(v) = e.env { return v; }
    if OVERLAY_ON.load(Ordering::Relaxed) {
        if let Some(v) = overlay_lookup(t, cx) { return v; }
    }
    if let Some(v) = e.fast { return v; }
    e.dec.get(&key_of(t.scope, cx)).copied().unwrap_or(t.default)
}

/// `get` in the current thread-local family scope.
#[inline]
pub fn get_cur(t: &TunableDef) -> i32 { get(t, &current()) }

/// Lookups that ran before the freeze (a mis-ordered boot reads defaults silently otherwise).
pub fn early_lookups() -> u64 { EARLY.load(Ordering::Relaxed) }

/// Per-family lookup counts of an entry since process start (§4.8a inertness accounting).
pub fn consulted(t: &TunableDef) -> [u64; NFAM] {
    let mut out = [0u64; NFAM];
    if let Some(row) = CONSULT.get(t.slot as usize) {
        for (i, c) in row.iter().enumerate() { out[i] = c.load(Ordering::Relaxed); }
    }
    out
}

pub fn is_frozen() -> bool { FROZEN.get().is_some() }

/// Every value a lookup of `t` can return in this process: the env alias alone when it is set;
/// else the default, the frozen table's decisions, and (under `--autotune`) the whole domain. Load
/// code uses it to pre-resolve per-value resources (smem opt-ins, raw handles) outside any capture.
pub fn reachable(t: &TunableDef) -> Vec<i32> {
    if let Some(v) = env_cache().get(t.slot as usize).copied().flatten() { return vec![v]; }
    let mut v = vec![t.default];
    if let Some(e) = FROZEN.get().and_then(|f| f.entries.get(t.slot as usize)) {
        v.extend(e.dec.values().copied());
    }
    if TUNER_ARMED.load(Ordering::SeqCst) { v.extend(t.domain.iter().copied()); }
    v.sort_unstable();
    v.dedup();
    v
}
pub fn status() -> Option<BootStatus> { FROZEN.get().map(|f| f.status.clone()) }

/// Active env overrides: (entry id, env alias, value).
pub fn env_overrides() -> Vec<(&'static str, &'static str, i32)> {
    let env = env_cache();
    REGISTRY.iter().zip(env.iter())
        .filter_map(|(t, v)| v.map(|v| (t.id, t.env, v)))
        .collect()
}

// ---- tuner overlay (armed only by the --autotune harness) ----

static OVERLAY_ON: AtomicBool = AtomicBool::new(false);
static TUNER_ARMED: AtomicBool = AtomicBool::new(false);
thread_local! {
    static OVERLAY: RefCell<Vec<(u16, Option<Key>, i32)>> = RefCell::new(Vec::new());
}

fn overlay_lookup(t: &TunableDef, cx: &Ctx) -> Option<i32> {
    OVERLAY.with(|o| {
        let o = o.borrow();
        if o.is_empty() { return None; }
        let k = key_of(t.scope, cx);
        o.iter().rev().find(|(s, key, _)| *s == t.slot && key.map_or(true, |kk| kk == k)).map(|x| x.2)
    })
}

/// Arm the overlay machinery (the `--autotune` entry point only; serving never calls it).
pub fn arm_tuner() { TUNER_ARMED.store(true, Ordering::SeqCst); }

/// A candidate assignment in force for the life of the guard (thread-local; the tuner sets it only
/// while it runs / captures one candidate unit). `key = None` applies in every context.
pub struct OverlayGuard(std::marker::PhantomData<*const ()>);

pub fn overlay(assign: &[(&TunableDef, Option<Key>, i32)]) -> anyhow::Result<OverlayGuard> {
    anyhow::ensure!(TUNER_ARMED.load(Ordering::SeqCst), "tune overlay used outside --autotune");
    for (t, _, v) in assign {
        anyhow::ensure!(domain_has(t.domain, *v), "overlay {}={} outside its domain {:?}", t.id, v, t.domain);
        anyhow::ensure!(!matches!(t.class, Class::N), "overlay: {} is N-class (numerics) — never tuned without consent", t.id);
    }
    OVERLAY.with(|o| {
        let mut o = o.borrow_mut();
        o.clear();
        o.extend(assign.iter().map(|(t, k, v)| (t.slot, *k, *v)));
    });
    OVERLAY_ON.store(true, Ordering::SeqCst);
    Ok(OverlayGuard(std::marker::PhantomData))
}

impl Drop for OverlayGuard {
    fn drop(&mut self) {
        OVERLAY.with(|o| o.borrow_mut().clear());
        OVERLAY_ON.store(false, Ordering::SeqCst);
    }
}
