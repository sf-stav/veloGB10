//! TUNE T1 (PLAN/AUTOTUNE_DESIGN.md §5): the in-round measurement core of `gb10_inference --autotune`
//! and its self-test (§5.7); TUNE T2 (§6-§8): the search, the final gate and the table writer (the
//! `Tuner`, at the end of this file). A child module of `exl3_forward` (declared after the launch
//! macros), so it drives the SERVED units — verify_shadow / verify_kernels / draft_block /
//! forward_step_kernels — without widening their visibility.
//!
//! # Units and the idempotence contract (§5.1)
//!
//! | unit          | launches (one graph per arm)                                       | mode |
//! |---------------|--------------------------------------------------------------------|------|
//! | `verify_dry`  | verify_shadow -> verify_kernels(m') -> taps snapshot -> xq_accept   | SR   |
//! | `draft(s)`    | draft_block(s) (the tap restore runs eagerly before each replay)   | SR   |
//! | `step`        | forward_step_kernels(m = 1)                                        | LI   |
//!
//! Declared write sets (the digest covers every row a replay can leave behind that the canonical
//! round could read; `C` = comparable with the canonical execution at a different width, `F` = the
//! full set, compared arm against arm at the same width):
//! * verify_dry(m') at position p, slot s — C: logits rows < m', argmax rows < m', taps_keep rows < m',
//!   trunk K/V rows p..p+m'-1 of every attention layer (K and V, every kv-head), trunk indexer raw-key
//!   rows p..p+m'-1; F adds accept_out[0..m'+2) and the pooled indexer blocks the rows touch (a
//!   partial block reads stale keys past the rows, so it is NOT comparable across widths).
//!   Also written (scratch, rewritten before read by the canonical verify): resid/x/qkv/chain
//!   buffers, gdn_ring rows < m', qkv_save/qkv_post/a_save/b_save rows < m', conv/PLE shadows, acc2.
//! * draft(s) at position p — C: d_dev[0..s), dconf[0..s), head K/V rows p..p+s-1, head indexer raw
//!   keys p..p+s-1; F adds the head stream (resid row 0), the last pass's draft logits row, the head
//!   pooled blocks. Also written: toks/pos/slots row 0 (re-staged by xq_draft_setup every pass).
//! * step — LI only (it advances the live GDN/conv state; never replayed).
//!
//! Canonical-last: the canonical (served) execution runs AFTER every replay of its round and its write
//! set covers every replay's (fixed chain: canonical verify m = k+1 >= m', canonical chain k >= s), so
//! a replay can never leave a trace in the trajectory — the §5.7 test 3 checks it empirically (greedy
//! tokens AND [mtp-stats] rounds / drafted / accepted identical with SR on and off).
//!
//! # SR round (§5.2, protocol v2/v3 — see the §5.2 and §5.7 amendments), hooked into the served `mtp_round` / `mtp_verify`
//! 1. `sr_before_draft` (after the tap restore + draft_meta upload): every arm's graph is ensured
//!    FIRST (a capture syncs and idles the device, so it never lands inside the timed block); then
//!    the untimed WARM-UP replay(s) of the base (tap restore, graph, digests — the same kernel
//!    sequence as a timed arm, so position 0's predecessor is a replay like everyone else's); then
//!    per arm in the round's BALANCED order — tap restore, L2 conditioning, event A, the arm's draft
//!    graph, event B, untimed digests; then the tap restore again, so the canonical chain reads
//!    exactly the served tap.
//! 2. the canonical draft chain; `sr_after_draft` digests its C set.
//! 3. the served verify uploads; `sr_before_verify`: graphs first, the base warm-up replay(s), then
//!    per arm at this round's m' (one width per round, rotating) in the balanced order — L2
//!    conditioning, event A, the arm's verify_dry graph, event B, digests; event on the canonical start.
//! 4. the canonical verify (+ commit); `sr_after_verify`: canonical end event + its C digest.
//! Events and digests are read at the next round's first hook (the served sync retired them).
//! Pairing: the base arm runs in every round, so every candidate sample has a base sample on
//! identical inputs (same routing, context and drafts); Δ = t_cand − t_base is paired.
//!
//! Why v2 (the .14 self-test, 2026-09-27, receipts selftest-1790470336 / -1790470533): the FIRST
//! verify replay of every round was ~0.6-1.1 ms slower than the next (median µs by position, m = 6:
//! 39607 / 38803 / 38598), the draft replays showed nothing, clocks steady. The L2 conditioning runs
//! before EVERY timed arm, so it is not what differs at position 0; what differs is the predecessor
//! of the whole block — the verify SR follows the canonical draft chain + the served host sync /
//! readback / PLE build / uploads (the device idles), while the draft SR follows the heavy canonical
//! verify. So the warm-up goes at the START of the block, before the first arm's conditioning (each
//! timed arm keeps its own conditioning right before event A). The v1 per-round random order did
//! not balance first positions (base first 49x vs base2 35x) and biased the paired HL; v2 walks a
//! Williams design per analysis cell (every arm in every position equally often, carry-over
//! balanced), records each replay's position (`apos`), and the analysis can remove a fitted
//! position effect (`tune::position_offsets`, `--autotune-stratify`, default on).
//!
//! # LI (§5.3)
//! `li_step`, hooked into the served `forward_step`: each plain step runs the graph of the arm the
//! balanced (Williams) schedule picks (bitwise arms: the trajectory is the same whichever runs),
//! timed with events around the graph launch — the served conditions, unpaired. The first step(s)
//! of every LI segment run the base as an untimed warm-up.
//!
//! Nothing here runs unless the `--autotune` entry point armed it (`armed()` is false in serving).

use super::*;
use super::tune::{self, Ctx, Fam};
use anyhow::{bail, Result};
use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, DevicePtr};
use std::cell::RefCell;
use std::collections::HashMap;

use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------------------------
// Arming + the thread-local session
// ---------------------------------------------------------------------------------------------

static ARMED: AtomicBool = AtomicBool::new(false);

/// True only while the `--autotune` harness has a session installed on this process.
#[inline]
pub(crate) fn armed() -> bool { ARMED.load(Ordering::Relaxed) }

thread_local! {
    static SESSION: RefCell<Option<Session>> = RefCell::new(None);
}

fn with_session<T>(f: impl FnOnce(&mut Session) -> Result<T>) -> Result<Option<T>> {
    SESSION.with(|c| match c.borrow_mut().as_mut() {
        Some(s) => f(s).map(Some),
        None => Ok(None),
    })
}

/// The served round's own values at a hook point.
pub(crate) struct SrAt {
    pub slot: usize,
    pub k: usize,
    /// the canonical verify width (k + 1)
    pub m: usize,
    pub p: usize,
    pub tap_row: usize,
    pub qsa: bool,
    pub bucket: usize,
}

pub(crate) fn sr_before_draft(model: &FwdModel, sc: &mut Scratch, head: &DraftHead, at: &SrAt) -> Result<()> {
    with_session(|s| s.before_draft(model, sc, head, at)).map(|_| ())
}
pub(crate) fn sr_after_draft(model: &FwdModel, sc: &mut Scratch, _head: &DraftHead, at: &SrAt) -> Result<()> {
    with_session(|s| s.after_draft(model, sc, at)).map(|_| ())
}
pub(crate) fn sr_before_verify(model: &FwdModel, sc: &mut Scratch, at: &SrAt) -> Result<()> {
    with_session(|s| s.before_verify(model, sc, at)).map(|_| ())
}
pub(crate) fn sr_after_verify(model: &FwdModel, sc: &mut Scratch, at: &SrAt) -> Result<()> {
    with_session(|s| s.after_verify(model, sc, at)).map(|_| ())
}
/// LI plain step: Ok(true) = the harness launched this step's unit (the caller skips its own).
pub(crate) fn li_step(model: &FwdModel, sc: &mut Scratch, m: usize, qsa: bool, bucket: usize) -> Result<bool> {
    Ok(with_session(|s| s.li_step(model, sc, m, qsa, bucket))?.unwrap_or(false))
}

// ---------------------------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------------------------

/// A candidate configuration: the registry assignment its graphs are captured under (T2: keyed —
/// `None` applies in every context the unit reads, `Some(key)` only there, e.g. one shape).
#[derive(Clone)]
pub(crate) struct Arm {
    pub name: String,
    pub assign: Vec<(&'static tune::TunableDef, Option<tune::Key>, i32)>,
}

impl Arm {
    pub(crate) fn base() -> Arm { Arm { name: "base".into(), assign: Vec::new() } }
    /// Identity of the configuration (graphs are cached per signature, so a base shared by
    /// successive race calls is captured once).
    pub(crate) fn sig(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut feed = |x: u64| { for b in x.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); } };
        for (t, k, v) in &self.assign {
            feed(t.slot as u64);
            feed(match k { None => u64::MAX, Some(k) => {
                let mut kh = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(k, &mut kh);
                std::hash::Hasher::finish(&kh)
            } });
            feed(*v as i64 as u64);
        }
        h
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum UnitKey {
    Verify { m: usize, qsa: bool, bucket: usize, slot: usize, pen: bool },
    Draft { s: usize, qsa: bool, bucket: usize, slot: usize, pen: bool },
    Step { m: usize, qsa: bool, bucket: usize, pen: bool },
}

#[derive(Clone)]
pub(crate) struct SrCfg {
    pub verify_on: bool,
    pub draft_on: bool,
    pub li_on: bool,
    /// the replay widths m', one per round, rotating (each must be <= the canonical width)
    pub verify_widths: Vec<usize>,
    /// draft replay passes s (<= the canonical chain k)
    pub draft_s: usize,
    /// tuner graph memory cap (the design's 512 MB, measured by cuMemGetInfo)
    pub mem_cap: usize,
}

/// One timing sample (a replay, a canonical execution or an LI step).
#[derive(Clone, Debug)]
pub(crate) struct Sample {
    pub unit: &'static str,
    /// the arm set generation (Session::set_arms); samples pair only within one phase
    pub phase: usize,
    pub m: usize,
    /// arm index; usize::MAX = the canonical (served) execution; WARMUP_ARM = an untimed warm-up
    pub arm: usize,
    pub prompt: usize,
    pub round: usize,
    /// the token position p of the round (not the arm's place in the block: that is `apos`)
    pub pos: usize,
    pub regime: u8,
    pub ms: f64,
    pub digest_ok: bool,
    /// "sr" (a timed replay), "warm" (an untimed warm-up replay / LI warm-up step), "li"
    pub mode: &'static str,
    /// v2: the replay's place in its round's timed block (0 = the first timed arm, after the
    /// warm-up); LI: the place in the schedule block. NO_APOS for the warm-up and the canonical.
    pub apos: u8,
    /// v2: the number of timed replays in that block (the position model's cell size)
    pub npos: u8,
}

/// `Sample::arm` of an untimed warm-up replay of the base (never paired, never in a race).
pub(crate) const WARMUP_ARM: usize = usize::MAX - 1;
/// `Sample::apos` of a sample that has no place in a timed block (warm-up, canonical).
pub(crate) const NO_APOS: u8 = u8::MAX;
/// Upper bound of `--autotune-warmup` (base replays before the timed arms of every block).
const MAX_WARMUP: usize = 4;
/// v3 (2026-09-27): the default `--autotune-warmup` — 2 (was 1). The .13 self-tests at 1 left a
/// first-position offset of +0.25 / +0.12 ms at verify m = 3 / 6 (receipts selftest-1790485556 /
/// -1790485758); at 2 the fitted offsets were <= 6 us in every cell (selftest-1790486005).
const DEFAULT_WARMUP: usize = 2;

struct Pending {
    unit: &'static str,
    m: usize,
    arm: usize,
    ev: (sys::CUevent, sys::CUevent),
    /// digest slots: (C, F); canonical C slot for the unit
    dslots: Option<(usize, usize, usize)>,
    pos: usize,
    regime: u8,
    apos: u8,
    npos: u8,
    /// the warm-up C digest accumulates over `wmul` identical replays (xq_digest64 adds mod 2^64):
    /// it must equal wmul x the canonical's C digest
    wmul: u64,
}

const DIG_SLOTS: usize = 64;
const DRAFT_WARM: (usize, usize) = (28, 29);
const DRAFT_CANON: usize = 30;
const VERIFY_WARM: (usize, usize) = (60, 61);
const VERIFY_CANON: usize = 62;
const MAX_ARMS: usize = 14;
const REG_CAP: usize = 2048; // regions (2 u64 each)
const VOFF: usize = 1024;    // verify region-table offset

/// The analysis cell of a replay for the balanced order: unit (0 draft, 1 verify) x width x regime.
fn order_cell(unit: u64, m: usize, regime: bool) -> u64 { (unit << 40) | ((m as u64) << 8) | regime as u64 }

/// One step of a round's SR block (§5.2 v2): the base's untimed warm-up(s), then every runnable arm
/// once, in the round's balanced order, with its place in the timed block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Warm,
    Timed { arm: usize, apos: u8 },
}

/// The block of one round: `run` = the balanced row filtered to the arms whose graph exists (the
/// base, arm 0, is always attempted first-class; no base graph = no warm-up and no pairing anyway).
fn block_plan(run: &[usize], warmup: usize) -> Vec<Step> {
    let mut v: Vec<Step> = Vec::with_capacity(warmup + run.len());
    if run.contains(&0) { v.extend(std::iter::repeat(Step::Warm).take(warmup)); }
    v.extend(run.iter().enumerate().map(|(i, &a)| Step::Timed { arm: a, apos: i as u8 }));
    v
}

pub(crate) struct Session {
    cfg: SrCfg,
    /// the model's blocking compute stream (every tuner graph is instantiated on and replayed into it)
    stream: sys::CUstream,
    arms: Vec<Arm>,
    /// tuner graphs per (unit, arm signature) — never the serving map (§4.4)
    graphs: HashMap<(UnitKey, u64), crate::gpu::CudaGraph>,
    /// §4.8b node-parameter hash of each tuner graph (function, geometry, smem, argument bytes, edges)
    ghash: HashMap<(UnitKey, u64), u64>,
    /// §4.8a registry slots consulted while capturing a BASE graph, per unit kind ("verify"/"draft"/"step")
    pub consulted_by: HashMap<u16, std::collections::BTreeSet<&'static str>>,
    /// arm-set generation (bumped by set_arms)
    pub phase: usize,
    events: Vec<sys::CUevent>,
    ev_next: usize,
    li_events: [sys::CUevent; 2],
    dig: CudaSlice<u64>,
    regs: CudaSlice<u64>,
    sink: CudaSlice<u32>,
    /// v2: the balanced arm order, one cursor per analysis cell (unit x width x regime x arms)
    order: tune::BalancedOrder,
    /// v2: untimed base replays at the start of every SR block (and LI segment), 0..=MAX_WARMUP
    pub warmup: usize,
    round: usize,
    round_open: bool,
    vw_next: usize,
    pending: Vec<Pending>,
    draft_canon: Option<usize>,
    verify_canon: Option<usize>,
    canon_ev: Option<sys::CUevent>,
    /// (arm, step index, regime, apos, npos) of the LI step whose events are still unread
    li_pending: Option<(usize, usize, u8, u8, u8)>,
    li_sched: LiSched,
    /// warm-up steps left in the current LI segment (re-armed by finish / set_arms)
    li_warm_left: usize,
    pub samples: Vec<Sample>,
    pub xfail: Vec<String>,
    pub canon_checks: (usize, usize),
    pub arm_checks: (usize, usize),
    mem0: usize,
    mem_cap_hit: bool,
    pub prompt: usize,
    pub captures: usize,
}

impl Session {
    pub(crate) fn new(model: &FwdModel, cfg: SrCfg, arms: Vec<Arm>, seed: u64) -> Result<Session> {
        anyhow::ensure!(!arms.is_empty() && arms.len() <= MAX_ARMS, "tune session: 1..={MAX_ARMS} arms");
        // arm 0 is the base every other arm pairs against (the served defaults, or under T2 the
        // defaults + the decisions adopted so far — bitwise either way, so its digest must still
        // equal the canonical execution's)
        model.dev.bind_to_thread()?;
        let mut events = Vec::new();
        // per round: 2 per draft arm + 2 per verify arm + 2 per warm-up of each unit + the canonical pair
        for _ in 0..(4 * MAX_ARMS + 4 * MAX_WARMUP + 8) { events.push(ev_create()?); }
        let li_events = [ev_create()?, ev_create()?];
        let mem0 = cudarc::driver::result::mem_get_info().map(|x| x.0).unwrap_or(0);
        let n = arms.len();
        Ok(Session {
            cfg,
            stream: model.stream.stream,
            arms,
            graphs: HashMap::new(),
            ghash: HashMap::new(),
            consulted_by: HashMap::new(),
            phase: 0,
            events,
            ev_next: 0,
            li_events,
            // real zeros (AGENTS §2.2: alloc_zeros does not zero)
            dig: model.dev.htod_sync_copy(&vec![0u64; DIG_SLOTS])?,
            regs: model.dev.htod_sync_copy(&vec![0u64; 2 * REG_CAP])?,
            sink: model.dev.htod_sync_copy(&[0u32; 4])?,
            order: tune::BalancedOrder::new(seed),
            warmup: DEFAULT_WARMUP,
            round: 0,
            round_open: false,
            vw_next: 0,
            pending: Vec::new(),
            draft_canon: None,
            verify_canon: None,
            canon_ev: None,
            li_pending: None,
            li_sched: LiSched::new(n, seed),
            li_warm_left: 1,
            samples: Vec::new(),
            xfail: Vec::new(),
            canon_checks: (0, 0),
            arm_checks: (0, 0),
            mem0,
            mem_cap_hit: false,
            prompt: 0,
            captures: 0,
        })
    }

    /// Replace the arm set between rounds (T2 race stages). Graphs stay cached by signature.
    pub(crate) fn set_arms(&mut self, arms: Vec<Arm>) -> Result<()> {
        anyhow::ensure!(!self.round_open && self.pending.is_empty(), "tune session: set_arms inside a round");
        anyhow::ensure!(!arms.is_empty() && arms.len() <= MAX_ARMS, "tune session: 1..={MAX_ARMS} arms");
        // the SR order cursors persist (per cell and arm count); the LI schedule restarts with the set
        self.li_sched = LiSched::new(arms.len(), self.phase as u64 + 1);
        self.li_warm_left = self.warmup;
        self.arms = arms;
        self.phase += 1;
        Ok(())
    }

    /// v2: set the warm-up count (base replays before the timed arms of every block).
    pub(crate) fn set_warmup(&mut self, w: usize) -> Result<()> {
        anyhow::ensure!(w <= MAX_WARMUP, "--autotune-warmup must be 0..={MAX_WARMUP}");
        self.warmup = w;
        self.li_warm_left = w;
        Ok(())
    }

    /// Destroy every tuner graph (a family is done) and re-arm the memory cap.
    pub(crate) fn clear_graphs(&mut self) {
        self.graphs.clear();
        self.ghash.clear();
        self.mem0 = cudarc::driver::result::mem_get_info().map(|x| x.0).unwrap_or(self.mem0);
        self.mem_cap_hit = false;
    }

    /// Node hashes of arm `a`'s graphs vs the base's, over every unit both captured this far:
    /// Some(true) = identical everywhere (INERT), Some(false) = a difference, None = nothing to compare.
    pub(crate) fn arm_inert(&self, a: usize) -> Option<bool> {
        let (sb, sa) = (self.arms[0].sig(), self.arms.get(a)?.sig());
        let mut any = false;
        for ((k, s), h) in self.ghash.iter() {
            if *s != sa || *h == 0 { continue; } // 0 = not hashable (argument bytes unreadable)
            if let Some(hb) = self.ghash.get(&(*k, sb)).filter(|x| **x != 0) {
                any = true;
                if hb != h { return Some(false); }
            }
        }
        if any { Some(true) } else { None }
    }

    /// Untimed digest of `regs` right now (outside any round): regions at table offset 1800, slot 63.
    pub(crate) fn digest_now(&mut self, model: &FwdModel, regs: &[Reg]) -> Result<u64> {
        const OFF: usize = 1800;
        anyhow::ensure!(!self.round_open, "digest_now inside a round");
        self.upload_regions(model, OFF, regs)?;
        let l = Launcher { dev: &model.dev, stream: &model.stream };
        xqlaunch!(l, "xq_memset_f32", (1, 1, 1), (2, 1, 1), 0, (*self.dig.device_ptr() as u64, 126i64, 2i64))?;
        self.digest(&l, OFF, regs, 63)?;
        model.dev.synchronize()?;
        Ok(model.dev.dtoh_sync_copy(&self.dig)?[63])
    }

    fn ev(&mut self) -> Result<sys::CUevent> {
        anyhow::ensure!(self.ev_next < self.events.len(), "tune session: event pool exhausted");
        let e = self.events[self.ev_next];
        self.ev_next += 1;
        Ok(e)
    }

    /// First hook of a round: read the previous round's events + digests, zero the digest slots.
    fn open_round(&mut self, model: &FwdModel) -> Result<()> {
        if self.round_open { return Ok(()); }
        self.resolve(model)?;
        let l = Launcher { dev: &model.dev, stream: &model.stream };
        xqlaunch!(l, "xq_memset_f32", (1, 1, 1), (2 * DIG_SLOTS as u32, 1, 1), 0,
                  (*self.dig.device_ptr() as u64, 0i64, (2 * DIG_SLOTS) as i64))?;
        self.round += 1;
        self.round_open = true;
        self.ev_next = 0;
        Ok(())
    }

    /// Read every pending sample's events and digests (the served sync already retired them).
    fn resolve(&mut self, model: &FwdModel) -> Result<()> {
        if self.pending.is_empty() { return Ok(()); }
        let dig = model.dev.dtoh_sync_copy(&self.dig)?;
        let pend = std::mem::take(&mut self.pending);
        // base C/F digests of this round per unit (arm 0)
        let base_of = |unit: &str| pend.iter().find(|p| p.unit == unit && p.arm == 0).and_then(|p| p.dslots)
            .map(|(c, f, _)| (dig[c], dig[f]));
        for p in &pend {
            let ms = ev_elapsed_ms(p.ev.0, p.ev.1)?;
            let mut ok = true;
            if let Some((c, f, canon)) = p.dslots {
                if p.arm == WARMUP_ARM {
                    // the warm-up is the base config: its C set must equal the canonical's (x replays)
                    let _ = f;
                    self.canon_checks.0 += 1;
                    if dig[c] != dig[canon].wrapping_mul(p.wmul) {
                        self.canon_checks.1 += 1;
                        ok = false;
                        self.xfail.push(format!(
                            "TUNE_XCHECK FAIL: base {} warm-up replay != canonical (m {} round {} pos {} prompt {}): \
                             C {:016x} vs {} x {:016x}",
                            p.unit, p.m, self.round, p.pos, self.prompt, dig[c], p.wmul, dig[canon]));
                    }
                } else if p.arm == 0 {
                    self.canon_checks.0 += 1;
                    if dig[c] != dig[canon] {
                        self.canon_checks.1 += 1;
                        ok = false;
                        self.xfail.push(format!(
                            "TUNE_XCHECK FAIL: base {} replay != canonical (m {} round {} pos {} prompt {}): C {:016x} vs {:016x}",
                            p.unit, p.m, self.round, p.pos, self.prompt, dig[c], dig[canon]));
                    }
                } else if let Some((bc, bf)) = base_of(p.unit) {
                    self.arm_checks.0 += 1;
                    if dig[c] != bc || dig[f] != bf {
                        self.arm_checks.1 += 1;
                        ok = false;
                        let a = &self.arms[p.arm];
                        let asg: Vec<String> = a.assign.iter().map(|(t, k, v)| match k {
                            None => format!("{}={}", t.id, v),
                            Some(k) => format!("{}{:?}={}", t.id, k, v),
                        }).collect();
                        self.xfail.push(format!(
                            "TUNE_XCHECK FAIL: {} arm '{}' [{}] differs from base (m {} round {} pos {} prompt {}): \
                             C {:016x} vs {:016x}, F {:016x} vs {:016x}",
                            p.unit, a.name, asg.join(","), p.m, self.round, p.pos, self.prompt, dig[c], bc, dig[f], bf));
                    }
                }
            }
            let mode = if p.arm == WARMUP_ARM { "warm" } else { "sr" };
            self.samples.push(Sample { unit: p.unit, phase: self.phase, m: p.m, arm: p.arm, prompt: self.prompt,
                                       round: self.round, pos: p.pos, regime: p.regime, ms, digest_ok: ok, mode,
                                       apos: p.apos, npos: p.npos });
        }
        Ok(())
    }

    /// Drain everything (after the pass's last round): device sync, then resolve. The next LI
    /// segment starts with its warm-up again.
    pub(crate) fn finish(&mut self, model: &FwdModel) -> Result<()> {
        model.dev.synchronize()?;
        self.resolve(model)?;
        self.resolve_li()?;
        self.round_open = false;
        self.li_warm_left = self.warmup;
        Ok(())
    }

    fn upload_regions(&mut self, model: &FwdModel, off: usize, regs: &[Reg]) -> Result<()> {
        anyhow::ensure!(off + regs.len() <= REG_CAP, "tune: region table overflow ({} + {})", off, regs.len());
        let mut v = Vec::with_capacity(2 * regs.len());
        for r in regs { v.push(r.ptr); v.push(r.words); }
        let mut dst = self.regs.slice_mut(2 * off..2 * (off + regs.len()));
        model.dev.htod_sync_copy_into(&v, &mut dst)?;
        Ok(())
    }

    /// Untimed digest of regions [off, off + n) of the table into slot.
    fn digest(&self, l: &Launcher, off: usize, regs: &[Reg], slot: usize) -> Result<()> {
        if regs.is_empty() { return Ok(()); }
        let maxw = regs.iter().map(|r| r.words).max().unwrap_or(0).max(1);
        let gx = ((maxw + 255) / 256).clamp(1, 128) as u32;
        xqlaunch!(l, "xq_digest64", (gx, regs.len() as u32, 1), (256, 1, 1), 0,
                  (&self.regs, off as i32, &self.dig, slot as i32))
    }

    fn l2_touch(&self, l: &Launcher, ptr: u64, bytes: usize, min_ns: i64) -> Result<()> {
        let n16 = (bytes / 16) as i64;
        if n16 == 0 && min_ns == 0 { return Ok(()); }
        anyhow::ensure!(ptr % 16 == 0, "l2_touch: unaligned region");
        let grid = ((n16 + 1023) / 1024).clamp(1, 384) as u32;
        xqlaunch!(l, "xq_l2_touch", (grid, 1, 1), (256, 1, 1), 0,
                  (ptr, n16, &self.sink, 0i32, min_ns))
    }

    /// §5.4: before a verify replay, the tail of the pruned draft lm_head (the last draft pass's
    /// stream) — else the full head's tail.
    fn condition_verify(&self, l: &Launcher, model: &FwdModel) -> Result<()> {
        let q = model.lm_head_draft.as_ref().unwrap_or(&model.lm_head);
        let bytes = q.tr.len() * 2;
        let tail = bytes.min(24 << 20);
        let off = (bytes - tail) & !15;
        self.l2_touch(l, *q.tr.device_ptr() as u64 + off as u64, bytes - off, 60_000)
    }

    /// §5.4: before a draft replay, the previous round's re-prime reads (head entry + K/V projections).
    fn condition_draft(&self, l: &Launcher, head: &DraftHead) -> Result<()> {
        let qs = [&head.fc_hidden, &head.fc_embed, &head.attn.k_proj, &head.attn.v_proj];
        for (i, q) in qs.iter().enumerate() {
            let bytes = (q.tr.len() * 2).min(24 << 20) & !15;
            self.l2_touch(l, *q.tr.device_ptr() as u64, bytes, if i + 1 == qs.len() { 60_000 } else { 0 })?;
        }
        Ok(())
    }

    /// The arm's graph for this unit: eager warm run (SR units only — idempotent under
    /// canonical-last) under the arm's overlay, then capture + instantiate. Err(stop) past the cap.
    fn ensure_graph(&mut self, model: &FwdModel, sc: &mut Scratch, head: Option<&DraftHead>,
                    key: UnitKey, arm: usize, tap_row: usize) -> Result<bool> {
        let sig = self.arms[arm].sig();
        if self.graphs.contains_key(&(key, sig)) { return Ok(true); }
        if self.mem_cap_hit { return Ok(false); }
        let assign: Vec<(&tune::TunableDef, Option<tune::Key>, i32)> = self.arms[arm].assign.clone();
        let _ov = if assign.is_empty() { None } else { Some(tune::overlay(&assign)?) };
        // §4.8a: which registry entries does this unit consult? (base captures only)
        let before: Option<Vec<u64>> = (arm == 0)
            .then(|| tune::REGISTRY.iter().map(|t| tune::consulted(t).iter().sum()).collect());
        let l = Launcher { dev: &model.dev, stream: &model.stream };
        if !matches!(key, UnitKey::Step { .. }) {
            if let UnitKey::Draft { slot, .. } = key { tap_restore(&l, model, sc, slot, tap_row)?; }
            run_unit(model, &l, sc, head, key)?;
            model.dev.synchronize()?;
        }
        let raw = model.stream.stream;
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(raw, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "tune capture: BeginCapture {r:?}");
        }
        let body = run_unit(model, &l, sc, head, key);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r2 = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
        if body.is_err() && !graph.is_null() { unsafe { sys::cuGraphDestroy(graph); } }
        body.context("tune capture: unit body")?;
        anyhow::ensure!(r2 == sys::CUresult::CUDA_SUCCESS, "tune capture: EndCapture {r2:?}");
        if let Some(b) = before {
            let unit = match key { UnitKey::Verify { .. } => "verify", UnitKey::Draft { .. } => "draft", UnitKey::Step { .. } => "step" };
            for (i, t) in tune::REGISTRY.iter().enumerate() {
                let now: u64 = tune::consulted(t).iter().sum();
                if now > b[i] { self.consulted_by.entry(t.slot).or_default().insert(unit); }
            }
        }
        // the served capture path exactly: WP27 node priorities (when on), then its instantiate
        let mm = match key { UnitKey::Verify { m, .. } => m, UnitKey::Draft { s, .. } => s, UnitKey::Step { m, .. } => m };
        crate::exl3_wp27::finish(graph, "tune", mm);
        // §4.8b: hash the instantiated-to-be graph's kernel nodes before it is consumed
        let gh = graph_hash(graph).map(|x| x.0).unwrap_or(0);
        let gh = if gh == 0 { 0 } else { gh ^ ((crate::exl3_wp27::prio_now() as u64) << 63 | 0x5157) };
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r3 = unsafe {
            let r3 = crate::exl3_wp27::instantiate(&mut exec, graph);
            sys::cuGraphDestroy(graph);
            r3
        };
        anyhow::ensure!(r3 == sys::CUresult::CUDA_SUCCESS, "tune capture: instantiate {r3:?}");
        // upload now (no execution) so the first TIMED launch does not carry the graph upload
        let r4 = unsafe { sys::cuGraphUpload(exec, raw) };
        let g = crate::gpu::CudaGraph::from_exec(exec, raw);
        anyhow::ensure!(r4 == sys::CUresult::CUDA_SUCCESS, "tune capture: cuGraphUpload {r4:?}");
        self.graphs.insert((key, sig), g);
        self.ghash.insert((key, sig), gh);
        self.captures += 1;
        let free = cudarc::driver::result::mem_get_info().map(|x| x.0).unwrap_or(self.mem0);
        if self.mem0.saturating_sub(free) > self.cfg.mem_cap {
            self.mem_cap_hit = true;
            println!("TUNE: tuner graph memory cap reached ({} MB) — no further candidate captures",
                     self.cfg.mem_cap >> 20);
        }
        Ok(true)
    }

    fn launch_timed(&mut self, key: UnitKey, arm: usize) -> Result<(sys::CUevent, sys::CUevent)> {
        let (a, b) = (self.ev()?, self.ev()?);
        let sig = self.arms[arm].sig();
        let g = self.graphs.get(&(key, sig)).context("tune: arm graph missing")?;
        ev_record(a, self.stream)?;
        g.launch();
        ev_record(b, self.stream)?;
        Ok((a, b))
    }

    fn before_draft(&mut self, model: &FwdModel, sc: &mut Scratch, head: &DraftHead, at: &SrAt) -> Result<()> {
        self.open_round(model)?;
        self.draft_canon = None;
        let s = self.cfg.draft_s.min(at.k);
        if !self.cfg.draft_on || at.tap_row == usize::MAX || s == 0 || at.p + s > model.max_pos { return Ok(()); }
        let (c, f) = draft_regions(model, sc, head, s, at.p, at.slot)?;
        let mut all = c.clone();
        all.extend(f.iter().cloned());
        self.upload_regions(model, 0, &all)?;
        let key = UnitKey::Draft { s, qsa: at.qsa, bucket: at.bucket, slot: at.slot, pen: sc.pen_live };
        let l = Launcher { dev: &model.dev, stream: &model.stream };
        // v2: the round's balanced order for this cell; every graph BEFORE the block (a capture
        // syncs + idles the device — never between the warm-up and a timed arm)
        let row = self.order.next(order_cell(0, s, at.qsa), self.arms.len());
        let mut run: Vec<usize> = Vec::with_capacity(row.len());
        for a in row {
            if self.ensure_graph(model, sc, Some(head), key, a, at.tap_row)? { run.push(a); }
        }
        let plan = block_plan(&run, self.warmup);
        let nwarm = plan.iter().filter(|x| **x == Step::Warm).count();
        let npos = run.len() as u8;
        let mut w = 0usize;
        for st in plan {
            tap_restore(&l, model, sc, at.slot, at.tap_row)?;
            match st {
                Step::Warm => {
                    // untimed for the analysis (its events are kept only for the position report);
                    // the same digest kernels as a timed arm follow it, so the first timed arm's
                    // predecessor is a replay + digests exactly like every other position's
                    w += 1;
                    let ev = self.launch_timed(key, 0)?;
                    self.digest(&l, 0, &c, DRAFT_WARM.0)?;
                    self.digest(&l, 0, &all, DRAFT_WARM.1)?;
                    let last = w == nwarm;
                    self.pending.push(Pending { unit: "draft", m: s, arm: WARMUP_ARM, ev,
                                                dslots: last.then_some((DRAFT_WARM.0, DRAFT_WARM.1, DRAFT_CANON)),
                                                pos: at.p, regime: at.qsa as u8, apos: NO_APOS, npos, wmul: nwarm as u64 });
                }
                Step::Timed { arm: a, apos } => {
                    self.condition_draft(&l, head)?;
                    let ev = self.launch_timed(key, a)?;
                    self.digest(&l, 0, &c, 2 * a)?;
                    self.digest(&l, 0, &all, 2 * a + 1)?;
                    self.pending.push(Pending { unit: "draft", m: s, arm: a, ev, dslots: Some((2 * a, 2 * a + 1, DRAFT_CANON)),
                                                pos: at.p, regime: at.qsa as u8, apos, npos, wmul: 1 });
                }
            }
        }
        // the canonical chain reads exactly the served tap
        tap_restore(&l, model, sc, at.slot, at.tap_row)?;
        self.draft_canon = Some(c.len());
        Ok(())
    }

    fn after_draft(&mut self, model: &FwdModel, _sc: &mut Scratch, _at: &SrAt) -> Result<()> {
        if let Some(nc) = self.draft_canon.take() {
            let l = Launcher { dev: &model.dev, stream: &model.stream };
            self.digest_range(&l, 0, nc, DRAFT_CANON)?;
        }
        Ok(())
    }

    /// Digest the first `n` regions at table offset `off` (the canonical's C set).
    fn digest_range(&self, l: &Launcher, off: usize, n: usize, slot: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        // the region sizes are on the device; bound the grid by the largest region class (logits)
        xqlaunch!(l, "xq_digest64", (128u32, n as u32, 1), (256, 1, 1), 0,
                  (&self.regs, off as i32, &self.dig, slot as i32))
    }

    fn before_verify(&mut self, model: &FwdModel, sc: &mut Scratch, at: &SrAt) -> Result<()> {
        self.open_round(model)?;
        self.verify_canon = None;
        self.canon_ev = None;
        if self.cfg.verify_on && !self.cfg.verify_widths.is_empty() {
            let mp = self.cfg.verify_widths[self.vw_next % self.cfg.verify_widths.len()];
            self.vw_next += 1;
            if mp >= 2 && mp <= at.m && at.p + mp <= model.max_pos {
                let (c, f) = verify_regions(model, sc, mp, at.p, at.slot)?;
                let mut all = c.clone();
                all.extend(f.iter().cloned());
                self.upload_regions(model, VOFF, &all)?;
                let key = UnitKey::Verify { m: mp, qsa: at.qsa, bucket: at.bucket, slot: at.slot, pen: sc.pen_live };
                let l = Launcher { dev: &model.dev, stream: &model.stream };
                // v2: balanced order per (width, regime) cell; graphs first; the base warm-up replay(s)
                // absorb the block's cold start (the device idled through the served host sync /
                // readback / PLE build / uploads since the draft phase); then every arm, each with
                // its own L2 conditioning right before event A
                let row = self.order.next(order_cell(1, mp, at.qsa), self.arms.len());
                let mut run: Vec<usize> = Vec::with_capacity(row.len());
                for a in row {
                    if self.ensure_graph(model, sc, None, key, a, 0)? { run.push(a); }
                }
                let plan = block_plan(&run, self.warmup);
                let nwarm = plan.iter().filter(|x| **x == Step::Warm).count();
                let npos = run.len() as u8;
                let mut w = 0usize;
                for st in plan {
                    match st {
                        Step::Warm => {
                            w += 1;
                            let ev = self.launch_timed(key, 0)?;
                            self.digest(&l, VOFF, &c, VERIFY_WARM.0)?;
                            self.digest(&l, VOFF, &all, VERIFY_WARM.1)?;
                            let last = w == nwarm;
                            self.pending.push(Pending { unit: "verify_dry", m: mp, arm: WARMUP_ARM, ev,
                                                        dslots: last.then_some((VERIFY_WARM.0, VERIFY_WARM.1, VERIFY_CANON)),
                                                        pos: at.p, regime: at.qsa as u8, apos: NO_APOS, npos,
                                                        wmul: nwarm as u64 });
                        }
                        Step::Timed { arm: a, apos } => {
                            self.condition_verify(&l, model)?;
                            let ev = self.launch_timed(key, a)?;
                            self.digest(&l, VOFF, &c, 32 + 2 * a)?;
                            self.digest(&l, VOFF, &all, 33 + 2 * a)?;
                            self.pending.push(Pending { unit: "verify_dry", m: mp, arm: a, ev,
                                                        dslots: Some((32 + 2 * a, 33 + 2 * a, VERIFY_CANON)),
                                                        pos: at.p, regime: at.qsa as u8, apos, npos, wmul: 1 });
                        }
                    }
                }
                self.verify_canon = Some(c.len());
            }
        }
        // the canonical verify's own device time (served conditions; informational)
        let e = self.ev()?;
        ev_record(e, model.stream.stream)?;
        self.canon_ev = Some(e);
        Ok(())
    }

    fn after_verify(&mut self, model: &FwdModel, _sc: &mut Scratch, at: &SrAt) -> Result<()> {
        let l = Launcher { dev: &model.dev, stream: &model.stream };
        if let Some(a) = self.canon_ev.take() {
            let b = self.ev()?;
            ev_record(b, model.stream.stream)?;
            self.pending.push(Pending { unit: "verify", m: at.m, arm: usize::MAX, ev: (a, b), dslots: None,
                                        pos: at.p, regime: at.qsa as u8, apos: NO_APOS, npos: 0, wmul: 1 });
        }
        if let Some(nc) = self.verify_canon.take() {
            self.digest_range(&l, VOFF, nc, VERIFY_CANON)?;
        }
        self.round_open = false;
        Ok(())
    }

    fn resolve_li(&mut self) -> Result<()> {
        if let Some((arm, round, regime, apos, npos)) = self.li_pending.take() {
            let ms = ev_elapsed_ms(self.li_events[0], self.li_events[1])?;
            let mode = if arm == WARMUP_ARM { "warm" } else { "li" };
            self.samples.push(Sample { unit: "step", phase: self.phase, m: 1, arm, prompt: self.prompt, round,
                                       pos: 0, regime, ms, digest_ok: true, mode, apos, npos });
        }
        Ok(())
    }

    fn li_step(&mut self, model: &FwdModel, sc: &mut Scratch, m: usize, qsa: bool, bucket: usize) -> Result<bool> {
        if !self.cfg.li_on || m != 1 { return Ok(false); }
        self.resolve_li()?; // forward_step synchronized after the previous step
        // v2: the segment's first step(s) run the base as an untimed warm-up (the segment follows a
        // prefill + seam: a cold start no arm should pay); then the balanced (Williams) schedule
        let (arm, tag, apos) = if self.li_warm_left > 0 {
            self.li_warm_left -= 1;
            (0usize, WARMUP_ARM, NO_APOS)
        } else {
            let (a, p) = self.li_sched.next();
            (a, a, p)
        };
        let key = UnitKey::Step { m, qsa, bucket, pen: sc.pen_live };
        if !self.ensure_graph(model, sc, None, key, arm, 0)? { return Ok(false); }
        let g = self.graphs.get(&(key, self.arms[arm].sig())).context("tune: LI graph missing")?;
        ev_record(self.li_events[0], self.stream)?;
        g.launch();
        ev_record(self.li_events[1], self.stream)?;
        self.round += 1;
        self.li_pending = Some((tag, self.round, qsa as u8, apos, self.arms.len() as u8));
        Ok(true)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.graphs.clear();
        for &e in self.events.iter().chain(self.li_events.iter()) {
            unsafe { sys::cuEventDestroy_v2(e); }
        }
    }
}

fn ev_create() -> Result<sys::CUevent> {
    let mut e: sys::CUevent = std::ptr::null_mut();
    let r = unsafe { sys::cuEventCreate(&mut e, 0) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventCreate {r:?}");
    Ok(e)
}
fn ev_record(e: sys::CUevent, s: sys::CUstream) -> Result<()> {
    let r = unsafe { sys::cuEventRecord(e, s) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventRecord {r:?}");
    Ok(())
}
fn ev_elapsed_ms(a: sys::CUevent, b: sys::CUevent) -> Result<f64> {
    let mut ms: f32 = 0.0;
    let r = unsafe { sys::cuEventElapsedTime(&mut ms, a, b) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventElapsedTime {r:?}");
    Ok(ms as f64)
}

/// Balanced LI schedule (§5.3, v2): consecutive blocks of `arms` steps are the successive rows of a
/// Williams design (tune::williams_row) — every block is a permutation, every arm takes every
/// in-block position equally often and follows every other arm equally often (carry-over balanced)
/// over each period. Deterministic; the start row comes from `seed`. Returns (arm, in-block position).
struct LiSched {
    arms: usize,
    row: Vec<usize>,
    r: usize,
    j: usize,
}
impl LiSched {
    fn new(arms: usize, seed: u64) -> LiSched {
        let arms = arms.max(1);
        LiSched { arms, row: Vec::new(), r: (seed as usize) % tune::williams_period(arms), j: 0 }
    }
    fn next(&mut self) -> (usize, u8) {
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
fn tap_restore(l: &Launcher, model: &FwdModel, sc: &mut Scratch, slot: usize, tap_row: usize) -> Result<()> {
    if tap_row == usize::MAX { return Ok(()); }
    let he = model.cfg.hc_count.max(1) * model.cfg.hidden_size;
    let tb = model.tap_base(sc, slot)?;
    xqlaunch!(l, "xq_copy_f32", ((((he as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
              (&mut sc.resid, &sc.taps_keep, he as i64, 0i64, (tb + tap_row * he) as i64))
}

/// A unit's launches (graph-capturable: launches only). The selftest spin node (§5.7 test 2) closes
/// the unit when the arm's overlay sets it.
fn run_unit(model: &FwdModel, l: &Launcher, sc: &mut Scratch, head: Option<&DraftHead>, key: UnitKey) -> Result<()> {
    let cx = match key {
        UnitKey::Verify { m, slot, qsa, .. } => {
            let _tc = tune::enter(Ctx::fam(Fam::Verify, m).regime(qsa));
            let he = model.cfg.hc_count.max(1) * model.cfg.hidden_size;
            // verify_block minus verify_commit (the live state never moves in a replay)
            model.verify_shadow(l, sc, slot)?;
            model.verify_kernels(l, sc, m, qsa, slot)?;
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
            model.forward_step_kernels(l, sc, m, None, false, qsa)?;
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
    ptr: u64,
    words: u64,
}

fn reg(ptr: u64, bytes: usize) -> Result<Reg> {
    anyhow::ensure!(ptr % 4 == 0, "digest region not 4-byte aligned ({ptr:#x})");
    Ok(Reg { ptr, words: (bytes / 4) as u64 }) // a trailing partial word (never on this model) is skipped
}

fn dptr<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }

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
fn graph_hash(graph: sys::CUgraph) -> Result<(u64, usize, bool)> {
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
fn kv_rows(model: &FwdModel, base: u64, slot: usize, p: usize, n: usize, out: &mut Vec<Reg>) -> Result<()> {
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
fn verify_regions(model: &FwdModel, sc: &Scratch, m: usize, p: usize, slot: usize) -> Result<(Vec<Reg>, Vec<Reg>)> {
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
fn draft_regions(model: &FwdModel, sc: &Scratch, head: &DraftHead, s: usize, p: usize, slot: usize)
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
fn zero_positional(model: &FwdModel, slot: usize, n: usize) -> Result<()> {
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

const CORPUS: [&str; 2] = [
    "Write a complete ANSI C program that reads a text file named on the command line, counts how \
     many times each word occurs using a hash table with separate chaining, and prints the twenty \
     most frequent words with their counts. Include all necessary headers, check every allocation \
     and file operation, and comment each function.\n\n```c\n#include <stdio.h>\n",
    "The history of the printing press is a story about far more than machines. When Johannes \
     Gutenberg began experimenting with movable metal type in Mainz around 1440, he could not have \
     imagined how thoroughly his invention would reshape European society. In the decades that \
     followed,",
];

struct PassOut {
    tokens: Vec<u32>,
    rounds: usize,
    drafted: usize,
    accepted: usize,
    accepts: Vec<usize>,
}

fn fnv64(v: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in v {
        for b in t.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
    }
    h
}

/// Common state before every pass: zeroed positional planes + taps (so round 1's tap restore reads
/// the same bytes in every pass), a reset slot, greedy unpenalized rows, then the served prefill
/// (chunked prompt[..n-1] + the seam decode step) and the served head prime.
fn pass_setup(model: &FwdModel, sc: &mut Scratch, ids: &[u32], max_new: usize, chunk: usize,
              psc: &mut Option<PrefillScratch>) -> Result<u32> {
    let span = ids.len() + max_new + 2 * (MTP_MAX_K + 2) + 4;
    zero_positional(model, 0, span)?;
    model.reset_slot(0)?;
    {
        let l = Launcher { dev: &model.dev, stream: &model.stream };
        let n = sc.taps_keep.len();
        xqlaunch!(l, "xq_memset_f32", ((((n + 255) / 256).max(1)) as u32, 1, 1), (256, 1, 1), 0,
                  (&sc.taps_keep, 0i64, n as i64))?;
    }
    let rows: Vec<Option<SampParams>> = vec![None; MTP_MAX_K + 1];
    model.set_sampling(sc, &rows)?;
    let pens: Vec<Option<PenParams>> = vec![None; MTP_MAX_K + 1];
    model.set_penalties(sc, &pens)?;
    model.prefill_prompt(sc, ids, Some(chunk), psc)?;
    let pred0 = model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32;
    Ok(pred0)
}

/// One greedy fixed-depth (k = MTP_MAX_K) MTP generation through the SERVED mtp_round — the tuning
/// trajectory of §5.2. SR runs inside when a session is armed.
fn run_mtp_pass(model: &FwdModel, sc: &mut Scratch, head: &DraftHead, ids: &[u32], max_new: usize,
                chunk: usize, psc: &mut Option<PrefillScratch>) -> Result<PassOut> {
    let plen = ids.len();
    let pred0 = pass_setup(model, sc, ids, max_new, chunk, psc)?;
    // the served WP03 seam prime (tap snapshot into slot 0's window + the head's row plen-1)
    model.mtp_seam_prime(sc, psc.as_ref(), head, *ids.last().unwrap() as i32, plen, 0)?;
    let mut out = PassOut { tokens: vec![pred0], rounds: 0, drafted: 0, accepted: 0, accepts: Vec::new() };
    let (mut b, mut p, mut tap_row) = (pred0 as i32, plen, 0usize);
    let k = MTP_MAX_K;
    while !EOS_IDS.contains(&b) && out.tokens.len() < max_new && p + k + 2 < model.max_pos {
        let r = model.mtp_round(sc, head, 0, k, b, p, tap_row)?;
        out.rounds += 1;
        out.drafted += r.drafts.len();
        out.accepted += r.a;
        out.accepts.push(r.a);
        out.tokens.extend(r.emitted.iter().map(|&e| e as u32));
        b = r.last_tok;
        p = r.pos_next;
        tap_row = r.tap_row;
    }
    Ok(out)
}

/// Plain greedy decode (LI segment): the served seam, then `steps` served forward_step calls.
fn run_plain_pass(model: &FwdModel, sc: &mut Scratch, ids: &[u32], steps: usize, chunk: usize,
                  psc: &mut Option<PrefillScratch>, li: bool) -> Result<Vec<u32>> {
    let plen = ids.len();
    let pred0 = pass_setup(model, sc, ids, steps, chunk, psc)?;
    let mut toks = vec![pred0];
    if li { ARMED.store(true, Ordering::SeqCst); }
    let r = (|| -> Result<()> {
        for i in 0..steps.saturating_sub(1) {
            let t = *toks.last().unwrap() as i32;
            if EOS_IDS.contains(&t) || plen + i + 1 >= model.max_pos { break; }
            let ids = model.forward_step(sc, &[t], &[plen + i], &[0], None)?;
            toks.push(ids[0] as u32);
        }
        Ok(())
    })();
    ARMED.store(false, Ordering::SeqCst);
    r?;
    Ok(toks)
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn local_ipv4s() -> Vec<String> {
    let mut out = Vec::new();
    unsafe {
        let mut ifap: *mut libc::ifaddrs = std::ptr::null_mut();
        if libc::getifaddrs(&mut ifap) != 0 { return out; }
        let mut p = ifap;
        while !p.is_null() {
            let a = (*p).ifa_addr;
            if !a.is_null() && (*a).sa_family as i32 == libc::AF_INET {
                let sin = a as *const libc::sockaddr_in;
                out.push(std::net::Ipv4Addr::from(u32::from_be((*sin).sin_addr.s_addr)).to_string());
            }
            p = (*p).ifa_next;
        }
        libc::freeifaddrs(ifap);
    }
    out
}

/// Other gb10_inference processes on this box (/proc/*/comm; no spawn).
fn other_engines() -> Vec<u32> {
    let me = std::process::id();
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/proc") {
        for e in rd.flatten() {
            let Ok(pid) = e.file_name().to_string_lossy().parse::<u32>() else { continue };
            if pid == me { continue; }
            if let Ok(c) = std::fs::read_to_string(e.path().join("comm")) {
                if c.trim() == "gb10_inference" { v.push(pid); }
            }
        }
    }
    v
}

/// Resident compute apps other than this process (None = nvidia-smi unusable: cannot prove idle).
fn resident_apps() -> Option<Vec<String>> {
    let me = std::process::id().to_string();
    let out = tune::run_timeout("nvidia-smi", &["--query-compute-apps=pid,process_name,used_memory",
                                                "--format=csv,noheader"], std::time::Duration::from_secs(10))?;
    Some(out.lines().map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.contains("No running") && l.split(',').next().map(|p| p.trim()) != Some(me.as_str()))
        .collect())
}

/// §6.4.3-4 / §12 G-T8: only on an idle GPU (no other engine, no resident compute app). Err = the refusal reason.
fn preflight() -> std::result::Result<(), String> {
    let others = other_engines();
    if !others.is_empty() {
        return Err(format!("other gb10_inference processes are running (pids {others:?}) — the GPU is not idle"));
    }
    match resident_apps() {
        None => Err("nvidia-smi --query-compute-apps failed — cannot prove the GPU is idle".into()),
        Some(a) if !a.is_empty() => Err(format!("resident compute apps: {a:?}")),
        Some(_) => Ok(()),
    }
    .and_then(|_| {
        for (v, why) in [(crate::opt!("exl3-no-graph"), "graphs off"), (crate::opt!("mtp-dump"), "dump mode"),
                         (crate::opt!("exl3-gdn-ring-check"), "eager ring check"), (crate::opt!("wp12-xcheck"), "eager XCHECK"),
                         (crate::opt!("exl3-draft-graph"), "draft graphs toggled")] {
            if crate::opts::var_os(v).is_some() {
                return Err(format!("{v} is set ({why}) — the harness measures the served graphed path"));
            }
        }
        Ok(())
    })
}

/// nvidia-smi clocks / thermal snapshot (unsupported fields read "unknown", never zero).
fn gpu_env(phase: &str) -> serde_json::Value {
    let q = |fields: &str| tune::run_timeout("nvidia-smi", &[&format!("--query-gpu={fields}"),
                                                              "--format=csv,noheader,nounits"],
                                             std::time::Duration::from_secs(5));
    let full = "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active";
    let (names, out): (Vec<&str>, Option<String>) = match q(full) {
        Some(o) => (full.split(',').collect(), Some(o)),
        None => { let f = "clocks.sm,clocks.mem,temperature.gpu,power.draw"; (f.split(',').collect(), q(f)) }
    };
    let mut m = serde_json::Map::new();
    m.insert("phase".into(), phase.into());
    let vals: Vec<String> = out.as_deref().and_then(|o| o.lines().next())
        .map(|l| l.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
    for (i, n) in names.iter().enumerate() {
        let v = vals.get(i).cloned().unwrap_or_default();
        let v = if v.is_empty() || v.contains("N/A") || v.contains("Not Supported") { "unknown".to_string() } else { v };
        m.insert(n.to_string(), v.into());
    }
    serde_json::Value::Object(m)
}

/// The position model's cell: (unit, width, regime, timed replays in the block).
type PosCell = (&'static str, usize, u8, u8);
/// Fitted position offsets per cell (ms, indexed by `apos`); an empty table = no stratification.
type PosTable = HashMap<PosCell, Vec<f64>>;

fn pos_cell(s: &Sample) -> PosCell { (s.unit, s.m, s.regime, s.npos) }

/// §5.2 v2: fit the position model (tune::position_offsets) for every SR cell `want` accepts, over
/// ALL timed replays of the session in that cell (the effect belongs to the protocol, not to one arm
/// set: every phase's arms enter with their own arm effect).
fn pos_table(samples: &[Sample], want: &dyn Fn(&PosCell) -> bool) -> PosTable {
    let mut by: HashMap<PosCell, Vec<tune::PosObs>> = HashMap::new();
    for s in samples.iter().filter(|s| s.mode == "sr" && s.digest_ok && s.apos != NO_APOS && s.npos >= 2) {
        let cell = pos_cell(s);
        if !want(&cell) { continue; }
        by.entry(cell).or_default().push(tune::PosObs { round: s.round as u64, arm: ((s.phase as u64) << 16) | s.arm as u64,
                                                        pos: s.apos as usize, ms: s.ms });
    }
    by.into_iter().map(|(c, v)| (c, tune::position_offsets(&v, c.3 as usize))).collect()
}

/// A timed replay's time with its fitted position offset removed (the raw time when no offsets).
fn adj_ms(s: &Sample, offs: &PosTable) -> f64 {
    s.ms - offs.get(&pos_cell(s)).and_then(|f| f.get(s.apos as usize)).copied().unwrap_or(0.0)
}

/// Paired deltas (ms) of `arm` over the base arm for (unit, m): per (prompt, round) pair where both
/// ran and both digests held; position-stratified when `offs` holds the cell (v2).
fn paired(samples: &[Sample], unit: &str, m: usize, arm: usize, offs: &PosTable) -> Vec<f64> {
    let base: HashMap<(usize, usize), f64> = samples.iter()
        .filter(|s| s.mode == "sr" && s.unit == unit && s.m == m && s.arm == 0 && s.digest_ok)
        .map(|s| ((s.prompt, s.round), adj_ms(s, offs))).collect();
    samples.iter()
        .filter(|s| s.mode == "sr" && s.unit == unit && s.m == m && s.arm == arm && s.digest_ok)
        .filter_map(|s| base.get(&(s.prompt, s.round)).map(|b| adj_ms(s, offs) - b))
        .collect()
}

/// §5.7 test 1 (A/A null), v3 (2026-09-27, design §5.7 amendment): tau_selftest = max(abs, pct % of
/// the cell's base median) — defaults 0.02 ms and 0.05 %, `--autotune-selftest-tau <ms>` /
/// `--autotune-selftest-tau-pct <pct>`. PASS when |HL| < tau AND (the 95% CI covers 0 OR it lies
/// wholly inside +-tau). v1/v2 required the CI to cover 0, which a tight CI (large n) around a
/// practically-null offset fails by chance (3 cells at 95% ~ 14% family-wise) — receipt
/// selftest-1790486005: HL +0.0188 ms, CI95 [+0.0005, +0.0368] at verify m = 3.
#[derive(Clone, Copy, Debug, PartialEq)]
struct AaRule {
    abs_ms: f64,
    pct: f64,
}
impl AaRule {
    const DEFAULT: AaRule = AaRule { abs_ms: 0.02, pct: 0.05 };
    /// tau_selftest of a cell whose base replays have median `cell_median_ms`.
    fn tau(&self, cell_median_ms: f64) -> f64 {
        let rel = if cell_median_ms.is_finite() { self.pct / 100.0 * cell_median_ms.abs() } else { 0.0 };
        self.abs_ms.max(rel)
    }
    fn json(&self) -> serde_json::Value {
        serde_json::json!({"rule": "|HL| < tau AND (CI95 covers 0 OR CI95 within +-tau)",
                           "tau": format!("max({} ms, {} % of the cell's base median)", self.abs_ms, self.pct)})
    }
}
/// The A/A verdict at tau: (pass, the clause that decided).
fn aa_verdict(e: &tune::Est, tau: f64) -> (bool, &'static str) {
    if !(e.est.abs() < tau) { return (false, "|HL| >= tau"); }
    if e.lo <= 0.0 && 0.0 <= e.hi { return (true, "CI covers 0"); }
    if e.lo >= -tau && e.hi <= tau { return (true, "CI within +-tau"); }
    (false, "CI excludes 0 and leaves +-tau")
}
fn aa_ok(e: &tune::Est, tau: f64) -> bool { aa_verdict(e, tau).0 }
/// `--autotune-selftest-tau <ms>` (> 0, default 0.02) and `--autotune-selftest-tau-pct <pct>` (>= 0,
/// default 0.05): the A/A threshold of the self-test.
fn aa_args(args: &[String]) -> Result<AaRule> {
    let mut r = AaRule::DEFAULT;
    if let Some(v) = arg(args, "--autotune-selftest-tau") {
        r.abs_ms = v.parse::<f64>().ok().filter(|x| x.is_finite() && *x > 0.0)
            .with_context(|| format!("--autotune-selftest-tau must be a positive number of ms (got {v})"))?;
    }
    if let Some(v) = arg(args, "--autotune-selftest-tau-pct") {
        r.pct = v.parse::<f64>().ok().filter(|x| x.is_finite() && *x >= 0.0 && *x <= 10.0)
            .with_context(|| format!("--autotune-selftest-tau-pct must be a percentage in [0, 10] (got {v})"))?;
    }
    Ok(r)
}
/// Median of the base arm's timed SR replays in one (unit, m) cell (ms).
fn cell_median(samples: &[Sample], unit: &str, m: usize) -> f64 {
    let v: Vec<f64> = samples.iter().filter(|s| s.mode == "sr" && s.unit == unit && s.m == m && s.arm == 0 && s.digest_ok)
        .map(|s| s.ms).collect();
    tune::median(&v).unwrap_or(f64::NAN)
}
/// §5.7 test 2 (known positive): the spin read within +-0.03 ms of its value (unchanged by v2).
fn spin_ok(e: &tune::Est, spin_ms: f64) -> bool { (e.est - spin_ms).abs() <= 0.03 }

/// Per-position view of one (unit, m) cell for the self-test / receipts: (warm-up median, timed
/// median by position, per-arm position counts). Times in ms.
fn position_view(samples: &[Sample], unit: &str, m: usize, arms: usize)
    -> (Option<f64>, Vec<Option<f64>>, Vec<Vec<usize>>, usize) {
    let sel = |s: &&Sample| s.unit == unit && s.m == m && s.digest_ok;
    let warm: Vec<f64> = samples.iter().filter(sel).filter(|s| s.mode == "warm").map(|s| s.ms).collect();
    let timed: Vec<&Sample> = samples.iter().filter(sel).filter(|s| s.mode == "sr" && s.apos != NO_APOS).collect();
    let np = timed.iter().map(|s| s.apos as usize + 1).max().unwrap_or(0);
    let by_pos: Vec<Option<f64>> = (0..np).map(|p| tune::median(&timed.iter().filter(|s| s.apos as usize == p)
        .map(|s| s.ms).collect::<Vec<_>>())).collect();
    let mut counts = vec![vec![0usize; np]; arms];
    for s in &timed { if s.arm < arms { counts[s.arm][s.apos as usize] += 1; } }
    let rounds = timed.iter().map(|s| (s.prompt, s.round)).collect::<std::collections::HashSet<_>>().len();
    (tune::median(&warm), by_pos, counts, rounds)
}

/// The run's position effect per cell (unit x width x regime x timed replays, >= 8 rounds): the
/// warm-up median, the timed median by position and the fitted offsets (table provenance, receipts,
/// `--autotune-report`).
fn position_effect_json(samples: &[Sample], offs: &PosTable) -> serde_json::Value {
    let mut cells: Vec<PosCell> = samples.iter().filter(|s| s.mode == "sr" && s.apos != NO_APOS && s.npos >= 2)
        .map(pos_cell).collect::<std::collections::HashSet<_>>().into_iter().collect();
    cells.sort();
    let r3 = |x: f64| (x * 1e3).round() / 1e3;
    let mut out = Vec::new();
    for c in cells {
        let in_cell = |s: &&Sample| pos_cell(s) == c && s.digest_ok;
        let timed: Vec<&Sample> = samples.iter().filter(in_cell).filter(|s| s.mode == "sr" && s.apos != NO_APOS).collect();
        let rounds = timed.iter().map(|s| s.round).collect::<std::collections::HashSet<_>>().len();
        if rounds < 8 { continue; }
        let warm: Vec<f64> = samples.iter().filter(in_cell).filter(|s| s.mode == "warm").map(|s| s.ms).collect();
        let by_pos: Vec<Option<f64>> = (0..c.3 as usize).map(|p| tune::median(&timed.iter()
            .filter(|s| s.apos as usize == p).map(|s| s.ms).collect::<Vec<_>>())).collect();
        let seen: Vec<f64> = by_pos.iter().flatten().copied().collect();
        let spread = if seen.is_empty() { None } else {
            Some(seen.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - seen.iter().cloned().fold(f64::INFINITY, f64::min))
        };
        out.push(serde_json::json!({
            "unit": c.0, "m": c.1, "regime": if c.2 == 1 { "sparse" } else { "dense" }, "n_pos": c.3, "rounds": rounds,
            "warmup_median_ms": tune::median(&warm).map(r3),
            "median_ms_by_pos": by_pos.iter().map(|x| x.map(r3)).collect::<Vec<_>>(),
            "offsets_ms": offs.get(&c).map(|f| f.iter().map(|x| r3(*x)).collect::<Vec<_>>()),
            "spread_ms": spread.map(r3),
        }));
    }
    serde_json::Value::Array(out)
}

/// `--autotune-warmup N` (0..=4, default 2 since v3) and `--autotune-stratify on|off` (default on).
fn protocol_args(args: &[String]) -> Result<(usize, bool)> {
    let warmup = match arg(args, "--autotune-warmup") {
        None => DEFAULT_WARMUP,
        Some(v) => v.parse::<usize>().ok().filter(|&w| w <= MAX_WARMUP)
            .with_context(|| format!("--autotune-warmup must be 0..={MAX_WARMUP}"))?,
    };
    let stratify = match arg(args, "--autotune-stratify") {
        None | Some("on") => true,
        Some("off") => false,
        Some(v) => bail!("--autotune-stratify must be on or off (got {v})"),
    };
    Ok((warmup, stratify))
}

fn protocol_json(warmup: usize, stratify: bool) -> serde_json::Value {
    serde_json::json!({"version": 3, "warmup_replays": warmup, "order": "williams (balanced per unit x width x regime x arms)",
                       "analysis": if stratify { "position-stratified (median-polish offsets)" } else { "plain paired" },
                       "li": "williams schedule; first round(s) of every generation / segment untimed"})
}

/// `gb10_inference --autotune [--selftest]` / `--autotune-report <table>`. Returns the exit code:
/// 0 = PASS / table written, 1 = FAIL / table rejected by its gate, 2 = refused or error.
pub fn cli(args: &[String]) -> i32 {
    let r = if let Some(path) = arg(args, "--autotune-report") {
        report_cli(path)
    } else if args.iter().any(|a| a == "--selftest") {
        cli_inner(args)
    } else {
        tune_cli(args)
    };
    match r {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(e) => {
            eprintln!("AUTOTUNE_FAIL: {e:#}");
            2
        }
    }
}

/// `--autotune-report <table>`: parse + print (no GPU, no box rule: it only reads a file).
fn report_cli(path: &str) -> Result<bool> {
    let raw = std::fs::read(path).with_context(|| format!("read {path}"))?;
    let t: tune::Table = serde_json::from_slice(&raw).with_context(|| format!("parse {path}"))?;
    print!("{}", tune::report(&t, &raw));
    Ok(t.seal_ok())
}

fn cli_inner(args: &[String]) -> Result<bool> {
    // the box rule first (G-T8): on .11/.12 the refusal is the signal, before any pack check
    if let Err(why) = preflight() {
        println!("AUTOTUNE REFUSED: {why}");
        bail!("preflight refused: {why}");
    }
    let dir = arg(args, "--model-dir").context("--autotune --selftest requires --model-dir <EXL3 pack>")?.to_string();
    if !crate::exl3_serve::is_exl3_pack(&dir) { bail!("{dir} is not an EXL3 pack"); }
    let num = |k: &str, d: usize| -> Result<usize> {
        match arg(args, k) { None => Ok(d), Some(v) => v.parse().map_err(|_| anyhow::anyhow!("{k} must be an integer")) }
    };
    let max_pos = num("--max-seq-len", 8192)?;
    let chunk = num("--prefill-chunk", crate::exl3_serve::DEFAULT_PREFILL_CHUNK)?.max(1);
    let max_new = num("--max-new-tokens", 384)?.max(8);
    let li_steps = num("--autotune-li-steps", 64)?;
    let out_dir = arg(args, "--autotune-out").unwrap_or("tune/receipts").to_string();
    let seed: u64 = match arg(args, "--autotune-seed") {
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--autotune-seed must be an integer"))?,
        None => std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1),
    };
    let (warmup, stratify) = protocol_args(args)?;
    let aa_rule = aa_args(args)?;
    let spin_ns = 200_000i32;

    // the measured config = the served defaults unless a table is named (--tune-table, default off)
    tune::arm_tuner();
    let sel = tune::TableSel::parse(Some(arg(args, "--tune-table").unwrap_or("off")))?;
    let posture = tune::Posture { lanes: 1, mtp_max_k: MTP_MAX_K, dds: false, kv_fmt: kv_fmt_from_opts(),
                                  max_pos, prefill_chunk: chunk };
    tune::boot(&tune::BootReq { sel, model_dir: &dir, posture: posture.clone(), profile_mhz: None,
                                draft_on: false, tp: 1 });

    std::fs::create_dir_all(&out_dir).with_context(|| format!("mkdir {out_dir}"))?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let jsonl_path = format!("{out_dir}/selftest-{stamp}.jsonl");
    let mut jsonl = std::io::BufWriter::new(std::fs::File::create(&jsonl_path).with_context(|| format!("create {jsonl_path}"))?);
    let bin_sha = std::fs::read("/proc/self/exe").map(|b| tune::sha256_hex(&b)).unwrap_or_else(|_| "unknown".into());

    let t_load = std::time::Instant::now();
    let model = FwdModel::load(&dir, 1, max_pos)?;
    let head = model.mtp.as_ref().context("the pack has no mtp.* draft head — the self-test needs MTP rounds")?;
    let mut sc = FwdModel::scratch(model.dev(), &model.cfg, MTP_MAX_K + 1)?;
    let mut psc: Option<PrefillScratch> = None;
    let fp = tune::current_fingerprint(&dir, 1, tune::STOCK_PROFILE_MHZ);
    let fp_json = serde_json::to_value(&fp)?;
    let contract = tune::sha256_hex(fp_json.to_string().as_bytes())[..8].to_string();
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", dir.trim_end_matches('/')))?;
    let corpus: Vec<Vec<u32>> = match arg(args, "--autotune-corpus") {
        Some(f) => {
            let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(f)?)
                .context("--autotune-corpus must be a JSON array of prompt strings")?;
            texts.iter().map(|t| tok.encode(t, false)).collect::<Result<_>>()?
        }
        None => CORPUS.iter().map(|t| tok.encode(t, false)).collect::<Result<_>>()?,
    };
    anyhow::ensure!(corpus.iter().all(|c| c.len() >= 2), "every corpus prompt needs >= 2 tokens");
    println!("AUTOTUNE SELFTEST: model {dir}, load {:.1} s, corpus {} prompts {:?} tokens, max_new {max_new}, \
              k {}, verify widths [3, 6], draft s {}, LI steps {li_steps}, seed {seed}, receipts {jsonl_path}",
             t_load.elapsed().as_secs_f64(), corpus.len(), corpus.iter().map(|c| c.len()).collect::<Vec<_>>(),
             MTP_MAX_K, MTP_MAX_K);
    println!("AUTOTUNE SELFTEST: protocol v3 — {} untimed base warm-up replay(s) per block, balanced (Williams) \
              arm order per cell, analysis {}; A/A: |HL| < tau AND (CI95 covers 0 OR CI95 within +-tau), \
              tau = max({} ms, {} % of the cell's base median)", warmup,
             if stratify { "position-stratified (--autotune-stratify off = plain paired)" } else { "plain paired (not stratified)" },
             aa_rule.abs_ms, aa_rule.pct);
    writeln!(jsonl, "{}", serde_json::json!({
        "receipt": "gb10-exl3-tune-selftest/3", "fingerprint": fp_json, "posture": serde_json::to_value(&posture)?,
        "binary_sha256": bin_sha, "commit": env!("TUNE_GIT_COMMIT"), "dirty": env!("TUNE_GIT_DIRTY"),
        "kernel_build_id": env!("KERNEL_BUILD_ID"), "model_dir": dir, "seed": seed, "contract": contract,
        "date_unix": stamp, "tune_status": format!("{:?}", tune::status().map(|s| (s.state, s.reason))),
        "protocol": protocol_json(warmup, stratify), "aa_rule": aa_rule.json(),
    }))?;
    writeln!(jsonl, "{}", serde_json::json!({"env": gpu_env("start")}))?;

    // base2 = the base config under a distinct signature (a separately captured A/A twin graph)
    let arms = vec![
        Arm::base(),
        Arm { name: "base2".into(), assign: vec![(&tune::T_SELFTEST_SPIN, None, 0)] },
        Arm { name: "spin".into(), assign: vec![(&tune::T_SELFTEST_SPIN, None, spin_ns)] },
    ];
    let cfg = SrCfg { verify_on: true, draft_on: true, li_on: false, verify_widths: vec![3, 6],
                      draft_s: MTP_MAX_K, mem_cap: 512 << 20 };
    let mut session = Session::new(&model, cfg.clone(), arms.clone(), seed)?;
    session.set_warmup(warmup)?;
    SESSION.with(|c| *c.borrow_mut() = Some(session));

    let mut traj_ok = true;
    let mut li_tok_ok = true;
    let mut traj_lines = Vec::new();
    let run = (|| -> Result<()> {
        for (pi, ids) in corpus.iter().enumerate() {
            // pass A: the served path, harness disarmed
            let a = run_mtp_pass(&model, &mut sc, head, ids, max_new, chunk, &mut psc)?;
            writeln!(jsonl, "{}", serde_json::json!({"env": gpu_env(&format!("prompt{pi} after pass A"))}))?;
            // re-check tenants at the family boundary (§6.4.3)
            if let Some(ap) = resident_apps() { if !ap.is_empty() { bail!("a tenant appeared: {ap:?}"); } }
            // pass B: the same generation with SR armed
            with_session(|s| { s.prompt = pi; s.cfg = SrCfg { li_on: false, ..cfg.clone() }; Ok(()) })?;
            ARMED.store(true, Ordering::SeqCst);
            let b = run_mtp_pass(&model, &mut sc, head, ids, max_new, chunk, &mut psc);
            ARMED.store(false, Ordering::SeqCst);
            let b = b?;
            with_session(|s| s.finish(&model))?;
            writeln!(jsonl, "{}", serde_json::json!({"env": gpu_env(&format!("prompt{pi} after pass B"))}))?;
            let same = a.tokens == b.tokens && a.rounds == b.rounds && a.drafted == b.drafted
                && a.accepted == b.accepted && a.accepts == b.accepts;
            traj_ok &= same;
            traj_lines.push(format!(
                "prompt {pi}: served tokens {} hash {:016x} rounds {} drafted {} accepted {} | SR-on tokens {} hash {:016x} \
                 rounds {} drafted {} accepted {} -> {}",
                a.tokens.len(), fnv64(&a.tokens), a.rounds, a.drafted, a.accepted,
                b.tokens.len(), fnv64(&b.tokens), b.rounds, b.drafted, b.accepted, if same { "IDENTICAL" } else { "DIFFERENT" }));
            writeln!(jsonl, "{}", serde_json::json!({"trajectory": {
                "prompt": pi, "served": {"tokens": a.tokens.len(), "hash": format!("{:016x}", fnv64(&a.tokens)),
                                          "rounds": a.rounds, "drafted": a.drafted, "accepted": a.accepted},
                "sr_on": {"tokens": b.tokens.len(), "hash": format!("{:016x}", fnv64(&b.tokens)),
                          "rounds": b.rounds, "drafted": b.drafted, "accepted": b.accepted},
                "identical": same }}))?;
            // LI segment: plain steps through the served forward_step, arms from the LI schedule
            if li_steps > 1 {
                with_session(|s| { s.prompt = pi; s.cfg = SrCfg { verify_on: false, draft_on: false, li_on: true, ..cfg.clone() }; Ok(()) })?;
                let plain = run_plain_pass(&model, &mut sc, ids, li_steps, chunk, &mut psc, true)?;
                with_session(|s| s.finish(&model))?;
                let n = plain.len().min(a.tokens.len());
                let ok = plain[..n] == a.tokens[..n];
                li_tok_ok &= ok;
                traj_lines.push(format!("prompt {pi}: LI plain greedy {} tokens == MTP served prefix: {}", n, if ok { "YES" } else { "NO" }));
            }
        }
        Ok(())
    })();
    let session = SESSION.with(|c| c.borrow_mut().take()).context("session vanished")?;
    run?;
    writeln!(jsonl, "{}", serde_json::json!({"env": gpu_env("end")}))?;

    // ---- receipts: every sample (v2: + the arm's place in its timed block and the block size)
    let arm_name = |a: usize| if a == usize::MAX { "canonical" } else if a == WARMUP_ARM { "warmup" }
                              else { arms.get(a).map_or("?", |x| x.name.as_str()) };
    for s in &session.samples {
        writeln!(jsonl, "{}", serde_json::json!({
            "case": format!("{}:m{}", s.unit, s.m), "variant": arm_name(s.arm), "trial": s.round,
            "time_us": s.ms * 1e3, "correct": s.digest_ok, "contract": format!("{contract}:{}:graph", s.mode),
            "round": s.round, "m": s.m, "fam": s.unit, "regime": if s.regime == 1 { "sparse" } else { "dense" },
            "pos": s.pos, "prompt": s.prompt, "mode": s.mode,
            "arm_pos": if s.apos == NO_APOS { serde_json::Value::Null } else { serde_json::json!(s.apos) },
            "n_pos": s.npos }))?;
    }

    // ---- §5.7 tests (v2: the verdict follows --autotune-stratify; both analyses are printed)
    let mut pass = true;
    let mut lines: Vec<String> = Vec::new();
    let cells: [(&str, usize); 3] = [("verify_dry", 3), ("verify_dry", 6), ("draft", MTP_MAX_K)];
    let fmt = |e: &tune::Est| format!("n={} HL={:+.4} ms CI95=[{:+.4}, {:+.4}]", e.n, e.est, e.lo, e.hi);
    let offs = pos_table(&session.samples, &|_| true);
    let none = PosTable::new();
    let (used, other, used_name, other_name) = if stratify { (&offs, &none, "position-stratified", "plain paired") }
                                               else { (&none, &offs, "plain paired", "position-stratified") };
    // (4) the per-position medians: the bias the protocol removes, visible
    for (unit, m) in cells {
        let (warm, by_pos, counts, rounds) = position_view(&session.samples, unit, m, arms.len());
        let f = offs.iter().filter(|(c, _)| c.0 == unit && c.1 == m).max_by_key(|(c, _)| c.3).map(|(_, f)| f.clone());
        let ms = |x: Option<f64>| x.map_or("-".to_string(), |v| format!("{v:.3}"));
        let bal = counts.iter().all(|c| c.iter().max().unwrap_or(&0) - c.iter().min().unwrap_or(&0) <= 1);
        lines.push(format!(
            "SELFTEST POSITION {unit} m={m}: rounds {rounds} | warm-up median {} ms | timed median by position [{}] ms | \
             fitted offsets [{}] ms | arm x position counts {} -> {}",
            ms(warm), by_pos.iter().map(|x| ms(*x)).collect::<Vec<_>>().join(", "),
            f.map_or("-".into(), |f| f.iter().map(|x| format!("{x:+.3}")).collect::<Vec<_>>().join(", ")),
            counts.iter().enumerate().map(|(a, c)| format!("{}{:?}", arm_name(a), c)).collect::<Vec<_>>().join(" "),
            if bal { "balanced" } else { "UNBALANCED" }));
    }
    for (unit, m) in cells {
        let d = paired(&session.samples, unit, m, 1, used);
        let alt = tune::hl_paired(&paired(&session.samples, unit, m, 1, other));
        let alt_s = alt.map_or("n/a".into(), |e| fmt(&e));
        match tune::hl_paired(&d) {
            Some(e) if e.n >= 20 => {
                let med = cell_median(&session.samples, unit, m);
                let tau = aa_rule.tau(med);
                let (ok, why) = aa_verdict(&e, tau);
                pass &= ok;
                lines.push(format!("SELFTEST 1 A/A {unit} m={m}: {} ({used_name}) | tau {tau:.4} ms (max({}, {} % x base median \
                                    {med:.3})) -> {} ({why}) | {other_name}: {alt_s}",
                                   fmt(&e), aa_rule.abs_ms, aa_rule.pct, if ok { "PASS" } else { "FAIL" }));
            }
            other => {
                pass = false;
                lines.push(format!("SELFTEST 1 A/A {unit} m={m}: INSUFFICIENT samples ({})", other.map_or(0, |e| e.n)));
            }
        }
    }
    let spin_ms = spin_ns as f64 / 1e6;
    for (unit, m) in cells {
        let d = paired(&session.samples, unit, m, 2, used);
        let alt = tune::hl_paired(&paired(&session.samples, unit, m, 2, other));
        let alt_s = alt.map_or("n/a".into(), |e| fmt(&e));
        match tune::hl_paired(&d) {
            Some(e) if e.n >= 20 => {
                let ok = spin_ok(&e, spin_ms);
                pass &= ok;
                lines.push(format!("SELFTEST 2 known-positive spin {unit} m={m}: {} ({used_name}; target {spin_ms:+.3} +- 0.030) \
                                    -> {} | {other_name}: {alt_s}", fmt(&e), if ok { "PASS" } else { "FAIL" }));
            }
            other => {
                pass = false;
                lines.push(format!("SELFTEST 2 known-positive spin {unit} m={m}: INSUFFICIENT samples ({})", other.map_or(0, |e| e.n)));
            }
        }
    }
    let xok = session.xfail.is_empty() && session.canon_checks.0 > 0;
    pass &= traj_ok && xok;
    for l in &traj_lines { lines.push(format!("SELFTEST 3 trajectory {l}")); }
    lines.push(format!("SELFTEST 3 trajectory invariance (greedy tokens + [mtp-stats] rounds/drafted/accepted, SR on vs off): {}",
                       if traj_ok { "PASS" } else { "FAIL" }));
    lines.push(format!("SELFTEST digests: base + warm-up replays == canonical {}/{}; arms == base {}/{} -> {}",
                       session.canon_checks.0 - session.canon_checks.1, session.canon_checks.0,
                       session.arm_checks.0 - session.arm_checks.1, session.arm_checks.0,
                       if xok { "PASS" } else { "FAIL" }));
    for f in session.xfail.iter().take(20) { lines.push(f.clone()); }
    // LI (informational: unpaired, served conditions; the segment warm-ups excluded)
    let li_of = |arm: usize| -> Vec<f64> { session.samples.iter().filter(|s| s.mode == "li" && s.arm == arm).map(|s| s.ms).collect() };
    {
        let li_warm: Vec<f64> = session.samples.iter().filter(|s| s.unit == "step" && s.mode == "warm").map(|s| s.ms).collect();
        let li_pos: Vec<String> = (0..arms.len()).map(|p| tune::median(&session.samples.iter()
            .filter(|s| s.mode == "li" && s.apos as usize == p).map(|s| s.ms).collect::<Vec<_>>())
            .map_or("-".into(), |x| format!("{x:.3}"))).collect();
        lines.push(format!("SELFTEST INFO LI step position: warm-up median {} ms (n {}) | median by in-block position [{}] ms",
                           tune::median(&li_warm).map_or("-".into(), |x| format!("{x:.3}")), li_warm.len(), li_pos.join(", ")));
    }
    for (arm, name) in [(1usize, "A/A base2"), (2, "spin")] {
        if let Some(e) = tune::hl_two_sample(&li_of(0), &li_of(arm)) {
            lines.push(format!("SELFTEST INFO LI step {name} vs base: {}", fmt(&e)));
        }
    }
    lines.push(format!("SELFTEST INFO LI plain greedy == served MTP greedy prefix: {}", if li_tok_ok { "YES" } else { "NO" }));
    let canon: Vec<f64> = session.samples.iter().filter(|s| s.arm == usize::MAX).map(|s| s.ms).collect();
    if !canon.is_empty() {
        let mut c = canon.clone();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lines.push(format!("SELFTEST INFO canonical verify m={}: median {:.3} ms over {} rounds", MTP_MAX_K + 1, c[c.len() / 2], c.len()));
    }
    lines.push(format!("SELFTEST INFO tuner graphs captured {} (mem cap hit: {})", session.captures, session.mem_cap_hit));
    let verdict = format!("SELFTEST RESULT: {} (commit {} dirty {}, binary sha256 {}, kernel_build_id {}, tune_build_id {}, \
                           model {dir}, date_unix {stamp}, receipts {jsonl_path})",
                          if pass { "PASS" } else { "FAIL" }, env!("TUNE_GIT_COMMIT"), env!("TUNE_GIT_DIRTY"), bin_sha,
                          env!("KERNEL_BUILD_ID"), env!("TUNE_BUILD_ID"));
    lines.push(verdict);
    for l in &lines { println!("{l}"); }
    writeln!(jsonl, "{}", serde_json::json!({"position_effect": position_effect_json(&session.samples, &offs),
                                             "protocol": protocol_json(warmup, stratify)}))?;
    writeln!(jsonl, "{}", serde_json::json!({"summary": lines, "pass": pass}))?;
    jsonl.flush()?;
    drop(session); // tuner graphs + events before the model's context goes away
    Ok(pass)
}

// =============================================================================================
// T2 (PLAN/AUTOTUNE_DESIGN.md §6-§8, §11 T2): the search, the final gate and the table.
//
//   gb10_inference --autotune --model-dir <EXL3 pack> [served posture flags] [--autotune-budget 20]
//
// 1. refuse unless this is an idle .13/.14 (§6.4 / G-T8); boot the registry at the defaults;
//    load the model at the SERVED posture (--max-batch / --max-seq-len / --kv-cache /
//    --prefill-chunk / --draft-confidence, the server's own defaults) — the table's posture section
//    must equal what will be served or boot falls back.
// 2. probe: base-only SR rounds at every served verify width — the base unit medians (tau, budget)
//    and, per unit, which registry entries the capture consulted (§4.8a).
// 3. static pruning (§6.1): class S only, a proven domain (non-empty xcheck), no Load scope, no
//    structural / harness / umbrella entries, no env-overridden entry, consulted by some unit,
//    valid() values only.
// 4. dense sweep (§6.2): families in ledger-weighted order (ev / seconds), each a §6.3 race of
//    paired within-round SR differences (HL + Wilcoxon, tau, default preference, sequential
//    stopping, confirmation on fresh rounds), every sample digest-checked (§5.6) and every
//    candidate graph node-hashed (§4.8b: an identical graph leaves the race as INERT). Width
//    groups: race over the representative widths, confirm over every served width, split a width
//    back only when it is clearly slower there. Declared interacting pairs race as a joint grid.
//    Sweep 2 re-races the adopted families against the final base until nothing changes.
// 5. sparse point: the QSA families past the dense/sparse switch (a real long prompt, labelled).
// 6. prefill point: the chunk-class MT rule (the B3 known answer, §5.7.4) — snapshot-restored
//    chunk replays, paired per repetition, KV-row digests.
// 7. the final gate (§6.5): LI whole-round A/B (table vs defaults) in the served posture (DDS on,
//    the served calibrator + WP23 guard), 3 classes; the automatic hash gate (fixed chain AND
//    DDS: greedy tokens + rounds / drafted / accepted identical).
// 8. the table (§7.1): fingerprint (build, PTX, registry, model, GPU, driver, TP degree, clock
//    profile), served posture, provenance, decisions with evidence, family statuses, the gate.
//    Written only when the gate passes; a rejected table goes to the receipts directory only.
// =============================================================================================

/// What a family's candidates are measured on (§5.1 units).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    /// verify_dry replays -> Verify-family decisions (per width, or per family for a Fam knob)
    Verify,
    /// draft-chain replays -> DraftChain + DraftPass decisions (m = 1)
    Draft,
    /// both units summed per round (verify + draft x passes/round share) -> one Global / Shape value
    Round,
}

impl Target {
    fn name(self) -> &'static str {
        match self { Target::Verify => "verify_dry", Target::Draft => "draft", Target::Round => "round(verify+draft)" }
    }
}

/// One searchable knob instance: a registry entry, optionally pinned to one key (a shape).
#[derive(Clone)]
struct Knob {
    def: &'static tune::TunableDef,
    key: Option<tune::Key>,
    label: String,
}

/// One family of the plan: one knob (a coordinate step) or a declared interacting pair (§6.2 joint
/// grid, <= 9 combinations). `cands` = value tuples (one per knob), the base tuple excluded.
#[derive(Clone)]
struct Family {
    name: String,
    knobs: Vec<Knob>,
    cands: Vec<Vec<i32>>,
    regime: u8,
    ev_ms: f64,
}

/// §6.2 ledger (dense regime): (registry id, expected realizable ms per round at the §10 central
/// estimate, shares of the m = 6 inventory). Everything else registered is listed NOT SEARCHED.
const DENSE_SPECS: &[(&str, f64)] = &[
    ("moe.a1b3", 0.25), ("moe.coop", 0.15), ("moe.fh", 0.10),
    ("hc.rb", 0.15), ("wp11.r2", 0.20), ("wp11.r4", 0.10), ("wp11.cvt_prmt", 0.05), ("hc.mix4", 0.05),
    ("wp12.step", 0.30), ("wp12.ab", 0.07), ("wp12.diet", 0.05), ("wp12.conv", 0.05),
    ("lmh.nst", 0.25), ("lmh.had", 0.10), ("lmh.off", 0.10),
    ("wp10.fold", 0.10), ("wp10.gemv", 0.05), ("router.fused", 0.05),
    ("wp09.am_nb", 0.05), ("wp09.argmax", 0.03),
    ("attn.dense_v3", 0.10),
    ("wp27.forks", 0.30), ("wp27.prio", 0.10),
    // T0c (p5e): W4/HC geometry + levers (default on; every value bitwise)
    ("w4hc.g", 0.10), ("w4hc.pf", 0.08), ("w4hc.la", 0.05),
    ("w4hc.fuse", 0.05), ("w4hc.mix", 0.03), ("w4hc.inj", 0.05), ("w4hc.m1", 0.03),
    // W4/SMALL items (default on); w4s.gdn lives in verify_commit (LI only: INERT for the SR units)
    ("w4s.router", 0.05), ("w4s.dattn", 0.05), ("w4s.ple", 0.03), ("w4s.draft", 0.05), ("w4s.gdn", 0.03),
    // the opt-in packages' levers: consulted (so raced) only where the package is on — INERT otherwise
    ("w4dense.fix", 0.03), ("w4dense.suh", 0.03), ("w4dense.multi", 0.03), ("w4dense.silu", 0.03),
    ("w4dense.gsat", 0.05),
    ("w4moe.order", 0.05), ("w4moe.g", 0.05), ("w4moe.shovl", 0.05),
    // A5-K1: shared expert ‖ routed gate/up on the default WP20 path
    ("moe.shovl", 0.30),
    // A5-K2: router fold with coalesced weight staging
    ("moe.router_coal", 0.30),
    // A5-K7: xq_had_suh_multi folded into the WP20 gate/up prologue
    ("moe.gu_fold", 0.30),
];
/// §6.2 declared interacting pairs (joint grids): (a, a's values, b, b's values, ev).
const PAIRS: &[(&str, &[i32], &str, &[i32], f64)] = &[
    ("wp20.mode", &[0, 1, 2], "wp20.diet", &[0, 1], 0.60),
    ("hc.bxr", &[32, 40, 48], "wp11.r1", &[0, 1, 2], 0.50),
    // p5e: PSK is opt-in (default off) — the package re-evaluated per width at each trellis prefetch
    // distance (psk.on = 0 with another pf is an identical graph: INERT, never timed)
    ("psk.on", &[0, 1], "psk.pf", &[0, 1, 2], 0.30),
    // T0c (p5e): the opt-in W4 packages re-evaluated in-round per width (their standalone estimates
    // lacked exactly this), each at its ring depths (an off package at another depth is INERT)
    ("w4dense.on", &[0, 1], "w4dense.nst", &[4, 6, 8], 0.30),
    ("w4moe.parts", &[0, 3, 7], "w4moe.nst", &[0, 4, 8], 0.30),
];
/// Sparse-regime families (S17-S19): searched at the sparse context point only.
const SPARSE_SPECS: &[(&str, f64)] = &[
    ("qsa.score_mr_ctas", 0.30), ("qsa.sel_gather", 0.20), ("qsa.topk_v2", 0.10),
    ("qsa.score_generic", 0.05), ("qsa.sel_vec", 0.05),
    // T0c (p5e): the opt-in row-shared union gather (sparse verifies, m >= 2)
    ("w4qsa.on", 0.20),
    // A5-K3: latency-lean top-k select (asc3), sparse regime only
    ("attn.qsa_select", 0.30),
];
/// Per-shape PSK grid families (T4e): ev per shape instance.
const PSK_EV_PER_SHAPE: f64 = 0.08;
/// Registered but never searched by default, with the reason (the report lists them).
const NOT_SEARCHED: &[(&str, &str)] = &[
    ("qsa.score_mr", "structural: 0 brings back the bucket-keyed graph set (qsa_bucket_free)"),
    ("selftest.spin_ns", "harness: the self-test's known positive"),
    ("reprime.graph", "graph vs eager: the tuner measures graphs; eager is the diagnostic escape"),
    ("draft.graph", "graph vs eager: the tuner measures graphs; eager is the diagnostic escape"),
    ("wp27.graphs", "redundant with the per-family wp27.forks"),
    ("wp09.off", "umbrella escape: its parts (wp09.argmax, wp09.am_nb) are searched"),
    ("wp11.off", "umbrella escape: its parts (wp11.r1/r2/r4/cvt) are searched"),
    ("prefill.wide_mt", "searched at the prefill point (chunk classes)"),
    ("w4hc.off", "umbrella escape: its levers (w4hc.fuse/mix/inj/m1) and geometry (w4hc.g/pf/la) are searched"),
];
/// Share of a draft chain's replay delta that lands on a served DDS round (§10: ~3.5 passes per
/// round against the fixed chain's s passes): Round-target objective = dv + DRAFT_SHARE/s * dd.
const DRAFT_PASSES_PER_ROUND: f64 = 3.5;

const PY_PROMPT: &str = "Write a Python module that implements an LRU cache class with get, put and \
    delete methods, backed by a dict and a doubly linked list, with type hints, docstrings and a \
    small unittest suite at the end.\n\n```python\n";
const SCIFI_PROMPT: &str = "Write the opening scene of a science fiction short story. A salvage crew \
    boards a derelict generation ship that has been drifting between stars for three hundred \
    years, and discovers that the ship's crew is still alive and has no idea how much time has \
    passed.\n\n";

/// One adopted decision (with the overlay keys it contributes to the base of later families).
#[derive(Clone)]
struct Dec {
    family: String,
    target: Target,
    knob: Knob,
    value: i32,
    /// table decision scope(s) (one per family written: Draft fans out to DraftChain + DraftPass)
    scopes: Vec<tune::DecScope>,
    keys: Vec<tune::Key>,
    evidence: serde_json::Value,
    /// the confirmed per-round gain (ms, negative = faster) the prediction check sums
    gain_ms: f64,
}

/// The served DDS lane state the final gate mirrors (exl3_serve::step_mtp's DDS branch).
struct DdsLane {
    cal: crate::exl3_serve::DraftCal,
    fit: crate::exl3_serve::CostFit,
    ema_ms: f64,
    ema_tok: f64,
    rounds: usize,
    ctr: u32,
    drafted: usize,
    accepted: usize,
}

impl DdsLane {
    fn new() -> DdsLane {
        DdsLane { cal: crate::exl3_serve::DraftCal::new(), fit: Default::default(), ema_ms: 0.0, ema_tok: 0.0,
                  rounds: 0, ctr: 0, drafted: 0, accepted: 0 }
    }
}

/// One generation's outcome (hash gate).
#[derive(Clone, Debug, PartialEq)]
struct GenOut {
    tokens: Vec<u32>,
    rounds: usize,
    drafted: usize,
    accepted: usize,
}

struct Tuner<'a> {
    model: &'a FwdModel,
    head: &'a DraftHead,
    sc: Scratch,
    psc: Option<PrefillScratch>,
    chunk: usize,
    k: usize,
    draft_conf: f64,
    max_new: usize,
    corpus: Vec<Vec<u32>>,
    corpus_label: &'static str,
    gen_next: usize,
    /// (b, p, tap_row, emitted) of the running tuning generation
    gen: Option<(i32, usize, usize, usize)>,
    round_ms: Vec<f64>,
    widths_rep: Vec<usize>,
    widths_all: Vec<usize>,
    /// base unit medians: ("verify", m, regime) / ("draft", s, regime)
    base_ms: HashMap<(&'static str, usize, u8), f64>,
    base_mad: HashMap<(&'static str, usize, u8), f64>,
    decisions: Vec<Dec>,
    families: serde_json::Map<String, serde_json::Value>,
    jsonl: std::io::BufWriter<std::fs::File>,
    t0: std::time::Instant,
    budget_s: f64,
    rcfg: tune::RaceCfg,
    regime: u8,
    gpu_log: Vec<serde_json::Value>,
    throttle_events: usize,
    mem_cap: usize,
    /// §5.2 v2: untimed base replays at the start of every block / generation (--autotune-warmup)
    warmup: usize,
    /// §5.2 v2: remove the fitted position effect before pairing (--autotune-stratify, default on)
    stratify: bool,
    /// v2: the prefill point's per-chunk balanced order cursors (persist across race calls)
    pf_order: tune::BalancedOrder,
    /// the prefill point's B3 known answer: None = not run, Some(true) = HOLDS, Some(false) = DEVIATES
    known_answer: Option<bool>,
}

fn sm_clock(env: &serde_json::Value) -> Option<f64> {
    env.get("clocks.sm").and_then(|v| v.as_str()).and_then(|s| s.trim().parse::<f64>().ok())
}
fn throttled(env: &serde_json::Value) -> bool {
    // clocks_event_reasons: 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal,
    // 0x80 HW power brake (0x1 idle / 0x2 app clocks / 0x100 sync boost are not throttling)
    env.get("clocks_event_reasons.active").and_then(|v| v.as_str())
        .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .map_or(false, |m| m & (0x4 | 0x8 | 0x20 | 0x40 | 0x80) != 0)
}

impl<'a> Tuner<'a> {
    fn elapsed_s(&self) -> f64 { self.t0.elapsed().as_secs_f64() }

    fn say(&mut self, line: String) {
        println!("{line}");
        let _ = writeln!(self.jsonl, "{}", serde_json::json!({"log": line, "t_s": self.elapsed_s()}));
        let _ = self.jsonl.flush();
    }

    fn env_snapshot(&mut self, phase: &str) -> serde_json::Value {
        let e = gpu_env(phase);
        if throttled(&e) { self.throttle_events += 1; }
        let _ = writeln!(self.jsonl, "{}", serde_json::json!({"env": e.clone(), "t_s": self.elapsed_s()}));
        self.gpu_log.push(e.clone());
        e
    }

    /// §6.4.3: abort (keeping the receipts) the moment another tenant appears.
    fn tenants_ok(&mut self) -> Result<()> {
        if let Some(a) = resident_apps() {
            if !a.is_empty() { bail!("a tenant appeared on the GPU mid-run: {a:?} — aborting (receipts kept)"); }
        }
        let o = other_engines();
        if !o.is_empty() { bail!("another gb10_inference appeared mid-run (pids {o:?}) — aborting (receipts kept)"); }
        Ok(())
    }

    // ---- the tuning trajectory (§5.2): a real greedy fixed-depth generation, SR armed per round

    fn start_prompt(&mut self) -> Result<()> {
        let idx = self.gen_next % self.corpus.len();
        self.gen_next += 1;
        let ids = self.corpus[idx].clone();
        with_session(|s| { s.prompt = idx; Ok(()) })?;
        let pred0 = pass_setup(self.model, &mut self.sc, &ids, self.max_new, self.chunk, &mut self.psc)?;
        self.model.mtp_seam_prime(&mut self.sc, self.psc.as_ref(), self.head, *ids.last().unwrap() as i32, ids.len(), 0)?;
        self.gen = Some((pred0 as i32, ids.len(), 0, 1));
        Ok(())
    }

    fn one_round(&mut self) -> Result<()> {
        let fresh = match self.gen {
            None => true,
            Some((b, p, _, e)) => EOS_IDS.contains(&b) || e >= self.max_new || p + self.k + 2 >= self.model.max_pos,
        };
        if fresh { self.start_prompt()?; }
        let (b, p, tap_row, emitted) = self.gen.unwrap();
        if EOS_IDS.contains(&b) { self.gen = None; return Ok(()); }
        let t = std::time::Instant::now();
        ARMED.store(true, Ordering::SeqCst);
        let r = self.model.mtp_round(&mut self.sc, self.head, 0, self.k, b, p, tap_row);
        ARMED.store(false, Ordering::SeqCst);
        let r = r?;
        self.round_ms.push(t.elapsed().as_secs_f64() * 1e3);
        self.gen = Some((r.last_tok, r.pos_next, r.tap_row, emitted + r.emitted.len()));
        Ok(())
    }

    fn run_rounds(&mut self, n: usize) -> Result<()> {
        for _ in 0..n { self.one_round()?; }
        let model = self.model;
        with_session(|s| s.finish(model))?;
        Ok(())
    }

    fn set_arms(&mut self, arms: Vec<Arm>, verify_on: bool, draft_on: bool, widths: Vec<usize>) -> Result<usize> {
        let (k, cap) = (self.k, self.mem_cap);
        Ok(with_session(|s| {
            s.set_arms(arms)?;
            s.cfg = SrCfg { verify_on, draft_on, li_on: false, verify_widths: widths, draft_s: k, mem_cap: cap };
            Ok(s.phase)
        })?.context("tune session missing")?)
    }

    /// The adopted decisions as table records (one per scope).
    fn table_decisions(&self) -> Vec<tune::Decision> {
        let mut v = Vec::new();
        for d in &self.decisions {
            for sc in &d.scopes {
                v.push(tune::Decision { id: d.knob.def.id.into(), rev: d.knob.def.rev, class: d.knob.def.class.name().into(),
                                        scope: sc.clone(), value: d.value, default: Some(d.knob.def.default),
                                        evidence: d.evidence.clone() });
            }
        }
        v
    }

    /// The base assignment (every adopted decision, keyed), optionally without one family's.
    fn base_assign(&self, skip_family: Option<&str>) -> Vec<(&'static tune::TunableDef, Option<tune::Key>, i32)> {
        let mut v = Vec::new();
        for d in &self.decisions {
            if Some(d.family.as_str()) == skip_family { continue; }
            for k in &d.keys { v.push((d.knob.def, Some(*k), d.value)); }
        }
        v
    }

    // ---- probe (§6.1 / §4.8a / tau and budget inputs)

    fn probe(&mut self, regime: u8) -> Result<()> {
        let widths = self.widths_all.clone();
        let phase = self.set_arms(vec![Arm::base()], true, true, widths.clone())?;
        self.run_rounds(2 * widths.len() + 4)?;
        let samples: Vec<Sample> = with_session(|s| Ok(s.samples.iter().filter(|x| x.phase == phase && x.arm == 0 && x.digest_ok)
            .cloned().collect::<Vec<_>>()))?.unwrap_or_default();
        let mut groups: HashMap<(&'static str, usize), Vec<f64>> = HashMap::new();
        for s in &samples {
            let unit = if s.unit == "verify_dry" { "verify" } else if s.unit == "draft" { "draft" } else { continue };
            groups.entry((unit, s.m)).or_default().push(s.ms);
        }
        let mut line = format!("PROBE ({}): base unit medians", if regime == 1 { "sparse" } else { "dense" });
        let mut keys: Vec<_> = groups.keys().copied().collect();
        keys.sort();
        for key in keys {
            let v = &groups[&key];
            let med = tune::median(v).unwrap_or(f64::NAN);
            let mad = tune::median(&v.iter().map(|x| (x - med).abs()).collect::<Vec<_>>()).unwrap_or(f64::NAN);
            self.base_ms.insert((key.0, key.1, regime), med);
            self.base_mad.insert((key.0, key.1, regime), mad);
            line += &format!(" | {} m{} {:.3} ms (MAD {:.3}, n {})", key.0, key.1, med, mad, v.len());
        }
        let canon = tune::median(&self.round_ms[self.round_ms.len().saturating_sub(16)..]).unwrap_or(f64::NAN);
        line += &format!(" | canonical round {canon:.1} ms wall");
        self.say(line);
        let xf = with_session(|s| Ok(s.xfail.clone()))?.unwrap_or_default();
        if !xf.is_empty() {
            for l in xf.iter().take(8) { self.say(l.clone()); }
            bail!("the BASE replay differs from the canonical execution in the probe — the harness is not \
                   trustworthy on this build (fix before tuning)");
        }
        Ok(())
    }

    fn unit_ms(&self, unit: &'static str, regime: u8) -> f64 {
        let v: Vec<f64> = self.base_ms.iter().filter(|((u, _, r), _)| *u == unit && *r == regime).map(|(_, v)| *v).collect();
        if v.is_empty() { return if unit == "verify" { 45.0 } else { 8.0 }; }
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn tau_for(&self, target: Target, regime: u8) -> f64 {
        let base = match target {
            Target::Verify => self.unit_ms("verify", regime),
            Target::Draft => self.unit_ms("draft", regime),
            Target::Round => self.unit_ms("verify", regime) + self.unit_ms("draft", regime) * DRAFT_PASSES_PER_ROUND / self.k as f64,
        };
        tune::tau_ms(base)
    }

    /// Estimated seconds for one family race (stage 1 + ~half of stage 2 + a 50% confirmation).
    fn est_s(&self, f: &Family, t: Target) -> f64 {
        let rounds = self.rcfg.n_stage1 as f64 + (self.rcfg.n_max - self.rcfg.n_stage1) as f64 * 0.3
            + 0.5 * self.rcfg.n_confirm as f64;
        let canon = tune::median(&self.round_ms).unwrap_or(70.0);
        let unit = match t {
            Target::Verify => self.unit_ms("verify", f.regime),
            Target::Draft => self.unit_ms("draft", f.regime),
            Target::Round => self.unit_ms("verify", f.regime) + self.unit_ms("draft", f.regime),
        } + 0.3;
        // v2: + the base warm-up replay(s) every block
        rounds * (canon + (f.cands.len() + 1 + self.warmup) as f64 * unit) / 1e3
    }

    // ---- static pruning (§6.1) + the plan (§6.2)

    /// Why a registry entry is not searched (None = searchable).
    fn prune_reason(&self, def: &tune::TunableDef) -> Option<String> {
        if let Some((_, why)) = NOT_SEARCHED.iter().find(|(id, _)| *id == def.id) { return Some(why.to_string()); }
        if def.class != tune::Class::S { return Some(format!("class {}: not searched by default (never without consent)", def.class.name())); }
        if def.xcheck.is_empty() { return Some("DOMAIN UNPROVEN (no bit-diff harness)".into()); }
        if def.scope == tune::Scope::Load { return Some("Load scope: both layouts would have to be resident".into()); }
        if def.scope == tune::Scope::Chunk { return Some("prefill unit: registered, not searched in this build".into()); }
        if tune::env_overrides().iter().any(|(id, _, _)| *id == def.id) {
            return Some(format!("OVERRIDDEN(flag {}) — the override flag wins over every arm", def.env));
        }
        None
    }

    /// Units (per the probe captures) that consulted `def`.
    fn consulted_units(&self, def: &tune::TunableDef) -> Vec<&'static str> {
        with_session(|s| Ok(s.consulted_by.get(&def.slot).map(|x| x.iter().copied().collect::<Vec<_>>()).unwrap_or_default()))
            .ok().flatten().unwrap_or_default()
    }

    fn targets_for(&self, knobs: &[Knob]) -> Vec<Target> {
        let mut v = false;
        let mut d = false;
        for k in knobs {
            let u = self.consulted_units(k.def);
            v |= u.contains(&"verify");
            d |= u.contains(&"draft");
        }
        let per_fam = knobs.iter().all(|k| matches!(k.def.scope, tune::Scope::Width | tune::Scope::Fam));
        if per_fam {
            let mut t = Vec::new();
            if v { t.push(Target::Verify); }
            if d { t.push(Target::Draft); }
            t
        } else if v || d {
            vec![Target::Round]
        } else {
            Vec::new()
        }
    }

    /// Candidate values of one knob: its domain minus the base value, valid at the representative
    /// contexts of its targets.
    fn knob_cands(&self, k: &Knob, regime: u8) -> Vec<i32> {
        let base_v = self.decisions.iter().find(|d| d.knob.def.slot == k.def.slot && d.knob.key == k.key)
            .map(|d| d.value).unwrap_or(k.def.default);
        let ctxs: Vec<Ctx> = match k.key {
            Some(tune::Key::S(kk, n, b)) => vec![Ctx::UTIL.shape(kk, n, b as i32)],
            _ => self.widths_rep.iter().map(|&m| Ctx::fam(Fam::Verify, m).regime(regime == 1))
                .chain(std::iter::once(Ctx::fam(Fam::DraftChain, 1).regime(regime == 1))).collect(),
        };
        k.def.domain.iter().copied()
            .filter(|&v| v != base_v)
            .filter(|&v| ctxs.iter().any(|c| (k.def.valid)(c, v)))
            .collect()
    }

    /// The model's split-K chain shapes (singles and same-shape pairs), as psk_log_shapes walks them.
    fn psk_shapes(&self) -> (Vec<(i32, i32, i32, u32)>, Vec<(i32, i32, i32, u32)>) {
        use std::collections::BTreeSet;
        let mut singles: BTreeSet<(i32, i32, i32, u32)> = BTreeSet::new();
        let mut pairs: BTreeSet<(i32, i32, i32, u32)> = BTreeSet::new();
        let mut visit = |attn: Option<&AttnLayer>, gdn: Option<&GdnLayer>, moe: &MoeLayer| {
            let mut one = |q: &Quad| { if q.ks > 1 { singles.insert((q.k, q.n, q.bits, q.ks)); } };
            if let Some(g) = gdn { one(&g.in_qkv); one(&g.in_z); one(&g.out_proj); }
            if let Some(a) = attn {
                one(&a.q_proj); one(&a.o_proj); one(&a.k_proj); one(&a.v_proj);
                if let Some(ix) = &a.idx { one(&ix.qk); }
            }
            one(&moe.sh_gate); one(&moe.sh_up); one(&moe.sh_down);
            let mut two = |a: &Quad, b: &Quad| {
                if a.ks > 1 && a.ks == b.ks && a.k == b.k && a.n == b.n && a.bits == b.bits {
                    pairs.insert((a.k, a.n, a.bits, a.ks));
                }
            };
            if let Some(a) = attn { two(&a.k_proj, &a.v_proj); }
            two(&moe.sh_gate, &moe.sh_up);
        };
        for l in &self.model.layers {
            match &l.mixer {
                Mixer::Gdn(g) => visit(None, Some(g), &l.moe),
                Mixer::Attn(a) => visit(Some(a), None, &l.moe),
            }
        }
        visit(Some(&self.head.attn), None, &self.head.moe);
        for q in [&self.head.fc_hidden, &self.head.fc_embed] { if q.ks > 1 { singles.insert((q.k, q.n, q.bits, q.ks)); } }
        (singles.into_iter().collect(), pairs.into_iter().collect())
    }

    /// Build the dense or sparse plan; every registry entry ends up either in a family or in the
    /// report's not-searched list (with its reason).
    fn build_plan(&mut self, regime: u8) -> Vec<(Family, Target)> {
        let reg = |id: &str| tune::REGISTRY.iter().copied().find(|t| t.id == id);
        let mut fams: Vec<Family> = Vec::new();
        let mut paired: Vec<&str> = Vec::new();
        if regime == 0 {
            for (a, av, b, bv, ev) in PAIRS {
                let (Some(da), Some(db)) = (reg(a), reg(b)) else { continue };
                if self.prune_reason(da).is_some() || self.prune_reason(db).is_some() { continue; }
                paired.push(a);
                paired.push(b);
                let ka = Knob { def: da, key: None, label: a.to_string() };
                let kb = Knob { def: db, key: None, label: b.to_string() };
                let base = (da.default, db.default);
                let mut cands: Vec<Vec<i32>> = Vec::new();
                for &x in av.iter() {
                    for &y in bv.iter() {
                        if (x, y) != base { cands.push(vec![x, y]); }
                    }
                }
                cands.truncate(9);
                fams.push(Family { name: format!("{a} x {b}"), knobs: vec![ka, kb], cands, regime, ev_ms: *ev });
            }
        }
        let specs: &[(&str, f64)] = if regime == 0 { DENSE_SPECS } else { SPARSE_SPECS };
        for (id, ev) in specs {
            if paired.contains(id) { continue; }
            let Some(def) = reg(id) else { continue };
            if self.prune_reason(def).is_some() { continue; }
            let k = Knob { def, key: None, label: id.to_string() };
            let cands: Vec<Vec<i32>> = self.knob_cands(&k, regime).into_iter().map(|v| vec![v]).collect();
            fams.push(Family { name: id.to_string(), knobs: vec![k], cands, regime, ev_ms: *ev });
        }
        if regime == 0 {
            // T4e: psk.g per single shape, psk.g_pair per pair shape (Shape scope, keyed overlay)
            let (singles, pairs) = self.psk_shapes();
            let (s1, s2) = psk_slots(&self.model.dev);
            for (list, def, members, slots) in [(&singles, &T_PSK_G, 1u32, s1), (&pairs, &T_PSK_G2, 2u32, s2)] {
                if self.prune_reason(def).is_some() { continue; }
                for &(kk, n, bits, ks) in list.iter() {
                    let items = ((n.max(0) as u32) / 128) * ks * members;
                    let rule = items.min(slots) as i32;
                    let key = tune::Key::S(kk, n, bits.clamp(0, 255) as u8);
                    let k = Knob { def, key: Some(key), label: format!("{}[{kk}x{n} b{bits}]", def.id) };
                    let mut c: Vec<i32> = self.knob_cands(&k, regime).into_iter().filter(|&v| v > 0 && v != rule).collect();
                    c.sort_by_key(|&v| (v - rule).abs());
                    c.truncate(6);
                    fams.push(Family { name: k.label.clone(), knobs: vec![k], cands: c.into_iter().map(|v| vec![v]).collect(),
                                       regime, ev_ms: PSK_EV_PER_SHAPE });
                }
            }
            // chain.pair per pair shape
            if let Some(def) = reg("chain.pair") {
                if self.prune_reason(def).is_none() {
                    for &(kk, n, bits, _) in pairs.iter() {
                        let key = tune::Key::S(kk, n, bits.clamp(0, 255) as u8);
                        let k = Knob { def, key: Some(key), label: format!("chain.pair[{kk}x{n} b{bits}]") };
                        let c: Vec<Vec<i32>> = self.knob_cands(&k, regime).into_iter().map(|v| vec![v]).collect();
                        fams.push(Family { name: k.label.clone(), knobs: vec![k], cands: c, regime, ev_ms: 0.05 });
                    }
                }
            }
        }
        // targets (per the probe's consulted map), INERT (not consulted), no candidates
        let mut plan: Vec<(Family, Target)> = Vec::new();
        for f in fams {
            let rname = if regime == 1 { format!("{} [sparse]", f.name) } else { f.name.clone() };
            if f.cands.is_empty() {
                self.families.insert(rname, serde_json::json!({"status": "NO VALID CANDIDATE (static pruning)"}));
                continue;
            }
            let ts = self.targets_for(&f.knobs);
            if ts.is_empty() {
                self.families.insert(rname, serde_json::json!({"status": "INERT (not consulted by any unit)"}));
                continue;
            }
            for t in ts { plan.push((f.clone(), t)); }
        }
        let mut items: Vec<tune::PlanItem> = plan.iter().enumerate()
            .map(|(i, (f, t))| tune::PlanItem { name: format!("{i}"), ev_ms: f.ev_ms, est_s: self.est_s(f, *t) }).collect();
        tune::order_by_ev(&mut items);
        items.iter().map(|it| plan[it.name.parse::<usize>().unwrap()].clone()).collect()
    }

    // ---- the race sampler: fresh SR rounds with arms [base] + the active candidates

    fn obs_for(&self, samples: &[Sample], offs: &PosTable, phase: usize, target: Target, arms: usize, regime: u8)
        -> Vec<Vec<tune::Obs>> {
        pair_obs(samples, offs, phase, target, arms, self.k, regime)
    }

    fn sample(&mut self, fam: &Family, target: Target, base: &[(&'static tune::TunableDef, Option<tune::Key>, i32)],
              active: &[usize], want: usize, confirm: bool) -> Result<Vec<tune::Sampled>> {
        let mut arms = vec![Arm { name: "base".into(), assign: base.to_vec() }];
        for &c in active {
            let mut asg = base.to_vec();
            for (i, kn) in fam.knobs.iter().enumerate() { asg.push((kn.def, kn.key, fam.cands[c][i])); }
            arms.push(Arm { name: format!("{}={:?}", fam.name, fam.cands[c]), assign: asg });
        }
        let n_arms = arms.len();
        let widths = if confirm && target != Target::Draft { self.widths_all.clone() } else { self.widths_rep.clone() };
        let phase = self.set_arms(arms, target != Target::Draft, target != Target::Verify, widths)?;
        let cap_rounds = want * 3 + 12;
        let mut rounds = 0usize;
        let (mut obs, mut outs): (Vec<Vec<tune::Obs>>, Vec<Option<String>>);
        let stratify = self.stratify;
        loop {
            self.run_rounds(4)?;
            rounds += 4;
            let (samples, xfail, inert, offs): (Vec<Sample>, Vec<String>, Vec<Option<bool>>, PosTable) = with_session(|s| {
                let mine: Vec<Sample> = s.samples.iter().filter(|x| x.phase == phase).cloned().collect();
                // v2: the position model of this phase's cells, fitted over the whole session's
                // replays in those cells (a short phase alone cannot separate arm from position)
                let offs = if stratify {
                    let cells: std::collections::HashSet<PosCell> = mine.iter().filter(|x| x.mode == "sr").map(pos_cell).collect();
                    pos_table(&s.samples, &|c| cells.contains(c))
                } else { PosTable::new() };
                Ok((mine, s.xfail.clone(), (0..n_arms).map(|a| s.arm_inert(a)).collect(), offs))
            })?.context("tune session missing")?;
            obs = self.obs_for(&samples, &offs, phase, target, n_arms, fam.regime);
            outs = vec![None; n_arms];
            for a in 1..n_arms {
                if samples.iter().any(|x| x.arm == a && !x.digest_ok) {
                    let first = xfail.iter().rev().find(|l| l.contains(&format!("'{}=", fam.name)))
                        .or_else(|| xfail.iter().rev().find(|l| l.contains("differs from base")))
                        .cloned().unwrap_or_default();
                    outs[a] = Some(format!("XCHECK FAIL (a kernel bug on live activations): {first}"));
                } else if inert[a] == Some(true) {
                    outs[a] = Some("INERT (identical graph: same kernels, geometry, arguments and edges as the base)".into());
                }
            }
            if samples.iter().any(|x| (x.arm == 0 || x.arm == WARMUP_ARM) && (x.mode == "sr" || x.mode == "warm") && !x.digest_ok) {
                bail!("the BASE replay stopped matching the canonical execution (family {}) — harness failure", fam.name);
            }
            let done = (1..n_arms).all(|a| outs[a].is_some() || obs[a].len() >= want);
            if done || rounds >= cap_rounds { break; }
        }
        Ok((1..n_arms).map(|a| match outs[a].take() {
            Some(why) => tune::Sampled::Out(why),
            None => tune::Sampled::Obs(std::mem::take(&mut obs[a])),
        }).collect())
    }

    /// One family race + its decisions. Returns (adopted, a clock / thermal event happened).
    fn run_family(&mut self, fam: &Family, target: Target, sweep: usize) -> Result<(bool, bool)> {
        let t_fam = std::time::Instant::now();
        let e0 = self.env_snapshot(&format!("family {} [{}] start", fam.name, target.name()));
        let base = self.base_assign(Some(&fam.name));
        // sweep 2: the family's own adopted value becomes a candidate again, the default the base
        let mut fam = fam.clone();
        if sweep > 1 {
            let mine: Vec<i32> = fam.knobs.iter().map(|k| self.decisions.iter()
                .find(|d| d.family == fam.name && d.knob.def.slot == k.def.slot && d.knob.key == k.key && d.target == target)
                .map(|d| d.value).unwrap_or(k.def.default)).collect();
            let dflt: Vec<i32> = fam.knobs.iter().map(|k| k.def.default).collect();
            if mine != dflt && !fam.cands.contains(&mine) { fam.cands.push(mine); }
            fam.cands.retain(|c| *c != dflt);
        }
        let tau = self.tau_for(target, fam.regime);
        let rcfg = self.rcfg;
        let mut err: Option<anyhow::Error> = None;
        let out = {
            let mut sampler = |active: &[usize], want: usize, confirm: bool| -> Vec<tune::Sampled> {
                if err.is_some() { return active.iter().map(|_| tune::Sampled::Out("ERROR".into())).collect(); }
                match self.sample(&fam, target, &base, active, want, confirm) {
                    Ok(v) => v,
                    Err(e) => {
                        err = Some(e);
                        active.iter().map(|_| tune::Sampled::Out("ERROR".into())).collect()
                    }
                }
            };
            tune::race(fam.cands.len(), &rcfg, tau, &mut sampler)
        };
        with_session(|s| { s.clear_graphs(); Ok(()) })?;
        if let Some(e) = err { return Err(e); }
        let e1 = self.env_snapshot(&format!("family {} [{}] end", fam.name, target.name()));
        let drift = match (sm_clock(&e0), sm_clock(&e1)) {
            (Some(a), Some(b)) if a > 0.0 => ((b - a) / a).abs() > 0.05,
            _ => false,
        };
        // a NEW throttle flag during the block (a box that sits power-capped throughout is steady state)
        let thr = throttled(&e1) && !throttled(&e0);
        // receipts: every candidate's fate + estimate
        let cand_json: Vec<serde_json::Value> = out.cands.iter().map(|c| serde_json::json!({
            "value": fam.cands.get(c.idx), "fate": c.fate, "n": c.obs.len(),
            "hl_delta_ms": c.est.map(|e| e.est), "ci95_ms": c.est.map(|e| [e.lo, e.hi]) })).collect();
        let _ = writeln!(self.jsonl, "{}", serde_json::json!({"family": fam.name, "target": target.name(), "sweep": sweep,
            "regime": fam.regime, "tau_ms": tau, "verdict": out.verdict, "candidates": cand_json,
            "confirm": out.confirm.map(|e| serde_json::json!({"n": e.n, "hl": e.est, "ci95": [e.lo, e.hi]})),
            "calls": out.calls, "seconds": t_fam.elapsed().as_secs_f64(), "clock_drift": drift, "throttled": thr }));
        let key = format!("{}{} [{}]", fam.name, if fam.regime == 1 { " (sparse)" } else { "" }, target.name());
        let lead = out.leader.and_then(|i| out.cands[i].est.map(|e| (fam.cands[i].clone(), e)));
        let mut line = format!("FAMILY {key} sweep {sweep}: {} | tau {tau:.4} ms | {} candidate(s) | {:.1} s",
                               out.verdict, fam.cands.len(), t_fam.elapsed().as_secs_f64());
        if let Some((v, e)) = &lead { line += &format!(" | leader {v:?} HL {:+.4} CI95 [{:+.4}, {:+.4}] n {}", e.est, e.lo, e.hi, e.n); }
        if let Some(c) = out.confirm { line += &format!(" | confirm HL {:+.4} CI95 [{:+.4}, {:+.4}] n {}", c.est, c.lo, c.hi, c.n); }
        for c in &out.cands {
            if c.fate.starts_with("XCHECK") || c.fate.starts_with("INERT") {
                line += &format!(" | {:?}: {}", fam.cands[c.idx], c.fate);
            }
        }
        if drift || thr { line += " | CLOCK/THERMAL EVENT during the family (flagged in the receipts)"; }
        self.say(line);
        // decisions
        let mut adopted = false;
        // a re-race replaces this family's previous decision for this target either way
        self.decisions.retain(|d| !(d.family == fam.name && d.target == target));
        if let Some(w) = out.winner {
            let e = out.cands[w].est.unwrap();
            let conf = out.confirm.unwrap();
            let values = fam.cands[w].clone();
            let runner = out.cands.iter().filter(|c| c.idx != w && c.est.is_some())
                .min_by(|a, b| a.est.unwrap().est.partial_cmp(&b.est.unwrap().est).unwrap())
                .map(|c| serde_json::json!({"value": fam.cands[c.idx], "hl_delta_ms": c.est.unwrap().est}));
            let split = tune::width_split(&out.confirm_obs);
            let rg = if fam.regime == 1 { "sparse" } else { "dense" };
            let base_ms = match target {
                Target::Verify => self.unit_ms("verify", fam.regime),
                Target::Draft => self.unit_ms("draft", fam.regime),
                Target::Round => self.unit_ms("verify", fam.regime) + self.unit_ms("draft", fam.regime),
            };
            for (i, kn) in fam.knobs.iter().enumerate() {
                let (scopes, keys): (Vec<tune::DecScope>, Vec<tune::Key>) = match (target, kn.def.scope) {
                    (Target::Verify, tune::Scope::Width) => {
                        let mut ms: Vec<u16> = self.widths_all.iter().map(|&m| m as u16)
                            .filter(|m| split.iter().find(|x| x.0 == *m).map_or(true, |x| x.2)).collect();
                        ms.sort_unstable();
                        let sc = tune::DecScope { fam: Some("Verify".into()), m: Some(ms.clone()), regime: Some(rg.into()), ..Default::default() };
                        let keys = ms.iter().map(|&m| tune::Key::W(Fam::Verify as u8, m, fam.regime)).collect();
                        (vec![sc], keys)
                    }
                    (Target::Verify, tune::Scope::Fam) => (vec![tune::DecScope { fam: Some("Verify".into()), ..Default::default() }],
                                                           vec![tune::Key::F(Fam::Verify as u8)]),
                    (Target::Draft, tune::Scope::Width) => (
                        [Fam::DraftChain, Fam::DraftPass].iter().map(|f| tune::DecScope { fam: Some(f.name().into()), m: Some(vec![1]),
                            regime: Some(rg.into()), ..Default::default() }).collect(),
                        vec![tune::Key::W(Fam::DraftChain as u8, 1, fam.regime), tune::Key::W(Fam::DraftPass as u8, 1, fam.regime)]),
                    (Target::Draft, tune::Scope::Fam) => (
                        [Fam::DraftChain, Fam::DraftPass].iter().map(|f| tune::DecScope { fam: Some(f.name().into()), ..Default::default() }).collect(),
                        vec![tune::Key::F(Fam::DraftChain as u8), tune::Key::F(Fam::DraftPass as u8)]),
                    (_, tune::Scope::Shape) => match kn.key {
                        Some(tune::Key::S(k, n, b)) => (vec![tune::DecScope { k: Some(k), n: Some(n), bits: Some(b), ..Default::default() }],
                                                        vec![tune::Key::S(k, n, b)]),
                        _ => continue,
                    },
                    (_, tune::Scope::Global) => (vec![tune::DecScope::default()], vec![tune::Key::G]),
                    _ => continue,
                };
                // boot drops a decision any of whose keys fails valid() (and a Width scope with no m):
                // adopt exactly what boot will load, per scope
                let v = values[i];
                let scopes: Vec<tune::DecScope> = scopes.into_iter().filter(|sc| {
                    match tune::decision_keys(kn.def, sc) {
                        Ok(ks) => !ks.is_empty() && ks.iter().all(|k| tune::key_valid(kn.def, k, v)),
                        Err(_) => false,
                    }
                }).collect();
                let keys: Vec<tune::Key> = keys.into_iter().filter(|k| tune::key_valid(kn.def, k, v)).collect();
                if scopes.is_empty() || keys.is_empty() {
                    self.say(format!("FAMILY {key}: {}={v} adopted but invalid at every key of this target (valid()) — not recorded",
                                     kn.def.id));
                    continue;
                }
                let ev = serde_json::json!({
                    "family": fam.name, "unit": target.name(), "n": e.n, "hl_delta_ms": e.est, "ci95_ms": [e.lo, e.hi],
                    "tau_ms": tau, "base_ms": base_ms, "runner_up": runner, "digest_ok": e.n + conf.n,
                    "confirm": {"n": conf.n, "hl_delta_ms": conf.est, "ci95_ms": [conf.lo, conf.hi]},
                    "widths": split.iter().map(|(m, x, keep)| serde_json::json!({"m": m, "n": x.map(|x| x.n),
                        "hl": x.map(|x| x.est), "keep": keep})).collect::<Vec<_>>(),
                    "regime": rg, "sweep": sweep, "joint": fam.knobs.len() > 1,
                });
                let gain = match target {
                    Target::Draft => conf.est * DRAFT_PASSES_PER_ROUND / self.k.max(1) as f64,
                    _ => conf.est,
                } / fam.knobs.len() as f64;
                self.decisions.push(Dec { family: fam.name.clone(), target, knob: kn.clone(), value: values[i], scopes, keys,
                                          evidence: ev, gain_ms: gain });
                adopted = true;
            }
        }
        let status = if adopted { format!("TUNED: {} (sweep {sweep})", out.verdict) } else { out.verdict.clone() };
        self.families.insert(key, serde_json::json!({"status": status, "tau_ms": tau,
            "candidates": out.cands.iter().map(|c| serde_json::json!({"value": fam.cands.get(c.idx), "fate": c.fate,
                "hl": c.est.map(|e| e.est), "ci95": c.est.map(|e| [e.lo, e.hi]), "n": c.obs.len()})).collect::<Vec<_>>(),
            "seconds": t_fam.elapsed().as_secs_f64(), "clock_or_thermal_event": drift || thr }));
        Ok((adopted, drift || thr))
    }

    /// §5.4: a family whose block saw the SM clock move > 5% or a throttle flag is re-run once.
    fn run_family_checked(&mut self, fam: &Family, target: Target, sweep: usize) -> Result<()> {
        let (_, event) = self.run_family(fam, target, sweep)?;
        if event {
            self.say(format!("FAMILY {} [{}]: clock / thermal event — discarded and re-run once (§5.4)", fam.name, target.name()));
            let (_, again) = self.run_family(fam, target, sweep)?;
            if again {
                self.say(format!("FAMILY {} [{}]: the event repeated — decision kept but flagged", fam.name, target.name()));
            }
        }
        Ok(())
    }

    /// Sweep 1 over the plan in order, then sweep 2 (+) over the adopted families until nothing
    /// changes (§6.2), all within the budget.
    fn search(&mut self, plan: Vec<(Family, Target)>, max_sweeps: usize) -> Result<()> {
        let n = plan.len();
        let mut skipped = 0usize;
        for (f, t) in plan.iter() {
            // §6.6: a family that no longer fits is left UNTUNED (budget); a cheaper later one may still fit
            let est = self.est_s(f, *t);
            if self.elapsed_s() + est > self.budget_s {
                let key = format!("{}{} [{}]", f.name, if f.regime == 1 { " (sparse)" } else { "" }, t.name());
                self.families.insert(key, serde_json::json!({"status": "UNTUNED (budget)", "est_s": est}));
                skipped += 1;
                continue;
            }
            self.tenants_ok()?;
            self.run_family_checked(f, *t, 1)?;
        }
        if skipped > 0 {
            self.say(format!("BUDGET: {skipped} of {n} families left UNTUNED (budget {:.0} s, spent {:.0} s)",
                             self.budget_s, self.elapsed_s()));
        }
        for sweep in 2..=max_sweeps {
            let again: Vec<(Family, Target)> = plan.iter().filter(|(f, t)| self.decisions.iter()
                .any(|d| d.family == f.name && d.target == *t)).cloned().collect();
            if again.is_empty() { break; }
            let snapshot = |t: &Self| -> Vec<(String, i32)> {
                let mut v: Vec<(String, i32)> = t.decisions.iter()
                    .map(|d| (format!("{}|{:?}|{:?}|{}", d.family, d.target, d.knob.key, d.knob.def.id), d.value)).collect();
                v.sort();
                v
            };
            let before = snapshot(self);
            for (f, t) in &again {
                if self.elapsed_s() + self.est_s(f, *t) > self.budget_s {
                    self.say(format!("BUDGET: sweep {sweep} stopped before {} (budget)", f.name));
                    return Ok(());
                }
                self.tenants_ok()?;
                self.run_family_checked(f, *t, sweep)?;
            }
            let after = snapshot(self);
            if before == after {
                self.say(format!("SWEEP {sweep}: no decision changed — converged"));
                break;
            }
            self.say(format!("SWEEP {sweep}: decisions changed ({} -> {}) — another sweep", before.len(), after.len()));
        }
        Ok(())
    }

    // ---- the prefill point (§5.1 prefill(c), §5.7.4 the B3 known answer)

    fn prefill_phase(&mut self, tiled: &[u32]) -> Result<()> {
        let def = &T_WIDE_MT;
        if let Some((_, env, v)) = tune::env_overrides().into_iter().find(|(id, _, _)| *id == def.id) {
            self.families.insert("prefill.wide_mt".into(),
                serde_json::json!({"status": format!("NOT SEARCHED: OVERRIDDEN(flag {env}={v}) — the override flag wins over every arm")}));
            return Ok(());
        }
        if !self.model.prefix_snapshots() {
            self.families.insert("prefill.wide_mt".into(), serde_json::json!({"status": "UNTUNED (no prefix snapshots: --prefix-cache off posture)"}));
            return Ok(());
        }
        let mut sizes: Vec<usize> = [26usize, 72, 160, 512, self.chunk].iter().copied().filter(|&c| c > 16 && c <= self.chunk).collect();
        sizes.sort_unstable();
        sizes.dedup();
        let p0 = 64usize;
        let mut known: Vec<String> = Vec::new();
        let mut deviate: Vec<String> = Vec::new();
        for c in sizes {
            if tiled.len() < p0 + c { continue; }
            let est = if c >= 2048 { 60.0 } else { 12.0 };
            // the prefill point has its own 2-minute reserve beyond the search budget
            if self.elapsed_s() + est > self.budget_s + 120.0 {
                self.families.insert(format!("prefill.wide_mt c={c}"), serde_json::json!({"status": "UNTUNED (budget)"}));
                continue;
            }
            self.tenants_ok()?;
            let prefix: Vec<i32> = tiled[..p0].iter().map(|&t| t as i32).collect();
            let toks: Vec<i32> = tiled[p0..p0 + c].iter().map(|&t| t as i32).collect();
            // the chunk's pre-state: a real prefix, snapshotted (restored before every replay)
            self.model.reset_slot(0)?;
            {
                let psc = self.psc.as_mut().context("prefill scratch")?;
                self.model.prefill_chunk(psc, &prefix, 0, 0, false, None)?;
                self.model.snapshot_slot(0, psc)?;
            }
            self.model.dev.synchronize()?;
            let class = tune::chunk_class(c);
            // p5e: the default (0) IS the PFX1 rule (B3's MT by chunk rows for the MoE expert GEMMs);
            // a fixed MT equal to the rule's value at this c launches exactly the base's kernels, so
            // it is not a candidate
            let rule_mt = pfx1_moe_mt(c) as i32;
            let cands: Vec<i32> = def.domain.iter().copied().filter(|&v| v != def.default && v != rule_mt).collect();
            let arms: Vec<i32> = std::iter::once(def.default).chain(cands.iter().copied()).collect();
            let mut regs: Vec<Reg> = Vec::new();
            for kc in self.model.k_cache.iter() { kv_rows(self.model, dptr(kc), 0, p0, c, &mut regs)?; }
            let mut err: Option<anyhow::Error> = None;
            let rcfg = if c >= 1024 { tune::RaceCfg { n_stage1: 6, n_max: 8, n_confirm: 6, step: 2 } }
                       else { tune::RaceCfg { n_stage1: 6, n_max: 12, n_confirm: 6, step: 3 } };
            let mut base_times: Vec<f64> = Vec::new();
            // v2: timed replays per block size for the position model; (warm-up, by-position) medians
            let mut hist: HashMap<usize, Vec<tune::PosObs>> = HashMap::new();
            let mut warm_times: Vec<f64> = Vec::new();
            let mut rep_id: u64 = 0;
            let (warmup, stratify) = (self.warmup, self.stratify);
            let mut porder = std::mem::take(&mut self.pf_order);
            let out = {
                let mut sampler = |active: &[usize], want: usize, _confirm: bool| -> Vec<tune::Sampled> {
                    let mut res: Vec<Vec<tune::Obs>> = vec![Vec::new(); active.len()];
                    let mut bad: Vec<Option<String>> = vec![None; active.len()];
                    if err.is_some() { return active.iter().map(|_| tune::Sampled::Out("ERROR".into())).collect(); }
                    // one timed replay of arm `a` (usize::MAX = the base) from the restored pre-state
                    let mut replay = |a: usize| -> Result<(f64, u64)> {
                        let v = if a == usize::MAX { arms[0] } else { arms[a + 1] };
                        {
                            let psc = self.psc.as_mut().context("prefill scratch")?;
                            self.model.restore_slot(0, psc)?;
                        }
                        self.model.dev.synchronize()?;
                        let t0 = std::time::Instant::now();
                        {
                            let _ov = if v != def.default { Some(tune::overlay(&[(def, None, v)])?) } else { None };
                            let psc = self.psc.as_mut().context("prefill scratch")?;
                            self.model.prefill_chunk(psc, &toks, p0, 0, false, None)?;
                        }
                        self.model.dev.synchronize()?;
                        let ms = t0.elapsed().as_secs_f64() * 1e3;
                        let dg = with_session(|s| s.digest_now(self.model, &regs))?.unwrap_or(0);
                        Ok((ms, dg))
                    };
                    let npos = active.len() + 1;
                    let r = (|| -> Result<()> {
                        // v2: the call starts after host-side race work (and, the first time, a family
                        // boundary): untimed base warm-up replay(s) first — every timed arm is then
                        // preceded by a replay + restore + sync, whatever its place
                        let mut warm_dg: Option<u64> = None;
                        for _ in 0..warmup {
                            let (ms, dg) = replay(usize::MAX)?;
                            warm_times.push(ms);
                            warm_dg = Some(dg);
                        }
                        let mut reps: Vec<HashMap<usize, (f64, u64, usize)>> = Vec::new();
                        for _ in 0..want {
                            // v2: the balanced order (Williams over base + active, per chunk and block
                            // size, continuing across race calls); row index 0 = the base
                            let row = porder.next(c as u64, npos);
                            let mut t: HashMap<usize, (f64, u64, usize)> = HashMap::new();
                            for (place, &ix) in row.iter().enumerate() {
                                let a = if ix == 0 { usize::MAX } else { active[ix - 1] };
                                let (ms, dg) = replay(a)?;
                                t.insert(a, (ms, dg, place));
                            }
                            rep_id += 1;
                            let h = hist.entry(npos).or_default();
                            for (&a, &(ms, _, place)) in t.iter() {
                                let v = if a == usize::MAX { arms[0] } else { arms[a + 1] };
                                h.push(tune::PosObs { round: rep_id, arm: v as u32 as u64, pos: place, ms });
                            }
                            reps.push(t);
                        }
                        let f = match hist.get(&npos) {
                            Some(h) if stratify => tune::position_offsets(h, npos),
                            _ => vec![0.0; npos],
                        };
                        for t in &reps {
                            let (tb, db, pb) = t[&usize::MAX];
                            base_times.push(tb);
                            if let Some(wd) = warm_dg {
                                anyhow::ensure!(wd == db, "prefill c={c}: the base chunk is not deterministic (warm-up KV digest \
                                                           {wd:016x} != base {db:016x}) — harness failure");
                            }
                            for (j, &a) in active.iter().enumerate() {
                                let (ta, da, pa) = t[&a];
                                if da != db {
                                    bad[j] = Some(format!("XCHECK FAIL: prefill c={c} MT={} KV digest {da:016x} != base {db:016x}", arms[a + 1]));
                                }
                                res[j].push((c.min(u16::MAX as usize) as u16, (ta - f[pa]) - (tb - f[pb])));
                            }
                        }
                        Ok(())
                    })();
                    if let Err(e) = r { err = Some(e); }
                    res.into_iter().zip(bad).map(|(o, b)| match b { Some(w) => tune::Sampled::Out(w), None => tune::Sampled::Obs(o) }).collect()
                };
                // tau from a first estimate of the base chunk time (0.1% of it, >= 0.02 ms)
                let tau = tune::tau_ms(if c >= 1024 { 2000.0 } else { 300.0 });
                tune::race(cands.len(), &rcfg, tau, &mut sampler)
            };
            self.pf_order = porder;
            if let Some(e) = err { return Err(e); }
            self.model.reset_slot(0)?;
            // v2: the prefill point's position effect (per block size), for the family record
            let pf_pos: Vec<serde_json::Value> = {
                let mut ks: Vec<usize> = hist.keys().copied().collect();
                ks.sort_unstable();
                ks.into_iter().map(|np| {
                    let h = &hist[&np];
                    let by: Vec<Option<f64>> = (0..np).map(|p| tune::median(&h.iter().filter(|o| o.pos == p)
                        .map(|o| o.ms).collect::<Vec<_>>())).collect();
                    serde_json::json!({"n_pos": np, "reps": h.len() / np.max(1), "median_ms_by_pos": by,
                                       "offsets_ms": tune::position_offsets(h, np)})
                }).collect()
            };
            let base_med = tune::median(&base_times).unwrap_or(f64::NAN);
            // the effective MT after the race: the adopted fixed MT, else the rule's value at c
            let pick = out.winner.map(|w| cands[w]).unwrap_or(rule_mt);
            let lead = out.leader.and_then(|i| out.cands[i].est.map(|e| (cands[i], e)));
            let mut line = format!("PREFILL c={c} (class {class}): base = the PFX1 rule (MT {rule_mt}) {:.2} ms | {} -> MT {pick}",
                                   base_med, out.verdict);
            if let Some((v, e)) = lead { line += &format!(" | leader MT {v} HL {:+.3} ms CI95 [{:+.3}, {:+.3}] n {}", e.est, e.lo, e.hi, e.n); }
            for cc in &out.cands { if cc.fate.starts_with("XCHECK") { line += &format!(" | MT {}: {}", cands[cc.idx], cc.fate); } }
            self.say(line);
            known.push(format!("c={c}->{pick}"));
            if pick != rule_mt { deviate.push(format!("c={c}: MT {pick} beat the rule's MT {rule_mt}")); }
            self.families.insert(format!("prefill.wide_mt c={c}"), serde_json::json!({"status": out.verdict, "base_ms": base_med,
                "pick": pick, "candidates": out.cands.iter().map(|x| serde_json::json!({"mt": cands[x.idx], "fate": x.fate,
                    "hl": x.est.map(|e| e.est), "ci95": x.est.map(|e| [e.lo, e.hi]), "n": x.obs.len()})).collect::<Vec<_>>(),
                "warmup_median_ms": tune::median(&warm_times), "position": pf_pos}));
            if let (Some(w), Some(conf)) = (out.winner, out.confirm) {
                let e = out.cands[w].est.unwrap();
                self.decisions.push(Dec {
                    family: format!("prefill.wide_mt c={c}"), target: Target::Round,
                    knob: Knob { def, key: None, label: def.id.into() }, value: cands[w],
                    scopes: vec![tune::DecScope { chunk: Some(vec![class]), regime: Some("dense".into()), ..Default::default() }],
                    keys: vec![tune::Key::C(class, 0)],
                    evidence: serde_json::json!({"unit": format!("prefill(c={c})"), "n": e.n, "hl_delta_ms": e.est,
                        "ci95_ms": [e.lo, e.hi], "base_ms": base_med, "digest_ok": e.n + conf.n,
                        "confirm": {"n": conf.n, "hl_delta_ms": conf.est, "ci95_ms": [conf.lo, conf.hi]}}),
                    gain_ms: 0.0, // TTFT, not a decode round: outside the round prediction check
                });
            }
        }
        // p5e: the served default IS B3's rule (PFX1 (a): MT 1 for c <= 112, 2 below 384, 4 from 384), so
        // the known answer is that no fixed MT beats it by tau at any chunk size
        self.say(format!("KNOWN ANSWER (B3, §5.7.4): MT by chunk {} — expected the PFX1 rule (MT 1 up to c ~100, 2 for \
                          ~100-350, 4 from 512) to stand: {}", known.join(" "),
                         if deviate.is_empty() { "HOLDS (no fixed MT beat the rule by tau)".to_string() }
                         else { format!("DEVIATES ({}) — a finding for the B3 ledger, flagged", deviate.join("; ")) }));
        self.known_answer = Some(deviate.is_empty());
        Ok(())
    }

    // ---- the final gate (§6.5): LI whole-round A/B + the automatic hash gate

    /// One served DDS round on slot 0 (exl3_serve::step_mtp's DDS branch: the calibrator's keep,
    /// the WP23 target, labels, decay, the lane EMAs and cost fit). `det_cost` feeds the WP23 guard
    /// a deterministic per-round cost (29 + 6 ms/draft, §6.6's fit) instead of the wall clock, so
    /// two arms' draft decisions are comparable (the hash gate); the LI A/B uses the wall clock.
    fn dds_round(&mut self, lane: &mut DdsLane, b: i32, p: usize, tap_row: usize, det_cost: bool,
                 samp: Option<(f32, f32, u64)>) -> Result<(MtpRoundOut, f64, bool)> {
        let k = self.k;
        let k_eff = k.min(self.model.max_pos.saturating_sub(p + 2));
        anyhow::ensure!(k_eff >= 1, "dds_round at the end of the window");
        let rows: Vec<Option<SampParams>> = (0..k + 1).map(|i| samp.and_then(|(t, tp, seed)|
            SampParams::sampled(t, tp, 0, 0.0, seed, lane.ctr.wrapping_add(i as u32)))).collect();
        self.model.set_sampling(&mut self.sc, &rows)?;
        let pens: Vec<Option<PenParams>> = vec![None; k + 1];
        self.model.set_penalties(&mut self.sc, &pens)?;
        let guard = crate::exl3_serve::wp23_guard_on(k);
        let target = crate::exl3_serve::wp23_target(self.draft_conf, guard, lane.rounds, lane.fit.slope(), lane.ema_tok, lane.ema_ms);
        let g0 = self.model.graph_count();
        let t0 = std::time::Instant::now();
        let (o, confs) = {
            let cal = &lane.cal;
            let mut reach = 1.0f64;
            let mut keep = |i: usize, c: f32| { reach *= cal.estimate_at(i, c); reach >= target };
            self.model.mtp_round_adaptive(&mut self.sc, self.head, 0, k_eff, b, p, tap_row, &mut keep, None)?
        };
        self.model.dev.synchronize()?; // retire the async re-prime: the round's whole cost
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let captured = self.model.graph_count() != g0;
        for (i, &c) in confs.iter().enumerate() {
            if i < o.a { lane.cal.add_label(i, c, true); } else { lane.cal.add_label(i, c, false); break; }
        }
        lane.cal.decay_step();
        lane.ctr = lane.ctr.wrapping_add((k + 1) as u32);
        lane.rounds += 1;
        lane.drafted += o.drafts.len();
        lane.accepted += o.a;
        let toks = (o.a + 1) as f64;
        let w = o.drafts.len();
        let cost = if det_cost { 29.0 + 6.0 * w as f64 } else { ms };
        if lane.rounds > 4 {
            if lane.ema_tok <= 0.0 { lane.ema_ms = cost; lane.ema_tok = toks; }
            else { lane.ema_ms = 0.9375 * lane.ema_ms + 0.0625 * cost; lane.ema_tok = 0.9375 * lane.ema_tok + 0.0625 * toks; }
            if (det_cost || !captured) && (lane.ema_ms <= 0.0 || cost <= 3.0 * lane.ema_ms) { lane.fit.add(w as f64, cost); }
        }
        Ok((o, ms, captured))
    }

    /// A fresh generation on slot 0: positional planes zeroed over the whole span it can reach
    /// (every arm and every repeat starts from the same device state), the served seam prime.
    fn setup_gen(&mut self, ids: &[u32], max_new: usize) -> Result<(i32, usize)> {
        let pred0 = pass_setup(self.model, &mut self.sc, ids, max_new, self.chunk, &mut self.psc)?;
        self.model.mtp_seam_prime(&mut self.sc, self.psc.as_ref(), self.head, *ids.last().unwrap() as i32, ids.len(), 0)?;
        Ok((pred0 as i32, ids.len()))
    }

    /// One whole generation in DDS (det_cost) or fixed-chain mode — the hash gate's unit.
    fn gen_once(&mut self, ids: &[u32], max_new: usize, dds: bool) -> Result<GenOut> {
        let (mut b, mut p) = self.setup_gen(ids, max_new)?;
        let mut out = GenOut { tokens: vec![b as u32], rounds: 0, drafted: 0, accepted: 0 };
        let mut tap_row = 0usize;
        let mut lane = DdsLane::new();
        while !EOS_IDS.contains(&b) && out.tokens.len() < max_new && p + self.k + 2 < self.model.max_pos {
            let r = if dds {
                self.dds_round(&mut lane, b, p, tap_row, true, None)?.0
            } else {
                let rows: Vec<Option<SampParams>> = vec![None; self.k + 1];
                self.model.set_sampling(&mut self.sc, &rows)?;
                self.model.mtp_round(&mut self.sc, self.head, 0, self.k, b, p, tap_row)?
            };
            out.rounds += 1;
            out.drafted += r.drafts.len();
            out.accepted += r.a;
            out.tokens.extend(r.emitted.iter().map(|&e| e as u32));
            b = r.last_tok;
            p = r.pos_next;
            tap_row = r.tap_row;
        }
        Ok(out)
    }

    /// Run `f` with arm B in force (the table overlay + B's graph set swapped into the served map)
    /// or arm A (defaults, the served map as is).
    fn with_arm<T>(&mut self, b_arm: bool, map_b: &mut HashMap<usize, crate::gpu::CudaGraph>,
                   assign: &[(&'static tune::TunableDef, Option<tune::Key>, i32)],
                   f: &mut dyn FnMut(&mut Self) -> Result<T>) -> Result<T> {
        if !b_arm || assign.is_empty() { return f(self); }
        std::mem::swap(&mut *self.model.graphs.lock().unwrap(), map_b);
        let r = {
            let _ov = tune::overlay(assign)?;
            f(self)
        };
        std::mem::swap(&mut *self.model.graphs.lock().unwrap(), map_b);
        r
    }

    fn final_gate(&mut self, li_rounds: usize, hash_tokens: usize, prompts: &[(&'static str, Vec<u32>, bool)])
        -> Result<(bool, serde_json::Value)> {
        // exactly what boot will put in force: the table's decisions through build_decisions
        let probe = tune::Table { format: tune::TABLE_FORMAT.into(), origin: "user".into(), fingerprint: Default::default(),
                                  posture: Default::default(), provenance: serde_json::Value::Null,
                                  decisions: self.table_decisions(), families: serde_json::Value::Null,
                                  confirm: serde_json::Value::Null, table_sha256: String::new() };
        let (assign, ignored) = tune::table_assign(&probe, false);
        if !ignored.is_empty() {
            self.say(format!("FINAL GATE: WARNING {} decision(s) boot would ignore: {}", ignored.len(), ignored.join("; ")));
        }
        if assign.is_empty() {
            // L10: a clean 0-decision run IS a passed gate (the table is the built-in defaults: the
            // A/B has no B arm and the hashes are identical by construction) — it must say so, or
            // boot's auto selection refuses it as "unconfirmed". Not clean = decisions boot would
            // ignore (the table claims something it cannot put in force) or a DEVIATING known answer.
            let known = match self.known_answer { None => "not run", Some(true) => "holds", Some(false) => "deviates" };
            let clean = ignored.is_empty() && self.known_answer != Some(false);
            let reason = if clean {
                "0 decisions: the table is the built-in defaults (no B arm to A/B; hashes identical by construction)".to_string()
            } else if !ignored.is_empty() {
                format!("0 decisions in force but {} decision(s) boot would ignore — the table is not what it claims", ignored.len())
            } else {
                "0 decisions but the B3 known answer DEVIATES — the run is not clean".to_string()
            };
            self.say(format!("FINAL GATE: no decision adopted — {} ({reason}; known answer {known})",
                             if clean { "PASS" } else { "FAIL" }));
            return Ok((clean, serde_json::json!({"skipped": "no decision adopted", "pass": clean, "reason": reason,
                                                 "known_answer": known, "hashes_identical": true,
                                                 "ignored": ignored})));
        }
        // decode decisions are judged by the whole-round A/B; prefill (Chunk) decisions change no
        // decode round — they were confirmed by their own paired race and ride the hash gate
        let decode_decisions = self.decisions.iter().any(|d| !d.family.starts_with("prefill."));
        // graph sets: A = the served defaults, B = the table (both captured capture-only, slot 0)
        let t_cap = std::time::Instant::now();
        self.model.precapture_decode_graphs(&mut self.sc, 1, self.k, self.draft_conf > 0.0)?;
        let mut map_b: HashMap<usize, crate::gpu::CudaGraph> = HashMap::new();
        {
            let a = assign.clone();
            let dds = self.draft_conf > 0.0;
            self.with_arm(true, &mut map_b, &a, &mut |t: &mut Self| t.model.precapture_decode_graphs(&mut t.sc, 1, t.k, dds))?;
        }
        self.say(format!("FINAL GATE: graph sets captured (defaults + table, slot 0) in {:.1} s", t_cap.elapsed().as_secs_f64()));
        // ---- LI whole-round A/B, served posture (DDS + the WP23 guard on the wall clock)
        let mut li = serde_json::Map::new();
        let mut pass_li = true;
        let mut sum_est = 0.0f64;
        let mut measured_ansic: Option<f64> = None;
        for (name, ids, sampled) in prompts.iter().filter(|x| x.0 != "python") {
            if !decode_decisions { break; }
            self.tenants_ok()?;
            let samp = if *sampled { Some((1.0f32, 0.95f32, 0x5C1F1u64)) } else { None };
            // v2: a deterministic Williams schedule (n = 2: A B | B A | ... — every arm in every
            // in-block position and after each arm equally often) instead of random blocks
            let mut sched = LiSched::new(2, ids.len() as u64);
            let (mut t_a, mut t_b): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
            // (arm, in-block position, ms) of every timed round, for the position view
            let mut placed: Vec<(usize, u8, f64)> = Vec::new();
            let mut warm_ms: Vec<f64> = Vec::new();
            let mut gen: Option<(i32, usize, usize, usize, DdsLane)> = None;
            let mut guard_rounds = 0usize;
            let mut warm_left = 0usize;
            while (t_a.len() < li_rounds || t_b.len() < li_rounds) && guard_rounds < 6 * li_rounds + 64 {
                guard_rounds += 1;
                let fresh = match &gen { None => true, Some((b, p, _, e, _)) =>
                    EOS_IDS.contains(b) || *e >= self.max_new.max(1024) || p + self.k + 2 >= self.model.max_pos };
                if fresh {
                    let (b, p) = self.setup_gen(ids, self.max_new.max(1024))?;
                    gen = Some((b, p, 0, 1, DdsLane::new()));
                    // v2: the first round(s) after a fresh generation (prefill + seam prime) are an
                    // untimed warm-up under the defaults — a cold start no arm should pay
                    warm_left = self.warmup;
                }
                let (b, p, tap_row, emitted, mut lane) = gen.take().unwrap();
                let warm = warm_left > 0;
                let (arm, apos) = if warm { warm_left -= 1; (0usize, NO_APOS) } else { sched.next() };
                let r = self.with_arm(arm == 1, &mut map_b, &assign, &mut |t: &mut Self| t.dds_round(&mut lane, b, p, tap_row, false, samp))?;
                let (o, ms, captured) = r;
                if warm {
                    warm_ms.push(ms);
                } else if !captured {
                    if arm == 1 { t_b.push(ms) } else { t_a.push(ms) }
                    placed.push((arm, apos, ms));
                }
                gen = Some((o.last_tok, o.pos_next, o.tap_row, emitted + o.emitted.len(), lane));
            }
            let e = tune::hl_two_sample(&t_a, &t_b);
            let (ma, mb) = (tune::median(&t_a).unwrap_or(f64::NAN), tune::median(&t_b).unwrap_or(f64::NAN));
            let regress = matches!(e, Some(x) if x.lo > 0.0);
            pass_li &= !regress;
            if let Some(x) = e { sum_est += x.est; }
            if *name == "ansic" { measured_ansic = e.map(|x| x.est); }
            let pos_med = |a: usize| -> Vec<Option<f64>> { (0..2u8).map(|p| tune::median(&placed.iter()
                .filter(|x| x.0 == a && x.1 == p).map(|x| x.2).collect::<Vec<_>>())).collect() };
            let pm = |v: &[Option<f64>]| v.iter().map(|x| x.map_or("-".into(), |x| format!("{x:.2}"))).collect::<Vec<_>>().join(", ");
            let (pa, pb) = (pos_med(0), pos_med(1));
            self.say(format!("FINAL LI A/B {name}: default {ma:.2} ms/round (n {}) | table {mb:.2} ms/round (n {}) | table-default HL {} -> {} \
                              | by in-block position: default [{}] table [{}] | warm-up rounds {} (median {})",
                             t_a.len(), t_b.len(),
                             e.map_or("n/a".into(), |x| format!("{:+.3} CI95 [{:+.3}, {:+.3}]", x.est, x.lo, x.hi)),
                             if regress { "REGRESSION" } else { "ok" }, pm(&pa), pm(&pb), warm_ms.len(),
                             tune::median(&warm_ms).map_or("-".into(), |x| format!("{x:.2}"))));
            li.insert(name.to_string(), serde_json::json!({"default": ma, "table": mb, "n_default": t_a.len(), "n_table": t_b.len(),
                "hl_delta_ms": e.map(|x| x.est), "ci95_ms": e.map(|x| [x.lo, x.hi]), "regression": regress, "sampled": sampled,
                "median_ms_by_block_pos": {"default": pa, "table": pb}, "warmup_rounds": warm_ms.len(),
                "warmup_median_ms": tune::median(&warm_ms)}));
        }
        let beats = !decode_decisions || sum_est < 0.0;
        if !decode_decisions {
            self.say("FINAL LI A/B: skipped — only prefill (chunk-class) decisions were adopted; the hash gate still runs".into());
        }
        // prediction check (§6.5): measured round gain >= 0.5 x the sum of the adopted unit gains
        let predicted: f64 = self.decisions.iter().map(|d| d.gain_ms).sum();
        let pred_ok = match measured_ansic {
            Some(m) if predicted < 0.0 => m <= 0.5 * predicted,
            _ => true,
        };
        self.say(format!("PREDICTION CHECK: measured AnsiC round delta {} vs adopted unit gains {predicted:+.3} ms -> {}",
                         measured_ansic.map_or("n/a".into(), |m| format!("{m:+.3}")),
                         if pred_ok { "OK" } else { "FLAGGED for attribution (the parts do not add up)" }));
        // ---- the automatic hash gate: fixed chain AND DDS (deterministic guard cost), A vs B
        let mut hashes = serde_json::Map::new();
        let mut pass_hash = true;
        for (name, ids, sampled) in prompts.iter() {
            if *sampled { continue; }
            for dds in [false, true] {
                self.tenants_ok()?;
                let a = self.gen_once(ids, hash_tokens, dds)?;
                let bo = self.with_arm(true, &mut map_b, &assign, &mut |t: &mut Self| t.gen_once(ids, hash_tokens, dds))?;
                let same_tok = a.tokens == bo.tokens;
                let same_ctr = (a.rounds, a.drafted, a.accepted) == (bo.rounds, bo.drafted, bo.accepted);
                pass_hash &= same_tok && same_ctr;
                let mode = if dds { "dds" } else { "fixed" };
                self.say(format!("HASH GATE {name} {mode}: default {:016x} ({} tok, rounds {} drafted {} accepted {}) | table {:016x} \
                                  ({} tok, rounds {} drafted {} accepted {}) -> {}",
                                 fnv64(&a.tokens), a.tokens.len(), a.rounds, a.drafted, a.accepted,
                                 fnv64(&bo.tokens), bo.tokens.len(), bo.rounds, bo.drafted, bo.accepted,
                                 if same_tok && same_ctr { "IDENTICAL" } else if same_tok { "TOKENS equal, COUNTERS DIFFER" } else { "DIFFERENT" }));
                hashes.insert(format!("{name}.{mode}"), serde_json::json!({"default": format!("{:016x}", fnv64(&a.tokens)),
                    "table": format!("{:016x}", fnv64(&bo.tokens)), "tokens": a.tokens.len(),
                    "mtp_stats_default": [a.rounds, a.drafted, a.accepted], "mtp_stats_table": [bo.rounds, bo.drafted, bo.accepted],
                    "identical": same_tok && same_ctr}));
            }
        }
        drop(map_b); // table graphs (the served map keeps the defaults)
        let pass = pass_li && beats && pass_hash;
        let conf = serde_json::json!({
            "li_round_ms": li, "beats_defaults": beats, "sum_hl_delta_ms": sum_est, "no_regression": pass_li,
            "prediction_check": {"measured_ansic_ms": measured_ansic, "predicted_ms": predicted, "ok": pred_ok},
            "hashes": hashes, "mtp_stats_identical": pass_hash, "pass": pass,
            "note": "LI = served DDS rounds (wall-clock WP23 guard), arms alternated per round on one trajectory \
                     (v2: a Williams A B | B A schedule; the first round(s) of every generation are an untimed warm-up); \
                     rounds that captured a graph are excluded; control rounds (1/128 plain) are not run",
        });
        Ok((pass, conf))
    }
}


/// Paired observations of every candidate arm (1..arms) against the base (arm 0) in one phase:
/// per round, (m', t_arm - t_base) for the verify unit, (0, draft delta) for the draft unit, or
/// (m', dv + (passes per round / s) * dd) for a Round target. Only digest-clean samples pair; the
/// warm-ups (mode "warm") never do. v2: each time has its cell's fitted position offset removed
/// first when `offs` holds the cell (an empty table = the plain paired difference).
fn pair_obs(samples: &[Sample], offs: &PosTable, phase: usize, target: Target, arms: usize, k: usize, regime: u8)
    -> Vec<Vec<tune::Obs>> {

    let mut by_round: HashMap<usize, (HashMap<usize, (usize, f64)>, HashMap<usize, f64>)> = HashMap::new();
    for s in samples.iter().filter(|s| s.phase == phase && s.mode == "sr" && s.arm < arms && s.digest_ok && s.regime == regime) {
        let e = by_round.entry(s.round).or_default();
        let ms = adj_ms(s, offs);
        match s.unit {
            "verify_dry" => { e.0.insert(s.arm, (s.m, ms)); }
            "draft" => { e.1.insert(s.arm, ms); }
            _ => {}
        }
    }
    let w = DRAFT_PASSES_PER_ROUND / k.max(1) as f64;
    let mut out: Vec<Vec<tune::Obs>> = vec![Vec::new(); arms];
    let mut rounds: Vec<usize> = by_round.keys().copied().collect();
    rounds.sort_unstable();
    for r in rounds {
        let (v, d) = &by_round[&r];
        for a in 1..arms {
            let dv = match (v.get(&0), v.get(&a)) { (Some(b), Some(x)) => Some((x.0, x.1 - b.1)), _ => None };
            let dd = match (d.get(&0), d.get(&a)) { (Some(b), Some(x)) => Some(x - b), _ => None };
            let o = match target {
                Target::Verify => dv.map(|(m, x)| (m as u16, x)),
                Target::Draft => dd.map(|x| (0u16, x)),
                Target::Round => match (dv, dd) { (Some((m, x)), Some(y)) => Some((m as u16, x + w * y)), _ => None },
            };
            if let Some(o) = o { out[a].push(o); }
        }
    }
    out
}

/// `gb10_inference --autotune` (T2). Ok(true) = a table was written; Ok(false) = the gate rejected it.
fn tune_cli(args: &[String]) -> Result<bool> {
    // the box rule first (G-T8)
    if let Err(why) = preflight() {
        println!("AUTOTUNE REFUSED: {why}");
        bail!("preflight refused: {why}");
    }
    let dir = arg(args, "--model-dir").context("--autotune requires --model-dir <EXL3 pack>")?.to_string();
    if !crate::exl3_serve::is_exl3_pack(&dir) { bail!("{dir} is not an EXL3 pack"); }
    let num = |k: &str, d: usize| -> Result<usize> {
        match arg(args, k) { None => Ok(d), Some(v) => v.parse().map_err(|_| anyhow::anyhow!("{k} must be an integer")) }
    };
    // ---- the SERVED posture (the server's own flags and defaults: exl3_serve::run)
    let max_seq_len = num("--max-seq-len", 4096)?;
    let lanes = num("--max-batch", 8)?.max(1);
    let max_pos = max_seq_len + crate::batch::decode_headroom(false);
    let chunk = num("--prefill-chunk", crate::exl3_serve::DEFAULT_PREFILL_CHUNK)?.max(1);
    let draft_conf: f64 = match arg(args, "--draft-confidence") {
        None => 0.4,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--draft-confidence must be a number"))?,
    };
    anyhow::ensure!((0.0..1.0).contains(&draft_conf), "--draft-confidence must be in [0, 1)");
    if let Some(v) = arg(args, "--kv-cache") {
        anyhow::ensure!(matches!(v, "f32" | "f16" | "fp8" | "q8"), "--kv-cache must be f32, f16, fp8 or q8");
        crate::opts::set(crate::opt!("kv-cache"), v);
    }
    if !matches!(arg(args, "--prefix-cache"), Some("off")) { crate::opts::set(crate::opt!("exl3-prefix"), "1"); }
    let k = mtp_depth_default();
    let posture = tune::Posture { lanes, mtp_max_k: k, dds: draft_conf > 0.0, kv_fmt: kv_fmt_from_opts(), max_pos,
                                  prefill_chunk: chunk };
    // ---- the run's own knobs
    let budget_min: f64 = match arg(args, "--autotune-budget") {
        None => 20.0,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--autotune-budget must be minutes"))?,
    };
    let max_new = num("--autotune-max-new", 512)?.max(32);
    let li_rounds = num("--autotune-li-rounds", 120)?.max(16);
    let hash_tokens = num("--autotune-hash-tokens", 1024)?.max(64);
    let max_sweeps = num("--autotune-sweeps", 3)?.max(1);
    let out_dir = arg(args, "--autotune-out").unwrap_or("tune/receipts").to_string();
    let origin = arg(args, "--autotune-origin").unwrap_or("user").to_string();
    anyhow::ensure!(origin == "user" || origin == "shipped", "--autotune-origin must be user or shipped");
    let only: Option<Vec<String>> = arg(args, "--autotune-families").map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let skip_sparse = args.iter().any(|a| a == "--autotune-skip-sparse");
    // the sparse context point: default just past the dense/sparse switch (cheap: ~2K-token prompt);
    // --autotune-sparse-ctx <N> puts it deeper (e.g. 32768: ~33 s per prompt restart, labelled)
    let sparse_ctx: Option<usize> = match arg(args, "--autotune-sparse-ctx") {
        None => None,
        Some(v) => Some(v.parse().map_err(|_| anyhow::anyhow!("--autotune-sparse-ctx must be tokens"))?),
    };
    let skip_prefill = args.iter().any(|a| a == "--autotune-skip-prefill");
    let skip_gate = args.iter().any(|a| a == "--autotune-skip-gate");
    let seed: u64 = match arg(args, "--autotune-seed") {
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--autotune-seed must be an integer"))?,
        None => std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1),
    };
    let profile = match arg(args, "--tune-profile") {
        Some(v) => v.parse::<u32>().map_err(|_| anyhow::anyhow!("--tune-profile must be MHz"))?,
        None => tune::detect_clock_profile().unwrap_or(tune::STOCK_PROFILE_MHZ),
    };
    let (warmup, stratify) = protocol_args(args)?;
    // the measured config = the built-in defaults (a new table is tuned from scratch)
    tune::arm_tuner();
    // §13 instrumentation: bind the device-0 primary context up front (allocation-free) so
    // mem_get_info works at every probe point below (same ctx FwdModel::load reuses).
    // Three-point attribution receipt: pre-boot / post-boot / after-load (the check line below).
    let _ctx_guard = cudarc::driver::CudaDevice::new(0).ok();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let rss_bytes = || -> Option<u64> {
        std::fs::read_to_string("/proc/self/status").ok()?.lines().find(|l| l.starts_with("VmRSS:"))?
            .split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024)
    };
    let mem_probe = |tag: &str| {
        let fmt = |o: Option<u64>| o.map(|b| format!("{:.1} GiB", gib(b))).unwrap_or_else(|| "n/a".into());
        let cu = cudarc::driver::result::mem_get_info().ok();
        println!("AUTOTUNE: mem {tag}: CUDA free {} of {}; MemAvailable {}; process RSS {}",
                 fmt(cu.map(|c| c.0 as u64)), fmt(cu.map(|c| c.1 as u64)),
                 fmt(crate::memwatch::mem_available_bytes()), fmt(rss_bytes()));
    };
    mem_probe("pre-boot");
    tune::boot(&tune::BootReq { sel: tune::TableSel::Off, model_dir: &dir, posture: posture.clone(), profile_mhz: Some(profile),
                                draft_on: false, tp: 1 });
    mem_probe("after tune::boot");
    let overridden = tune::env_overrides();
    std::fs::create_dir_all(&out_dir).with_context(|| format!("mkdir {out_dir}"))?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let jsonl_path = format!("{out_dir}/autotune-{stamp}.jsonl");
    let jsonl = std::io::BufWriter::new(std::fs::File::create(&jsonl_path).with_context(|| format!("create {jsonl_path}"))?);
    let bin_sha = std::fs::read("/proc/self/exe").map(|b| tune::sha256_hex(&b)).unwrap_or_else(|_| "unknown".into());

    // §13 baseline BEFORE the load: on GB10 unified memory cuMemGetInfo's `free` reads ~0
    // whenever a large model is resident (a healthy serve box measures 0.98 by 1-free/total),
    // so the absolute formula can never pass after weights load. The footprint the ceiling
    // means is OUR delta from this post-boot, pre-load baseline (ctx bound above).
    let mem_base = cudarc::driver::result::mem_get_info().ok();
    let avail_base = crate::memwatch::mem_available_bytes();
    let t_load = std::time::Instant::now();
    let model = FwdModel::load(&dir, lanes, max_pos)?;
    let head = model.mtp.as_ref().context("the pack has no mtp.* draft head — the tuner measures MTP rounds")?;
    // §13: refuse above the 80% memory ceiling (unified LPDDR5x is shared with the OS).
    // What the load's footprint IS (2026-09-27 attribution, TUNE_ceiling_bug_RESOLVED.md): the
    // device-side weights/state (what nvidia-smi shows) PLUS the RAM-resident PLE n-gram table —
    // ordinary host Vec pages (~30 GiB on Flash-Next) that nvidia-smi never shows but that lower
    // cuMemGetInfo's free exactly like a device allocation on the unified pool. Serve holds the
    // same table, so this footprint equals serve's. CUDA free is MemFree: it excludes the
    // reclaimable page cache the load just filled (the safetensors), so it reads ~1 GiB on a
    // healthy box — the survivability floor is Linux MemAvailable, which counts that cache.
    let mem_after = cudarc::driver::result::mem_get_info().ok();
    let avail_after = crate::memwatch::mem_available_bytes();
    model.ple_promote_auto()?; // the served residency (CF-P1d: auto decides after the load)
    let (ple_b, ple_mode) = (model.ple_ram_bytes(), model.ple_residency());
    let avail_s = match (avail_base, avail_after) {
        (Some(a0), Some(a1)) => format!("MemAvailable {:.0} -> {:.0} GiB", gib(a0), gib(a1)),
        _ => "MemAvailable n/a".into(),
    };
    match (mem_base, mem_after) {
        (Some((free0, total)), Some((free1, _))) => {
            let used_b = free0.saturating_sub(free1) as u64;
            let used = used_b as f64 / total.max(1) as f64;
            // real consumption = MemAvailable Δ: the CUDA-free Δ (MemFree) also counts the page
            // cache the load's file reads grew, so footprint − PLE overstates the device side
            // (.14 v4 smoke: 61.0 vs nvidia-smi 54.1 GiB; MemAvailable Δ − PLE = 53.2 GiB)
            let real = match (avail_base, avail_after) { (Some(a0), Some(a1)) => Some(a0.saturating_sub(a1)), _ => None };
            let (dev_b, dev_how) = match real {
                Some(r) => (r.saturating_sub(ple_b), "MemAvailable Δ − PLE"),
                None => (used_b.saturating_sub(ple_b), "footprint − PLE"),
            };
            println!("AUTOTUNE: memory after load: footprint {:.1}% of {:.0} GiB (CUDA free {:.0} -> {:.0} GiB; {avail_s}; \
                      PLE RAM table {:.1} GiB [{ple_mode}]; device-side ≈ {dev_how} ≈ {:.1} GiB)",
                     used * 100.0, gib(total as u64), gib(free0 as u64), gib(free1 as u64), gib(ple_b), gib(dev_b));
            if let Some(r) = real {
                let cache = used_b as i64 - r as i64;
                println!("AUTOTUNE: memory attribution: MemAvailable Δ {:.1} GiB = real consumption; the footprint's CUDA-free Δ \
                          ({:.1} GiB) also counts {:+.1} GiB of reclaimable page-cache growth; process RSS {} (host pages: \
                          PLE table + host buffers; nvidia-smi shows only the device side)",
                         gib(r), gib(used_b), cache as f64 / (1u64 << 30) as f64,
                         rss_bytes().map(|b| format!("{:.1} GiB", gib(b))).unwrap_or_else(|| "n/a".into()));
            }
            if used > 0.80 {
                println!("AUTOTUNE REFUSED: the load consumed {:.0}% of device memory (> 80% ceiling, §13)", used * 100.0);
                bail!("memory ceiling");
            }
        }
        _ => println!("AUTOTUNE: memory ceiling check SKIPPED ({} mem_get_info unavailable)",
                      if mem_base.is_none() { "pre-load" } else { "after-load" }),
    }
    // survivability floor: what the OS can still hand out (free + reclaimable cache), not MemFree
    match avail_after {
        Some(a1) if a1 < (8u64 << 30) => {
            println!("AUTOTUNE REFUSED: only {:.1} GiB MemAvailable after the load (< 8 GiB survivability floor, §13; \
                      {avail_s}, PLE RAM table {:.1} GiB [{ple_mode}])", gib(a1), gib(ple_b));
            bail!("memory ceiling");
        }
        Some(_) => {}
        None => println!("AUTOTUNE: survivability floor SKIPPED (/proc/meminfo MemAvailable unavailable)"),
    }
    let sc = FwdModel::scratch(model.dev(), &model.cfg, lanes.max(k + 1))?;
    let mut psc0 = model.prefill_scratch(chunk)?;
    model.prefill_warmup(&mut psc0, 0, &FwdModel::prefill_warmup_widths(chunk))?;
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", dir.trim_end_matches('/')))?;
    let corpus: Vec<Vec<u32>> = match arg(args, "--autotune-corpus") {
        Some(f) => {
            let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(f)?)
                .context("--autotune-corpus must be a JSON array of prompt strings")?;
            texts.iter().map(|t| tok.encode(t, false)).collect::<Result<_>>()?
        }
        None => CORPUS.iter().chain(std::iter::once(&PY_PROMPT)).map(|t| tok.encode(t, false)).collect::<Result<_>>()?,
    };
    anyhow::ensure!(corpus.iter().all(|c| c.len() >= 2), "every corpus prompt needs >= 2 tokens");
    let corpus_sha = tune::sha256_hex(serde_json::to_string(&corpus)?.as_bytes());
    let fp = tune::current_fingerprint(&dir, 1, profile);

    // the whole run fits the budget: the search gets it minus the prefill point and the final gate
    // (LI 3 classes x 2 x li_rounds rounds + 3 prompts x 2 modes x 2 arms x hash_tokens tokens)
    let reserve_s = if skip_prefill { 0.0 } else { 120.0 }
        + if skip_gate { 0.0 } else { 3.0 * 2.0 * li_rounds as f64 * 0.07 + 12.0 * hash_tokens as f64 / 90.0 + 30.0 };
    let rep: Vec<usize> = [3usize, 5, 6, 8].iter().copied().filter(|&m| m <= k + 1).collect();
    let all: Vec<usize> = (2..=k + 1).collect();
    let mut session = Session::new(&model, SrCfg { verify_on: true, draft_on: true, li_on: false, verify_widths: all.clone(),
                                                   draft_s: k, mem_cap: 512 << 20 }, vec![Arm::base()], seed)?;
    session.set_warmup(warmup)?;
    SESSION.with(|c| *c.borrow_mut() = Some(session));
    let mut t = Tuner {
        model: &model, head, sc, psc: Some(psc0), chunk, k, draft_conf, max_new, corpus: corpus.clone(), corpus_label: "dense",
        gen_next: 0, gen: None, round_ms: Vec::new(), widths_rep: rep, widths_all: all,
        base_ms: HashMap::new(), base_mad: HashMap::new(), decisions: Vec::new(), families: serde_json::Map::new(),
        jsonl, t0: std::time::Instant::now(), budget_s: (budget_min * 60.0 - reserve_s).max(60.0), rcfg: tune::RaceCfg::default(), regime: 0,
        gpu_log: Vec::new(), throttle_events: 0, mem_cap: 512 << 20,
        warmup, stratify, pf_order: tune::BalancedOrder::new(seed ^ 0x9F), known_answer: None,
    };
    let _ = writeln!(t.jsonl, "{}", serde_json::json!({
        "receipt": "gb10-exl3-autotune/1", "fingerprint": serde_json::to_value(&fp)?, "posture": serde_json::to_value(&posture)?,
        "binary_sha256": bin_sha, "commit": env!("TUNE_GIT_COMMIT"), "dirty": env!("TUNE_GIT_DIRTY"),
        "kernel_build_id": env!("KERNEL_BUILD_ID"), "tune_build_id": env!("TUNE_BUILD_ID"), "model_dir": dir, "seed": seed,
        "date_unix": stamp, "budget_min": budget_min, "overridden_env": format!("{overridden:?}"),
        "protocol": protocol_json(warmup, stratify) }));
    t.say(format!("AUTOTUNE: measurement protocol v3 — {warmup} untimed base warm-up replay(s) per block, balanced (Williams) \
                   arm order per cell, analysis {}", if stratify { "position-stratified" } else { "plain paired" }));
    t.say(format!("AUTOTUNE: model {dir} loaded in {:.1} s | posture lanes {lanes} max_pos {max_pos} k {k} dds {} kv {} chunk {chunk} \
                   | profile {profile} MHz tp 1 | budget {budget_min} min (search {:.1} min, gate + prefill reserve {:.1} min) \
                   | seed {seed} | receipts {jsonl_path}",
                  t_load.elapsed().as_secs_f64(), draft_conf > 0.0, kv_fmt_name(kv_fmt_from_opts()),
                  t.budget_s / 60.0, reserve_s / 60.0));
    for (id, env, v) in &overridden {
        t.say(format!("AUTOTUNE: {env} is set -> {id}={v} is OVERRIDDEN(flag) and will not be searched"));
    }
    let e0 = t.env_snapshot("start");
    let run = (|| -> Result<()> {
        // ---- dense: probe, plan, search
        t.probe(0)?;
        let mut plan = t.build_plan(0);
        if let Some(o) = &only { plan.retain(|(f, _)| o.iter().any(|x| f.name.starts_with(x.as_str()))); }
        let est: f64 = plan.iter().map(|(f, tt)| t.est_s(f, *tt)).sum();
        let order: Vec<String> = plan.iter().map(|(f, tt)| format!("{} [{}] ({} cand)", f.name, tt.name(), f.cands.len())).collect();
        t.say(format!("PLAN (dense): {} families, est {:.1} min, budget {budget_min} min — order: {}",
                      plan.len(), est / 60.0, order.join(", ")));
        t.search(plan, max_sweeps)?;
        // ---- sparse point (a real long prompt past the dense/sparse switch)
        let sparse_len = sparse_ctx.unwrap_or(0).max(model.qsa_limit() + 32);
        let tiled: Vec<u32> = {
            let src: Vec<u32> = corpus.iter().flatten().copied().collect();
            let want = sparse_len.max(p_len_for_prefill(chunk));
            src.iter().cycle().take(want.min(max_pos.saturating_sub(max_new + 2 * (MTP_MAX_K + 2) + 8))).copied().collect()
        };
        if skip_sparse {
            t.families.insert("sparse point".into(), serde_json::json!({"status": "SKIPPED (--autotune-skip-sparse)"}));
        } else if !model.cfg.has_indexer() || tiled.len() < sparse_len {
            t.families.insert("sparse point".into(), serde_json::json!({"status": format!(
                "UNTUNED (posture max_pos {max_pos} leaves no sparse positions past the switch at {})", model.qsa_limit())}));
        } else {
            t.corpus = vec![tiled[..sparse_len].to_vec()];
            t.corpus_label = "sparse (tiled corpus past the dense/sparse switch — an approximation)";
            t.gen = None;
            t.gen_next = 0;
            t.regime = 1;
            t.probe(1)?;
            let mut plan = t.build_plan(1);
            if let Some(o) = &only { plan.retain(|(f, _)| o.iter().any(|x| f.name.starts_with(x.as_str()))); }
            t.say(format!("PLAN (sparse, ctx {sparse_len} tokens, {}): {} families", t.corpus_label, plan.len()));
            t.search(plan, 1)?;
            t.corpus = corpus.clone();
            t.gen = None;
            t.regime = 0;
        }
        // ---- prefill point (the B3 known answer)
        if skip_prefill {
            t.families.insert("prefill.wide_mt".into(), serde_json::json!({"status": "SKIPPED (--autotune-skip-prefill)"}));
        } else {
            let tiled_pf: Vec<u32> = corpus.iter().flatten().copied().cycle().take(64 + chunk).collect();
            t.prefill_phase(&tiled_pf)?;
        }
        Ok(())
    })();
    // the session's tuner graphs + events go before anything else touches the stream
    let session = SESSION.with(|c| c.borrow_mut().take());
    let mut position_effect = serde_json::Value::Array(Vec::new());
    if let Some(s) = session.as_ref() {
        for smp in &s.samples {
            let variant = if smp.arm == usize::MAX { serde_json::json!("canonical") }
                          else if smp.arm == WARMUP_ARM { serde_json::json!("warmup") } else { serde_json::json!(smp.arm) };
            let _ = writeln!(t.jsonl, "{}", serde_json::json!({
                "case": format!("{}:m{}", smp.unit, smp.m), "variant": variant, "phase": smp.phase, "trial": smp.round,
                "time_us": smp.ms * 1e3, "correct": smp.digest_ok, "contract": format!("{}:{}:graph", &fp.tune_build_id, smp.mode),
                "round": smp.round, "m": smp.m, "regime": if smp.regime == 1 { "sparse" } else { "dense" },
                "pos": smp.pos, "prompt": smp.prompt, "mode": smp.mode,
                "arm_pos": if smp.apos == NO_APOS { serde_json::Value::Null } else { serde_json::json!(smp.apos) },
                "n_pos": smp.npos }));
        }
        // §5.2 v2: the run's replay-position effect (table provenance -> --autotune-report)
        let offs = pos_table(&s.samples, &|_| true);
        position_effect = position_effect_json(&s.samples, &offs);
        let _ = writeln!(t.jsonl, "{}", serde_json::json!({"position_effect": position_effect.clone()}));
    }
    drop(session);
    run?;
    // ---- the final gate
    let prompts: Vec<(&'static str, Vec<u32>, bool)> = vec![
        ("ansic", tok.encode(CORPUS[0], false)?, false),
        ("python", tok.encode(PY_PROMPT, false)?, false),
        ("prose", tok.encode(CORPUS[1], false)?, false),
        ("scifi", tok.encode(SCIFI_PROMPT, false)?, true),
    ];
    let (pass, confirm) = if skip_gate {
        t.say("FINAL GATE: SKIPPED (--autotune-skip-gate) — the table is written UNGATED (diagnostic runs only)".into());
        (true, serde_json::json!({"skipped": "--autotune-skip-gate", "pass": false}))
    } else {
        t.final_gate(li_rounds, hash_tokens, &prompts)?
    };
    let e1 = t.env_snapshot("end");
    // ---- the table (§7.1)
    let clocks: Vec<f64> = t.gpu_log.iter().filter_map(sm_clock).collect();
    let temps: Vec<f64> = t.gpu_log.iter().filter_map(|e| e.get("temperature.gpu").and_then(|v| v.as_str())
        .and_then(|s| s.trim().parse::<f64>().ok())).collect();
    let mm = |v: &[f64]| if v.is_empty() { serde_json::Value::Null } else {
        serde_json::json!([v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)]) };
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim().to_string();
    let ips: Vec<String> = local_ipv4s();
    let uuid = tune::run_timeout("nvidia-smi", &["--query-gpu=uuid", "--format=csv,noheader"], std::time::Duration::from_secs(5))
        .map(|s| s.trim().to_string()).unwrap_or_default();
    let noise: serde_json::Map<String, serde_json::Value> = t.base_mad.iter()
        .map(|((u, m, r), v)| (format!("{u}_m{m}{}", if *r == 1 { "_sparse" } else { "" }), serde_json::json!(v))).collect();
    let decisions: Vec<tune::Decision> = t.table_decisions();
    // the not-searched list (every registered entry is accounted for)
    for def in tune::REGISTRY.iter() {
        if let Some(why) = t.prune_reason(def) {
            if !t.families.keys().any(|k| k.starts_with(def.id)) {
                t.families.insert(def.id.to_string(), serde_json::json!({"status": format!("NOT SEARCHED: {why}")}));
            }
        }
    }
    let mut table = tune::Table {
        format: tune::TABLE_FORMAT.into(), origin: origin.clone(), fingerprint: fp.clone(), posture: posture.clone(),
        provenance: serde_json::json!({
            "binary_sha256": bin_sha, "kernel_build_id": env!("KERNEL_BUILD_ID"), "tune_build_id": env!("TUNE_BUILD_ID"),
            "source_build_id": option_env!("SOURCE_BUILD_ID").unwrap_or(""), "commit": env!("TUNE_GIT_COMMIT"),
            "dirty": env!("TUNE_GIT_DIRTY"), "box": format!("{host} ({})", ips.join(",")), "gpu_uuid": uuid,
            "date_unix": stamp, "clocks": {"sm_mhz": mm(&clocks), "temp_c": mm(&temps), "throttle_events": t.throttle_events},
            "resident_apps": [], "corpus_sha256": corpus_sha, "mode": "full", "classes": ["S"],
            "budget_min": budget_min, "elapsed_min": t.elapsed_s() / 60.0, "noise_floor_ms": noise, "seed": seed,
            "receipts": jsonl_path, "env_start": e0, "env_end": e1,
            "protocol": protocol_json(warmup, stratify), "position_effect": position_effect,
        }),
        decisions,
        families: serde_json::Value::Object(t.families.clone()),
        confirm,
        table_sha256: String::new(),
    };
    let path = match arg(args, "--autotune-table") { Some(p) => p.to_string(), None => tune::table_file_name(&table) };
    // a rejected or ungated table never lands where `--tune-table auto` looks, and carries
    // confirm.pass = false (boot's auto selection refuses it anyway); load it only by explicit path
    let fname = path.rsplit('/').next().unwrap_or("table.json").to_string();
    let (dest, written) = if skip_gate {
        (format!("{out_dir}/ungated-{stamp}.{fname}"), true)
    } else if pass {
        (path.clone(), true)
    } else {
        (format!("{out_dir}/rejected-{stamp}.{fname}"), false)
    };
    let sha = tune::write_table(&mut table, &dest)?;
    let raw = std::fs::read(&dest)?;
    print!("{}", tune::report(&table, &raw));
    t.say(format!("AUTOTUNE RESULT: {} {dest} (file sha256 {sha}, {} decision(s), {:.1} min; commit {} dirty {}, binary {bin_sha}, \
                   tune_build_id {}, model {dir})",
                  if skip_gate { "UNGATED TABLE (diagnostic; auto never loads it):" }
                  else if written { "TABLE WRITTEN" } else { "REJECTED by the final gate — kept out of ./tune:" },
                  table.decisions.len(), t.elapsed_s() / 60.0, env!("TUNE_GIT_COMMIT"), env!("TUNE_GIT_DIRTY"), env!("TUNE_BUILD_ID")));
    Ok(written)
}

/// Prefill point: the tiled corpus must hold the prefix + the largest chunk.
fn p_len_for_prefill(chunk: usize) -> usize { 64 + chunk }

#[cfg(test)]
mod tests {
    use super::*;

    /// The LI schedule is balanced: every block of `arms` consecutive draws is a permutation, the
    /// reported in-block position is the draw's place in its block, and over a whole period every
    /// arm takes every in-block position equally often (v2: a Williams walk, not random blocks).
    #[test]
    fn li_schedule_is_balanced_blocks() {
        for arms in 1..=5usize {
            for seed in [0u64, 3, 11] {
                let mut s = LiSched::new(arms, seed);
                let p = tune::williams_period(arms);
                let mut at = vec![vec![0usize; arms]; arms];
                for _ in 0..p {
                    let draws: Vec<(usize, u8)> = (0..arms).map(|_| s.next()).collect();
                    let mut seen: Vec<usize> = draws.iter().map(|d| d.0).collect();
                    seen.sort();
                    assert_eq!(seen, (0..arms).collect::<Vec<_>>());
                    for (j, d) in draws.iter().enumerate() {
                        assert_eq!(d.1 as usize, j);
                        at[d.0][j] += 1;
                    }
                }
                for a in 0..arms { assert_eq!(at[a], vec![p / arms; arms], "arms {arms} seed {seed}"); }
            }
        }
    }

    /// The SR block: the warm-up(s) come first and only when the base is runnable; the timed arms
    /// follow in the row's order with positions 0..n (a skipped arm consumes no position).
    #[test]
    fn block_plan_puts_the_warmup_first() {
        let p = block_plan(&[2, 0, 1], 2);
        assert_eq!(p, vec![Step::Warm, Step::Warm, Step::Timed { arm: 2, apos: 0 }, Step::Timed { arm: 0, apos: 1 },
                           Step::Timed { arm: 1, apos: 2 }]);
        // no base graph (memory cap): no warm-up, positions stay dense
        assert_eq!(block_plan(&[2, 1], 1), vec![Step::Timed { arm: 2, apos: 0 }, Step::Timed { arm: 1, apos: 1 }]);
        assert_eq!(block_plan(&[0], 0), vec![Step::Timed { arm: 0, apos: 0 }]);
        // cells separate units, widths and regimes
        assert_ne!(order_cell(0, 7, false), order_cell(1, 7, false));
        assert_ne!(order_cell(1, 3, false), order_cell(1, 6, false));
        assert_ne!(order_cell(1, 3, false), order_cell(1, 3, true));
    }

    /// Region words: a partial trailing word is skipped, alignment is enforced.
    #[test]
    fn digest_regions_are_word_aligned() {
        assert_eq!(reg(0x1000, 10).unwrap().words, 2);
        assert!(reg(0x1002, 8).is_err());
        assert_eq!(fnv64(&[1, 2, 3]), fnv64(&[1, 2, 3]));
        assert_ne!(fnv64(&[1, 2, 3]), fnv64(&[3, 2, 1]));
    }

    fn smp(unit: &'static str, phase: usize, m: usize, arm: usize, round: usize, ms: f64, ok: bool) -> Sample {
        Sample { unit, phase, m, arm, prompt: 0, round, pos: 100 + round, regime: 0, ms, digest_ok: ok, mode: "sr",
                 apos: arm.min(7) as u8, npos: 2 }
    }

    /// Pairing: only the same round's base sample pairs; digest failures and other phases are out;
    /// the Round objective adds the draft delta at the served passes-per-round share.
    #[test]
    fn pair_obs_pairs_within_rounds_only() {
        let s = vec![
            smp("verify_dry", 1, 3, 0, 10, 40.0, true), smp("verify_dry", 1, 3, 1, 10, 39.5, true),
            smp("draft", 1, 7, 0, 10, 8.0, true), smp("draft", 1, 7, 1, 10, 7.3, true),
            smp("verify_dry", 1, 6, 0, 11, 48.0, true), smp("verify_dry", 1, 6, 1, 11, 48.2, true),
            // round 12: the candidate's digest failed -> no pair
            smp("verify_dry", 1, 5, 0, 12, 45.0, true), smp("verify_dry", 1, 5, 1, 12, 44.0, false),
            // another phase -> ignored
            smp("verify_dry", 2, 3, 0, 13, 40.0, true), smp("verify_dry", 2, 3, 1, 13, 30.0, true),
            // the canonical execution (arm usize::MAX) never pairs
            smp("verify", 1, 8, usize::MAX, 10, 55.0, true),
        ];
        let none = PosTable::new();
        let v = pair_obs(&s, &none, 1, Target::Verify, 2, 7, 0);
        assert_eq!(v[1].len(), 2);
        assert!((v[1][0].1 - -0.5).abs() < 1e-12 && v[1][0].0 == 3);
        assert!((v[1][1].1 - 0.2).abs() < 1e-9 && v[1][1].0 == 6);
        let d = pair_obs(&s, &none, 1, Target::Draft, 2, 7, 0);
        assert_eq!(d[1].len(), 1);
        assert!((d[1][0].1 - -0.7).abs() < 1e-9);
        let r = pair_obs(&s, &none, 1, Target::Round, 2, 7, 0);
        assert!(pair_obs(&s, &none, 1, Target::Verify, 2, 7, 1)[1].is_empty()); // another regime never pairs
        // v2: a cell's fitted position offsets are removed before pairing (base at apos 0, arm at 1)
        let mut offs = PosTable::new();
        offs.insert(("verify_dry", 3, 0, 2), vec![0.25, -0.25]);
        let vs = pair_obs(&s, &offs, 1, Target::Verify, 2, 7, 0);
        assert!((vs[1][0].1 - (-0.5 + 0.25 + 0.25)).abs() < 1e-9, "{:?}", vs[1]);
        assert!((vs[1][1].1 - 0.2).abs() < 1e-9); // m = 6 has no offsets: unchanged
        assert_eq!(r[1].len(), 1); // round 11 has no draft pair
        let want = -0.5 + DRAFT_PASSES_PER_ROUND / 7.0 * -0.7;
        assert!((r[1][0].1 - want).abs() < 1e-9, "{:?}", r[1]);
        assert!(v[0].is_empty()); // the base never pairs with itself
    }

    /// Arm signatures: equal assignments share cached graphs; a different value, key or knob does not.
    #[test]
    fn arm_signatures_identify_configurations() {
        let a = Arm { name: "a".into(), assign: vec![(&T_WP09_AM_NB, None, 16)] };
        let b = Arm { name: "b".into(), assign: vec![(&T_WP09_AM_NB, None, 16)] };
        let c = Arm { name: "c".into(), assign: vec![(&T_WP09_AM_NB, None, 32)] };
        let d = Arm { name: "d".into(), assign: vec![(&T_WP09_AM_NB, Some(tune::Key::G), 16)] };
        let e = Arm { name: "e".into(), assign: vec![(&T_HC_BXR, None, 16)] };
        assert_eq!(a.sig(), b.sig());
        assert_ne!(a.sig(), c.sig());
        assert_ne!(a.sig(), d.sig());
        assert_ne!(a.sig(), e.sig());
        assert_ne!(Arm::base().sig(), a.sig());
    }

    #[test]
    fn clock_and_throttle_fields_parse() {
        let e = serde_json::json!({"clocks.sm": "2405", "clocks_event_reasons.active": "0x0000000000000001"});
        assert_eq!(sm_clock(&e), Some(2405.0));
        assert!(!throttled(&e)); // 0x1 = GPU idle, not a throttle
        let t = serde_json::json!({"clocks.sm": "unknown", "clocks_event_reasons.active": "0x0000000000000040"});
        assert_eq!(sm_clock(&t), None); // unsupported reads "unknown", never zero
        assert!(throttled(&t)); // HW thermal slowdown
    }

    /// Every registered entry is either in a default search list, a declared pair, the per-shape
    /// PSK/chain families, or NOT_SEARCHED / pruned by class-scope-xcheck (the report accounts for all).
    #[test]
    fn every_registry_entry_has_a_search_status() {
        for t in tune::REGISTRY.iter() {
            let listed = DENSE_SPECS.iter().any(|x| x.0 == t.id) || SPARSE_SPECS.iter().any(|x| x.0 == t.id)
                || PAIRS.iter().any(|p| p.0 == t.id || p.2 == t.id)
                || ["psk.g", "psk.g_pair", "chain.pair"].contains(&t.id)
                || NOT_SEARCHED.iter().any(|x| x.0 == t.id);
            let pruned = t.class != tune::Class::S || t.xcheck.is_empty()
                || matches!(t.scope, tune::Scope::Load | tune::Scope::Chunk);
            assert!(listed || pruned, "{} is neither searched nor listed as not searched", t.id);
            if listed && !NOT_SEARCHED.iter().any(|x| x.0 == t.id) {
                assert_eq!(t.class, tune::Class::S, "{} searched but not class S", t.id);
                assert!(!t.xcheck.is_empty(), "{} searched without a proven domain", t.id);
            }
        }
        // pair grids stay within the design's 9 combinations
        for (_, a, _, b, _) in PAIRS { assert!(a.len() * b.len() - 1 <= 9); }
    }

    // ---- §5.2 v2: a synthetic sampler through the production order / plan / pairing code ----

    struct Xs(u64);
    impl Xs {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        /// symmetric triangular noise, |x| <= amp
        fn tri(&mut self, amp: f64) -> f64 { (self.u() + self.u() - 1.0) * amp }
    }

    /// One self-test cell (verify_dry m = 6) over `rounds` rounds with arms base, base2 (the A/A
    /// twin) and spin (+0.200 ms). Every replay i of a round (warm-ups included) pays `pen(i, P_r)`
    /// on top of the round's content and the arm's effect; P_r varies per round (the cold start is
    /// not a constant). v1 = the old protocol: a fresh Fisher–Yates order every round (the removed
    /// Session::shuffled_arms), no warm-up. v2 = the production path: BalancedOrder rows per cell,
    /// block_plan with the warm-up, Samples recorded as resolve() records them. Returns the samples
    /// and how often each arm ran first among the TIMED replays.
    fn simulate(v2: bool, warmup: usize, rounds: usize, seed: u64, pen: &dyn Fn(usize, f64) -> f64)
        -> (Vec<Sample>, Vec<usize>) {
        let eff = [0.0, 0.0, 0.2];
        let n = eff.len();
        let mut rng = Xs(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut order = tune::BalancedOrder::new(seed);
        let (mut out, mut first) = (Vec::new(), vec![0usize; n]);
        for r in 0..rounds {
            let content = 30.0 + 10.0 * rng.u();
            let p_r = 0.5 + rng.u();
            let row = if v2 { order.next(order_cell(1, 6, false), n) } else {
                let mut v: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() { let j = ((rng.u() * (i + 1) as f64) as usize).min(i); v.swap(i, j); }
                v
            };
            first[row[0]] += 1;
            for (i, st) in block_plan(&row, if v2 { warmup } else { 0 }).into_iter().enumerate() {
                let (arm, apos, mode, e) = match st {
                    Step::Warm => (WARMUP_ARM, NO_APOS, "warm", 0.0),
                    Step::Timed { arm, apos } => (arm, apos, "sr", eff[arm]),
                };
                let ms = content + e + pen(i, p_r) + rng.tri(0.03);
                out.push(Sample { unit: "verify_dry", phase: 1, m: 6, arm, prompt: 0, round: r, pos: 100, regime: 0, ms,
                                  digest_ok: true, mode, apos, npos: n as u8 });
            }
        }
        (out, first)
    }

    fn est(s: &[Sample], arm: usize, offs: &PosTable) -> tune::Est {
        tune::hl_paired(&paired(s, "verify_dry", 6, arm, offs)).expect("pairs")
    }

    /// The receipts' failure, reproduced and fixed: a first-replay penalty (P_r ~ 0.5..1.5 ms, the
    /// .14 evidence) biases the v1 protocol's paired HL by the realized first-position imbalance —
    /// the §5.7 A/A / known-positive criteria fail on many seeds, and the error's sign follows the
    /// imbalance — while v2 (warm-up + balanced order, plain or stratified) is unbiased: every arm
    /// first equally often and the criteria hold on every seed.
    #[test]
    fn synthetic_first_position_penalty_biases_v1_not_v2() {
        let pen = |i: usize, p: f64| if i == 0 { p } else { 0.0 };
        let (rounds, seeds) = (120usize, 1..=24u64);
        let mut v1_fail = 0usize;
        let mut signed = 0usize;
        let mut v1_max = 0.0f64;
        let mut v2_max = 0.0f64;
        let (mut v2_cover, mut v2_n) = (0usize, 0usize);
        let mut log = Vec::new();
        for seed in seeds.clone() {
            // ---- v1
            let (s, first) = simulate(false, 0, rounds, seed, &pen);
            let none = PosTable::new();
            let (aa, sp) = (est(&s, 1, &none), est(&s, 2, &none));
            if !(aa_ok(&aa, 0.02) && spin_ok(&sp, 0.2)) { v1_fail += 1; }
            let d = first[1] as i64 - first[0] as i64; // base2 first more often -> base2 reads slower
            if d.abs() >= 10 {
                signed += 1;
                assert_eq!(aa.est > 0.0, d > 0, "v1 seed {seed}: first {first:?} A/A HL {:+.4}", aa.est);
            }
            v1_max = v1_max.max(aa.est.abs());
            log.push(format!("seed {seed}: v1 first {first:?} A/A {:+.4} spin {:+.4} -> {}", aa.est, sp.est,
                             if aa_ok(&aa, 0.02) && spin_ok(&sp, 0.2) { "pass" } else { "FAIL" }));
            // ---- v2 (the same content/noise model)
            let (s2, first2) = simulate(true, 1, rounds, seed, &pen);
            assert_eq!(first2, vec![rounds / 3; 3], "v2 seed {seed}: the balanced order must put every arm first equally often");
            // no warm-up sample is ever paired, and every timed replay carries its position
            assert!(s2.iter().filter(|x| x.mode == "warm").all(|x| x.arm == WARMUP_ARM && x.apos == NO_APOS));
            assert_eq!(s2.iter().filter(|x| x.mode == "warm").count(), rounds);
            for offs in [PosTable::new(), pos_table(&s2, &|_| true)] {
                let (aa2, sp2) = (est(&s2, 1, &offs), est(&s2, 2, &offs));
                // no bias: the estimate is inside the A/A band on EVERY seed; the 95% CI itself
                // misses 0 by chance on ~5% of seeds (noise only), so coverage is counted
                assert!(aa2.est.abs() < 0.02, "v2 seed {seed} (stratified {}): A/A {aa2:?}", !offs.is_empty());
                assert!(spin_ok(&sp2, 0.2) && (sp2.est - 0.2).abs() < 0.02, "v2 seed {seed}: spin {sp2:?}");
                v2_cover += (aa2.lo <= 0.0 && 0.0 <= aa2.hi) as usize;
                v2_n += 1;
                v2_max = v2_max.max(aa2.est.abs());
            }
        }
        println!("{}\nv1: {v1_fail}/24 seeds fail, max |A/A| {v1_max:.4} ms; v2: max |A/A| {v2_max:.4} ms, A/A CI covers 0 \
                  {v2_cover}/{v2_n}", log.join("\n"));
        assert!(v2_cover * 10 >= v2_n * 9, "v2 A/A coverage {v2_cover}/{v2_n} (nominal 95%)");
        assert!(v1_fail >= 6, "the v1 protocol should fail the §5.7 criteria on many seeds ({v1_fail}/24):\n{}", log.join("\n"));
        assert!(signed >= 3, "too few strongly imbalanced v1 seeds to check the sign rule ({signed}):\n{}", log.join("\n"));
        assert!(v1_max > 3.0 * v2_max && v1_max > 0.04, "v1 max |A/A| {v1_max} vs v2 {v2_max}");
    }

    /// A residual position gradient the warm-up does not absorb (replays 1..3 pay +0.6 / +0.1 /
    /// -0.7 ms): the balanced order keeps the plain paired HL centred but wide; the position model
    /// recovers the gradient and the stratified CI is several times tighter, centred on the truth.
    #[test]
    fn stratification_removes_a_residual_gradient() {
        let pen = |i: usize, p: f64| match i { 0 => p, 1 => 0.6, 2 => 0.1, _ => -0.7 };
        let (s, _) = simulate(true, 1, 240, 7, &pen);
        let offs = pos_table(&s, &|_| true);
        let f = &offs[&("verify_dry", 6, 0, 3)];
        for (p, want) in [0.6, 0.1, -0.7].iter().enumerate() { assert!((f[p] - want).abs() < 0.02, "{f:?}"); }
        let none = PosTable::new();
        let (plain, strat) = (est(&s, 1, &none), est(&s, 1, &offs));
        assert!(plain.lo <= 0.0 && 0.0 <= plain.hi, "{plain:?}");
        assert!(aa_ok(&strat, 0.02), "{strat:?}");
        assert!((strat.hi - strat.lo) * 4.0 < plain.hi - plain.lo, "stratified {strat:?} vs plain {plain:?}");
        let sp = est(&s, 2, &offs);
        assert!((sp.est - 0.2).abs() < 0.01, "{sp:?}");
        // the report view: the warm-up and the timed positions, with the fitted offsets
        let pe = position_effect_json(&s, &offs);
        let c = &pe.as_array().unwrap()[0];
        assert_eq!(c["n_pos"], 3);
        assert_eq!(c["rounds"], 240);
        assert!(c["warmup_median_ms"].as_f64().unwrap() > c["median_ms_by_pos"][2].as_f64().unwrap());
        assert!((c["offsets_ms"][2].as_f64().unwrap() + 0.7).abs() < 0.02);
    }

    #[test]
    fn protocol_flags_parse_and_refuse() {
        let a = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(protocol_args(&a(&["--autotune"])).unwrap(), (2, true)); // v3: warm-up default 2
        assert_eq!(protocol_args(&a(&["--autotune-warmup", "1"])).unwrap(), (1, true));
        assert_eq!(protocol_args(&a(&["--autotune-warmup", "0", "--autotune-stratify", "off"])).unwrap(), (0, false));
        assert_eq!(protocol_args(&a(&["--autotune-warmup", "4"])).unwrap(), (4, true));
        assert!(protocol_args(&a(&["--autotune-warmup", "5"])).is_err());
        assert!(protocol_args(&a(&["--autotune-stratify", "maybe"])).is_err());
        // v3: the self-test's A/A threshold
        assert_eq!(aa_args(&a(&["--autotune"])).unwrap(), AaRule::DEFAULT);
        assert_eq!(aa_args(&a(&["--autotune-selftest-tau", "0.037", "--autotune-selftest-tau-pct", "0.1"])).unwrap(),
                   AaRule { abs_ms: 0.037, pct: 0.1 });
        assert!(aa_args(&a(&["--autotune-selftest-tau", "0"])).is_err());
        assert!(aa_args(&a(&["--autotune-selftest-tau", "x"])).is_err());
        assert!(aa_args(&a(&["--autotune-selftest-tau-pct", "-1"])).is_err());
    }

    /// v3 A/A rule on the .13 receipts (2026-09-27 05:05-05:17 UTC; stratified HL / CI95 and the base
    /// medians of each cell): tau = max(0.02 ms, 0.05 % of the median). The rule is what the brief
    /// specified; the receipt that motivated it (selftest-1790486005 verify m = 3) still FAILS it at
    /// the defaults — the CI's upper end (+0.0368) leaves +-0.020 — and passes from tau 0.0368 ms up.
    #[test]
    fn aa_rule_on_the_selftest_receipts() {
        let r = AaRule::DEFAULT;
        let est = |n: usize, hl: f64, lo: f64, hi: f64| tune::Est { n, est: hl, lo, hi };
        // (receipt cell, estimate, base median ms, verdict under v3 defaults, verdict under v1/v2)
        let cells = [
            ("1790485556 verify m3", est(127, -0.0452, -0.1378, 0.0280), 31.798, false, false),
            ("1790485556 verify m6", est(126, 0.0248, -0.0399, 0.0993), 39.055, false, false),
            ("1790485556 draft m7", est(253, 0.0014, -0.0030, 0.0063), 5.758, true, true),
            ("1790485758 verify m3", est(236, -0.0021, -0.0631, 0.0518), 31.984, true, true),
            ("1790485758 verify m6", est(236, 0.0260, -0.0221, 0.0842), 39.352, false, false),
            ("1790485758 draft m7", est(472, -0.0001, -0.0036, 0.0036), 5.724, true, true),
            ("1790486005 verify m3", est(236, 0.0188, 0.0005, 0.0368), 31.917, false, false),
            ("1790486005 verify m6", est(236, -0.0037, -0.0227, 0.0124), 39.345, true, true),
            ("1790486005 draft m7", est(472, 0.0019, -0.0045, 0.0085), 5.832, true, true),
        ];
        for (name, e, med, v3, old) in cells {
            let tau = r.tau(med);
            assert!((tau - 0.02).abs() < 1e-12, "{name}: tau {tau} (every cell here is under 40 ms)");
            assert_eq!(aa_ok(&e, tau), v3, "{name}: {e:?} tau {tau}");
            assert_eq!(e.est.abs() < 0.02 && e.lo <= 0.0 && 0.0 <= e.hi, old, "{name}: the v1/v2 rule");
        }
        // the motivating cell: the |HL| clause holds, the CI clause does not ...
        let m3 = est(236, 0.0188, 0.0005, 0.0368);
        assert_eq!(aa_verdict(&m3, 0.02), (false, "CI excludes 0 and leaves +-tau"));
        // ... until tau covers the CI's upper end
        assert_eq!(aa_verdict(&m3, 0.0368), (true, "CI within +-tau"));
        assert!(!aa_ok(&m3, 0.0367));
        // the relative part only binds above 40 ms (e.g. the m = 8 canonical verify, 45.7 ms)
        assert!((r.tau(45.726) - 0.022863).abs() < 1e-9);
        assert_eq!(r.tau(f64::NAN), 0.02);
    }
}
