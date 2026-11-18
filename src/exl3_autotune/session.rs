//! The in-round measurement session: arms, samples, pending events, the balanced (Williams) block plan and `impl Session`.
//! Part of the `--autotune` core (see `mod.rs` for the file map). Nothing here runs unless `armed()` is true; canonical-last ordering keeps served output bit-identical.

use super::*;

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
pub(super) enum UnitKey {
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
pub(super) const MAX_WARMUP: usize = 4;
/// v3 (2026-09-27): the default `--autotune-warmup` — 2 (was 1). The .13 self-tests at 1 left a
/// first-position offset of +0.25 / +0.12 ms at verify m = 3 / 6 (receipts selftest-1790485556 /
/// -1790485758); at 2 the fitted offsets were <= 6 us in every cell (selftest-1790486005).
pub(super) const DEFAULT_WARMUP: usize = 2;

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
pub(super) fn order_cell(unit: u64, m: usize, regime: bool) -> u64 { (unit << 40) | ((m as u64) << 8) | regime as u64 }

/// One step of a round's SR block (§5.2 v2): the base's untimed warm-up(s), then every runnable arm
/// once, in the round's balanced order, with its place in the timed block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Step {
    Warm,
    Timed { arm: usize, apos: u8 },
}

/// The block of one round: `run` = the balanced row filtered to the arms whose graph exists (the
/// base, arm 0, is always attempted first-class; no base graph = no warm-up and no pairing anyway).
pub(super) fn block_plan(run: &[usize], warmup: usize) -> Vec<Step> {
    let mut v: Vec<Step> = Vec::with_capacity(warmup + run.len());
    if run.contains(&0) { v.extend(std::iter::repeat(Step::Warm).take(warmup)); }
    v.extend(run.iter().enumerate().map(|(i, &a)| Step::Timed { arm: a, apos: i as u8 }));
    v
}

pub(crate) struct Session {
    pub(super) cfg: SrCfg,
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
    pub(super) mem_cap_hit: bool,
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

    pub(super) fn before_draft(&mut self, model: &FwdModel, sc: &mut Scratch, head: &DraftHead, at: &SrAt) -> Result<()> {
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

    pub(super) fn after_draft(&mut self, model: &FwdModel, _sc: &mut Scratch, _at: &SrAt) -> Result<()> {
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

    pub(super) fn before_verify(&mut self, model: &FwdModel, sc: &mut Scratch, at: &SrAt) -> Result<()> {
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

    pub(super) fn after_verify(&mut self, model: &FwdModel, _sc: &mut Scratch, at: &SrAt) -> Result<()> {
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

    pub(super) fn li_step(&mut self, model: &FwdModel, sc: &mut Scratch, m: usize, qsa: bool, bucket: usize) -> Result<bool> {
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
