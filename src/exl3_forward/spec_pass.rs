//! Speculative draft-pass state: the SpecMode / SpecGeom / SpecPass types, their counters and the
//! `impl FwdModel` spec_* methods (geometry, rows, keys, begin / next / end of a pass).
//! See `mod.rs` for the file map.

use super::*;

// ---- HOST / RO-5(a): speculative next draft pass (PLAN/SURPASS_PLAN_2026-09-26.md, RO-5(a)) ----
// In the DDS round (mtp_round_adaptive) pass i+1 used to be launched only after the host had read
// pass i's confidence and `keep` had said "continue" — a host round trip (event wake + readback +
// keep + graph launch + launch latency, ~50 µs) of GPU idle per pass. With the speculation, pass
// i+1 is enqueued BEFORE pass i's readback is waited on, when the lane's recent continuation rate
// at pass i clears the threshold. Exactness (the trajectory cannot move):
//  - pass i+1's inputs are unchanged by launching it early: the same graph, after pass i in stream
//    order; its only host-side inputs (draft_meta, slot_ids, penalty rows) were staged before
//    pass 0 (slot_ids = the lane's slot in every regime since ISSUES 128), and the readback/keep of pass i never write
//    device memory. So every conf/draft, hence every `keep` decision (called with the same
//    (i, conf) sequence), is bit-identical.
//  - when the DDS stops at i with pass i+1 in flight ("wasted"), its PERSISTENT writes — the head's
//    K|V rows at x = p+i+1, the raw indexer-key row x and the pooled-plane block of x — are
//    restored from a copy the xq_spec_rows kernel saved right before the pass (stream order: pass
//    i -> save -> pass i+1 -> restore -> verify), bit for bit. The K|V rows are the lane's slot
//    (xq_draft_setup stages sc.slots from draft_meta); the raw-key row and the pooled block are
//    addressed ON DEVICE through slot_ids[0], the exact slot the pass's qsa_key_write_b /
//    qsa_pool_update_b use — the lane's own slot (before ISSUES 128 a dense round of a multi-lane
//    server could leave ANOTHER lane's slot there; HOST only guarantees that a wasted pass leaves
//    no trace of its own wherever the pass wrote). Its SCRATCH writes (sc.toks/pos/slots
//    row 0, sc.resid, sc.cos, the chain/attention/MoE/logit scratch, dconf[i+1], d_dev[i+1],
//    sc.argmax) are the same buffers the last real pass writes; the verify at width i+1 rewrites
//    toks/pos/slots/slot_ids (upload), resid (xq_embed_resid, first launch), cos (xq_cos_gather),
//    logits/argmax before reading them; xq_accept reads d_dev[0..k) only (k = i+1), dconf[j] is
//    read back only for executed passes; the re-prime and the next round's pass 0 rewrite their
//    inputs (tap restore, draft_setup) first. No RNG is consumed by a draft pass.
//  - Readbacks: two pinned slots (pass i -> slot i & 1) + one event each, so pass i's value is read
//    while pass i+1 runs; the NULL-stream copy of pass i and its event are enqueued BEFORE pass
//    i+1 (a legacy-stream op issued after pass i+1 would wait for it).
// Escapes: --spec-pass=0 (the pre-HOST loop), --wp05-off=dds (the whole old path).
// Diagnostics: --spec-pass=always (speculate every eligible pass: maximal waste, the counters
// and hashes must not move), --spec-pass-xcheck=1 (after every restore, the head rows are
// diffed against a host snapshot taken before the pass: prints SPEC_XCHECK lines).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SpecMode { Off, Auto, Always }

fn spec_pass_mode() -> SpecMode {
    static M: std::sync::OnceLock<SpecMode> = std::sync::OnceLock::new();
    *M.get_or_init(|| match crate::opts::var(crate::opt!("spec-pass")).as_deref() {
        Ok("0") | Ok("off") => SpecMode::Off,
        Ok("always") => SpecMode::Always,
        _ => SpecMode::Auto,
    })
}

/// Auto threshold on the lane's continuation rate at pass i. Break-even: a wasted pass costs
/// ~1.1 ms of verify delay, a confirmed one saves ~50 µs, so speculate only at >= ~96%.
fn spec_pass_q() -> f32 {
    static Q: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *Q.get_or_init(|| crate::opts::var(crate::opt!("spec-pass-q")).ok().and_then(|s| s.parse::<f32>().ok())
        .filter(|q| (0.0..=1.0).contains(q)).unwrap_or(0.96))
}

/// TP-H: --spec-pass-skew=<rank> — diagnostic only (see spec_pass_next).
fn spec_skew_rank() -> Option<usize> {
    static R: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::opts::var(crate::opt!("spec-pass-skew")).ok().and_then(|s| s.parse::<usize>().ok()))
}

fn spec_xcheck_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("spec-pass-xcheck")).map_or(false, |v| !v.is_empty() && v != "0"))
}

/// EMA weight of one round's outcome in the per-(slot, pass) continuation rate.
const SPEC_EMA_A: f32 = 1.0 / 32.0;

/// FwdModel::spec_geom: the persistent head rows one draft pass at a position writes. The K|V
/// rows are fixed on the host (the lane's slot); the raw-key row and the pooled block are given
/// as their SLOT-0 row plus a per-slot stride, because the pass writes them at slot = the device
/// slot_ids[0] (xq_spec_rows resolves it on device; the XCHECK host copy reads it back).
#[derive(Debug, Clone, Copy)]
struct SpecGeom {
    kv0: u64, kv_stride: usize, rb: usize, nkvrows: usize,
    /// raw indexer-key row of the position in slot 0 (0 = no plane), slot stride, row bytes
    key0: u64, key_ss: usize, key_b: usize,
    /// pooled-plane block row of the position in slot 0 (0 = no plane / out of range), stride, bytes
    pl0: u64, pl_ss: usize, pl_b: usize,
    /// slots in the key/pool planes (a slot_ids[0] outside 0..nslots skips both rows)
    nslots: usize,
}
impl SpecGeom {
    fn bytes(&self) -> usize { self.nkvrows * self.rb + self.key_b + self.pl_b }
    /// Host addresses of the raw-key row and pooled block for key slot `ks` (0 = absent).
    fn key_pl_rows(&self, ks: i32) -> (u64, u64) {
        if ks < 0 || ks as usize >= self.nslots { return (0, 0); }
        let s = ks as u64;
        (if self.key0 != 0 { self.key0 + s * self.key_ss as u64 } else { 0 },
         if self.pl0 != 0 { self.pl0 + s * self.pl_ss as u64 } else { 0 })
    }
}

static SPEC_ROUNDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SPEC_LAUNCHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SPEC_WASTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Per-Scratch speculation state (lives in the decode Wp05Stage, next to the pinned readback).
pub(crate) struct SpecPass {
    /// readback-retired events of the two pinned readback slots (pass i -> slot i & 1)
    ev: [cudarc::driver::sys::CUevent; 2],
    /// [2*nkv K|V rows | raw key row | plane block row] of the speculated position (xq_spec_rows)
    save: CudaSlice<u8>,
    /// per slot: continuation-rate EMA per pass index, and where the lane's next round starts
    /// (a round that starts elsewhere = a new request in the slot: the EMA restarts at 1.0)
    ema: Vec<[f32; MTP_MAX_K]>,
    pub(super) next_p: Vec<usize>,
    /// --spec-pass-xcheck: (position, device slot_ids[0], host snapshot of the rows) taken
    /// before the pass
    xsnap: Option<(usize, i32, Vec<u8>)>,
    /// TP only: pass-graph keys the NORMAL (non-speculative) path has already run. Speculation is eligible only for
    /// keys in this set. The set is a function of the lockstep round sequence, so it is identical on every rank;
    /// the rank-local "is this graph captured" state is NOT (boot precapture stops at a per-rank memory cap, and graph
    /// bytes differ per rank), which let one rank speculate a pass its peer did not: "TP pre-verify FAILED: ranks chose
    /// different ... speculative head-pass launches" at --max-batch 16 (found 2026-10-03; the old build was fine at 8).
    pub(super) seen: std::collections::HashSet<usize>,
    dev: Arc<CudaDevice>,
}
// Raw CUevent handles; all use is on the owning scheduler thread through &mut self.
unsafe impl Send for SpecPass {}
unsafe impl Sync for SpecPass {}

impl SpecPass {
    pub(super) fn new(dev: &Arc<CudaDevice>, save_bytes: usize, slots: usize) -> Result<Self> {
        dev.bind_to_thread()?;
        let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING;
        let e0 = cudarc::driver::result::event::create(flags)?;
        let e1 = match cudarc::driver::result::event::create(flags) {
            Ok(e) => e,
            Err(e) => { unsafe { let _ = cudarc::driver::result::event::destroy(e0); } return Err(e.into()); }
        };
        // never read before the save kernel writes it (alloc_zeros is async; content irrelevant)
        let save = dev.alloc_zeros::<u8>(save_bytes.max(16))?;
        Ok(Self { ev: [e0, e1], save, ema: vec![[1.0; MTP_MAX_K]; slots.max(1)],
                  next_p: vec![usize::MAX; slots.max(1)], xsnap: None, seen: Default::default(), dev: dev.clone() })
    }
    /// Record readback slot `s`'s event on the NULL stream (after its two dtoh copies).
    pub(super) fn record(&self, s: usize) -> Result<()> {
        let r = unsafe { cudarc::driver::sys::cuEventRecord(self.ev[s & 1], *self.dev.cu_stream()) };
        anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "spec pass event record ({r:?})");
        Ok(())
    }
    /// Wait until readback slot `s` has landed (does NOT wait for work enqueued after it).
    pub(super) fn wait(&self, s: usize) -> Result<()> {
        // NIGHT1: under --gpu-wait (hybrid sleeps then polls instead of the driver's spin)
        crate::host_wait::event_wait(self.ev[s & 1]).map_err(|e| anyhow::anyhow!("spec pass event sync ({e:?})"))?;
        Ok(())
    }
}

impl Drop for SpecPass {
    fn drop(&mut self) {
        let _ = self.dev.bind_to_thread();
        for e in self.ev {
            unsafe { let _ = cudarc::driver::result::event::destroy(e); }
        }
    }
}

impl Wp05PfStage {
    pub(super) fn new(dev: &Arc<CudaDevice>, c: usize, ple: bool) -> Result<Self> {
        let bytes = (c * 5 + 4) * 4 + if ple { c * 2560 * 2 } else { 0 };
        Ok(Self { pin: PinnedHost::new(dev, bytes)?, c, inflight_tab: false, inflight_ple: false })
    }
    pub(super) fn off_ple(&self) -> usize { (self.c * 5 + 4) * 4 }
}

/// WP05: the decode stage mirrors this scratch's upload buffers exactly (width w >= need rows).
pub(super) fn wp05_stage_fits(sc: &Scratch, need: usize) -> bool {
    sc.wp05.as_ref().map_or(false, |s| {
        s.w >= need && s.w == sc.toks.len() && s.w == sc.pos.len() && s.w * 2 == sc.slots.len()
            && s.w == sc.slot_ids.len() && s.w * 2560 == sc.ple_emb.len()
            && sc.draft_meta.len() == 4 && sc.dconf.len() >= need && sc.d_dev.len() >= need
    })
}

impl FwdModel {
    // ---- HOST / RO-5(a) speculative next draft pass: helpers of mtp_round_adaptive (see SpecPass) ----

    /// Where a draft pass at position `x` of `slot` writes PERSISTENT head state: the 2*nkv K|V
    /// cache rows (xq_kv_make's (slot, K|V, kv-head, pos) rows of kv_rowbytes, one per plane of
    /// max_pos rows — addressed through sc.slots, i.e. the lane's `slot`), the raw indexer-key row
    /// ([s][max_pos][hdx] bf16) and the pooled-plane block row of x ([s][max_pos/ratio][hdx] bf16)
    /// — those two at s = the DEVICE slot_ids[0] (qsa_key_write_b / qsa_pool_update_b's `slot`),
    /// so they are returned as slot-0 rows + slot strides. Absent planes = (0, 0).
    fn spec_geom(&self, head: &DraftHead, slot: usize, x: usize) -> SpecGeom {
        let cfg = &self.cfg;
        let nkv = cfg.num_kv_heads;
        let rb = kv_rowbytes(self.kv_fmt, cfg.head_dim);
        let kv0 = *head.kv.device_ptr() as u64 + ((slot * 2 * nkv * self.max_pos + x) * rb) as u64;
        let hdx = cfg.indexer_head_dim;
        let (key0, key_ss, key_b) = match self.qsa_keys_p(HEAD_ATT) {
            Some(k) if hdx > 0 => (k + (x * hdx * 2) as u64, self.max_pos * hdx * 2, hdx * 2),
            _ => (0, 0, 0),
        };
        let (pl0, pl_ss, pl_b) = match self.qsa_pool_p(HEAD_ATT) {
            Some(pp) if hdx > 0 => {
                let nbs = (self.max_pos / cfg.indexer_compress_ratio.max(1)).max(1);
                let j = x / cfg.indexer_compress_ratio.max(1);
                if j < nbs { (pp + (j * hdx * 2) as u64, nbs * hdx * 2, hdx * 2) } else { (0, 0, 0) }
            }
            _ => (0, 0, 0),
        };
        SpecGeom { kv0, kv_stride: self.max_pos * rb, rb, nkvrows: 2 * nkv,
                   key0, key_ss, key_b, pl0, pl_ss, pl_b, nslots: self.width }
    }

    /// xq_spec_rows: dir 0 saves the rows of `spec_geom(x)` into `save`, dir 1 restores them.
    /// `d_sids` = sc.slot_ids (the key/pool rows' slot is read from slot_ids[0] on device).
    fn spec_rows(&self, l: &Launcher, save: &mut CudaSlice<u8>, d_sids: u64, head: &DraftHead, slot: usize,
                 x: usize, dir: i32) -> Result<()> {
        let g = self.spec_geom(head, slot, x);
        anyhow::ensure!(g.rb % 4 == 0 && g.key_b % 4 == 0 && g.pl_b % 4 == 0 && g.kv_stride % 4 == 0
                        && g.key_ss % 4 == 0 && g.pl_ss % 4 == 0
                        && g.key_b <= 0xFFFF && g.pl_b <= 0xFFFF && g.nslots <= 0xFFFF
                        && g.key_ss <= u32::MAX as usize && g.pl_ss <= u32::MAX as usize
                        && g.bytes() <= save.len() && d_sids != 0,
                        "spec pass rows: geometry {g:?} vs save {}", save.len());
        let sstrides = (g.key_ss as u64 | ((g.pl_ss as u64) << 32)) as i64;
        let kp_bytes = (g.key_b | (g.pl_b << 16)) as i32;
        let nslots_dir = (g.nslots as i32) | (dir << 16);
        xqlaunch!(l, "xq_spec_rows", ((g.nkvrows + 2) as u32, 1, 1), (128, 1, 1), 0,
                  (save, g.kv0, g.kv_stride as i64, g.rb as i32, g.nkvrows as i32,
                   g.key0, g.pl0, sstrides, kp_bytes, d_sids, nslots_dir))
    }

    /// --spec-pass-xcheck: synchronous host copy of the rows of `g` with the key/pool rows at
    /// key slot `ks` (the device slot_ids[0], read back by the caller), plus the (offset, bytes)
    /// segments actually present (absent rows stay zero in the copy and are not compared).
    /// Legacy-stream DtoH: it orders after every kernel enqueued before it on the blocking
    /// compute stream.
    fn spec_rows_host(&self, g: &SpecGeom, ks: i32) -> Result<(Vec<u8>, Vec<(usize, usize)>)> {
        let mut v = vec![0u8; g.bytes()];
        let (key_row, pl_row) = g.key_pl_rows(ks);
        let mut segs: Vec<(u64, usize)> = (0..g.nkvrows).map(|r| (g.kv0 + (r * g.kv_stride) as u64, g.rb)).collect();
        segs.push((key_row, g.key_b));
        segs.push((pl_row, g.pl_b));
        let mut present = Vec::with_capacity(segs.len());
        let mut off = 0usize;
        self.dev.bind_to_thread()?;
        for (src, n) in segs {
            if n > 0 && src != 0 {
                let r = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(v[off..].as_mut_ptr() as *mut std::ffi::c_void, src, n) };
                anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "spec xcheck dtoh ({r:?})");
                present.push((off, n));
            }
            off += n;
        }
        Ok((v, present))
    }

    /// --spec-pass-xcheck: the device slot_ids[0] (after a full synchronize).
    fn spec_key_slot_host(&self, sc: &Scratch) -> Result<i32> {
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
        Ok(self.dtoh(&sc.slot_ids)?.first().copied().unwrap_or(-1))
    }

    /// The DDS loop's per-pass graph key for pass `i` of a round at `p`: ONE formula for the
    /// loop's own launch/capture key and the speculative launch of the next pass (review item:
    /// a second copy could silently diverge on a merge that changes the key; the loop still
    /// checks the two agree when it consumes a speculatively launched pass).
    pub(super) fn dds_pass_key(&self, sc: &Scratch, slot: usize, p: usize, i: usize) -> usize {
        // p3c merge (HOST x WP04): p2's formula — pkey_of (== the boot precapture's key) with the
        // WP04 KEY bucket (0 when the scorer is bucket-free), plus the WP15 draft-mirror bit.
        let pq = self.qsa_live(p + i);
        let (_, pb) = self.qsa_grid_bucket_key(p + i);
        pkey_of(slot, i, pq, pb) | self.pen_draft_key(sc) | sk(sc) // WP15 draft mirror + DHEADP bits + TP-C scratch id
    }

    /// DHEADP (w3): which tail a draft pass runs — ONE decision for draft_pass and the draft graph
    /// keys (a graph records exactly one tail, so the key must say which).
    pub(super) fn draft_tail(&self, sc: &Scratch) -> DraftTail {
        // WP24: a real-q pass takes the old tail (full slice head, + xq_pen_draft when penalized):
        // DHEAD's candidate set is only guaranteed to hold the argmax's block, not the slice's exact
        // top-K. A5-L4 (spec.ratio_dhead=1): an UNPENALIZED real-q pass takes DHEAD anyway, with q'
        // built over the candidate set (exact for any q' the draft is drawn from and judged against).
        if sc.rq_mode == 2 {
            return match (self.dhead.as_ref(), self.lm_head_draft.as_ref()) {
                (Some(dh), Some(_)) if dh.rq_live() && !(sc.pen_live && pen_draft_on()) => DraftTail::DheadRq,
                _ => DraftTail::Old,
            };
        }
        match (self.dhead.as_ref(), self.lm_head_draft.as_ref()) {
            (Some(dh), Some(_)) => {
                if !(sc.pen_live && pen_draft_on()) {
                    DraftTail::Dhead
                } else if sc.pen_mono && dh.penalized_live() {
                    DraftTail::DheadP
                } else {
                    DraftTail::Old
                }
            }
            _ => DraftTail::Old,
        }
    }

    /// The draft graph keys' penalty bits: bit 40 = the WP15 draft mirror is live (as before
    /// DHEADP), bit 41 = the penalized passes take DHEADP (a monotone and a non-monotone request
    /// never share a graph). --dheadp-off=1 leaves bit 41 clear: today's keys exactly.
    /// WP24: + bits 42-43 = the draft passes' rq mode (rq_draft_key; 0 by default).
    pub(super) fn pen_draft_key(&self, sc: &Scratch) -> usize {
        let dheadp = self.draft_tail(sc) == DraftTail::DheadP;
        if dheadp {
            if let Some(dh) = self.dhead.as_ref() { dh.mark_penalized_round(); }
        }
        (((sc.pen_live && pen_draft_on()) as usize) << 40) | ((dheadp as usize) << 41)
            | rq_draft_key(sc)
    }

    /// Round start: is the speculation live this round? Creates the state on first use and
    /// restarts the slot's continuation EMA when the round does not continue the lane's last one
    /// (a new request in the slot; `+1` = a control round in between).
    pub(super) fn spec_pass_begin(&self, sc: &mut Scratch, slot: usize, p: usize) -> Result<bool> {
        static BROKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mode = spec_pass_mode();
        if mode == SpecMode::Off || BROKEN.load(std::sync::atomic::Ordering::Relaxed) { return Ok(false); }
        let Some(st) = sc.wp05.as_mut() else { return Ok(false) };
        if st.spec.is_none() {
            let cfg = &self.cfg;
            let bytes = 2 * cfg.num_kv_heads * kv_rowbytes(self.kv_fmt, cfg.head_dim) + 2 * (2 * cfg.indexer_head_dim);
            match SpecPass::new(&self.dev, bytes, self.width) {
                Ok(s) => {
                    crate::dprintln!("[spec-pass] ON (--spec-pass={}: continuation threshold {:.3}, EMA 1/32 per (slot, pass); \
                              graphed passes only; a wasted pass's head rows are restored) — --spec-pass=0 = the pre-HOST loop",
                             if mode == SpecMode::Always { "always" } else { "auto" }, spec_pass_q());
                    st.spec = Some(s);
                }
                Err(e) => {
                    // TP-H: a one-sided degrade would leave the peer speculating (and, with a sharded
                    // head, issuing head-pass collectives this rank does not) — fail loudly instead.
                    if self.tp.is_some() {
                        anyhow::bail!("[spec-pass] state creation failed under TP ({e:#}) — refusing a one-sided \
                                       speculation-off (--spec-pass=0 on both ranks is the explicit escape)");
                    }
                    BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
                    crate::dprintln!("[spec-pass] OFF: state creation failed ({e:#}) — the pre-HOST loop runs");
                    return Ok(false);
                }
            }
        }
        let sp = st.spec.as_mut().unwrap();
        if slot >= sp.ema.len() {
            sp.ema.resize(slot + 1, [1.0; MTP_MAX_K]);
            sp.next_p.resize(slot + 1, usize::MAX);
        }
        let np = sp.next_p[slot];
        if np != p && np.wrapping_add(1) != p {
            sp.ema[slot] = [1.0; MTP_MAX_K];
        }
        sp.xsnap = None;
        let r = SPEC_ROUNDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if r % 128 == 0 { // cumulative; >= 1 line per 1,024-token code request (~180 rounds)
            let (la, wa) = (SPEC_LAUNCHED.load(std::sync::atomic::Ordering::Relaxed),
                            SPEC_WASTED.load(std::sync::atomic::Ordering::Relaxed));
            crate::drprintln!("[spec-pass] rounds {r} launched {la} ({:.2}/round) wasted {wa} ({:.1}% of launched)",
                     la as f64 / r as f64, 100.0 * wa as f64 / la.max(1) as f64);
        }
        Ok(true)
    }

    /// At iteration i (pass i launched, its readback + event queued): launch pass i+1 NOW when the
    /// predictor says it will very likely run — save its position's head rows, replay its graph,
    /// queue its {dconf, d} readback into pinned slot (i+1) & 1 and record that slot's event.
    /// Returns the launched pass's graph key (None = not launched: no graph yet, context end,
    /// or the lane's continuation rate at pass i is below the threshold).
    pub(super) fn spec_pass_next(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, slot: usize, p: usize,
                      i: usize) -> Result<Option<usize>> {
        let j = i + 1;
        let x = p + j;
        if j >= MTP_MAX_K || x >= self.max_pos { return Ok(None); }
        // TP-H diagnostic negative control (--spec-pass-skew=<rank>, rides the env snapshot): that
        // rank never speculates while its peer may — the pre-verify launch-count agreement must
        // fire (abort before the verify), never mis-pair a barrier.
        if let (Some(t), Some(r)) = (self.tp.as_ref(), spec_skew_rank()) {
            if t.rank as usize == r { return Ok(None); }
        }
        let go = {
            let sp = sc.wp05.as_ref().and_then(|s| s.spec.as_ref()).context("spec pass state")?;
            match spec_pass_mode() {
                SpecMode::Always => true,
                SpecMode::Auto => sp.ema.get(slot).map_or(false, |e| e[i] >= spec_pass_q()),
                SpecMode::Off => false,
            }
        };
        if !go { return Ok(None); }
        let key = self.dds_pass_key(sc, slot, p, j);
        if !self.graphs.lock().unwrap().contains_key(&key) { return Ok(None); }
        // TP: only a pass the normal path has already run (identical on every rank — see SpecPass::seen)
        if self.tp.is_some() && !sc.wp05.as_ref().and_then(|s| s.spec.as_ref()).map_or(false, |sp| sp.seen.contains(&key)) {
            return Ok(None);
        }
        let (d_conf, d_d) = (*sc.dconf.device_ptr() as u64, *sc.d_dev.device_ptr() as u64);
        let d_sids = *sc.slot_ids.device_ptr() as u64;
        // pass i may refresh the same pooled block: snapshot once it has finished
        let xsnap = if spec_xcheck_on() {
            let ks = self.spec_key_slot_host(sc)?;
            Some((x, ks, self.spec_rows_host(&self.spec_geom(head, slot, x), ks)?.0))
        } else { None };
        let st = sc.wp05.as_mut().unwrap();
        let rb = st.off_rb() + 8 * (j & 1);
        let Wp05Stage { pin, spec, .. } = st;
        let sp = spec.as_mut().unwrap();
        if xsnap.is_some() { sp.xsnap = xsnap; }
        self.spec_rows(l, &mut sp.save, d_sids, head, slot, x, 0)?;
        self.graphs.lock().unwrap().get(&key).unwrap().launch();
        pin.dtoh(rb, d_conf + 4 * j as u64, 4)?;
        pin.dtoh(rb + 4, d_d + 4 * j as u64, 4)?;
        sp.record(j)?;
        SPEC_LAUNCHED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(key))
    }

    /// Loop end: restore the head rows of a wasted speculative pass (BEFORE the verify and the
    /// re-prime are enqueued: the re-prime may legitimately rewrite that row), XCHECK them, and
    /// fold this round's outcome into the slot's continuation EMA (pass j ran => pass j+1 ran?).
    pub(super) fn spec_pass_end(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, slot: usize, kmax: usize,
                     w: usize, wasted: Option<usize>) -> Result<()> {
        let d_sids = *sc.slot_ids.device_ptr() as u64;
        if let Some(x) = wasted {
            let sp = sc.wp05.as_mut().and_then(|s| s.spec.as_mut()).context("spec pass state")?;
            self.spec_rows(l, &mut sp.save, d_sids, head, slot, x, 1)?;
            SPEC_WASTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Some((sx, sks, snap)) = sp.xsnap.take().filter(|_| spec_xcheck_on()) {
                let ks = self.spec_key_slot_host(sc)?;
                let g = self.spec_geom(head, slot, x);
                let (now, present) = self.spec_rows_host(&g, ks)?;
                let sp = sc.wp05.as_ref().and_then(|s| s.spec.as_ref()).context("spec pass state")?;
                let saved = self.dtoh(&sp.save)?;
                let bad = now.iter().zip(snap.iter()).filter(|(a, b)| a != b).count();
                // the save buffer holds only the rows that exist (an absent row is never saved)
                let bad_s: usize = present.iter()
                    .map(|&(o, n)| saved[o..o + n].iter().zip(&snap[o..o + n]).filter(|(a, b)| a != b).count()).sum();
                crate::dprintln!("SPEC_XCHECK slot {slot} pos {x} (snap pos {sx}) key/pool slot {ks} (snap {sks}{}): head rows after restore: \
                          {bad} of {} bytes differ from the pre-speculation snapshot (K|V {}x{} B, key {} B, plane {} B); \
                          save buffer {bad_s} differ — {}",
                         if ks >= 0 && ks as usize != slot { ", cross-slot: device slot_ids[0] != lane slot" } else { "" },
                         snap.len(), g.nkvrows, g.rb, g.key_b, g.pl_b,
                         if bad == 0 && bad_s == 0 && sx == x && sks == ks { "OK" } else { "MISMATCH" });
            }
        }
        let sp = sc.wp05.as_mut().and_then(|s| s.spec.as_mut()).context("spec pass state")?;
        if let Some(e) = sp.ema.get_mut(slot) {
            for jj in 0..w.min(MTP_MAX_K) {
                if jj + 1 < kmax {
                    let ran = if jj + 1 < w { 1.0f32 } else { 0.0 };
                    e[jj] += SPEC_EMA_A * (ran - e[jj]);
                }
            }
        }
        Ok(())
    }

}
