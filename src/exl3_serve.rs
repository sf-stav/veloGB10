//! S-A3-e — EXL3 first-class serve backend (TP=1; S-A3-g owns TP).
//!
//! Consumes the SAME `BatchRequest` protocol server.rs speaks, so the entire HTTP surface
//! (/v1/chat/completions, /v1/completions, /v1/tokenize, /v1/models, /health) is 100% shared
//! with the NVFP4 serving path. The engine underneath is the S-A3-d EXL3 stack
//! (`FwdModel::forward_step`), with the width>1 kernel fixes (xq_hc_norm / xq_hc_inject) that
//! make DISTINCT concurrent lanes correct.
//!
//! Posture: MTP chain-verify for greedy AND sampled lanes (API-parity G2: sampled lanes sample
//! every verify row on device and accept while sample == draft — distribution-exact with the
//! argmax draft; greedy stays bitwise lossless). The §6 profitability control is PER REQUEST
//! (API-parity G1): a request whose drafts do not pay runs plain; the next request speculates.

use crate::batch::{BatchRequest, TokEvent};
use crate::exl3_forward::{FwdModel, Scratch};
use crate::server::{create_router, AppState};
use crate::tokenizer::{QwenTokenizer, ThinkingMode};
use anyhow::{Context as _, Result};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

struct Lane {
    pos: usize,          // next trunk position to write
    last_tok: u32,       // last committed token
    generated: usize,
    max_new: usize,
    min_new: usize,
    ignore_eos: bool,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    // WP15: this request's penalties (None = unpenalized: no penalty kernel runs for it).
    pen: Option<crate::exl3_forward::PenParams>,
    // WP08: the rival's streaming loop detector (None = --loop-detect off) + log context.
    loop_det: Option<crate::loop_detect::LoopDetector>,
    prompt_hash: u64,
    history: Vec<u32>,
    // API-parity G2: sampler stream (uniform = hash(seed, ctr); ctr advances per sampled row).
    seed: u64,
    ctr: u32,
    // WP24: the real-q draft RNG counter (draft pass i of a round draws dctr + i; a separate
    // stream from the verify rows' `ctr`). Advances by the served depth per real-q round.
    dctr: u32,
    // WP24 gate (--probe-spec-sampling): true until the request's first MTP step (its first
    // round's d_0 feeds the gate's negative control). Bookkeeping only.
    gate_first: bool,
    // API-parity G1: this request's MTP round-time and tokens-per-round EMAs + rounds run.
    // cost = ema_ms / ema_tok (ratio of means). NOT an EMA of per-round ms/tok: that mean
    // of ratios over-weights accept-0 rounds (Jensen) and read ~34 ms/tok for rounds that
    // really cost ~22, turning MTP off on every request (2026-09-25, owner: "SUPER SLOW").
    ema_ms: f64,
    ema_tok: f64,
    mtp_rounds: usize,
    wp23: Wp23Lane, // WP23: DDS cost-guard fit + per-width round counts (the [dds] line)
    // per-request round anatomy, logged at finish ([mtp-stats]): comparable to the rival's
    // draft_accepted/draft_rejected counters (tokens = accepted + rounds).
    st_rounds: usize,
    st_drafted: usize,
    st_accepted: usize,
    st_ms: f64,
    st_plain: usize,
    st_ctrl: usize, // WP03: control rounds this request ran (the cadence gate reads it)
    // S-A3-f-d: chain-verify state. tap_row = taps_keep row holding the tap that
    // predicted last_tok (usize::MAX after a control round: resid holds it).
    mtp: Option<usize>,
    tx: tokio::sync::mpsc::UnboundedSender<TokEvent>,
}

struct Exl3Scheduler {
    model: Arc<FwdModel>,
    sc: Scratch,
    width: usize,
    lanes: Vec<Option<Lane>>,
    eos: Vec<u32>,
    chunk: usize, // S-A3-f-b: prefill chunk width (tokens per forward sweep)
    psc: Option<crate::exl3_forward::PrefillScratch>,
    // S-A3-f-d MTP policy: depth k (--exl3-mtp-k, default MTP_MAX_K = 7 since WP23), auto-disable
    // when the measured per-token cost loses to plain (§6), control-round bookkeeping.
    mtp_k: usize,
    ctrl_round: usize,
    // API-parity G1: plain decode cost (ms/token at m=1), refreshed by control rounds. It is
    // a property of the ENGINE (not of the content), so it persists across requests; the
    // MTP side is per request (Lane::ema_mtp) because acceptance is a property of content.
    ema_plain: f64,
    // API-parity DDS: dynamic draft stop target (0 = fixed depth) + the shared calibrator.
    draft_conf: f64,
    cal: DraftCal,
    // Prefix cache (2026-09-26): per slot, the prompt tokens whose recurrent state is snapshotted
    // (prompt[..snap]) — a new prompt that starts with them resumes prefill at `snap`.
    prefix_on: bool,
    cache: Vec<Option<Vec<u32>>>,
    // WP16: intermediate recurrent checkpoints at aligned chunk ends (None = off: --prefix-cache
    // off, --wp16-off=1, or a --prefix-ckpt-mem-gb cap that holds none) + the XCHECK diagnostic.
    wp16: Option<crate::exl3_forward::CkptStore<crate::exl3_forward::RecurCkpt>>,
    wp16_xcheck: bool,
    // WP08: stop_on_loop (window, min_reps) — the rival's (300, 3) unless --loop-detect off.
    loop_cfg: Option<(usize, usize)>,
    // WP15: penalty window (sustain = decay = this; the rival's penalty_range 1024).
    pen_range: usize,
    // WP24: --spec-sampling ratio (real-q: sampled drafts, accept min(1, p/q'), exact residual)
    // for sampled lanes; false = match (today). draft_temp = --draft-temperature (None = the
    // request's temperature).
    spec_ratio: bool,
    draft_temp: Option<f32>,
    // WP24 gate (--probe-spec-sampling) round bookkeeping; None in the server.
    gate: Option<GateStats>,
    // for the [loop] log's text tail only
    tok: Arc<QwenTokenizer>,
    // WP02 liveness. exit_on_fatal = --exit-on-fatal (opt-in, owner decision): a sticky CUDA error
    // exits 70 so a supervisor restarts the server; off, the engine goes DEAD (/health and every
    // request answer 503). `fatal` is set by admit/step when the context is poisoned.
    exit_on_fatal: bool,
    fatal: Option<String>,
    inject_panic: Option<usize>, // --inject-panic-steps=<n>: panic at decode step n (liveness gate)
    steps: usize,
    // TP-C: Some = this scheduler is one rank of a TP group (SPMD lockstep, head-authoritative timing)
    tp: Option<TpSync>,
    // TP-D: the last admit's resume points (prompt-end reuse, WP16/C1 checkpoint q) — proven
    // identical on every rank by the seam lockstep
    last_resume: (usize, usize),
}

/// WP24 gate: MTP round anatomy of one arm (non-plain rounds only) + the first-round drafts.
#[derive(Default, Debug)]
struct GateStats {
    rounds: u64,
    drafted: u64,
    accepted: u64,
    ratio_rounds: u64,
    d0: Vec<i32>,
    // TP-C harness: MTP round wall ms (the head's under TP), control rounds, plain steps
    ms: f64,
    ctrl: u64,
    plain: u64,
}

/// TP-C: the SPMD lockstep of a TP scheduler (None at TP=1 — every hook below is then a no-op and
/// the TP=1 serve path is unchanged). Every rank runs the same scheduler on the same requests;
/// the head (rank 0) is AUTHORITATIVE for every decision that reads a wall clock (design §4.2):
/// after each round the ranks agree on (step, accept, verify width, hash of emitted ids + drafts +
/// draft confidences) — the load-bearing agree() guard, AGENTS §2.10 — and the node ADOPTS the
/// head's measured round / plain-step ms and graph-capture flag. Everything downstream (the §6
/// per-request MTP-off, the control cadence, the WP23 cost guard's target, the cost fit) is then a
/// pure function of replicated state, so both ranks choose the same draft stop and the same verify
/// width by construction; the agree token proves it every round.
/// TP-E: per-call-site wall time of the host lockstep waits (always on; ~50 ns per call). Printed and
/// reset whenever the engine goes idle, so every line covers one request (or one idle-to-idle span).
const WP_SITES: [&str; 8] = ["fenceA", "xchg", "fenceB", "pv_xchg", "pv_agree", "ctl_recv", "ctl_send", "step_go"];
const WP_EDGES_US: [u64; 5] = [20, 50, 200, 1000, 2000];
#[derive(Default, Clone)]
pub(crate) struct WaitProf {
    n: [u64; 8],
    sum_ns: [u64; 8],
    max_ns: [u64; 8],
    hist: [[u64; 6]; 8],
    sleeps: [u64; 8],
    rounds: u64,
}

impl WaitProf {
    #[inline]
    fn rec(&mut self, site: usize, t0: std::time::Instant, s0: u64) {
        let ns = t0.elapsed().as_nanos() as u64;
        self.n[site] += 1;
        self.sum_ns[site] += ns;
        self.max_ns[site] = self.max_ns[site].max(ns);
        let us = ns / 1000;
        let b = WP_EDGES_US.iter().position(|&e| us < e).unwrap_or(WP_EDGES_US.len());
        self.hist[site][b] += 1;
        self.sleeps[site] += crate::net::wait_sleeps().saturating_sub(s0);
    }
    fn line(&self, rank: i32) -> String {
        let mut o = format!("[tp-wait] rank {rank} rounds {}", self.rounds);
        let mut tot = 0u64;
        for i in 0..WP_SITES.len() {
            if self.n[i] == 0 { continue; }
            tot += if i < 5 || i == 7 { self.sum_ns[i] } else { 0 };
            let h = &self.hist[i];
            o += &format!(" | {} n {} mean {:.1}us max {:.0}us sleeps {} h[<20,<50,<200,<1k,<2k,>=2k]us {}/{}/{}/{}/{}/{}",
                          WP_SITES[i], self.n[i], self.sum_ns[i] as f64 / self.n[i] as f64 / 1e3,
                          self.max_ns[i] as f64 / 1e3, self.sleeps[i], h[0], h[1], h[2], h[3], h[4], h[5]);
        }
        if self.rounds > 0 {
            o += &format!(" | lockstep(fenceA+xchg+fenceB+pv+go) {:.3} ms/round", tot as f64 / self.rounds as f64 / 1e6);
        }
        // TP-G: transport health (cumulative): tail-guard fires (MUST stay 0) and GPU-proved epochs
        o += &format!(" | xport tail_fires {} gpu_rx_skips {} abort {} reuse-gate binds {}", crate::net::traced_tail_fires(),
                      crate::net::traced_gpu_rx_skips(), crate::net::traced_abort_status(), crate::net::traced_gate_waits());
        o
    }
}

pub(crate) struct TpSync {
    pub rank: i32,
    pub wp: WaitProf,
    /// agree_ext step counter (24-bit field on the wire)
    step: u64,
    /// G-T1-b graphed: digest the logits rows + the tap of every step/round and compare across
    /// ranks (a dtoh per step — the harness's identity pass only)
    pub ident: bool,
    pub ident_bufs: u64,
    pub ident_bad: u64,
    pub agrees: u64,
    /// TP-D: pre-verify agrees (DDS: the verify width + drafts, BEFORE the verify graph launches)
    pub pre_agrees: u64,
    /// TP-D: the serve control plane (rank 0: one stream per node; a node: the stream to the head).
    /// Empty in the TP-C harness (run_spec), where nothing is head-decided mid-step.
    ctl: Vec<std::net::TcpStream>,
}

impl TpSync {
    pub(crate) fn new(rank: i32) -> Self {
        TpSync { rank, step: 0, ident: false, ident_bufs: 0, ident_bad: 0, agrees: 0, pre_agrees: 0, ctl: Vec::new(),
                 wp: WaitProf::default() }
    }

    /// TP-D: a boolean the HEAD decides inside a step (a prefill chunk boundary's "client gone"),
    /// shipped over the control stream; every rank returns the head's value at the same program
    /// point. No control plane (the harness) = false on every rank (nothing is cancellable there).
    fn head_flag(&mut self, v: bool) -> Result<bool> {
        if self.ctl.is_empty() {
            return Ok(false);
        }
        if self.rank == 0 {
            for s in self.ctl.iter_mut() {
                crate::tp_serve::send_serving(s, &crate::tp_serve::ServingMsg::HeadFlag { v })?;
            }
            Ok(v)
        } else {
            match crate::tp_serve::recv_serving(&mut self.ctl[0])? {
                crate::tp_serve::ServingMsg::HeadFlag { v } => Ok(v),
                other => anyhow::bail!("TP control plane out of step: expected HeadFlag, got {other:?}"),
            }
        }
    }

    /// TP-E: the per-step "go" of a LIVE engine over the RDMA link instead of the TCP control stream.
    /// Measured (TP-E [tp-wait]): the node waited ~0.53 ms per round in the TCP recv of an EMPTY Step
    /// (kernel TCP over the CX7 netdev: NIC interrupt moderation + wakeup), and the head then waited
    /// the same ~0.5 ms at its pre-verify exchange — the node started every round half a millisecond
    /// late. Now, whenever a lane is live (replicated state, so both ranks take this path together),
    /// the head publishes (step number, event count) in an RDMA exchange (~8 us); the TCP Step is sent
    /// and read ONLY when the step carries events (admit / cancel — a few per request). Shape =
    /// exchange (rendezvous) then agree (fence): the exchange reuses hot-path ring slot memory, and a
    /// plain / seam step may run collectives right after it (TP-C fence-B rule). The node adopts the
    /// head's words; the head's own words come back as the node's zeros (ignored).
    /// `--exl3-tp-fastctl=0` (diagnostic, rides the env snapshot) = the TP-D TCP-every-step path.
    fn step_go(&mut self, step_no: u64, n_events: usize) -> Result<(u64, usize)> {
        let tag = 0x60_000000u32 | (step_no as u32 & 0xFF_FFFF);
        let mine: [u32; 3] = if self.rank == 0 { [tag, step_no as u32, n_events as u32] } else { [0, 0, 0] };
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, 3 + 4)?;
        let (hs, hn) = if self.rank == 0 { (step_no, n_events) } else {
            anyhow::ensure!(peer[0] & 0xFF00_0000 == 0x6000_0000, "TP step-go: head sent tag {:#x}", peer[0]);
            (((step_no & !0xFFFF_FFFF) | peer[1] as u64), peer[2] as usize)
        };
        // the fence takes the next lockstep step number (a plain agree field, like fence A): its field
        // then differs from the stale fence-B (bit 23) token it replaces and from the next pre-verify's.
        self.step += 1;
        let r = tp_agree_eq(agree_field(self.step, AGREE_GO), 0, 0, (hs as u32) ^ ((hn as u32) << 24) ^ 0x60);
        self.wp.rec(7, t0, s0);
        if let Err(why) = r {
            crate::net::abort_link();
            anyhow::bail!("TP step-go fence FAILED at control step {hs}: {why}");
        }
        Ok((hs, hn))
    }

    /// TP-D (design §4.2 belt-and-braces): agree on the DDS verify width + the drafts BEFORE the
    /// verify graph launches — the width selects the graph and therefore the barrier sequence, so a
    /// divergent draft stop must be caught before any mismatched barrier runs, not one round later.
    ///
    /// Shape = exchange (rendezvous) -> agree (fence): net_agree is ONE mirrored token slot, so two
    /// agrees with no rendezvous between them race (the peer, asleep in its spin, never sees the
    /// first token before the second overwrites it — the 2026-09-29 served smoke died exactly so:
    /// fence B of round n, then this agree after draft passes that have no collective). The
    /// exchange is the rendezvous that makes the fence-B token observed before this one is
    /// written; the agree after it is the fence that keeps the verify's ring epochs off the
    /// exchange slots (TP-C's fence-B rule).
    ///
    /// TP-H: `launched` = the head draft passes this rank issued this round (w, plus one when a
    /// speculative pass ran past the DDS stop). The sharded MTP head puts collectives in every draft
    /// pass, so a rank-local speculative-launch decision that diverged would mis-pair the verify's
    /// first barrier with the wasted pass's; the count is agreed HERE, before the verify launches
    /// (fail-stop, never a mis-paired all-reduce). Free at world 2 with a replicated head too.
    fn pre_verify(&mut self, w: usize, drafts: &[i32], launched: usize) -> Result<()> {
        let step = agree_field(self.step + 1, AGREE_PV);
        let h = fnv32(0x5EED, drafts.iter().map(|&d| d as u32));
        let tag = 0x9E00_0000u32 | (step as u32 & 0xFFFF);
        let mine = [tag, w as u32, h, launched as u32];
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, 4 + 4)?; // + the 16-byte tail guard (as tp_round)
        self.wp.rec(3, t0, s0);
        if peer != mine {
            crate::net::abort_link();
            anyhow::bail!("TP pre-verify FAILED (round {}): this rank width {w} drafts {h:08x} head passes launched {launched}, \
                           peer {:?} — ranks chose different verify widths, drafts or speculative head-pass launches; \
                           link aborted before the verify", self.step + 1, peer);
        }
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let r = tp_agree_eq(step, w.min(255) as u8, (w + 1).min(15) as u8, h);
        self.wp.rec(4, t0, s0);
        if let Err(why) = r {
            crate::net::abort_link();
            anyhow::bail!("TP pre-verify agree FAILED (round {}, draft width {w}, drafts hash {h:08x}: {why}) — ranks chose \
                           different verify widths or drafts; link aborted before the verify", self.step + 1);
        }
        self.pre_agrees += 1;
        Ok(())
    }
}

/// TP-E: the 24-bit agree step field = [kind:2 | lockstep counter mod 2^22]. `net_agree` returns the
/// peer's CURRENT token as soon as its step field matches, so the field of every agree must differ from
/// the peer's previous token (the stale value a waiting rank may read). Adjacent agrees always differ in
/// kind (B -> GO -> PV -> A -> B, or B -> A on plain rounds), and a same-kind collision needs the counter
/// 2^22 apart. (Before TP-E, fence A used the raw 24-bit counter and pre-verify `S & 0x3FFFFF | 1<<22`:
/// equal once S reached [2^22, 2^23) — ~4 M rounds — which the new comparison would turn into an abort.)
const AGREE_A: u64 = 0;
const AGREE_PV: u64 = 1;
const AGREE_B: u64 = 2;
const AGREE_GO: u64 = 3;
fn agree_field(counter: u64, kind: u64) -> u64 { (counter & 0x3F_FFFF) | (kind << 22) }

/// TP-E: the lockstep agree WITH the comparison. `net::agree_ext` only rendezvouses on the step field
/// and returns the PEER's (accept, hash); until TP-E the EXL3 callers checked `is_none()` alone, so a
/// divergent accept count / emitted-ids hash / boot config would have sailed through every "agree ok"
/// (only the timeout could fire). The NVFP4 path (batch.rs) always compared. Ok(()) = both ranks
/// published the identical token; Err = timeout/abort or a mismatch (message says which).
pub(crate) fn tp_agree_eq(step: u64, accept: u8, k_verify: u8, hash: u32) -> std::result::Result<(), String> {
    let h_ext = hash ^ ((k_verify as u32 & 0xF) << 27);
    match crate::net::agree_ext(step, accept, k_verify, hash) {
        None => Err("link aborted or peer timeout".into()),
        Some((pa, ph)) if pa == accept && ph == h_ext => Ok(()),
        Some((pa, ph)) => Err(format!("MISMATCH: this rank (accept {accept}, hash {h_ext:08x}) vs peer (accept {pa}, hash {ph:08x})")),
    }
}

fn fnv32(h: u32, words: impl Iterator<Item = u32>) -> u32 {
    let mut h = if h == 0 { 0x811c_9dc5 } else { h };
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
    }
    h
}

/// WP02: a CUDA error that poisons the context — every later call fails too, so the engine is dead
/// (cudarc's DriverError prints the CUresult name; anyhow's {:?} carries the whole chain).
fn sticky_cuda_error(e: &anyhow::Error) -> bool {
    let s = format!("{e:?}");
    ["ILLEGAL_ADDRESS", "LAUNCH_FAILED", "MISALIGNED_ADDRESS", "ILLEGAL_INSTRUCTION", "ECC_UNCORRECTABLE",
     "HARDWARE_STACK_ERROR", "INVALID_PC", "INVALID_ADDRESS_SPACE", "CUDA_ERROR_ASSERT", "LAUNCH_TIMEOUT",
     "CONTEXT_IS_DESTROYED"].iter().any(|t| s.contains(t))
}

/// WP02: the scheduler thread's liveness guard. A panic unwinding out of the scheduler drops its
/// lanes' senders (every in-flight request then ends "error: engine stopped", never "length"),
/// then this marks the engine DEAD; with --exit-on-fatal it exits 70 for the supervisor. An abort
/// (a panic inside a CudaSlice drop, AGENTS §5a) never reaches here: the supervisor restarting on
/// abort is that half.
struct FatalGuard {
    exit_on_fatal: bool,
}

impl Drop for FatalGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            crate::server::set_engine_dead("scheduler thread panicked");
            if self.exit_on_fatal {
                eprintln!("[exl3-serve] FATAL: scheduler thread panicked — exit 70 (--exit-on-fatal)");
                std::thread::sleep(std::time::Duration::from_millis(300));
                std::process::exit(70);
            }
        }
    }
}

/// Port of exllamav3's DraftConfidenceCalibrator (vcruz305 fork 523ecd3,
/// exllamav3/generator/draft_confidence.py): an online map from the draft head's argmax LOGIT
/// to the observed acceptance rate, in 1-logit bins of exponentially decayed (tested, accepted)
/// counts. One calibrator for the whole server, as in the rival (one per generator).
pub(crate) struct DraftCal {
    bins: std::collections::BTreeMap<i64, (f64, f64)>,
    total: f64,
    // WP23: depth-keyed bins [depth bucket] -> logit bin -> (tested, accepted), fed alongside the
    // pooled map (which stays byte-for-byte the pre-WP23 calibrator). None = logit-only (K <= 5).
    dbins: Option<Vec<std::collections::BTreeMap<i64, (f64, f64)>>>,
}

impl DraftCal {
    const BIN_W: f64 = 1.0;
    const DECAY: f64 = 0.995;
    const MIN_COUNT: f64 = 8.0;
    const BURN_IN: f64 = 64.0;
    pub(crate) fn new() -> Self {
        let k = crate::exl3_forward::mtp_depth_default();
        let dbins = if wp23_on(crate::opt!("exl3-dds-depth-bins"), k) {
            Some(vec![Default::default(); WP23_DEPTH_BUCKETS])
        } else {
            None
        };
        Self { bins: Default::default(), total: 0.0, dbins }
    }
    /// `i` = the draft's pass index in its round (0 = the first draft); only the WP23 depth bins
    /// read it — the pooled map is updated exactly as before.
    pub(crate) fn add_label(&mut self, i: usize, score: f32, accepted: bool) {
        let idx = (score as f64 / Self::BIN_W).floor() as i64;
        let b = self.bins.entry(idx).or_insert((0.0, 0.0));
        b.0 += 1.0;
        if accepted { b.1 += 1.0; }
        self.total += 1.0;
        if let Some(d) = self.dbins.as_mut() {
            let b = d[wp23_depth_bucket(i)].entry(idx).or_insert((0.0, 0.0));
            b.0 += 1.0;
            if accepted { b.1 += 1.0; }
        }
    }
    /// Once per verification round, so the map tracks drift (prose vs code) over ~hundreds of rounds.
    pub(crate) fn decay_step(&mut self) {
        for b in self.bins.values_mut() { b.0 *= Self::DECAY; b.1 *= Self::DECAY; }
        self.total *= Self::DECAY;
        if let Some(d) = self.dbins.as_mut() {
            for m in d.iter_mut() {
                for b in m.values_mut() { b.0 *= Self::DECAY; b.1 *= Self::DECAY; }
            }
        }
    }
    /// WP23: acceptance estimate for pass `i`'s draft. Depth bins on: the nearest populated bin
    /// at or below the score in i's depth bucket; a bucket with none there (sparse at deep levels
    /// on prose) falls back to the pooled estimate. Depth bins off: exactly `estimate`.
    pub(crate) fn estimate_at(&self, i: usize, score: f32) -> f64 {
        let Some(d) = self.dbins.as_ref() else { return self.estimate(score) };
        if self.total < Self::BURN_IN { return 1.0; }
        let idx = (score as f64 / Self::BIN_W).floor() as i64;
        let near = d[wp23_depth_bucket(i)].range(..=idx).rev()
            .find(|(_, v)| v.0 >= Self::MIN_COUNT).map(|(_, v)| *v);
        match near { Some((t, a)) => a / t, None => self.estimate(score) }
    }
    /// Estimated conditional acceptance for a draft with this score: the nearest populated bin at
    /// or below it (else the lowest populated); optimistic 1.0 while still learning.
    fn estimate(&self, score: f32) -> f64 {
        if self.total < Self::BURN_IN { return 1.0; }
        let idx = (score as f64 / Self::BIN_W).floor() as i64;
        let mut below: Option<(f64, f64)> = None;
        let mut lowest: Option<(f64, f64)> = None;
        for (&k, &v) in self.bins.iter() {
            if v.0 < Self::MIN_COUNT { continue; }
            if lowest.is_none() { lowest = Some(v); }
            if k <= idx { below = Some(v); } else { break; }
        }
        match below.or(lowest) { Some((t, a)) => a / t, None => 1.0 }
    }
}

// ---- WP23 (PLAN/SURPASS_PLAN_2026-09-26.md): draft depth 6/7 — DDS depth bins + cost guard ----
// Draft-only: the verify decides every emitted token (greedy bitwise at any width <= 16 by batch
// invariance); these only choose how many drafts a round proposes.

/// The pre-WP23 depth ceiling: at a served depth <= this, both WP23 DDS parts default OFF, so
/// --exl3-mtp-k=5 reproduces the K5 build's DDS decisions (the brief's confound check).
const WP23_LEGACY_MAX_K: usize = 5;
/// Depth buckets of the WP23 bins: pass {0} | {1,2} | {3,4} | {5..} — levels 6/7 share a bucket so
/// it fills on code at ~1.5 labels/round (a per-level bucket at 1-logit bins stays under MIN_COUNT).
const WP23_DEPTH_BUCKETS: usize = 4;
fn wp23_depth_bucket(i: usize) -> usize {
    match i { 0 => 0, 1 | 2 => 1, 3 | 4 => 2, _ => 3 }
}

/// WP23 part switch: `name`=0|off / 1|on forces it (diagnostic A/B, never a serving default);
/// unset = on iff the served depth k exceeds the pre-WP23 ceiling.
fn wp23_on(name: crate::opts::OptId, k: usize) -> bool {
    match crate::opts::var(name).as_deref() {
        Ok("0") | Ok("off") => false,
        Ok("1") | Ok("on") => true,
        _ => k > WP23_LEGACY_MAX_K,
    }
}

/// WP23 cost guard switch (--exl3-dds-guard=0|1; default on iff k > 5). k = the scheduler's
/// served depth (one per process), so the env is read once.
pub(crate) fn wp23_guard_on(k: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| wp23_on(crate::opt!("exl3-dds-guard"), k))
}

/// WP23 cost guard: the lane's least-squares slope of round ms on drafts per round (exponentially
/// weighted, this request only), shrunk toward the 6.0 ms/draft prior (PLAN WP23: a row past m=6
/// costs ~6.0-6.3 ms incl. its draft pass). With little spread in drafts/round (code that always
/// drafts to the cap) the prior dominates; DDS-varied rounds (prose, python) identify the slope.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CostFit {
    /// the prior slope ĉ is shrunk toward (ms per draft): 6.0 at TP=1; TP-I #7: dds_prior(tp)
    prior: f64,
    w: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
    n: usize,
}

impl Default for CostFit {
    fn default() -> Self {
        CostFit { prior: Self::PRIOR_SLOPE, w: 0.0, sx: 0.0, sy: 0.0, sxx: 0.0, sxy: 0.0, n: 0 }
    }
}

/// TP-I #7 (T2 lever): the WP23 cost guard's prior ĉ for a TP=2 lane. The 6.0 ms/draft seed was fitted
/// to TP=1 costs; a TP=2 draft costs a verify row at the TP=2 width slope (1.92 ms/row, TP-I re-ledger:
/// r0 union m=4 22.91 -> m=8 30.60 ms/verify) plus one sharded draft pass (0.76 ms) = ~2.7 ms. With
/// code that always drafts to the cap the fit never identifies the slope, so the prior IS the guard.
///   unset / on : 2.7 at TP (world 2) — only lanes of a TP scheduler; TP=1 keeps 6.0 by construction
///   0 / off    : 6.0 (the TP=1 constant, the pre-TP-I behaviour)
///   <ms>       : that prior (diagnostic)
/// --tp-dds-prior rides the exl3_env snapshot; the value joins the TP boot agree hash. The guard's
/// timing inputs stay the HEAD's (tp_round), so both ranks' targets are identical.
pub(crate) const TP2_DDS_PRIOR: f64 = 2.7;
/// CLI-1 (--print-config): the TP=2 prior this process resolves.
pub fn dds_prior_tp() -> f64 { dds_prior(true) }
pub(crate) fn dds_prior(tp: bool) -> f64 {
    if !tp { return CostFit::PRIOR_SLOPE; }
    static P: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *P.get_or_init(|| match crate::opts::var(crate::opt!("tp-dds-prior")).map(|v| v.trim().to_ascii_lowercase()) {
        Ok(v) if v == "0" || v == "off" || v == "false" => CostFit::PRIOR_SLOPE,
        Ok(v) if v == "1" || v == "on" || v.is_empty() => TP2_DDS_PRIOR,
        Ok(v) => v.parse::<f64>().ok().filter(|x| *x > 0.0).unwrap_or(TP2_DDS_PRIOR),
        Err(_) => TP2_DDS_PRIOR,
    })
}

impl CostFit {
    pub(crate) fn with_prior(prior: f64) -> Self { CostFit { prior, ..Default::default() } }
    const DECAY: f64 = 0.97;       // ~33-round memory: the context (and a row's cost) drifts slowly
    const PRIOR_SLOPE: f64 = 6.0;  // ms per draft (the brief's seed)
    const PRIOR_W: f64 = 4.0;      // prior weight in (drafts - mean)^2 units
    pub(crate) fn add(&mut self, x: f64, y: f64) {
        let d = Self::DECAY;
        self.w = d * self.w + 1.0;
        self.sx = d * self.sx + x;
        self.sy = d * self.sy + y;
        self.sxx = d * self.sxx + x * x;
        self.sxy = d * self.sxy + x * y;
        self.n += 1;
    }
    /// ĉ (ms per draft), >= 0.
    pub(crate) fn slope(&self) -> f64 {
        let (cxx, cxy) = if self.w > 0.0 {
            ((self.sxx - self.sx * self.sx / self.w).max(0.0), self.sxy - self.sx * self.sy / self.w)
        } else {
            (0.0, 0.0)
        };
        ((cxy + Self::PRIOR_W * self.prior) / (cxx + Self::PRIOR_W)).max(0.0)
    }
}

/// WP23: the DDS target for this round. `base` = --draft-confidence (0.4 served). Once the lane
/// has 16 rounds and its cost EMAs, a draft must be worth its cost: accept probability >= ĉ ms x
/// the lane's tokens/ms — max(base, clamp(ĉ·ema_tok/ema_ms, 0.15, 0.8)). The guard only RAISES
/// the target (long context: rows cost more; fast code rounds: every ms is worth more tokens).
pub(crate) fn wp23_target(base: f64, guard: bool, rounds: usize, c_hat: f64, ema_tok: f64, ema_ms: f64) -> f64 {
    if !guard || rounds < 16 || ema_ms <= 0.0 || ema_tok <= 0.0 {
        return base;
    }
    base.max((c_hat * ema_tok / ema_ms).clamp(0.15, 0.8))
}

/// WP23 per-lane state: the cost fit, the last round's target, rounds by verified draft width.
#[derive(Clone, Copy, Debug, Default)]
struct Wp23Lane {
    fit: CostFit,
    target: f64,
    widths: [usize; crate::exl3_forward::MTP_MAX_K + 1],
}

impl Lane {
    fn log_stats(&self, slot: usize, reason: &str) {
        if self.st_rounds == 0 && self.st_plain == 0 { return; }
        let r = self.st_rounds.max(1) as f64;
        self.log_wp23(slot);
        eprintln!("[mtp-stats] slot={slot} finish={reason} gen={} rounds={} drafted={} accepted={} ({:.1}%) \
                   tok/round={:.2} ms/round={:.1} plain_steps={} temp={} ctrl={}",
                  self.generated, self.st_rounds, self.st_drafted, self.st_accepted,
                  100.0 * self.st_accepted as f64 / self.st_drafted.max(1) as f64,
                  (self.st_accepted as f64 + self.st_rounds as f64) / r, self.st_ms / r,
                  self.st_plain, self.temperature, self.st_ctrl);
    }

    /// WP02: the lane's client is gone (receiver dropped: disconnect, or a stop-string hit on the
    /// HTTP side) — its slot is freed now; always leaves a [mtp-stats] finish=cancelled line.
    fn log_cancel(&self, slot: usize) {
        if self.st_rounds == 0 && self.st_plain == 0 {
            eprintln!("[mtp-stats] slot={slot} finish=cancelled gen={} rounds=0 plain_steps=0 temp={}",
                      self.generated, self.temperature);
        } else {
            self.log_stats(slot, "cancelled");
        }
    }

    /// WP23: one line per finished request that ran MTP rounds — rounds by verified draft width
    /// (widths[w] = rounds that verified w drafts; w=6/7 are the new depths), the cost fit ĉ and
    /// the last round's DDS target. Printed just before [mtp-stats].
    fn log_wp23(&self, slot: usize) {
        if self.st_rounds == 0 { return; }
        let g = &self.wp23;
        eprintln!("[dds] slot={slot} widths={:?} c_hat={:.2} ms/draft (fit rounds {}) target={:.3}",
                  g.widths, g.fit.slope(), g.fit.n, g.target);
    }

    /// Sampler params for `n` consecutive rows starting at the lane's counter (None = greedy).
    fn samp_rows(&self, n: usize) -> Vec<Option<crate::exl3_forward::SampParams>> {
        (0..n).map(|i| crate::exl3_forward::SampParams::sampled(
            self.temperature, self.top_p, self.top_k, self.min_p, self.seed,
            self.ctr.wrapping_add(i as u32))).collect()
    }

    /// WP08: feed one emitted token (one that did not already end the request) to the loop
    /// detector — the rival's job.py:1010-1013 order. True = end the response ("loop_detected").
    fn loop_feed(&mut self, t: u32) -> bool {
        self.loop_det.as_mut().map_or(false, |d| d.feed(t))
    }

    /// WP08: one line per detection, with what a replay needs (a loop can be an engine bug:
    /// replay greedy + plain before blaming the model).
    fn log_loop(&self, slot: usize, tok: &QwenTokenizer) {
        let Some(d) = self.loop_det.as_ref() else { return };
        let tail_ids = &self.history[self.history.len().saturating_sub(96)..];
        let text = tok.decode(tail_ids, false).unwrap_or_default();
        let n = text.chars().count();
        let tail: String = text.chars().skip(n.saturating_sub(200)).collect();
        eprintln!("[loop] detected slot={slot} period={} (detector {}) gen={} pos={} seed={} temp={} \
                   top_p={} top_k={} min_p={} pen={:?} mtp={} prompt_hash={:016x} tail={:?}",
                  d.fundamental_period().map_or("-".into(), |p| p.to_string()),
                  d.period().map_or("-".into(), |p| p.to_string()),
                  self.generated, self.pos, self.seed, self.temperature, self.top_p, self.top_k,
                  self.min_p, self.pen, self.mtp.is_some(), self.prompt_hash, tail);
    }
}

/// C1 v2: prefill scratch rows — 2C - 1 with tail checkpoints on (merge_grid chunks), else C.
fn psc_rows(c: usize) -> usize {
    if crate::exl3_forward::tail_ckpt_on() && !crate::exl3_forward::wp16_off() { 2 * c - 1 } else { c }
}

/// FNV-1a over the prompt ids — a replay key for the [loop] log (not a security hash).
fn prompt_hash(p: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in p {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl Exl3Scheduler {
    /// Prefill one lane's prompt into its slot, chunked (S-A3-f-b): C tokens per
    /// forward sweep through the wide-M path. The final sweep's last token is
    /// NOT processed here — see the S-A3-h seam contract below.
    /// NOTE: the sequential path primed the M1 draft head per prompt token; the
    /// chunked prefill skips that work entirely (pure TTFT win) and the head is
    /// primed once after the seam step (PLAN/S_A3_F_B_CHUNKED_PREFILL.md).
    /// S-A3-h item 1 seam contract (bench byte-parity): chunked prefill covers
    /// prompt[..plen-1] ONLY — the LAST prompt token runs through `forward_step`,
    /// the graphed/fused decode step, in `admit` (the exact bench `--prefill-chunk`
    /// contract, exl3_forward::run). The f-g prefill row and the f-f decode row for
    /// the same token differ in low bits (N.6 class): using the prefill row at the
    /// prompt/decode seam broke bench-vs-serve byte parity AND primed the M1 tap
    /// from a drifted hidden (bug5's self-consistent-but-wrong chains). One
    /// composition per position on both sides, by construction.
    /// WP02: returns Ok(false) when the client went away between chunks (no Finish is sent; the
    /// caller leaves cache[slot] = None, so the partial slot state is never reused). `tx` None =
    /// not cancellable (the WP16 XCHECK's fresh reference prefill).
    /// WP16: `aligned` = this run's chunk grid starts at 0 or at an aligned checkpoint (the caller
    /// did CkptStore::begin_aligned): its chunks extend the slot's checkpointed prefix and the
    /// chunk ends `wp16_due` picks become checkpoints. Never set for a prompt-end prefix-hit resume.
    /// C1 (`tail` Some only with prefix.tail_ckpt on): the grid is `run_grid(from, n, C, realign,
    /// tail)` — realigned to the C grid and split at the message boundary `tail`, which becomes a
    /// checkpoint of this (tracked) run. Knob off: realign = false, tail = None = today's grid.
    fn prefill_range(&mut self, slot: usize, prompt: &[u32], from: usize, to: usize,
                     tx: Option<&tokio::sync::mpsc::UnboundedSender<TokEvent>>, aligned: bool,
                     tail: Option<usize>, realign: bool) -> Result<bool> {
        let tp = std::time::Instant::now();
        let c = self.chunk.max(1);
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_rows(c))?);
        }
        let psc = self.psc.as_mut().unwrap();
        let mut pos = from;
        let n = to.min(prompt.len().saturating_sub(1)); // the seam token is a decode step
        if pos >= n { return Ok(true); }
        let mut taken: Vec<usize> = Vec::new();
        // C1 v2 (knob on == realign): merge away the chunk the split / realign would add (psc holds
        // 2C - 1 rows then, see run())
        let grid = if realign {
            crate::exl3_forward::merge_grid(from, crate::exl3_forward::run_grid(from, n, c, true, tail), tail, c)
        } else {
            crate::exl3_forward::run_grid(from, n, c, false, None)
        };
        let nchunks = grid.len();
        for end in grid {
            if pos > from {
                let mut gone = tx.map_or(false, |t| t.is_closed());
                // TP-D: the HEAD decides (its client); every rank stops at this same chunk boundary
                if let (Some(t), true) = (self.tp.as_mut(), tx.is_some()) {
                    gone = t.head_flag(gone)?;
                }
                if gone {
                    println!("[exl3-serve] prefill slot={slot} cancelled at pos {pos} of {n} (client gone)");
                    return Ok(false);
                }
            }
            let toks: Vec<i32> = prompt[pos..end].iter().map(|&t| t as i32).collect();
            self.model.prefill_chunk(psc, &toks, pos, slot, false, None)?;
            if aligned {
                if let Some(st) = self.wp16.as_mut() {
                    st.extend(slot, &prompt[pos..end]);
                    if crate::exl3_forward::wp16_due(end, n, c) || tail == Some(end) {
                        let model = &self.model;
                        if let Some(mut b) = st.acquire(|| model.ckpt_alloc()) {
                            match model.ckpt_save(slot, psc, &mut b) {
                                Ok(()) => { st.insert(slot, end, b); taken.push(end); }
                                Err(e) => { st.release(b); return Err(e); }
                            }
                        }
                    }
                }
            }
            pos = end;
        }
        // S-A3-f-d Item 2: serve-path prefill telemetry (TTFT's main term).
        let ms = tp.elapsed().as_secs_f64() * 1e3;
        let done = n - from;
        println!("[exl3-serve] prefill slot={} n={} (pos {}..{}) in {:.1} ms ({:.0} tok/s, C={})",
                 slot, done, from, n, ms, if ms > 0.0 { done as f64 * 1e3 / ms } else { 0.0 }, c);
        if let Some(t) = tail {
            println!("[exl3-serve] C1 tail slot={slot}: grid split + checkpoint at {t} (message boundary - {}; {nchunks} chunk(s){})",
                     crate::exl3_forward::TAIL_BACKOFF,
                     if realign && from % c != 0 { ", realigned to the C grid" } else { "" });
        } else if realign && from % c != 0 {
            println!("[exl3-serve] C1 slot={slot}: run from {from} realigned to the C grid ({nchunks} chunk(s))");
        }
        if !taken.is_empty() {
            if let Some(st) = self.wp16.as_ref() {
                let per = self.model.ckpt_layout().bytes() as f64;
                println!("[exl3-serve] WP16 slot={slot}: checkpoints taken at {taken:?}; slot holds {:?}; \
                          live {} x {:.0} MB = {:.2} GiB (allocated {} of cap {})",
                         st.positions(slot), st.live(), per / 1e6, st.live() as f64 * per / (1u64 << 30) as f64,
                         st.n_alloc, st.cap);
            }
        }
        Ok(true)
    }

    /// Prefill prompt[..plen-1] into `slot`, resuming from a cached prefix when possible, and
    /// snapshot the recurrent state at the END of the chunked prefill (prompt[..plen-1]; the last
    /// token is the seam decode step). No extra split: a fresh prompt prefills bit-identically to
    /// --prefix-cache off (a message-boundary split cost a ~175 ms small chunk and moved chunk
    /// boundaries). The next chat turn re-renders this prompt verbatim up to plen-1 in both
    /// thinking modes (verified via /v1/tokenize: thinking on differs only at the final token).
    /// WP16: when the prompt-end snapshot does not match (a new conversation on a shared system
    /// prompt, a new question on the same document), resume from the deepest intermediate
    /// checkpoint <= min(LCP(prompt, the checkpointed prefix), plen-1) instead of prefilling from 0
    /// — the same chunk grid as a fresh prefill, so bit-identical to it. The prompt-end hit path
    /// is unchanged (whenever it matches it is the deeper resume and it still wins).
    /// C1 (prefix.tail_ckpt on): `ckpt_at` (the message boundary: this prompt without its
    /// generation prompt, server.rs) becomes a tail checkpoint of this run — the resume point of
    /// the next chat turn, which diverges right there — and every run is tracked (rows extend the
    /// checkpointed prefix) with its grid realigned to C. See wp16.rs T_PREFIX_TAIL_CKPT.
    fn prefill_cached(&mut self, slot: usize, prompt: &[u32], ckpt_at: Option<usize>,
                      tx: &tokio::sync::mpsc::UnboundedSender<TokEvent>) -> Result<bool> {
        let plen = prompt.len();
        let end = plen.saturating_sub(1);
        self.last_resume = (0, 0);
        if !self.prefix_on || !self.model.prefix_snapshots() {
            // PFX1 (b): the chunked prefill below head-fills rows 0..plen-2 (no head-KV memset)
            self.model.reset_slot_for_prefill(slot)?;
            self.cache[slot] = None;
            return self.prefill_range(slot, prompt, 0, end, Some(tx), false, None, false);
        }
        let tail_on = self.wp16.is_some() && crate::exl3_forward::tail_ckpt_on();
        let reuse = match &self.cache[slot] {
            Some(t) if t.len() <= end && prompt[..t.len()] == t[..] => t.len(),
            _ => 0,
        };
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_rows(self.chunk.max(1)))?);
        }
        let c = self.chunk.max(1);
        // WP16: only on a prompt-end miss (a hit is always the deeper, unchanged resume).
        let (q, lcp) = match (&self.wp16, reuse) {
            (Some(st), 0) => st.resume_point(slot, prompt, end, c),
            _ => (0, 0),
        };
        let mut tracked = reuse == 0 && self.wp16.is_some();
        if reuse > 0 {
            self.model.restore_slot(slot, self.psc.as_mut().unwrap())?;
            println!("[exl3-serve] prefix-cache hit slot={slot}: reuse {reuse} of {plen} prompt tokens");
            // an unaligned grid from `reuse`: no checkpoint is taken in it, none above it survives.
            // C1: with tail checkpoints the run is tracked instead (realigned grid, tail split) when
            // the slot's checkpointed prefix is exactly prompt[..reuse].
            if let Some(st) = self.wp16.as_mut() {
                if tail_on && st.prefix_ok(slot, prompt, reuse, c) {
                    st.begin_aligned(slot, reuse, c);
                    tracked = true;
                } else {
                    st.begin_unaligned(slot, reuse);
                }
            }
        } else if q > 0 {
            let st = self.wp16.as_mut().unwrap();
            let dropped = st.begin_aligned(slot, q, c);
            st.touch(slot, q);
            let ck = st.get(slot, q).context("WP16: resume checkpoint vanished")?;
            // replaces reset_slot: recurrent state + PLE ring/hist + head-fill carry at q; every
            // positional row < q (trunk/head KV, QSA planes) is the checkpointed prefix's own
            self.model.ckpt_restore(slot, self.psc.as_mut().unwrap(), ck)?;
            self.model.ckpt_poison_head_kv_from(slot, q)?; // PFX1 (b) poison gate only (no-op otherwise)
            println!("[exl3-serve] WP16 checkpoint resume slot={slot}: resume at {q} of {plen} prompt tokens \
                      (LCP {lcp}, C={c}; {dropped} checkpoint(s) above dropped) — prefill {q}..{end}");
        } else {
            self.model.reset_slot_for_prefill(slot)?; // PFX1 (b): prefill_range from 0 head-fills
            if let Some(st) = self.wp16.as_mut() { st.begin_aligned(slot, 0, c); }
        }
        self.cache[slot] = None;
        self.last_resume = (reuse, q);
        let from = reuse.max(q);
        let tail = if tail_on && tracked { crate::exl3_forward::tail_point(ckpt_at, from, end, c, crate::exl3_forward::tail_ckpt_extra_chunk()) } else { None };
        match self.prefill_range(slot, prompt, from, end, Some(tx), tracked, tail, tail_on && tracked) {
            Ok(true) => {}
            // WP02: cancelled mid-prefill — no snapshot, cache stays None. WP16: the checkpoints
            // taken so far stay (they describe prompt[..pos], rows this run did write).
            Ok(false) => return Ok(false),
            Err(e) => {
                if let Some(st) = self.wp16.as_mut() { st.clear_slot(slot); } // slot state unknown
                return Err(e);
            }
        }
        if q > 0 && self.wp16_xcheck {
            self.wp16_xcheck(slot, prompt, q)?;
        }
        self.model.snapshot_slot(slot, self.psc.as_ref().unwrap())?;
        self.cache[slot] = Some(prompt[..end].to_vec());
        Ok(true)
    }

    /// --wp16-xcheck=1 (diagnostic): after a checkpoint resume of `slot` (state R live), prefill
    /// the same prompt FRESH from 0 into a scratch slot — a free lane's slot, or (--max-batch 1)
    /// the resumed slot itself — and bit-compare GDN S / conv / PLE ring / PLE hist / head-fill
    /// carry and the seam step's first-token logits. The resumed state is then restored, so the
    /// served output stays the resumed path's (scratch mode: exactly; same-slot mode: the KV rows
    /// are the fresh run's rewrite of the same positions).
    fn wp16_xcheck(&mut self, slot: usize, prompt: &[u32], q: usize) -> Result<()> {
        let t0 = std::time::Instant::now();
        let plen = prompt.len();
        let end = plen - 1;
        let last = prompt[end] as i32;
        let model = self.model.clone();
        let v = model.cfg.vocab_size;
        let x = (0..self.lanes.len()).find(|&i| i != slot && self.lanes[i].is_none()).unwrap_or(slot);
        let mut r = model.ckpt_alloc()?;
        model.ckpt_save(slot, self.psc.as_ref().unwrap(), &mut r)?;
        // raw logits: greedy, unpenalized (admit re-stages both before the real seam step)
        model.set_sampling(&mut self.sc, &[None])?;
        model.set_penalties(&mut self.sc, &[None])?;
        let id_r = model.forward_step(&mut self.sc, &[last], &[end], &[slot], None)?[0];
        let lg_r: Vec<u16> = model.logits_host(&self.sc)?[..v].to_vec();
        if x != slot {
            self.cache[x] = None; // the scratch slot's rows get overwritten
            if let Some(st) = self.wp16.as_mut() { st.clear_slot(x); }
        }
        println!("[wp16-xcheck] slot={slot} resumed at {q}: fresh reference prefill 0..{end} into {} slot {x}",
                 if x != slot { "scratch" } else { "the same (no free lane)" });
        model.reset_slot_for_prefill(x)?;
        self.prefill_range(x, prompt, 0, end, None, false, None, false)?;
        let mut f = model.ckpt_alloc()?;
        model.ckpt_save(x, self.psc.as_ref().unwrap(), &mut f)?;
        let d = model.ckpt_diff(&r, &f)?;
        let id_f = model.forward_step(&mut self.sc, &[last], &[end], &[x], None)?[0];
        let lg_f: Vec<u16> = model.logits_host(&self.sc)?[..v].to_vec();
        let nl = lg_r.iter().zip(&lg_f).filter(|(a, b)| a != b).count();
        // serve the resumed state: recurrent state + PLE ring/hist + carry back to R (the seam step
        // above advanced them; the real seam step rewrites KV row plen-1 identically)
        model.ckpt_restore(slot, self.psc.as_mut().unwrap(), &r)?;
        if x != slot { model.reset_slot(x)?; }
        let exact = d.exact() && nl == 0 && id_r == id_f;
        println!("[wp16-xcheck] slot={slot} resume {q} vs fresh (slot {x}) at plen {plen}: {}; first-token logits \
                  differ {nl}/{v} (argmax {id_r} vs {id_f}) -> {} ({:.0} ms)",
                 d.line(), if exact { "EXACT" } else { "MISMATCH" }, t0.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    /// S-A3-h: head prime after the seam decode step — snapshot the tap the seam
    /// step left in sc.resid (the f-f decode composition, byte-exact: the same tap
    /// class as the bench-mtp prime and every round's re-prime; the f-d
    /// tap_snapshot_from_prefill workaround fed round 1's verify restore a
    /// ≤1e-3-drifted prefill tap, whose chain was self-consistent but wrong), then
    /// feed the head the last prompt token with that tap (round 1's draft-extend
    /// reuses the snapshot via tap_row 0).
    /// WP03: the slot's own taps window (row 0) + a KV-only head row plen-1 paired with the stream
    /// after plen-2 (psc's head-fill carry) — FwdModel::mtp_seam_prime (--a-seam = old arm).
    fn mtp_prime(&mut self, slot: usize, last_prompt_tok: u32, plen: usize) -> Result<()> {
        let head = self.model.mtp.as_ref()
            .context("mtp_prime called without a draft head")?;
        self.model.mtp_seam_prime(&mut self.sc, self.psc.as_ref(), head, last_prompt_tok as i32, plen, slot)
    }

    /// TP-C: one lockstep point (a seam step, a plain batched step, an MTP round or a control
    /// step). (1) agree_ext(step, accept, verify width m, FNV of `words`) — the load-bearing
    /// tripwire (AGENTS §2.10): a divergence aborts the link on both ranks; (2) a u32 exchange
    /// that ships the head's (ms, plain ms, captured) — returned on EVERY rank, so every timing-
    /// derived decision downstream is the head's; (3) with `ident` on, the digests of the logits
    /// rows 0..m, the residual and the taps window are compared across ranks (G-T1-b graphed).
    fn tp_round(&mut self, what: &str, a: usize, m: usize, words: &[u32], ms: f64, plain_ms: f64,
                captured: bool) -> Result<(f64, f64, bool)> {
        let Some(t) = self.tp.as_mut() else { return Ok((ms, plain_ms, captured)) };
        t.step += 1;
        let (step, rank, ident) = (t.step, t.rank, t.ident);
        let h = fnv32(0, words.iter().copied());
        // Fence A = the agree token itself: both ranks finished this step's forward, so every hot-
        // path epoch payload of it has been read on both sides (the exchange below reuses ring slot
        // memory: its send buffer is send slot 0, its receive lands in recv slot gen & 7).
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let r = tp_agree_eq(agree_field(step, AGREE_A), a.min(255) as u8, m.min(15) as u8, h);
        t.wp.rec(0, t0, s0);
        t.wp.rounds += 1;
        if let Err(why) = r {
            crate::net::abort_link();
            anyhow::bail!("TP agree() FAILED at {what} (lockstep step {step}, accept {a}, width {m}, hash {h:08x}): {why} — \
                           ranks diverged or the link aborted");
        }
        let digests = if ident { self.model.tp_ident_digests(&self.sc, m)? } else { Vec::new() };
        let tag = 0xC0DE_0000u32 | (step as u32 & 0xFFFF);
        let (mb, pb) = (ms.to_bits(), plain_ms.to_bits());
        let mut mine = vec![tag, mb as u32, (mb >> 32) as u32, pb as u32, (pb >> 32) as u32, captured as u32,
                            digests.len() as u32];
        mine.extend(digests.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]));
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, mine.len() + 4)?;
        let tw = self.tp.as_mut().unwrap();
        tw.wp.rec(1, t0, s0);
        // Fence B: nobody re-enters the hot path (whose next epochs reuse the ring slots) until both
        // ranks have read their exchange payload. (Without it a rank that finished its exchange first
        // raced into the next forward and overwrote the slower rank's unread exchange slot — the
        // 2026-09-29 smoke: "recv payload tail never landed", abort code 2.)
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let r = tp_agree_eq(agree_field(step, AGREE_B), 0, 0, tag);
        tw.wp.rec(2, t0, s0);
        if let Err(why) = r {
            crate::net::abort_link();
            anyhow::bail!("TP post-exchange fence FAILED at {what} (lockstep step {step}): {why}");
        }
        anyhow::ensure!(peer[0] == tag, "TP step sync at {what}: peer at {:#x}, this rank at {tag:#x}", peer[0]);
        anyhow::ensure!(peer[6] as usize == digests.len(), "TP step sync at {what}: peer sent {} digests, this rank {}",
                        peer[6], digests.len());
        let t = self.tp.as_mut().unwrap();
        t.agrees += 1;
        if ident {
            let bad = (0..digests.len()).filter(|&i| mine[7 + 2 * i] != peer[7 + 2 * i] || mine[8 + 2 * i] != peer[8 + 2 * i]).count();
            t.ident_bufs += digests.len() as u64;
            t.ident_bad += bad as u64;
            if bad > 0 && t.ident_bad as usize == bad {
                eprintln!("[tp-ident] FIRST MISMATCH at {what} (lockstep step {step}, width {m}): {bad} of {} digests differ \
                           (logits rows 0..{m}, resid, taps)", digests.len());
            }
        }
        if rank == 0 {
            Ok((ms, plain_ms, captured))
        } else {
            Ok((f64::from_bits(peer[1] as u64 | ((peer[2] as u64) << 32)),
                f64::from_bits(peer[3] as u64 | ((peer[4] as u64) << 32)), peer[5] != 0))
        }
    }

    /// One decode round over all active lanes (batched; lanes are independent slots).
    /// S-A3-f-d: lanes with a loaded M1 head run chain-verify rounds (speculation ON by
    /// default per AGENTS 1b — --exl3-no-mtp is the owner's diagnostics hatch, never a
    /// default). API-parity G2: sampled lanes speculate too (device-sampled verify rows).
    /// Lanes whose request turned MTP off (G1) batch through the plain decode path.
    fn step(&mut self) -> Result<()> {
        // WP02 cancel sweep (the NVFP4 decode_step sweep, batch.rs): a lane whose client is gone
        // (disconnect, or the HTTP side dropped the receiver on a stop-string hit) frees its slot
        // NOW instead of decoding to EOS/max_new while every later request waits behind it.
        // TP-D: under TP the HEAD sweeps (run_tp_head) and ships each cancel as a Step event — a
        // rank-local sweep would free a lane on one rank only.
        for s in 0..self.lanes.len() {
            if self.tp.is_none() && self.lanes[s].as_ref().map_or(false, |l| l.tx.is_closed()) {
                if let Some(lane) = self.lanes[s].take() {
                    lane.log_cancel(s);
                }
            }
        }
        let slots: Vec<usize> = self.lanes.iter().enumerate()
            .filter(|(_, l)| l.is_some())
            .map(|(i, _)| i)
            .collect();
        if slots.is_empty() {
            return Ok(());
        }
        self.steps += 1;
        if self.inject_panic == Some(self.steps) {
            panic!("--inject-panic-steps: injected scheduler panic at decode step {} (liveness gate)", self.steps);
        }
        crate::tel::note_step();
        let mtp_on = self.model.mtp.is_some()
            && !crate::exl3_forward::mtp_disabled_by_opt();
        let mut plain: Vec<usize> = Vec::new();
        for &s in &slots {
            let lane = self.lanes[s].as_ref().unwrap();
            if mtp_on && lane.mtp.is_some() {
                self.step_mtp(s)?;
            } else {
                plain.push(s);
            }
        }
        if !plain.is_empty() {
            self.step_plain(&plain)?;
        }
        Ok(())
    }

    /// WP24: this lane's real-q draft sampler for its next MTP round — Some only for a SAMPLED
    /// lane (greedy stays on the bitwise match path) under --spec-sampling ratio (mode 2), or
    /// with --wp24-dump set (mode 1: today's drafts, top-32 lists recorded for rung 0).
    fn rq_stage(&self, s: usize) -> Option<crate::exl3_forward::RqStage> {
        let lane = self.lanes[s].as_ref()?;
        crate::exl3_forward::RqStage::for_lane(self.spec_ratio, self.draft_temp, lane.temperature,
                                               lane.top_p, lane.top_k, lane.min_p, lane.seed, lane.dctr)
    }

    /// One chain-verify round for one lane, with the §6 profitability control:
    /// every 128th round runs PLAIN as the control and the MTP path auto-disables
    /// (loudly) if its per-token cost loses to plain.
    fn step_mtp(&mut self, s: usize) -> Result<()> {
        let k = self.mtp_k;
        let af = crate::exl3_forward::a_flags();
        let head = self.model.mtp.as_ref().unwrap();
        let (b, p, tap_row, rows, pen, remaining, lane_cost) = {
            let lane = self.lanes[s].as_ref().unwrap();
            (lane.last_tok as i32, lane.pos, lane.mtp.unwrap(), lane.samp_rows(k + 1), lane.pen,
             lane.max_new.saturating_sub(lane.generated),
             if lane.ema_tok > 0.0 { lane.ema_ms / lane.ema_tok } else { 0.0 })
        };
        let xcheck = crate::exl3_forward::a_xcheck_on();
        if xcheck {
            self.model.tap_xcheck_verify(&self.sc, s, tap_row)?; // WP03: was this lane's tap clobbered?
        }
        // WP02 end-of-window clamp. Verify rows p..p+k (draft rows p..p+k-1, head re-prime
        // p+1..p+a) write the max_pos-row KV / indexer-key planes and read cos_tab: the last rounds
        // of a window-filling answer wrote up to p+k past them — p+7 at WP23's depth 7 — (K head 0
        // into head 1's rows 0-1, the last V head into the next slot or past the allocation).
        // (WP23 carried an identical copy of this clamp; p2 keeps this one.) p+k <= max_pos-2 also keeps every
        // written row's pooled-key block inside floor(max_pos/ratio). The remaining-budget half
        // drafts no token the request cannot emit (--a-remclamp=0 = off). k_eff == 0 -> a plain
        // step (with the budget clamp that is the request's last token).
        let mut k_eff = k.min(self.model.max_pos().saturating_sub(p + 2));
        if af.rem_clamp {
            k_eff = k_eff.min(remaining.saturating_sub(1));
        }
        self.ctrl_round += 1;
        // WP03 cadence: 1/1024 while this request's MTP cost is clearly below plain (< 0.8x), else
        // 1/128. ema_plain stays context-local: it is refreshed only by control rounds at the
        // current context, never by a boot-time plain cost (under-priced at long context).
        let period = if af.ctrl == 0 && self.ema_plain > 0.0 && lane_cost > 0.0
            && lane_cost < 0.8 * self.ema_plain { 1024 } else { 128 };
        let control = self.ctrl_round % period == 0;
        let plain_step = control || k_eff == 0;
        // G2: stage this lane's sampler rows (verify rows 0..k, or row 0 of a plain step).
        // WP24: a sampled lane under --spec-sampling ratio stages real-q rows + the draft sampler
        // (plain steps stay plain); None = exactly set_sampling (the match rule, today).
        let rq = if plain_step { None } else { self.rq_stage(s) };
        let rq_ratio = rq.map_or(false, |r| r.mode == 2);
        self.model.set_spec_sampling(&mut self.sc, if plain_step { &rows[..1] } else { &rows }, rq)?;
        // WP15: the same penalty params on every row of the round (and the draft mirror's row 0).
        self.model.set_penalties(&mut self.sc, &vec![pen; if plain_step { 1 } else { k + 1 }])?;
        let t0 = std::time::Instant::now();
        let g0 = self.model.graph_count(); // WP23: a round that captures a graph is not cost-fit
        let (out, plain_ms, confs) = if plain_step {
            // plain round: consume b at p like a plain decode step. WP03: no draft pass runs, so
            // first write the head's row p KV-only (tap that predicted b, b), and after the step
            // snapshot its tap into row 0 of the lane's OWN window (sc.resid is shared: another
            // lane's round overwrites it before this lane's next round). --a-ctrl=old = neither.
            let ctrl_new = af.ctrl != 2;
            let tc = if ctrl_new {
                self.model.mtp_plain_prime(&mut self.sc, head, s, b, p, tap_row)?;
                self.model.dev().synchronize()?; // the control's plain-cost timing stays the step alone
                std::time::Instant::now()
            } else {
                t0
            };
            let ids = self.model.forward_step(&mut self.sc, &[b], &[p], &[s], None)?;
            let ms = tc.elapsed().as_secs_f64() * 1e3;
            if ctrl_new {
                self.model.tap_snapshot(&mut self.sc, s)?;
            }
            // synthesize an accept-0 round; e_0 becomes the new bonus
            (crate::exl3_forward::MtpRoundOut {
                a: 0,
                emitted: vec![ids[0]],
                pos_next: p + 1,
                last_tok: ids[0],
                // new: the snapshot row; old: the plain step leaves the true tap in sc.resid
                tap_row: if ctrl_new { 0 } else { usize::MAX },
                drafts: Vec::new(),
                diag_ms: 0.0,
            }, ms, Vec::new())
        } else if self.draft_conf > 0.0 {
            // API-parity DDS: draft until the running product of calibrated per-position
            // acceptance estimates falls below the target (the rival's -dds -dc), then verify
            // only the proposed window; every verified position labels the calibrator.
            // WP23: the target carries the lane's cost guard, the estimate its draft's depth bucket
            // (both identity at k <= 5 unless forced: the K5 build's decisions exactly).
            let target = {
                let lane = self.lanes[s].as_mut().unwrap();
                let t = wp23_target(self.draft_conf, wp23_guard_on(k), lane.mtp_rounds,
                                    lane.wp23.fit.slope(), lane.ema_tok, lane.ema_ms);
                lane.wp23.target = t;
                t
            };
            let cal = &self.cal;
            let mut reach = 1.0f64;
            let mut keep = |i: usize, c: f32| { reach *= cal.estimate_at(i, c); reach >= target };
            // TP-D: agree the verify width + drafts before the verify launches (None at TP=1)
            let tp_on = self.tp.is_some();
            let tp = &mut self.tp;
            let mut pv = |w: usize, d: &[i32], n: usize| -> Result<()> {
                match tp.as_mut() { Some(t) => t.pre_verify(w, d, n), None => Ok(()) }
            };
            let pre: Option<&mut dyn FnMut(usize, &[i32], usize) -> Result<()>> =
                if tp_on && tp_preverify_on() { Some(&mut pv) } else { None };
            let (o, confs) = self.model.mtp_round_adaptive(&mut self.sc, head, s, k_eff, b, p, tap_row, &mut keep, pre)?;
            for (i, &c) in confs.iter().enumerate() {
                if i < o.a { self.cal.add_label(i, c, true); } else { self.cal.add_label(i, c, false); break; }
            }
            self.cal.decay_step();
            (o, 0.0, confs)
        } else {
            (self.model.mtp_round(&mut self.sc, head, s, k_eff, b, p, tap_row)?, 0.0, Vec::new())
        };
        // WP24: the dump / XCHECK host diagnostics are not round cost (they would trip the §6
        // per-request auto-off on a diagnostics run); 0 when off.
        let ms = (t0.elapsed().as_secs_f64() * 1e3 - out.diag_ms).max(0.0);
        // TP-C: lockstep — agree on (step, accept, verify width, emitted + drafts + confidences) and
        // take the HEAD's ms / plain ms / capture flag (identity at TP=1: no TP state).
        let captured_local = self.model.graph_count() != g0;
        let (ms, plain_ms, captured) = if self.tp.is_some() {
            let mut w: Vec<u32> = out.emitted.iter().map(|&x| x as u32).collect();
            w.extend(out.drafts.iter().map(|&x| x as u32));
            w.extend(confs.iter().map(|c| c.to_bits()));
            let m = if plain_step { 1 } else { out.drafts.len() + 1 };
            self.tp_round(if plain_step { "plain-round" } else { "mtp-round" }, out.a, m, &w, ms, plain_ms, captured_local)?
        } else {
            (ms, plain_ms, captured_local)
        };
        if xcheck {
            self.model.tap_xcheck_record(&self.sc, s, out.tap_row)?;
        }
        // G2: advance the lane's sampler counter past every row this round consumed.
        {
            let lane = self.lanes[s].as_mut().unwrap();
            lane.ctr = lane.ctr.wrapping_add(if plain_step { 1 } else { (k + 1) as u32 });
            // WP24: past every draft counter this round could use (passes 0..k_eff-1 incl. a
            // HOST speculative pass the DDS then stopped before; k_eff <= k)
            if rq_ratio { lane.dctr = lane.dctr.wrapping_add(k as u32); }
            // WP24 gate bookkeeping (None in the server): round anatomy per arm + the request's
            // first-round draft d_0 (the negative control: d_0 ~ q', a law != p)
            if let Some(g) = self.gate.as_mut() {
                if !plain_step {
                    g.rounds += 1;
                    g.drafted += out.drafts.len() as u64;
                    g.accepted += out.a as u64;
                    g.ratio_rounds += rq_ratio as u64;
                    g.ms += ms;
                } else if control {
                    g.ctrl += 1;
                } else {
                    g.plain += 1;
                }
                if lane.gate_first {
                    if let Some(&d0) = out.drafts.first() { g.d0.push(d0); }
                }
            }
            lane.gate_first = false;
        }
        // API-parity G1: §6 profitability, scoped to THIS request. The plain reference is
        // an engine property (control rounds, persists); the MTP side is this request's
        // content. A request whose drafts do not pay finishes plain; nothing global dies
        // (the old process-lifetime `mtp_dead` turned one prose request into ~30 tok/s
        // for every later request).
        let toks = (out.a + 1) as f64;
        if control {
            self.ema_plain = if self.ema_plain <= 0.0 { plain_ms } else {
                0.875 * self.ema_plain + 0.125 * plain_ms
            };
            if let Some(lane) = self.lanes[s].as_mut() { lane.st_ctrl += 1; }
        } else if plain_step {
            // WP02: the window/budget-clamped last step (k_eff == 0) — a plain step, not a round
            if let Some(lane) = self.lanes[s].as_mut() { lane.st_plain += 1; }
        } else {
            let ema_plain = self.ema_plain;
            let lane = self.lanes[s].as_mut().unwrap();
            lane.mtp_rounds += 1;
            lane.st_rounds += 1;
            lane.st_drafted += out.drafts.len();
            lane.st_accepted += out.a;
            lane.st_ms += ms;
            // the first rounds of a slot can include graph capture; keep them out of the cost
            if lane.mtp_rounds > 4 {
                if lane.ema_tok <= 0.0 {
                    lane.ema_ms = ms;
                    lane.ema_tok = toks;
                } else {
                    lane.ema_ms = 0.9375 * lane.ema_ms + 0.0625 * ms;
                    lane.ema_tok = 0.9375 * lane.ema_tok + 0.0625 * toks;
                }
            }
            let cost = if lane.ema_tok > 0.0 { lane.ema_ms / lane.ema_tok } else { 0.0 };
            if lane.mtp_rounds >= 64 && ema_plain > 0.0 && cost > ema_plain * 1.02 {
                eprintln!("[exl3-serve] MTP off for this request (slot {s}): {:.2} ms/tok ({:.1} ms/round, \
                           {:.2} tok/round) vs plain {:.2} after {} rounds (AGENTS §6, per-request)",
                           cost, lane.ema_ms, lane.ema_tok, ema_plain, lane.mtp_rounds);
                // the lane is at a clean seam (round committed, last_tok unconsumed at
                // lane.pos — the same state a control round leaves), so plain can take over.
                lane.mtp = None;
            }
        }
        // WP23: rounds by verified draft width, and the cost fit (ms on drafts) — MTP rounds of a
        // DDS lane past the EMAs' 4-round warm-up, minus any round that captured a graph (a lazy
        // first-use capture) or ran > 3x the lane's mean round (a host stall): outliers, not cost.
        if !plain_step {
            let dds = self.draft_conf > 0.0;
            if let Some(lane) = self.lanes[s].as_mut() {
                let w = out.drafts.len();
                if let Some(c) = lane.wp23.widths.get_mut(w) { *c += 1; }
                if dds && lane.mtp_rounds > 4 && !captured && (lane.ema_ms <= 0.0 || ms <= 3.0 * lane.ema_ms) {
                    lane.wp23.fit.add(w as f64, ms);
                }
            }
        }

        // ---- emit the round's tokens (stop at EOS / max_new mid-list)
        for &t32 in out.emitted.iter() {
            let t = t32 as u32;
            let (finish, reason, generated) = {
                let lane = self.lanes[s].as_mut().unwrap();
                lane.last_tok = t;
                lane.pos += 1;
                lane.generated += 1;
                lane.history.push(t);
                let fin = lane.generated >= lane.max_new
                    || (self.eos.contains(&t) && !lane.ignore_eos && lane.generated >= lane.min_new);
                let rsn = if lane.generated >= lane.max_new { "length" }
                    else if fin { "stop" } else { "" };
                // WP08: a loop ends the round mid-list like a stop token (rest dropped).
                if !fin && lane.loop_feed(t) {
                    (true, "loop_detected".to_string(), lane.generated)
                } else {
                    (fin, rsn.to_string(), lane.generated)
                }
            };
            if let Some(lane) = self.lanes[s].as_mut() {
                if lane.mtp.is_some() {
                    lane.mtp = Some(out.tap_row);
                }
            }
            // WP02: a failed send = the client is gone — cancel now, free the slot. TP-D: under TP
            // the head's next sweep ships the cancel instead (rank-local drops would desync).
            let sent = self.lanes[s].as_ref().map_or(false, |lane| lane.tx.send(crate::batch::TokEvent::Tok(t)).is_ok());
            if !sent && self.tp.is_none() {
                if let Some(lane) = self.lanes[s].take() {
                    lane.log_cancel(s);
                }
                return Ok(());
            }
            if finish {
                if let Some(lane) = self.lanes[s].take() {
                    if reason == "loop_detected" { lane.log_loop(s, &self.tok); }
                    lane.log_stats(s, &reason);
                    self.model.dump_expert_hist(); // TP-I #6 diagnostic (no-op unless --tp-ep-hist)
                    let _ = lane.tx.send(crate::batch::TokEvent::Finish { reason });
                    // after the Finish: the DHEADP line may sync the device once (penalized requests)
                    self.model.dhead_request_report(); // DHEADP fallback line (penalized requests) + DHEAD XCHECK lines (when on)
                }
                return Ok(()); // lane finished; remaining emitted tokens dropped
            }
        }
        // consumed all emitted. S-A3-h: the emit loop's per-token pos increments
        // leave lane.pos = out.pos_next = the position of last_tok — mtp_round's
        // next-p contract (identical to bench_mtp_exl3's `p = out.pos_next`, whose
        // chain is the LOSSLESS reference). The old `lane.pos = out.pos_next - 1`
        // clobber shifted every round 2+ one position back (RoPE/state seam) and
        // produced the residual bug-5 wrong-but-consistent chains.
        Ok(())
    }

    /// Plain batched decode for the given slots (the pre-f-d step() body).
    fn step_plain(&mut self, slots: &[usize]) -> Result<()> {
        let toks: Vec<i32> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().last_tok as i32)
            .collect();
        let poss: Vec<usize> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().pos)
            .collect();
        // API-parity G2: sampled lanes sample ON DEVICE (xq_sample_rows after the argmax;
        // the same sampler the MTP verify rows use) — no 248K-float host readback per token.
        let rows: Vec<Option<crate::exl3_forward::SampParams>> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().samp_rows(1)[0])
            .collect();
        self.model.set_sampling(&mut self.sc, &rows)?;
        let pens: Vec<Option<crate::exl3_forward::PenParams>> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().pen)
            .collect();
        self.model.set_penalties(&mut self.sc, &pens)?;
        let ids = self.model.forward_step(&mut self.sc, &toks, &poss, slots, None)?;
        if self.tp.is_some() {
            // TP-C: lockstep on the batched plain step's emitted ids (no wall-clock decision here)
            let w: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
            self.tp_round("plain-step", 0, slots.len(), &w, 0.0, 0.0, false)?;
        }
        if let Some(g) = self.gate.as_mut() { g.plain += slots.len() as u64; }

        for (j, &s) in slots.iter().enumerate() {
            let t = {
                let lane = self.lanes[s].as_mut().unwrap();
                let next = ids[j] as u32;
                lane.ctr = lane.ctr.wrapping_add(1);
                lane.st_plain += 1;
                lane.last_tok = next;
                lane.pos += 1;
                lane.generated += 1;
                lane.history.push(next);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
                next
            };
            let (finish, reason) = {
                let lane = self.lanes[s].as_mut().unwrap();
                if lane.generated >= lane.max_new {
                    (true, "length")
                } else if self.eos.contains(&t) && !lane.ignore_eos && lane.generated >= lane.min_new {
                    (true, "stop")
                } else if lane.loop_feed(t) {
                    (true, "loop_detected") // WP08
                } else {
                    (false, "")
                }
            };
            let tp_on = self.tp.is_some();
            let lane = self.lanes[s].as_mut().unwrap();
            // TP-D: under TP a failed send is not a cancel here (the head's next sweep ships it)
            if lane.tx.send(TokEvent::Tok(t)).is_err() && !tp_on {
                // WP02: the client is gone — cancel now, free the slot
                lane.log_cancel(s);
                self.lanes[s] = None;
                continue;
            }
            if finish {
                if reason == "loop_detected" { lane.log_loop(s, &self.tok); }
                lane.log_stats(s, reason);
                self.model.dump_expert_hist(); // TP-I #6 diagnostic (no-op unless --tp-ep-hist)
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.to_string() });
                self.lanes[s] = None;
                self.model.dhead_request_report(); // DHEADP fallback line (penalized requests) + DHEAD XCHECK lines (when on)
            }
        }
        Ok(())
    }

    fn run(mut self, mut rx: UnboundedReceiver<BatchRequest>) {
        // HOST / RO-7: the mask this thread inherited from `pub fn run`'s pin (--cpu-affinity)
        if let Ok(m) = crate::cpu_affinity::current_thread_affinity() {
            println!("[exl3-serve] scheduler thread cpu mask: {}", crate::cpu_affinity::format_cpu_list(&m));
        }
        // Requests beyond the lane count WAIT in arrival order (the standard server contract);
        // they used to be refused as "server busy", which reached a non-streaming client as an
        // empty HTTP 200. A client that hangs up while queued is dropped before admission.
        let mut waiting: std::collections::VecDeque<BatchRequest> = std::collections::VecDeque::new();
        loop {
            if self.fatal.is_some() {
                return self.die(waiting, rx);
            }
            if waiting.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                match rx.blocking_recv() {
                    Some(req) => waiting.push_back(req),
                    None => break,
                }
            }
            while let Ok(req) = rx.try_recv() {
                waiting.push_back(req);
            }
            while self.fatal.is_none() && self.lanes.iter().any(|l| l.is_none()) {
                let Some(req) = waiting.pop_front() else { break };
                if req.tx.is_closed() {
                    continue;
                }
                self.admit(req);
            }
            if !waiting.is_empty() {
                waiting.retain(|r| !r.tx.is_closed());
            }
            if self.fatal.is_some() || self.lanes.iter().all(|l| l.is_none()) {
                continue;
            }
            if let Err(e) = self.step() {
                eprintln!("[exl3-serve] decode step failed: {e:#}");
                // Fail every live lane loudly — never emit garbage as if it were text.
                for s in 0..self.lanes.len() {
                    if let Some(lane) = self.lanes[s].take() {
                        let _ = lane.tx.send(TokEvent::Finish { reason: format!("error: {e}") });
                    }
                }
                if sticky_cuda_error(&e) {
                    self.fatal = Some(format!("{e:#}"));
                }
            }
        }
    }

    /// WP02 liveness: the CUDA context is poisoned (sticky error) — nothing on this engine can run
    /// again. Mark it DEAD (/health 503, new requests 503 at the HTTP layer), fail every live and
    /// queued request with an error Finish, then either exit 70 (--exit-on-fatal: the supervisor
    /// restarts the server) or keep answering every late request with the same error.
    fn die(mut self, mut waiting: std::collections::VecDeque<BatchRequest>,
           mut rx: UnboundedReceiver<BatchRequest>) {
        let why = self.fatal.clone().unwrap_or_default();
        crate::server::set_engine_dead(&why);
        let reason = format!("error: engine stopped (fatal CUDA error: {why})");
        eprintln!("[exl3-serve] FATAL: {reason}");
        for s in 0..self.lanes.len() {
            if let Some(lane) = self.lanes[s].take() {
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.clone() });
            }
        }
        while let Ok(req) = rx.try_recv() {
            waiting.push_back(req);
        }
        for req in waiting.drain(..) {
            let _ = req.tx.send(TokEvent::Finish { reason: reason.clone() });
        }
        if self.exit_on_fatal {
            eprintln!("[exl3-serve] FATAL: exit 70 (--exit-on-fatal; a supervisor restarts the server)");
            std::thread::sleep(std::time::Duration::from_millis(300)); // let the error events flush
            std::process::exit(70);
        }
        while let Some(req) = rx.blocking_recv() {
            let _ = req.tx.send(TokEvent::Finish { reason: reason.clone() });
        }
    }

    fn admit(&mut self, req: BatchRequest) {
        let t_admit = std::time::Instant::now();
        // prefix cache: prefer the free lane whose snapshot is the longest prefix of this prompt,
        // then a lane with no snapshot, then any free lane.
        let pick = {
            let free: Vec<usize> = (0..self.lanes.len()).filter(|&i| self.lanes[i].is_none()).collect();
            let hit = free.iter().copied().filter_map(|i| match &self.cache[i] {
                Some(t) if t.len() < req.prompt.len() && req.prompt[..t.len()] == t[..] => Some((t.len(), i)),
                _ => None,
            }).max();
            hit.map(|(_, i)| i)
                // WP16: else the free lane whose checkpoints resume this prompt deepest
                .or_else(|| {
                    let st = self.wp16.as_ref()?;
                    let end = req.prompt.len().saturating_sub(1);
                    free.iter().copied()
                        .map(|i| (st.resume_point(i, &req.prompt, end, self.chunk.max(1)).0, i))
                        .filter(|&(q, _)| q > 0).max().map(|(_, i)| i)
                })
                .or_else(|| free.iter().copied().find(|&i| self.cache[i].is_none()))
                .or_else(|| free.first().copied())
        };
        let free = match pick {
            Some(i) => i,
            None => {
                let _ = req.tx.send(TokEvent::Finish {
                    reason: "error: server busy (all lanes occupied)".into(),
                });
                return;
            }
        };
        if req.schema.is_some() {
            let _ = req.tx.send(TokEvent::Finish {
                reason: "error: json-schema constraint not supported on the exl3 serve path yet".into(),
            });
            return;
        }
        if req.image_embeds.is_some() || !req.image_spans.is_empty() {
            let _ = req.tx.send(TokEvent::Finish {
                reason: "error: vision input not supported on the exl3 serve path".into(),
            });
            return;
        }
        if req.prompt.is_empty() {
            let _ = req.tx.send(TokEvent::Finish { reason: "error: empty prompt".into() });
            return;
        }
        let max_new = req.max_new.min(self.model.max_pos().saturating_sub(req.prompt.len() + 1));
        match self.prefill_cached(free, &req.prompt, req.ckpt_at, &req.tx) {
            Ok(true) => {}
            // WP02: the client left mid-prefill — the slot is free again, nobody to tell
            Ok(false) => return,
            Err(e) => {
                eprintln!("[exl3-serve] prefill failed (slot {free}): {e:#}");
                let _ = req.tx.send(TokEvent::Finish { reason: format!("error: prefill failed: {e}") });
                if self.tp.is_some() || sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                return;
            }
        }
        // S-A3-h seam step: the LAST prompt token runs through forward_step — the
        // graphed/fused decode step the bench path uses for it — so its KV/GDN state
        // and the tap left in sc.resid are byte-class identical to bench ground
        // truth. Its argmax is the first generated token (the S-A3-e contract).
        let plen = req.prompt.len();
        // API-parity G2: the first generated token is SAMPLED for sampled requests (it was
        // the argmax). Seed: the request's, else fresh entropy; counter 0 = this token.
        let seed = req.seed.unwrap_or_else(|| {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64).unwrap_or(0);
            t ^ ((free as u64) << 56) ^ 0x5DEECE66D
        });
        let first = crate::exl3_forward::SampParams::sampled(
            req.temperature, req.top_p, req.top_k, req.min_p, seed, 0);
        if let Err(e) = self.model.set_sampling(&mut self.sc, &[first]) {
            let _ = req.tx.send(TokEvent::Finish { reason: format!("error: sampler staging failed: {e}") });
            if self.tp.is_some() { self.fatal = Some(format!("{e:#}")); } // TP-D: never a one-rank skip
            return;
        }
        // WP15: penalties from the first generated token on (the seam step's row), over the
        // prompt tail + everything generated — the rival's past_ids = the full sequence.
        let pen = crate::exl3_forward::PenParams::new(req.rep_penalty, req.presence_penalty,
                                                      req.frequency_penalty, self.pen_range, self.pen_range);
        if pen.is_some() {
            eprintln!("[exl3-serve] penalties slot={free}: {pen:?} (window {} + {} tokens)", self.pen_range, self.pen_range);
        }
        let staged = match pen {
            Some(_) => self.model.pen_seed(&mut self.sc, free, &req.prompt[..plen - 1])
                .and_then(|_| self.model.set_penalties(&mut self.sc, &[pen])),
            None => self.model.set_penalties(&mut self.sc, &[None]),
        };
        if let Err(e) = staged {
            let _ = req.tx.send(TokEvent::Finish { reason: format!("error: penalty staging failed: {e}") });
            if self.tp.is_some() { self.fatal = Some(format!("{e:#}")); } // TP-D: never a one-rank skip
            return;
        }
        let last = match self.model.forward_step(
            &mut self.sc,
            &[*req.prompt.last().unwrap() as i32],
            &[plen - 1],
            &[free],
            None,
        ) {
            Ok(ids) => ids[0] as u32,
            Err(e) => {
                eprintln!("[exl3-serve] seam decode step failed (slot {free}): {e:#}");
                let _ = req.tx.send(TokEvent::Finish {
                    reason: format!("error: seam step failed: {e}"),
                });
                if self.tp.is_some() || sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                return;
            }
        };
        if self.tp.is_some() {
            // TP-C: lockstep on the seam step (the request's first token)
            // TP-D: + the slot, the prefix-cache / WP16-C1 resume points and the checkpoint store's
            // shape (live, allocated, this slot's checkpoint positions) — proven equal on every rank
            let (reuse, q) = self.last_resume;
            let mut w = vec![last, plen as u32, free as u32, reuse as u32, q as u32];
            if let Some(st) = self.wp16.as_ref() {
                w.extend([st.live() as u32, st.n_alloc as u32,
                          fnv32(0, st.positions(free).iter().map(|&x| x as u32))]);
            }
            if let Err(e) = self.tp_round("seam", 0, 1, &w, 0.0, 0.0, false) {
                eprintln!("[exl3-serve] TP seam lockstep failed (slot {free}): {e:#}");
                let _ = req.tx.send(TokEvent::Finish { reason: format!("error: TP lockstep failed: {e}") });
                self.fatal = Some(format!("{e:#}"));
                return;
            }
        }
        if req.tx.send(TokEvent::Tok(last)).is_err() && self.tp.is_none() {
            // WP02: the client is gone — no lane (the prefix snapshot stays valid). TP-D: under TP
            // the lane is created on every rank and the head's next sweep cancels it.
            eprintln!("[mtp-stats] slot={free} finish=cancelled gen=1 rounds=0 plain_steps=0 temp={}", req.temperature);
            return;
        }
        // WP04-v2: engine-side TTFT (receipt = the handler's hand-off, AFTER tokenization and
        // template rendering), so a client TTFT can be split into HTTP/tokenizer vs engine time.
        println!("[exl3-serve] first token slot={free} {:.1} ms after receipt (queued {:.1} ms)",
                 req.received_at.elapsed().as_secs_f64() * 1e3,
                 t_admit.duration_since(req.received_at).as_secs_f64() * 1e3);
        if max_new <= 1 {
            let _ = req.tx.send(TokEvent::Finish { reason: "length".into() });
            return;
        }
        if self.eos.contains(&last) && !req.ignore_eos && 1 >= req.min_new {
            let _ = req.tx.send(TokEvent::Finish { reason: "stop".into() });
            return;
        }
        // S-A3-f-d: greedy lanes on an M1 pack speculate by default (AGENTS 1b).
        // Prime the head once: snapshot the tap that predicted `last` (now the
        // seam decode step's sc.resid — exact, same composition as the bench-mtp
        // prime and every round's re-prime), then feed the head the last prompt
        // token with that tap (round 1's draft-extend reuses tap_row 0).
        let mtp = if self.model.mtp.is_some() && !crate::exl3_forward::mtp_disabled_by_opt() {
            match self.mtp_prime(free, *req.prompt.last().unwrap(), plen) {
                Ok(()) => {
                    if crate::exl3_forward::a_xcheck_on() {
                        if let Err(e) = self.model.tap_xcheck_record(&self.sc, free, 0) {
                            eprintln!("[a-xcheck] record failed: {e:#}");
                        }
                    }
                    Some(0usize)
                }
                Err(e) => {
                    eprintln!("[exl3-serve] MTP prime failed (lane runs plain): {e:#}");
                    if self.tp.is_some() || sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                    None
                }
            }
        } else {
            None
        };
        // WP08: the rival feeds the detector every emitted token, the first one included.
        let mut loop_det = self.loop_cfg
            .map(|(w, r)| crate::loop_detect::LoopDetector::for_stop_on_loop(w, r));
        if let Some(d) = loop_det.as_mut() { d.feed(last); }
        self.lanes[free] = Some(Lane {
            pos: req.prompt.len(),
            last_tok: last,
            generated: 1,
            max_new,
            min_new: req.min_new,
            ignore_eos: req.ignore_eos,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            pen,
            loop_det,
            prompt_hash: prompt_hash(&req.prompt),
            history: req.prompt.clone(),
            seed,
            ctr: 1,
            dctr: 0,
            gate_first: true,
            ema_ms: 0.0,
            ema_tok: 0.0,
            mtp_rounds: 0,
            // TP-I #7: a TP lane's cost prior is the TP=2 one (dds_prior; TP=1 = 6.0 unchanged)
            wp23: Wp23Lane { fit: CostFit::with_prior(dds_prior(self.tp.is_some())), ..Default::default() },
            st_rounds: 0,
            st_drafted: 0,
            st_accepted: 0,
            st_ms: 0.0,
            st_plain: 0,
            st_ctrl: 0,
            mtp,
            tx: req.tx,
        });
    }
}

// ---- TP-D: serving at TP=2 (design §5: run_tp_head / run_tp_mirror around the SPMD scheduler) ----

/// A request the EXL3 path refuses before admission (checked on the head BEFORE an Admit ships,
/// so a refused request never reaches the node).
fn exl3_reject(req: &BatchRequest) -> Option<String> {
    if req.schema.is_some() {
        return Some("error: json-schema constraint not supported on the exl3 serve path yet".into());
    }
    if req.image_embeds.is_some() || !req.image_spans.is_empty() {
        return Some("error: vision input not supported on the exl3 serve path".into());
    }
    if req.prompt.is_empty() {
        return Some("error: empty prompt".into());
    }
    None
}

/// The seed a request without one gets (admit's TP=1 derivation, minus the slot term).
fn entropy_seed() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64).unwrap_or(0);
    t ^ 0x5DEECE66D
}

impl Exl3Scheduler {
    /// TP-D: when the engine goes idle, one line of lockstep counters (the session's agree receipts).
    fn tp_log_idle(&mut self, step_no: u64) {
        if self.lanes.iter().all(|l| l.is_none()) {
            if let Some(t) = self.tp.as_mut() {
                println!("[tp-serve] rank {} idle after control step {step_no}: lockstep agrees {} ok, pre-verify agrees {} ok{}",
                         t.rank, t.agrees, t.pre_agrees,
                         if t.ident { format!(", graphed digests {} compared, {} differ", t.ident_bufs, t.ident_bad) } else { String::new() });
                println!("{}", t.wp.line(t.rank));
                t.wp = WaitProf::default();
            }
        }
    }

    fn tp_ship(&mut self, m: &crate::tp_serve::ServingMsg) -> Result<()> {
        let live = self.lanes.iter().any(|l| l.is_some());
        let t = self.tp.as_mut().context("TP ship without TP state")?;
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        for s in t.ctl.iter_mut() {
            crate::tp_serve::send_serving(s, m)?;
        }
        if live { t.wp.rec(6, t0, s0); }
        Ok(())
    }

    /// TP-D: a TP rank cannot serve on after a failed lockstep, a failed step or a fatal admit:
    /// fail every live, pending and queued request loudly, mark the engine dead (/health 503 while
    /// the process winds down), and return the error — the caller exits 70.
    fn tp_fail(&mut self, why: String, waiting: &mut std::collections::VecDeque<BatchRequest>,
               rx: Option<&mut UnboundedReceiver<BatchRequest>>) -> anyhow::Error {
        crate::server::set_engine_dead(&why);
        let reason = format!("error: TP engine stopped ({why})");
        for s in 0..self.lanes.len() {
            if let Some(lane) = self.lanes[s].take() {
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.clone() });
            }
        }
        if let Some(rx) = rx {
            while let Ok(req) = rx.try_recv() { waiting.push_back(req); }
        }
        for req in waiting.drain(..) {
            let _ = req.tx.send(TokEvent::Finish { reason: reason.clone() });
        }
        anyhow::anyhow!("{why}")
    }

    /// TP-D: the TP=2 HEAD's scheduler loop. Every scheduler-visible decision is taken HERE and
    /// shipped before it is applied: per iteration the head sweeps cancels (a closed client
    /// channel), admits waiting requests (validated here; the seed resolved here so sampled ranks
    /// draw the same stream), ships ONE `Step { events }` to every node, then applies exactly those
    /// events in order and runs the SPMD step the node runs. Everything else the step decides is a
    /// pure function of replicated state (slot pick, prefix-cache / WP16 / C1 resume points, EOS /
    /// max_new / loop stops) or is head-authoritative inside the step (round timing — TP-C's
    /// exchange; a prefill chunk's cancel — HeadFlag); the agree tripwires prove it every
    /// seam / round (and before every DDS verify).
    fn run_tp_head(mut self, mut rx: UnboundedReceiver<BatchRequest>, streams: Vec<std::net::TcpStream>) -> Result<()> {
        use crate::tp_serve::{ServingMsg, StepEvents, TpEvent, WireRequest};
        if let Ok(m) = crate::cpu_affinity::current_thread_affinity() {
            println!("[exl3-serve] TP head scheduler thread cpu mask: {}", crate::cpu_affinity::format_cpu_list(&m));
        }
        self.tp.as_mut().context("run_tp_head without TP state")?.ctl = streams;
        let mut waiting: std::collections::VecDeque<BatchRequest> = std::collections::VecDeque::new();
        let mut step_no: u64 = 0;
        let fast_ctl = crate::opts::var(crate::opt!("exl3-tp-fastctl")).map_or(true, |v| v != "0");
        println!("[exl3-serve] TP control: {}", if fast_ctl { "step-go over RDMA while live, TCP Step only with events (TP-E)" }
                 else { "TCP Step every step (--exl3-tp-fastctl=0)" });
        loop {
            if waiting.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                match rx.blocking_recv() {
                    Some(req) => waiting.push_back(req),
                    None => {
                        let _ = self.tp_ship(&ServingMsg::Shutdown);
                        return Ok(());
                    }
                }
            }
            while let Ok(req) = rx.try_recv() {
                waiting.push_back(req);
            }
            let mut events: Vec<TpEvent> = Vec::new();
            let mut admits: std::collections::VecDeque<BatchRequest> = std::collections::VecDeque::new();
            let mut free = 0usize;
            for s in 0..self.lanes.len() {
                match &self.lanes[s] {
                    None => free += 1,
                    Some(l) if l.tx.is_closed() => {
                        events.push(TpEvent::Cancel { lane: s });
                        free += 1;
                    }
                    Some(_) => {}
                }
            }
            while free > 0 {
                let Some(mut req) = waiting.pop_front() else { break };
                if req.tx.is_closed() {
                    continue;
                }
                if let Some(why) = exl3_reject(&req) {
                    let _ = req.tx.send(TokEvent::Finish { reason: why });
                    continue;
                }
                if req.seed.is_none() {
                    req.seed = Some(entropy_seed()); // head-resolved: every rank samples one stream
                }
                events.push(TpEvent::Admit(WireRequest::from(&req)));
                admits.push_back(req);
                free -= 1;
            }
            if !waiting.is_empty() {
                waiting.retain(|r| !r.tx.is_closed());
            }
            if events.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                continue;
            }
            step_no += 1;
            // TP-E: a live engine's step goes over RDMA; the TCP Step only when it carries events
            let go = fast_ctl && self.lanes.iter().any(|l| l.is_some());
            if go {
                let n = events.len();
                if let Err(e) = self.tp.as_mut().context("TP state")?.step_go(step_no, n) {
                    return Err(self.tp_fail(format!("step-go failed at step {step_no}: {e:#}"), &mut waiting, Some(&mut rx)));
                }
            }
            if !go || !events.is_empty() {
                let msg = ServingMsg::Step(StepEvents { step: step_no, events: events.clone() });
                if let Err(e) = self.tp_ship(&msg) {
                    return Err(self.tp_fail(format!("control plane send failed at step {step_no}: {e:#}"), &mut waiting, Some(&mut rx)));
                }
            }
            for ev in events {
                match ev {
                    TpEvent::Cancel { lane } => {
                        if let Some(l) = self.lanes[lane].take() {
                            l.log_cancel(lane);
                        }
                    }
                    TpEvent::Admit(_) => {
                        let req = admits.pop_front().context("admit queue out of step")?;
                        self.admit(req);
                    }
                }
                if let Some(f) = self.fatal.clone() {
                    return Err(self.tp_fail(format!("admit failed at step {step_no}: {f}"), &mut waiting, Some(&mut rx)));
                }
            }
            if self.lanes.iter().any(|l| l.is_some()) {
                if let Err(e) = self.step() {
                    eprintln!("[exl3-serve] TP decode step failed: {e:#}");
                    return Err(self.tp_fail(format!("decode step {step_no} failed: {e:#}"), &mut waiting, Some(&mut rx)));
                }
            }
            self.tp_log_idle(step_no);
        }
    }

    /// TP-D: the node's mirror loop — apply each head `Step` (cancels, admits with the head's
    /// resolved sampling params + seed) in order, then run the identical SPMD step. The node's
    /// token channels have no receiver (sends are no-ops; no rank-local cancel exists under TP).
    /// Ok = the head shut the session down (or its stream closed); Err = a failed step (the node
    /// process exits and its supervisor re-arms).
    fn run_tp_mirror(mut self) -> Result<()> {
        use crate::tp_serve::{ServingMsg, TpEvent};
        let mut step_no: u64 = 0;
        let mut admitted: u64 = 0;
        let fast_ctl = crate::opts::var(crate::opt!("exl3-tp-fastctl")).map_or(true, |v| v != "0");
        println!("[exl3-serve] TP control (node): {}", if fast_ctl { "step-go over RDMA while live (TP-E)" } else { "TCP Step every step" });
        loop {
            let live = self.lanes.iter().any(|l| l.is_some());
            if fast_ctl && live {
                let (hs, n) = self.tp.as_mut().context("mirror without TP state")?.step_go(step_no + 1, 0)?;
                anyhow::ensure!(hs == step_no + 1, "TP mirror: head step-go {hs} after local step {step_no}");
                if n == 0 {
                    step_no = hs;
                    self.step().with_context(|| format!("TP node: decode step {step_no} failed"))?;
                    self.tp_log_idle(step_no);
                    continue;
                }
                // events ride the TCP Step below (the head sent it after the go)
            }
            let msg = {
                let t = self.tp.as_mut().context("mirror without TP state")?;
                let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
                let r = crate::tp_serve::recv_serving(&mut t.ctl[0]);
                if live { t.wp.rec(5, t0, s0); }  // mid-request only: the head's inter-step host time + TCP
                match r {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[exl3-serve] TP node: head control stream closed ({e:#}) — session over after {step_no} steps, {admitted} requests");
                        return Ok(());
                    }
                }
            };
            match msg {
                ServingMsg::Shutdown => {
                    eprintln!("[exl3-serve] TP node: Shutdown from the head after {step_no} steps, {admitted} requests");
                    return Ok(());
                }
                ServingMsg::Step(se) => {
                    anyhow::ensure!(se.step == step_no + 1, "TP mirror: head step {} after local step {step_no}", se.step);
                    step_no = se.step;
                    for ev in se.events {
                        match ev {
                            TpEvent::Cancel { lane } => {
                                if let Some(l) = self.lanes.get_mut(lane).and_then(|l| l.take()) {
                                    l.log_cancel(lane);
                                }
                            }
                            TpEvent::Admit(w) => {
                                let (tx, _) = tokio::sync::mpsc::unbounded_channel::<TokEvent>();
                                let min_p = w.min_p;
                                let mut req = w.into_request(tx);
                                req.min_p = min_p;
                                admitted += 1;
                                self.admit(req);
                            }
                        }
                        if let Some(f) = self.fatal.clone() {
                            anyhow::bail!("TP node: admit failed at step {step_no}: {f}");
                        }
                    }
                    if self.lanes.iter().any(|l| l.is_some()) {
                        self.step().with_context(|| format!("TP node: decode step {step_no} failed"))?;
                    }
                    self.tp_log_idle(step_no);
                }
                other => anyhow::bail!("TP mirror: unexpected control message {other:?}"),
            }
        }
    }
}

/// TP-D: bring up the EXL3 TP attachment from a TpContext (the sanity + branch handshakes).
fn tp_attach(mut ctx: crate::tp::TpContext) -> Result<crate::exl3_forward::xtp::TpAttach> {
    ctx.sanity()?;
    ctx.branch_check(&crate::tp::TpBranch::Exl3Xtp)?;
    let (rank, world, link) = ctx.into_parts();
    Ok(crate::exl3_forward::xtp::TpAttach { rank, world, link })
}

/// TP-D: --exl3-tp-preverify=0 (diagnostic, prices the belt): skip the pre-verify width/drafts
/// exchange + agree (the post-round agree still proves the width, one round late — TP-C's shape).
/// Rides TpConfig's env snapshot, so both ranks decide alike.
fn tp_preverify_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-tp-preverify")).map_or(true, |v| v != "0"))
}

/// TP-D: --exl3-tp-ident=1 (diagnostic; rides TpConfig's env snapshot, so both ranks agree):
/// the served scheduler digests the logits rows, residual and taps at every lockstep point and
/// compares them across ranks (TP-C's G-T1-b graphed identity, through the API). Costs a dtoh per
/// step — a gate knob, never a serving default.
fn tp_ident_knob(s: &mut Exl3Scheduler) {
    if crate::opts::var(crate::opt!("exl3-tp-ident")).map_or(false, |v| v == "1") {
        if let Some(t) = s.tp.as_mut() {
            t.ident = true;
            println!("[exl3-serve] TP ident digests ON (--exl3-tp-ident=1, diagnostic): logits rows + resid + taps compared across ranks every lockstep point");
        }
    }
}

/// TP-D: every resolved scheduler knob, hashed and AGREED across ranks before the first request
/// (both ranks resolved them from the same argv + shipped env; this proves it, including the
/// memory-derived WP16 checkpoint cap and the tune table actually frozen).
fn tp_boot_agree(s: &mut Exl3Scheduler) -> Result<()> {
    let tune = crate::exl3_tune::status().map(|t| format!("{}|{:?}|{}|{}", t.state, t.table_sha8, t.decisions_s, t.decisions_d))
        .unwrap_or_default();
    let mut w: Vec<u32> = vec![s.width as u32, s.chunk as u32, s.mtp_k as u32, s.prefix_on as u32,
                               s.pen_range as u32, s.spec_ratio as u32, s.model.max_pos() as u32,
                               s.model.mtp.is_some() as u32, crate::exl3_forward::mtp_disabled_by_opt() as u32];
    let dc = s.draft_conf.to_bits();
    w.extend([dc as u32, (dc >> 32) as u32, s.draft_temp.map_or(0, |t| t.to_bits())]);
    w.extend(s.loop_cfg.map_or([0, 0], |(a, b)| [a as u32, b as u32]));
    w.extend(s.wp16.as_ref().map_or([0, 0], |st| [1, st.cap as u32]));
    w.push(crate::exl3_forward::tail_ckpt_on() as u32);
    w.push(crate::exl3_forward::kv_fmt_from_opts() as u32);
    // TP-I #7: the WP23 cost-guard prior both ranks resolved (it only matters if the targets differ)
    w.push((dds_prior(true) as f32).to_bits());
    // TP-I2 item 1: the prefill-overlap granularity both ranks resolved (--tp-prefill-overlap, TpConfig v19)
    w.push(crate::exl3_forward::xtp::pf_overlap_code());
    // TP-H2: the sampled-row vocab-parallel tail both ranks resolved (--tp-vp-sampled, TpConfig v21)
    w.push(crate::exl3_forward::xtp::vp_sampled_code());
    // TP-SP1: sequence-parallel prefill (--tp-seq-parallel, TpConfig v22) — an ON code only, so an OFF
    // boot's hash is the pre-SP posture's
    if crate::exl3_forward::xtp::seq_parallel_on() {
        w.push(0x5350_0000 | crate::exl3_forward::xtp::seq_parallel_code());
    }
    // CLI-1: every SPMD registry option either rank has set (both installed the head's registry
    // (TpConfig v23); this proves it). Pushed only when one is set, so a boot with no option set
    // keeps the pre-registry hash.
    let spmd = crate::opts::spmd_digest();
    if let Some(d) = spmd { w.extend([0x4F50_5453, d]); }
    w.extend(s.eos.iter().copied());
    w.extend(tune.bytes().map(|b| b as u32));
    let h = fnv32(0xB007, w.iter().copied());
    let rank = s.tp.as_ref().map_or(0, |t| t.rank);
    println!("[exl3-serve] TP=2 rank {rank}: boot config hash {h:08x} (width {} chunk {} k {} dds {} prefix {} wp16 cap {} tail {} tune {tune} \
              dds-guard prior {} ms/draft (--tp-dds-prior; TP=1 6.0) prefill-overlap {} vp-sampled {}{})",
             s.width, s.chunk, s.mtp_k, s.draft_conf, s.prefix_on, s.wp16.as_ref().map_or(0, |st| st.cap),
             crate::exl3_forward::tail_ckpt_on(), dds_prior(true),
             crate::exl3_forward::xtp::pf_overlap_desc(crate::exl3_forward::xtp::pf_overlap_code()),
             crate::exl3_forward::xtp::vp_sampled_desc(crate::exl3_forward::xtp::vp_sampled_code()),
             if crate::exl3_forward::xtp::seq_parallel_on() {
                 format!(" seq-parallel {}", crate::exl3_forward::xtp::seq_parallel_desc(crate::exl3_forward::xtp::seq_parallel_code()))
             } else { String::new() });
    println!("[exl3-serve] TP=2 rank {rank}: options {} (spmd digest {})", crate::opts::set_summary(),
             spmd.map_or("none".to_string(), |d| format!("{d:08x}")));
    if let Err(why) = tp_agree_eq(0xB0_0700, 0, 0, h) {
        crate::net::abort_link();
        anyhow::bail!("TP boot agree FAILED: the ranks resolved different serve configs (hash {h:08x} on rank {rank}): {why}");
    }
    Ok(())
}

/// TP-D: the TP=2 HEAD serve (rank 0): load this rank's shard, prove the config equal with the
/// node, wait for the node's mirror to arm (Ready), then serve HTTP with the scheduler running
/// `run_tp_head` over the retained control streams.
/// PACK-FIX: `watch` (armed by the caller before the link bring-up) keeps watching the node control
/// streams through this rank's load + boot agree — a node boot failure it reports (or its death)
/// exits the head at once with the node's reason; it is disarmed right before the Ready reads.
pub fn run_tp_head_serve(args: &[String], model_dir: &str, ctx: crate::tp::TpContext,
                         mut streams: Vec<std::net::TcpStream>, watch: crate::cluster::NodeWatch) -> Result<()> {
    let attach = tp_attach(ctx)?;
    let mut parts = build_serve(args, model_dir, Some(attach))?;
    tp_ident_knob(&mut parts.sched);
    tp_boot_agree(&mut parts.sched)?;
    watch.disarm(&streams);
    for (i, s) in streams.iter_mut().enumerate() {
        match crate::tp_serve::recv_serving(s).context("node Ready")? {
            crate::tp_serve::ServingMsg::Ready => println!("[exl3-serve] TP node rank {} READY (mirror armed)", i + 1),
            other => anyhow::bail!("expected Ready from node rank {}, got {other:?}", i + 1),
        }
    }
    println!("[exl3-serve] TP=2 serving: every §1b feature as TP=1 (graphs, MTP k={}, DDS {}, WP27 forks, prefix cache {}, \
              WP16 {}) — head-authoritative Step/Cancel/Admit(seed) + HeadFlag, agree per seam/round + pre-verify",
             parts.sched.mtp_k, parts.sched.draft_conf, parts.sched.prefix_on,
             parts.sched.wp16.as_ref().map_or("off".to_string(), |st| format!("cap {}", st.cap)));
    serve_http(parts, Some(streams))
}

/// TP-D: the TP=2 NODE serve (rank >= 1): the same boot as the head (same argv), then the mirror
/// loop on this thread, pinned to core 9.
/// PACK-FIX: a boot failure (attach, load, boot agree, pinning) is reported to the head over the
/// control stream before returning — the head watches the stream through its own load and exits
/// loudly with this reason, instead of loading for minutes and then losing the node in a probe.
pub fn run_tp_node_serve(args: &[String], model_dir: &str, ctx: crate::tp::TpContext,
                         mut stream: std::net::TcpStream) -> Result<()> {
    let boot = (|| -> Result<(Exl3Scheduler, Option<Vec<usize>>)> {
        let attach = tp_attach(ctx)?;
        let parts = build_serve(args, model_dir, Some(attach))?;
        let mut sched = parts.sched;
        tp_ident_knob(&mut sched);
        tp_boot_agree(&mut sched)?;
        anyhow::ensure!(crate::net::pin_thread(9), "TP node mirror thread failed to pin to core 9 — TP refuses to run unpinned");
        Ok((sched, parts.cpu_aff.0))
    })();
    let (mut sched, cpu_mask) = match boot {
        Ok(x) => x,
        Err(e) => {
            crate::cluster::node_report_failure(&mut stream, &format!("TP node boot failed: {e:#}"));
            return Err(e);
        }
    };
    set_worker_mask(tp_worker_mask(&cpu_mask, true));
    crate::tp_serve::send_serving(&mut stream, &crate::tp_serve::ServingMsg::Ready)?;
    println!("[exl3-serve] TP node: mirror armed (Ready sent) — waiting for the head's steps");
    sched.tp.as_mut().context("node without TP state")?.ctl = vec![stream];
    sched.run_tp_mirror()
}

/// WP24: `--spec-sampling match|ratio` and `--draft-temperature <T>` (EXL3 serve, TP=1 — no
/// TpConfig transport needed while EXL3 has no TP path). match (default) = today: drafts are the
/// head's argmax, a sampled verify row accepts while its sample == the draft. ratio = real-q
/// speculative sampling for SAMPLED requests: each draft pass samples its draft from q' (the
/// head's top-32 under --draft-temperature and the request's top-k / top-p), a verify row accepts
/// it with probability min(1, p/q'), an exact residual on reject — distribution-exact vs plain
/// sampling (gate: --probe-spec-sampling), but seeded sampled bytes change. Greedy requests are
/// unaffected. --draft-temperature default = the request's temperature.
pub fn parse_spec_sampling(args: &[String]) -> Result<(bool, Option<f32>)> {
    let spec_ratio = match arg(args, "--spec-sampling") {
        None | Some("ratio") => true, // owner ruling 2026-09-27: ratio is the default (χ² gate PASS, +5.1% T=1 thinking)
        Some("match") => false,
        Some(v) => anyhow::bail!("--spec-sampling must be match or ratio (got '{v}')"),
    };
    let draft_temp: Option<f32> = match arg(args, "--draft-temperature") {
        None => None,
        Some(v) => {
            let t: f32 = v.parse().map_err(|_| anyhow::anyhow!("--draft-temperature must be a number"))?;
            anyhow::ensure!(t.is_finite() && t > 0.0, "--draft-temperature must be > 0 (omit it = the request's temperature)");
            Some(t)
        }
    };
    if spec_ratio || draft_temp.is_some() {
        println!("[exl3-serve] spec sampling: {} (draft temperature {}){}",
                 if spec_ratio { "ratio (real-q: sampled drafts on the full draft-slice head — on DHEAD when the DHEAD-RQ line says so, accept min(1, p/q'))" } else { "match" },
                 draft_temp.map_or("= request temperature".to_string(), |t| t.to_string()),
                 if !spec_ratio && draft_temp.is_some() { " — --draft-temperature only applies with --spec-sampling ratio" } else { "" });
    }
    Ok((spec_ratio, draft_temp))
}

/// Detection: a dir is an EXL3 pack iff quantization_config.json exists and the safetensors
/// index carries `.trellis` entries (the A3 spec's quadruple manifest).
pub fn is_exl3_pack(dir: &str) -> bool {
    let dir = dir.trim_end_matches('/');
    let qc = std::path::Path::new(dir).join("quantization_config.json");
    if !qc.exists() {
        return false;
    }
    let idx = std::path::Path::new(dir).join("model.safetensors.index.json");
    if !idx.exists() {
        return false;
    }
    std::fs::read_to_string(&idx).map(|s| s.contains(".trellis")).unwrap_or(false)
}

/// Everything a booted serve needs: the scheduler, the HTTP state, and the host-side knobs the
/// launch applies (TP-D: shared by the TP=1 server, the TP=2 head and the TP=2 node mirror).
struct ServeParts {
    sched: Exl3Scheduler,
    state: AppState,
    port: u16,
    cpu_aff: (Option<Vec<usize>>, String),
    exit_on_fatal: bool,
    rx: UnboundedReceiver<BatchRequest>,
}

/// Boot the EXL3 HTTP server (TP=1). Mirrors run_server's serving surface.
pub fn run(args: &[String], model_dir: &str) -> Result<()> {
    let parts = build_serve(args, model_dir, None)?;
    serve_http(parts, None)
}

/// TP-D: resolve every serve knob from `args`, load (TP: this rank's shard) and build the
/// scheduler + HTTP state. At TP the head and the node run this with the SAME args (the head ships
/// its argv in TpConfig.exl3_mode), so both resolve every knob identically; `tp_boot_agree` then
/// proves the resolved scheduler config equal across ranks before a request is served.
fn build_serve(args: &[String], model_dir: &str, tp: Option<crate::exl3_forward::xtp::TpAttach>) -> Result<ServeParts> {
    let tp_rank: Option<i32> = tp.as_ref().map(|a| a.rank);
    let who = match tp_rank { Some(r) => format!("TP=2 rank {r}"), None => "TP=1".to_string() };
    let port: u16 = arg(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(8000);
    let max_seq_len: usize = arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let width: usize = arg(args, "--max-batch").and_then(|s| s.parse().ok()).unwrap_or(8);
    let max_pos = max_seq_len + crate::batch::decode_headroom(false);
    // --prefix-cache on|off (default on, as the NVFP4 server): per-slot recurrent-state snapshots
    // at the message boundary so a multi-turn re-send prefills only its new suffix.
    let prefix_on = !matches!(arg(args, "--prefix-cache"), Some("off"));
    if prefix_on { crate::opts::set(crate::opt!("exl3-prefix"), "1"); }
    // WP16: --prefix-ckpt-mem-gb <GiB> — the LRU byte cap of the intermediate recurrent checkpoints
    // (allocated lazily up to it). Default 4 (placeholder: the owner sets the final default);
    // 0 = no checkpoints (today's behaviour, as --wp16-off=1).
    let wp16_gb: f64 = match arg(args, "--prefix-ckpt-mem-gb") {
        None => 4.0,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--prefix-ckpt-mem-gb must be a number (GiB)"))?,
    };
    anyhow::ensure!(wp16_gb.is_finite() && wp16_gb >= 0.0, "--prefix-ckpt-mem-gb must be >= 0");
    // --ple-ram on|off|auto: where the ~30 GiB PLE n-gram table lives (RAM vs per-token pread
    // through the page cache). Output-identical either way; transported to the loader via env.
    if let Some(v) = arg(args, "--ple-ram") {
        anyhow::ensure!(matches!(v, "on" | "off" | "auto"), "--ple-ram must be on, off or auto");
        crate::opts::set(crate::opt!("ple-ram"), v);
    }
    // --kv-cache f32|f16|fp8|q8: attention KV storage format (PLAN/KV_CACHE_FORMATS.md). Default
    // q8 (owner decision 2026-09-26, after the WP25 quality gate); f32 is exact; f16 halves, fp8 (e4m3 + per-row f32 scale) and q8 (the rival's H32-rotated int8 +
    // f16 per-32 scale, WP25) quarter the long-context KV bytes. Transported to the loader via
    // [kv-cache] (an explicit env value wins when the flag is absent).
    if let Some(v) = arg(args, "--kv-cache") {
        anyhow::ensure!(matches!(v, "f32" | "f16" | "fp8" | "q8"), "--kv-cache must be f32, f16, fp8 or q8");
        crate::opts::set(crate::opt!("kv-cache"), v);
    }
    let kv_fmt = crate::exl3_forward::kv_fmt_from_opts();
    anyhow::ensure!(kv_fmt != crate::exl3_forward::KV_Q4X,
                    "[kv-cache]=q4x is the WP25 probe's deliberately broken control; it is never served");
    let kv_name = crate::exl3_forward::kv_fmt_name(kv_fmt);
    // --reasoning-effort / --thinking: the server-wide thinking defaults, as on the NVFP4 server.
    // Unset = the model's own chat template decides (Qwen3.8: thinking on at xhigh, the model
    // card's default). A request's reasoning_effort / chat_template_kwargs.enable_thinking wins.
    let reasoning_effort = match arg(args, "--reasoning-effort") {
        None => None,
        Some(e @ ("none" | "no_think" | "low" | "medium" | "high" | "xhigh")) => Some(e.to_string()),
        Some(other) => anyhow::bail!("--reasoning-effort must be none|low|medium|high|xhigh (got '{other}')"),
    };
    let thinking = match arg(args, "--thinking") {
        None => crate::tokenizer::ThinkingMode::Auto,
        Some(v) => crate::tokenizer::ThinkingMode::parse(v)
            .ok_or_else(|| anyhow::anyhow!("--thinking must be auto|on|off (got '{v}')"))?,
    };
    // HOST / RO-7: --cpu-affinity auto|off (default auto; src/cpu_affinity.rs). Resolved here so
    // a bad value fails before the load; applied after it (the load's threads keep every core).
    let cpu_aff = crate::cpu_affinity::resolve(
        crate::cpu_affinity::AffinityMode::parse(arg(args, "--cpu-affinity"))?)?;
    // TUNE T0 (PLAN/AUTOTUNE_DESIGN.md §7): resolve + freeze the tune table BEFORE the load (Load-scope
    // knobs are read there). --tune-table auto (default: a matching table from ./tune/, S section) |
    // off (built-in defaults) | <path>; --tune-draft on also applies section D (draft-only);
    // --tune-profile <MHz> picks the clock profile (default: detected, else the 2400 MHz stock one).
    // A missing or mismatched table is a loud FALLBACK to the defaults (== today, bitwise).
    {
        use crate::exl3_tune as tune;
        let sel = tune::TableSel::parse(arg(args, "--tune-table"))?;
        let draft_on = match arg(args, "--tune-draft") {
            None | Some("off") => false,
            Some("on") => true,
            Some(o) => anyhow::bail!("--tune-draft must be on or off (got '{o}')"),
        };
        let profile_mhz = match arg(args, "--tune-profile") {
            None => None,
            Some(v) => Some(v.parse::<u32>().map_err(|_| anyhow::anyhow!("--tune-profile must be MHz (got '{v}')"))?),
        };
        let posture = tune::Posture {
            lanes: width,
            mtp_max_k: crate::exl3_forward::mtp_depth_default(),
            dds: arg(args, "--draft-confidence").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.4) > 0.0,
            kv_fmt: crate::exl3_forward::kv_fmt_from_opts(),
            max_pos,
            prefill_chunk: arg(args, "--prefill-chunk").and_then(|v| v.parse().ok()).unwrap_or(2048),
        };
        tune::boot(&tune::BootReq { sel, model_dir, posture, profile_mhz, draft_on,
                                    tp: if tp_rank.is_some() { 2 } else { 1 } });
    }
    println!("[exl3-serve] {who}: loading EXL3 pack {model_dir} (width {width}, max_pos {max_pos}, kv-cache {kv_name})");
    let model = FwdModel::load_tp(model_dir, width, max_pos, tp)?;
    if model.mtp.is_some() {
        println!("[exl3-serve] M1 pack head loaded (MTP draft head for chain-verify serving)");
    }
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", model_dir.trim_end_matches('/')))?;
    // Stop tokens = the §3 union: generation_config eos (int or list) + tokenizer/config eos.
    let cfg_eos: u32 = {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/config.json", model_dir.trim_end_matches('/')))?,
        )?;
        cfg.get("eos_token_id").and_then(|e| e.as_u64()).unwrap_or(248046) as u32
    };
    let eos = tok.stop_token_ids(cfg_eos);
    println!("[exl3-serve] stop_ids={eos:?}");

    let mtp_k = crate::exl3_forward::mtp_depth_default();
    // API-parity DDS: dynamic draft stop (the rival's -dds -dc 0.6). CLI --draft-confidence <p>;
    // 0 = fixed-depth chain. Output-neutral (greedy lossless, sampled distribution-exact).
    // Default 0.4, not the rival's 0.6: our draft pass is cheaper relative to a verify row, so
    // longer drafts pay (sweep 2026-09-25, k5 greedy 1024 tok: 0.4 -> ansic 87.9 / python 77.6 /
    // prose 55.8; 0.6 -> 85.2 / 74.2 / 55.6; 0.2-0.7 all within +-2).
    let draft_conf: f64 = arg(args, "--draft-confidence").and_then(|s| s.parse().ok()).unwrap_or(0.4);
    anyhow::ensure!((0.0..1.0).contains(&draft_conf), "--draft-confidence must be in [0, 1)");
    // WP08 (owner decision 2026-09-26): the rival's streaming loop detector, ON by default at its
    // launcher setting stop_on_loop = (300, 3) (chat.py -lw 300 -lmr 3): a response whose last
    // 300 tokens are one repeating sequence of period <= 100 ends as a normal stop
    // (stop_reason "loop_detected"). --loop-detect off disables; --loop-window / --loop-min-reps
    // tune it (the rival's asserts: window > 1, 1 < reps < window). It only changes output when
    // the model is already looping.
    let loop_cfg = match arg(args, "--loop-detect") {
        None | Some("on") => {
            let w: usize = match arg(args, "--loop-window") {
                None => 300,
                Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--loop-window must be an integer"))?,
            };
            let r: usize = match arg(args, "--loop-min-reps") {
                None => 3,
                Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--loop-min-reps must be an integer"))?,
            };
            anyhow::ensure!(w > 1, "--loop-window must be > 1");
            anyhow::ensure!(r > 1 && r < w, "--loop-min-reps must be > 1 and < --loop-window");
            Some((w, r))
        }
        Some("off") => None,
        Some(v) => anyhow::bail!("--loop-detect must be on or off (got '{v}')"),
    };
    // WP15: penalty window, the rival's -penr (penalty_range, default 1024) used as both its
    // full-strength (sustain) and fading (decay) spans. Penalties themselves are per request.
    let pen_range: usize = match arg(args, "--penalty-range") {
        None => 1024,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--penalty-range must be an integer"))?,
    };
    anyhow::ensure!(pen_range <= crate::exl3_forward::PEN_RANGE_MAX,
                    "--penalty-range must be <= {}", crate::exl3_forward::PEN_RANGE_MAX);
    // WP24 (owner-consent class: seeded sampled bytes change; default match = today bit for bit).
    let (spec_ratio, draft_temp) = parse_spec_sampling(args)?;
    // Server-wide penalty defaults for requests that send none (the shared server flags; the
    // rival's defaults 1.0 / 0 / 0 when absent). Same validity rules as a request's values.
    let dflt = |name: &str, d: f32| -> Result<f32> {
        match arg(args, name) {
            None => Ok(d),
            Some(v) => v.parse::<f32>().map_err(|_| anyhow::anyhow!("{name} must be a number")),
        }
    };
    let default_rep_penalty = dflt("--default-repetition-penalty", 1.0)?;
    let default_presence_penalty = dflt("--default-presence-penalty", 0.0)?;
    let default_frequency_penalty = dflt("--default-frequency-penalty", 0.0)?;
    crate::server::validate_penalties(Some(default_rep_penalty), Some(default_presence_penalty),
                                      Some(default_frequency_penalty), None)
        .map_err(|e| anyhow::anyhow!("server penalty defaults: {e}"))?;
    println!("[exl3-serve] loop detector {} ; penalty window {pen_range} + {pen_range} tokens \
              (defaults rep {default_rep_penalty} presence {default_presence_penalty} frequency {default_frequency_penalty})",
             match loop_cfg { Some((w, r)) => format!("ON (window {w}, min reps {r})"), None => "OFF".into() });
    let tok = Arc::new(tok);
    // BUG-5 FIX (width half): mtp_round verifies at m=k+1 rows — the scratch must
    // hold the verify width even when the lane count is 1 (w1 serve panicked on
    // sc.slots[2] with a 2-element buffer). Model width (state slots) stays = lanes.
    let mut sc = FwdModel::scratch(model.dev(), &model.cfg, width.max(mtp_k + 1))?;
    // S-A3-f-b: chunked prefill width (CLI --prefill-chunk; user-facing knob).
    // S-A3-f-d Item 2: default 512 -> 2048 — the matched-C table (7838 toks,
    // fixed flash256 kernel) shows 112 tok/s @ C=512 vs 234 @ C=2048 (the KV
    // re-scan term ~N^2/2C); serve == bench at matched C, so the chunk width is
    // the whole serve-side lever. Scratch cost is ~4x of a few hundred MB.
    let chunk: usize = arg(args, "--prefill-chunk").and_then(|v| v.parse().ok()).unwrap_or(2048);
    if tp_rank.is_some() {
        anyhow::ensure!(psc_rows(chunk.max(1)) <= crate::exl3_forward::xtp::TP_MAX_ROWS,
                        "TP: --prefill-chunk {chunk} gives prefill chunks up to {} rows (C1 tail checkpoints merge up to 2C-1), \
                         above the TP all-reduce partial buffer ({} rows)", psc_rows(chunk.max(1)), crate::exl3_forward::xtp::TP_MAX_ROWS);
    }
    // S-A3-u: build the prefill scratch at startup and warm cuBLASLt (lazy kernel loads + plans
    // ~0.4 s) so request 1 does not pay it. --exl3-no-prefill-warmup skips (diagnostics).
    let mut psc0 = model.prefill_scratch(psc_rows(chunk.max(1)))?;
    if crate::opts::var(crate::opt!("exl3-no-prefill-warmup")).is_err() {
        model.prefill_warmup(&mut psc0, 0, &crate::exl3_forward::FwdModel::prefill_warmup_widths(chunk.max(1)))?;
    }
    // WP02 --exit-on-fatal (opt-in; the owner's auto-restart decision): a sticky CUDA error or a
    // scheduler panic exits 70 so a supervisor (serve_ours.sh SUPERVISE=1) restarts the server.
    // Default off: the engine goes DEAD and /health + every request answer 503 until restarted.
    let exit_on_fatal = args.iter().any(|a| a == "--exit-on-fatal");
    let inject_panic: Option<usize> = crate::opts::var(crate::opt!("inject-panic-steps")).ok().and_then(|v| v.parse().ok());
    if let Some(n) = inject_panic {
        eprintln!("[exl3-serve] DIAGNOSTIC: --inject-panic-steps={n} — injected scheduler fault at decode step {n}");
    }
    let _ = crate::exl3_forward::a_flags(); // log any GB10_A_* diagnostic escape at boot
    // WP04: --graph-precapture on|off (default on) — capture every decode graph the served config
    // reaches (step, verify widths, draft passes / chain, both attention regimes, every lane slot)
    // BEFORE listening, against the scheduler's own scratch, so no request pays a lazy capture.
    // Capture-only (WP04-v2): no kernel runs, so every device byte stays as an `off` boot has it.
    let precapture = match arg(args, "--graph-precapture") {
        None | Some("on") => true,
        Some("off") => false,
        Some(other) => anyhow::bail!("--graph-precapture must be on or off (got '{other}')"),
    };
    if precapture {
        model.precapture_decode_graphs(&mut sc, width, mtp_k, draft_conf > 0.0)?;
    } else {
        println!("[exl3-serve] graph precapture OFF (--graph-precapture off): decode graphs capture lazily");
    }
    // PFX1 (e): after the boot captures — the cuBLASLt plans of every prefill tail length are
    // built in the background (host-only work; requests are served meanwhile, lazily as before).
    let _ = model.lt_prewarm_start();
    // WP16: intermediate recurrent checkpoints — memory accounting at boot (allocated lazily).
    let wp16 = {
        let lay = model.ckpt_layout();
        let per = lay.bytes() as u64;
        let cap_bytes = (wp16_gb * (1u64 << 30) as f64) as u64;
        let cap = if per > 0 { (cap_bytes / per) as usize } else { 0 };
        if !prefix_on {
            println!("[exl3-serve] WP16 prefix checkpoints OFF (--prefix-cache off)");
            None
        } else if crate::exl3_forward::wp16_off() {
            println!("[exl3-serve] WP16 prefix checkpoints OFF (--wp16-off=1 — diagnostic escape: the prompt-end \
                      snapshot only, today's behaviour)");
            None
        } else if cap == 0 {
            println!("[exl3-serve] WP16 prefix checkpoints OFF (--prefix-ckpt-mem-gb {wp16_gb} holds no {:.1} MB checkpoint)",
                     per as f64 / 1e6);
            None
        } else {
            println!("[exl3-serve] WP16 prefix checkpoints ON: {}; cap {wp16_gb} GiB (--prefix-ckpt-mem-gb; default 4 is a \
                      placeholder — the owner sets the final default) = {cap} checkpoint(s) = {:.2} GiB at most, \
                      allocated lazily, LRU across slots, kept across a conversation's cache-hit turns. Taken at chunk \
                      ends of aligned prefills (grid from 0 or from an aligned checkpoint; never after a prompt-end \
                      prefix hit): end % {} == 0, plus every C={chunk} boundary in a prefill's last 2 chunks; keyed \
                      by C. Resumed on a prompt-end miss from the deepest checkpoint <= min(LCP, plen-1) \
                      (a 128K document holds ~18 = ~{:.1} GiB)",
                     lay.describe(), (cap as u64 * per) as f64 / (1u64 << 30) as f64,
                     crate::exl3_forward::WP16_STRIDE, 18.0 * per as f64 / (1u64 << 30) as f64);
            Some(crate::exl3_forward::CkptStore::new(width, cap))
        }
    };
    // C1: tail (message-boundary) checkpoints — opt-in, output-changing (registry prefix.tail_ckpt, N).
    let tail_ckpt_boot = wp16.is_some() && crate::exl3_forward::tail_ckpt_on();
    if tail_ckpt_boot {
        println!("[exl3-serve] C1 tail checkpoints ON (prefix.tail_ckpt = {} / --prefix-tail-ckpt=1|2: {}; OUTPUT-CHANGING, \
                  owner consent): every prefill also checkpoints at its message boundary (the prompt without its \
                  generation prompt) minus {} tokens when that saves >= {} tokens over the C grid point below it; runs \
                  are realigned to the C grid; the next chat turn resumes there",
                 if crate::exl3_forward::tail_ckpt_extra_chunk() { 2 } else { 1 },
                 if crate::exl3_forward::tail_ckpt_extra_chunk() {
                     "also splits costing one extra chunk when they promise >= 1024 tokens"
                 } else { "only splits absorbed into an existing chunk boundary" },
                 crate::exl3_forward::TAIL_BACKOFF, crate::exl3_forward::TAIL_MIN_GAIN);
    } else if crate::exl3_forward::tail_ckpt_on() {
        println!("[exl3-serve] C1 tail checkpoints requested but WP16 checkpoints are off — ignored");
    }
    let wp16_xcheck = wp16.is_some() && crate::exl3_forward::wp16_xcheck_on();
    if wp16_xcheck {
        println!("[exl3-serve] WP16 XCHECK ON (--wp16-xcheck=1, diagnostic): every checkpoint resume also prefills \
                  fresh from 0 into a scratch slot ({}) and bit-compares GDN S / conv / PLE ring + hist / head-fill \
                  carry and the first-token logits ([wp16-xcheck] lines)",
                 if width > 1 { "a free lane's slot" } else { "--max-batch 1: the resumed slot itself, then the resumed state is restored" });
    }
    let sched = Exl3Scheduler {
        model: model.clone(), sc, width,
        lanes: (0..width).map(|_| None).collect(), eos: eos.clone(), chunk, psc: Some(psc0),
        mtp_k, ctrl_round: 0, ema_plain: 0.0, draft_conf, cal: DraftCal::new(),
        prefix_on, cache: (0..width).map(|_| None).collect(),
        wp16, wp16_xcheck,
        loop_cfg, pen_range, spec_ratio, draft_temp, gate: None, tok: tok.clone(),
        tp: tp_rank.map(TpSync::new),
        exit_on_fatal, fatal: None, inject_panic, steps: 0, last_resume: (0, 0),
    };
    println!("[exl3-serve] liveness: sticky CUDA error / scheduler thread crash -> {}",
             if exit_on_fatal { "exit 70 (--exit-on-fatal)" } else { "engine DEAD, /health 503 (no --exit-on-fatal)" });
    println!("[exl3-serve] prefill chunk width = {chunk} tokens/sweep (wide-M)");
    if model.mtp.is_some() && !crate::exl3_forward::mtp_disabled_by_opt() {
        println!("[exl3-serve] MTP chain-verify ON (k={mtp_k}, greedy + sampled lanes, dynamic draft stop {}; --exl3-no-mtp / --exl3-mtp-k override)",
                 if draft_conf > 0.0 { format!("at {draft_conf}") } else { "OFF".to_string() });
    } else if model.mtp.is_some() {
        println!("[exl3-serve] MTP chain-verify OFF (--exl3-no-mtp set — diagnostics hatch)");
    }
    // WP23: depth 6/7 posture (both parts default on iff k > 5; --exl3-mtp-k=5 = the K5 build).
    if model.mtp.is_some() && draft_conf > 0.0 {
        println!("[exl3-serve] WP23 DDS: depth ceiling k={mtp_k} (max {}), depth-keyed bins {}, cost guard {} \
                  (target = max({draft_conf}, clamp(c_hat*tok/ms, 0.15, 0.8)) after 16 rounds, c_hat seeded 6.0 ms/draft; \
                  --exl3-dds-depth-bins / --exl3-dds-guard = 0|1 force)",
                 crate::exl3_forward::MTP_MAX_K,
                 if wp23_on(crate::opt!("exl3-dds-depth-bins"), mtp_k) { "ON" } else { "off" },
                 if wp23_guard_on(mtp_k) { "ON" } else { "off" });
    }

    let (stx, srx) = tokio::sync::mpsc::unbounded_channel::<crate::batch::BatchRequest>();
    let model_name = std::path::Path::new(model_dir)
        .file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "exl3-model".into());
    let state = AppState {
        sampling_defaults: crate::server::SamplingDefaults::QWEN38_CARD,
        scheduler: stx,
        tokenizer: tok,
        model_name: model_name.clone(),
        // No output cap in the request = generate until a stop token or the end of the context
        // window (the vLLM contract; was a silent 1024). --max-tokens N sets a server-wide cap.
        default_max_tokens: arg(args, "--max-tokens").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
        default_rep_penalty,
        default_presence_penalty,
        default_frequency_penalty,
        reasoning_effort,
        thinking,
        output_prompts: 0,
        max_seq_len,
        decode_headroom: crate::batch::decode_headroom(false),
        // exl3 snapshots at the prompt end: no ckpt_at render needed — except C1 tail checkpoints,
        // which take one at the message boundary (ckpt_at) of every prompt
        prefix_cache: prefix_on && tail_ckpt_boot,
        vision_tower: None,
        vision_gpu: None,
        vision_cpu: false,
        stop_ids: eos,
        otel: None,
    };
    Ok(ServeParts { sched, state, port, cpu_aff, exit_on_fatal, rx: srx })
}

/// TP-D: the host-thread mask at TP — the big cores minus the launch core (9, the pinned scheduler
/// thread) and the RDMA proxy core (19). TP=1: the resolved mask unchanged.
fn tp_worker_mask(m: &Option<Vec<usize>>, tp: bool) -> Option<Vec<usize>> {
    let m = m.clone()?;
    if !tp { return Some(m); }
    // TP-F: the dual-rail prefill transport pins its second proxy on TP_AUX_PROXY_CORE (18).
    let aux = if crate::exl3_forward::xtp::prefill_xport_mode() == 2 { crate::exl3_forward::xtp::TP_AUX_PROXY_CORE as usize } else { usize::MAX };
    let f: Vec<usize> = m.into_iter().filter(|&c| c != 9 && c != 19 && c != aux).collect();
    if f.is_empty() { None } else { Some(f) }
}

/// TP-D: threads the pinned (single-core) TP scheduler thread spawns — the prefill PLE row workers —
/// would inherit its one-core mask and serialize on core 9; they re-pin to this mask instead.
fn set_worker_mask(m: Option<Vec<usize>>) {
    if let Some(m) = m {
        println!("[exl3-serve] TP: scheduler pinned to core 9; spawned host workers use cpus {}",
                 crate::cpu_affinity::format_cpu_list(&m));
        crate::cpu_affinity::set_worker_mask(m);
    }
}

/// Spawn the scheduler thread and serve HTTP on this thread (never returns on success).
/// `tp_ctl` Some = the TP=2 head: the scheduler runs `run_tp_head` over the node control streams,
/// on a thread pinned to core 9 (the TP launch-thread rule), and the HTTP/worker threads keep the
/// big-core mask minus the launch (9) and proxy (19) cores.
fn serve_http(parts: ServeParts, tp_ctl: Option<Vec<std::net::TcpStream>>) -> Result<()> {
    let ServeParts { sched, state, port, cpu_aff, exit_on_fatal, rx: srx } = parts;
    let model_name = state.model_name.clone();
    // HOST / RO-7: pin THIS thread (it runs the HTTP runtime below) to the big cores BEFORE the
    // scheduler thread is spawned — a new thread inherits its creator's mask (Linux), and so do
    // the threads either of them spawns later (prefill PLE workers, tokio's blocking pool).
    println!("[exl3-serve] {}", cpu_aff.1);
    let mask = tp_worker_mask(&cpu_aff.0, tp_ctl.is_some());
    if let Some(cpus) = mask.as_ref() {
        if let Err(e) = crate::cpu_affinity::pin_current_thread(cpus) {
            println!("[exl3-serve] cpu affinity: pinning FAILED ({e:#}) — threads left to the OS scheduler");
        }
    }
    match tp_ctl {
        None => {
            std::thread::spawn(move || {
                let _guard = FatalGuard { exit_on_fatal };
                sched.run(srx)
            });
        }
        Some(streams) => {
            std::thread::spawn(move || {
                // any exit of the TP scheduler thread ends the process (a TP head cannot serve on
                // without its lockstep): exit 70, the NVFP4 TP head's contract
                let _guard = FatalGuard { exit_on_fatal: true };
                if !crate::net::pin_thread(9) {
                    eprintln!("\n*** FATAL: TP head scheduler failed to pin to core 9 — TP refuses to run unpinned. Exiting. ***\n");
                    std::process::exit(70);
                }
                set_worker_mask(mask);
                let r = sched.run_tp_head(srx, streams);
                match r {
                    Ok(()) => eprintln!("[exl3-serve] TP head: request channel closed — session over"),
                    Err(e) => eprintln!("\n*** FATAL: the TP scheduler failed: {e:#}. Exiting (70). ***\n"),
                }
                std::thread::sleep(std::time::Duration::from_millis(300)); // let the error events flush
                std::process::exit(70);
            });
        }
    }
    let rt = tokio::runtime::Builder::new_current_thread().enable_all()
        .build().context("tokio runtime")?;
    println!("[exl3-serve] listening on 0.0.0.0:{port} (model id: {model_name})");
    rt.block_on(async move {
        let app = create_router(state);
        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await.unwrap();
        // WP02: TCP_NODELAY — an SSE token event is a few dozen bytes; Nagle held them for the ACK
        axum::serve(listener, app).tcp_nodelay(true).await.unwrap();
    });
    Ok(())
}

// ---- WP24 gate, stage B (served level): see src/wp24.rs for stage A and the statistics ----

/// The gate's three arms: ratio MTP (under test), match MTP (the control: today's rule, known
/// exact), plain sampled decode with MTP off (the reference law).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateArm {
    Plain = 0,
    Ratio = 1,
    Match = 2,
}

struct GateArmOut {
    seqs: Vec<Vec<u32>>,
    stats: GateStats,
    secs: f64,
}

const WP24_GATE_PROMPT: &str = "Write a short story about a lighthouse keeper who finds a message in a bottle.\n\nThe lighthouse keeper";

/// One arm of one condition: `trials` seeded requests of `prompt` through the REAL scheduler
/// (admit -> prefix-cache restore -> seam step -> prime -> step() rounds), `positions` tokens
/// each (ignore_eos, so every trial yields exactly that many), in waves of `width` lanes. The
/// plain arm clears each lane's MTP state after admit (the per-request §6 fallback's exact
/// hand-off: the lane then decodes through step_plain). Trial seeds: distinct per (condition,
/// arm, trial) — independent RNG streams everywhere (AGENTS §3).
#[allow(clippy::too_many_arguments)]
fn wp24_gate_arm(sched: &mut Exl3Scheduler, arm: GateArm, prompt: &[u32], t: f32, top_p: f32,
                 top_k: usize, trials: usize, positions: usize, base: u64) -> Result<GateArmOut> {
    sched.spec_ratio = arm == GateArm::Ratio;
    sched.cal = DraftCal::new(); // each arm learns its own DDS map (never another arm's labels)
    sched.gate = Some(GateStats::default());
    let t0 = std::time::Instant::now();
    let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(trials);
    let mut done = 0usize;
    while done < trials {
        let nb = (trials - done).min(sched.width);
        let mut rxs = Vec::with_capacity(nb);
        for j in 0..nb {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<TokEvent>();
            let seed = crate::wp24::mix64(base ^ ((arm as u64 + 1) << 56) ^ (done + j) as u64);
            sched.admit(BatchRequest {
                prompt: prompt.to_vec(),
                max_new: positions,
                temperature: t,
                top_p,
                top_k,
                rep_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                min_new: 0,
                ignore_eos: true,
                tx,
                seed: Some(seed),
                ckpt_at: None,
                domain: crate::batch::Domain::General,
                received_at: std::time::Instant::now(),
                image_embeds: None,
                image_spans: Vec::new(),
                schema: None,
            });
            if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("WP24 gate: engine fatal at admit: {f}"); }
            rxs.push(rx);
        }
        if arm == GateArm::Plain {
            for l in sched.lanes.iter_mut().flatten() { l.mtp = None; }
        }
        let mut steps = 0usize;
        while sched.lanes.iter().any(|l| l.is_some()) {
            sched.step()?;
            if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("WP24 gate: engine fatal: {f}"); }
            steps += 1;
            anyhow::ensure!(steps <= 4 * positions + 64, "WP24 gate: lanes still live after {steps} steps");
        }
        for mut rx in rxs {
            let (mut toks, mut reason) = (Vec::with_capacity(positions), None);
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    TokEvent::Tok(x) => toks.push(x),
                    TokEvent::Finish { reason: r } => reason = Some(r),
                }
            }
            anyhow::ensure!(reason.as_deref() == Some("length") && toks.len() == positions,
                            "WP24 gate: a {arm:?} trial ended {reason:?} after {} tokens (want {positions}, \"length\")",
                            toks.len());
            seqs.push(toks);
        }
        done += nb;
    }
    let stats = sched.gate.take().unwrap_or_default();
    Ok(GateArmOut { seqs, stats, secs: t0.elapsed().as_secs_f64() })
}

/// `--probe-spec-sampling --model-dir <EXL3 pack>` stage B: the served-level distribution gate.
/// Per condition (T in {0.7, 1.0} x top_p in {0.8, 0.95}, top_k 20, >= 2 seeds): `--trials`
/// requests per arm of one raw prompt (`--prompt`, default a story opening), each generating
/// `--positions` tokens (default 6: x1 = the seam step, identical code in every arm — a built-in
/// control; x2.. = MTP rounds; the remaining-budget clamp lets round 1 draft up to positions-2).
/// Tests (reference-free two-sample chi-square, wp24::gof2 — |z| < 4 passes):
///   GATE    ratio-vs-plain per position and on the joint (x2, x3);
///   CONTROL match-vs-plain, the same tests (today's rule is exact: a failure = the harness is
///           broken, and the condition FAILS);
///   NEG     the ratio arm's first-round drafts d_0 vs plain x2 (d_0 ~ q' != p: must be REJECTED
///           in at least one condition, else the gate had no power on this prompt).
/// Success signals asserted, not just the absence of errors: every trial returns exactly
/// `positions` tokens with finish "length"; the ratio arm ran real-q rounds (>0) and the match
/// arm none; both MTP arms drafted. Output: one line per condition, then RESULT. `--quick` = the
/// owner's config (T 1.0, top_p 0.95) only, as the first rung of the ladder.
pub fn wp24_served_gate(args: &[String], model_dir: &str) -> Result<bool> {
    let num = |name: &str, d: usize| -> Result<usize> {
        match arg(args, name) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| anyhow::anyhow!("{name} must be an integer")),
        }
    };
    let trials = num("--trials", 600)?.max(50);
    let positions = num("--positions", 6)?.clamp(3, 32);
    let width = num("--max-batch", 4)?.clamp(1, 16);
    let n_seeds = num("--seeds", 2)?.max(2);
    let top_k = num("--top-k", 20)?;
    let max_seq_len = num("--max-seq-len", 4096)?;
    let draft_conf: f64 = match arg(args, "--draft-confidence") {
        None => 0.4,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--draft-confidence must be a number"))?,
    };
    anyhow::ensure!((0.0..1.0).contains(&draft_conf), "--draft-confidence must be in [0, 1)");
    let (_, draft_temp) = parse_spec_sampling(args)?; // --draft-temperature applies to the ratio arm
    let text = arg(args, "--prompt").unwrap_or(WP24_GATE_PROMPT).to_string();
    // the served posture: prefix snapshots on (run() sets this for --prefix-cache on), KV format
    // from --kv-cache / [kv-cache] / the default, graphs precaptured like the server boot
    crate::opts::set(crate::opt!("exl3-prefix"), "1");
    if let Some(v) = arg(args, "--kv-cache") {
        anyhow::ensure!(matches!(v, "f32" | "f16" | "fp8" | "q8"), "--kv-cache must be f32, f16, fp8 or q8");
        crate::opts::set(crate::opt!("kv-cache"), v);
    }
    anyhow::ensure!(!crate::exl3_forward::mtp_disabled_by_opt(), "WP24 gate: --exl3-no-mtp is set — the MTP arms need the head");
    let max_pos = max_seq_len + crate::batch::decode_headroom(false);
    println!("WP24 served gate: loading {model_dir} (width {width}, max_pos {max_pos}, kv-cache {})",
             crate::exl3_forward::kv_fmt_name(crate::exl3_forward::kv_fmt_from_opts()));
    let model = FwdModel::load(model_dir, width, max_pos)?;
    anyhow::ensure!(model.mtp.is_some(), "WP24 gate: the pack has no MTP draft head");
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", model_dir.trim_end_matches('/')))?;
    let cfg_eos: u32 = {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/config.json", model_dir.trim_end_matches('/')))?)?;
        cfg.get("eos_token_id").and_then(|e| e.as_u64()).unwrap_or(248046) as u32
    };
    let eos = tok.stop_token_ids(cfg_eos);
    let prompt = tok.encode(&text, false)?;
    let mtp_k = crate::exl3_forward::mtp_depth_default();
    anyhow::ensure!(prompt.len() >= 2 && prompt.len() + positions + mtp_k + 4 < model.max_pos(),
                    "WP24 gate: prompt of {} tokens does not fit the window", prompt.len());
    let mut sc = FwdModel::scratch(model.dev(), &model.cfg, width.max(mtp_k + 1))?;
    let chunk = 2048usize;
    let psc0 = model.prefill_scratch(chunk)?;
    model.precapture_decode_graphs(&mut sc, width, mtp_k, draft_conf > 0.0)?;
    let tok = Arc::new(tok);
    let mut sched = Exl3Scheduler {
        model: model.clone(), sc, width,
        lanes: (0..width).map(|_| None).collect(), eos, chunk, psc: Some(psc0),
        mtp_k, ctrl_round: 0, ema_plain: 0.0, draft_conf, cal: DraftCal::new(),
        prefix_on: true, cache: (0..width).map(|_| None).collect(),
        wp16: None, wp16_xcheck: false,
        loop_cfg: None, pen_range: 1024, spec_ratio: false, draft_temp, gate: None, tok: tok.clone(), tp: None,
        exit_on_fatal: false, fatal: None, inject_panic: None, steps: 0, last_resume: (0, 0),
    };
    let quick = args.iter().any(|a| a == "--quick");
    wp24_gate_conditions(&mut sched, &prompt, &text, trials, positions, n_seeds, top_k, draft_conf, draft_temp, quick)
}

/// The WP24 served gate's condition loop on a constructed scheduler (TP-C: shared by the TP=1
/// gate above and the TP=2 spec program, where every rank runs it in lockstep).
#[allow(clippy::too_many_arguments)]
fn wp24_gate_conditions(sched: &mut Exl3Scheduler, prompt: &[u32], text: &str, trials: usize, positions: usize,
                        n_seeds: usize, top_k: usize, draft_conf: f64, draft_temp: Option<f32>, quick: bool) -> Result<bool> {
    let mtp_k = sched.mtp_k;
    sched.draft_conf = draft_conf;
    sched.draft_temp = draft_temp;
    println!("WP24 served gate: prompt {} tokens {:?}; {trials} trials x 3 arms (plain | ratio | match) per condition, \
              {positions} positions, top_k {top_k}, {n_seeds} seeds, DDS {draft_conf}, depth {mtp_k}, draft temperature {}",
             prompt.len(), text.chars().take(60).collect::<String>(),
             draft_temp.map_or("= T".to_string(), |x| x.to_string()));
    type Hist = std::collections::HashMap<u64, u64>;
    let hist = |seqs: &[Vec<u32>], j: usize| -> Hist {
        let mut h = Hist::new();
        for s in seqs { *h.entry(s[j] as u64).or_insert(0) += 1; }
        h
    };
    let pair = |seqs: &[Vec<u32>], j: usize| -> Hist {
        let mut h = Hist::new();
        for s in seqs { *h.entry(((s[j] as u64) << 32) | s[j + 1] as u64).or_insert(0) += 1; }
        h
    };
    let t_all = std::time::Instant::now();
    let (mut all_ok, mut neg_rejected, mut conds) = (true, false, 0usize);
    // --quick: the owner's daily config only (T 1.0, top_p 0.95) — the ladder's first rung; the
    // full gate is all four configs
    let configs: &[(f32, f32)] = if quick { &[(1.0, 0.95)] } else { &[(0.7, 0.8), (0.7, 0.95), (1.0, 0.8), (1.0, 0.95)] };
    let total = configs.len() * n_seeds;
    for &(t, top_p) in configs {
        for si in 0..n_seeds {
            let base = crate::wp24::mix64(0x5EED_2400_0000_0000u64
                ^ (((t.to_bits() as u64) << 32) | top_p.to_bits() as u64)
                ^ (si as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let plain = wp24_gate_arm(sched, GateArm::Plain, prompt, t, top_p, top_k, trials, positions, base)?;
            let ratio = wp24_gate_arm(sched, GateArm::Ratio, prompt, t, top_p, top_k, trials, positions, base)?;
            let matc = wp24_gate_arm(sched, GateArm::Match, prompt, t, top_p, top_k, trials, positions, base)?;
            let (mut zr, mut zm) = (Vec::with_capacity(positions + 1), Vec::with_capacity(positions + 1));
            for j in 0..positions {
                zr.push(crate::wp24::gof2_maps(&hist(&ratio.seqs, j), &hist(&plain.seqs, j)).1);
                zm.push(crate::wp24::gof2_maps(&hist(&matc.seqs, j), &hist(&plain.seqs, j)).1);
            }
            zr.push(crate::wp24::gof2_maps(&pair(&ratio.seqs, 1), &pair(&plain.seqs, 1)).1);
            zm.push(crate::wp24::gof2_maps(&pair(&matc.seqs, 1), &pair(&plain.seqs, 1)).1);
            let mut d0h = Hist::new();
            for &d in &ratio.stats.d0 { *d0h.entry(d as u64).or_insert(0) += 1; }
            let neg = crate::wp24::gof2_maps(&d0h, &hist(&plain.seqs, 1));
            let signals = ratio.stats.ratio_rounds > 0 && ratio.stats.drafted > 0
                && matc.stats.ratio_rounds == 0 && matc.stats.drafted > 0;
            let gate_ok = zr.iter().all(|z| z.abs() < 4.0);
            let ctrl_ok = zm.iter().all(|z| z.abs() < 4.0);
            let ok = gate_ok && ctrl_ok && signals;
            all_ok &= ok;
            neg_rejected |= neg.1 >= 4.0 && !ratio.stats.d0.is_empty();
            conds += 1;
            let fz = |v: &[f64]| v.iter().map(|z| format!("{z:+.2}")).collect::<Vec<_>>().join(" ");
            let arm = |a: &GateArmOut| {
                let s = &a.stats;
                format!("rounds {} ratio-rounds {} accept {:.1}% tok/round {:.2} ({:.0} s)", s.rounds, s.ratio_rounds,
                        100.0 * s.accepted as f64 / s.drafted.max(1) as f64,
                        (s.accepted + s.rounds) as f64 / s.rounds.max(1) as f64, a.secs)
            };
            println!("  [served] T={t} top_p={top_p} seed#{si}: n={trials}/arm | GATE ratio-vs-plain z[pos 1..{positions}, pair 2-3] = {} | \
                      CONTROL match-vs-plain z = {} | NEG d0-vs-plain-x2 z={:+.1} ({} d0) | ratio: {} | match: {} | plain {:.0} s{} -> {}",
                     fz(&zr), fz(&zm), neg.1, ratio.stats.d0.len(), arm(&ratio), arm(&matc), plain.secs,
                     if signals { "" } else { " | SUCCESS SIGNALS MISSING (ratio rounds / drafts)" },
                     if ok { "PASS" } else if !ctrl_ok { "FAIL (control failed: harness suspect)" } else { "FAIL" });
            if conds == 1 {
                let per = t_all.elapsed().as_secs_f64();
                println!("  [served] projected total ~{:.0} min ({total} conditions at {:.0} s each)", per * total as f64 / 60.0, per);
            }
        }
    }
    if !neg_rejected {
        println!("  [served] negative control never rejected: the gate had no power to see q' != p on this prompt");
        println!("RESULT: SERVED_GATE_NO_POWER");
        return Ok(false);
    }
    println!("RESULT: {} ({conds} conditions{}, {:.1} min)",
             if all_ok { "SERVED_DISTRIBUTION_OK (ratio-mode MTP emits the plain sampler's law at every tested position)" }
             else { "SERVED_DISTRIBUTION_MISMATCH" },
             if quick { ", --quick: T 1.0 / top_p 0.95 only — run without --quick for the full gate" } else { "" },
             t_all.elapsed().as_secs_f64() / 60.0);
    Ok(all_ok)
}

#[cfg(test)]
mod wp23_tests {
    use super::*;

    fn cal(depth: bool) -> DraftCal {
        DraftCal {
            bins: Default::default(),
            total: 0.0,
            dbins: if depth { Some(vec![Default::default(); WP23_DEPTH_BUCKETS]) } else { None },
        }
    }

    /// deterministic label stream: (pass index, score, accepted), deeper passes less likely
    fn stream(n: usize) -> Vec<(usize, f32, bool)> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        (0..n).map(|_| {
            let i = (next() % 7) as usize;
            let score = 8.0 + (next() % 2400) as f32 / 100.0;
            let acc = ((next() % 100) as f32 / 100.0) < (score / 32.0 - 0.06 * i as f32);
            (i, score, acc)
        }).collect()
    }

    fn feed(c: &mut DraftCal, labels: &[(usize, f32, bool)]) {
        for (j, &(i, sc, a)) in labels.iter().enumerate() {
            c.add_label(i, sc, a);
            if j % 3 == 2 { c.decay_step(); }
        }
    }

    #[test]
    fn depth_bins_off_is_the_legacy_calibrator() {
        let mut c = cal(false);
        feed(&mut c, &stream(4000));
        for i in 0..8 {
            for q in 0..400 {
                let sc = 4.0 + q as f32 * 0.1;
                assert_eq!(c.estimate_at(i, sc).to_bits(), c.estimate(sc).to_bits(), "i {i} score {sc}");
            }
        }
    }

    #[test]
    fn depth_bins_keep_the_pooled_map_identical() {
        let (mut a, mut b) = (cal(false), cal(true));
        let l = stream(4000);
        feed(&mut a, &l);
        feed(&mut b, &l);
        assert_eq!(a.total.to_bits(), b.total.to_bits());
        assert_eq!(a.bins.len(), b.bins.len());
        for ((ka, va), (kb, vb)) in a.bins.iter().zip(b.bins.iter()) {
            assert_eq!(ka, kb);
            assert_eq!((va.0.to_bits(), va.1.to_bits()), (vb.0.to_bits(), vb.1.to_bits()));
        }
    }

    #[test]
    fn depth_estimate_uses_its_bucket_else_the_pool() {
        let mut c = cal(true);
        // pass 0: logit 20 always accepted; pass 5 (bucket 3): logit 20 accepted 1 in 4
        for _ in 0..40 { c.add_label(0, 20.5, true); }
        for j in 0..40 { c.add_label(5, 20.5, j % 4 == 0); }
        assert!(c.total >= DraftCal::BURN_IN);
        let pooled = c.estimate(20.5);
        assert!((pooled - 50.0 / 80.0).abs() < 1e-12);
        assert!((c.estimate_at(0, 20.5) - 1.0).abs() < 1e-12);
        assert!((c.estimate_at(6, 20.5) - 0.25).abs() < 1e-12); // pass 6 shares bucket 3
        // bucket 2 (passes 3, 4) has no labels: the pooled estimate
        assert_eq!(c.estimate_at(3, 20.5).to_bits(), pooled.to_bits());
        // below every populated bin of the bucket: the pooled fallback (its lowest-bin rule)
        assert_eq!(c.estimate_at(5, 12.0).to_bits(), c.estimate(12.0).to_bits());
        // burn-in: optimistic 1.0
        let mut d = cal(true);
        for _ in 0..10 { d.add_label(5, 20.5, false); }
        assert_eq!(d.estimate_at(5, 20.5), 1.0);
    }

    #[test]
    fn cost_fit_prior_and_slope() {
        assert_eq!(CostFit::default().slope(), 6.0);
        // constant drafts/round: no spread, the prior stays
        let mut g = CostFit::default();
        for _ in 0..200 { g.add(7.0, 56.0); }
        assert!((g.slope() - 6.0).abs() < 1e-9, "{}", g.slope());
        // spread drafts with a true 9 ms/draft slope: identified (shrunk slightly toward 6)
        let mut h = CostFit::default();
        for r in 0..200 { let x = (r % 7 + 1) as f64; h.add(x, 30.0 + 9.0 * x); }
        let s = h.slope();
        assert!(s > 8.7 && s < 9.0, "{s}");
        // never negative
        let mut n = CostFit::default();
        for r in 0..200 { let x = (r % 7 + 1) as f64; n.add(x, 90.0 - 30.0 * x); }
        assert!(n.slope() >= 0.0);
    }

    #[test]
    fn guard_target_rules() {
        // off, too early, or no EMAs: the base target
        assert_eq!(wp23_target(0.4, false, 100, 6.0, 5.6, 56.0), 0.4);
        assert_eq!(wp23_target(0.4, true, 15, 6.0, 5.6, 56.0), 0.4);
        assert_eq!(wp23_target(0.4, true, 100, 6.0, 0.0, 0.0), 0.4);
        // AnsiC-like: 6 ms/draft at 0.1 tok/ms -> 0.6
        assert!((wp23_target(0.4, true, 16, 6.0, 5.6, 56.0) - 0.6).abs() < 1e-12);
        // prose-like: 6 * 2.7 / 45 = 0.36 < base -> base (the guard only raises)
        assert_eq!(wp23_target(0.4, true, 100, 6.0, 2.7, 45.0), 0.4);
        // long context, expensive rows: clamped at 0.8
        assert_eq!(wp23_target(0.4, true, 100, 20.0, 5.0, 60.0), 0.8);
        // an operator base above the guard wins
        assert_eq!(wp23_target(0.9, true, 100, 20.0, 5.0, 60.0), 0.9);
    }

    #[test]
    fn depth_buckets_cover_the_window() {
        let b: Vec<usize> = (0..crate::exl3_forward::MTP_MAX_K).map(wp23_depth_bucket).collect();
        assert_eq!(b, vec![0, 1, 1, 2, 2, 3, 3]);
        assert!(b.iter().all(|&x| x < WP23_DEPTH_BUCKETS));
    }
}

// ---------------------------------------------------------------------------------------------
// TP-C: the spec program (`--probe-exl3-tpspec`) — the SERVED scheduler, offline (no HTTP), at
// TP=1 (single process) or as one rank of TP=2 (SPMD, every rank runs this function). Every
// request goes through Exl3Scheduler::admit (prefix cache -> chunked prefill -> seam step ->
// head prime) and step() (graphed DDS rounds / plain steps / control rounds), exactly the
// serve's code; at TP=2 each step is a lockstep point (TpSync). Gates:
//   G-T1-e  greedy: every spec arm (DDS at the served depth, fixed chains at each --spec-depths)
//           emits exactly the plain arm's tokens, per prompt and rep (and plain is rep-stable);
//   G-T1-g  acceptance: tok/round of the DDS arm per prompt class (compare TP=2 vs the TP=1 run
//           of this same program);
//   speed   DDS ms/round and tok/s, plain ms/token (bring-up readings, not perf claims);
//   G-T1-b  (graphed) with ident on: logits rows / residual / taps digests identical across ranks
//           at every lockstep point of rep 0;
//   G-T1-h  the WP24 served distribution gate (ratio + match vs plain, chi-square) on the same
//           scheduler (--spec-samp quick|full).
// ---------------------------------------------------------------------------------------------

/// Options of the spec program; ride TpConfig.exl3_mode as
/// "spec:gen=N:reps=N:depths=1,2:dds=0|1:width=N:prefix=0|1:samp=0|1|2:trials=N:ident=0|1".
#[derive(Clone, Debug, PartialEq)]
pub struct SpecOpts {
    pub gen: usize,
    pub reps: usize,
    pub depths: Vec<usize>,
    pub dds: bool,
    pub width: usize,
    pub prefix: bool,
    /// 0 = no sampled gate, 1 = WP24 --quick, 2 = the full ladder
    pub samp: u8,
    pub trials: usize,
    pub ident: bool,
}

impl Default for SpecOpts {
    fn default() -> Self {
        SpecOpts { gen: 1024, reps: 3, depths: (1..=crate::exl3_forward::MTP_MAX_K).collect(), dds: true, width: 1,
                   prefix: true, samp: 0, trials: 600, ident: true }
    }
}

impl SpecOpts {
    pub fn parse(mode: &str) -> Result<Self> {
        let mut it = mode.split(':');
        anyhow::ensure!(it.next() == Some("spec"), "EXL3 TP: '{mode}' is not a spec program");
        let mut o = SpecOpts::default();
        for kv in it {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            let b = |v: &str| v == "1";
            match k {
                "gen" => o.gen = v.parse().context("spec gen=N")?,
                "reps" => o.reps = v.parse().context("spec reps=N")?,
                "depths" => o.depths = if v.is_empty() { Vec::new() } else {
                    v.split(',').map(|x| x.parse::<usize>().context("spec depths=a,b")).collect::<Result<_>>()?
                },
                "dds" => o.dds = b(v),
                "width" => o.width = v.parse().context("spec width=N")?,
                "prefix" => o.prefix = b(v),
                "samp" => o.samp = v.parse().context("spec samp=0|1|2")?,
                "trials" => o.trials = v.parse().context("spec trials=N")?,
                "ident" => o.ident = b(v),
                _ => anyhow::bail!("EXL3 TP: unknown spec option '{kv}' in '{mode}'"),
            }
        }
        anyhow::ensure!(o.depths.iter().all(|&d| (1..=crate::exl3_forward::MTP_MAX_K).contains(&d)),
                        "spec depths must be in 1..={}", crate::exl3_forward::MTP_MAX_K);
        anyhow::ensure!(o.width >= 1 && o.reps >= 1 && o.gen >= 2, "spec: width/reps >= 1, gen >= 2");
        Ok(o)
    }
    pub fn to_mode(&self) -> String {
        format!("spec:gen={}:reps={}:depths={}:dds={}:width={}:prefix={}:samp={}:trials={}:ident={}",
                self.gen, self.reps, self.depths.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
                self.dds as u8, self.width, self.prefix as u8, self.samp, self.trials, self.ident as u8)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SpecArm {
    Plain,
    Dds,
    Fixed(usize),
}

impl SpecArm {
    fn name(&self) -> String {
        match self { SpecArm::Plain => "plain".into(), SpecArm::Dds => "dds".into(), SpecArm::Fixed(d) => format!("d{d}") }
    }
}

struct SpecArmOut {
    toks: Vec<u32>,
    st: GateStats,
    secs: f64,
    reason: String,
}

/// One greedy request of `prompt` through the real scheduler under `arm` (plain = the §6 per-request
/// fallback's exact hand-off: the lane's MTP state cleared after admit, then step_plain).
fn spec_arm(sched: &mut Exl3Scheduler, arm: SpecArm, prompt: &[u32], gen: usize) -> Result<SpecArmOut> {
    let kdef = crate::exl3_forward::mtp_depth_default();
    match arm {
        SpecArm::Plain | SpecArm::Dds => { sched.mtp_k = kdef; sched.draft_conf = 0.4; }
        SpecArm::Fixed(d) => { sched.mtp_k = d; sched.draft_conf = 0.0; }
    }
    sched.cal = DraftCal::new();
    sched.gate = Some(GateStats::default());
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<TokEvent>();
    let t0 = std::time::Instant::now();
    sched.admit(BatchRequest {
        prompt: prompt.to_vec(), max_new: gen, temperature: 0.0, top_p: 1.0, top_k: 0,
        rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0, min_p: 0.0, min_new: 0,
        ignore_eos: false, tx, seed: Some(0x7C0D_E5EE_D000_0001), ckpt_at: None,
        domain: crate::batch::Domain::General, received_at: std::time::Instant::now(),
        image_embeds: None, image_spans: Vec::new(), schema: None,
    });
    if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("spec: engine fatal at admit: {f}"); }
    if arm == SpecArm::Plain {
        for l in sched.lanes.iter_mut().flatten() { l.mtp = None; }
    }
    let mut steps = 0usize;
    while sched.lanes.iter().any(|l| l.is_some()) {
        sched.step()?;
        if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("spec: engine fatal: {f}"); }
        steps += 1;
        anyhow::ensure!(steps <= 4 * gen + 64, "spec: lane still live after {steps} steps");
    }
    let secs = t0.elapsed().as_secs_f64();
    let (mut toks, mut reason) = (Vec::with_capacity(gen), String::new());
    while let Ok(ev) = rx.try_recv() {
        match ev {
            TokEvent::Tok(x) => toks.push(x),
            TokEvent::Finish { reason: r } => reason = r,
        }
    }
    anyhow::ensure!(!reason.starts_with("error"), "spec: {} request failed: {reason}", arm.name());
    Ok(SpecArmOut { toks, st: sched.gate.take().unwrap_or_default(), secs, reason })
}

fn spec_pack(prompts: &[(String, Vec<u32>)]) -> Vec<u32> {
    let mut v = vec![prompts.len() as u32];
    for (n, ids) in prompts {
        let mut b = n.as_bytes().to_vec();
        b.resize(16, 0);
        v.extend(b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        v.push(ids.len() as u32);
        v.extend_from_slice(ids);
    }
    v
}

fn spec_unpack(v: &[u32]) -> Result<Vec<(String, Vec<u32>)>> {
    let n = *v.first().context("spec: empty prompt pack")? as usize;
    let mut i = 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        anyhow::ensure!(i + 5 <= v.len(), "spec: truncated prompt pack");
        let b: Vec<u8> = v[i..i + 4].iter().flat_map(|w| w.to_le_bytes()).take_while(|&c| c != 0).collect();
        let len = v[i + 4] as usize;
        i += 5;
        anyhow::ensure!(i + len <= v.len(), "spec: truncated prompt pack");
        out.push((String::from_utf8_lossy(&b).to_string(), v[i..i + len].to_vec()));
        i += len;
    }
    Ok(out)
}

/// The spec program (see the section comment). `ctx` = Some on both TP ranks (the head passes
/// `prompts`, the node None — they arrive over the link); None = TP=1 (prompts required).
pub fn run_spec(model_dir: &str, ctx: Option<crate::tp::TpContext>, prompts: Option<Vec<(String, Vec<u32>)>>,
                max_pos: usize, o: &SpecOpts) -> Result<()> {
    use sha2::{Digest, Sha256};
    let (prompts, attach, rank) = match ctx {
        Some(mut ctx) => {
            ctx.sanity()?;
            ctx.branch_check(&crate::tp::TpBranch::Exl3Xtp)?;
            let packed = prompts.as_ref().map(|p| spec_pack(p));
            let (seq, _, _) = match &packed {
                Some(v) => ctx.broadcast_prompt(Some((v, 0, 0)))?,
                None => ctx.broadcast_prompt(None)?,
            };
            let prompts = spec_unpack(&seq)?;
            let (rank, world, link) = ctx.into_parts();
            anyhow::ensure!(crate::net::pin_thread(9), "launch thread failed to pin to core 9 — TP refuses to run unpinned");
            (prompts, Some(crate::exl3_forward::xtp::TpAttach { rank, world, link }), rank)
        }
        None => (prompts.context("spec (TP=1): no prompts")?, None, 0),
    };
    let tp_on = attach.is_some();
    let who = if tp_on { format!("TP=2 rank {rank}") } else { "TP=1".to_string() };
    if o.prefix { crate::opts::set(crate::opt!("exl3-prefix"), "1"); }
    println!("[tpspec] {who}: program {} — {} prompt(s) {:?}; loading {model_dir} (width {}, max_pos {max_pos}, kv-cache {})",
             o.to_mode(), prompts.len(), prompts.iter().map(|(n, p)| format!("{n}:{}", p.len())).collect::<Vec<_>>(),
             o.width, crate::exl3_forward::kv_fmt_name(crate::exl3_forward::kv_fmt_from_opts()));
    let t_load = std::time::Instant::now();
    let model = FwdModel::load_tp(model_dir, o.width, max_pos, attach)?;
    anyhow::ensure!(model.mtp.is_some(), "spec: the pack has no MTP draft head");
    println!("[tpspec] {who}: loaded in {:.1}s (trunk per rank: {} q / {} kv heads, GDN {} k / {} v heads)",
             t_load.elapsed().as_secs_f32(), model.tc.num_heads, model.tc.num_kv_heads,
             model.tc.lin_num_k_heads, model.tc.lin_num_v_heads);
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", model_dir.trim_end_matches('/')))?;
    let cfg_eos: u32 = {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/config.json", model_dir.trim_end_matches('/')))?)?;
        cfg.get("eos_token_id").and_then(|e| e.as_u64()).unwrap_or(248046) as u32
    };
    let eos = tok.stop_token_ids(cfg_eos);
    let mtp_k = crate::exl3_forward::mtp_depth_default();
    let width = o.width;
    let mut sc = FwdModel::scratch(model.dev(), &model.cfg, width.max(mtp_k + 1))?;
    let chunk = 2048usize;
    let psc0 = model.prefill_scratch(chunk)?;
    model.precapture_decode_graphs(&mut sc, width, mtp_k, true)?;
    let tok = Arc::new(tok);
    let mut sched = Exl3Scheduler {
        model: model.clone(), sc, width,
        lanes: (0..width).map(|_| None).collect(), eos: eos.clone(), chunk, psc: Some(psc0),
        mtp_k, ctrl_round: 0, ema_plain: 0.0, draft_conf: 0.4, cal: DraftCal::new(),
        prefix_on: o.prefix, cache: (0..width).map(|_| None).collect(),
        wp16: None, wp16_xcheck: false,
        loop_cfg: Some((300, 3)), pen_range: 1024, spec_ratio: true, draft_temp: None, gate: None, tok: tok.clone(),
        exit_on_fatal: false, fatal: None, inject_panic: None, steps: 0, last_resume: (0, 0),
        tp: if tp_on { Some(TpSync::new(rank)) } else { None },
    };
    let mut arms = vec![SpecArm::Plain];
    if o.dds { arms.push(SpecArm::Dds); }
    arms.extend(o.depths.iter().map(|&d| SpecArm::Fixed(d)));
    println!("[tpspec] {who}: arms {:?} x {} reps, gen {} greedy (thinking-off chat ids), served depth {mtp_k}, DDS 0.4, \
              prefix cache {}, loop detect (300, 3), precapture done ({} graphs)",
             arms.iter().map(|a| a.name()).collect::<Vec<_>>(), o.reps, o.gen, if o.prefix { "on" } else { "off" },
             model.graph_count());
    let fnv64 = |t: &[u32]| -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &x in t { h ^= x as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
        h
    };
    let text_sha = |t: &[u32]| -> String {
        let body: Vec<u32> = t.iter().copied().filter(|x| !eos.contains(x)).collect();
        let s = tok.decode(&body, true).unwrap_or_default();
        Sha256::digest(s.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect()
    };
    let t_all = std::time::Instant::now();
    let mut e_ok = true;
    let (mut n_cmp, mut n_bad) = (0usize, 0usize);
    let mut plain_ref: std::collections::BTreeMap<String, Vec<u32>> = Default::default();
    // per (class, arm): (tok/round, ms/round, tok/s) per rep; plain: ms/token
    let mut acc: std::collections::BTreeMap<(String, String), Vec<(f64, f64, f64)>> = Default::default();
    for rep in 0..o.reps {
        if let Some(t) = sched.tp.as_mut() { t.ident = o.ident && rep == 0; }
        for (name, ids) in &prompts {
            let mut rf: Option<Vec<u32>> = None;
            for &arm in &arms {
                let r = spec_arm(&mut sched, arm, ids, o.gen)?;
                let st = &r.st;
                let same = match arm {
                    SpecArm::Plain => {
                        let prev = plain_ref.entry(name.clone()).or_insert_with(|| r.toks.clone());
                        rf = Some(r.toks.clone());
                        *prev == r.toks
                    }
                    _ => rf.as_ref().map_or(false, |p| *p == r.toks),
                };
                n_cmp += 1;
                if !same { n_bad += 1; e_ok = false; }
                let div = rf.as_ref().and_then(|p| p.iter().zip(&r.toks).position(|(a, b)| a != b));
                let n = r.toks.len();
                let (tpr, mpr) = if st.rounds > 0 {
                    ((st.accepted + st.rounds) as f64 / st.rounds as f64, st.ms / st.rounds as f64)
                } else { (0.0, 0.0) };
                let tps = if r.secs > 0.0 { n as f64 / r.secs } else { 0.0 };
                let key = (name.clone(), arm.name());
                acc.entry(key).or_default().push(if arm == SpecArm::Plain { (0.0, r.secs * 1e3 / n.max(1) as f64, tps) } else { (tpr, mpr, tps) });
                if arm == SpecArm::Plain && rep == 0 && rank == 0 {
                    let body: Vec<u32> = r.toks.iter().copied().filter(|x| !eos.contains(x)).collect();
                    let t = tok.decode(&body, true).unwrap_or_default();
                    let (hd, tl): (String, String) = (t.chars().take(160).collect(), t.chars().rev().take(120).collect::<Vec<_>>().into_iter().rev().collect());
                    println!("TPSPEC-TEXT {who} {name}: head {hd:?} ... tail {tl:?}");
                }
                println!("TPSPEC {who} rep {rep} {name} {:<5} n={n} ({}) ids {:016x} text {} | {} | rounds {} drafted {} accepted {} \
                          tok/round {tpr:.3} ms/round {mpr:.2} ctrl {} plain {} | {tps:.1} tok/s ({:.1} s)",
                         arm.name(), r.reason, fnv64(&r.toks), text_sha(&r.toks),
                         if arm == SpecArm::Plain { if same { "plain (rep-stable)".to_string() } else { "PLAIN DIFFERS FROM REP 0".to_string() } }
                         else if same { "== plain".to_string() } else { format!("MISMATCH vs plain at {:?}", div) },
                         st.rounds, st.drafted, st.accepted, st.ctrl, st.plain, r.secs);
            }
        }
        if let Some(t) = sched.tp.as_ref() {
            println!("[tpspec] {who} rep {rep}: lockstep agrees {} ok (agree_ext tripwire armed; proxy watchdog 10 s){}",
                     t.agrees, if t.ident_bufs > 0 { format!("; G-T1-b graphed digests {} compared, {} differ", t.ident_bufs, t.ident_bad) } else { String::new() });
        }
    }
    println!("TPSPEC {who} SUMMARY (mean over {} reps; DDS/fixed: tok/round | ms/round | tok/s; plain: ms/token | tok/s):", o.reps);
    for ((name, arm), v) in &acc {
        let k = v.len() as f64;
        let m = |f: fn(&(f64, f64, f64)) -> f64| v.iter().map(f).sum::<f64>() / k;
        let mn = |f: fn(&(f64, f64, f64)) -> f64| v.iter().map(f).fold(f64::INFINITY, f64::min);
        let mx = |f: fn(&(f64, f64, f64)) -> f64| v.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
        if arm == "plain" {
            println!("  {name:<7} {arm:<5} ms/token {:.2} (min {:.2} max {:.2}) | {:.1} tok/s", m(|x| x.1), mn(|x| x.1), mx(|x| x.1), m(|x| x.2));
        } else {
            println!("  {name:<7} {arm:<5} tok/round {:.3} (min {:.3} max {:.3}) | ms/round {:.2} (min {:.2} max {:.2}) | {:.1} tok/s",
                     m(|x| x.0), mn(|x| x.0), mx(|x| x.0), m(|x| x.1), mn(|x| x.1), mx(|x| x.1), m(|x| x.2));
        }
    }
    let (ident_bufs, ident_bad, agrees) = sched.tp.as_ref().map_or((0, 0, 0), |t| (t.ident_bufs, t.ident_bad, t.agrees));
    let (sp_ok, sp_bad) = (crate::exl3_forward::xtp::SPINE_OK.load(std::sync::atomic::Ordering::Relaxed),
                           crate::exl3_forward::xtp::SPINE_BAD.load(std::sync::atomic::Ordering::Relaxed));
    if tp_on {
        println!("[tpspec] {who}: graphs captured {} ; TP-SPINE audit {sp_ok} graphs OK, {sp_bad} discarded (a discarded shape runs eager)",
                 model.graph_count());
    }
    println!("G-T1-e {who}: {} — {n_cmp} requests compared ({n_bad} differ) over {} prompt(s) x {} arm(s) x {} rep(s), gen {} ({:.1} min)",
             if e_ok { "LOSSLESS_OK (every spec arm == plain, byte-identical; plain rep-stable)" } else { "MISMATCH" },
             prompts.len(), arms.len(), o.reps, o.gen, t_all.elapsed().as_secs_f64() / 60.0);
    let mut samp_ok = true;
    if o.samp > 0 {
        let text = WP24_GATE_PROMPT;
        let p = tok.encode(text, false)?;
        sched.mtp_k = mtp_k;
        let ok = wp24_gate_conditions(&mut sched, &p, text, o.trials.max(50), 6, 2, 20, 0.4, None, o.samp == 1)?;
        samp_ok = ok;
        println!("G-T1-h {who}: {} (WP24 served gate on this scheduler, {}; trials {}/arm, width {width})",
                 if ok { "SERVED_DISTRIBUTION_OK" } else { "FAIL / NO_POWER" }, if o.samp == 1 { "--quick" } else { "full ladder" }, o.trials);
    }
    // (re-read after the sampled gate: its rounds are lockstep points too)
    let (ident_bufs, ident_bad, agrees) = sched.tp.as_ref().map_or((0, 0, 0), |t| (t.ident_bufs, t.ident_bad, t.agrees));
    let mut all_ok = e_ok && samp_ok && ident_bad == 0;
    if let Some(t) = sched.tp.as_ref() {
        let peer = crate::net::exchange_u32s(&[all_ok as u32, t.agrees as u32], 4)?;
        println!("[tpspec] {who}: this rank {} | peer {} (peer agrees {}, this rank {agrees}; graphed digests {ident_bufs} compared, {ident_bad} differ)",
                 if all_ok { "OK" } else { "FAIL" }, if peer[0] != 0 { "OK" } else { "FAIL" }, peer[1]);
        all_ok &= peer[0] != 0 && peer[1] as u64 == agrees;
    }
    println!("RESULT: {} ({who}; G-T1-e {}{}{})", if all_ok { "TPSPEC_OK" } else { "TPSPEC_FAIL" },
             if e_ok { "OK" } else { "MISMATCH" },
             if o.samp > 0 { if samp_ok { " ; G-T1-h OK" } else { " ; G-T1-h FAIL" } } else { "" },
             if tp_on { if ident_bad == 0 { " ; RANK_IDENTITY_OK (graphed)" } else { " ; RANK_IDENTITY_FAIL" } } else { "" });
    anyhow::ensure!(all_ok, "spec program gates failed ({who})");
    Ok(())
}
