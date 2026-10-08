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

/// S1 (REL_V0_7_3): delegates to the ONE shared `=`-aware parser (`crate::arg_value`) so
/// `--lane-order=fcfs` and `--lane-order fcfs` behave identically on EXL3 (the old body
/// matched only the space form and silently ignored the `=` form).
fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    crate::arg_value(args, name)
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
    // WP08: the reference implementation's streaming loop detector (None = --loop-detect off) + log context.
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
    // per-request round anatomy, logged at finish ([mtp-stats]): comparable to the reference implementation's
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
    // CF-FCFS: this request's admission order (the scheduler's monotonically increasing admit
    // counter at admit time). fcfs lane order is the ONLY thing that reads it. It is a pure
    // function of replicated state — every TP rank runs the same admits in the same head-driven
    // order, so every rank assigns the same value to the same request (AGENTS 2.10).
    admit_seq: u64,
    // CF-FCHS: tokens this lane has generated since it last became the fcfs front lane. Reset when
    // a lane rotates to the back; compared against --lane-quantum to bound one turn's monopoly.
    fcfs_credit: usize,
    tx: crate::server::TokTx,
}

struct Exl3Scheduler {
    model: Arc<FwdModel>,
    sc: Scratch,
    width: usize,
    lanes: Vec<Option<Lane>>,
    eos: Vec<u32>,
    chunk: usize, // S-A3-f-b: prefill chunk width (tokens per forward sweep)
    /// `--prefill-interleave N`: decode steps run for the live lanes after each non-final prefill chunk (0 = off)
    interleave: usize,
    psc: Option<crate::exl3_forward::PrefillScratch>,
    // S-A3-f-d MTP policy: depth k (--exl3-mtp-k, default MTP_MAX_K = 7 since WP23), auto-disable
    // when the measured per-token cost loses to plain (§6), control-round bookkeeping.
    mtp_k: usize,
    /// CF-P1e `--spec-lanes-max`: when the busy lanes share one plain batched step instead of serial speculation.
    spec_policy: SpecLanes,
    /// CF-FCFS `--lane-order` / `--lane-quantum`: the serial rounds' order, the fcfs quantum in
    /// generated tokens, the next admission's sequence number, and the first-16 rotation log count.
    lane_order: LaneOrder,
    lane_quantum: usize,
    admit_ctr: u64,
    fcfs_rots: usize,
    /// CF-P1e auto policy: learned scale of the shared-step time model, the last decision (hysteresis) and the
    /// consecutive shared-step count (stale-estimate refresh). All updated from lockstep-adopted values.
    shared_scale: f64,
    shared_last: bool,
    shared_streak: usize,
    policy_logs: usize,
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
    // WP08: stop_on_loop (window, min_reps) — the reference implementation's (300, 3) unless --loop-detect off.
    loop_cfg: Option<(usize, usize)>,
    // WP15: penalty window (sustain = decay = this; the reference implementation's penalty_range 1024).
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
    // WP02 liveness. exit_on_fatal = --exit-on-fatal (default ON since v0.7.3, owner decision
    // 2026-10-06): a sticky CUDA error exits 70 so a supervisor restarts the server; off, the
    // engine goes DEAD (/health and every request answer 503). `fatal` is set by admit/step when
    // the context is poisoned.
    exit_on_fatal: bool,
    fatal: Option<String>,
    inject_panic: Option<usize>, // --inject-panic-steps=<n>: panic at decode step n (liveness gate)
    steps: usize,
    // TP-C: Some = this scheduler is one rank of a TP group (SPMD lockstep, head-authoritative timing)
    tp: Option<TpSync>,
    // TP-D: the last admit's resume points (prompt-end reuse, WP16/C1 checkpoint q) — proven
    // identical on every rank by the seam lockstep
    last_resume: (usize, usize),
    // VIS-2: the text config's interleaved mrope_section (rope_parameters.mrope_section; None = the
    // model declares no interleaved mrope -> image requests are refused)
    mrope_section: Option<[usize; 3]>,
    // VIS-4: the last image admit's (image tokens, rope delta, FNV of the fp16 rows) — proven equal on
    // every rank by the seam lockstep (None = a text admit)
    vis_seam: Option<(u32, i32, u64)>,
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
        o += &format!(" | k2-prefetch hints armed {} used {}", crate::exl3_forward::xtp::PF_ARMED.load(std::sync::atomic::Ordering::Relaxed),
                      crate::exl3_forward::xtp::PF_USED.load(std::sync::atomic::Ordering::Relaxed));
        o
    }
}

pub(crate) struct TpSync {
    pub rank: i32,
    /// TP-4D: the TP world (2 = the pairwise lockstep of TP-C..TP-I, byte-for-byte; > 2 = the head-hub lockstep
    /// of `tp_lockstep`, ONE merged hub op per lockstep point).
    pub world: i32,
    /// TP-4D (audit C5): per-wait deadline of a world > 2 lockstep hub op, ms (0 = unbounded; always 0 at world 2,
    /// which keeps the pairwise channel and its own 10 s `net_agree` timeout).
    pub deadline_ms: u64,
    /// TP-4D: rounds that reported a device-epoch skew between ranks (diagnostic print throttle)
    epoch_warns: u64,
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
    pub(crate) fn new(rank: i32, world: i32) -> Self {
        let deadline_ms = if world > 2 { lockstep_deadline_ms() } else { 0 };
        TpSync { rank, world, deadline_ms, epoch_warns: 0, step: 0, ident: false, ident_bufs: 0, ident_bad: 0, agrees: 0,
                 pre_agrees: 0, ctl: Vec::new(), wp: WaitProf::default() }
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
    /// `--exl3-tp-fastctl=0` (diagnostic, rides TpConfig's CLI option-registry snapshot) = the TP-D TCP-every-step path.
    fn step_go(&mut self, step_no: u64, n_events: usize) -> Result<(u64, usize)> {
        if self.world > 2 {
            let mut x = crate::net::link_hub(self.deadline_ms)?;
            return self.step_go_hub(&mut x, step_no, n_events);
        }
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
        if self.world > 2 {
            let mut x = crate::net::link_hub(self.deadline_ms)?;
            return self.pre_verify_hub(&mut x, w, drafts, launched);
        }
        let step = agree_field(self.step + 1, AGREE_PV);
        let h = fnv32(0x5EED, drafts.iter().map(|&d| d as u32));
        let tag = 0x9E00_0000u32 | (step as u32 & 0xFFFF);
        let mine = [tag, w as u32, h, launched as u32];
        // S-B9-REL-TP2RACE (issue #10): the reporter's diagnostic — `device_epoch & 7` next to the
        // exchange generation's slot `gen & 7` — plus the deterministic fault-injection hook. With
        // --tp-race-probe <ms> the node's proxy holds every epoch release, so its last screen epoch
        // sits validated-but-unconsumed when we get here; aligning our next exchange generation onto
        // that epoch's recv slot (gen & 7 == device_epoch & 7) makes our frame land exactly on the
        // withheld payload. Both ranks run the identical alignment (same device epoch, same lockstep
        // gen), so the paired generations stay equal. On the dedicated-slot transport this changes
        // nothing (the frame never touches the doorbell rings); on the old hot-ring transport the
        // node screens a corrupted candidate set and this bail or the proxy's CLOBBER line fires.
        let e_last = crate::net::traced_device_epoch();
        let g0 = crate::net::traced_xchg_gen();
        let g = if crate::net::race_probe_hold_ms() > 0 {
            let g1 = crate::net::traced_gen_add(crate::net::probe_gen_delta(g0, e_last & 7));
            eprintln!("[tp-race-probe] rank {}: pre-verify aligned xchg gen {g0} -> {g1} onto epoch {e_last} \
                       (device_epoch&7 = {}, gen&7 = {}), hold {} ms on rank 1",
                      self.rank, e_last & 7, g1 & 7, crate::net::race_probe_hold_ms());
            g1
        } else { g0 };
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, 4 + 4)?; // + the 16-byte tail guard (as tp_round)
        self.wp.rec(3, t0, s0);
        if peer != mine {
            crate::net::abort_link();
            anyhow::bail!("TP pre-verify FAILED (round {}): this rank width {w} drafts {h:08x} head passes launched {launched}, \
                           peer {:?} — device_epoch&7 = {}, xchg gen&7 = {} — ranks chose different verify widths, drafts \
                           or speculative head-pass launches; link aborted before the verify",
                          self.step + 1, peer, e_last & 7, g & 7);
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

// ------------------------------------------------------------------------------------------------------
// TP-4D: the world > 2 lockstep — ONE merged head-hub op per lockstep point.
//
// World 2 keeps the TP-C..TP-I sequence above (fence A agree, u32 exchange, fence B agree). Those agrees and
// that exchange are PAIRWISE ops over the HOT-PATH ring memory (the exchange stages in send slot 0 and lands in
// recv slot gen & 7, the rings the device doorbell epochs use), which is the only reason fence A/B and
// `drain_sends` exist. At world > 2 every lockstep frame rides `net::hub_all` on the DEDICATED control slots
// (`net_exchange_one`: its own send staging slot, its own per-sender recv rings of depth TP_CTRL_RING, a stable
// copy per sender) — no byte of a hub frame shares memory with the hot path, so
//   * fence B has nothing to protect: after the op the next forward may start at once (its epochs cannot reach
//     the control slots), and the op's two rounds are already a barrier (a rank leaves round 2 only after the
//     head has heard from every rank);
//   * fence A's "both ranks finished the forward" is the op's own arrival condition (the frame is built from
//     the finished forward's host results);
//   * `drain_sends` (exl3_forward.rs, gated to world 2) has no aliasing left to drain.
// The claim is void if a hub frame ever shares memory with the hot path (the model in `net::hub_tests` assumes
// separate slots). Every rank ends up holding EVERY rank's frame, runs the same pure verdict
// (`tp_lockstep::*_verdict`) over the same data and therefore reaches the same verdict: all ranks abort together,
// none is left parked in a later exchange. Transport failure (dead peer, code 10 / deadline, code 12 / link
// abort) makes `hub_all` fail on the ranks that see it; the dead rank's block reads zero on the rest, which the
// verdict rejects (tags differ), so every live rank ends in an error (`net::hub_tests` kills / hangs a node).

/// The per-wait deadline of a world > 2 lockstep hub op: `--tp-lockstep-timeout-ms`, default 30 s, 0 = unbounded.
pub(crate) fn lockstep_deadline_ms() -> u64 {
    crate::opts::var(crate::opt!("tp-lockstep-timeout-ms")).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(30_000)
}

impl TpSync {
    fn hub_fail(&self, what: &str, why: &dyn std::fmt::Display) -> anyhow::Error {
        // keep an abort code a failed exchange already recorded (10 = peer dead, 12 = lockstep deadline)
        crate::net::abort_link_keep_code();
        anyhow::anyhow!("TP lockstep {what} FAILED at rank {} (world {}): {why}", self.rank, self.world)
    }

    /// world > 2 `step_go`: [tag, counter field, head step, head events] from every rank; the verdict proves all
    /// ranks are at the same lockstep counter and the nodes adopt the head's (step, events).
    pub(crate) fn step_go_hub<X: crate::net::HubXport>(&mut self, x: &mut X, step_no: u64, n_events: usize)
                                                       -> Result<(u64, usize)> {
        use crate::tp_lockstep as ls;
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        self.step += 1;
        let field = agree_field(self.step, AGREE_GO) as u32;
        let frame = ls::go_frame(self.rank == 0, field, step_no, n_events);
        let all = crate::net::hub_all(x, &frame, ls::GO_WIRE).map_err(|e| self.hub_fail("step-go", &format!("{e:#}")))?;
        self.wp.rec(7, t0, s0);
        let (hs, hn) = ls::go_verdict(&all, self.world as usize)
            .map_err(|why| self.hub_fail("step-go", &format!("fence FAILED at control step {step_no}: {why}")))?;
        // same adoption rule as the pairwise path: the head's low 32 bits under this rank's own high bits
        let hs = if self.rank == 0 { step_no } else { (step_no & !0xFFFF_FFFF) | hs as u64 };
        Ok((hs, hn as usize))
    }

    /// world > 2 `pre_verify`: every rank's (width, drafts hash, head passes launched) at the same lockstep
    /// counter — compared by EVERY rank, BEFORE the verify graph (whose width selects the barrier sequence) launches.
    pub(crate) fn pre_verify_hub<X: crate::net::HubXport>(&mut self, x: &mut X, w: usize, drafts: &[i32], launched: usize)
                                                          -> Result<()> {
        use crate::tp_lockstep as ls;
        let field = agree_field(self.step + 1, AGREE_PV) as u32;
        let h = fnv32(0x5EED, drafts.iter().map(|&d| d as u32));
        let frame = ls::pv_frame(field, w, h, launched);
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let all = crate::net::hub_all(x, &frame, ls::PV_WIRE).map_err(|e| self.hub_fail("pre-verify", &format!("{e:#}")))?;
        self.wp.rec(3, t0, s0);
        ls::pv_verdict(&all, self.world as usize).map_err(|why| {
            self.hub_fail("pre-verify", &format!("round {}: this rank width {w} drafts {h:08x} head passes launched \
                {launched}; {why} — ranks chose different verify widths, drafts or speculative head-pass launches; \
                link aborted before the verify", self.step + 1))
        })?;
        self.pre_agrees += 1;
        Ok(())
    }

    /// world > 2 `tp_round`: the merged lockstep op (tripwire words + the head's timing + a device-epoch probe) and,
    /// with `ident` on, a second op with the digests of `digests(m)` compared against the head's. `digests` runs
    /// only after the verdict proved every rank chose the same width `m`.
    pub(crate) fn round_hub<X: crate::net::HubXport>(&mut self, x: &mut X, what: &str, a: usize, m: usize, words: &[u32],
                                                     ms: f64, plain_ms: f64, captured: bool, epoch: u64,
                                                     digests: impl FnOnce(usize) -> Result<Vec<u64>>)
                                                     -> Result<(f64, f64, bool)> {
        use crate::tp_lockstep as ls;
        self.step += 1;
        let (step, world) = (self.step, self.world as usize);
        let h = fnv32(0, words.iter().copied());
        let field = agree_field(step, AGREE_A) as u32;
        let frame = ls::rnd_frame(&ls::RoundIn { step, field, accept: a.min(255), width: m.min(15), hash: h, epoch, ms,
                                                 plain_ms, captured });
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let all = crate::net::hub_all(x, &frame, ls::RND_WIRE)
            .map_err(|e| self.hub_fail("round", &format!("at {what} (lockstep step {step}): {e:#}")))?;
        self.wp.rec(1, t0, s0);
        self.wp.rounds += 1;
        let out = ls::rnd_verdict(&all, world).map_err(|why| {
            self.hub_fail("round", &format!("agree() FAILED at {what} (lockstep step {step}, accept {a}, width {m}, \
                hash {h:08x}): {why} — ranks diverged or the link aborted"))
        })?;
        if self.ident {
            let d = digests(m).map_err(|e| self.hub_fail("ident", &format!("digest failed at {what}: {e:#}")))?;
            let frame = ls::id_frame(step, &d);
            let all = crate::net::hub_all(x, &frame, ls::id_wire(d.len()))
                .map_err(|e| self.hub_fail("ident", &format!("at {what} (lockstep step {step}): {e:#}")))?;
            let bad = ls::id_verdict(&all, world, d.len())
                .map_err(|why| self.hub_fail("ident", &format!("step sync at {what}: {why}")))?;
            self.ident_bufs += d.len() as u64;
            self.ident_bad += bad as u64;
            if bad > 0 && self.ident_bad as usize == bad {
                eprintln!("[tp-ident] FIRST MISMATCH at {what} (lockstep step {step}, width {m}): {bad} of {} digests differ \
                           (logits rows 0..{m}, resid, taps; some rank differs from the head)", d.len());
            }
        }
        self.agrees += 1;
        if !out.epoch_skew.is_empty() {
            self.epoch_warns += 1;
            if self.epoch_warns <= 5 || self.epoch_warns % 1000 == 0 {
                eprintln!("[tp-round] EPOCH-PROBE at {what} (lockstep step {step}, #{}): device epoch differs from the head's on \
                           (rank, delta) {:?}", self.epoch_warns, out.epoch_skew);
            }
        }
        Ok((out.ms, out.plain_ms, out.captured))
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
                eprintln!("[exl3-serve] FATAL: scheduler thread panicked — exit 70 (--exit-on-fatal, default)");
                std::thread::sleep(std::time::Duration::from_millis(300));
                crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7
            }
        }
    }
}

/// Port of exllamav3's DraftConfidenceCalibrator (vcruz305 fork 523ecd3,
/// exllamav3/generator/draft_confidence.py): an online map from the draft head's argmax LOGIT
/// to the observed acceptance rate, in 1-logit bins of exponentially decayed (tested, accepted)
/// counts. One calibrator for the whole server, as in the reference implementation (one per generator).
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
/// --tp-dds-prior rides TpConfig's CLI option-registry snapshot; the value joins the TP boot agree hash. The guard's
/// timing inputs stay the HEAD's (tp_round), so both ranks' targets are identical.
pub(crate) const TP2_DDS_PRIOR: f64 = 2.7;
/// World-4 default: an ESTIMATE, not a measurement — TP-4A (24.7-17.59)/5 = 1.42 ms/row + 0.73 draft = 2.15; see TP-4F_PREP_REPORT.md.
pub(crate) const TP4_DDS_PRIOR: f64 = 2.1;
/// CLI-1 (--print-config): the TP=2 prior this process resolves.
pub fn dds_prior_tp() -> f64 { dds_prior(true) }
/// CLI-1 (--print-config): the TP=4 prior this process resolves.
pub fn dds_prior_tp4() -> f64 { dds_prior_w(4) }
pub(crate) fn dds_prior(tp: bool) -> f64 { dds_prior_w(if tp { 2 } else { 1 }) }
/// The prior for a scheduler of `world` ranks (1 = no TP): 6.0 / 2.7 / 2.1 at world 1 / 2 / 4 (any other world
/// keeps the TP=2 value); `--tp-dds-prior` (off = 6.0, a number = that prior) applies at every TP world.
pub(crate) fn dds_prior_w(world: usize) -> f64 {
    if world < 2 { return CostFit::PRIOR_SLOPE; }
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    resolve_dds_prior(RAW.get_or_init(|| crate::opts::var(crate::opt!("tp-dds-prior")).ok()).as_deref(), world)
}
/// Pure resolution of the prior from the raw `--tp-dds-prior` value (None = unset) at a TP `world` >= 2. The
/// world-2 arms are the pre-TP-4F code verbatim: on/unset/garbage = 2.7, off = 6.0, a positive number = itself.
pub(crate) fn resolve_dds_prior(raw: Option<&str>, world: usize) -> f64 {
    let dflt = if world == 4 { TP4_DDS_PRIOR } else { TP2_DDS_PRIOR };
    match raw.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if v == "0" || v == "off" || v == "false" => CostFit::PRIOR_SLOPE,
        Some(v) if v == "1" || v == "on" || v.is_empty() => dflt,
        Some(v) => v.parse::<f64>().ok().filter(|x| *x > 0.0).unwrap_or(dflt),
        None => dflt,
    }
}
/// The prior's word in the TP boot agree hash (`tp_boot_agree`).
pub(crate) fn boot_prior_word(world: usize) -> u32 { (dds_prior_w(world) as f32).to_bits() }

impl Exl3Scheduler {
    /// Ranks of this scheduler's TP group for the world-aware DDS prior: 1 = no TP (an impossible tp-without-model
    /// pairing keeps the pre-TP-4F value, world 2).
    fn tp_world(&self) -> usize {
        if self.tp.is_none() { 1 } else { self.model.tp_rank().map_or(2, |(_, w)| w as usize) }
    }
}

#[cfg(test)]
mod dds_prior_w4_tests {
    use super::*;

    /// The pre-TP-4F `dds_prior(true)` body, verbatim (only the option read is replaced by `raw`): the frozen
    /// reference that world 2 must equal bit for bit.
    fn old_world2(raw: Option<&str>) -> f64 {
        match raw.map(|v| v.trim().to_ascii_lowercase()) {
            Some(v) if v == "0" || v == "off" || v == "false" => CostFit::PRIOR_SLOPE,
            Some(v) if v == "1" || v == "on" || v.is_empty() => TP2_DDS_PRIOR,
            Some(v) => v.parse::<f64>().ok().filter(|x| *x > 0.0).unwrap_or(TP2_DDS_PRIOR),
            None => TP2_DDS_PRIOR,
        }
    }

    const RAWS: [Option<&str>; 16] = [None, Some(""), Some(" "), Some("on"), Some("ON"), Some("1"), Some("off"), Some("OFF"),
        Some("0"), Some("false"), Some("2.7"), Some("3.5"), Some("0.0"), Some("-1"), Some("garbage"), Some(" 1.25 ")];

    #[test]
    fn world_1_and_2_are_bit_identical_to_the_pre_world_aware_prior() {
        assert_eq!(dds_prior(false).to_bits(), CostFit::PRIOR_SLOPE.to_bits());
        assert_eq!(dds_prior_w(1).to_bits(), 6.0f64.to_bits());
        assert_eq!(dds_prior_w(0).to_bits(), 6.0f64.to_bits());
        for raw in RAWS {
            assert_eq!(resolve_dds_prior(raw, 2).to_bits(), old_world2(raw).to_bits(), "raw {raw:?}");
        }
        // unset (the test process has no --tp-dds-prior): the runtime entry points are the old constants
        assert_eq!(dds_prior(true).to_bits(), 2.7f64.to_bits());
        assert_eq!(dds_prior_tp().to_bits(), 2.7f64.to_bits());
        assert_eq!(dds_prior_w(2).to_bits(), 2.7f64.to_bits());
    }

    #[test]
    fn world_2_boot_hash_word_and_hash_are_unchanged() {
        let sample = [128u32, 2048, 7, 1, 512, 0, 262144, 1, 0, 0x3F00_0000, 0, 0, 1, 2, 3];
        let hash = |word: u32| fnv32(0xB007, sample.iter().copied().chain([word]).chain([9, 9, 9]));
        assert_eq!(boot_prior_word(2), (2.7f64 as f32).to_bits(), "the world-2 word");
        assert_eq!(boot_prior_word(2), (dds_prior(true) as f32).to_bits(), "== the pre-change `(dds_prior(true) as f32).to_bits()`");
        assert_eq!(hash(boot_prior_word(2)), hash((dds_prior(true) as f32).to_bits()));
        // every world-2 value the flag can produce hashes exactly as the old code did
        for raw in RAWS {
            assert_eq!(hash((resolve_dds_prior(raw, 2) as f32).to_bits()), hash((old_world2(raw) as f32).to_bits()), "raw {raw:?}");
        }
        // world 4's word differs (its prior is 2.1): ranks must agree among themselves, and do, both resolve the same flag
        assert_eq!(boot_prior_word(4), (2.1f64 as f32).to_bits());
        assert_ne!(hash(boot_prior_word(4)), hash(boot_prior_word(2)));
    }

    #[test]
    fn world_4_default_is_the_derived_value_and_an_override_still_overrides() {
        assert_eq!(TP4_DDS_PRIOR, 2.1);
        assert_eq!(dds_prior_w(4).to_bits(), TP4_DDS_PRIOR.to_bits());
        assert_eq!(dds_prior_tp4().to_bits(), TP4_DDS_PRIOR.to_bits());
        // unset / on / unusable = the world's default (the TP=2 arms' "unusable falls back to default" rule)
        for raw in [None, Some(""), Some("on"), Some("1"), Some("garbage"), Some("-1"), Some("0.0")] {
            assert_eq!(resolve_dds_prior(raw, 4), TP4_DDS_PRIOR, "raw {raw:?}");
        }
        for raw in [Some("off"), Some("0"), Some("false")] { assert_eq!(resolve_dds_prior(raw, 4), 6.0, "raw {raw:?}"); }
        assert_eq!(resolve_dds_prior(Some("2.7"), 4), 2.7, "a numeric override overrides at W=4 (even to the W=2 value)");
        assert_eq!(resolve_dds_prior(Some("3.5"), 4), 3.5);
        assert_eq!(resolve_dds_prior(Some(" 1.25 "), 4), 1.25);
        // the override is world-independent: the same number at world 2 and 4
        for n in ["0.5", "1.9", "4.4"] { assert_eq!(resolve_dds_prior(Some(n), 2), resolve_dds_prior(Some(n), 4)); }
        // any other TP world keeps the TP=2 constant
        assert_eq!(resolve_dds_prior(None, 3), 2.7);
    }
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

/// `--spec-lanes-max`: never / a fixed lane threshold / cost-based (default).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SpecLanes { Never, Over(usize), Auto }

/// CF-FCFS `--lane-order`: the order the SERIAL speculative rounds run in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LaneOrder {
    /// Today's order, unchanged: every scheduler step gives each busy MTP-capable lane one round,
    /// in slot order. This arm must stay bit-for-bit identical to the pre-flag build.
    Rr,
    /// Run to completion, first come first served: only the front lane (the oldest busy
    /// MTP-capable lane that still has quantum left) steps; it does so until it finishes or
    /// exhausts its quantum, then it rotates to the back.
    Fcfs,
}

impl LaneOrder {
    pub(crate) fn parse(v: Option<&str>) -> Result<LaneOrder> {
        match v {
            None | Some("") | Some("rr") | Some("round-robin") => Ok(LaneOrder::Rr),
            Some("fcfs") => Ok(LaneOrder::Fcfs),
            Some(other) => anyhow::bail!("--lane-order: expected rr|fcfs, got {other:?}"),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self { LaneOrder::Rr => "rr", LaneOrder::Fcfs => "fcfs" }
    }
}

/// CF-FCFS: the one lane that gets this scheduler step's speculative round, or None when this step
/// runs no speculative round at all.
///
/// PURE FUNCTION of replicated state (AGENTS 2.10, TP SPMD): it reads only `mtp_capable`,
/// `admit_seq` and `fcfs_credit` per lane — no wall clock, no rank-local counter, nothing the
/// head alone sees. Every rank admits the same requests in the same order (the head ships the
/// admits), so every rank's `admit_seq` and token counters are identical and every rank picks the
/// same lane. The rr arm returns None here and the caller keeps today's loop verbatim.
///
/// `lanes` is the busy-lane list in slot order (the caller's `slots`), already split into the
/// lanes this round would speculate (`capable`, MTP on and lane.mtp armed) and the MTP-off lanes
/// that batch separately. Rotation accounting (`fcfs_credit`, the rotation log) is the caller's.
fn fcfs_front<'a>(capable: &[(usize, u64, usize)], quantum: usize) -> Option<usize> {
    // Oldest first (admit_seq is unique per live request: the counter only ever increments).
    let mut order: Vec<&(usize, u64, usize)> = capable.iter().collect();
    order.sort_by_key(|(_, seq, _)| *seq);
    // A lane that is not the front yet starts with a full quantum; only a lane that already had a
    // turn can be over its quantum (its credit is reset when it rotated to the back).
    order.into_iter().find(|(_, _, credit)| *credit < quantum).map(|(slot, _, _)| *slot)
}

/// CF-FCFSB: charge the front lane for the tokens it just generated and rotate it to the BACK of
/// the queue when it used its quantum. PURE function of replicated state: on rotation the lane is
/// re-stamped with the next admission sequence number (`*admit_ctr`, then the counter advances) so
/// `fcfs_front` — which orders by `admit_seq` — now sees it as the youngest lane. Without the
/// re-stamp the rotated lane keeps the lowest `admit_seq` and is picked again on the very next
/// step: the "rotation" was a log line and fcfs degenerated to run-to-completion by admission
/// order. Returns true when the lane rotated. `rr` never calls this.
fn fcfs_charge(seq: &mut u64, credit: &mut usize, delta: usize, quantum: usize, admit_ctr: &mut u64) -> bool {
    *credit += delta;
    if *credit >= quantum {
        *credit = 0;
        *seq = *admit_ctr;
        *admit_ctr += 1;
        true
    } else {
        false
    }
}

impl SpecLanes {
    pub(crate) fn parse(v: Option<&str>) -> Result<SpecLanes> {
        match v.map(str::trim) {
            None | Some("auto") | Some("") => Ok(SpecLanes::Auto),
            Some("0") | Some("never") | Some("off") => Ok(SpecLanes::Never),
            Some(n) => n.parse::<usize>().map(SpecLanes::Over)
                .map_err(|_| anyhow::anyhow!("--spec-lanes-max must be auto, 0 or a lane count (got '{n}')")),
        }
    }
}

/// Model of one shared plain step over n lanes (ms, TP=1 code class, fitted on .14 2026-10-02: n=4 46.5, n=8 61.7,
/// n=16 90): the learned `shared_scale` corrects it for the topology / content actually served.
fn shared_step_prior_ms(n: usize) -> f64 { 31.5 + 3.6 * n as f64 }

impl Lane {
    fn log_stats(&self, slot: usize, reason: &str) {
        if self.st_rounds == 0 && self.st_plain == 0 { return; }
        let r = self.st_rounds.max(1) as f64;
        crate::tel::add_spec(self.st_rounds as u64, self.st_drafted as u64, self.st_accepted as u64);
        self.log_wp23(slot);
        crate::reprintln!("[mtp-stats] slot={slot} finish={reason} gen={} rounds={} drafted={} accepted={} ({:.1}%) \
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
            crate::reprintln!("[mtp-stats] slot={slot} finish=cancelled gen={} rounds=0 plain_steps=0 temp={}",
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
        crate::reprintln!("[dds] slot={slot} widths={:?} c_hat={:.2} ms/draft (fit rounds {}) target={:.3}",
                  g.widths, g.fit.slope(), g.fit.n, g.target);
    }

    /// Sampler params for `n` consecutive rows starting at the lane's counter (None = greedy).
    fn samp_rows(&self, n: usize) -> Vec<Option<crate::exl3_forward::SampParams>> {
        (0..n).map(|i| crate::exl3_forward::SampParams::sampled(
            self.temperature, self.top_p, self.top_k, self.min_p, self.seed,
            self.ctr.wrapping_add(i as u32))).collect()
    }

    /// WP08: feed one emitted token (one that did not already end the request) to the loop
    /// detector — the reference implementation's job.py:1010-1013 order. True = end the response ("loop_detected").
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
        crate::reprintln!("[loop] detected slot={slot} period={} (detector {}) gen={} pos={} seed={} temp={} \
                   top_p={} top_k={} min_p={} pen={:?} mtp={} prompt_hash={:016x} tail={:?}",
                  d.fundamental_period().map_or("-".into(), |p| p.to_string()),
                  d.period().map_or("-".into(), |p| p.to_string()),
                  self.generated, self.pos, self.seed, self.temperature, self.top_p, self.top_k,
                  self.min_p, self.pen, self.mtp.is_some(), self.prompt_hash, tail);
    }
}

/// C1 v2: prefill scratch rows — the merged-chunk cap with tail checkpoints on (2C - 1; under TP
/// min(2C - 1, 4,095) so `--prefill-chunk 4095` fits the all-reduce partial buffer, TP-4X1), else C.
fn psc_rows(c: usize, tp: bool) -> usize {
    if crate::exl3_forward::tail_ckpt_on() && !crate::exl3_forward::wp16_off() {
        crate::exl3_forward::merge_cap(c, tp).max(c)
    } else { c }
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
    /// `key`: the prompt as the prefix cache / WP16 checkpoints see it (VIS-2: image rows carry
    /// pixel-hash ids so two different same-size images never share a cached prefix); == `prompt`
    /// for text.
    fn prefill_range(&mut self, slot: usize, prompt: &[u32], key: &[u32], from: usize, to: usize,
                     tx: Option<&crate::server::TokTx>, aligned: bool,
                     tail: Option<usize>, realign: bool) -> Result<bool> {
        let tp = std::time::Instant::now();
        let c = self.chunk.max(1);
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_rows(c, self.tp.is_some()))?);
        }
        let mut pos = from;
        let n = to.min(prompt.len().saturating_sub(1)); // the seam token is a decode step
        if pos >= n { return Ok(true); }
        let mut taken: Vec<usize> = Vec::new();
        // C1 v2 (knob on == realign): merge away the chunk the split / realign would add (psc holds
        // 2C - 1 rows then, see run())
        let absorb = crate::exl3_forward::absorb_rmin();
        let grid = crate::exl3_forward::plan_grid(from, n, c, realign, tail, psc_rows(c, self.tp.is_some()), absorb, aligned);
        let nchunks = grid.len();
        if absorb.is_some() || c > 2048 {
            // TP-4X1 receipts: the grid actually run (only with the new options on; default logs unchanged)
            let mut rows = Vec::with_capacity(grid.len());
            let mut s = from;
            for &e in &grid { rows.push(e - s); s = e; }
            let shown = if rows.len() > 16 {
                format!("{:?} .. {:?}", &rows[..4], &rows[rows.len() - 6..])
            } else { format!("{rows:?}") };
            crate::rprintln!("[exl3-serve] prefill grid slot={slot} n={} (pos {from}..{n}) C={c}: {nchunks} chunk(s), rows {shown}{}",
                     n - from, if absorb.is_some() { " (absorb-tail)" } else { "" });
        }
        let (mut il_chunks, mut il_steps) = (0usize, 0usize);
        for end in grid {
            let psc = self.psc.as_mut().unwrap();
            if pos > from {
                let mut gone = tx.map_or(false, |t| t.is_closed());
                // TP-D: the HEAD decides (its client); every rank stops at this same chunk boundary
                if let (Some(t), true) = (self.tp.as_mut(), tx.is_some()) {
                    gone = t.head_flag(gone)?;
                }
                if gone {
                    crate::rprintln!("[exl3-serve] prefill slot={slot} cancelled at pos {pos} of {n} (client gone)");
                    return Ok(false);
                }
            }
            let toks: Vec<i32> = prompt[pos..end].iter().map(|&t| t as i32).collect();
            self.model.prefill_chunk(psc, &toks, pos, slot, false, None)?;
            crate::metrics::sched_touch(); // H3: prefill chunk COMPLETE (long prefill stays live)
            if aligned {
                if let Some(st) = self.wp16.as_mut() {
                    st.extend(slot, &key[pos..end]);
                    if crate::exl3_forward::wp16_due(end, n, c) || tail == Some(end) {
                        let model = &self.model;
                        // TP-4D (audit C2): at world > 2 the allocation outcome is agreed across ranks (a rank-local
                        // failure would desynchronise the checkpoint stores); world <= 2: the plain local alloc.
                        let n_before = st.n_alloc;
                        let mut tp_alloc = self.tp.as_mut().filter(|t| t.world > 2);
                        let mut hub_err: Option<anyhow::Error> = None;
                        let got = st.acquire(|| {
                            let r = model.ckpt_alloc();
                            match tp_alloc.as_mut() { Some(t) => t.wp16_alloc(n_before, r, &mut hub_err), None => r }
                        });
                        if let Some(e) = hub_err { return Err(e); }
                        if let Some(mut b) = got {
                            match model.ckpt_save(slot, psc, &mut b) {
                                Ok(()) => { st.insert(slot, end, b); taken.push(end); }
                                Err(e) => { st.release(b); return Err(e); }
                            }
                        }
                    }
                }
            }
            pos = end;
            // --prefill-interleave: let the already-decoding lanes advance between chunks. lanes[slot] is
            // None while it prefills and step() never touches self.psc, so the two work on disjoint state.
            if end < n && self.interleave > 0 && self.tp.is_none() && tx.is_some() {
                il_chunks += 1;
                for _ in 0..self.interleave {
                    if self.fatal.is_some() || self.lanes.iter().all(|l| l.is_none()) { break; }
                    if let Err(e) = self.step() {
                        self.fail_step(&e);
                        return Err(e);
                    }
                    il_steps += 1;
                }
            }
        }
        if il_chunks > 0 {
            println!("[exl3-serve] prefill-interleave slot={slot} chunks={il_chunks} decode_steps={il_steps}");
        }
        // S-A3-f-d Item 2: serve-path prefill telemetry (TTFT's main term).
        let ms = tp.elapsed().as_secs_f64() * 1e3;
        let done = n - from;
        crate::rprintln!("[exl3-serve] prefill slot={} n={} (pos {}..{}) in {:.1} ms ({:.0} tok/s, C={})",
                 slot, done, from, n, ms, if ms > 0.0 { done as f64 * 1e3 / ms } else { 0.0 }, c);
        if let Some(t) = tail {
            crate::rprintln!("[exl3-serve] C1 tail slot={slot}: grid split + checkpoint at {t} (message boundary - {}; {nchunks} chunk(s){})",
                     crate::exl3_forward::TAIL_BACKOFF,
                     if realign && from % c != 0 { ", realigned to the C grid" } else { "" });
        } else if realign && from % c != 0 {
            crate::rprintln!("[exl3-serve] C1 slot={slot}: run from {from} realigned to the C grid ({nchunks} chunk(s))");
        }
        if !taken.is_empty() {
            if let Some(st) = self.wp16.as_ref() {
                let per = self.model.ckpt_layout().bytes() as f64;
                crate::rprintln!("[exl3-serve] WP16 slot={slot}: checkpoints taken at {taken:?}; slot holds {:?}; \
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
    fn prefill_cached(&mut self, slot: usize, prompt: &[u32], key: &[u32], ckpt_at: Option<usize>,
                      tx: &crate::server::TokTx) -> Result<bool> {
        anyhow::ensure!(key.len() == prompt.len(), "prefix key length {} != prompt length {}", key.len(), prompt.len());
        let plen = prompt.len();
        let end = plen.saturating_sub(1);
        self.last_resume = (0, 0);
        if !self.prefix_on || !self.model.prefix_snapshots() {
            // PFX1 (b): the chunked prefill below head-fills rows 0..plen-2 (no head-KV memset)
            self.model.reset_slot_for_prefill(slot)?;
            self.cache[slot] = None;
            return self.prefill_range(slot, prompt, key, 0, end, Some(tx), false, None, false);
        }
        let tail_on = self.wp16.is_some() && crate::exl3_forward::tail_ckpt_on();
        let reuse = match &self.cache[slot] {
            Some(t) if t.len() <= end && key[..t.len()] == t[..] => t.len(),
            _ => 0,
        };
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_rows(self.chunk.max(1), self.tp.is_some()))?);
        }
        let c = self.chunk.max(1);
        // WP16: only on a prompt-end miss (a hit is always the deeper, unchanged resume).
        let (q, lcp) = match (&self.wp16, reuse) {
            (Some(st), 0) => st.resume_point(slot, key, end, c),
            _ => (0, 0),
        };
        let mut tracked = reuse == 0 && self.wp16.is_some();
        if reuse > 0 {
            self.model.restore_slot(slot, self.psc.as_mut().unwrap())?;
            crate::rprintln!("[exl3-serve] prefix-cache hit slot={slot}: reuse {reuse} of {plen} prompt tokens");
            // an unaligned grid from `reuse`: no checkpoint is taken in it, none above it survives.
            // C1: with tail checkpoints the run is tracked instead (realigned grid, tail split) when
            // the slot's checkpointed prefix is exactly prompt[..reuse].
            if let Some(st) = self.wp16.as_mut() {
                if tail_on && st.prefix_ok(slot, key, reuse, c) {
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
            crate::rprintln!("[exl3-serve] WP16 checkpoint resume slot={slot}: resume at {q} of {plen} prompt tokens \
                      (LCP {lcp}, C={c}; {dropped} checkpoint(s) above dropped) — prefill {q}..{end}");
        } else {
            self.model.reset_slot_for_prefill(slot)?; // PFX1 (b): prefill_range from 0 head-fills
            if let Some(st) = self.wp16.as_mut() { st.begin_aligned(slot, 0, c); }
        }
        self.cache[slot] = None;
        self.last_resume = (reuse, q);
        let from = reuse.max(q);
        // v0.7.3 (#8.2): report the cache-served prefix (prefill skipped) to the request
        // handler ONCE, before the first token: usage.prompt_tokens_details.cached_tokens
        // and the /metrics counter. from == 0 (fresh prefill) stays silent — the
        // handler's default is 0, so engines without a prefix cache report 0.
        if from > 0 {
            let _ = tx.send(TokEvent::Admitted { cached_tokens: from as u32 });
        }
        let tail = if tail_on && tracked { crate::exl3_forward::tail_point(ckpt_at, from, end, c, crate::exl3_forward::tail_ckpt_extra_chunk()) } else { None };
        match self.prefill_range(slot, prompt, key, from, end, Some(tx), tracked, tail, tail_on && tracked) {
            Ok(true) => {}
            // WP02: cancelled mid-prefill — no snapshot, cache stays None. WP16: the checkpoints
            // taken so far stay (they describe prompt[..pos], rows this run did write).
            Ok(false) => return Ok(false),
            Err(e) => {
                if let Some(st) = self.wp16.as_mut() { st.clear_slot(slot); } // slot state unknown
                return Err(e);
            }
        }
        if q > 0 && self.wp16_xcheck && key == prompt { // VIS-2: the xcheck's scratch prefill is text-only
            self.wp16_xcheck(slot, prompt, q)?;
        }
        self.model.snapshot_slot(slot, self.psc.as_ref().unwrap())?;
        self.cache[slot] = Some(key[..end].to_vec());
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
        self.prefill_range(x, prompt, prompt, 0, end, None, false, None, false)?;
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
        if t.world > 2 {
            // TP-4D: ONE merged head-hub op (tp_lockstep::RND) instead of agree / exchange / agree
            let mut x = crate::net::link_hub(t.deadline_ms)?;
            let epoch = crate::net::traced_device_epoch();
            let (model, sc) = (&self.model, &self.sc);
            return t.round_hub(&mut x, what, a, m, words, ms, plain_ms, captured, epoch, |m| model.tp_ident_digests(sc, m));
        }
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
        crate::metrics::sched_step(); // v0.7.3 gauge: rounds + busy
        // WP02 cancel sweep (the NVFP4 decode_step sweep, batch.rs): a lane whose client is gone
        // (disconnect, or the HTTP side dropped the receiver on a stop-string hit) frees its slot
        // NOW instead of decoding to EOS/max_new while every later request waits behind it.
        // TP-D: under TP the HEAD sweeps (run_tp_head) and ships each cancel as a Step event — a
        // rank-local sweep would free a lane on one rank only.
        for s in 0..self.lanes.len() {
            if self.tp.is_none() {
                if let Some(l) = self.lanes[s].as_ref() {
                    let closed = l.tx.is_closed();
                    let over = !closed && crate::server::stream_backlog_exceeded(l.tx.len()); // LR-5
                    if closed || over {
                        let lane = self.lanes[s].take().unwrap();
                        if over {
                            crate::metrics::stream_backlog_cancel();
                            crate::reprintln!("[exl3-serve] slot {s}: stream cancelled — unconsumed event backlog at the --stream-backlog-events limit (reader stalled; LR-5)");
                        }
                        lane.log_cancel(s);
                    }
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
        // CF-P1e load-adaptive batching: past `--spec-lanes-max` busy lanes, serial speculative rounds stop
        // scaling (one lane per round), so every busy lane shares one batched plain step instead; the
        // lanes' MTP heads stay in sync (see step_plain's `sync_head`), so speculation resumes when load drops.
        if mtp_on && self.use_shared_plain(&slots) {
            self.shared_streak += 1;
            // Stale-estimate refresh: lanes' speculative-speed EMAs freeze while they ride shared steps, so every
            // 128th shared round one lane (rotating) runs a serial speculative round instead; the others skip
            // this iteration. ~0.5% of rounds; keeps the cost-based choice tracking the content.
            if self.spec_policy == SpecLanes::Auto && self.shared_streak % 128 == 0 {
                let s = slots[(self.shared_streak / 128) % slots.len()];
                if self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some()) {
                    return self.step_mtp(s);
                }
            }
            return self.step_plain(&slots, true);
        }
        self.shared_streak = 0;
        // CF-FCFS: the fcfs arm steps ONE lane (the front) and leaves the rest untouched this
        // scheduler step; the MTP-off lanes still batch together exactly as below. rr (the default)
        // runs the original loop, unchanged and in slot order.
        if self.lane_order == LaneOrder::Fcfs {
            let mut plain: Vec<usize> = Vec::new();
            let mut capable: Vec<(usize, u64, usize)> = Vec::new();
            for &s in &slots {
                let lane = self.lanes[s].as_ref().unwrap();
                if mtp_on && lane.mtp.is_some() {
                    capable.push((s, lane.admit_seq, lane.fcfs_credit));
                } else {
                    plain.push(s);
                }
            }
            let front = fcfs_front(&capable, self.lane_quantum);
            match front {
                Some(s) => {
                    let before = self.lanes[s].as_ref().unwrap().generated;
                    self.step_mtp(s)?;
                    // step_mtp may have FINISHED this lane (stop / length / cancel / a dead client):
                    // it frees the slot (self.lanes[s] = None) before returning. The front lane is
                    // then simply gone — its turn ended early and the next step picks the next-oldest
                    // — so every post-step access must tolerate the slot being empty.
                    let lane = match self.lanes[s].as_mut() {
                        Some(lane) => lane,
                        None => {
                            if !plain.is_empty() {
                                self.step_plain(&plain, false)?;
                            }
                            return Ok(());
                        }
                    };
                    let delta = lane.generated - before;
                    let (mut seq, mut credit) = (lane.admit_seq, lane.fcfs_credit);
                    // fcfs_charge works on the replicated (seq, credit) tuple; BOTH fields are
                    // written back every step (the credit accumulation lives inside charge —
                    // forgetting this write-back on the no-rotate path pins credit at 0 and the
                    // front lane never reaches the quantum: run-to-completion again).
                    let rotated = fcfs_charge(&mut seq, &mut credit, delta, self.lane_quantum, &mut self.admit_ctr);
                    lane.fcfs_credit = credit;
                    if rotated {
                        lane.admit_seq = seq; // re-stamped: the queue position IS the sequence number
                        self.fcfs_rots += 1; // TOTAL rotations; the first 16 are logged and the total is printed at a clean idle point
                        if self.fcfs_rots <= 16 {
                            crate::reprintln!("[exl3-serve] lane-order fcfs: slot {s} used its quantum ({} tokens) \
                                       and rotates to the back (rotation {}/16 logged)", self.lane_quantum, self.fcfs_rots);
                        }
                    }
                }
                None => {
                    // Unreachable with the charge-at-quantum bookkeeping: a lane's credit is reset
                    // the moment it reaches the quantum, so some lane always has credit < quantum.
                    // Kept as a guard: this step services no capable lane.
                }
            }
            if !plain.is_empty() {
                self.step_plain(&plain, false)?;
            }
            return Ok(());
        }
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
            self.step_plain(&plain, false)?;
        }
        crate::metrics::sched_touch(); // H3: round COMPLETE — the age gauge measures from here
        Ok(())
    }

    /// CF-P1e: do the busy lanes share one plain batched step this round (true) or run serial speculative rounds?
    /// `Auto` compares the estimated aggregate tokens/ms of the two arms (4% hysteresis): serial = the busy
    /// MTP-capable lanes' measured round speed (sum tokens / sum ms) plus one batched step for any MTP-off lanes;
    /// shared = n tokens per modelled shared-step time. A lane without an estimate yet falls back to a lane count of 4.
    fn use_shared_plain(&mut self, slots: &[usize]) -> bool {
        let n = slots.len();
        if n < 2 { return false; }
        match self.spec_policy {
            SpecLanes::Never => false,
            SpecLanes::Over(k) => n > k,
            SpecLanes::Auto => {
                let (mut tok, mut ms, mut capable, mut known) = (0.0f64, 0.0f64, 0usize, 0usize);
                for &s in slots {
                    let l = self.lanes[s].as_ref().unwrap();
                    if l.mtp.is_some() {
                        capable += 1;
                        if l.ema_tok > 0.0 && l.ema_ms > 0.0 { tok += l.ema_tok; ms += l.ema_ms; known += 1; }
                    }
                }
                if capable == 0 { return false; }          // nothing speculates: the plain batch already runs
                if known < capable { return n > 4; }       // an estimate is missing: the fixed threshold
                let plain_only = n - capable;
                let shared_ms = self.shared_scale * shared_step_prior_ms(n);
                if plain_only > 0 {
                    tok += plain_only as f64;
                    ms += self.shared_scale * shared_step_prior_ms(plain_only);
                }
                let (serial_rate, shared_rate) = (tok / ms, n as f64 / shared_ms);
                let pick = if shared_rate > serial_rate * 1.04 { true } else if serial_rate > shared_rate * 1.04 { false } else { self.shared_last };
                if pick != self.shared_last && self.policy_logs < 64 {
                    self.policy_logs += 1;
                    crate::reprintln!("[exl3-serve] spec-lanes auto: {} with {n} busy lanes (serial speculation {:.0} tok/s est, \
                               one shared step {:.0} tok/s est, scale {:.2})",
                              if pick { "SHARED plain step" } else { "serial speculation" },
                              serial_rate * 1e3, shared_rate * 1e3, self.shared_scale);
                }
                self.shared_last = pick;
                pick
            }
        }
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
            // acceptance estimates falls below the target (the reference implementation's -dds -dc), then verify
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
                crate::reprintln!("[exl3-serve] MTP off for this request (slot {s}): {:.2} ms/tok ({:.1} ms/round, \
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
                    self.model.dump_esel_hist(); // S-A3-o diagnostic (no-op unless --exl3-esel-hist)
                    self.model.dump_route_log(); // CF-P1e step 0 diagnostic (no-op unless --exl3-route-log)
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
    /// CF-P1e: fold a measured shared-round time (ms, lockstep-adopted under TP) into the model's learned scale.
    /// The first rounds of a width can include a graph capture, so an outlier above 3x the model is ignored.
    fn learn_shared_ms(&mut self, n: usize, ms: f64) {
        if n < 2 { return; }
        let r = ms / shared_step_prior_ms(n);
        if !(0.05..3.0).contains(&r) { return; }
        self.shared_scale = if self.shared_streak <= 1 && self.shared_scale == 1.0 { r } else { 0.9 * self.shared_scale + 0.1 * r };
    }

    /// `sync_head` (CF-P1e): the lanes are MTP-capable but this round is a shared plain step — keep each lane's draft
    /// head exactly as a control round does (write the head's row p KV-only from the lane's saved tap BEFORE the
    /// step; snapshot the step's tap row into the lane's window AFTER it), so its next speculative round is valid.
    fn step_plain(&mut self, slots: &[usize], sync_head: bool) -> Result<()> {
        if sync_head {
            let model = self.model.clone();
            if let Some(head) = model.mtp.as_ref() {
                for &s in slots {
                    let (b, p, tr) = { let l = self.lanes[s].as_ref().unwrap(); (l.last_tok as i32, l.pos, l.mtp) };
                    if let Some(tap_row) = tr {
                        model.mtp_plain_prime(&mut self.sc, head, s, b, p, tap_row)?;
                    }
                }
            }
        }
        let t_round = std::time::Instant::now();   // CF-P1e: the shared round's cost = head primes + the step
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
            let own = t_round.elapsed().as_secs_f64() * 1e3;
            let (ms, _, _) = self.tp_round("plain-step", 0, slots.len(), &w, own, 0.0, false)?;
            if sync_head { self.learn_shared_ms(slots.len(), ms); }
        } else if sync_head {
            self.learn_shared_ms(slots.len(), t_round.elapsed().as_secs_f64() * 1e3);
        }
        if let Some(g) = self.gate.as_mut() { g.plain += slots.len() as u64; }
        if sync_head {
            let model = self.model.clone();
            for (j, &s) in slots.iter().enumerate() {
                if self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some()) {
                    model.tap_snapshot_row(&mut self.sc, s, j)?;
                    self.lanes[s].as_mut().unwrap().mtp = Some(0);
                }
            }
        }

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
                self.model.dump_esel_hist(); // S-A3-o diagnostic (no-op unless --exl3-esel-hist)
                self.model.dump_route_log(); // CF-P1e step 0 diagnostic (no-op unless --exl3-route-log)
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
        let mut rots_printed = 0usize;
        loop {
            if self.fatal.is_some() {
                return self.die(waiting, rx);
            }
            if waiting.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                // Clean idle point: report the TOTAL number of fcfs rotations this scheduler made
                // (the per-rotation log caps at 16 lines; G1 needs the true count).
                if self.lane_order == LaneOrder::Fcfs && self.fcfs_rots > rots_printed {
                    crate::rprintln!("[exl3-serve] lane-order fcfs idle: {rots_printed}->{} total rotations", self.fcfs_rots);
                    rots_printed = self.fcfs_rots;
                }
                crate::metrics::sched_idle(); // H3: about to block for work (busy must not stick at 1)
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
                self.fail_step(&e);
            }
        }
    }

    /// A failed decode step (from `run` or a prefill-interleave hook): fail every live lane loudly
    /// (never emit garbage as if it were text) and mark the engine dead on a sticky CUDA error.
    fn fail_step(&mut self, e: &anyhow::Error) {
        eprintln!("[exl3-serve] decode step failed: {e:#}");
        for s in 0..self.lanes.len() {
            if let Some(lane) = self.lanes[s].take() {
                let _ = lane.tx.send(TokEvent::Finish { reason: format!("error: {e}") });
            }
        }
        if sticky_cuda_error(e) {
            self.fatal = Some(format!("{e:#}"));
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
            eprintln!("[exl3-serve] FATAL: exit 70 (--exit-on-fatal, default; a supervisor restarts the server)");
            std::thread::sleep(std::time::Duration::from_millis(300)); // let the error events flush
            crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7 + LR-1
        }
        while let Some(req) = rx.blocking_recv() {
            let _ = req.tx.send(TokEvent::Finish { reason: reason.clone() });
        }
    }

    /// VIS-2 (TP=1): arm `slot` for an image request — the 3-axis RoPE map + image-token rows
    /// (FwdModel::set_slot_vision) and the prefill splice data (psc.vis_img / vis_src). Returns the
    /// prefix-cache key (image rows replaced by pixel-hash ids, so a cached prefix is reused only
    /// for the SAME image) and a generated-token cap (none since VIS-3).
    fn vision_setup(&mut self, slot: usize, req: &BatchRequest) -> Result<(Vec<u32>, usize)> {
        let sec = self.mrope_section.context("bad request: this server has no vision tower for this model (text-only)")?;
        let embeds = req.image_embeds.as_ref().context("image spans without image embeddings")?;
        let h = self.model.cfg.hidden_size;
        let plen = req.prompt.len();
        let n_img: usize = req.image_spans.iter().map(|s| s.num_tokens).sum();
        anyhow::ensure!(embeds.len() == n_img * h, "image embeddings {} != {n_img} tokens x {h}", embeds.len());
        anyhow::ensure!(n_img <= crate::exl3_forward::VIS_MROPE_ROWS,
                        "bad request: {n_img} image tokens exceed the per-request limit of {}", crate::exl3_forward::VIS_MROPE_ROWS);
        let (pos3, delta) = crate::vision_encoder::mrope_positions(plen, &req.image_spans)?;
        self.model.set_slot_vision(slot, &pos3, delta, sec)?;
        let mut src = vec![-1i32; plen];
        let mut key = req.prompt.clone();
        let mut off = 0usize;
        for sp in &req.image_spans {
            let hh = (sp.pixel_hash as u32) ^ ((sp.pixel_hash >> 32) as u32);
            for j in 0..sp.num_tokens {
                src[sp.start + j] = (off + j) as i32;
                // ids >= 2^31 never collide with a vocabulary id; only the cache compares them
                key[sp.start + j] = 0x8000_0000 | (hh.wrapping_add((j as u32).wrapping_mul(0x9E37_79B1)) & 0x7FFF_FFFF);
            }
            off += sp.num_tokens;
        }
        // the embedding table is fp16: image rows enter the residual in the same precision class
        // (VIS-4: also the TP wire's precision, so every rank splices bit-identical rows)
        let rows: Vec<f32> = embeds.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let (_, digest) = crate::tp_serve::image_rows_wire(&rows);
        self.vis_seam = Some((n_img as u32, delta as i32, digest));
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_rows(self.chunk.max(1), self.tp.is_some()))?);
        }
        let img = self.model.vis_rows_upload(&rows)?;
        let psc = self.psc.as_mut().unwrap();
        psc.vis_img = Some(img);
        psc.vis_src = src;
        crate::rprintln!("[exl3-serve] VIS-2 slot={slot}: {} image(s), {n_img} image tokens in a {plen}-token prompt; rope delta {delta}",
                 req.image_spans.len());
        // VIS-3: the QSA indexer ropes its pooled keys through the same per-slot map, so an image
        // request runs past the dense window like text (no generated-token cap)
        Ok((key, usize::MAX))
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
        if req.prompt.is_empty() {
            let _ = req.tx.send(TokEvent::Finish { reason: "error: empty prompt".into() });
            return;
        }
        let mut max_new = req.max_new.min(self.model.max_pos().saturating_sub(req.prompt.len() + 1));
        // VIS-2: an image request arms the slot's 3-axis RoPE map and the prefill splice; every other
        // request puts the slot back on plain text RoPE (a previous image request may have armed it).
        self.vis_seam = None;
        let key: Vec<u32> = if req.image_spans.is_empty() {
            if let Err(e) = self.model.clear_slot_vision(free) {
                let _ = req.tx.send(TokEvent::Finish { reason: format!("error: slot reset failed: {e}") });
                self.fatal = Some(format!("{e:#}"));
                return;
            }
            req.prompt.clone()
        } else {
            match self.vision_setup(free, &req) {
                Ok((key, cap)) => {
                    max_new = max_new.min(cap);
                    key
                }
                Err(e) => {
                    let _ = self.model.clear_slot_vision(free);
                    if let Some(p) = self.psc.as_mut() { p.vis_img = None; p.vis_src.clear(); }
                    let msg = format!("{e:#}");
                    let reason = if msg.starts_with("bad request:") || msg.starts_with("unprocessable:") {
                        format!("error: {msg}")
                    } else {
                        format!("error: vision setup failed: {msg}") // an engine fault, not the client's
                    };
                    let _ = req.tx.send(TokEvent::Finish { reason });
                    return;
                }
            }
        };
        let pre = self.prefill_cached(free, &req.prompt, &key, req.ckpt_at, &req.tx);
        if let Some(p) = self.psc.as_mut() { p.vis_img = None; p.vis_src.clear(); } // VIS-2: splice data is per prefill
        match pre {
            Ok(true) => {}
            // WP02: the client left mid-prefill — the slot is free again, nobody to tell
            Ok(false) => return,
            Err(e) => {
                crate::reprintln!("[exl3-serve] prefill failed (slot {free}): {e:#}");
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
        // prompt tail + everything generated — the reference implementation's past_ids = the full sequence.
        let pen = crate::exl3_forward::PenParams::new(req.rep_penalty, req.presence_penalty,
                                                      req.frequency_penalty, self.pen_range, self.pen_range);
        if pen.is_some() {
            crate::reprintln!("[exl3-serve] penalties slot={free}: {pen:?} (window {} + {} tokens)", self.pen_range, self.pen_range);
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
            if let Some((n, d, h)) = self.vis_seam {
                // VIS-4: the image rows, their count and the rope delta are identical on every rank
                w.extend([0x7155_0000, n, d as u32, h as u32, (h >> 32) as u32]);
            }
            if let Some(st) = self.wp16.as_ref() {
                w.extend([st.live() as u32, st.n_alloc as u32,
                          fnv32(0, st.positions(free).iter().map(|&x| x as u32))]);
            }
            if let Err(e) = self.tp_round("seam", 0, 1, &w, 0.0, 0.0, false) {
                crate::reprintln!("[exl3-serve] TP seam lockstep failed (slot {free}): {e:#}");
                let _ = req.tx.send(TokEvent::Finish { reason: format!("error: TP lockstep failed: {e}") });
                self.fatal = Some(format!("{e:#}"));
                return;
            }
        }
        if req.tx.send(TokEvent::Tok(last)).is_err() && self.tp.is_none() {
            // WP02: the client is gone — no lane (the prefix snapshot stays valid). TP-D: under TP
            // the lane is created on every rank and the head's next sweep cancels it.
            crate::reprintln!("[mtp-stats] slot={free} finish=cancelled gen=1 rounds=0 plain_steps=0 temp={}", req.temperature);
            return;
        }
        // WP04-v2: engine-side TTFT (receipt = the handler's hand-off, AFTER tokenization and
        // template rendering), so a client TTFT can be split into HTTP/tokenizer vs engine time.
        crate::rprintln!("[exl3-serve] first token slot={free} {:.1} ms after receipt (queued {:.1} ms)",
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
                    crate::reprintln!("[exl3-serve] MTP prime failed (lane runs plain): {e:#}");
                    if self.tp.is_some() || sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                    None
                }
            }
        } else {
            None
        };
        // WP08: the reference implementation feeds the detector every emitted token, the first one included.
        let mut loop_det = self.loop_cfg
            .map(|(w, r)| crate::loop_detect::LoopDetector::for_stop_on_loop(w, r));
        if let Some(d) = loop_det.as_mut() { d.feed(last); }
        let admit_seq = self.admit_ctr;   // CF-FCFS: FCFS order; the same on every TP rank
        self.admit_ctr += 1;
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
            wp23: Wp23Lane { fit: CostFit::with_prior(dds_prior_w(self.tp_world())), ..Default::default() },
            st_rounds: 0,
            st_drafted: 0,
            st_accepted: 0,
            st_ms: 0.0,
            st_plain: 0,
            st_ctrl: 0,
            mtp,
            admit_seq,
            fcfs_credit: 0,
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
                crate::rprintln!("[tp-serve] rank {} idle after control step {step_no}: lockstep agrees {} ok, pre-verify agrees {} ok{}",
                         t.rank, t.agrees, t.pre_agrees,
                         if t.ident { format!(", graphed digests {} compared, {} differ", t.ident_bufs, t.ident_bad) } else { String::new() });
                crate::rprintln!("{}", t.wp.line(t.rank)); // H5: twin of the line above - never direct on a served path
                t.wp = WaitProf::default();
            }
        }
    }

    /// VIS-4: the binary channel — raw image-row payloads written right after a Step frame, in the
    /// order of that Step's image admits (every node reads them back by `image_bytes`).
    fn tp_ship_raw(&mut self, payloads: &[Vec<u8>]) -> Result<()> {
        if payloads.is_empty() { return Ok(()); }
        use std::io::Write;
        let t = self.tp.as_mut().context("TP ship without TP state")?;
        let t0 = std::time::Instant::now();
        let total: usize = payloads.iter().map(|p| p.len()).sum();
        for s in t.ctl.iter_mut() {
            for p in payloads { s.write_all(p)?; }
            s.flush()?;
        }
        crate::rprintln!("[exl3-serve] VIS-4: shipped {} image payload(s), {:.1} MB to {} node(s) in {:.1} ms",
                 payloads.len(), total as f64 / 1e6, t.ctl.len(), t0.elapsed().as_secs_f64() * 1e3);
        Ok(())
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
        let mut rots_printed = 0usize;
        let fast_ctl = crate::opts::var(crate::opt!("exl3-tp-fastctl")).map_or(true, |v| v != "0");
        println!("[exl3-serve] TP control: {}", if fast_ctl { "step-go over RDMA while live, TCP Step only with events (TP-E)" }
                 else { "TCP Step every step (--exl3-tp-fastctl=0)" });
        loop {
            if waiting.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                // Clean idle point (TP twin of the single-process loop): total fcfs rotations.
                if self.lane_order == LaneOrder::Fcfs && self.fcfs_rots > rots_printed {
                    crate::rprintln!("[exl3-serve] lane-order fcfs idle: {rots_printed}->{} total rotations", self.fcfs_rots);
                    rots_printed = self.fcfs_rots;
                }
                crate::metrics::sched_idle(); // H3: about to block for work (busy must not stick at 1)
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
            let mut payloads: Vec<Vec<u8>> = Vec::new(); // VIS-4: image rows of this step's admits, in order
            let mut free = 0usize;
            for s in 0..self.lanes.len() {
                match &self.lanes[s] {
                    None => free += 1,
                    Some(l) if l.tx.is_closed()
                             || crate::server::stream_backlog_exceeded(l.tx.len()) => { // LR-5
                        if !l.tx.is_closed() {
                            crate::metrics::stream_backlog_cancel();
                            crate::reprintln!("[exl3-serve] slot {s}: stream cancelled — unconsumed event backlog at the --stream-backlog-events limit (reader stalled; LR-5; Cancel shipped to the mirrors)");
                        }
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
                let mut w = WireRequest::from(&req);
                if let Some(e) = req.image_embeds.as_ref().filter(|_| !req.image_spans.is_empty()) {
                    // VIS-4: the rows ride the binary channel right after this step's frame
                    let (bytes, digest) = crate::tp_serve::image_rows_wire(e);
                    w.image_bytes = bytes.len() as u64;
                    w.image_digest = digest;
                    payloads.push(bytes);
                }
                events.push(TpEvent::Admit(w));
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
                if let Err(e) = self.tp_ship_raw(&payloads) {
                    return Err(self.tp_fail(format!("image payload send failed at step {step_no}: {e:#}"), &mut waiting, Some(&mut rx)));
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
        crate::rprintln!("[exl3-serve] TP control (node): {}", if fast_ctl { "step-go over RDMA while live (TP-E)" } else { "TCP Step every step" });
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
                        // H7: END-OF-LIFE line — direct print; the node exits right after the
                        // return and no logq flush can be guaranteed there.
                        eprintln!("[exl3-serve] TP node: head control stream closed ({e:#}) — session over after {step_no} steps, {admitted} requests");
                        return Ok(());
                    }
                }
            };
            match msg {
                ServingMsg::Shutdown => {
                    // H7: END-OF-LIFE line — direct print (same reason as the closed-stream line).
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
                                let (tx, _rx, _bl) = crate::server::tok_channel(); // mirror dummy; LR-5 gated off under TP
                                let min_p = w.min_p;
                                let (ib, idg) = (w.image_bytes, w.image_digest);
                                let mut req = w.into_request(tx);
                                req.min_p = min_p;
                                if ib > 0 {
                                    // VIS-4: this admit's image rows follow the Step frame (binary channel)
                                    use std::io::Read;
                                    let mut buf = vec![0u8; ib as usize];
                                    let t = self.tp.as_mut().context("mirror without TP state")?;
                                    t.ctl[0].read_exact(&mut buf).context("TP node: image payload read")?;
                                    anyhow::ensure!(crate::tp_serve::fnv64(&buf) == idg,
                                                    "TP node: image payload digest mismatch (step {step_no})");
                                    req.image_embeds = Some(buf.chunks_exact(2)
                                        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect());
                                }
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

impl TpSync {
    /// TP-4D (audit C2) world > 2: the outcome of ONE WP16 checkpoint allocation attempt as a lockstep decision.
    /// `CkptStore::acquire` freezes its cap when THIS rank's `cuMemAlloc` fails; ranks whose allocation outcomes
    /// differ hold different checkpoint sets and every later store-dependent decision (resume point, chunk grid)
    /// goes rank-local. Every rank sends (buffers held, ok); Ok(ranks that failed) — if any, ALL ranks must treat
    /// the allocation as failed (and drop a buffer they did get) so the stores stay identical.
    pub(crate) fn alloc_agree_hub<X: crate::net::HubXport>(&mut self, x: &mut X, n_alloc: usize, ok: bool)
                                                           -> Result<Vec<usize>> {
        use crate::tp_lockstep as ls;
        let all = crate::net::hub_all(x, &ls::alloc_frame(n_alloc, ok), ls::ALLOC_WIRE)
            .map_err(|e| self.hub_fail("wp16 alloc", &format!("{e:#}")))?;
        ls::alloc_verdict(&all, self.world as usize).map_err(|why| self.hub_fail("wp16 alloc", &why))
    }

    /// The body of `CkptStore::acquire`'s allocation closure with the cross-rank agreement applied: `r` = this rank's
    /// local `ckpt_alloc` result, `n_before` = the buffers this rank held before it. Every rank runs this at the same
    /// program point (the store state is identical by invariant) and gets the same outcome: Ok = every rank got its
    /// buffer, Err = at least one rank's allocation failed (a buffer this rank did get is dropped) so `acquire` freezes
    /// the cap on ALL ranks. A transport failure of the agree itself lands in `hub_err` (the caller must return it:
    /// `acquire` would otherwise swallow it as an allocation failure).
    pub(crate) fn alloc_outcome<B, X: crate::net::HubXport>(&mut self, x: &mut X, n_before: usize, r: Result<B>,
                                                            hub_err: &mut Option<anyhow::Error>) -> Result<B> {
        match self.alloc_agree_hub(x, n_before, r.is_ok()) {
            Ok(failed) if failed.is_empty() => r,
            Ok(failed) => match r {
                Ok(_dropped) => Err(anyhow::anyhow!("checkpoint allocation failed on rank(s) {failed:?} (agreed across ranks)")),
                Err(e) => Err(e),
            },
            Err(e) => { let m = format!("{e:#}"); *hub_err = Some(e); Err(anyhow::anyhow!(m)) }
        }
    }

    /// World 2 keeps today's rank-local decision (NO exchange, no behaviour change: the gap stays open there, see
    /// PLAN/TP-4D_REPORT.md); world > 2 runs `alloc_outcome` over the hub's control slots.
    fn wp16_alloc<B>(&mut self, n_before: usize, r: Result<B>, hub_err: &mut Option<anyhow::Error>) -> Result<B> {
        if self.world <= 2 { return r; }
        match crate::net::link_hub(self.deadline_ms) {
            Ok(mut x) => self.alloc_outcome(&mut x, n_before, r, hub_err),
            Err(e) => { let m = format!("{e:#}"); *hub_err = Some(e); Err(anyhow::anyhow!(m)) }
        }
    }
}

/// TP-4D: the head's rank/host map (`rank r = host (addr)`, r >= 1), set once by main after the node sync so the API
/// startup line can name which box is which rank. Display only — nothing reads it for a decision.
static TP_RANK_MAP: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
pub fn set_tp_rank_map(map: Vec<String>) { let _ = TP_RANK_MAP.set(map); }

/// TP-D: bring up the EXL3 TP attachment from a TpContext (the sanity + branch handshakes).
fn tp_attach(mut ctx: crate::tp::TpContext) -> Result<crate::exl3_forward::xtp::TpAttach> {
    ctx.sanity()?;
    ctx.branch_check(&crate::tp::TpBranch::Exl3Xtp)?;
    let (rank, world, link) = ctx.into_parts();
    Ok(crate::exl3_forward::xtp::TpAttach { rank, world, link })
}

/// TP-D: --exl3-tp-preverify=0 (diagnostic, prices the belt): skip the pre-verify width/drafts
/// exchange + agree (the post-round agree still proves the width, one round late — TP-C's shape).
/// Rides TpConfig's CLI option-registry snapshot, so both ranks decide alike.
fn tp_preverify_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-tp-preverify")).map_or(true, |v| v != "0"))
}

/// TP-D: --exl3-tp-ident=1 (diagnostic; rides TpConfig's CLI option-registry snapshot, so both ranks agree):
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

/// TP-4D: the world > 2 boot-agree frame magic (class prefix; a frame of another class at this point is a desync).
const BOOT_AGREE_MAGIC: u32 = 0xB007_A612;

/// TP-4D: Ok when every rank sent the boot magic and rank 0's config hash; Err names every differing rank with both
/// hashes (`all[r] = [magic, hash]`).
fn boot_hash_verdict(all: &[Vec<u32>], my_rank: usize) -> std::result::Result<(), String> {
    let mut bad = Vec::new();
    for (r, f) in all.iter().enumerate() {
        if f[0] != BOOT_AGREE_MAGIC {
            bad.push(format!("rank {r} sent frame class {:#010x}, not a boot-agree frame", f[0]));
        } else if f[1] != all[0][1] {
            bad.push(format!("rank {r} hash {:08x} != head {:08x}", f[1], all[0][1]));
        }
    }
    if bad.is_empty() { Ok(()) } else { Err(format!("{} (this is rank {my_rank})", bad.join("; "))) }
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
    w.push(boot_prior_word(s.tp_world()));
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
    let world = s.tp.as_ref().map_or(2, |t| t.world);
    println!("[exl3-serve] TP={world} rank {rank}: boot config hash {h:08x} (width {} chunk {} k {} dds {} prefix {} wp16 cap {} tail {} tune {tune} \
              dds-guard prior {} ms/draft (--tp-dds-prior; TP=1 6.0) prefill-overlap {} vp-sampled {}{})",
             s.width, s.chunk, s.mtp_k, s.draft_conf, s.prefix_on, s.wp16.as_ref().map_or(0, |st| st.cap),
             crate::exl3_forward::tail_ckpt_on(), dds_prior_w(s.tp_world()),
             crate::exl3_forward::xtp::pf_overlap_desc(crate::exl3_forward::xtp::pf_overlap_code()),
             crate::exl3_forward::xtp::vp_sampled_desc(crate::exl3_forward::xtp::vp_sampled_code()),
             if crate::exl3_forward::xtp::seq_parallel_on() {
                 format!(" seq-parallel {}", crate::exl3_forward::xtp::seq_parallel_desc(crate::exl3_forward::xtp::seq_parallel_code()))
             } else { String::new() });
    println!("[exl3-serve] TP={world} rank {rank}: options {} (spmd digest {})", crate::opts::set_summary(),
             spmd.map_or("none".to_string(), |d| format!("{d:08x}")));
    if world > 2 {
        // TP-4D: world > 2 — every rank's hash through the head hub (probe-tolerant, unbounded: the four model
        // loads finish at different times; a dead peer still aborts within the dead-peer probe): EVERY rank sees
        // EVERY hash and names the offender, instead of one head-side verdict. world 2 keeps the pairwise agree.
        let all = crate::net::exchange_u32s_all(&[BOOT_AGREE_MAGIC, h], 4)
            .context("TP boot agree (head hub; peer dead or link aborted)")?;
        if let Err(why) = boot_hash_verdict(&all, rank as usize) {
            crate::net::abort_link_keep_code();
            anyhow::bail!("TP boot agree FAILED: the ranks resolved different serve configs (hash {h:08x} on rank {rank}): {why}");
        }
        return Ok(());
    }
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
    // S7 (REL_V0_7_3): refuse a bad --otel-endpoint BEFORE the RDMA attach (inside
    // build_serve it runs after tp_attach). Returning Err (not exit) lets the caller report it.
    if let Some(cfg) = crate::otel::config_from_args(|f| arg(args, &format!("--{f}")).map(|s| s.to_string())) {
        if let Err(e) = cfg.hostport() { anyhow::bail!("[otel] {e}"); }
    }
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
    let world = parts.sched.tp.as_ref().map_or(2, |t| t.world);
    if world <= 2 {
        println!("[exl3-serve] TP={world} serving: every §1b feature as TP=1 (graphs, MTP k={}, DDS {}, WP27 forks, prefix cache {}, \
                  WP16 {}) — head-authoritative Step/Cancel/Admit(seed) + HeadFlag, agree per seam/round + pre-verify",
                 parts.sched.mtp_k, parts.sched.draft_conf, parts.sched.prefix_on,
                 parts.sched.wp16.as_ref().map_or("off".to_string(), |st| format!("cap {}", st.cap)));
    }
    if world > 2 {
        // TP-4D (review F1): at world > 2 some §1b features are OFF, INERT or refused, so the world-2 "every §1b feature as
        // TP=1" claim is not printed. The `TP=N serving: ` prefix is the launcher's exactly-once marker (tp4_launch.sh
        // SERVE_SERVING_RE); the per-feature state rides the next line. Option-resolved state, read from the same predicates
        // the load uses; the `TP-H ...` / `[exl3-tp] ...` load lines above show what attached.
        {
            use crate::exl3_forward::xtp;
            let s = &parts.sched;
            let mtp_on = s.model.mtp.is_some() && !crate::exl3_forward::mtp_disabled_by_opt();
            let graphs = if crate::opts::var(crate::opt!("exl3-no-graph")).is_ok()
                || crate::opts::var(crate::opt!("mtp-dump")).map_or(false, |v| !v.is_empty()) { "OFF (--exl3-no-graph / --mtp-dump)" }
                else if arg(args, "--graph-precapture") == Some("off") { "ON (lazy capture: --graph-precapture off)" }
                else { "ON (precaptured at boot)" };
            let mtp = if mtp_on { format!("ON k={}", s.mtp_k) }
                else if s.model.mtp.is_some() { "OFF (--exl3-no-mtp)".to_string() } else { "OFF (pack has no MTP head)".to_string() };
            let dds = if mtp_on && s.draft_conf > 0.0 { format!("ON at {}", s.draft_conf) } else { "OFF".to_string() };
            let dec = xtp::dec_xport_mode() == 1;
            println!("[exl3-serve] TP={world} serving: head-authoritative Step/Cancel/Admit(seed) + HeadFlag, hub lockstep agree per \
                      seam/round + pre-verify — some TP=1 §1b features are OFF / INERT / refused at world {world}; per-feature state on the \
                      next line, table in PLAN/TP-4D_REPORT.md");
            println!("[exl3-serve] TP={world} features: graphs {graphs}; MTP {mtp}; DDS {dds}; prefix cache {}; WP16 {}; WP27 forks not \
                      world-gated (--wp27-off / --wp27-forks / PDL decide); TP-H greedy vocab-parallel head {}; sampled / penalized / \
                      ratio-rule rows = {}; sharded MTP-head screen {}; decode \
                      transport {}; prefill transport {}; prefill overlap {}; sequence-parallel prefill {}; EP deal '{}'; pre-verify \
                      drain SKIPPED at world > 2 (it runs at world 2 only; hardware-unproven); refused if passed: --tp-oneshot, --tp-dec-grecv != 0, \
                      --ep-deal freq outside world 4",
                     if s.prefix_on { "ON" } else { "OFF (--prefix-cache off)" },
                     s.wp16.as_ref().map_or("OFF".to_string(), |st| format!("ON (cap {} checkpoints)", st.cap)),
                     if xtp::vp_head_on() && dec { "ON" } else { "OFF (--tp-vp-head off / --tp-dec-xport 0)" },
                     if xtp::vp_head_on() && dec && xtp::vp_sampled_on() { "ON (vocab-parallel all-gather, TP-4H2)" }
                         else { "REPLICATED head (--tp-vp-sampled off / --tp-vp-head off / --tp-dec-xport 0)" },
                     if mtp_on && xtp::dh_shard_on() && dec { "ON" } else { "OFF (no MTP head / --tp-dh-shard off / --tp-dec-xport 0)" },
                     if dec { format!("folded K1 + multi-block K2, {} exchange(s) per reduce{}, receive cpu_done only",
                                      if xtp::tp_reduce_single_for(world) { 1 } else { (world as u32).trailing_zeros() },
                                      if xtp::tp_reduce_single_for(world) { " (--tp-reduce single: ONE all-peers stage, the world-4 default)" } else { "" }) }
                         else { "serial K1/K2 + cvt (--tp-dec-xport 0)".to_string() },
                     match xtp::prefill_xport_mode() { 0 => "serial rounds-aware K1/K2", 1 => "single-rail pipelined (hardware-unproven at world > 2)", _ => "dual-rail pipelined (hardware-unproven at world > 2)" },
                     if xtp::pf_overlap_rows() == 0 { "OFF" } else { "INERT" },
                     if !xtp::seq_parallel_on() { "OFF" }
                         else if world == 4 { "ON (TP-4S quarter rows: recursive-halving reduce-scatter + recursive-doubling all-gather; hardware-unproven until PLAN/TP-4S_GATES.md rungs 1-2 pass)" }
                         else { "INERT" },
                     xtp::ep_deal_kind());
        }
        // TP-4D: the API startup line names the world, the lockstep protocol and the rank/host map
        println!("[exl3-serve] TP={world} API up: head = rank 0; {}; lockstep = head-hub (one merged op per point, per-wait \
                  deadline {} ms — --tp-lockstep-timeout-ms); Step/Cancel/Admit/HeadFlag over {} node control streams",
                 TP_RANK_MAP.get().map_or("rank/host map unavailable".to_string(), |m| m.join("; ")),
                 parts.sched.tp.as_ref().map_or(0, |t| t.deadline_ms), world - 1);
    }
    serve_http(parts, Some(streams))
}

/// TP-D: the TP=2 NODE serve (rank >= 1): the same boot as the head (same argv), then the mirror
/// loop on this thread, pinned to core 9.
/// PACK-FIX: a boot failure (attach, load, boot agree, pinning) is reported to the head over the
/// control stream before returning — the head watches the stream through its own load and exits
/// loudly with this reason, instead of loading for minutes and then losing the node in a probe.
pub fn run_tp_node_serve(args: &[String], model_dir: &str, ctx: crate::tp::TpContext,
                         mut stream: std::net::TcpStream) -> Result<()> {
    let node_world = ctx.world;   // TP-4Z1: the worker mask needs the world to know whether the single-stage proxy (core 17) is live
    // S7 (REL_V0_7_3): same hoisted refusal as on the head - BEFORE the attach, and reported
    // through the boot failure path instead of process::exit(1).
    if let Some(cfg) = crate::otel::config_from_args(|f| arg(args, &format!("--{f}")).map(|s| s.to_string())) {
        if let Err(e) = cfg.hostport() { anyhow::bail!("[otel] {e}"); }
    }
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
    set_worker_mask(tp_worker_mask(&cpu_mask, true, node_world));
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

/// CF-P1d pre-load fit check: the parts of a serving footprint, in bytes (estimates; see `fit_plan`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitParts {
    pub weights: u64,  // device weights on this rank (model shards / world + the vision tower on the head)
    pub table: u64,    // the PLE n-gram table (host RAM when RAM-resident)
    pub per_lane: u64, // one lane: KV + indexer keys + MTP-head KV over max_pos, + its recurrent state (x1.15)
    pub fixed: u64,    // scratch, prefill scratch, graphs, vendor workspaces, prefix-checkpoint cap
    pub floor: u64,    // the memory watchdog's floor (--mem-watchdog-gb)
}

/// What a configuration needs and whether it fits: Ok(table_in_ram_possible) or Err(the refusal text).
pub fn fit_plan(p: FitParts, lanes: usize, avail: u64, mode: &str, max_pos: usize) -> Result<bool, String> {
    let gb = |b: u64| b as f64 / 1e9;
    let base = p.weights + p.fixed + p.per_lane * lanes as u64 + p.floor;
    let with_table = base + p.table;
    let line = format!("model weights {:.1} GB + {lanes} lane(s) x {:.2} GB (KV for {max_pos} tokens) + {:.1} GB working memory \
                        + {:.1} GB safety floor{} = {:.1} GB; this machine has {:.1} GB available",
                       gb(p.weights), gb(p.per_lane), gb(p.fixed), gb(p.floor),
                       if mode == "ram" { format!(" + n-gram table {:.1} GB in RAM", gb(p.table)) } else { String::new() },
                       gb(if mode == "ram" { with_table } else { base }), gb(avail));
    let fits_ssd = base <= avail;
    if mode == "ram" && with_table > avail {
        let mut msg = format!("this configuration does not fit in memory: {line}.");
        if fits_ssd {
            msg += " It fits with the n-gram table on SSD: use --ple-ram ssd (or leave --ple-ram at auto) — identical output, decode ~1-2% slower.";
        }
        return Err(msg + &suggest(p, lanes, avail, max_pos));
    }
    if !fits_ssd {
        return Err(format!("this configuration does not fit in memory, even with the n-gram table on SSD: {line}.{}",
                           suggest(p, lanes, avail, max_pos)));
    }
    Ok(with_table <= avail)
}

fn suggest(p: FitParts, lanes: usize, avail: u64, max_pos: usize) -> String {
    let room = avail.saturating_sub(p.weights + p.fixed + p.floor);
    let mut out = String::new();
    if p.per_lane > 0 && room / p.per_lane < lanes as u64 {
        let n = room / p.per_lane;
        if n >= 1 { out += &format!(" At this context length, at most --max-batch {n} fits."); }
        let per_tok = p.per_lane as f64 / max_pos.max(1) as f64;
        let ctx = (room as f64 / lanes.max(1) as f64 / per_tok) as u64 / 1024 * 1024;
        if ctx >= 4096 { out += &format!(" With --max-batch {lanes}, at most --max-seq-len {ctx} fits."); }
    }
    if room < p.per_lane {
        out += " The model alone leaves too little memory on one box: run it over two boxes (--tp 2).";
    }
    out
}

/// Estimate the footprint of an EXL3 pack from its files and config (no GPU, no weight reads).
pub fn fit_parts(model_dir: &str, max_pos: usize, world: usize, head: bool, kv_bytes: f64, ckpt_gb: f64) -> Option<FitParts> {
    let dir = std::path::Path::new(model_dir);
    let size = |n: &str| std::fs::metadata(dir.join(n)).map(|m| m.len()).unwrap_or(0);
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).ok()?).ok()?;
    let all: std::collections::BTreeSet<String> = idx["weight_map"].as_object()?.values()
        .filter_map(|v| v.as_str().map(str::to_string)).collect();
    // the n-gram table is counted as `table`, not device weights (in a shipped shard dir it sits under seg/)
    let (ngram, files): (std::collections::BTreeSet<String>, std::collections::BTreeSet<String>) =
        all.into_iter().partition(|f| f.contains("ngram"));
    let shards: u64 = files.iter().map(|f| size(f)).sum();
    // A shipped per-rank shard dir (B32, `gb10_shard.json`) already holds this rank's share; a full pack is split by world.
    let share = if dir.join("gb10_shard.json").is_file() { 1 } else { world.max(1) as u64 };
    let mut weights = shards / share;
    if head { weights += size("vision_tower_bf16.safetensors"); }
    let mut table: u64 = ngram.iter().map(|f| size(f)).sum();
    if table == 0 {
        table = std::fs::read_dir(dir).ok()?.flatten()
            .filter(|e| e.file_name().to_string_lossy().contains("ngram"))
            .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0)).sum();
    }
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    let t = cfg.get("text_config").unwrap_or(&cfg);
    let g = |k: &str| t[k].as_u64().unwrap_or(0);
    let types = t["layer_types"].as_array()?;
    let n_full = types.iter().filter(|v| v.as_str() == Some("full_attention")).count() as u64;
    let n_lin = types.len() as u64 - n_full;
    let kvh_hd = g("num_key_value_heads") * g("head_dim");
    // per token: trunk K+V (kv-cache format) + indexer keys (bf16) + the MTP head's own K+V (f16)
    // trunk KV heads are split across ranks up to their count (2 KV heads: TP=2 and TP=4 hold one each)
    let kv_split = (world.max(1) as u64).min(g("num_key_value_heads").max(1)) as f64;
    let per_tok = n_full as f64 * 2.0 * kvh_hd as f64 * kv_bytes / kv_split
        + (n_full * g("indexer_kv_heads") * g("indexer_head_dim") * 2) as f64
        + (g("mtp_num_hidden_layers").max(1) * 2 * kvh_hd * 2) as f64;
    // per lane fixed: GDN recurrent state (f32) + conv ring
    let state = n_lin * g("linear_num_value_heads") * g("linear_key_head_dim") * g("linear_value_head_dim") * 4
        + n_lin * (2 * g("linear_num_key_heads") * g("linear_key_head_dim") + g("linear_num_value_heads") * g("linear_value_head_dim"))
            * g("linear_conv_kernel_dim").max(1) * 2;
    let per_lane = ((per_tok * max_pos as f64 + state as f64) * 1.15) as u64;
    // Working memory beyond weights + table + lanes, measured on .14 2026-10-02 (Flash-Next 3.05 TP=1, 131K): 14.3 GB
    // at 1 lane and 14.8 GB at 8 lanes after the boot (scratch, prefill scratch, graphs, vendor workspaces, staging),
    // + the prefix-checkpoint pool, which grows to its cap while serving.
    let fixed = 15_000_000_000 + (ckpt_gb * (1u64 << 30) as f64) as u64;
    let floor_gb: f64 = crate::opts::var(crate::opt!("mem-watchdog-gb")).ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
    Some(FitParts { weights, table, per_lane, fixed, floor: (floor_gb * 1e9) as u64 })
}

/// The served `--prefill-chunk` default (owner 2026-10-02, v0.7.1: steady 4,095-row chunks, TP-4X1).
pub const DEFAULT_PREFILL_CHUNK: usize = 4095;

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
    let tp_world: u32 = tp.as_ref().map_or(1, |a| a.world as u32);
    let who = match tp_rank { Some(r) => format!("TP={tp_world} rank {r}"), None => "TP=1".to_string() };
    // v0.7.3 ("tel"): --otel-* on the EXL3 server, the SAME shared parser as the NVFP4 path
    // (otel::config_from_args). Parsed + refused on EVERY rank BEFORE the model load (a bad
    // endpoint aborts the boot); the SINK is built later, only where the HTTP API lives.
    let otel_cfg = crate::otel::config_from_args(
        |f| arg(args, &format!("--{f}")).map(|s| s.to_string()));
    if let Some(cfg) = &otel_cfg {
        if let Err(e) = cfg.hostport() { eprintln!("[otel] {e}"); crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 1); } // H7
    }
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
        anyhow::ensure!(matches!(v, "auto" | "ram" | "ssd" | "on" | "off"), "--ple-ram must be auto, ram or ssd (on/off are the old spellings of ram/ssd)");
        crate::opts::set(crate::opt!("ple-ram"), v);
    }
    // --kv-cache f32|f16|fp8|q8: attention KV storage format (PLAN/KV_CACHE_FORMATS.md). Default
    // q8 (owner decision 2026-09-26, after the WP25 quality gate); f32 is exact; f16 halves, fp8 (e4m3 + per-row f32 scale) and q8 (the reference implementation's H32-rotated int8 +
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
            prefill_chunk: arg(args, "--prefill-chunk").and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_PREFILL_CHUNK),
        };
        tune::boot(&tune::BootReq { sel, model_dir, posture, profile_mhz, draft_on,
                                    tp: tp_world });
    }
    // CF-P1d: refuse BEFORE a minute-long load when the configuration cannot fit (with the arithmetic and what to
    // change), instead of a watchdog exit half-way through the boot.
    {
        let kvb = match kv_name { "f32" => 4.0, "f16" => 2.0, "fp8" => 1.0, _ => 1.0625 };
        let mode = match crate::opts::var(crate::opt!("ple-ram")).as_deref() { Ok("on") | Ok("ram") => "ram", Ok("off") | Ok("ssd") => "ssd", _ => "auto" };
        let world = tp_world.max(1) as usize;
        if let Some(parts) = fit_parts(model_dir, max_pos, world, tp_rank.map_or(true, |r| r == 0), kvb, wp16_gb) {
            let avail = std::fs::read_to_string("/proc/meminfo").ok()
                .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:"))
                    .and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()))
                .unwrap_or(0) * 1024;
            if avail > 0 {
                if let Err(msg) = fit_plan(parts, width, avail, mode, max_pos) {
                    eprintln!("error: {msg}");
                    std::process::exit(2);
                }
            }
        }
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
    // API-parity DDS: dynamic draft stop (the reference implementation's -dds -dc 0.6). CLI --draft-confidence <p>;
    // 0 = fixed-depth chain. Output-neutral (greedy lossless, sampled distribution-exact).
    // Default 0.4, not the reference implementation's 0.6: our draft pass is cheaper relative to a verify row, so
    // longer drafts pay (sweep 2026-09-25, k5 greedy 1024 tok: 0.4 -> ansic 87.9 / python 77.6 /
    // prose 55.8; 0.6 -> 85.2 / 74.2 / 55.6; 0.2-0.7 all within +-2).
    let draft_conf: f64 = arg(args, "--draft-confidence").and_then(|s| s.parse().ok()).unwrap_or(0.4);
    anyhow::ensure!((0.0..1.0).contains(&draft_conf), "--draft-confidence must be in [0, 1)");
    // WP08 (owner decision 2026-09-26): the reference implementation's streaming loop detector, ON by default at its
    // launcher setting stop_on_loop = (300, 3) (chat.py -lw 300 -lmr 3): a response whose last
    // 300 tokens are one repeating sequence of period <= 100 ends as a normal stop
    // (stop_reason "loop_detected"). --loop-detect off disables; --loop-window / --loop-min-reps
    // tune it (the reference implementation's asserts: window > 1, 1 < reps < window). It only changes output when
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
    // WP15: penalty window, the reference implementation's -penr (penalty_range, default 1024) used as both its
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
    // reference implementation's defaults 1.0 / 0 / 0 when absent). Same validity rules as a request's values.
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
    // v0.7.1 (owner 2026-10-02, D-TP4-6): default 2048 -> 4095 (TP-4X1: W=4 prefill -3% at 8K, -5.4% at 32K).
    let chunk: usize = arg(args, "--prefill-chunk").and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_PREFILL_CHUNK);
    if tp_rank.is_some() {
        anyhow::ensure!(psc_rows(chunk.max(1), true) <= crate::exl3_forward::xtp::TP_MAX_ROWS,
                        "TP: --prefill-chunk {chunk} gives prefill chunks up to {} rows, above the TP all-reduce \
                         partial buffer ({} rows; C1 tail checkpoints merge up to min(2C-1, {}) rows)",
                        psc_rows(chunk.max(1), true), crate::exl3_forward::xtp::TP_MAX_ROWS,
                        crate::exl3_forward::xtp::TP_MAX_ROWS - 1);
    }
    // S-A3-u: build the prefill scratch at startup and warm cuBLASLt (lazy kernel loads + plans
    // ~0.4 s) so request 1 does not pay it. --exl3-no-prefill-warmup skips (diagnostics).
    let mut psc0 = model.prefill_scratch(psc_rows(chunk.max(1), tp_rank.is_some()))?;
    if crate::opts::var(crate::opt!("exl3-no-prefill-warmup")).is_err() {
        model.prefill_warmup(&mut psc0, 0, &crate::exl3_forward::FwdModel::prefill_warmup_widths(chunk.max(1)))?;
    }
    // WP02 --exit-on-fatal (default ON since v0.7.3, owner decision 2026-10-06; was opt-in):
    // a sticky CUDA error or a scheduler panic exits 70 so a supervisor (serve_ours.sh SUPERVISE=1)
    // restarts the server. --exit-on-fatal off keeps the old behaviour: the engine goes DEAD and
    // /health + every request answer 503 until restarted. Registered Bool: bare flag = on.
    let exit_on_fatal = crate::opts::var(crate::opt!("exit-on-fatal")).map_or(true, |v| v != "0");
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
    // VIS-2: the vision tower (TP=1; TP > 1 serves images from VIS-4). Loaded when config.json declares
    // a vision_config and the text config an interleaved mrope; a load failure serves text-only with a
    // visible notice (image requests then answer 400).
    // VIS-4: the head (TP=1, or rank 0) runs the tower; a node only needs the mrope section (it splices
    // the rows the head ships).
    let (vision_tower, vision_gpu, mrope_section) = if tp_rank.map_or(true, |r| r == 0) {
        vision_boot(model_dir)
    } else {
        let sec = mrope_section_of(model_dir);
        println!("[exl3-serve] vision (node): mrope section {sec:?}; image rows arrive from the head");
        (None, None, sec)
    };
    // CF-P1d: every lane's KV, the scratch, graphs and the vision tower now exist — `--ple-ram auto` decides RAM vs
    // SSD for the n-gram table on the memory that is actually left (each rank for itself; identical output either way).
    model.ple_promote_auto()?;
    let spec_policy = SpecLanes::parse(arg(args, "--spec-lanes-max"))?;
    // CF-FCFS: --lane-order reaches every rank through the SPMD option registry (Scope::Spmd), so
    // the head's value is installed on the nodes before the scheduler is built. The `arg()` read is
    // the TP=1 path's own value; under TP both ranks see the head's shipped value (identical).
    let lane_order = LaneOrder::parse(arg(args, "--lane-order")
                                      .map(str::trim)
                                      .filter(|v| !v.is_empty()))?;
    let lane_quantum = match arg(args, "--lane-quantum") {
        None => 256,
        Some(v) if v.trim().is_empty() => 256,
        Some(v) => v.trim().parse::<usize>().map_err(|e| anyhow::anyhow!("--lane-quantum: {e}"))?,
    };
    anyhow::ensure!(lane_quantum > 0, "--lane-quantum must be > 0");
    let interleave: usize = arg(args, "--prefill-interleave").and_then(|v| v.parse().ok()).unwrap_or(0);
    if interleave > 0 {
        println!("[exl3-serve] prefill interleave = {interleave} decode step(s) per prefill chunk (--prefill-interleave; skipped under TP)");
    }
    if width > 1 && model.mtp.is_some() {
        println!("[exl3-serve] load-adaptive batching (--spec-lanes-max): {} (--max-batch {width})", match spec_policy {
            SpecLanes::Auto => "auto — each round picks serial speculation or ONE shared plain step by estimated aggregate tokens/s".to_string(),
            SpecLanes::Over(k) => format!("one shared plain step with more than {k} busy lanes, serial speculation otherwise"),
            SpecLanes::Never => "off — always serial speculative rounds".to_string(),
        });
    }
    let sched = Exl3Scheduler {
        model: model.clone(), sc, width,
        lanes: (0..width).map(|_| None).collect(), eos: eos.clone(), chunk, interleave, psc: Some(psc0),
        mtp_k, spec_policy, lane_order, lane_quantum, admit_ctr: 0, fcfs_rots: 0,
        shared_scale: 1.0, shared_last: false, shared_streak: 0, policy_logs: 0, ctrl_round: 0, ema_plain: 0.0, draft_conf, cal: DraftCal::new(),
        prefix_on, cache: (0..width).map(|_| None).collect(),
        wp16, wp16_xcheck,
        loop_cfg, pen_range, spec_ratio, draft_temp, gate: None, tok: tok.clone(),
        tp: tp_rank.map(|r| TpSync::new(r, tp_world as i32)),
        exit_on_fatal, fatal: None, inject_panic, steps: 0, last_resume: (0, 0), mrope_section, vis_seam: None,
    };
    if width > 1 && model.mtp.is_some() {
        println!("[exl3-serve] serial lane order (--lane-order): {}{}", lane_order.name(),
                 if lane_order == LaneOrder::Fcfs {
                     format!(" — run to completion, one lane at a time, FCFS by admission order, \
                              quantum {} generated tokens (--lane-quantum)", lane_quantum)
                 } else {
                     " — every busy lane one round per scheduler step, in slot order (unchanged)".to_string()
                 });
    }
    println!("[exl3-serve] liveness: sticky CUDA error / scheduler thread crash -> {}",
             if exit_on_fatal { "exit 70 (--exit-on-fatal, default)" } else { "engine DEAD, /health 503 (--exit-on-fatal off)" });
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
    // v0.7.3 (#8.3): --model-name (alias --served-model-name) wins verbatim; the default
    // is the model card's `base_model:` line, else the directory name — the same
    // resolution order as the NVFP4 server (server::resolve_model_name).
    let model_name = crate::server::resolve_model_name(
        arg(args, "--model-name"), arg(args, "--served-model-name"), model_dir);
    crate::metrics::set_max_batch(width);
    // v0.7.3 ("tel"): the sink lives ONLY where the HTTP API lives — TP=1, or TP rank 0
    // (the head). A node never builds one: it has no HTTP hooks to fill the ring and no
    // sender to drain it, so under TP only the head emits. `--otel-model-id` /
    // `--otel-topology` override the auto values inside OtelSink::new.
    let otel = match (&otel_cfg, tp_rank) {
        (Some(cfg), None) | (Some(cfg), Some(0)) => Some(crate::otel::OtelSink::new(
            cfg.clone(), &model_name,
            // tp_rank (not tp — that was moved into the load) still tells single vs TP.
            &crate::otel::topology_from_world(if tp_rank.is_some() { Some(tp_world) } else { None }))),
        _ => None,
    };
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
        vision_tower,
        vision_gpu,
        vision_cpu: false,
        stop_ids: eos,
        otel,
    };
    Ok(ServeParts { sched, state, port, cpu_aff, exit_on_fatal, rx: srx })
}

/// VIS-2: the text config's interleaved mrope_section (None = the model declares no interleaved mrope).
fn mrope_section_of(model_dir: &str) -> Option<[usize; 3]> {
    crate::vision_tower::vision_geometry(model_dir).ok()??;
    let raw = std::fs::read_to_string(std::path::Path::new(model_dir).join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let rp = v.get("text_config").unwrap_or(&v).get("rope_parameters")?;
    if rp.get("mrope_interleaved").and_then(|x| x.as_bool()) != Some(true) { return None; }
    let a = rp.get("mrope_section")?.as_array()?;
    if a.len() != 3 { return None; }
    Some([a[0].as_u64()? as usize, a[1].as_u64()? as usize, a[2].as_u64()? as usize])
}

/// VIS-2: load the vision tower for the EXL3 server (CPU weights + the f32 GPU tower) and the text
/// config's interleaved mrope_section. Any failure = text-only with a printed reason.
#[allow(clippy::type_complexity)]
fn vision_boot(model_dir: &str) -> (Option<Arc<crate::vision_tower::VisualTower>>,
                                    Option<Arc<std::sync::Mutex<crate::vision_gpu::GpuVisualTower>>>,
                                    Option<[usize; 3]>) {
    match crate::vision_tower::vision_geometry(model_dir) {
        Ok(Some(_)) => {}
        Ok(None) => return (None, None, None),
        Err(e) => { println!("[exl3-serve] vision: geometry probe failed ({e:#}) — text-only"); return (None, None, None); }
    }
    let Some(sec) = mrope_section_of(model_dir) else {
        println!("[exl3-serve] vision: the text config declares no interleaved mrope_section — text-only");
        return (None, None, None);
    };
    let t0 = std::time::Instant::now();
    let tower = match crate::vision_tower::VisualTower::load(model_dir) {
        Ok(t) => Arc::new(t),
        Err(e) => { println!("[exl3-serve] vision: tower load failed ({e:#}) — text-only"); return (None, None, None); }
    };
    let gpu = match cudarc::driver::CudaDevice::new(0).map_err(anyhow::Error::from)
        .and_then(|d| crate::vision_gpu::GpuVisualTower::new(d, &tower)) {
        Ok(g) => g,
        Err(e) => { println!("[exl3-serve] vision: GPU tower unavailable ({e:#}) — text-only"); return (None, None, None); }
    };
    let d = tower.dims;
    println!("[exl3-serve] vision ON: tower from {}: {} blocks x {} wide -> {} ({:.1} s); images resized to a longer side <= {} px \
              (--image-max-edge), mrope section {sec:?}; QSA indexer pooled keys follow the image positions (VIS-3)",
             tower.source, d.depth, d.hidden, d.out_hidden, t0.elapsed().as_secs_f64(),
             if tower.preproc.max_edge == 0 { "unlimited".to_string() } else { tower.preproc.max_edge.to_string() });
    (Some(tower), Some(Arc::new(std::sync::Mutex::new(gpu))), Some(sec))
}

/// TP-D: the host-thread mask at TP — the big cores minus the launch core (9, the pinned scheduler
/// thread) and the RDMA proxy core (19). TP=1: the resolved mask unchanged.
fn tp_worker_mask(m: &Option<Vec<usize>>, tp: bool, world: i32) -> Option<Vec<usize>> {
    let m = m.clone()?;
    if !tp { return Some(m); }
    // TP-F: the dual-rail prefill transport pins its second proxy on TP_AUX_PROXY_CORE (18).
    let aux = if crate::exl3_forward::xtp::prefill_xport_mode() == 2 { crate::exl3_forward::xtp::TP_AUX_PROXY_CORE as usize } else { usize::MAX };
    // TP-4F2: `--tp-reduce single` pins a third proxy (the single-stage reduce ctx) on TP_SINGLE_PROXY_CORE (17).
    let single = if crate::exl3_forward::xtp::tp_reduce_single_for(world) { crate::exl3_forward::xtp::TP_SINGLE_PROXY_CORE as usize } else { usize::MAX };
    let f: Vec<usize> = m.into_iter().filter(|&c| c != 9 && c != 19 && c != aux && c != single).collect();
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
    // v0.7.3 ("tel"): the ONE sender task rides THIS runtime (serve_http is head/TP=1 only).
    let otel_sink = state.otel.clone();
    // HOST / RO-7: pin THIS thread (it runs the HTTP runtime below) to the big cores BEFORE the
    // scheduler thread is spawned — a new thread inherits its creator's mask (Linux), and so do
    // the threads either of them spawns later (prefill PLE workers, tokio's blocking pool).
    println!("[exl3-serve] {}", cpu_aff.1);
    let mask = tp_worker_mask(&cpu_aff.0, tp_ctl.is_some(), tp_ctl.as_ref().map_or(1, |c| c.len() as i32 + 1));
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
                    crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7
                }
                set_worker_mask(mask);
                let r = sched.run_tp_head(srx, streams);
                match r {
                    Ok(()) => eprintln!("[exl3-serve] TP head: request channel closed — session over"),
                    Err(e) => eprintln!("\n*** FATAL: the TP scheduler failed: {e:#}. Exiting (70). ***\n"),
                }
                std::thread::sleep(std::time::Duration::from_millis(300)); // let the error events flush
                crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7 + LR-1
            });
        }
    }
    let rt = tokio::runtime::Builder::new_current_thread().enable_all()
        .build().context("tokio runtime")?;
    rt.block_on(async move {
        // v0.7.3 ("tel"): run_sender is fully async (timed connect/write/read) — a slow or
        // absent receiver delays only itself; decode and SSE serving continue (the design
        // contract: ring, drop-on-full, one task off the compute path, zero cost when off).
        if let Some(sink) = otel_sink { tokio::spawn(crate::otel::run_sender(sink)); }
        let app = create_router(state);
        let listener = crate::server::http_listen(port).await;
        println!("[exl3-serve] listening on {} (model id: {model_name})",
                 listener.local_addr().map_or_else(|_| format!("port {port}"), |a| a.to_string()));
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
            let (tx, rx, bl) = crate::server::tok_channel(); // LR-5 (H6): drained counted below
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
            rxs.push((rx, bl));
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
        for (mut rx, bl) in rxs {
            let (mut toks, mut reason) = (Vec::with_capacity(positions), None);
            while let Some(ev) = crate::server::try_recv_counted(&mut rx, &bl) { // LR-5 (H6)
                match ev {
                    TokEvent::Tok(x) => toks.push(x),
                    TokEvent::Finish { reason: r } => reason = Some(r),
                    // #8.2: admission telemetry — no action for a gate drain.
                    TokEvent::Admitted { .. } => {}
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
        lanes: (0..width).map(|_| None).collect(), eos, chunk, interleave: 0, psc: Some(psc0),
        mtp_k, spec_policy: SpecLanes::Never, lane_order: LaneOrder::Rr, lane_quantum: 256, admit_ctr: 0, fcfs_rots: 0, shared_scale: 1.0, shared_last: false, shared_streak: 0, policy_logs: 0, ctrl_round: 0, ema_plain: 0.0, draft_conf, cal: DraftCal::new(),
        prefix_on: true, cache: (0..width).map(|_| None).collect(),
        wp16: None, wp16_xcheck: false,
        loop_cfg: None, pen_range: 1024, spec_ratio: false, draft_temp, gate: None, tok: tok.clone(), tp: None,
        exit_on_fatal: false, fatal: None, inject_panic: None, steps: 0, last_resume: (0, 0), mrope_section: None, vis_seam: None,
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
    let (tx, mut rx, bl) = crate::server::tok_channel(); // LR-5 (H6): drained counted below
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
    while let Some(ev) = crate::server::try_recv_counted(&mut rx, &bl) { // LR-5 (H6)
        match ev {
            TokEvent::Tok(x) => toks.push(x),
            TokEvent::Finish { reason: r } => reason = r,
            // #8.2: admission telemetry — no action for a spec gate drain.
            TokEvent::Admitted { .. } => {}
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

/// TP-4D: fold the world > 2 end-of-program hub result: `all[r] = [rank r's verdict (1 = OK), rank r's agree count]`.
/// Ok only when EVERY rank reports OK and every rank's agree count equals this rank's (the same lockstep points were
/// crossed everywhere). The line names each rank, so one failing rank of four is visible in every rank's log.
fn tpspec_fold(all: &[Vec<u32>], my_rank: usize, agrees: u64) -> (bool, String) {
    let mut ok = true;
    let mut parts = Vec::new();
    for (r, f) in all.iter().enumerate() {
        let good = f[0] != 0 && f[1] as u64 == agrees;
        ok &= good;
        parts.push(format!("rank {r}{} {} (agrees {})", if r == my_rank { " (this)" } else { "" },
                           if f[0] != 0 { "OK" } else { "FAIL" }, f[1]));
    }
    (ok, format!("{} | this rank agrees {agrees}", parts.join(" | ")))
}

/// The spec program (see the section comment). `ctx` = Some on both TP ranks (the head passes
/// `prompts`, the node None — they arrive over the link); None = TP=1 (prompts required).
pub fn run_spec(model_dir: &str, ctx: Option<crate::tp::TpContext>, prompts: Option<Vec<(String, Vec<u32>)>>,
                max_pos: usize, o: &SpecOpts) -> Result<()> {
    use sha2::{Digest, Sha256};
    let (prompts, attach, rank, world) = match ctx {
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
            (prompts, Some(crate::exl3_forward::xtp::TpAttach { rank, world, link }), rank, world)
        }
        None => (prompts.context("spec (TP=1): no prompts")?, None, 0, 1),
    };
    let tp_on = attach.is_some();
    // "TP=2 rank N" at world 2 (the label the gate scripts grep), "TP=4 rank N" at world 4
    let who = if tp_on { format!("TP={world} rank {rank}") } else { "TP=1".to_string() };
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
        lanes: (0..width).map(|_| None).collect(), eos: eos.clone(), chunk, interleave: 0, psc: Some(psc0),
        mtp_k, spec_policy: SpecLanes::Never, lane_order: LaneOrder::Rr, lane_quantum: 256, admit_ctr: 0, fcfs_rots: 0, shared_scale: 1.0, shared_last: false, shared_streak: 0, policy_logs: 0, ctrl_round: 0, ema_plain: 0.0, draft_conf: 0.4, cal: DraftCal::new(),
        prefix_on: o.prefix, cache: (0..width).map(|_| None).collect(),
        wp16: None, wp16_xcheck: false,
        loop_cfg: Some((300, 3)), pen_range: 1024, spec_ratio: true, draft_temp: None, gate: None, tok: tok.clone(),
        exit_on_fatal: false, fatal: None, inject_panic: None, steps: 0, last_resume: (0, 0), mrope_section: None, vis_seam: None,
        tp: if tp_on { Some(TpSync::new(rank, world)) } else { None },
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
        if t.world > 2 {
            // TP-4D: world > 2 — every rank's verdict + agree count through the head hub (unbounded: the ranks
            // finish the gate's last rep at different times); every rank prints the same per-rank table
            let all = crate::net::exchange_u32s_all(&[all_ok as u32, t.agrees as u32], 4)?;
            let (ok, line) = tpspec_fold(&all, t.rank as usize, agrees);
            println!("[tpspec] {who}: {line} (graphed digests {ident_bufs} compared, {ident_bad} differ)");
            all_ok &= ok;
        } else {
            let peer = crate::net::exchange_u32s(&[all_ok as u32, t.agrees as u32], 4)?;
            println!("[tpspec] {who}: this rank {} | peer {} (peer agrees {}, this rank {agrees}; graphed digests {ident_bufs} compared, {ident_bad} differ)",
                     if all_ok { "OK" } else { "FAIL" }, if peer[0] != 0 { "OK" } else { "FAIL" }, peer[1]);
            all_ok &= peer[0] != 0 && peer[1] as u64 == agrees;
        }
    }
    println!("RESULT: {} ({who}; G-T1-e {}{}{})", if all_ok { "TPSPEC_OK" } else { "TPSPEC_FAIL" },
             if e_ok { "OK" } else { "MISMATCH" },
             if o.samp > 0 { if samp_ok { " ; G-T1-h OK" } else { " ; G-T1-h FAIL" } } else { "" },
             if tp_on { if ident_bad == 0 { " ; RANK_IDENTITY_OK (graphed)" } else { " ; RANK_IDENTITY_FAIL" } } else { "" });
    anyhow::ensure!(all_ok, "spec program gates failed ({who})");
    Ok(())
}

/// TP-4D: CPU models of every world > 2 lockstep op of the serve path — the REAL `TpSync::*_hub` functions and the
/// REAL `net::hub_all` over the in-memory `net::hub_mock` transport, one thread per rank (4 ranks). What these prove:
/// the merged hub ops are symmetric (every rank runs the same op sequence and reaches the same verdict), counters
/// stay in lockstep, nodes adopt the head's words, a divergent / dead / hung rank fails EVERY live rank (none is left
/// parked), and the WP16 allocation agree keeps four checkpoint stores identical. What they do NOT prove: anything
/// about RDMA placement, the proxy, the hot-path rings or the GPU (hardware questions for the W=4 gate run).
#[cfg(test)]
mod tp_hub_tests {
    use super::*;
    use crate::net::hub_mock::{run_ranks, Net};
    use std::time::Duration;

    const W: usize = 4;

    fn sync(rank: usize) -> TpSync { TpSync::new(rank as i32, W as i32) }

    #[test]
    fn step_go_hub_adopts_head_words_and_advances_counters_in_lockstep() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let mut got = Vec::new();
            for round in 0..5u64 {
                let (events, local_step) = if r == 0 { ((round % 3) as usize, 100 + round) } else { (0, 100 + round) };
                got.push(t.step_go_hub(hub, local_step, events).expect("a healthy step-go"));
            }
            (got, t.step)
        });
        for (r, (got, step)) in out.iter().enumerate() {
            assert_eq!(*step, 5, "rank {r}: one lockstep counter tick per step-go");
            let want: Vec<(u64, usize)> = (0..5u64).map(|i| (100 + i, (i % 3) as usize)).collect();
            assert_eq!(got, &want, "rank {r} adopts the head's (step, events)");
        }
        assert!(net.clobbers().is_empty(), "control ring clobbers: {:?}", net.clobbers());
    }

    #[test]
    fn a_rank_one_lockstep_point_ahead_fails_every_rank_and_names_it() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            if r == 2 { t.step += 1; }
            t.step_go_hub(hub, 7, 0).map_err(|e| format!("{e:#}"))
        });
        for (r, o) in out.iter().enumerate() {
            let e = o.as_ref().expect_err(&format!("rank {r} must fail: the ranks are at different lockstep points"));
            assert!(e.contains("step-go") && e.contains("FAILED") && e.contains("rank 2 counter field"), "rank {r}: {e}");
        }
    }

    #[test]
    fn pre_verify_hub_compares_width_drafts_and_launches_on_every_rank() {
        let net = Net::new(W, true);
        let ok = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let res = t.pre_verify_hub(hub, 6, &[1, 2, 3, 4, 5], 7);
            (res.is_ok(), t.pre_agrees)
        });
        assert!(ok.iter().all(|&(good, n)| good && n == 1), "{ok:?}");
        for (what, r_bad) in [("drafts", 3usize), ("width", 1), ("launched", 2)] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                let (w, d, l) = match (what, r == r_bad) {
                    ("drafts", true) => (6, vec![1, 2, 3, 4, 6], 7),
                    ("width", true) => (5, vec![1, 2, 3, 4, 5], 7),
                    ("launched", true) => (6, vec![1, 2, 3, 4, 5], 8),
                    _ => (6, vec![1, 2, 3, 4, 5], 7),
                };
                t.pre_verify_hub(hub, w, &d, l).map_err(|e| format!("{e:#}"))
            });
            for (r, o) in out.iter().enumerate() {
                let e = o.as_ref().expect_err(&format!("{what}: rank {r} must fail on a divergent rank {r_bad}"));
                assert!(e.contains("pre-verify") && e.contains(&format!("rank {r_bad}")), "{what} rank {r}: {e}");
            }
        }
    }

    #[test]
    fn round_hub_ships_head_timing_and_reports_ident_digests() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            t.ident = true;
            let words = [3u32, 1, 4, 1, 5];
            // rank 3's second digest differs: counted, but it is diagnostic (the round itself stays OK)
            let res = t.round_hub(hub, "verify", 3, 5, &words, if r == 0 { 12.5 } else { 99.0 }, 20.0, true, 1000 + r as u64,
                                  |m| { assert_eq!(m, 5); Ok(vec![0xAA, if r == 3 { 0xBC } else { 0xBB }, 0xCC]) });
            (res.expect("a healthy round"), t.agrees, t.ident_bufs, t.ident_bad, t.step)
        });
        for (r, (res, agrees, bufs, bad, step)) in out.iter().enumerate() {
            assert_eq!(*res, (12.5, 20.0, true), "rank {r} gets the head's timing, not its own");
            assert_eq!((*agrees, *bufs, *bad, *step), (1, 3, 1, 1), "rank {r}");
        }
    }

    #[test]
    fn round_hub_fails_every_rank_on_a_divergent_accept_count() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            t.round_hub(hub, "verify", if r == 1 { 2 } else { 3 }, 5, &[1, 2, 3], 1.0, 1.0, false, 0, |_| Ok(vec![]))
                .map(|_| ()).map_err(|e| format!("{e:#}"))
        });
        for (r, o) in out.iter().enumerate() {
            let e = o.as_ref().expect_err(&format!("rank {r} must fail"));
            assert!(e.contains("rank 1 accept count"), "rank {r}: {e}");
        }
    }

    #[test]
    fn a_node_that_dies_fails_every_live_rank_and_a_hung_one_trips_the_deadline() {
        // dead node: rank 2 "crashes" before its step-go (its thread ends: the mock's dead-peer probe fires)
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            if r == 2 { return Err("node crashed".to_string()); }
            sync(r).step_go_hub(hub, 9, 1).map(|_| ()).map_err(|e| format!("{e:#}"))
        });
        for (r, o) in out.iter().enumerate() {
            let e = o.as_ref().expect_err(&format!("rank {r}: a dead rank 2 must fail the op"));
            if r != 2 { assert!(e.contains("step-go") && e.contains("FAILED"), "rank {r}: {e}"); }
        }
        assert!(net.abort_code(0) == 10 || net.abort_code(0) == 0, "the head saw the dead peer (code 10) or a failed verdict");
        // hung node: alive but never posts; the head's deadline fires (code 12) and every other rank fails too
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_millis(300)), |r, hub| {
            if r == 3 { std::thread::sleep(Duration::from_millis(1500)); return Err("hung".to_string()); }
            sync(r).step_go_hub(hub, 9, 1).map(|_| ()).map_err(|e| format!("{e:#}"))
        });
        for r in 0..3 { assert!(out[r].is_err(), "rank {r} must not stay parked behind a hung rank 3"); }
        assert_eq!(net.abort_code(0), 12, "the head's wait hit the lockstep deadline");
    }

    #[test]
    fn checkpoint_allocation_outcome_is_agreed_so_four_stores_stay_identical() {
        use crate::exl3_forward::CkptStore;
        // rank 2's third allocation fails (its device memory is tighter); the others would succeed.
        let local = |r: usize, n_before: usize| -> Result<u32> {
            if r == 2 && n_before >= 2 { anyhow::bail!("CUDA_ERROR_OUT_OF_MEMORY (model)") } else { Ok(n_before as u32 + 1) }
        };
        // WITHOUT the agree (today's world-2 behaviour): the stores diverge — the C2 bug, reproduced
        let plain: Vec<(usize, usize)> = (0..W).map(|r| {
            let mut st: CkptStore<u32> = CkptStore::new(1, 8);
            st.begin_aligned(0, 0, 2048);
            for i in 0..5 {
                let n = st.n_alloc;
                if let Some(b) = st.acquire(|| local(r, n)) { st.insert(0, 2048 * (i + 1), b); }
            }
            (st.n_alloc, st.live())
        }).collect();
        assert!(plain.iter().any(|&p| p != plain[0]), "without the agree rank 2 freezes at 2 while the others grow: {plain:?}");
        // WITH the agree: every rank's store ends identical
        let net = Net::new(W, true);
        let agreed = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let mut st: CkptStore<u32> = CkptStore::new(1, 8);
            st.begin_aligned(0, 0, 2048);
            let mut hub_err = None;
            for i in 0..5 {
                let n = st.n_alloc;
                let got = st.acquire(|| { let l = local(r, n); t.alloc_outcome(hub, n, l, &mut hub_err) });
                assert!(hub_err.is_none(), "rank {r}: a healthy hub");
                if let Some(b) = got { st.insert(0, 2048 * (i + 1), b); }
            }
            (st.n_alloc, st.live(), st.positions(0))
        });
        assert!(agreed.iter().all(|a| *a == agreed[0]), "the four stores must be identical: {agreed:?}");
        assert_eq!(agreed[0].0, 2, "the cap froze at the two buffers every rank could allocate");
        assert!(net.clobbers().is_empty());
    }

    #[test]
    fn a_failed_alloc_agree_is_reported_not_swallowed_as_an_allocation_failure() {
        use crate::exl3_forward::CkptStore;
        // rank 3 dies before the allocation agree: the live ranks' `acquire` would freeze the cap on the Err, so the
        // transport failure must come back through `hub_err`
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            if r == 3 { return None; }
            let mut t = sync(r);
            let mut st: CkptStore<u32> = CkptStore::new(1, 8);
            let mut hub_err = None;
            let _ = st.acquire(|| t.alloc_outcome(hub, 0, Ok(1u32), &mut hub_err));
            hub_err.map(|e| format!("{e:#}"))
        });
        for r in 0..3 {
            let e = out[r].as_ref().unwrap_or_else(|| panic!("rank {r}: the agree failure must surface in hub_err"));
            assert!(e.contains("wp16 alloc"), "rank {r}: {e}");
        }
    }

    #[test]
    fn boot_hash_verdict_names_every_differing_rank() {
        let ok: Vec<Vec<u32>> = (0..W).map(|_| vec![BOOT_AGREE_MAGIC, 0x1234]).collect();
        assert_eq!(boot_hash_verdict(&ok, 1), Ok(()));
        let mut bad = ok.clone();
        bad[2][1] = 0x9999;
        bad[3][0] = 0x6000_0001;
        let e = boot_hash_verdict(&bad, 1).unwrap_err();
        assert!(e.contains("rank 2 hash 00009999 != head 00001234") && e.contains("rank 3 sent frame class") && e.contains("this is rank 1"), "{e}");
        assert!(!e.contains("rank 1 hash") && !e.contains("rank 0"), "{e}");
    }

    #[test]
    fn tpspec_fold_requires_every_rank_ok_with_equal_agree_counts() {
        let good: Vec<Vec<u32>> = (0..W).map(|_| vec![1, 77]).collect();
        let (ok, line) = tpspec_fold(&good, 2, 77);
        assert!(ok && line.contains("rank 2 (this) OK (agrees 77)"), "{line}");
        let mut one_fail = good.clone();
        one_fail[3][0] = 0;
        let (ok, line) = tpspec_fold(&one_fail, 0, 77);
        assert!(!ok && line.contains("rank 3 FAIL"), "{line}");
        let mut skew = good.clone();
        skew[1][1] = 76;
        let (ok, _) = tpspec_fold(&skew, 0, 77);
        assert!(!ok, "a rank that crossed a different number of lockstep points must fail the fold");
    }
}

#[cfg(test)]
mod fit_tests {
    use super::{fit_parts, fit_plan, FitParts};
    const G: u64 = 1_000_000_000;

    #[test]
    fn fit_plan_refuses_with_the_arithmetic_and_the_way_out() {
        let p = FitParts { weights: 53 * G, table: 33 * G, per_lane: 3 * G, fixed: 19 * G, floor: 5 * G };
        // 1 lane: fits with the table in RAM.
        assert_eq!(fit_plan(p, 1, 124 * G, "auto", 131072), Ok(true));
        // 8 lanes: fits only on SSD; auto serves it, ram is refused and told to use ssd.
        assert_eq!(fit_plan(p, 8, 124 * G, "auto", 131072), Ok(false));
        let e = fit_plan(p, 8, 124 * G, "ram", 131072).unwrap_err();
        assert!(e.contains("--ple-ram ssd") && e.contains("does not fit"), "{e}");
        // 20 lanes: does not fit even on SSD; names the lane and context limits.
        let e = fit_plan(p, 20, 124 * G, "auto", 131072).unwrap_err();
        assert!(e.contains("even with the n-gram table on SSD") && e.contains("--max-batch 15"), "{e}");
        // the weights alone overflow: says TP=2.
        let e = fit_plan(FitParts { weights: 110 * G, ..p }, 1, 124 * G, "ssd", 131072).unwrap_err();
        assert!(e.contains("--tp 2"), "{e}");
    }

    /// The real Flash-Next 3.05 pack, where present: the estimate must bracket the 2026-10-02 measurement on .14
    /// (per lane at 131K ~2.8 GB measured; device weights ~53 GB; table 32.6 GB).
    #[test]
    fn flash_next_305_estimate_brackets_the_measurement() {
        let dir = std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join("models/Qwen3.8-Flash-Next-exl3-3.05bpw");
        if !dir.is_dir() { return; }
        let p = fit_parts(dir.to_str().unwrap(), 131073, 1, true, 1.0625, 4.0).unwrap();
        assert!((2_600_000_000..3_300_000_000).contains(&p.per_lane), "{p:?}");
        assert!((50 * G..56 * G).contains(&p.weights), "{p:?}");
        assert!((32 * G..34 * G).contains(&p.table), "{p:?}");
    }
}

#[cfg(test)]
mod spec_lanes_tests {
    use super::{fcfs_charge, fcfs_front, shared_step_prior_ms, LaneOrder, SpecLanes};

    /// CF-FCFS: the ordering function is a pure function of (admit_seq, credit) — no GPU, no
    /// scheduler. The credit/rotation bookkeeping itself is `fcfs_charge`, exercised by
    /// `fcfs_simulation_three_lanes` below, which drives this same pair exactly as step() does.
    #[test]
    fn fcfs_order_is_pure() {
        // (slot, admit_seq, fcfs_credit)
        let q = 4;
        // Two lanes admitted in order, both fresh: the OLDEST (lowest admit_seq) is the front, not
        // the lowest slot.
        let two = [(0usize, 0u64, 0usize), (1, 1, 0)];
        assert_eq!(fcfs_front(&two, q), Some(0));
        // Slot order and admission order disagree: admission order wins.
        let swapped = [(0usize, 7u64, 0usize), (1, 3, 0)];
        assert_eq!(fcfs_front(&swapped, q), Some(1));
        // A lane whose credit reached the quantum (as fcfs_charge leaves it: credit 0 AND the
        // HIGHEST admit_seq after rotation) would only be the front if it were the sole lane.
        let one_left = [(0usize, 0u64, 0usize)];
        assert_eq!(fcfs_front(&one_left, q), Some(0));
        // The front finishes (its lane is gone from the list) -> the next-oldest runs at once.
        let finished = [(1usize, 1u64, 0)];
        assert_eq!(fcfs_front(&finished, q), Some(1));
        // A new arrival does NOT jump the queue: it has the highest admit_seq.
        let arrived = [(0usize, 0u64, 0), (1, 1, 0), (2, 2, 0)];
        assert_eq!(fcfs_front(&arrived, q), Some(0));
        // A lane with a partially used credit is still the front (its quantum is not exhausted).
        let partial = [(0usize, 0u64, 3), (1, 1, 0)];
        assert_eq!(fcfs_front(&partial, q), Some(0));
        // No capable lane at all (all MTP-off) -> the caller runs the plain batch only.
        assert_eq!(fcfs_front(&[], q), None);
        // Defensive guard on an unreachable state (charge resets credit AT the quantum): still
        // answers, never panics.
        let all_done = [(0usize, 0u64, q), (1, 1, q + 3)];
        assert_eq!(fcfs_front(&all_done, q), None);
        // Purity: the same input always gives the same front, whatever the slot order.
        let a = [(5usize, 2u64, 0), (3, 9, 0), (7, 4, q)];
        let b = [(7usize, 4u64, q), (5, 2, 0), (3, 9, 0)];
        assert_eq!(fcfs_front(&a, q), fcfs_front(&b, q));
        assert_eq!(fcfs_front(&a, q), Some(5));
        // A huge quantum never rotates (one lane runs to completion).
        assert_eq!(fcfs_front(&two, 1 << 20), Some(0));
    }

    /// CF-FCFSB: a multi-step simulation that drives the SAME bookkeeping pair step() uses —
    /// `fcfs_front` to pick, `fcfs_charge` to charge and rotate — over 3 lanes whose turns are
    /// longer than the quantum, one lane finishing mid-turn and a new arrival. Asserts the actual
    /// SERVICE ORDER (who steps at each scheduler step): a lane that used its quantum goes BEHIND
    /// the others, and the old front only gets its next turn after every waiting lane had one.
    /// (Before the fix this test fails: the rotated lane kept the lowest admit_seq and stepped
    /// forever — the order degenerated to [0,0,0,...].)
    #[test]
    fn fcfs_simulation_three_lanes() {
        let q = 4usize;
        // (slot, admit_seq, credit, tokens_left). Lanes 0 and 1 have 10-token turns, lane 2 a
        // 5-token turn; each speculative round generates 2 tokens. Quantum 4 => 2 consecutive
        // rounds per turn, then rotate.
        let mut lanes: Vec<Option<(usize, u64, usize, usize)>> =
            vec![Some((0, 0, 0, 10)), Some((1, 1, 0, 10)), Some((2, 2, 0, 5))];
        let mut admit_ctr = 3u64;
        let mut rotations = 0usize;
        let mut service: Vec<usize> = Vec::new();
        let mut lane3_admitted = false;
        for _step in 0..100 {
            // Lane 3 is admitted once lane 2 has finished (the scheduler admits into a free slot).
            if !lane3_admitted && lanes[2].is_none() {
                lanes[2] = Some((2, admit_ctr, 0, 10));
                admit_ctr += 1;
                lane3_admitted = true;
            }
            let capable: Vec<(usize, u64, usize)> = lanes
                .iter()
                .filter_map(|l| l.as_ref().map(|(s, seq, c, _)| (*s, *seq, *c)))
                .collect();
            let Some(front) = fcfs_front(&capable, q) else { break };
            let lane = lanes
                .iter_mut()
                .find_map(|l| l.as_mut().filter(|(s, ..)| *s == front))
                .unwrap();
            let delta = lane.3.min(2);
            lane.3 -= delta;
            let (mut seq, mut credit) = (lane.1, lane.2);
            if fcfs_charge(&mut seq, &mut credit, delta, q, &mut admit_ctr) {
                rotations += 1;
            }
            lane.1 = seq;
            lane.2 = credit;
            service.push(front);
            if lane.3 == 0 {
                lanes.iter_mut().find(|l| matches!(l, Some((s, ..)) if *s == front)).unwrap().take();
            }
        }
        // Hand-derived from the bookkeeping above: two rounds per quantum, then rotate; lane 2's
        // last round is a 1-token partial (credit 1, no rotation); lanes 0/1 finish with credit 2.
        // The rotated lane is served again ONLY after both other lanes had their turns; the new
        // arrival (lane 3, admitted into slot 2) runs only after lanes 0 and 1 finished entirely.
        assert_eq!(service, [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2]);
        assert!(lane3_admitted);
        assert_eq!(rotations, 7); // 5 in the first two cycles + 2 from the lane-3 turn
        assert_eq!(admit_ctr, 11); // 3 initial + 7 rotation re-stamps + 1 admission stamp
    }

    #[test]
    fn lane_order_parse() {
        assert_eq!(LaneOrder::parse(None).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("")).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("rr")).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("round-robin")).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("fcfs")).unwrap(), LaneOrder::Fcfs);
        assert!(LaneOrder::parse(Some("fifo")).is_err());
        assert!(LaneOrder::parse(Some("RR")).is_err());
    }

    #[test]
    fn spec_lanes_parse() {
        assert_eq!(SpecLanes::parse(None).unwrap(), SpecLanes::Auto);
        assert_eq!(SpecLanes::parse(Some("auto")).unwrap(), SpecLanes::Auto);
        assert_eq!(SpecLanes::parse(Some("0")).unwrap(), SpecLanes::Never);
        assert_eq!(SpecLanes::parse(Some("off")).unwrap(), SpecLanes::Never);
        assert_eq!(SpecLanes::parse(Some("4")).unwrap(), SpecLanes::Over(4));
        assert!(SpecLanes::parse(Some("many")).is_err());
        assert!(SpecLanes::parse(Some("-1")).is_err());
    }

    /// The prior reproduces the 2026-10-02 TP=1 measurements within 5% (n = 4: 46.5, 8: 61.7, 16: 90 ms).
    #[test]
    fn shared_step_prior_matches_measurements() {
        for (n, ms) in [(4usize, 46.5f64), (8, 61.7), (16, 90.0)] {
            let p = shared_step_prior_ms(n);
            assert!((p - ms).abs() / ms < 0.05, "n={n}: {p} vs {ms}");
        }
    }
}


#[cfg(test)]
/// S1 (REL_V0_7_3 review): both spellings of every flag EXL3 reads with `arg()` must work.
/// The table-driven test drives the REAL parser through the REAL `arg()` — before the fix,
/// `--lane-order=fcfs` returned None (silently ignored) while `--print-config` showed it set.
mod s1_eq_aware_arg_tests {
    use super::*;

    const FLAGS: &[&str] = &[
        "--lane-order", "--lane-quantum", "--model-name", "--served-model-name",
        "--otel-endpoint", "--otel-model-id", "--otel-topology",
        "--port", "--max-seq-len", "--max-batch", "--prefix-cache", "--prefill-chunk",
        "--kv-cache", "--ple-ram", "--reasoning-effort", "--thinking", "--cpu-affinity",
        "--tune-table", "--tune-draft", "--tune-profile", "--draft-confidence",
        "--draft-temperature", "--spec-sampling", "--host",
    ];

    fn owned(args: &[&str]) -> Vec<String> { args.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn arg_accepts_both_spellings_of_every_flag() {
        for f in FLAGS {
            // space form
            let a = owned(&["serve", "/m", *f, "VALUE", "--after", "x"]);
            assert_eq!(arg(&a, f).as_deref(), Some("VALUE"), "{f} space form");
            // equals form
            let a = owned(&["serve", "/m", &format!("{f}=VALUE"), "--after", "x"]);
            assert_eq!(arg(&a, f).as_deref(), Some("VALUE"), "{f} = form (S1: was None)");
            // absent
            let a = owned(&["serve", "/m", "--other", "y"]);
            assert_eq!(arg(&a, f), None, "{f} absent");
        }
    }

    #[test]
    fn arg_equals_form_does_not_eat_the_next_token() {
        let a = owned(&["--lane-order=fcfs", "--lane-quantum", "4"]);
        assert_eq!(arg(&a, "--lane-order").as_deref(), Some("fcfs"));
        assert_eq!(arg(&a, "--lane-quantum").as_deref(), Some("4"));
        // a longer flag sharing a prefix must not match (the old strip bug class)
        let a = owned(&["--port=9000"]);
        assert_eq!(arg(&a, "--por"), None);
        assert_eq!(arg(&a, "--port").as_deref(), Some("9000"));
    }

    #[test]
    fn nvfp4_parse_arg_is_the_same_parser() {
        // the NVFP4-side reader is a thin delegate of the same shared fn; spot-check both
        // spellings through the EXL3 reader to prove ONE mechanism.
        let a = owned(&["--model-name=Qwen/Qwen3.8-Flash-Next"]);
        assert_eq!(arg(&a, "--model-name").as_deref(), Some("Qwen/Qwen3.8-Flash-Next"));
        let a = owned(&["--model-name", "Qwen/Qwen3.8-Flash-Next"]);
        assert_eq!(arg(&a, "--model-name").as_deref(), Some("Qwen/Qwen3.8-Flash-Next"));
    }
}
