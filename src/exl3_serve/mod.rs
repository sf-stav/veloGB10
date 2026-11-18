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
//!
//! File map (this is the former single `exl3_serve.rs`; every item keeps its `crate::exl3_serve::` path):
//! - `mod.rs`: shared types (`Lane`, `Exl3Scheduler`, `PackStats`, pool / prefill job types, `ServeParts`), pool constants, `arg`, and the test-module list.
//! - `builder.rs`: `build_serve`, the boot sequence. `serve.rs`: `run`, pack detection, vision boot, `serve_http`. `fit.rs`: memory fit and context clamp.
//! - `admission.rs` / `prefill.rs` / `pack_prefill.rs` / `pool.rs`: admit, chunked prefill, packed prefill, prefix pool and KV spill (`impl Exl3Scheduler`).
//! - `step.rs` / `scheduler.rs` / `mtp_round.rs` / `pack.rs` / `copy.rs`: the step wrapper, loop and plain step, the speculative round, lane-pack arms, copy-span arms.
//! - `lockstep.rs` / `tp_serve.rs`: the TP lockstep layer (`TpSync`, agree words) and the TP head / mirror loops with their entry points.
//! - `df2.rs` / `dds.rs` / `policy.rs` / `lanes.rs`: DFlash2 gating, draft-depth policy, lane order and arm constants, per-lane helpers (`impl Lane`).
//! - `wp24_gate.rs` / `spec_program.rs`: the served-level WP24 gate and the offline spec program.
//! - `<name>_tests.rs` and `review_p0_pins.rs`: the former inline unit-test modules (one file per module, declared at the end of this file).

mod admission;
mod builder;
mod copy;
mod dds;
mod df2;
mod fit;
mod lanes;
mod lockstep;
mod mtp_round;
mod pack;
mod pack_prefill;
mod policy;
mod pool;
mod prefill;
mod scheduler;
mod serve;
mod spec_program;
mod step;
mod tp_serve;
mod wp24_gate;

use crate::batch::{BatchRequest, TokEvent};
use crate::exl3_forward::{FwdModel, Scratch};
use crate::server::{create_router, AppState};
use crate::tokenizer::{QwenTokenizer, ThinkingMode};
use anyhow::{Context as _, Result};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;
use cudarc::driver::{DevicePtr as _, DeviceSlice as _};

use self::builder::*;
pub use self::copy::*;
pub use self::dds::*;
pub use self::df2::*;
pub use self::fit::*;
use self::lanes::*;
pub use self::lockstep::*;
use self::pack::*;
pub(crate) use self::policy::*;
use self::prefill::*;
pub use self::serve::*;
pub use self::spec_program::*;
pub use self::tp_serve::*;
pub use self::wp24_gate::*;

/// PFXPOOL: one prefix-pool entry's device buffer (the spill regions of its prompt, back to back) and the host PLE
/// n-gram pair captured with it.
pub(crate) struct PoolBuf {
    /// one slab: the entry's OWN positional rows `r0..r1` of every plane (PFXPOOL2: rows 0..r0 are its parent chain's,
    /// r0 = 0 for an entry that holds them all), then the recurrent snapshot planes at `snap_off`, then the carried
    /// checkpoints at `ck_off`, each `ck_bytes` long. Recycled through the scheduler's spare list when shifted out.
    /// None: freed by this rank alone under memory pressure (`pool_emergency`); the replicated policy drops the entry
    /// at the next admission (`pool_upkeep`), before anything can read it.
    buf: Option<cudarc::driver::CudaSlice<u8>>,
    hist: [i32; 2],
    r0: usize,
    r1: usize,
    snap_off: u64,
    ck_off: u64,
    ck_bytes: u64,
    /// the carried WP16 checkpoints over this prompt: (position, host PLE pair), in slab order; taken at chunk width
    /// `ckpt_c`. A prompt that shares only part of the entry (the next turn, a shorter prompt) resumes at one.
    ckpts: Vec<(usize, [i32; 2])>,
    ckpt_c: usize,
    /// how many of the entry's tokens were its lane's ALIGNED prefix (WP16 bookkeeping) at chunk width `ckpt_c`: a full
    /// restore adopts it, so the resumed run is tracked like an own-lane hit (C1 tail checkpoint, WP16 checkpoints)
    aligned: usize,
}

/// PFXPOOL: what a pool restore gave the prefill: the entry's whole prompt-end state (`Full(n)`: rows + recurrent
/// snapshot, resume at n like an own-lane hit) or a carried checkpoint installed as the lane's WP16 checkpoint
/// (`Ckpt(q)`: rows 0..q restored, the ordinary WP16 resume at q follows).
enum PoolHit { Full(usize, bool), Ckpt(usize) }

/// PFXPOOL: prompts shorter than this are not kept in the pool (reading them again costs about what a copy does).
const POOL_MIN_TOKENS: usize = 256;

/// PFXPOOL2: an entry reads its leading rows from an older entry when they share at least this many (fewer are
/// cheaper to keep than to chain), through at most this many ancestors (each one is a copy per plane at restore).
const POOL_SHARE_MIN: usize = 1024;

const POOL_MAX_DEPTH: usize = 24;

/// S1 (REL_V0_7_3): delegates to the ONE shared `=`-aware parser (`crate::arg_value`) so
/// `--lane-order=fcfs` and `--lane-order fcfs` behave identically on EXL3 (the old body
/// matched only the space form and silently ignored the `=` form).
/// TPSHIP (2026-11-07, AGENTS 1b-bis.8: fall back, do not refuse): the EXL3 engine's KV store formats are f32 / f16 /
/// fp8 / q8. bf16 (the NVFP4 trunk's KV format, which launch profiles carry over) has no EXL3 store; f16 is the nearest
/// (the same 2 bytes per value): serve f16, say so once, count it on /metrics. Anything else is still refused.
pub(crate) fn exl3_kv_cache_value(v: &str) -> anyhow::Result<&str> {
    if v == "bf16" {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            crate::metrics::exl3_kv_cache_fallback();
            crate::reprintln!("[exl3-serve] --kv-cache bf16: the EXL3 engine has no bf16 KV store; serving f16 instead \
(the same 2 bytes per value, the nearest format it has). Pass --kv-cache f16 (or q8) to silence this line.");
        });
        return Ok("f16");
    }
    anyhow::ensure!(matches!(v, "f32" | "f16" | "fp8" | "q8" | "q4k"), "--kv-cache must be f32, f16, fp8, q8 or q4k (bf16 is served as f16)");
    Ok(v)
}

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
    // THINKBUDGET1: vLLM's thinking_token_budget for this request (None = not sent, or the model has
    // no reasoning markers). Its room() clamps every speculative round's depth (round_depth) and its
    // filter() replaces the boundary token by the end marker at every emission point (the seam,
    // round_emit, step_plain). Pure function of replicated state: every TP rank decides the same.
    think: Option<crate::think_budget::ThinkBudget>,
    prompt_hash: u64,
    history: Vec<u32>,
    // WIDEVERIFY1: this request's copy-span proposer (None = --copy-span off / not armed), built at
    // admission over prompt + the seam token; `copy_new` = tokens emitted since its last commit (every
    // emission point appends; step_mtp commits them before the next proposal, a copy round right after
    // its own emit). st_copy_* = this request's copy rounds / accepted / emitted (the [mtp-stats] line).
    copy: Option<Box<crate::copy_span::CopySpan>>,
    copy_new: Vec<u32>,
    // COPYREC1 (--copy-span-record): the request's reply so far (the seam token first; every emission point appends)
    // and its prompt's last copy_record::OVERLAP ids; None = the record is off or read-only (nothing is kept).
    rec_reply: Option<Vec<u32>>,
    rec_tail: Vec<u32>,
    st_copy_rounds: usize,
    st_copy_accepted: usize,
    st_copy_emitted: usize,
    // COPYPF1 (--copy-span-prefill): the stretch schedule (rung, the copy cursor it belongs to; usize::MAX = none)
    // and this request's stretches / offered / accepted / emitted tokens / re-run rows (the [copy-span-prefill] line)
    cpv_next: (usize, usize),
    st_cpv_steps: usize,
    st_cpv_offered: usize,
    st_cpv_accepted: usize,
    st_cpv_emitted: usize,
    st_cpv_rerun: usize,
    // COPYLANES (--copy-span-lanes): this request's copy-work sums (copy rounds and stretches, TP-agreed ms) — the
    // price of its copy STARTS while other requests are busy
    cl_work: crate::copy_lanes::Work,
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
    // S-B9-REL-MTPREPROBE2 W1.1: rounds counted since the last estimator reset
    // (request start / re-arm) — the re-probe path blends a RUNNING MEAN over the
    // first 64 of them (see mtp_reprobe::blend_cost_ema), never a one-round seed.
    ema_n: usize,
    /// P18 SPEC8K: measured round-to-round WALL latency of this lane (ms): the end of one round to the
    /// end of the next, so it contains the wait for the rounds of the other busy lanes, which
    /// ema_ms (the time of one round, or for the lane of a packed round one SHARE of it) does not.
    /// Read only by the Auto arm of use_shared_plain, whose job is to price one arm against the other:
    /// both sides must be the aggregate that the n clients see. 0.0 until the second round of the lane.
    ema_wall_ms: f64,
    /// P18 SPEC8K: the instant round_account finished the previous round of this lane, the anchor for
    /// ema_wall_ms.
    last_wall: Option<std::time::Instant>,
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
    // S-B9-CF-PACK4S: which arm ran each of this request's rounds (`[mtp-stats] packed=`, and
    // `arms=` with --round-trace): the provenance of every emitted token.
    arms: crate::lane_pack::ArmTrace,
    // the request's first generated token ids (seam token first; at most FIRST_TOKS), printed
    // with --round-trace as `toks=`: the distribution gate's statistic, read with `arms=`
    first_toks: Vec<u32>,
    // S-A3-f-d: chain-verify state. tap_row = taps_keep row holding the tap that
    // predicted last_tok (usize::MAX after a control round: resid holds it).
    mtp: Option<usize>,
    // S-B9-REL-MTPREPROBE: re-probe state for the §6 per-request MTP auto-off
    // (src/mtp_reprobe.rs). mtp=None + phase Off = speculation off but the draft head
    // stays primed and a re-probe comes after a backoff; force_control = a pending
    // confirm wants ONE fresh control round at the current context.
    reprobe: crate::mtp_reprobe::ReProbe,
    force_control: bool,
    // CF-FCFS: this request's admission order (the scheduler's monotonically increasing admit
    // counter at admit time). fcfs lane order is the ONLY thing that reads it. It is a pure
    // function of replicated state — every TP rank runs the same admits in the same head-driven
    // order, so every rank assigns the same value to the same request (AGENTS 2.10).
    admit_seq: u64,
    // CF-FCHS: tokens this lane has generated since it last became the fcfs front lane. Reset when
    // a lane rotates to the back; compared against --lane-quantum to bound one turn's monopoly.
    fcfs_credit: usize,
    // S-B9-CF-PACK4: an image request (its slot carries a 3-axis RoPE map) — never packed in v1.
    img: bool,
    // Log probabilities for every emitted token (`BatchRequest::logprobs`): Some(n) = a
    // `TokEvent::Logprob` with the n most likely alternatives before each `Tok`. The lane keeps
    // speculating (owner 2026-11-02): a round's emitted token j is the choice of verify row j (row r0 + j of a
    // packed round), read back once per round in `round_emit`; plain steps read their row as before.
    logprobs: Option<u8>,
    // W2-EXL3: the request's compiled JSON-schema FSM (None = unconstrained), its current
    // state, and whether the constraint is live yet (false while the lane is still inside the
    // think block its prompt opened: the mask arms on the think-close token). A schema lane
    // never speculates (mtp = None for its whole life), so every token it emits comes from a
    // masked plain step.
    schema: Option<std::sync::Arc<crate::json_schema::SchemaMask>>,
    schema_state: u32,
    schema_armed: bool,
    tx: crate::server::TokTx,
}

/// S-B9-CF-PACK4: host-side anatomy of the packed rounds since boot: wall ms of the three stages
/// (the head re-prime is enqueued, so its GPU time lands in the next round's first draft pass,
/// exactly as the lone round's device re-prime does). The sums are CUMULATIVE; the idle line
/// prints their per-round means, and only when a round ran since the previous line.
#[derive(Default, Clone, Copy)]
struct PackStats {
    rounds: u64,
    rows: u64,
    toks: u64,
    ms_draft: f64,
    ms_verify: f64,
    ms_finish: f64,
    // the verify's own parts (FwdModel::pack_verify): staging, graph replay + sync, readback
    ms_stage: f64,
    ms_graph: f64,
    ms_read: f64,
    // per packed width R (index = rows, 0..=16): rounds, graph-and-sync ms, tokens emitted — the served cost curve
    by_r: [(u64, f64, u64); 17],
    printed: u64,
    // S-B9-CF-PACK4S: packed rounds that replayed the real-q family, and lane rounds of sampled
    // lanes (real-q, match rule) and of greedy lanes served by packed verifies
    rq_rounds: u64,
    lanes_rq: u64,
    lanes_match: u64,
    lanes_greedy: u64,
    // S-B9-CF-PACK5: host ms of the packed round's two lockstep points (TP only): the pre-verify
    // (wasted-pass drain + exchange + agree) and the post-verify tp_round (fence A + exchange + fence B)
    ms_pre: f64,
    ms_post: f64,
    // PACK4-REVIEW A3: true from the moment a pair round starts its packed part (after the
    // control-cadence fallback) until it completes; an error that leaves it set came from the
    // packed round, and the arm is disabled for the rest of the boot (`pack_disable`)
    in_round: bool,
}

/// TURNRESUME: one kept recurrent state per lane (`bufs[slot]`, allocated at boot) holding the lane's rows
/// `toks[slot]` (None = nothing kept: the rows above the prompt-end snapshot belong to something else), and the
/// rendered next-turn prefix waiting for a background pass on that lane (`pending[slot]`).
struct TurnArm {
    nt: crate::turn_resume::NtMode,
    chunk: usize,
    bufs: Vec<crate::exl3_forward::RecurCkpt>,
    toks: Vec<Option<Vec<u32>>>,
    pending: Vec<Option<Vec<u32>>>,
    /// when each pending hint was posted, and how long the box must have been idle since before a pass starts
    pending_at: Vec<std::time::Instant>,
    delay: std::time::Duration,
    /// rate limit of the pool's memory upkeep while the turn arm waits in short ticks
    last_upkeep: std::time::Instant,
}

/// TURNRESUME: what the idle wait returned.
enum TurnWake {
    Req(BatchRequest),
    Closed,
    Again,
}

struct Exl3Scheduler {
    model: Arc<FwdModel>,
    sc: Scratch,
    width: usize,
    lanes: Vec<Option<Lane>>,
    eos: Vec<u32>,
    chunk: usize, // S-A3-f-b: prefill chunk width (tokens per forward sweep)
    /// ISSUES[94] DF2: the loaded DFlash2 round (artifact + ring), present iff the boot gate
    /// ENGAGED it. One shared round per scheduler: the DF2 chain is a single token stream, so
    /// only a lane whose position matches the ring's `nprev` may run a DF2 round (the lone-lane
    /// case by construction; every other lane/step runs the pack-MTP paths unchanged).
    df2: Option<crate::dflash2::round::Df2Round>,
    /// The chain-verify depth the DF2 round uses: k = --df2-block - 1 (default
    /// `dflash2::block() - 1`).
    df2_k: usize,
    /// WIDEVERIFY1: the armed copy-span arm (`--copy-span on` at TP=1 with speculation live), else None.
    copy_arm: Option<CopyArm>,
    /// COPYREC1: the cross-request copy record (`--copy-span-record on` with the copy arm armed), else None. Every TP
    /// rank holds its own, fed the same ids in the same order (prompts at admit, replies at the lane's release).
    record: Option<Box<crate::copy_record::CopyRecord>>,
    /// COPYPF1: `--copy-span-prefill` resolved on (one box, the copy arm armed, no DFlash2 chain) and the
    /// step's device buffers (allocated on its first step). false = step_mtp never calls the arm.
    cpv_on: bool,
    cpv: Option<crate::exl3_forward::CpvBufs>,
    /// ECHOPRICE: the measured costs of --copy-span-prefill-price (copy rounds vs stretches; TP-agreed times)
    cpv_price: crate::copy_prefill::Price,
    /// COPYLANES: `--copy-span-lanes` resolved on (the copy arm armed, --lane-order rr, two or more lanes),
    /// `--copy-span-lanes-price`, and the measured copy costs its price reads (TP-agreed times). false = nothing below
    /// is called and every step runs as without the flag.
    cl_on: bool,
    cl_priced: bool,
    cl_price: crate::copy_lanes::Price,
    /// LANEWIDE: `--lane-wide-verify` (state, buffers allocated on the first wide round, the measured-rate policy)
    wide: crate::lane_wide::WideSched,
    /// `--prefill-interleave N` (public PR #23): scheduler steps run for the lanes already decoding
    /// after each non-final prefill chunk of a joining prompt. 0 (default) = none: the whole prompt
    /// prefills before the next step, exactly as before. Always 0 under TP (build_serve).
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
    /// LANERES1 `--lane-reserve`: the lane-reservation policy. None = off = today's admission,
    /// byte-for-byte (the run loop and the pick keep their original shapes).
    reserve: Option<crate::lane_reserve::Cfg>,
    /// The policy's forced-take limiter state (one per scheduler).
    reserve_st: crate::lane_reserve::State,
    /// LANERES2 `--kv-spill`: the SSD spill tier for evicted lane snapshots. None = off =
    /// today's behavior, byte-for-byte (no capture, no restore, no counters).
    spill: Option<crate::kv_spill::Store>,
    /// The spill policy (cap, LRU, cost model); a zero-cap policy when the tier is off.
    spill_pol: crate::kv_spill::Policy,
    /// This scheduler's rank (0 on a single box): spill files are per-rank.
    spill_rank: usize,
    /// PFXPOOL `--prefix-pool-gb`: prompt states kept in device (unified) memory ACROSS lanes. A zero-cap pool
    /// when off (--prefix-pool-gb 0 or --prefix-cache off). Replicated policy under TP (see prefix_pool.rs).
    pool: crate::prefix_pool::Pool<PoolBuf>,
    /// PFXPOOL: slabs of entries shifted out, kept for the next capture (a fresh allocation provisions new pages at
    /// ~25-30 GB/s: 50-400 ms for a long prompt; a recycled slab costs nothing). Freed first under the memory floor.
    pool_spare: Vec<cudarc::driver::CudaSlice<u8>>,
    /// PFXPOOL: MemAvailable the pool keeps free on every rank (a capture that would go below it first evicts, then
    /// is dropped; PFXPOOL2: an admission that finds a rank below it shifts entries out until every rank is above);
    /// --prefix-pool-floor-gb.
    pool_floor: u64,
    /// PFXPOOL2: per lane, the lineage of its positional rows (which prefill computation wrote which rows; see
    /// prefix_pool::Segs) — pool entries whose rows agree share one copy. Replicated (SPMD admission).
    slot_segs: Vec<crate::prefix_pool::Segs>,
    /// PFXPOOL2: the generation counter `slot_segs` labels prefill computations with.
    row_gen: u64,
    /// PFXPOOL2: when `pool_emergency` last read MemAvailable.
    pool_emerg_at: Option<std::time::Instant>,
    /// Per-slot release stamps (released_at, normal finish) for the policy. Stamped only while
    /// the policy is on; read to build SlotViews and the forced-take log line.
    rel: Vec<Option<(std::time::Instant, bool)>>,
    /// S-B9-CF-PACK4 `--lane-pack` (see step_packed): the mode, the packed verify's own scratch
    /// (Some only with the flag on), which regime families are captured [dense, sparse], the
    /// rotation counter, the skip-line budget, "a prefill is in flight" and the idle-line stats.
    lane_pack: crate::lane_pack::LanePack,
    scp: Option<Scratch>,
    pack_ready: [bool; 2],
    /// S-B9-CF-PACK5: sampled lanes may pack on this boot (TP=1: yes; under TP: `--lane-pack-sampled-tp`,
    /// see `pack_sampled_under_tp`). A function of the topology and a shipped option: the same on every rank.
    pack_sampled: bool,
    pack_rot: u64,
    pack_logs: usize,
    in_prefill: bool,
    pack_stats: PackStats,
    /// CF-P1e auto policy: learned scale of the shared-step time model, the last decision (hysteresis) and the
    /// consecutive shared-step count (stale-estimate refresh). All updated from lockstep-adopted values.
    shared_scale: f64,
    shared_last: bool,
    shared_streak: usize,
    /// F4 (PACK5G finding 1): rounds riding the packed/serial arm at >=4 busy lanes since the
    /// last shared plain step, under `--lane-pack on` + Auto (the shared-cost probe; see step()).
    shared_probe_ctr: u32,
    // S-C4XOVER: measured delivered rate per arm (tokens / ms of the rounds that ran),
    // the honest basis of the Auto pick: both modelled inputs were biased at four busy
    // lanes (the serial cycle estimate ~+30% optimistic vs delivery, the shared prior's
    // learned scale ~-18% pessimistic), which pinned the policy to the slower arm.
    spec_win_tok: [f64; 2],        // window accumulators: tokens the arm's rounds delivered
    spec_win_ms: [f64; 2],         // ... the ms those rounds cost
    spec_win_n: [u8; 2],           // ... the busiest lane count seen in the window
    spec_win_acc: [u32; 2],        // ... rounds accumulated since the last fold
    spec_arm_rate: [f64; 2],       // EWMA of delivered aggregate tok/ms (0 = never measured)
    spec_arm_n: [u8; 2],           // busy lanes the rate was measured at (last fold's tag)
    spec_arm_fold: [u64; 2],       // scheduler step of the last fold (freshness)
    /// POSTURE1 --spec-arm-meter v2: the measured rate of each arm PER EXACT busy-lane count (index n, capped at
    /// SPEC_RATE_NMAX): (EWMA tok/ms, scheduler step of its last fold). v1 (the default) never reads it.
    spec_rate_n: Vec<[(f64, u64); 2]>,
    /// ULTRA v2: folds per (exact busy count, arm) (young-estimate test), the settle countdown and the busy count it
    /// follows, and the arm whose next fold comes from a probe window (the replace rule applies to it)
    spec_rate_folds: Vec<[u32; 2]>,
    meter_settle: u32,
    meter_last_n: usize,
    probe_fold: [bool; 2],
    spec_est_pick: Option<bool>,   // S-C4XOVER v3: the estimate path's last pick (None before one)
    spec_probe: Option<(u8, u32)>, // arm-rate probe window in flight: (arm, steps left)
    spec_probe_ctr: u32,           // rounds since the last probe of the idle arm
    spec_margin: f64,           // S-C4XOVER v2: last MEASURED arm margin (max/min - 1; 0 = none)
    spec_margin_n: u8,          // ...and the busy count it was measured at (eager again on change)
    policy_logs: usize,
    ctrl_round: usize,
    /// S-B9-REL-MTPREPROBE: the re-probe schedule (shrunken by --mtp-reprobe-drill) and the
    /// master switch (--mtp-reprobe=off = the previous one-way behaviour, exactly).
    reprobe_cfg: crate::mtp_reprobe::ReProbeCfg,
    reprobe_on: bool,
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
    /// POSTURE1 --sched-trace: per-slot phase tracking (diagnostic only; never read by a decision)
    tr: SchedTrace,
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
    /// PACKPF: slot -> the prompt a packed prefill already ran there, waiting for its admit.
    pf_done: std::collections::HashMap<usize, PfDone>,
    /// PACKPF: prompts whose packed prefill is in flight (FIFO order; their lanes are reserved).
    pf_jobs: Vec<PfJob>,
    // VIS-2: the text config's interleaved mrope_section (rope_parameters.mrope_section; None = the
    // model declares no interleaved mrope -> image requests are refused)
    mrope_section: Option<[usize; 3]>,
    // VIS-4: the last image admit's (image tokens, rope delta, FNV of the fp16 rows) — proven equal on
    // every rank by the seam lockstep (None = a text admit)
    vis_seam: Option<(u32, i32, u64)>,
    // Public issue #24: the prefill in flight as /metrics sees it — (slot, positions written so
    // far). Set per completed chunk, cleared when the admit ends (the lane then carries `pos`).
    kv_pf: Option<(usize, usize)>,
    // TURNRESUME (--turn-resume / --next-turn-prefill): the kept next-turn state per lane + the pending hints
    // (None = off, or asked for and fell back at boot).
    turn: Option<Box<TurnArm>>,
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

#[derive(Default, Clone)]
pub(crate) struct WaitProf {
    n: [u64; 8],
    sum_ns: [u64; 8],
    max_ns: [u64; 8],
    hist: [[u64; 6]; 8],
    sleeps: [u64; 8],
    rounds: u64,
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
    /// `--tp-width-drill N` (test drill, per box, one-shot; None = not armed): see `width_said`.
    drill_width: Option<usize>,
    /// rounds the drill has lied about: 0, then 1 for the rest of the boot (it is one-shot)
    pub drill_hits: u64,
}

impl TpSync {
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
            crate::deprintln!("[tp-race-probe] rank {}: pre-verify aligned xchg gen {g0} -> {g1} onto epoch {e_last} \
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
        let r = tp_agree_eq(step, w, w + 1, h);
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

impl TpSync {
    fn hub_fail(&self, what: &str, why: &dyn std::fmt::Display) -> anyhow::Error {
        // keep an abort code a failed exchange already recorded (10 = peer dead, 12 = lockstep deadline)
        crate::net::abort_link_keep_code();
        anyhow::anyhow!("TP lockstep {what} FAILED at rank {} (world {}): {why}", self.rank, self.world)
    }
}

/// WP02: the scheduler thread's liveness guard. A panic unwinding out of the scheduler drops its
/// lanes' senders (every in-flight request then ends "error: engine stopped", never "length"),
/// then this marks the engine DEAD; with --exit-on-fatal it exits 70 for the supervisor. An abort
/// (a panic inside a CudaSlice drop, AGENTS §5a) never reaches here: the supervisor restarting on
/// abort is that half.
struct FatalGuard {
    exit_on_fatal: bool,
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

/// WP23 per-lane state: the cost fit, the last round's target, rounds by verified draft width.
#[derive(Clone, Copy, Debug, Default)]
struct Wp23Lane {
    fit: CostFit,
    target: f64,
    widths: [usize; crate::exl3_forward::MTP_MAX_K + 1],
}

/// `--sched-trace` state: the `<think>` / `</think>` ids (resolved once) and per slot (admit_seq, history entries
/// scanned, the request is past its reasoning).
#[derive(Default)]
struct SchedTrace {
    marks: Option<(Option<u32>, Option<u32>)>,
    lanes: Vec<(u64, usize, bool)>,
}

/// PACKPF: prefill_cached's decisions for one prompt (prefill_plan).
#[derive(Clone, Debug)]
struct PfPlan { from: usize, end: usize, tracked: bool, tail: Option<usize>, realign: bool, q: usize, snapshot: bool }

/// PACKPF: a prompt a packed prefill finished, waiting for its admit (which then skips the prefill).
struct PfDone { key: Vec<u32>, resume: (usize, usize), end: usize }

/// PACKPF: one prompt in flight in pack_prefill_waiting.
struct PfJob { slot: usize, at: std::time::Instant, key: Vec<u32>, plan: PfPlan, grid: Vec<usize>, pos: usize, gi: usize,
               taken: Vec<usize>, resume: (usize, usize), passes: usize, shared: usize, t0: std::time::Instant }

impl Exl3Scheduler {
    /// TP-C: one lockstep point (a seam step, a plain batched step, an MTP round or a control
    /// step). (1) agree_ext(step, accept, verify width m, FNV of `words`) — the load-bearing
    /// tripwire (AGENTS §2.10): a divergence aborts the link on both ranks; (2) a u32 exchange
    /// that ships the head's (ms, plain ms, captured) — returned on EVERY rank, so every timing-
    /// derived decision downstream is the head's; (3) with `ident` on, the digests of the logits
    /// rows 0..m, the residual and the taps window are compared across ranks (G-T1-b graphed).
    ///
    /// S-B9-CF-PACK5: `digest_pack` = the round's rows live in the PACKED scratch (a packed verify):
    /// the `--exl3-tp-ident` digests then read it, not the served scratch (whose argmax / logits are
    /// only `--max-batch`-wide: a 16-row pack indexed past them).
    fn tp_round(&mut self, what: &str, a: usize, m: usize, words: &[u32], ms: f64, plain_ms: f64,
                captured: bool, digest_pack: bool) -> Result<(f64, f64, bool)> {
        let Some(t) = self.tp.as_mut() else { return Ok((ms, plain_ms, captured)) };
        let dsc = if digest_pack { self.scp.as_ref().unwrap_or(&self.sc) } else { &self.sc };
        Self::tp_round_with(t, &self.model, Some(dsc), what, a, m, words, ms, plain_ms, captured)
    }

    /// `tp_round`'s body over the TP state alone (COPYPF1-TP: the stretch's judge agree runs while the step holds
    /// the scratches). `dsc` = the scratch the `--exl3-tp-ident` digests read; None = no digests on this lockstep
    /// point (every rank passes None at the same point, so the digest counts still match).
    #[allow(clippy::too_many_arguments)]
    fn tp_round_with(t: &mut TpSync, model: &crate::exl3_forward::FwdModel, dsc: Option<&crate::exl3_forward::Scratch>,
                     what: &str, a: usize, m: usize, words: &[u32], ms: f64, plain_ms: f64, captured: bool)
                     -> Result<(f64, f64, bool)> {
        if t.world > 2 {
            // TP-4D: ONE merged head-hub op (tp_lockstep::RND) instead of agree / exchange / agree
            let mut x = crate::net::link_hub(t.deadline_ms)?;
            let epoch = crate::net::traced_device_epoch();
            return t.round_hub(&mut x, what, a, m, words, ms, plain_ms, captured, epoch,
                               |m| match dsc { Some(sc) => model.tp_ident_digests(sc, m), None => Ok(Vec::new()) });
        }
        t.step += 1;
        let (step, rank, ident) = (t.step, t.rank, t.ident);
        let h = fnv32(0, words.iter().copied());
        // Fence A = the agree token itself: both ranks finished this step's forward, so every hot-
        // path epoch payload of it has been read on both sides (the exchange below reuses ring slot
        // memory: its send buffer is send slot 0, its receive lands in recv slot gen & 7).
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let r = tp_agree_eq(agree_field(step, AGREE_A), a, t.width_said(m), h);
        t.wp.rec(0, t0, s0);
        t.wp.rounds += 1;
        if let Err(why) = r {
            crate::net::abort_link();
            anyhow::bail!("TP agree() FAILED at {what} (lockstep step {step}, accept {a}, width {m}, hash {h:08x}): {why} — \
                           ranks diverged or the link aborted");
        }
        let digests = if ident {
            match dsc { Some(sc) => model.tp_ident_digests(sc, m)?, None => Vec::new() }
        } else { Vec::new() };
        let tag = 0xC0DE_0000u32 | (step as u32 & 0xFFFF);
        let (mb, pb) = (ms.to_bits(), plain_ms.to_bits());
        let mut mine = vec![tag, mb as u32, (mb >> 32) as u32, pb as u32, (pb >> 32) as u32, captured as u32,
                            digests.len() as u32];
        mine.extend(digests.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]));
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, mine.len() + 4)?;
        let tw = &mut *t;
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
        t.agrees += 1;
        if ident {
            let bad = (0..digests.len()).filter(|&i| mine[7 + 2 * i] != peer[7 + 2 * i] || mine[8 + 2 * i] != peer[8 + 2 * i]).count();
            t.ident_bufs += digests.len() as u64;
            t.ident_bad += bad as u64;
            if bad > 0 && t.ident_bad as usize == bad {
                crate::deprintln!("[tp-ident] FIRST MISMATCH at {what} (lockstep step {step}, width {m}): {bad} of {} digests differ \
                           (logits rows 0..{m}, resid, taps)", digests.len());
            }
            // MTPAGREE1: name WHICH digests differ (index < m: logits row / vp key; m: resid; m+1: taps), every time,
            // at --log-level debug, so the first divergent round says trunk vs head.
            if bad > 0 {
                let which: Vec<String> = (0..digests.len()).filter(|&i| mine[7 + 2 * i] != peer[7 + 2 * i] || mine[8 + 2 * i] != peer[8 + 2 * i])
                    .map(|i| format!("{}:{:08x}{:08x}/{:08x}{:08x}", if i < m { format!("logits{i}") } else { ["resid", "taps", "x", "chain_xh", "head_kv"].get(i - m).map(|s| s.to_string()).unwrap_or_else(|| format!("d{i}")) },
                                     mine[8 + 2 * i], mine[7 + 2 * i], peer[8 + 2 * i], peer[7 + 2 * i])).collect();
                crate::deprintln!("[tp-ident] {what} step {step} width {m}: differing digests (mine/peer) {}", which.join(" "));
            }
        }
        if rank == 0 {
            Ok((ms, plain_ms, captured))
        } else {
            Ok((f64::from_bits(peer[1] as u64 | ((peer[2] as u64) << 32)),
                f64::from_bits(peer[3] as u64 | ((peer[4] as u64) << 32)), peer[5] != 0))
        }
    }
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
    /// OUT1: what the one SERVING line needs at listen time (tp, ctx, batch, kv, prefix,
    /// mtp-head-present, mtp-off).
    boot_facts: (u32, usize, usize, String, bool, bool, bool, bool),
}

#[cfg(test)]
mod dds_prior_w4_tests;

#[cfg(test)]
mod ctx_clamp_tests;

#[cfg(test)]
mod wp23_tests;

/// TP-4D: CPU models of every world > 2 lockstep op of the serve path — the REAL `TpSync::*_hub` functions and the
/// REAL `net::hub_all` over the in-memory `net::hub_mock` transport, one thread per rank (4 ranks). What these prove:
/// the merged hub ops are symmetric (every rank runs the same op sequence and reaches the same verdict), counters
/// stay in lockstep, nodes adopt the head's words, a divergent / dead / hung rank fails EVERY live rank (none is left
/// parked), and the WP16 allocation agree keeps four checkpoint stores identical. What they do NOT prove: anything
/// about RDMA placement, the proxy, the hot-path rings or the GPU (hardware questions for the W=4 gate run).
#[cfg(test)]
mod tp_hub_tests;

#[cfg(test)]
mod fit_tests;

#[cfg(test)]
mod spec_lanes_tests;

#[cfg(test)]
mod prefill_interleave_tests;

#[cfg(test)]
/// S1 (REL_V0_7_3 review): both spellings of every flag EXL3 reads with `arg()` must work.
/// The table-driven test drives the REAL parser through the REAL `arg()` — before the fix,
/// `--lane-order=fcfs` returned None (silently ignored) while `--print-config` showed it set.
mod s1_eq_aware_arg_tests;

#[cfg(test)]
mod step_image_payload_tests;

/// S-B9-REL-SMALLFIXES item 1 (owner 2026-10-08): `--mtp off` is honoured by the EXL3 server.
/// Before, `--mtp` was documented in `--help` and ignored here: the banner stayed 'MTP chain-verify
/// ON' and the only working switch was the diagnostics hatch `--exl3-no-mtp`.
#[cfg(test)]
mod mtp_switch_tests;

/// S-B9-REL-SMALLFIXES item 5(a): the scheduler's side of `stop_p` / `stop_rank`.
#[cfg(test)]
mod stop_info_tests;

/// S-B9-CF-PACK4: `--lane-pack` wiring that no unit can run (the suite never builds a Scratch,
/// AGENTS 1a-ter): source pins, so a later edit cannot silently drop a refusal or the gate.
#[cfg(test)]
mod lane_pack_wiring_tests;

/// S-B9-REL-SMALLFIXES 9a(iii) + 9a(iv): the scheduler side of the per-row attention regime.
#[cfg(test)]
mod regime_sched_tests;

/// S-B9-REL-VISIONFIX H2: the `--vision-fail-gemm` drill through the real argument path
/// (`opts::parse_cli`, what `main` runs on argv, then the accessor the serve path calls).
#[cfg(test)]
mod vision_drill_flag_tests;

// ------------------------------------------------------------------------------------------------------
// S-B9-CF-AGREE16: the lockstep words must tell every verify width apart (0..=31), 15 from 16 included.
// These tests drive the real field packing (`tp_agree_fields`), the real token (`net::agree_token`) and the
// real comparison (`tp_agree_decide`); only the wire is replaced: `net_agree` hands each rank the other
// rank's token for the same step field, which is what `pair` does.
// ------------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tp_agree_width_tests;

/// S-B9-CF-AGREE16B: the one-shot, rank-dependent width drill (W3) and the hoisted `--max-batch` refusal (W4).
#[cfg(test)]
mod agree16b_tests;

/// S-B9-REL-INTEG F1 (review of the orchestrator's `2d52ad79`): `--lane-pack-sampled-tp` through the real
/// argument path (`opts::parse_cli`, then the accessor the boot reads), and the three properties the option
/// rests on: it rides to every rank (Spmd, in the TpConfig snapshot and the boot-agree digest), `off`
/// restores the earlier graph set (greedy family only), and a bad value is refused at startup.
#[cfg(test)]
mod lane_pack_sampled_tp_tests;

/// S-B9-REL-INTEG: the `--lane-pack on` boot warning for penalty defaults (the owner's posture has
/// `--default-presence-penalty 1.5`, under which no request packs).
#[cfg(test)]
mod lane_pack_penalty_warning_tests;

// ---- ISSUES[94] DF2 gate tests: the reason table is host-pure (df2_gate_reason touches no
// GPU and no filesystem), so the flag-parse fallback outcomes are pinned here exactly as the
// boot line prints them. A non-existent dir is NOT a gate reason: it engages and fails at the
// LOAD (reason `load failed: <e>`), which needs the GPU and cannot be tested on the host.
#[cfg(test)]
mod df2_gate_tests;

/// ISSUES 121: the world > 2 spill agree end to end over the four-rank mock hub (the real `hub_all`, frame and
/// verdict): healthy writes agree on every rank, one rank's failed write drops the snapshot on every rank, and no rank
/// aborts its link either way. The shipped build failed this on every rank at the first TP=4 capture.
#[cfg(test)]
mod spill_agree_tests;

/// ISSUES 127 / 128 (static review P0s): source pins for the DF2 width clamp and the draft slot staging (the ISSUES 126
/// ring re-alignment was measured and NOT kept: archive/reviewfix1-issue126-attempt). The behaviour needs a GPU; these keep a later edit from quietly restoring the old code.
#[cfg(test)]
mod review_p0_pins;

#[cfg(test)]
mod token_logprobs_tests;

#[cfg(test)]
mod tpship_kv_tests;

#[cfg(test)]
mod tp_preflight_tests;

#[cfg(test)]
mod copy_span_arm_tests;
