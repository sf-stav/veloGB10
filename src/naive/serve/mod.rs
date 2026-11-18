//! S5: single-stream tensor-parallel serving of Naive-N0.5-Flash behind the engine's OpenAI HTTP
//! server (PLAN/NAIVE_N05_PORT.md §5.2).
//!
//! Process model = the engine's cluster: the head (`--server --model-dir <pack> --tp 2`) syncs the
//! nodes (`--node`), brings up the RDMA link, every rank loads ITS shard (shard.rs), the node sends
//! Ready, the head binds HTTP. Requests arrive as `BatchRequest`s on the server's channel; the head
//! runs them ONE AT A TIME (the single-stream goal) in SPMD lockstep with the node over the retained
//! control stream:
//!
//! ```text
//!   head -> node:  Admit { prompt, .. }      (one per request)
//!                  Tok { t, stop }           (one per decode step: the head's sampled token; the
//!                                             node feeds it — head-authoritative, so the ranks can
//!                                             never diverge on a sampling tie or RNG detail)
//!                  Shutdown
//!   node -> head:  Ready                     (tp_serve::ServingMsg::Ready, once)
//! ```
//! The data plane (two fp32 all-reduces per layer) is the RDMA doorbell link (tp.rs). lm_head is
//! replicated and runs on the head only; sampling runs on the head's CPU (S6 moves it to the GPU
//! and makes the head vocab-parallel).
//!
//! Map of this directory (`Engine`'s impl and the serve entry points are cut by topic; every item keeps its `crate::naive::serve::` path):
//! - `mod.rs` (this file): control messages, speculation config / guard, the `Engine` struct and prefix-snapshot planning.
//! - `engine.rs`: `impl Engine` (load, prefix snapshots, speculative rounds, request begin, lane steps, prefill, decode).
//! - `flags.rs`: serve flags (`lanes_asked`, `pdl_flag`, `spec_cfg`, ...) and `sample_params`.
//! - `loops.rs`: the head's one-at-a-time and lane loops and the node's lockstep loop.
//! - `head.rs`: `run_head` and `run_node` (cluster bring-up around the loops).
//! - `probes.rs`: the `--probe-naive-*` TP / loop / prefill / shard-dir / spec probes.
//! - `prefix_tests.rs`: the inline unit-test module.

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::decode::{DecodeStep, SampleParams};
use super::drafter_gpu::GpuDrafter;
use super::model::{dptr, ws_bytes, GpuModel, Ws, MAX_ROWS};
use super::pack::NaivePack;
use super::shard::Shard;
use super::tp::NaiveTp;
use crate::batch::{BatchRequest, TokEvent};
use crate::tokenizer::QwenTokenizer;

/// Head -> node control messages (length-prefixed JSON on the retained cluster stream).
#[derive(Serialize, Deserialize, Debug)]
pub enum Ctl {
    /// `min_new` / `ignore_eos` ride along so the node's stop rule is literally the head's (a head-only
    /// min_new delayed the head's stop and tripped the lockstep check on the flag, not the data).
    /// `reuse` = the position the prefill resumes at (prefix reuse; 0 = from the start), `snap` = the chunk boundary
    /// where this prompt's own snapshot is taken: the head decides both, every rank does the same
    Admit { prompt: Vec<u32>, max_new: usize, seed: u64, sp: SampleParams, min_new: usize, ignore_eos: bool,
            #[serde(default)] reuse: usize, #[serde(default)] snap: Option<usize> },
    Tok { t: u32, stop: bool },
    /// S8: one speculative round's committed tokens (the lagging lockstep check of the node)
    Round { toks: Vec<u32>, stop: bool },
    /// G1 (`--naive-lanes N`), top of an iteration: the lanes the head finished after the previous step (applied in
    /// DESCENDING order, each a swap-remove: lane n-1 moves into the hole), then the requests it admits now (lane =
    /// the active count at admission, slot = a free KV slot, owner = it takes the drafter context)
    LIter { fin: Vec<usize>, admits: Vec<LAdmit> },
    /// G1, before the step: `drop` = admitted lanes finished by their first token (descending, swap-remove), `kind` =
    /// 0 a multi-request plain step over every active lane, 1 a speculative round of lane 0, 2 a plain step of lane
    /// 0 fed to the drafter (the guard); `check` = every lane's tokens of the previous step (the lockstep check)
    LStep { drop: Vec<usize>, kind: u8, check: Vec<Vec<u32>> },
    Shutdown,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LAdmit { pub lane: usize, pub slot: usize, pub prompt: Vec<u32>, pub max_new: usize, pub seed: u64, pub sp: SampleParams, pub owner: bool,
                    /// prefix reuse in this slot: the resume point and the snapshot point (the head's decision, as in `Admit`)
                    #[serde(default)] pub reuse: usize, #[serde(default)] pub snap: Option<usize> }

/// S8: the head's drafter (the node has none: drafts travel in a slot exchange).
pub struct HeadSpec {
    pub drafter: GpuDrafter,
    /// head-only drafter: the FULL lm_head (the model's is the rank's vocabulary slice; the Markov stage wants every
    /// logit). None = the sharded drafter, which scans the rank's slice through `model.lm_head`.
    pub lm_full: Option<super::gpu::GpuQuad>,
    pub ws: Ws,
    /// Work item A: the sampled drafter's full-vocabulary q, one f32 row of `vocab` per draft row
    /// (`(block-1) * vocab` floats) plus `2 * (block-1)` floats of (max, Z) per row. Persistent: the
    /// address is baked into the verify graphs, so it is allocated once at load.
    pub qbuf: CudaSlice<f32>,
    pub qstat: CudaSlice<f32>,
}

/// `--spec-source dspark` (default when the drafter dir exists next to the pack) with `--spec-block B`
/// (B = 1 anchor + B-1 drafts per verify, 2..=16; default 8).
#[derive(Clone, Debug)]
pub struct SpecCfg { pub draft_dir: Option<String>, pub block: usize, pub conf: f32, pub guard_off: f32, pub guard_on: f32, pub probe_every: usize,
                     pub draft_temp: f32, pub drafter: String, pub sampler: String,
                     /// G1 (`--naive-lane-ctx on|off`, default on): every lane keeps its own drafter context
                     pub lane_ctx: bool }
/// `--spec-drafter auto|head|sharded` (work item B): where the drafter runs under TP. `sharded` = every rank runs its
/// shard of the drafter (heads, MLP, fc rows, vocabulary slice) with fp32 doorbell exchanges, ~1/W of the bytes per
/// rank; `head` = the whole drafter on rank 0 while the nodes wait (the S8 path). `auto` = sharded at world > 1 with
/// argmax drafts (`--spec-draft-temp 0`), head otherwise (sampled drafts need the whole vocabulary's q on one rank).
pub const SPEC_DRAFTER: &str = "auto";
/// `--spec-sampler auto|rows|batched` (work item C): how the verify block's rows are sampled under TP. `batched` =
/// the rows sampled phase-major with ONE doorbell exchange per sampler phase for all rows (8 per block); `rows` =
/// the rows in sequence, 8 exchanges each (the item-A path). Bit-identical draws either way. `auto` = batched at
/// world > 1, rows at world 1 (no exchanges there).
pub const SPEC_SAMPLER: &str = "auto";
/// `--spec-draft-temp F` (work item A): the drafter draws its sampled drafts from q at F x the request's temperature.
/// Exactness does not depend on F (the accept test uses the same q the drafts were drawn from); F only moves the
/// acceptance rate: F -> 0 is the old greedy-draft rule (q one-hot, accept with p(d)), F = 1 is q as trained.
pub const SPEC_DRAFT_TEMP: f32 = 0.0;
// Default 0.0 (2026-10-20 sweep, TP4, production sampling, build ba18d10d): the drafter's q as trained is FLATTER
// than the target's p, so drafts drawn from it accept less, monotonically in F -- thinking prose 1.45-1.56 (F 1.0),
// 1.51-1.83 (0.6), 1.63-1.72 (0.3), 1.81-1.94 (0.0) accepted/round; the C-code cell shows the same trend with wider
// spread. F = 0 is argmax drafts with the accept test u < p(d) (q one-hot), i.e. the old rule's acceptance with the
// exact residual, still distribution-exact. The drafter is good at RANKING, not at probabilities.
/// Default verify-width cap: the drafter's trained block (anchor + 6 masks). The verify width of a round is
/// adaptive (`--spec-conf`): only the leading drafts whose confidence clears the threshold are verified, so a
/// wide draft never costs expert bytes it is unlikely to pay back.
pub const SPEC_CONF: f32 = 0.25;
pub const SPEC_BLOCK: usize = 7;
/// The acceptance guard (owner note §4 / v0.7.4 lesson): speculation is kept while the per-request EMA of
/// accepted drafts per round stays above SPEC_GUARD_ON; below SPEC_GUARD_OFF the request runs PLAIN steps
/// for SPEC_PROBE_EVERY tokens, then re-probes with speculative rounds. A pure function of the acceptance
/// counts (identical on every rank), never of timings (which differ per rank): the node mirrors it.
/// At block 4 a verify costs ~2.3 plain steps, so speculation wins at > ~1.3 accepted per round.
/// Defaults with the adaptive width (2026-10-19 calibration at TP2: a round of ~2.3 verified rows costs ~1.8
/// plain steps and pays from ~1 accepted): `--spec-guard-off` / `--spec-guard-on` override.
pub const SPEC_GUARD_OFF: f32 = 0.8;
pub const SPEC_GUARD_ON: f32 = 1.1;
pub const SPEC_PROBE_EVERY: usize = 16;   // `--spec-probe-every`: plain steps before re-probing (was 48)
pub const SPEC_PROBE_ROUNDS: usize = 3;

/// The guard's state (per request, mirrored on every rank).
pub struct SpecGuard { ema: f32, on: bool, plain_left: usize, probe_left: usize, off: f32, on_at: f32, every: usize }
impl SpecGuard {
    pub fn new() -> Self { Self::with(SPEC_GUARD_OFF, SPEC_GUARD_ON) }
    pub fn with(off: f32, on_at: f32) -> Self { Self::with_every(off, on_at, SPEC_PROBE_EVERY) }
    pub fn with_every(off: f32, on_at: f32, every: usize) -> Self { SpecGuard { ema: 3.0, on: true, plain_left: 0, probe_left: 0, off, on_at, every: every.max(1) } }
    /// Decide the next action: true = a speculative round, false = a plain step.
    pub fn speculate(&mut self) -> bool {
        if self.on { return true; }
        if self.plain_left > 0 { self.plain_left -= 1; return false; }
        true // re-probe
    }
    /// A speculative round's acceptance.
    pub fn observe(&mut self, accepted: usize) {
        self.ema = 0.7 * self.ema + 0.3 * accepted as f32;
        if self.on {
            if self.ema < self.off { self.on = false; self.plain_left = self.every; self.probe_left = SPEC_PROBE_ROUNDS; }
        } else {
            // in a probe: stay off (and plain for another stretch) unless the EMA recovered
            if self.probe_left > 0 { self.probe_left -= 1; }
            if self.ema >= self.on_at { self.on = true; }
            else if self.probe_left == 0 { self.plain_left = self.every; self.probe_left = SPEC_PROBE_ROUNDS; }
        }
    }
}
impl Default for SpecGuard { fn default() -> Self { Self::new() } }

/// The served engine on one rank: the model (its shard), a prefill workspace (16-row chunks) and the
/// graphed decode step (m = 1; the head's carries the sampler).
pub struct Engine {
    pub model: GpuModel,
    pub max_pos: usize,
    pf_ws: Ws,
    pub dec: DecodeStep,
    pub chunk: usize,
    /// S8: the verify block step (m = block), the head's drafter, the prefill tap buffer (head), the drafts exchange slot
    pub ver: Option<DecodeStep>,
    /// adaptive verify width: the steps for m = 2 .. block-1 (`vers[m - 2]`); `ver` is m = block
    pub vers: Vec<DecodeStep>,
    pub spec: Option<HeadSpec>,
    pf_taps: Option<CudaSlice<f32>>,
    xd: CudaSlice<f32>,
    pub block: usize,
    /// the confidence threshold a draft must clear to be verified (0 = verify every draft)
    pub conf: f32,
    /// `--spec-draft-temp`: multiplier on the request temperature for the drafter's q (sampled requests)
    pub draft_temp: f32,
    /// the verify width of the last speculative round (0 = it degenerated to a plain step)
    last_width: usize,
    /// the acceptance guard's thresholds (`--spec-guard-off` / `--spec-guard-on`)
    pub guard: (f32, f32, usize),
    /// the current request's sampler knobs (work item A: the drafter must draw from the request's T)
    pub sp_cur: SampleParams,
    /// the current request's seed and speculative-round counter (the accept draws are keyed on both)
    pub seed_cur: u64,
    pub smp_round: u64,
    /// rows whose residual draw the accept kernel could not make (rtok = -1: no candidate list or an
    /// empty residual) and that committed the target's own draw instead; reported per request
    pub smp_fallbacks: u64,
    /// of those, rows with NO candidate list at all (the gather did not run: a bug, never expected)
    pub smp_nolist: u64,
    /// work item B: the drafter runs sharded on every rank (else head-only, nodes idle through the draft)
    pub sharded: bool,
    /// work item C: the verify block's rows are sampled with the batched exchange (else per row)
    pub batched: bool,
    /// a node's own drafts of the current round (sharded drafter): checked against the head's exchanged drafts
    pub node_drafts: Option<(Vec<u32>, usize)>,
    /// seconds spent in `draft()` this request (printed as draft ms per round)
    pub draft_secs: f64,
    /// the current round's drafts are argmax draws (no q written): the accept kernel uses a one-hot q
    pub draft_onehot: bool,
    /// G1 (`--naive-lanes N`): concurrent requests (1 = the one-at-a-time serve), their shared sampler table and the
    /// multi-request steps for m = 2..=lanes (`lsteps[m - 2]`); the model holds `lanes` KV slots
    pub lanes: usize,
    pub lane_tab: Option<super::decode::LaneTab>,
    pub lsteps: Vec<DecodeStep>,
    /// prefix reuse state (`--prefix-cache on`): one snapshot per KV slot (lanes 1 = slot 0); empty = off
    pub pfx: Vec<PrefixSnap>,
    /// G1 + speculation: every lane's own drafter context (per drafter layer the K and V ring), so a request that is left
    /// alone can speculate whichever lane it was admitted on. The drafter's own ring buffers hold the LIVE lane's context
    /// (`ctx_live`); `lane_ctx[j]` holds lane j's while it is parked, and a spare while j is live (making a lane live swaps
    /// buffer handles: the drafter's kernels are eager, no graph holds a ring address). Empty = one shared context (lanes 1,
    /// or no drafter on this rank).
    lane_ctx: Vec<Vec<(CudaSlice<u16>, CudaSlice<u16>)>>,
    ctx_live: Option<usize>,
    /// the request whose speculation state (`sp_cur`, `seed_cur`, `smp_round`) the engine holds (G1: lanes load theirs)
    pub spec_id: u64,
}

/// SWA ring depth (positions): a power of two >= 2 x the 128 window; small, the ring is per rank.
pub const SWA_RING: usize = 4096;

/// Prefix reuse (`--prefix-cache on`, one request at a time): without a message boundary from the template, the
/// snapshot goes at least this many tokens before the prompt's end (the next turn re-renders the generation prompt)
pub const PREFIX_MARGIN: usize = 64;

/// Prefix reuse (`--prefix-cache on`, one request at a time). After a request ends the slot still holds every
/// position's K/V in the full-length caches (DSA layers, the indexer keys), so a prompt that starts with the previous
/// prompt's first `at` tokens can resume its prefill there. `at` is always a prefill CHUNK BOUNDARY, so the resumed
/// prefill runs exactly the chunks a fresh one would and leaves the same state, bit for bit. The sliding windows do not
/// survive a request (its generation wraps the rings), so their state at `at` is copied aside when the chunk ending at
/// `at` completes and put back when a request resumes there: the target's SWA entries of [at - window, at) and the
/// drafter's whole context ring.
pub struct PrefixSnap {
    /// the prompt tokens [0, at) the slot holds, with their snapshot (None = nothing reusable)
    pub toks: Option<Vec<u32>>,
    /// the snapshot holds the drafter context too (the prefill that took it fed the drafter)
    pub dr_ok: bool,
    /// (target layer index, K window [kv][window][hd], V window [kv][window][vd]) per SWA layer
    swa: Vec<(usize, CudaSlice<u16>, CudaSlice<u16>)>,
    /// the drafter's context rings (K, V) per layer, whole
    dr: Vec<(CudaSlice<u16>, CudaSlice<u16>)>,
}


/// The ring slots of positions [p0, p0 + n) in a ring of `s` slots (a power of two, n <= s): one or two contiguous
/// runs (first slot, length), in position order.
pub fn ring_runs(p0: usize, n: usize, s: usize) -> Vec<(usize, usize)> {
    let a = p0 & (s - 1);
    let first = n.min(s - a);
    if first == n { vec![(a, n)] } else { vec![(a, first), (0, n - first)] }
}

/// `Engine::prefix_plan` without the engine: `held` = the prompt prefix the slot holds a snapshot of.
pub fn plan_prefix(held: Option<&[u32]>, prompt: &[u32], ckpt_at: Option<usize>, chunk: usize, window: usize) -> (usize, Option<usize>) {
    let n = prompt.len();
    if n == 0 { return (0, None); }
    let reuse = match held { Some(t) if n > t.len() && prompt[..t.len()] == t[..] => t.len(), _ => 0 };
    let end = ckpt_at.unwrap_or(n.saturating_sub(PREFIX_MARGIN)).min(n - 1);
    let c = end / chunk * chunk;
    (reuse, (c > reuse && c >= window).then_some(c))
}

/// `--prefix-cache on` (the posture's flag) turns the Naive prefix reuse on; absent or `off` = every prompt prefilled whole.
pub fn prefix_flag(args: &[String]) -> bool {
    crate::arg_value(args, "--prefix-cache").map(|s| s == "on" || s == "1").unwrap_or(false)
}
/// prefill rows per step (default; `--prefill-chunk`): every expert's weights are read once per chunk, so
/// wider is cheaper per token; the workspace and the TP partial scale with it. 2048 since 2026-10-21 (owner decision:
/// +27% prefill at TP4, +36% at TP2; long-prompt streams differ from 1024, whose chunk boundaries fall elsewhere)
pub const PREFILL_CHUNK: usize = 2048;

// `super::<module>::` paths in the moved bodies resolve through these re-imports (children sit one level deeper).
use super::{config, cpu, decode, drafter_gpu, gpu, model};

mod engine;
mod flags;
mod loops;
use self::loops::*;
mod head;
mod probes;
pub use self::flags::*;
pub use self::head::*;
pub use self::probes::*;

#[cfg(test)]
mod prefix_tests;
