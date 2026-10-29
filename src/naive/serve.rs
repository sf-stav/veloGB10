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

impl Engine {
    /// `n_layers` < the model's = a SMOKE configuration (garbage text, the full serve machinery): the
    /// `--layers N` flag, shipped to the node in the head's argv.
    pub fn load(dev: Arc<CudaDevice>, pack: &NaivePack, shard: Shard, max_pos: usize, n_layers: usize, head: bool, chunk: usize) -> Result<Engine> {
        Self::load_spec(dev, pack, shard, max_pos, n_layers, head, chunk, None)
    }

    /// `spec` = Some: the verify block step on every rank; the drafter + full lm_head on the head only.
    #[allow(clippy::too_many_arguments)]
    pub fn load_spec(dev: Arc<CudaDevice>, pack: &NaivePack, shard: Shard, max_pos: usize, n_layers: usize, head: bool, chunk: usize,
                     spec: Option<&SpecCfg>) -> Result<Engine> {
        Self::load_spec_lanes(dev, pack, shard, max_pos, n_layers, head, chunk, spec, 1)
    }

    /// `load_spec` with G1 lanes (`--naive-lanes N`): N KV slots, the shared lane table, the m = 2..=N lane steps.
    #[allow(clippy::too_many_arguments)]
    pub fn load_spec_lanes(dev: Arc<CudaDevice>, pack: &NaivePack, shard: Shard, max_pos: usize, n_layers: usize, head: bool, chunk: usize,
                           spec: Option<&SpecCfg>, lanes: usize) -> Result<Engine> {
        ensure!((1..=4).contains(&lanes), "--naive-lanes {lanes} (1..=4)");
        crate::metrics::NAIVE_LANES.store(lanes as u64, std::sync::atomic::Ordering::Relaxed);
        // the lane count is this server's capacity: running vs waiting and the uptime clock come from it (was never set: "up 0 min")
        crate::metrics::set_max_batch(lanes);
        let t0 = std::time::Instant::now();
        ensure!(chunk >= 1 && chunk <= super::model::MAX_PREFILL_ROWS, "--prefill-chunk {chunk} (1..{})", super::model::MAX_PREFILL_ROWS);
        // every row of a chunk writes its K/V before any row attends: the SWA ring must hold the chunk + the window
        ensure!(chunk + pack.cfg.sliding_window <= SWA_RING, "--prefill-chunk {chunk} + window {} exceeds the SWA ring {SWA_RING}", pack.cfg.sliding_window);
        ensure!(n_layers >= 1 && n_layers <= pack.cfg.num_layers, "--layers {n_layers}");
        if n_layers < pack.cfg.num_layers { println!("[naive-serve] SMOKE configuration: {n_layers} of {} layers — the text is NOT the model's", pack.cfg.num_layers); }
        let model = GpuModel::load_slice_on(dev.clone(), pack, n_layers, max_pos, SWA_RING, lanes, shard)?;
        crate::dprintln!("[naive-serve] rank {}/{} shard loaded ({n_layers} layers, max_pos {max_pos}) in {:.1} s", shard.rank, shard.world,
                         t0.elapsed().as_secs_f64());
        let pf_ws = Ws::new(&dev, ws_bytes(&pack.cfg, chunk, max_pos, 0))?;
        let (mut ver, mut hs, mut pf_taps, mut block, mut conf) = (None, None, None, 1usize, 0f32);
        let mut draft_temp: f32 = SPEC_DRAFT_TEMP;
        let mut sharded = false;
        let mut batched = false;
        let mut guard = (SPEC_GUARD_OFF, SPEC_GUARD_ON, SPEC_PROBE_EVERY);
        let mut vers = Vec::new();
        // the plain step carries the drafter's taps too when speculation is on: a plain step's row is
        // context the drafter must see (a hole in its context after every plain step cost acceptance)
        let mut dec_taps: Vec<usize> = Vec::new();
        if let Some(sc) = spec {
            ensure!((2..=MAX_ROWS).contains(&sc.block), "--spec-block {} (2..=16)", sc.block);
            block = sc.block;
            conf = sc.conf;
            draft_temp = sc.draft_temp;
            guard = (sc.guard_off, sc.guard_on, sc.probe_every);
            let h = pack.cfg.hidden_size;
            let mut taps: Vec<usize> = Vec::new();
            // work item B: where the drafter runs (graceful fallback, never a refusal: a combination the sharded
            // drafter cannot serve runs head-only and says so)
            let world = model.shard.world;
            let want_sharded = match sc.drafter.as_str() { "sharded" => true, "head" => false, _ => world > 1 };
            sharded = want_sharded && world > 1 && sc.draft_temp <= 0.0;
            if want_sharded && world > 1 && !sharded {
                println!("[naive-serve] --spec-drafter {}: sampled drafts (--spec-draft-temp {}) need the whole vocabulary's q on one rank; running the drafter HEAD-ONLY instead", sc.drafter, sc.draft_temp);
            }
            if head || sharded {
                let dd = sc.draft_dir.as_ref().context("the head needs --draft-dir (or the drafter next to the pack)")?;
                super::drafter_gpu::load_module(&dev)?;
                let drafter = if sharded { GpuDrafter::load_sharded(&dev, dd, "model.safetensors", model.shard, model.v0, model.vl)? }
                              else { GpuDrafter::load(&dev, dd, "model.safetensors")? };
                ensure!(drafter.cfg.hidden == h && drafter.cfg.vocab == pack.cfg.vocab_size, "drafter geometry does not match the target");
                taps = drafter.cfg.taps.clone();
                let lm_full = if sharded { None } else { Some(super::gpu::upload_quad(&dev, pack, pack.quad("lm_head")?)?) };
                let ws = Ws::new(&dev, 512 << 20)?;
                pf_taps = Some(dev.alloc_zeros::<f32>(taps.len() * chunk * h)?);
                crate::dprintln!("[naive-serve] S8 drafter loaded from {dd}: {} layers, taps {:?}, block {block}, {} (rank {}/{}: {} q heads, {} kv heads, inter {}, fc rows {}, vocab [{}, {}))",
                                 drafter.cfg.layers, taps, if sharded { "SHARDED" } else { "head-only" }, model.shard.rank, world,
                                 drafter.nh_l, drafter.kv_l, drafter.inter_l, drafter.h_l, drafter.v0, drafter.v0 + drafter.vl);
                hs = Some(HeadSpec { drafter, lm_full, ws,
                                     qbuf: dev.alloc_zeros::<f32>(16 * pack.cfg.vocab_size + 2 * 1024)?,
                                     qstat: dev.alloc_zeros::<f32>(2 * (block - 1))? });
            }
            for m in 2..block { vers.push(DecodeStep::new_with_taps(&model, m, head, taps.clone())?); }
            ver = Some(DecodeStep::new_with_taps(&model, block, head, taps.clone())?);
            // work item C: the verify sampler's exchange pattern, identical on every rank (the doorbell sequence must match)
            batched = match sc.sampler.as_str() { "batched" => true, "rows" => false, _ => world > 1 };
            for v in vers.iter_mut() { v.set_batched(batched); }
            if let Some(v) = ver.as_mut() { v.set_batched(batched); }
            if head { println!("[naive-serve] verify sampler: {} (--spec-sampler {})", if batched { "BATCHED exchange (one per phase for all rows)" } else { "per-row exchange" }, sc.sampler); }
            // the verify graphs bake the q address at capture: assign it now, before `arm()`
            if let Some(h) = hs.as_ref() {
                let (qp, qz) = (dptr(&h.qbuf), dptr(&h.qstat));
                for v in vers.iter_mut() { v.smp_q = qp; v.smp_qstat = qz; v.smp_stride = pack.cfg.vocab_size; }
                if let Some(v) = ver.as_mut() { v.smp_q = qp; v.smp_qstat = qz; v.smp_stride = pack.cfg.vocab_size; }
            }
            dec_taps = taps;
        }
        let mut dec = DecodeStep::new_with_taps(&model, 1, head, dec_taps.clone())?;
        // G1: one sampler table for every step, and the multi-request steps (each with the drafter's taps, so the
        // request that owns the drafter context keeps feeding it while it shares a step)
        let (mut lane_tab, mut lsteps) = (None, Vec::new());
        if lanes > 1 {
            let tab = super::decode::LaneTab::new(&model, lanes)?;
            dec.use_lane_tab(&tab);
            for v in vers.iter_mut() { v.use_lane_tab(&tab); }
            if let Some(v) = ver.as_mut() { v.use_lane_tab(&tab); }
            for m in 2..=lanes {
                let mut st = DecodeStep::new_with_taps(&model, m, head, dec_taps.clone())?;
                st.use_lane_tab(&tab);
                st.set_multilane(true);
                lsteps.push(st);
            }
            lane_tab = Some(tab);
        }
        let xd = dev.alloc_zeros::<f32>(32)?;
        // G1 + speculation: a context per lane (a lane's ring set is the drafter's own geometry)
        let mut lane_ctx = Vec::new();
        if lanes > 1 && spec.is_some_and(|sc| sc.lane_ctx) {
            if let Some(h) = hs.as_ref() {
                for _ in 0..lanes {
                    let mut set = Vec::new();
                    for l in &h.drafter.layers {
                        set.push((dev.alloc_zeros::<u16>(cudarc::driver::DeviceSlice::len(&l.kc))?, dev.alloc_zeros::<u16>(cudarc::driver::DeviceSlice::len(&l.vc))?));
                    }
                    lane_ctx.push(set);
                }
            }
        }
        crate::metrics::NAIVE_LANE_CTX.store(!lane_ctx.is_empty(), std::sync::atomic::Ordering::Relaxed);
        dev.synchronize()?;
        Ok(Engine { model, max_pos, pf_ws, dec, chunk, ver, vers, spec: hs, pf_taps, xd, block, conf, draft_temp, last_width: 0, guard,
                    sp_cur: SampleParams::GREEDY, seed_cur: 0, smp_round: 0, smp_fallbacks: 0, smp_nolist: 0, sharded, batched, node_drafts: None, draft_secs: 0.0, draft_onehot: true,
                    lanes, lane_tab, lsteps, pfx: Vec::new(), lane_ctx, ctx_live: None, spec_id: u64::MAX })
    }

    /// Prefix reuse: allocate the snapshot buffers (`--prefix-cache on`), one set per KV slot (one per lane).
    pub fn enable_prefix(&mut self) -> Result<()> {
        let dev = self.model.dev.clone();
        let (w, hd, vd) = (self.model.cfg.sliding_window, self.model.cfg.head_dim, self.model.cfg.v_head_dim);
        let mut sets = Vec::new();
        for _ in 0..self.model.n_slots {
            let mut swa = Vec::new();
            for (i, l) in self.model.layers.iter().enumerate() {
                if l.kind != super::config::AttnKind::Swa { continue; }
                ensure!(w <= l.slots, "window {w} > ring {}", l.slots);
                swa.push((i, dev.alloc_zeros::<u16>(l.kv_l * w * hd)?, dev.alloc_zeros::<u16>(l.kv_l * w * vd)?));
            }
            let mut dr = Vec::new();
            if let Some(hs) = self.spec.as_ref() {
                for l in &hs.drafter.layers {
                    dr.push((dev.alloc_zeros::<u16>(cudarc::driver::DeviceSlice::len(&l.kc))?, dev.alloc_zeros::<u16>(cudarc::driver::DeviceSlice::len(&l.vc))?));
                }
            }
            sets.push(PrefixSnap { toks: None, dr_ok: false, swa, dr });
        }
        dev.synchronize()?;
        self.pfx = sets;
        Ok(())
    }

    pub fn pfx_on(&self) -> bool { !self.pfx.is_empty() }

    /// G1 + prefix reuse: the free KV slot a new request should take: the one whose snapshot it resumes furthest into,
    /// else one holding no snapshot (keep the others' conversations), else the most recently freed. Returns (index into
    /// `free`, resume point, snapshot point).
    pub fn prefix_pick(&self, free: &[usize], prompt: &[u32], ckpt_at: Option<usize>, feeds: bool) -> (usize, usize, Option<usize>) {
        let last = free.len() - 1;
        if !self.pfx_on() { return (last, 0, None); }
        let mut best = (last, 0usize, self.prefix_plan(free[last], prompt, ckpt_at, feeds).1);
        let mut empty: Option<usize> = None;
        for (i, &sl) in free.iter().enumerate() {
            let (r, sn) = self.prefix_plan(sl, prompt, ckpt_at, feeds);
            if r > best.1 { best = (i, r, sn); }
            if self.pfx[sl].toks.is_none() && empty.is_none() { empty = Some(i); }
        }
        if best.1 == 0 { if let Some(i) = empty { return (i, 0, self.prefix_plan(free[i], prompt, ckpt_at, feeds).1); } }
        best
    }

    /// Prefix reuse, the head's decision for a prompt: (the position its prefill resumes at, 0 = from the start; the
    /// chunk boundary its own snapshot goes to). `ckpt_at` = the template's message boundary (what the next turn
    /// replays verbatim), else the prompt less `PREFIX_MARGIN`; the snapshot is the last chunk boundary at or before it.
    /// `feeds`: this prefill feeds the drafter, so a snapshot without the drafter's context cannot be resumed.
    pub fn prefix_plan(&self, slot: usize, prompt: &[u32], ckpt_at: Option<usize>, feeds: bool) -> (usize, Option<usize>) {
        if !self.pfx_on() { return (0, None); }
        let p = &self.pfx[slot];
        let held = if feeds && self.spec.is_some() && !p.dr_ok { None } else { p.toks.as_deref() };
        plan_prefix(held, prompt, ckpt_at, self.chunk, self.model.cfg.sliding_window)
    }

    /// `to_snap`: the sliding-window state at position `at` -> the snapshot (else back into the rings). Every SWA
    /// layer's entries of [at - window, at) (per kv head, one or two runs of the ring) and the drafter's whole rings.
    fn prefix_copy(&self, slot: usize, at: usize, to_snap: bool, drafter: bool) -> Result<()> {
        let s = self.model.stream.stream;
        let (w, hd, vd) = (self.model.cfg.sliding_window, self.model.cfg.head_dim, self.model.cfg.v_head_dim);
        ensure!(at >= w, "prefix snapshot at {at} < window {w}");
        let cp = |dst: u64, src: u64, bytes: usize| -> Result<()> {
            unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, s) }.map_err(|e| anyhow::anyhow!("prefix snapshot copy: {e:?}"))
        };
        let ps = &self.pfx[slot];
        for (li, ks, vs) in &ps.swa {
            let l = &self.model.layers[*li];
            for (cache, snap, d) in [(dptr(&l.kc), dptr(ks), hd), (dptr(&l.vc), dptr(vs), vd)] {
                // the KV slot's block of the cache: [n_slots][kv][S][d]
                let cache = cache + ((slot * l.kv_l * l.slots * d) * 2) as u64;
                for kh in 0..l.kv_l {
                    let mut off = 0usize;
                    for (rs, n) in ring_runs(at - w, w, l.slots) {
                        let r = cache + (((kh * l.slots + rs) * d) * 2) as u64;
                        let q = snap + (((kh * w + off) * d) * 2) as u64;
                        if to_snap { cp(q, r, n * d * 2)?; } else { cp(r, q, n * d * 2)?; }
                        off += n;
                    }
                }
            }
        }
        if let Some(hs) = self.spec.as_ref().filter(|_| drafter) {
            for (l, (ks, vs)) in hs.drafter.layers.iter().zip(&ps.dr) {
                for (cache, snap, len) in [(dptr(&l.kc), dptr(ks), cudarc::driver::DeviceSlice::len(&l.kc)), (dptr(&l.vc), dptr(vs), cudarc::driver::DeviceSlice::len(&l.vc))] {
                    if to_snap { cp(snap, cache, len * 2)?; } else { cp(cache, snap, len * 2)?; }
                }
            }
        }
        Ok(())
    }

    /// The prompt's prefill with prefix reuse: resume at `reuse` (the slot must hold exactly that prefix: every rank
    /// checks its own), take this prompt's snapshot at `snap`. Same chunks as a whole prefill, so the same state.
    /// `slot` = the KV slot the prompt is prefilled into (lanes 1: 0); `feed` = the prefill feeds the drafter (its context
    /// is then part of the snapshot; with per-lane contexts the caller made the lane's context live first).
    pub fn prefill_resume(&mut self, slot: usize, prompt: &[u32], reuse: usize, snap: Option<usize>, feed: bool) -> Result<Option<u64>> {
        use std::sync::atomic::Ordering::Relaxed;
        let dr = feed && self.spec.is_some();
        if reuse > 0 {
            ensure!(self.pfx_on() && self.pfx[slot].toks.as_ref().is_some_and(|t| t.len() == reuse && prompt.len() > reuse && prompt[..reuse] == t[..])
                    && (!dr || self.pfx[slot].dr_ok),
                    "prefix reuse at {reuse} in slot {slot}: this rank's slot does not hold that prefix (the ranks' prefix state diverged)");
            self.prefix_copy(slot, reuse, false, dr)?;
            crate::metrics::NAIVE_PREFIX_HITS.fetch_add(1, Relaxed);
            crate::metrics::NAIVE_PREFIX_REUSED.fetch_add(reuse as u64, Relaxed);
        } else if self.pfx_on() {
            crate::metrics::NAIVE_PREFIX_MISSES.fetch_add(1, Relaxed);
            self.pfx[slot].toks = None; // the prefill below rewrites the slot from position 0
        }
        if let Some(c) = snap.filter(|&c| self.pfx_on() && c > reuse && c < prompt.len()) {
            self.prefill_feed(&prompt[reuse..c], reuse, false, feed)?;
            self.model.sync()?;
            self.prefix_copy(slot, c, true, dr)?;
            self.pfx[slot].toks = Some(prompt[..c].to_vec());
            self.pfx[slot].dr_ok = dr;
            return self.prefill_feed(&prompt[c..], c, true, feed);
        }
        self.prefill_feed(&prompt[reuse..], reuse, true, feed)
    }

    /// The verify width for this round's draft confidences: the longest prefix clearing the threshold.
    pub fn pick_width(&self, conf: &[f32]) -> usize {
        let n = self.block - 1;
        if self.conf <= 0.0 { return n.min(conf.len()); }
        conf.iter().take(n).take_while(|&&c| c >= self.conf).count()
    }

    /// True when the last speculative round ran as a plain step (verify width 0): its token was counted by
    /// the plain step's own note kernel, so the commit bookkeeping must not count it into `dec` again.
    pub fn last_round_plain(&self) -> bool { self.last_width == 0 }

    /// Every verify step sees the head's cancel flag for the next round.
    pub fn set_cancel_all(&mut self, cancel: bool) -> Result<()> {
        self.dec.set_cancel(&self.model, cancel)?;
        for v in self.vers.iter_mut() { v.set_cancel(&self.model, cancel)?; }
        if let Some(v) = self.ver.as_mut() { v.set_cancel(&self.model, cancel)?; }
        Ok(())
    }

    /// Committed tokens into every step's penalty table. A plain step already counted its own token into
    /// `dec` (the step's note kernel), so `from_plain` skips `dec`; a verify round counted nothing itself.
    pub fn note_commit_all(&mut self, toks: &[u32], from_plain: bool) -> Result<()> {
        if !from_plain { self.dec.note_commit(&self.model, toks)?; }
        for v in self.vers.iter_mut() { v.note_commit(&self.model, toks)?; }
        if let Some(v) = self.ver.as_mut() { v.note_commit(&self.model, toks)?; }
        Ok(())
    }

    /// A plain step inside a speculative request: the sampled token, and its row appended to the drafter's
    /// context (the head), so the next draft sees it.
    pub fn plain_step_spec(&mut self, t: u32, pos: usize) -> Result<(u32, bool)> {
        let r = self.dec.step(&self.model, &[t], &[pos])?;
        if let Some(hs) = self.spec.as_mut() {
            let taps = self.dec.tap_view(&self.model);
            hs.ws.reset();
            hs.drafter.append_context(&self.model.stream, &mut hs.ws, &taps, 1, pos, self.model.tp.as_ref())?;
        }
        Ok(r)
    }

    /// One speculative round on every rank. The head passes its drafts (block - 1 tokens); the node passes
    /// None and reads them from the slot exchange. Returns (committed tokens, rows accepted a, cancel flag):
    /// committed = drafts[..a] + the bonus sample of row a (a + 1 tokens, positions pos+1 ..). The target
    /// rows [t_last, d_1..d_{B-1}] at positions pos.. are verified in one graphed step; row j's sample is
    /// accepted while it equals d_{j+1} ("match": the samples are exact draws from the target's sampled
    /// distribution, so the output distribution is the plain one at every temperature; greedy is exact).
    pub fn spec_round(&mut self, drafts: Option<(&[u32], usize)>, t_last: u32, pos: usize) -> Result<(Vec<u32>, usize, bool)> {
        let b = self.block;
        let n = b - 1;
        // the drafts exchange: head writes slot 0..n and the verify width k in slot 31, node zeros; the
        // all-reduce gathers (adding zeros), so every rank runs the same width on the same drafts
        let dev = self.model.dev.clone();
        let mut slots = vec![0f32; 32];
        if let Some((d, k)) = drafts {
            ensure!(d.len() == n, "drafts {} != block-1 {n}", d.len());
            ensure!(k <= n, "verify width {k} > {n} drafts");
            for (i, t) in d.iter().enumerate() { slots[i] = *t as f32; }
            slots[31] = k as f32;
        }
        self.model.sync()?;
        dev.htod_sync_copy_into(&slots, &mut self.xd)?;
        if let Some(tp) = &self.model.tp { tp.all_reduce(&self.model.stream, &self.xd, 32)?; self.model.sync()?; }
        let got = dev.dtoh_sync_copy(&self.xd)?;
        let k = got[31] as usize;
        ensure!(k <= n, "exchanged verify width {k} > {n}");
        if let Some((nd, nk)) = self.node_drafts.take() {
            // sharded drafter: this node drafted the same block itself; the exchanged (head) drafts must match
            let ex: Vec<u32> = got[..n].iter().map(|x| *x as u32).collect();
            ensure!(ex == nd && k == nk, "SHARDED DRAFTER DIVERGED: node {}/{} drafts {nd:?} width {nk}, head {ex:?} width {k} (pos {pos})",
                    self.model.shard.rank, self.model.shard.world);
        }
        self.last_width = k;
        if k == 0 {
            // nothing confident enough to verify: a plain step (its row still feeds the drafter's context)
            let (tok, cancel) = self.plain_step_spec(t_last, pos)?;
            return Ok((vec![tok], 0, cancel));
        }
        let drafts: Vec<u32> = got[..k].iter().map(|x| *x as u32).collect();
        // the verify block: t_last + the k drafts
        let mut ids = Vec::with_capacity(k + 1); ids.push(t_last); ids.extend_from_slice(&drafts);
        let posv: Vec<usize> = (pos..pos + k + 1).collect();
        ensure!(pos + k + 1 <= self.max_pos, "speculative block past the context");
        // the verify step for k drafts (m = k + 1): `ver` is m = block, `vers[k - 1]` the narrower ones
        // (inline so the borrow is of these two fields only, not of `self.model`)
        let ver: &mut DecodeStep = if k + 1 == self.block { self.ver.as_mut().context("no verify step")? } else { &mut self.vers[k - 1] };
        // Work item A: a SAMPLED request takes the exact accept path (ratio test + residual); greedy
        // keeps the token-equality path bit-for-bit. NOTE: a node rank has no drafter (`spec` is None)
        // but MUST take this branch too — its accept kernel no-ops (smp_on stays 0) and its committed
        // list arrives in the ship below; taking the greedy branch there would pair the equality rule
        // with sampled drafts and never receive the head's tokens.
        let sampled = self.sp_cur.temperature > 0.0;
        self.smp_round += 1;
        let (xs, cancel) = if sampled {
            ver.step_block_smp(&self.model, &ids, &posv, Some((&drafts, self.seed_cur ^ (self.smp_round << 32), self.smp_round, self.draft_onehot)))?
        } else {
            ver.step_block(&self.model, &ids, &posv)?
        };
        let mut a = 0usize;
        let mut committed: Vec<u32>;
        if sampled {
            // rows accept independently (each has its own p, q and uniform); the block stops at the
            // first rejection, whose committed token is the residual draw
            let (acc, rtok) = ver.smp_accepts(&self.model, k)?;
            while a < k && acc[a] == 1 { a += 1; }
            committed = drafts[..a].to_vec();
            committed.push(if a < k {
                if rtok[a] < 0 {
                    self.smp_fallbacks += 1;
                    if rtok[a] == -1 { self.smp_nolist += 1; crate::metrics::NAIVE_SPEC_FALLBACK_NOLIST.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                    else { crate::metrics::NAIVE_SPEC_FALLBACK_EMPTY.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                    xs[a]
                } else { rtok[a] as u32 }
            } else { xs[k] });
            if self.model.shard.rank == 0 {
                crate::metrics::NAIVE_SPEC_SAMPLED_ROUNDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::metrics::NAIVE_SPEC_SAMPLED_ACCEPTED.fetch_add(a as u64, std::sync::atomic::Ordering::Relaxed);
            }
            ensure!(committed.iter().all(|&t| (t as usize) < self.model.cfg.vocab_size),
                    "spec round: committed an out-of-vocabulary id {:?} (a {a} of {k}, rtok {:?}, xs {:?})", committed, &rtok[..k], &xs[..xs.len().min(8)]);
        } else {
            while a < k && xs[a] == drafts[a] { a += 1; }
            committed = drafts[..a].to_vec();
            committed.push(xs[a]);
        }
        // TP: the chain is decided on the head (q lives only there); ship the committed list so every
        // rank commits the same tokens and writes the same KV.
        if sampled {
            if let Some(tp) = self.model.tp.as_ref() {
                let mut slots = vec![0f32; 32];
                if self.model.shard.rank == 0 {
                    for (i, t) in committed.iter().enumerate().take(31) { slots[i] = *t as f32; }
                    slots[31] = committed.len() as f32;
                }
                self.model.sync()?;
                dev.htod_sync_copy_into(&slots, &mut self.xd)?;
                tp.all_reduce(&self.model.stream, &self.xd, 32)?;
                self.model.sync()?;
                let got2 = dev.dtoh_sync_copy(&self.xd)?;
                let cl = got2[31] as usize;
                ensure!(cl >= 1 && cl <= 31, "spec ship: committed length {cl}");
                committed = got2[..cl].iter().map(|x| *x as u32).collect();
                a = cl - 1;
            }
        }
        // the drafter's context: rows 0..=a of the block (t_last and the accepted drafts) at positions pos..
        let taps = ver.tap_view(&self.model);
        if let Some(hs) = self.spec.as_mut() {
            hs.ws.reset();
            hs.drafter.append_context(&self.model.stream, &mut hs.ws, &taps, a + 1, pos, self.model.tp.as_ref())?;
        }
        Ok((committed, a, cancel))
    }

    /// A request begins on this rank: the sampler state of the plain step AND the verify block.
    pub fn begin_request(&mut self, sp: &SampleParams, seed: u64, prompt: &[u32]) -> Result<()> {
        self.begin_request_lane(sp, seed, prompt, 0)
    }

    /// G1: as `begin_request`, but the request takes lane `l` of BOTH the decode step and the verify
    /// block, so a second concurrent request cannot overwrite this one's knobs, RNG or penalty
    /// tables. `begin_request` is lane 0, which is what every single-stream caller wants unchanged.
    pub fn begin_request_lane(&mut self, sp: &SampleParams, seed: u64, prompt: &[u32], l: usize) -> Result<()> {
        // work item A: the drafter draws from the REQUEST's temperature and the accept draws are keyed
        // on (request seed, round)
        self.sp_cur = *sp;
        self.seed_cur = seed;
        self.smp_round = 0;
        self.smp_fallbacks = 0;
        self.smp_nolist = 0;
        self.draft_secs = 0.0;
        self.node_drafts = None;
        self.dec.begin_request_lane(&self.model, sp, seed, prompt, l)?;
        for (i, v) in self.vers.iter_mut().enumerate() { v.begin_request_lane(&self.model, sp, seed ^ (0x5EC0_0000_0000_0010 + i as u64), prompt, l)?; }
        if let Some(v) = self.ver.as_mut() { v.begin_request_lane(&self.model, sp, seed ^ 0x5EC0_0000_0000_0001, prompt, l)?; }
        Ok(())
    }

    /// The head's drafts for the next round: the anchor `t_last` at position `pos`, block - 1 tokens.
    pub fn draft(&mut self, t_last: u32, pos: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        let (t, c, _) = self.draft_q(t_last, pos)?;
        Ok((t, c))
    }

    /// Work item A: as `draft`, but a SAMPLED request draws d_j ~ q_j (Gumbel-max over the corrected,
    /// T-scaled logits) and materialises the full-vocabulary q the verify side needs for the ratio
    /// test. The third value says whether q is valid for this round (greedy, or no head = false).
    pub fn draft_q(&mut self, t_last: u32, pos: usize) -> Result<(Vec<u32>, Vec<f32>, bool)> {
        let n = self.block - 1;
        let sampled = self.sp_cur.temperature > 0.0;
        let (qp, qz) = match self.spec.as_ref() { Some(h) => (dptr(&h.qbuf), dptr(&h.qstat)), None => (0u64, 0u64) };
        let seed = self.seed_cur ^ ((pos as u64) << 24);
        let t0 = std::time::Instant::now();
        let draft_temp = self.draft_temp;
        let sp_t = self.sp_cur.temperature;
        let hs = self.spec.as_mut().context("no drafter on this rank")?;
        hs.ws.reset();
        let model = &self.model;
        // head-only: the full lm_head; sharded: this rank's vocabulary slice (the Markov stage gathers the argmax)
        let lm_q: &super::gpu::GpuQuad = hs.lm_full.as_ref().unwrap_or(&model.lm_head);
        let mut lm = |ws: &mut Ws, x16: u64, m: usize| -> Result<u64> { model.chain_ws(lm_q, x16, m, ws) };
        // argmax drafts unless the draft temperature is > 0 (F = 0 is the default: see SPEC_DRAFT_TEMP)
        let smp = if sampled && qp != 0 && draft_temp > 0.0 && hs.lm_full.is_some() {
            Some(super::drafter_gpu::SmpDraft { t: (sp_t * draft_temp).max(1e-3), rng: seed, smb: qp, qstat: qz })
        } else { None };
        let r = hs.drafter.round(&model.stream, &mut hs.ws, pos, t_last, n, dptr(&model.embed), &mut lm, smp, model.tp.as_ref())?;
        let ok = smp.is_some();
        self.draft_onehot = smp.is_none();
        self.draft_secs += t0.elapsed().as_secs_f64();
        Ok((r.tokens, r.confidence, ok))
    }

    /// Record the decode graph (after the TP transport is attached: the reduces are in the graph).
    pub fn arm(&mut self) -> Result<()> {
        let t0 = std::time::Instant::now();
        self.dec.capture(&self.model)?;
        for v in self.vers.iter_mut() { v.capture(&self.model)?; }
        if let Some(v) = self.ver.as_mut() { v.capture(&self.model)?; }
        for v in self.lsteps.iter_mut() { v.capture(&self.model)?; }
        crate::dprintln!("[naive-serve] decode graph captured in {:.2} s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Prefill `ids` at positions start.. in chunks of MAX_ROWS rows (eager; S6b replaces the 16-row
    /// chunks with the wide prefill path). Returns the LAST row's logits (f16 [V], a device address
    /// inside the prefill workspace, valid until the next prefill) when asked.
    pub fn prefill(&mut self, ids: &[u32], start: usize, want_logits: bool) -> Result<Option<u64>> {
        self.prefill_feed(ids, start, want_logits, true)
    }

    /// G1: the KV slot every row of the next step(s) reads and writes (the verify block, a prefill chunk: one request).
    pub fn set_slot_all(&mut self, slot: usize) -> Result<()> {
        if self.lanes <= 1 { return Ok(()); }
        self.model.sync()?;
        let n = cudarc::driver::DeviceSlice::len(&self.model.sids_buf);
        self.model.dev.htod_sync_copy_into(&vec![slot as i32; n], &mut self.model.sids_buf)?;
        Ok(())
    }

    /// G1: per-row KV slots of a multi-request step (row j = lane j's slot; the rest of the buffer keeps the last one).
    pub fn set_slots(&mut self, slots: &[usize]) -> Result<()> {
        self.model.sync()?;
        let n = cudarc::driver::DeviceSlice::len(&self.model.sids_buf);
        let mut v: Vec<i32> = slots.iter().map(|s| *s as i32).collect();
        let last = *v.last().unwrap_or(&0);
        v.resize(n, last);
        self.model.dev.htod_sync_copy_into(&v, &mut self.model.sids_buf)?;
        Ok(())
    }

    /// G1: one multi-request step over `ids`/`pos` (row j = lane j, KV slot `slots[j]`) -> each lane's token.
    /// `owner`: the lane whose request owns the drafter context; its row's taps are appended to that context
    /// (every rank that runs the drafter does the same), so the request can speculate again when it is alone.
    pub fn step_lanes(&mut self, ids: &[u32], pos: &[usize], slots: &[usize], owner: Option<usize>) -> Result<Vec<u32>> {
        let m = ids.len();
        ensure!(m >= 1 && m <= self.lanes && pos.len() == m && slots.len() == m, "lane step: {m} rows of {} lanes", self.lanes);
        self.set_slots(slots)?;
        let toks = if m == 1 { vec![self.dec.step(&self.model, ids, pos)?.0] } else { self.lsteps[m - 2].step_lanes(&self.model, ids, pos)? };
        if self.per_lane_ctx() && self.spec.is_some() {
            // every lane's row into ITS context (the same appends on every rank that runs the drafter)
            let h = self.model.cfg.hidden_size;
            for j in 0..m {
                self.ctx_make_live(j)?;
                let st: &DecodeStep = if m == 1 { &self.dec } else { &self.lsteps[m - 2] };
                let taps: Vec<(u64, usize)> = st.tap_view(&self.model).iter().map(|(a, p)| (a + (j * h * 4) as u64, *p)).collect();
                let hs = self.spec.as_mut().unwrap();
                hs.ws.reset();
                hs.drafter.append_context(&self.model.stream, &mut hs.ws, &taps, 1, pos[j], self.model.tp.as_ref())?;
            }
            return Ok(toks);
        }
        if let (Some(j), Some(hs)) = (owner, self.spec.as_mut()) {
            let st: &DecodeStep = if m == 1 { &self.dec } else { &self.lsteps[m - 2] };
            let h = self.model.cfg.hidden_size;
            let taps: Vec<(u64, usize)> = st.tap_view(&self.model).iter().map(|(a, p)| (a + (j * h * 4) as u64, *p)).collect();
            hs.ws.reset();
            hs.drafter.append_context(&self.model.stream, &mut hs.ws, &taps, 1, pos[j], self.model.tp.as_ref())?;
        }
        Ok(toks)
    }

    /// G1: every lane keeps its own drafter context (`lane_ctx`).
    pub fn per_lane_ctx(&self) -> bool { !self.lane_ctx.is_empty() }

    fn ctx_swap(&mut self, j: usize) {
        if let Some(hs) = self.spec.as_mut() {
            for (l, (k, v)) in hs.drafter.layers.iter_mut().zip(self.lane_ctx[j].iter_mut()) {
                std::mem::swap(&mut l.kc, k);
                std::mem::swap(&mut l.vc, v);
            }
        }
    }

    /// G1: lane `j`'s context into the drafter (the live one parked first). No-op without per-lane contexts.
    pub fn ctx_make_live(&mut self, j: usize) -> Result<()> {
        if self.lane_ctx.is_empty() || self.ctx_live == Some(j) { return Ok(()); }
        // no sync: the kernels already queued keep the raw addresses they were launched with, every buffer stays
        // allocated, and the next append runs on the same stream after them
        if let Some(k) = self.ctx_live.take() { self.ctx_swap(k); }
        self.ctx_swap(j);
        self.ctx_live = Some(j);
        Ok(())
    }

    /// G1 compaction: lane `from`'s sampler state into lane `to` (a finished request's lane is refilled by the last one),
    /// and its drafter context with it.
    pub fn copy_lane(&mut self, from: usize, to: usize) -> Result<()> {
        if let Some(t) = &self.lane_tab { t.copy_lane(&self.model, from, to)?; self.model.sync()?; }
        if !self.lane_ctx.is_empty() {
            if let Some(k) = self.ctx_live.take() { self.ctx_swap(k); }
            self.lane_ctx.swap(from, to);
        }
        Ok(())
    }

    /// `prefill`, optionally WITHOUT feeding the drafter's context (G1: a request admitted next to the one that owns it).
    pub fn prefill_feed(&mut self, ids: &[u32], start: usize, want_logits: bool, feed_drafter: bool) -> Result<Option<u64>> {
        ensure!(!ids.is_empty(), "empty prefill");
        ensure!(start + ids.len() <= self.max_pos, "prompt of {} tokens at {start} exceeds the context ({})", ids.len(), self.max_pos);
        let v = self.model.vl; // the logits are this rank's vocabulary SLICE (vocab-parallel head)
        let dev = self.model.dev.clone();
        let mut out = None;
        let n = ids.len();
        let mut i = 0;
        while i < n {
            let m = (n - i).min(self.chunk);
            let last = i + m == n;
            self.model.sync()?;
            let ids_d = dev.htod_sync_copy(&ids[i..i + m].iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
            let pos_d = dev.htod_sync_copy(&(start + i..start + i + m).map(|x| x as i32).collect::<Vec<i32>>())?;
            let sel: Option<(Vec<usize>, u64)> = match (&self.spec, &self.pf_taps) { (Some(hs), Some(pt)) => Some((hs.drafter.cfg.taps.clone(), dptr(pt))), _ => None };
            let sel_ref: Option<(&[usize], u64)> = sel.as_ref().map(|(l, d)| (l.as_slice(), *d));
            let lg = self.model.step_kernels_sel(&mut self.pf_ws, dptr(&ids_d), dptr(&pos_d), m, want_logits && last, None, sel_ref)?;
            self.model.sync()?; // the chunk's ids/pos buffers die here
            if let Some(tp) = &self.model.tp { tp.check_link("after a prefill chunk")?; }
            if let (true, Some(hs), Some(pt)) = (feed_drafter, self.spec.as_mut(), &self.pf_taps) {
                // the drafter's context for the prompt rows (head only)
                let h = self.model.cfg.hidden_size;
                let taps: Vec<(u64, usize)> = (0..hs.drafter.cfg.taps.len()).map(|j| (dptr(pt) + (j * m * h * 4) as u64, h)).collect();
                hs.ws.reset();
                hs.drafter.append_context(&self.model.stream, &mut hs.ws, &taps, m, start + i, self.model.tp.as_ref())?;
                self.model.sync()?;
            }
            // a 1-row logits buffer for a wide chunk, the last of m rows for a small one
            if let Some(p) = lg { out = Some(if m > MAX_ROWS { p } else { p + ((m - 1) * v * 2) as u64 }); }
            i += m;
        }
        Ok(out)
    }

    /// One graphed decode step: token `t` at `pos` -> (the next sampled token, the head's cancel flag),
    /// identical on every rank.
    pub fn decode_step(&mut self, t: u32, pos: usize) -> Result<(u32, bool)> {
        ensure!(pos < self.max_pos, "position {pos} beyond the context ({})", self.max_pos);
        self.dec.step(&self.model, &[t], &[pos])
    }
}

/// `--naive-pdl=0|1` (default 0 until the A/B on the posture lands): programmatic dependent launch for
/// every naive kernel (gpu.rs `set_pdl`). Read from the head's argv on every rank, so the ranks agree.
/// Default ON since 2026-10-20 (owner overnight: the A/B on THE POSTURE at TP4, build 9fed3324: plain step 21.2 -> 20.8 ms,
/// every speculative cell ~1% faster, 7/7 greedy + sampled streams byte-identical, cancel PASS). `--naive-pdl=0` turns it off.
/// The Naive server's lane cap (G1 serves 1..=4 requests per shared step).
pub const MAX_LANES: usize = 4;

/// G1 concurrent requests: the standard `--max-batch` (owner 2026-10-21; absent = the generic default 8), capped at
/// MAX_LANES; `--naive-lanes N` (N >= 1) still overrides it. Returns (lanes, what was asked). Head argv, every rank agrees.
pub fn lanes_asked(args: &[String]) -> (usize, usize) {
    let num = |f: &str| crate::arg_value(args, f).and_then(|s| s.parse::<usize>().ok());
    let asked = num("--naive-lanes").filter(|&n| n >= 1).or_else(|| num("--max-batch")).unwrap_or(8).max(1);
    (asked.min(MAX_LANES), asked)
}

pub fn lanes_flag(args: &[String]) -> usize { lanes_asked(args).0 }

pub fn pdl_flag(args: &[String]) -> bool {
    crate::arg_value(args, "--naive-pdl").map(|s| s == "1" || s == "on").unwrap_or(true)
}
/// `--naive-fuse-combine=0|1` (default 1): the F13 fused split-K combine. Head argv, every rank agrees.
pub fn fuse_flag(args: &[String]) -> bool {
    crate::arg_value(args, "--naive-fuse-combine").map(|s| s != "0" && s != "off").unwrap_or(true)
}

/// ISSUES 98: the drafter dir the HEAD ships to its nodes (through the cluster sync's draft round, the same
/// content-addressed blob cache the DFlash2 / DSpark drafters use): `--draft-dir` when it exists on the head, else the
/// drafter next to the pack; None with `--spec-source none|off` or when no drafter is on the head's disk. Prints nothing.
pub fn head_draft_dir(args: &[String], model_dir: &str) -> Option<String> {
    let src = crate::arg_value(args, "--spec-source").unwrap_or("dspark");
    if src == "none" || src == "off" { return None; }
    let ok = |p: &std::path::Path| p.join("config.json").is_file() && p.join("model.safetensors").is_file();
    if let Some(e) = crate::arg_value(args, "--draft-dir") {
        if ok(std::path::Path::new(e)) { return Some(e.to_string()); }
    }
    std::path::Path::new(model_dir.trim_end_matches('/')).parent().map(|p| p.join("Naive-N0.5-Flash-FP8-Draft"))
        .filter(|p| ok(p)).map(|p| p.to_string_lossy().to_string())
}

/// The speculative configuration from the head's argv: `--spec-source dspark|none` (default dspark when the
/// drafter dir exists), `--draft-dir <dir>` (default: `Naive-N0.5-Flash-FP8-Draft` next to the pack), `--spec-block B`.
pub fn spec_cfg(args: &[String], model_dir: &str) -> Option<SpecCfg> {
    let src = crate::arg_value(args, "--spec-source").unwrap_or("dspark");
    if src == "none" || src == "off" { return None; }
    let default_dir = std::path::Path::new(model_dir.trim_end_matches('/')).parent().map(|p| p.join("Naive-N0.5-Flash-FP8-Draft"));
    // `--draft-dir` is the head's path; a node rank (work item B: it loads its shard of the drafter) may not have
    // that path (the model reaches the node through the head's sync cache), so a path that does not exist on
    // THIS rank falls back to the drafter next to the rank's model dir (the sync cache's models/ directory).
    let explicit = crate::arg_value(args, "--draft-dir").map(|s| s.to_string());
    let default_ok = default_dir.filter(|p| p.join("config.json").is_file()).map(|p| p.to_string_lossy().to_string());
    let draft_dir = match (explicit, default_ok) {
        (Some(e), _) if std::path::Path::new(&e).join("config.json").is_file() => Some(e),
        (Some(e), Some(d)) => { println!("[naive-serve] --draft-dir {e} is not on this rank; using {d}"); Some(d) }
        (Some(e), None) => Some(e),
        (None, d) => d,
    };
    let block: usize = crate::arg_value(args, "--spec-block").and_then(|s| s.parse().ok()).unwrap_or(SPEC_BLOCK);
    // `--spec-conf T`: verify the longest prefix of drafts whose confidence head says >= T (0 = always the
    // whole block, the pre-adaptive behaviour). The drafter ships a trained confidence head for exactly this.
    // Work item D (first pass, 2026-10-20 sweep at TP4 with the sharded drafter, production sampling, 3.5 bpw): the
    // cheaper round pays with a HIGHER width threshold and a LOWER guard -- conf 0.45 + guard 0.5/0.8 gave thinking
    // prose 58.7/53.4/49.5 tok/s vs 54.8/47.1/46.5 at the old defaults (0.25, 0.8/1.1) and the C-code cell 74.3/75.9 vs
    // 73.4/65.1/63.0; conf alone (0.45 / 0.6) and the guard alone were each in between. Topology-aware: world 4 takes
    // the swept values; TP2 / world 1 keep the earlier calibration until swept. The flags override either.
    let world = crate::arg_value(args, "--tp").and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    // TP2 (2026-10-20 sweep at TP2 .13+.14, build 46d2bb3b, production sampling, seed 100, prose x3 / C x2): conf 0.45 alone
    // wins (prose 40.0/36.9/37.6 vs 33.2/33.2/36.1 tok/s at the old defaults, C 60.2/36.9 vs 56.6/33.9); the lower guard
    // alone does nothing there (34.7/34.1/36.5) and conf + guard is within noise of conf alone (37.4/37.6/37.3, C
    // 55.7/39.6). The posture table at TP2 then showed thinking prose LOSING 3-6% to the plain step with guard 0.8/1.1
    // (36.2-37.4 vs 38.2-38.6 tok/s: the 26 ms plain step is cheap relative to a round); guard 1.2/1.6 on the same seeds
    // gave prose 37.9/36.5/38.1 (within noise of plain) and C 48.9/61.5/61.1 (unchanged), so world 2 takes conf 0.45 +
    // guard 1.2/1.6 (--spec-probe-every 48 on top lost 20-40% on two C trajectories: rejected). World 1 is unswept and
    // keeps the earlier calibration.
    let (d_conf, d_off, d_on) = if world >= 4 { (0.45f32, 0.5f32, 0.8f32) } else if world >= 2 { (0.45f32, 1.2f32, 1.6f32) } else { (SPEC_CONF, SPEC_GUARD_OFF, SPEC_GUARD_ON) };
    let conf: f32 = crate::arg_value(args, "--spec-conf").and_then(|s| s.parse().ok()).unwrap_or(d_conf);
    let guard_off: f32 = crate::arg_value(args, "--spec-guard-off").and_then(|s| s.parse().ok()).unwrap_or(d_off);
    let guard_on: f32 = crate::arg_value(args, "--spec-guard-on").and_then(|s| s.parse().ok()).unwrap_or(d_on);
    println!("[naive-serve] spec defaults for world {world}: --spec-conf {conf} --spec-guard-off {guard_off} --spec-guard-on {guard_on} (flags override)");
    let probe_every: usize = crate::arg_value(args, "--spec-probe-every").and_then(|s| s.parse().ok()).unwrap_or(SPEC_PROBE_EVERY);
    let draft_temp: f32 = crate::arg_value(args, "--spec-draft-temp").and_then(|s| s.parse().ok()).unwrap_or(SPEC_DRAFT_TEMP);
    let drafter = crate::arg_value(args, "--spec-drafter").unwrap_or(SPEC_DRAFTER).to_string();
    let sampler = crate::arg_value(args, "--spec-sampler").unwrap_or(SPEC_SAMPLER).to_string();
    let lane_ctx = crate::arg_value(args, "--naive-lane-ctx").map(|s| s == "on" || s == "1").unwrap_or(true);
    Some(SpecCfg { draft_dir, block, conf, guard_off, guard_on, probe_every, draft_temp, drafter, sampler, lane_ctx })
}


fn sample_params(req: &BatchRequest) -> SampleParams {
    SampleParams { temperature: req.temperature, top_p: req.top_p, top_k: req.top_k, min_p: req.min_p,
                   rep_penalty: req.rep_penalty, presence_penalty: req.presence_penalty, frequency_penalty: req.frequency_penalty }
}

// ---------------------------------------------------------------------------------------------
// Sampling (CPU, the head): the OpenAI request knobs over the f32 logits row.
// ---------------------------------------------------------------------------------------------

fn argmax(l: &[f32]) -> usize { l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 }

fn seed_of(req: &BatchRequest) -> u64 {
    match req.seed {
        Some(s) => s,
        // no client seed: a fresh random one (RandomState is seeded from the OS); the node receives it in Admit
        None => { use std::hash::{BuildHasher, Hasher}; std::collections::hash_map::RandomState::new().build_hasher().finish() }
    }
}

// ---------------------------------------------------------------------------------------------
// The head
// ---------------------------------------------------------------------------------------------

/// Run one request on the head: ship Admit, prefill, then sample / ship / stream token by token.
fn serve_one(eng: &mut Engine, req: BatchRequest, stop_ids: &[u32], nodes: &mut [std::net::TcpStream]) -> Result<()> {
    let t0 = std::time::Instant::now();
    let prompt = req.prompt.clone();
    let room = eng.max_pos.saturating_sub(prompt.len());
    if prompt.is_empty() || room == 0 {
        let _ = req.tx.send(TokEvent::Finish { reason: "length".into() });
        bail!("prompt of {} tokens leaves no room in the context ({})", prompt.len(), eng.max_pos);
    }
    let max_new = req.max_new.min(room);
    let seed = seed_of(&req);
    let (reuse, snap) = eng.prefix_plan(0, &prompt, req.ckpt_at, true);
    for n in nodes.iter_mut() { crate::cluster::send_json(n, &Ctl::Admit { prompt: prompt.clone(), max_new, seed, sp: sample_params(&req), min_new: req.min_new, ignore_eos: req.ignore_eos, reuse, snap })?; }
    let _ = req.tx.send(TokEvent::Admitted { cached_tokens: reuse as u32 });
    eng.begin_request(&sample_params(&req), seed, &prompt)?;
    let row = eng.prefill_resume(0, &prompt, reuse, snap, true)?.context("head prefill without logits")?;
    let (mut t, mut cancel) = eng.dec.sample_row(&eng.model, row)?;
    let ttft = t0.elapsed();
    let mut generated: Vec<u32> = Vec::new();
    let mut pos = prompt.len();
    let mut reason = "length";
    let td = std::time::Instant::now();
    let mut step_secs = 0f64;
    if eng.ver.is_some() {
        // S8: speculative rounds. `t` is the first sampled token (position `pos`, not yet in the caches).
        let mut rounds = 0usize; let mut accepted = 0usize;
        // the first sampled token (the first anchor) is emitted once, here; every round then emits only its
        // committed tokens (the bonus becomes the next anchor and is never emitted twice)
        let mut stop0 = false;
        {
            let is_stop = stop_ids.contains(&t) && !req.ignore_eos && generated.len() + 1 > req.min_new;
            if is_stop { reason = "stop"; stop0 = true; }
            else {
                // a dead socket is NOT a stop on this rank alone (the node cannot see it): record the reason,
                // keep the bookkeeping identical, and let the exchanged cancel flag stop every rank together
                if req.tx.send(TokEvent::Tok(t)).is_err() { reason = "cancelled"; }
                generated.push(t); if generated.len() >= max_new { stop0 = true; }
            }
            for n in nodes.iter_mut() { crate::cluster::send_json(n, &Ctl::Round { toks: vec![t], stop: stop0 })?; }
            if !stop0 { eng.note_commit_all(&[t], false)?; }
        }
        let mut guard = SpecGuard::with_every(eng.guard.0, eng.guard.1, eng.guard.2);
        let mut plain_steps = 0usize;
        let mut width_sum = 0usize;
        // confidence calibration: per 0.2-wide confidence bucket, (verified, accepted) and (skipped) counts,
        // printed on the request line so `--spec-conf` can be set from data
        let (mut hv, mut ha, mut hs) = ([0usize; 5], [0usize; 5], [0usize; 5]);
        'rounds: while !stop0 {
            let ts = std::time::Instant::now();
            let mut from_plain = false;
            let (committed, a, c) = if guard.speculate() {
                let (drafts, conf) = eng.draft(t, pos)?;
                let k = eng.pick_width(&conf);
                let r = eng.spec_round(Some((&drafts, k)), t, pos)?;
                guard.observe(r.1);
                rounds += 1; accepted += r.1; width_sum += k;
                for (i, &c) in conf.iter().enumerate().take(eng.block - 1) {
                    let b = ((c.clamp(0.0, 0.999)) * 5.0) as usize;
                    if i < k { hv[b] += 1; if i < r.1 { ha[b] += 1; } } else { hs[b] += 1; }
                }
                if k == 0 { from_plain = true; }
                r
            } else {
                // a plain step (the guard says speculation does not pay right now); same bookkeeping shape
                let (nt, c) = eng.plain_step_spec(t, pos)?;
                plain_steps += 1; from_plain = true;
                (vec![nt], 0, c)
            };
            step_secs += ts.elapsed().as_secs_f64();
            let toks_out: Vec<u32> = committed.clone();
            let mut stop = false;
            let mut emitted = Vec::new();
            for &tok in &toks_out {
                let is_stop = stop_ids.contains(&tok) && !req.ignore_eos && generated.len() + 1 > req.min_new;
                if is_stop { reason = "stop"; stop = true; break; }
                // same rule as the plain loop: a dead socket only sets the reason; the stop that ships to the
                // nodes must be the pure function every rank computes (2026-10-18 TP=4 cancel abort:
                // `node [..]/false, head [..]/true` -- identical tokens, the head stopped one round early)
                if req.tx.send(TokEvent::Tok(tok)).is_err() { reason = "cancelled"; }
                generated.push(tok); emitted.push(tok);
                if generated.len() >= max_new { stop = true; break; }
            }
            if c || cancel { stop = true; }
            for n in nodes.iter_mut() { crate::cluster::send_json(n, &Ctl::Round { toks: toks_out.clone(), stop })?; }
            eng.note_commit_all(&emitted, from_plain)?;
            if stop { break 'rounds; }
            // the next anchor: the bonus sample (last committed), at position pos + a + 1
            pos += a + 1;
            t = *committed.last().unwrap();
            cancel = c;
            eng.set_cancel_all(req.tx.is_closed())?;
            if pos + eng.block > eng.max_pos { reason = "length"; break 'rounds; }
        }
        let _ = req.tx.send(TokEvent::Finish { reason: reason.into() });
        let dt = td.elapsed().as_secs_f64();
        println!("[naive-serve] req: prompt {} tok ({reuse} reused), ttft {:.2} s ({:.0} tok/s prefill), {} generated in {dt:.2} s ({:.1} tok/s; {rounds} spec rounds ({:.2} accepted/round, {:.2} verified/round of {}) + {plain_steps} plain steps, {:.1} ms per round/step, draft {:.1} ms/round {}, sampler {}; sampled {}, residual fallbacks {} (no list {})), finish {reason}",
                 prompt.len(), ttft.as_secs_f64(), (prompt.len() - reuse) as f64 / ttft.as_secs_f64().max(1e-9), generated.len(),
                 generated.len() as f64 / dt.max(1e-9), accepted as f64 / rounds.max(1) as f64, width_sum as f64 / rounds.max(1) as f64, eng.block - 1,
                 step_secs / (rounds + plain_steps).max(1) as f64 * 1e3, eng.draft_secs / rounds.max(1) as f64 * 1e3, if eng.sharded { "sharded" } else { "head-only" }, if eng.batched { "batched" } else { "rows" }, eng.sp_cur.temperature > 0.0, eng.smp_fallbacks, eng.smp_nolist);
        println!("[naive-serve] conf: verified(acc/n) by bucket .0 .2 .4 .6 .8: {} | skipped: {:?}",
                 (0..5).map(|b| format!("{}/{}", ha[b], hv[b])).collect::<Vec<_>>().join(" "), hs);
        return Ok(());
    }
    for i in 0..max_new {
        // the stop decision is a pure function of (token, i, pos, the exchanged cancel flag): every rank
        // computes the same one; the head's Tok message is a lagging lockstep check on the node
        let is_stop = stop_ids.contains(&t) && !req.ignore_eos && generated.len() + 1 > req.min_new;
        let last = is_stop || i + 1 == max_new || pos + 1 >= eng.max_pos || cancel;
        for n in nodes.iter_mut() { crate::cluster::send_json(n, &Ctl::Tok { t, stop: last })?; }
        if is_stop { reason = "stop"; break; }
        if req.tx.send(TokEvent::Tok(t)).is_err() { reason = "cancelled"; }
        else { generated.push(t); }
        if last { break; }
        // the client's departure is seen by the NEXT step's exchange (one step late, on every rank)
        eng.dec.set_cancel(&eng.model, req.tx.is_closed())?;
        let ts = std::time::Instant::now();
        (t, cancel) = eng.decode_step(t, pos)?;
        step_secs += ts.elapsed().as_secs_f64();
        pos += 1;
    }
    let _ = req.tx.send(TokEvent::Finish { reason: reason.into() });
    let dt = td.elapsed().as_secs_f64();
    println!("[naive-serve] req: prompt {} tok ({reuse} reused), ttft {:.2} s ({:.0} tok/s prefill), {} generated in {dt:.2} s ({:.1} tok/s; step {:.1} ms of {:.1} ms per token), finish {reason}",
             prompt.len(), ttft.as_secs_f64(), (prompt.len() - reuse) as f64 / ttft.as_secs_f64().max(1e-9), generated.len(),
             generated.len() as f64 / dt.max(1e-9), step_secs / generated.len().max(1) as f64 * 1e3, dt / generated.len().max(1) as f64 * 1e3);
    Ok(())
}

/// G1: one active request on the head.
struct HLane { req: BatchRequest, slot: usize, pos: usize, t: u32, gen: usize, max_new: usize, owner: bool, t0: std::time::Instant,
               ttft: f64, prompt_len: usize, steps: usize, rounds: usize, accepted: usize, reason: &'static str, last: Vec<u32>,
               spec: LaneSpec, guard: SpecGuard }

/// G1: a lane's own speculation state (admission order id, sampler knobs, seed, its accept-draw round counter); loaded
/// into the engine when the lane speculates alone. Every rank keeps the same, in the same order.
#[derive(Clone, Copy)]
struct LaneSpec { id: u64, sp: SampleParams, seed: u64, smp_round: u64 }

/// G1: lane 0 is about to run a speculative round or a drafter-fed plain step alone: its context into the drafter, its
/// speculation state into the engine (only when another request held it).
fn lane_spec_load(eng: &mut Engine, ls: &LaneSpec) -> Result<()> {
    eng.ctx_make_live(0)?;
    if eng.spec_id != ls.id {
        eng.sp_cur = ls.sp; eng.seed_cur = ls.seed; eng.smp_round = ls.smp_round;
        eng.smp_fallbacks = 0; eng.smp_nolist = 0; eng.draft_secs = 0.0; eng.node_drafts = None;
        eng.spec_id = ls.id;
    }
    Ok(())
}

/// G1 (`--naive-lanes N`): the head's multi-request loop. Up to N requests are active at once; between steps the head
/// admits waiting requests (prefill into a free KV slot, first token), and every step serves every active request:
/// one row each in a multi-request step, or — when ONE request is active and it owns the drafter context — the
/// speculative round of the single-stream path. The nodes are pure followers (`node_lanes`): the head ships every
/// decision (finishes, admissions, the step kind) before the step, so no rank decides anything on its own.
fn serve_lanes(eng: &mut Engine, srx: &mut tokio::sync::mpsc::UnboundedReceiver<BatchRequest>, stop_ids: &[u32],
               nodes: &mut [std::net::TcpStream]) -> Result<bool> {
    let n_l = eng.lanes;
    let mut act: Vec<HLane> = Vec::new();
    let mut free: Vec<usize> = (0..n_l).rev().collect();
    let mut fin: Vec<usize> = Vec::new();
    let mut next_id = 0u64;
    let send_all = |nodes: &mut [std::net::TcpStream], c: &Ctl| -> Result<()> { for n in nodes.iter_mut() { crate::cluster::send_json(n, c)?; } Ok(()) };
    let finish = |l: HLane| {
        let _ = l.req.tx.send(TokEvent::Finish { reason: l.reason.into() });
        let dt = l.t0.elapsed().as_secs_f64() - l.ttft;
        println!("[naive-serve] req (lane): prompt {} tok, ttft {:.2} s, {} generated in {dt:.2} s ({:.1} tok/s; {} steps, {} spec rounds ({:.2} accepted/round)), finish {}",
                 l.prompt_len, l.ttft, l.gen, l.gen as f64 / dt.max(1e-9), l.steps, l.rounds, l.accepted as f64 / l.rounds.max(1) as f64, l.reason);
    };
    loop {
        // 1. the finishes of the previous step, descending (swap-remove compaction; the sampler state follows)
        fin.sort_unstable(); fin.dedup(); fin.reverse();
        let fin_now = fin.clone();
        for &j in &fin_now {
            let last = act.len() - 1;
            if j != last { eng.copy_lane(last, j)?; }
            let l = act.swap_remove(j);
            free.push(l.slot);
            finish(l);
        }
        fin.clear();
        // 2. admissions (blocking only when idle)
        let mut reqs: Vec<BatchRequest> = Vec::new();
        while act.len() + reqs.len() < n_l {
            let r = if act.is_empty() && reqs.is_empty() {
                match srx.blocking_recv() { Some(r) => r, None => { send_all(nodes, &Ctl::LIter { fin: fin_now.clone(), admits: vec![] })?; return Ok(false); } }
            } else { match srx.try_recv() { Ok(r) => r, Err(_) => break } };
            reqs.push(r);
        }
        let mut admits = Vec::new();
        let mut new_lanes = Vec::new();
        for r in reqs {
            let prompt = r.prompt.clone();
            let room = eng.max_pos.saturating_sub(prompt.len());
            if prompt.is_empty() || room < 2 {
                let _ = r.tx.send(TokEvent::Finish { reason: "length".into() });
                println!("[naive-serve] req refused before admission: prompt {} tokens leaves no room in the context ({})", prompt.len(), eng.max_pos);
                continue;
            }
            let lane = act.len() + new_lanes.len();
            let owner = act.is_empty() && new_lanes.is_empty() && eng.ver.is_some();
            ensure!(!free.is_empty(), "no free KV slot");
            // the free slot whose prefix snapshot this prompt resumes furthest into (prefix reuse per slot)
            let (fi, reuse, snap) = eng.prefix_pick(&free, &prompt, r.ckpt_at, owner || eng.per_lane_ctx());
            let slot = free.remove(fi);
            let max_new = r.max_new.min(room - 1);
            let seed = seed_of(&r);
            admits.push(LAdmit { lane, slot, prompt: prompt.clone(), max_new, seed, sp: sample_params(&r), owner, reuse, snap });
            new_lanes.push((r, slot, max_new, owner, prompt.len()));
        }
        send_all(nodes, &Ctl::LIter { fin: fin_now, admits: admits.clone() })?;
        // 3. prefill every admitted request into its slot, sample its first token
        let mut drop: Vec<usize> = Vec::new();
        for (a, (r, slot, max_new, owner, plen)) in admits.iter().zip(new_lanes) {
            let t0 = std::time::Instant::now();
            let _ = r.tx.send(TokEvent::Admitted { cached_tokens: a.reuse as u32 });
            eng.dec.begin_request_lane(&eng.model, &a.sp, a.seed, &a.prompt, a.lane)?;
            eng.set_slot_all(slot)?;
            // per-lane contexts: every request's prompt feeds ITS context (else only the owner's feeds the one context)
            if eng.per_lane_ctx() { eng.ctx_make_live(a.lane)?; }
            let row = eng.prefill_resume(slot, &a.prompt, a.reuse, a.snap, owner || eng.per_lane_ctx())?.context("prefill without logits")?;
            let (t, _) = eng.dec.sample_row_lane(&eng.model, row, a.lane)?;
            let spec = LaneSpec { id: next_id, sp: a.sp, seed: a.seed, smp_round: 0 };
            next_id += 1;
            let mut l = HLane { req: r, slot, pos: plen, t, gen: 0, max_new, owner, t0, ttft: t0.elapsed().as_secs_f64(), prompt_len: plen,
                                steps: 0, rounds: 0, accepted: 0, reason: "length", last: vec![t],
                                spec, guard: SpecGuard::with_every(eng.guard.0, eng.guard.1, eng.guard.2) };
            let is_stop = stop_ids.contains(&t) && !l.req.ignore_eos && l.req.min_new == 0;
            if is_stop { l.reason = "stop"; drop.push(a.lane); }
            else {
                if l.req.tx.send(TokEvent::Tok(t)).is_err() { l.reason = "cancelled"; drop.push(a.lane); }
                l.gen = 1;
                if l.gen >= l.max_new && !drop.contains(&a.lane) { drop.push(a.lane); }
            }
            act.push(l);
        }
        drop.sort_unstable(); drop.reverse();
        for &j in &drop {
            let last = act.len() - 1;
            if j != last { eng.copy_lane(last, j)?; }
            let l = act.swap_remove(j);
            free.push(l.slot);
            finish(l);
        }
        if act.is_empty() { send_all(nodes, &Ctl::LStep { drop, kind: 255, check: vec![] })?; continue; }
        // 4. the step
        // a request alone speculates: any lane with per-lane contexts, only the owner of the one shared context without
        let solo_spec = act.len() == 1 && (act[0].owner || eng.per_lane_ctx()) && eng.ver.is_some();
        let kind: u8 = if solo_spec { if act[0].guard.speculate() { 1 } else { 2 } } else { 0 };
        let check: Vec<Vec<u32>> = act.iter().map(|l| l.last.clone()).collect();
        send_all(nodes, &Ctl::LStep { drop, kind, check })?;
        let outs: Vec<(Vec<u32>, usize)> = match kind {
            1 => {
                let (t, pos, slot) = (act[0].t, act[0].pos, act[0].slot);
                eng.set_slot_all(slot)?;
                lane_spec_load(eng, &act[0].spec)?;
                let (drafts, conf) = eng.draft(t, pos)?;
                let k = eng.pick_width(&conf);
                let (committed, a, _) = eng.spec_round(Some((&drafts, k)), t, pos)?;
                act[0].guard.observe(a);
                act[0].spec.smp_round = eng.smp_round;
                act[0].rounds += 1; act[0].accepted += a;
                if !eng.last_round_plain() { eng.dec.note_commit_lane(&eng.model, &committed, 0)?; }
                vec![(committed, a)]
            }
            2 => {
                let (t, pos, slot) = (act[0].t, act[0].pos, act[0].slot);
                eng.set_slot_all(slot)?;
                lane_spec_load(eng, &act[0].spec)?;
                let (nt, _) = eng.plain_step_spec(t, pos)?;
                vec![(vec![nt], 0)]
            }
            _ => {
                let ids: Vec<u32> = act.iter().map(|l| l.t).collect();
                let pos: Vec<usize> = act.iter().map(|l| l.pos).collect();
                let slots: Vec<usize> = act.iter().map(|l| l.slot).collect();
                let owner = act.iter().position(|l| l.owner);
                let toks = eng.step_lanes(&ids, &pos, &slots, owner)?;
                toks.into_iter().map(|t| (vec![t], 0)).collect()
            }
        };
        // 5. deliver, stop rules (the head decides; the nodes learn the finishes at the next iteration)
        for (j, (toks, a)) in outs.into_iter().enumerate() {
            let l = &mut act[j];
            l.steps += 1;
            let mut done = false;
            for &tok in &toks {
                let is_stop = stop_ids.contains(&tok) && !l.req.ignore_eos && l.gen + 1 > l.req.min_new;
                if is_stop { l.reason = "stop"; done = true; break; }
                if l.req.tx.send(TokEvent::Tok(tok)).is_err() { l.reason = "cancelled"; done = true; break; }
                l.gen += 1;
                if l.gen >= l.max_new { l.reason = "length"; done = true; break; }
            }
            l.pos += a + 1;
            l.t = *toks.last().unwrap();
            l.last = toks;
            if l.req.tx.is_closed() { l.reason = "cancelled"; done = true; }
            if l.pos + eng.block + 1 > eng.max_pos { l.reason = "length"; done = true; }
            if done { fin.push(j); }
        }
    }
}

/// G1: the node's follower of `serve_lanes`: the same admissions, compactions and steps, in the head's order.
fn node_lanes(eng: &mut Engine, stream: &mut std::net::TcpStream, first: Ctl) -> Result<()> {
    struct NLane { slot: usize, pos: usize, t: u32, owner: bool, last: Vec<u32>, spec: LaneSpec }
    let mut act: Vec<NLane> = Vec::new();
    let mut next_id = 0u64;
    let mut guard_obs: Vec<usize> = Vec::new(); let _ = &mut guard_obs;
    let mut msg = Some(first);
    loop {
        let m = match msg.take() { Some(m) => m, None => crate::cluster::recv_json::<Ctl>(stream).context("control stream (head gone?)")? };
        let (fin, admits) = match m { Ctl::LIter { fin, admits } => (fin, admits), Ctl::Shutdown => return Ok(()), other => bail!("lanes: unexpected {other:?}") };
        for &j in &fin { let last = act.len() - 1; if j != last { eng.copy_lane(last, j)?; } act.swap_remove(j); }
        if admits.is_empty() && act.is_empty() && !fin.is_empty() {
            // the head's request channel closed after these finishes (its last LIter); keep following
        }
        for a in &admits {
            ensure!(a.lane == act.len(), "lanes: admitted lane {} but {} active", a.lane, act.len());
            eng.dec.begin_request_lane(&eng.model, &a.sp, a.seed, &a.prompt, a.lane)?;
            eng.set_slot_all(a.slot)?;
            if eng.per_lane_ctx() { eng.ctx_make_live(a.lane)?; }
            let row = eng.prefill_resume(a.slot, &a.prompt, a.reuse, a.snap, a.owner || eng.per_lane_ctx())?.context("node prefill without logits")?;
            let (t, _) = eng.dec.sample_row_lane(&eng.model, row, a.lane)?;
            let spec = LaneSpec { id: next_id, sp: a.sp, seed: a.seed, smp_round: 0 };
            next_id += 1;
            act.push(NLane { slot: a.slot, pos: a.prompt.len(), t, owner: a.owner, last: vec![t], spec });
        }
        let (drop, kind, check) = match crate::cluster::recv_json::<Ctl>(stream)? {
            Ctl::LStep { drop, kind, check } => (drop, kind, check),
            Ctl::Shutdown => return Ok(()),
            other => bail!("lanes: unexpected {other:?} before a step"),
        };
        for &j in &drop { let last = act.len() - 1; if j != last { eng.copy_lane(last, j)?; } act.swap_remove(j); }
        if kind == 255 { continue; }
        ensure!(check.len() == act.len(), "NAIVE TP LOCKSTEP BROKEN (lanes): head {} lanes, node {}", check.len(), act.len());
        for (j, c) in check.iter().enumerate() {
            ensure!(*c == act[j].last, "NAIVE TP LOCKSTEP BROKEN (lane {j}): node {:?}, head {c:?}", act[j].last);
        }
        let outs: Vec<(Vec<u32>, usize)> = match kind {
            1 => {
                let (t, pos, slot) = (act[0].t, act[0].pos, act[0].slot);
                eng.set_slot_all(slot)?;
                lane_spec_load(eng, &act[0].spec)?;
                if eng.sharded { let (d, cf) = eng.draft(t, pos)?; let k = eng.pick_width(&cf); eng.node_drafts = Some((d, k)); }
                let (committed, a, _) = eng.spec_round(None, t, pos)?;
                act[0].spec.smp_round = eng.smp_round;
                if !eng.last_round_plain() { eng.dec.note_commit_lane(&eng.model, &committed, 0)?; }
                vec![(committed, a)]
            }
            2 => {
                let (t, pos, slot) = (act[0].t, act[0].pos, act[0].slot);
                eng.set_slot_all(slot)?;
                lane_spec_load(eng, &act[0].spec)?;
                let (nt, _) = eng.plain_step_spec(t, pos)?;
                vec![(vec![nt], 0)]
            }
            _ => {
                let ids: Vec<u32> = act.iter().map(|l| l.t).collect();
                let pos: Vec<usize> = act.iter().map(|l| l.pos).collect();
                let slots: Vec<usize> = act.iter().map(|l| l.slot).collect();
                let owner = act.iter().position(|l| l.owner);
                eng.step_lanes(&ids, &pos, &slots, owner)?.into_iter().map(|t| (vec![t], 0)).collect()
            }
        };
        for (j, (toks, a)) in outs.into_iter().enumerate() {
            act[j].pos += a + 1;
            act[j].t = *toks.last().unwrap();
            act[j].last = toks;
        }
    }
}

/// The head's serve: attach TP, load the shard, wait for the node, run HTTP + the one-at-a-time loop.
pub fn run_head(args: &[String], model_dir: &str, ctx: crate::tp::TpContext, mut streams: Vec<std::net::TcpStream>,
                watch: crate::cluster::NodeWatch) -> Result<()> {
    let world = ctx.world as usize;
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let c = &pack.cfg;
    let port: u16 = crate::arg_value(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(9000);
    let max_pos: usize = crate::arg_value(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(32768);
    ensure!(max_pos <= c.max_position_embeddings, "--max-seq-len {max_pos} exceeds the model's {}", c.max_position_embeddings);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    super::gpu::set_pdl(pdl_flag(args));
    super::gpu::set_fuse(fuse_flag(args));
    let n_layers: usize = crate::arg_value(args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(c.num_layers);
    let chunk: usize = crate::arg_value(args, "--prefill-chunk").and_then(|s| s.parse().ok()).unwrap_or(PREFILL_CHUNK);
    let spec = spec_cfg(args, model_dir);
    let tp = NaiveTp::attach(&dev, ctx, c.hidden_size, chunk.max(MAX_ROWS))?;
    let (lanes, asked) = lanes_asked(args);
    if asked > lanes {
        println!("[naive-serve] asked for {asked} requests at once (--max-batch {asked}): the Naive server shares a step among at most \
                  {MAX_LANES}, so it serves {lanes} at a time and further requests wait in the queue (none is rejected)");
    }
    let mut eng = Engine::load_spec_lanes(dev, &pack, Shard { rank: 0, world }, max_pos, n_layers, true, chunk, spec.as_ref(), lanes)?;
    crate::metrics::set_max_batch_requested(asked);
    if lanes > 1 {
        println!("[naive-serve] G1: up to {lanes} requests at once (--max-batch): one row per request in a shared step, {lanes} KV slots; {}",
                 if eng.per_lane_ctx() { "every lane keeps its own drafter context, so any request left alone speculates (--naive-lane-ctx on)" }
                 else { "one drafter context: only a request admitted into an idle engine speculates when alone (--naive-lane-ctx off, or no drafter)" });
    }
    eng.model.tp = Some(tp);
    let prefix = prefix_flag(args);
    if prefix {
        eng.enable_prefix()?;
        println!("[naive-serve] prefix reuse ON (--prefix-cache on): a prompt that repeats an earlier request's prompt up to a \
                  prefill chunk boundary resumes its prefill there (one conversation per KV slot: {lanes}; same chunks, same state)");
    }
    crate::metrics::NAIVE_PREFIX_ON.store(if eng.pfx_on() { 1 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
    let pfx_on = eng.pfx_on();
    watch.disarm(&streams);
    for (i, s) in streams.iter_mut().enumerate() {
        match crate::tp_serve::recv_serving(s).context("node Ready")? {
            crate::tp_serve::ServingMsg::Ready => crate::dprintln!("[naive-serve] node rank {} READY", i + 1),
            other => bail!("expected Ready from node rank {}, got {other:?}", i + 1),
        }
    }
    let tok = Arc::new(QwenTokenizer::from_file(&format!("{}/tokenizer.json", pack.dir)).context("tokenizer")?);
    let stop_ids = tok.stop_token_ids(c.eos_token_id);
    let model_name = crate::server::resolve_model_name(crate::arg_value(args, "--model-name"), crate::arg_value(args, "--served-model-name"), model_dir);
    let (stx, mut srx) = tokio::sync::mpsc::unbounded_channel::<BatchRequest>();
    let state = crate::server::AppState {
        // the pack's generation_config: temperature 1.0, top_p 0.95, no top-k
        sampling_defaults: crate::server::SamplingDefaults { think: (1.0, 0.95, 0), no_think: (1.0, 0.95, 0) },
        scheduler: stx,
        tokenizer: tok.clone(),
        model_name: model_name.clone(),
        default_max_tokens: crate::arg_value(args, "--max-tokens").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
        default_rep_penalty: 1.0,
        default_presence_penalty: 0.0,
        default_frequency_penalty: 0.0,
        reasoning_effort: None,
        thinking: crate::tokenizer::ThinkingMode::Auto,
        output_prompts: 0,
        max_seq_len: max_pos,
        decode_headroom: 0,
        // the message boundary (`ckpt_at`) places the Naive prefix snapshot
        prefix_cache: pfx_on,
        // no deferred schema start on this path
        schema_defer: false,
        vision_tower: None,
        vision_gpu: None,
        vision_cpu: false,
        stop_ids: stop_ids.clone(),
        otel: None,
    };
    println!("[naive-serve] TP={world} serving: single stream, head-authoritative tokens over the control stream, \
              fp32 doorbell all-reduces (2 per layer), lm_head replicated (head only), graphed decode step with GPU sampling; stop_ids {stop_ids:?}");
    std::thread::spawn(move || {
        if let Err(e) = eng.model.dev.bind_to_thread() { eprintln!("\n*** FATAL: CUDA context bind: {e:?} ***\n"); std::process::exit(70); }
        // the node captures its graph after Ready; the head captures here, before the first request (its
        // graph holds the reduces, which are barriers: both ranks must capture at the same program point)
        if let Err(e) = eng.arm() { eprintln!("\n*** FATAL: decode graph capture failed: {e:#} ***\n"); std::process::exit(70); }
        if eng.lanes > 1 {
            if let Err(e) = serve_lanes(&mut eng, &mut srx, &stop_ids, &mut streams) {
                eprintln!("\n*** FATAL: the Naive TP head failed in the lanes loop: {e:#}. Exiting (70). ***\n");
                std::thread::sleep(std::time::Duration::from_millis(300));
                std::process::exit(70);
            }
        }
        while let Some(req) = srx.blocking_recv() {
            if let Err(e) = serve_one(&mut eng, req, &stop_ids, &mut streams) {
                eprintln!("\n*** FATAL: the Naive TP head failed a request: {e:#}. Exiting (70). ***\n");
                std::thread::sleep(std::time::Duration::from_millis(300));
                std::process::exit(70);
            }
        }
        for n in streams.iter_mut() { let _ = crate::cluster::send_json(n, &Ctl::Shutdown); }
        eprintln!("[naive-serve] request channel closed — session over");
        std::process::exit(0);
    });
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().context("tokio runtime")?;
    rt.block_on(async move {
        let app = crate::server::create_router(state);
        let listener = crate::server::http_listen(port).await;
        let url_l = listener.local_addr().map_or_else(|_| format!("port {port}"), |a| a.to_string());
        // report the RESOLVED spec config, not a literal. This line used to hardcode `plain`, so it
        // read `plain` even when `--spec-source dspark` had resolved a live drafter — a status
        // field that cannot be wrong about engagement is exactly the untrustworthy-receipt class
        // (an engaged-but-unlabelled path reads as dead, and a dead one reads as engaged).
        let spec_label: String = match &spec {
            None => "off".to_string(),
            Some(sc) => format!(
                "{}:block{}",
                crate::arg_value(args, "--spec-source").unwrap_or("dspark"),
                sc.block
            ),
        };
        crate::boot_report::serving(&crate::boot_report::ServingLine {
            family: "naive", model: &model_name, tp: world as u32, ctx: max_pos, batch: lanes, kv: "f16", prefix: if pfx_on { "on" } else { "off" }, mtp: "none",
            spec: &spec_label, port, url: if url_l.starts_with("port ") { format!("http://0.0.0.0:{port}") } else { format!("http://{url_l}") },
        });
        axum::serve(listener, app).tcp_nodelay(true).await.unwrap();
    });
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The node
// ---------------------------------------------------------------------------------------------

/// The node's serve: attach TP, load the shard, Ready, then mirror Admit / Tok forever.
pub fn run_node(model_dir: &str, max_pos: usize, n_layers: Option<usize>, chunk: usize, head_argv: &[String], ctx: crate::tp::TpContext, mut stream: std::net::TcpStream) -> Result<()> {
    let (rank, world) = (ctx.rank as usize, ctx.world as usize);
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    super::gpu::set_pdl(pdl_flag(head_argv));
    super::gpu::set_fuse(fuse_flag(head_argv));
    let spec = spec_cfg(head_argv, model_dir); // the node has no drafter: only the block size matters here
    let tp = NaiveTp::attach(&dev, ctx, pack.cfg.hidden_size, chunk.max(MAX_ROWS))?;
    let mut eng = Engine::load_spec_lanes(dev, &pack, Shard { rank, world }, max_pos, n_layers.unwrap_or(pack.cfg.num_layers), false, chunk, spec.as_ref(), lanes_flag(head_argv))?;
    eng.model.tp = Some(tp);
    if prefix_flag(head_argv) { eng.enable_prefix()?; } // the head's decision rides every Admit; this rank keeps its own shard's snapshot
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", pack.dir)).context("tokenizer")?;
    let stop_ids = tok.stop_token_ids(pack.cfg.eos_token_id);
    crate::tp_serve::send_serving(&mut stream, &crate::tp_serve::ServingMsg::Ready)?;
    eng.arm()?;
    crate::dprintln!("[naive-serve] node rank {rank}: Ready sent, graph armed — mirroring the head");
    loop {
        match crate::cluster::recv_json::<Ctl>(&mut stream).context("control stream (head gone?)")? {
            Ctl::Admit { prompt, max_new, seed, sp, min_new, ignore_eos, reuse, snap } => {
                // the mirror: the same prefill, the same sampler (every rank derives the same token and the
                // same stop decision from the exchanged values); the head's Tok messages are read one step
                // LATE as a lockstep check, so the TCP hop is never on this rank's critical path
                eng.begin_request(&sp, seed, &prompt)?;
                let row = eng.prefill_resume(0, &prompt, reuse, snap, true)?.context("node prefill without logits")?;
                let (mut t, mut cancel) = eng.dec.sample_row(&eng.model, row)?;
                let mut pos = prompt.len();
                let mut gen = 0usize;
                if eng.ver.is_some() {
                    // S8 mirror: the same rounds (drafts arrive in the exchange), the same stop rule; the head's
                    // Round message is checked one round late
                    let mut pending: Option<(Vec<u32>, bool)> = None;
                    let mut stop0 = false;
                    {
                        let is_stop = stop_ids.contains(&t) && !ignore_eos && gen + 1 > min_new;
                        if is_stop { stop0 = true; } else { gen += 1; if gen >= max_new { stop0 = true; } }
                        pending = Some((vec![t], stop0));
                        if !stop0 { eng.note_commit_all(&[t], false)?; }
                    }
                    let mut guard = SpecGuard::with_every(eng.guard.0, eng.guard.1, eng.guard.2);
                    while !stop0 {
                        let mut from_plain = false;
                        let (committed, a, c) = if guard.speculate() {
                            if eng.sharded {
                                // work item B: this rank runs its shard of the drafter in lockstep with the head (the
                                // exchanges inside); its drafts are checked against the head's in the slot exchange
                                let (d, cf) = eng.draft(t, pos)?;
                                let k = eng.pick_width(&cf);
                                eng.node_drafts = Some((d, k));
                            }
                            let r = eng.spec_round(None, t, pos)?;
                            guard.observe(r.1);
                            r
                        } else {
                            let (nt, c) = eng.plain_step_spec(t, pos)?;
                            from_plain = true;
                            (vec![nt], 0, c)
                        };
                        let toks_out: Vec<u32> = committed.clone();
                        let mut stop = false; let mut emitted = Vec::new();
                        for &tok in &toks_out {
                            let is_stop = stop_ids.contains(&tok) && !ignore_eos && gen + 1 > min_new;
                            if is_stop { stop = true; break; }
                            gen += 1; emitted.push(tok);
                            if gen >= max_new { stop = true; break; }
                        }
                        if c || cancel { stop = true; }
                        if let Some((pt, ps)) = pending.take() {
                            match crate::cluster::recv_json::<Ctl>(&mut stream)? {
                                Ctl::Round { toks: ht, stop: hs } => ensure!(ht == pt && hs == ps, "NAIVE TP LOCKSTEP BROKEN (round): node {pt:?}/{ps}, head {ht:?}/{hs}"),
                                Ctl::Shutdown => return Ok(()),
                                other => bail!("unexpected {other:?} inside a request"),
                            }
                        }
                        pending = Some((toks_out, stop));
                        eng.note_commit_all(&emitted, from_plain || eng.last_round_plain())?;
                        if stop { break; }
                        pos += a + 1;
                        t = *committed.last().unwrap();
                        cancel = c;
                        if pos + eng.block > eng.max_pos { break; }
                    }
                    if let Some((pt, ps)) = pending.take() {
                        match crate::cluster::recv_json::<Ctl>(&mut stream)? {
                            Ctl::Round { toks: ht, stop: hs } => ensure!(ht == pt && hs == ps, "NAIVE TP LOCKSTEP BROKEN at the end (round): node {pt:?}/{ps}, head {ht:?}/{hs}"),
                            Ctl::Shutdown => return Ok(()),
                            other => bail!("unexpected {other:?} at the end of a request"),
                        }
                    }
                    continue;
                }
                let mut pending: Option<(u32, bool)> = None; // the previous step's (token, stop) awaiting the head's word
                for i in 0..max_new {
                    let is_stop = stop_ids.contains(&t) && !ignore_eos && gen + 1 > min_new;
                    let last = is_stop || i + 1 == max_new || pos + 1 >= eng.max_pos || cancel;
                    if let Some((pt, ps)) = pending.take() {
                        match crate::cluster::recv_json::<Ctl>(&mut stream)? {
                            Ctl::Tok { t: ht, stop: hs } => ensure!(ht == pt && hs == ps, "NAIVE TP LOCKSTEP BROKEN: node token {pt} stop {ps}, head token {ht} stop {hs}"),
                            Ctl::Shutdown => return Ok(()),
                            other => bail!("unexpected {other:?} inside a request"),
                        }
                    }
                    pending = Some((t, last));
                    if !is_stop { gen += 1; }
                    if last { break; }
                    (t, cancel) = eng.decode_step(t, pos)?;
                    pos += 1;
                }
                if let Some((pt, ps)) = pending.take() {
                    match crate::cluster::recv_json::<Ctl>(&mut stream)? {
                        Ctl::Tok { t: ht, stop: hs } => ensure!(ht == pt && hs == ps, "NAIVE TP LOCKSTEP BROKEN at the end: node {pt}/{ps}, head {ht}/{hs}"),
                        Ctl::Shutdown => return Ok(()),
                        other => bail!("unexpected {other:?} at the end of a request"),
                    }
                }
            }
            m @ Ctl::LIter { .. } => return node_lanes(&mut eng, &mut stream, m),
            Ctl::Shutdown => return Ok(()),
            other => bail!("unexpected {other:?} between requests"),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The TP identity probe: `--probe-naive-tp --model-dir <pack> --tp 2 --layers N --tokens T`
// ---------------------------------------------------------------------------------------------

/// Both ranks: N sharded layers + the transport, forward T fixed tokens; the head compares its
/// logits and residual taps with an UNSHARDED N-layer slice it loads alongside (small N).
pub fn run_probe(model_dir: &str, ctx: crate::tp::TpContext, n_layers: usize, tokens: usize, stream: Option<std::net::TcpStream>,
                 engine: bool, chunk: usize, argv: &[String], peers: Vec<std::net::TcpStream>) -> Result<()> {
    // FULL-DEPTH mode (n_layers == 0): the real prompt "The capital of France is" through every layer of the
    // sharded model (the serve configuration), rank 0 reporting the argmax of its vocabulary half —
    // " Paris" (12095) lives in rank 0's half at every world. No whole-model reference (it does not fit).
    if n_layers == 0 { return run_probe_full(model_dir, ctx, stream, engine, chunk, argv, peers); }
    let (rank, world) = (ctx.rank as usize, ctx.world as usize);
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    let tp = NaiveTp::attach(&dev, ctx, pack.cfg.hidden_size, MAX_ROWS)?;
    let test_stream = dev.fork_default_stream()?;
    for n in [4096usize, 16 * 4096, 2048 * 4096] {
        let bad = tp.self_test(&dev, &test_stream, n)?;
        println!("[naive-tp] rank {rank}: transport self-test {n} floats: {bad} wrong");
        ensure!(bad == 0, "NAIVE_TP_FAIL: transport self-test");
    }
    // the reduce latency: 500 back-to-back 16 KB all-reduces (the decode shape), stream time per reduce
    {
        let buf = dev.alloc_zeros::<f32>(4096)?;
        dev.synchronize()?;
        for _ in 0..20 { tp.all_reduce(&test_stream, &buf, 4096)?; }
        unsafe { cudarc::driver::result::stream::synchronize(test_stream.stream) }?;
        let t0 = std::time::Instant::now();
        let n = 500;
        for _ in 0..n { tp.all_reduce(&test_stream, &buf, 4096)?; }
        unsafe { cudarc::driver::result::stream::synchronize(test_stream.stream) }?;
        println!("[naive-tp] rank {rank}: 16 KB all-reduce: {:.1} us each ({n} back to back)", t0.elapsed().as_secs_f64() / n as f64 * 1e6);
    }
    let t0 = std::time::Instant::now();
    let mut sh = GpuModel::load_slice_on(dev.clone(), &pack, n_layers, 4096, SWA_RING, 1, Shard { rank, world })?;
    sh.tp = Some(tp);
    println!("[naive-tp] rank {rank}/{world}: {n_layers} sharded layers loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let pos: Vec<usize> = (0..tokens).collect();
    let _ctl = stream; // the node holds its control stream to the end (the head's watch reads a close as a failure)
    // round 1: the sharded forward on both ranks (the reduces are the barriers)
    let t1 = std::time::Instant::now();
    let (lg_sh, taps_sh) = sh.forward_rows(&ids, &pos)?;
    let fwd = t1.elapsed().as_secs_f64();
    println!("[naive-tp] rank {rank}: sharded forward {fwd:.3} s ({} reduces)", 2 * n_layers);
    // round 2: again — the second pass must be bit-identical to the first (transport determinism)
    let (lg_sh2, _) = sh.forward_rows(&ids, &pos)?;
    let same = lg_sh == lg_sh2;
    println!("[naive-tp] rank {rank}: second sharded pass bitwise identical to the first: {same}");
    if rank != 0 { ensure!(same, "NAIVE_TP_FAIL: rank {rank} not deterministic"); println!("NAIVE_TP_NODE_DONE"); return Ok(()); }
    // the unsharded reference on the head's GPU
    let r = GpuModel::load_slice_on(dev, &pack, n_layers, 4096, SWA_RING, 1, Shard::single())?;
    let (lg_r, taps_r) = r.forward_rows(&ids, &pos)?;
    let mut worst = 0f32;
    for l in 0..n_layers {
        let rel = super::cpu::rel_l2(&taps_sh[l], &taps_r[l]);
        worst = worst.max(rel);
        println!("  layer {l:2}: residual rel-L2 sharded vs whole {rel:.3e}");
    }
    // the sharded head holds rank 0's vocabulary slice: compare it with the same columns of the whole
    let (v, vl) = (pack.cfg.vocab_size, sh.vl);
    let mut lg_rs = Vec::with_capacity(tokens * vl);
    for t in 0..tokens { lg_rs.extend_from_slice(&lg_r[t * v..t * v + vl]); }
    let rel = super::cpu::rel_l2(&lg_sh, &lg_rs);
    let agree = (0..tokens).filter(|t| argmax(&lg_sh[t * vl..(t + 1) * vl]) == argmax(&lg_rs[t * vl..(t + 1) * vl])).count();
    println!("  logits (rank-0 vocabulary slice, {vl} of {v}) rel-L2 {rel:.3e}; argmax agreement {agree}/{tokens}; determinism {same}");
    // the sharded path differs from the whole-layer path only by fp32 summation order of the two
    // partials (and the f16 rounding of the halves' Hadamard outputs): 1e-3-class, not 1e-2
    if worst > 5e-3 || rel > 5e-3 || agree * 10 < tokens * 9 || !same { bail!("NAIVE_TP_FAIL"); }
    println!("NAIVE_TP_OK");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The loopback TP gate: `--probe-naive-loop --model-dir <pack> [--layers N] [--tokens T]`
// ---------------------------------------------------------------------------------------------

/// Two shards on ONE GPU, two threads, the partials crossing a host channel: the sharded math
/// without a link. The head thread compares against the whole-layer slice.
pub fn run_loop_probe(model_dir: &str, n_layers: usize, tokens: usize, world: usize) -> Result<()> {
    ensure!(world == 2 || world == 4, "--tp 2 or 4");
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    let mut tps = NaiveTp::loopback_group(&dev, pack.cfg.hidden_size, tokens.max(MAX_ROWS), world)?;
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let pos: Vec<usize> = (0..tokens).collect();
    fn run(dev: Arc<CudaDevice>, pack: Arc<NaivePack>, ids: Vec<u32>, pos: Vec<usize>, n_layers: usize, rank: usize, world: usize, tp: NaiveTp)
           -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        dev.bind_to_thread()?;
        let mut m = GpuModel::load_slice_on(dev, &pack, n_layers, 4096, SWA_RING, 1, Shard { rank, world })?;
        m.tp = Some(tp);
        m.forward_rows(&ids, &pos)
    }
    let tp0 = tps.remove(0);
    let mut handles = Vec::new();
    for (i, tp) in tps.into_iter().enumerate() {
        let (d, p, iv, q) = (dev.clone(), pack.clone(), ids.clone(), pos.clone());
        handles.push(std::thread::spawn(move || run(d, p, iv, q, n_layers, i + 1, world, tp)));
    }
    let (lg0, taps0) = run(dev.clone(), pack.clone(), ids.clone(), pos.clone(), n_layers, 0, world, tp0)?;
    let mut all_same = true;
    for (i, h) in handles.into_iter().enumerate() {
        let (_lg, taps) = h.join().map_err(|_| anyhow::anyhow!("rank {} thread panicked", i + 1))??;
        all_same &= taps.last() == taps0.last();
    }
    println!("[naive-loop] world {world}: every rank's final residual bitwise identical to rank 0's: {all_same}");
    let r = GpuModel::load_slice_on(dev.clone(), &pack, n_layers, 4096, SWA_RING, 1, Shard::single())?;
    let (lg_r, taps_r) = r.forward_rows(&ids, &pos)?;
    let mut worst = 0f32;
    for l in 0..n_layers {
        let rel = super::cpu::rel_l2(&taps0[l], &taps_r[l]);
        worst = worst.max(rel);
        println!("  layer {l:2}: residual rel-L2 sharded(loopback) vs whole {rel:.3e}");
    }
    let (v, vl) = (pack.cfg.vocab_size, lg0.len() / tokens);
    let mut lg_rs = Vec::with_capacity(tokens * vl);
    for t in 0..tokens { lg_rs.extend_from_slice(&lg_r[t * v..t * v + vl]); }
    let rel = super::cpu::rel_l2(&lg0, &lg_rs);
    let agree = (0..tokens).filter(|t| argmax(&lg0[t * vl..(t + 1) * vl]) == argmax(&lg_rs[t * vl..(t + 1) * vl])).count();
    println!("  logits (rank-0 vocabulary slice, {vl} of {v}) rel-L2 {rel:.3e}; argmax agreement {agree}/{tokens}");
    if worst > 5e-3 || rel > 5e-3 || agree * 10 < tokens * 9 { bail!("NAIVE_LOOP_FAIL"); }
    println!("NAIVE_LOOP_OK");
    Ok(())
}

/// `--probe-naive-prefill --model-dir <pack> [--layers N] [--tokens T]`: the wide prefill path (one chunk of T
/// rows) against the 16-row chunked path on the same tokens: the residual after every layer and the last
/// row's logits must be BITWISE identical (every kernel in the wide path is row-invariant by construction).
pub fn run_prefill_probe(model_dir: &str, n_layers: usize, tokens: usize) -> Result<()> {
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    let model = GpuModel::load_slice_on(dev.clone(), &pack, n_layers, 8192, SWA_RING, 1, Shard::single())?;
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let (h, vl) = (model.cfg.hidden_size, model.vl);
    // wide: one chunk
    let mut ws = Ws::new(&dev, ws_bytes(&model.cfg, tokens, 8192, n_layers))?;
    let ids_d = dev.htod_sync_copy(&ids.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
    let pos_d = dev.htod_sync_copy(&(0..tokens as i32).collect::<Vec<i32>>())?;
    let mut taps_w = Vec::new();
    let t0 = std::time::Instant::now();
    let lg_w = model.step_kernels(&mut ws, dptr(&ids_d), dptr(&pos_d), tokens, true, Some(&mut taps_w))?.unwrap();
    model.sync()?;
    let dt_w = t0.elapsed().as_secs_f64();
    let taps_w: Vec<Vec<f32>> = taps_w.iter().map(|t| super::model::dtoh::<f32>(*t, tokens * h)).collect::<Result<_>>()?;
    let lg_w: Vec<u16> = super::model::dtoh::<u16>(lg_w, vl)?;
    // chunked: 16 rows at a time (the caches are rewritten position by position identically)
    let mut taps_c: Vec<Vec<f32>> = vec![vec![0f32; tokens * h]; n_layers];
    let mut lg_c = Vec::new();
    let t1 = std::time::Instant::now();
    let mut i = 0;
    while i < tokens {
        let m = (tokens - i).min(MAX_ROWS);
        let mut ws2 = Ws::new(&dev, ws_bytes(&model.cfg, m, 8192, n_layers))?;
        let id2 = dev.htod_sync_copy(&ids[i..i + m].iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        let po2 = dev.htod_sync_copy(&(i as i32..(i + m) as i32).collect::<Vec<i32>>())?;
        let mut tp2 = Vec::new();
        let lg = model.step_kernels(&mut ws2, dptr(&id2), dptr(&po2), m, i + m == tokens, Some(&mut tp2))?;
        model.sync()?;
        for (l, t) in tp2.iter().enumerate() {
            let v = super::model::dtoh::<f32>(*t, m * h)?;
            taps_c[l][i * h..(i + m) * h].copy_from_slice(&v);
        }
        if let Some(p) = lg { lg_c = super::model::dtoh::<u16>(p + ((m - 1) * vl * 2) as u64, vl)?; }
        i += m;
    }
    let dt_c = t1.elapsed().as_secs_f64();
    // the decode path sums split-K fp32 partials, the wide kernel accumulates K in one pass: the two are
    // not bitwise (2e-4-class rounding differences, the f16 noise floor); the gate is the tolerance + argmax
    let mut ok = true;
    // the f16 activation drift grows ~1.5x per layer against any differently-rounded path (the slice gates
    // show the same trend); the bound follows it
    for l in 0..n_layers {
        let rel = super::cpu::rel_l2(&taps_w[l], &taps_c[l]);
        let bound = 1e-3 * 1.5f32.powi(l as i32 + 1);
        println!("  layer {l:2}: residual wide vs chunked-16: rel-L2 {rel:.2e} (bound {bound:.1e})");
        if rel > bound { ok = false; }
    }
    let f = |v: &[u16]| v.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect::<Vec<f32>>();
    let (lw, lc) = (f(&lg_w), f(&lg_c));
    let rel = super::cpu::rel_l2(&lw, &lc);
    let same_argmax = argmax(&lw) == argmax(&lc);
    println!("  last-row logits: rel-L2 {rel:.2e}, argmax equal {same_argmax}");
    println!("  wide {tokens} rows: {:.1} ms ({:.0} tok/s, {n_layers} layers); chunked-16: {:.1} ms ({:.0} tok/s)", dt_w * 1e3, tokens as f64 / dt_w, dt_c * 1e3, tokens as f64 / dt_c);
    if rel > 1e-2 || !same_argmax { ok = false; }
    if !ok { bail!("NAIVE_PREFILL_FAIL"); }
    println!("NAIVE_PREFILL_OK");
    Ok(())
}

/// `--probe-naive-shard-dir --model-dir <full pack> --shard-dir <shipped shard dir> --tp W --layers N`: the shipped
/// shard must load into the SAME rank model as the full pack sliced at load: rank-local partial forward (no
/// transport) on both, residual taps and local logits BITWISE identical.
pub fn run_shard_dir_probe(full_dir: &str, shard_dir: &str, n_layers: usize, tokens: usize) -> Result<()> {
    let full = Arc::new(NaivePack::open(full_dir)?);
    let shard_pack = Arc::new(NaivePack::open(shard_dir)?);
    let (world, rank) = shard_pack.shard.context("the shard dir has no gb10_shard.json")?;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let pos: Vec<usize> = (0..tokens).collect();
    let sh = Shard { rank, world };
    let a = GpuModel::load_slice_on(dev.clone(), &full, n_layers, 4096, SWA_RING, 1, sh)?;
    let (lg_a, taps_a) = a.forward_rows(&ids, &pos)?;
    drop(a);
    let b = GpuModel::load_slice_on(dev.clone(), &shard_pack, n_layers, 4096, SWA_RING, 1, sh)?;
    let (lg_b, taps_b) = b.forward_rows(&ids, &pos)?;
    let mut ok = true;
    for l in 0..n_layers {
        let d = taps_a[l].iter().zip(&taps_b[l]).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        println!("  layer {l:2}: residual (rank {rank} partial path) full-pack vs shipped shard: {d} of {} values differ", taps_a[l].len());
        if d > 0 { ok = false; }
    }
    let d = lg_a.iter().zip(&lg_b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    println!("  local logits: {d} of {} differ", lg_a.len());
    if d > 0 { ok = false; }
    if !ok { bail!("NAIVE_SHARD_DIR_FAIL"); }
    println!("NAIVE_SHARD_DIR_OK");
    Ok(())
}

/// The probe's load barrier over the control streams: every node says "loaded", the head waits for all of them, then
/// says "go". The ranks load at very different speeds (the head slices the whole pack, a node reads its shipped shard),
/// and the first reduce of a rank that is >10 s ahead trips the transport watchdog, which aborts the link: every later
/// reduce then returns a partial sum without an error (2026-10-21: the NLL probe's 12.7 nats/token).
fn probe_barrier(stream: &mut Option<std::net::TcpStream>, peers: &mut [std::net::TcpStream]) -> Result<()> {
    match stream.as_mut() {
        Some(s) => { crate::cluster::send_json(s, &1u8)?; let _: u8 = crate::cluster::recv_json(s).context("probe barrier: head's go")?; }
        None => {
            for p in peers.iter_mut() { let _: u8 = crate::cluster::recv_json(p).context("probe barrier: a node's loaded")?; }
            for p in peers.iter_mut() { crate::cluster::send_json(p, &2u8)?; }
        }
    }
    Ok(())
}

fn run_probe_full(model_dir: &str, ctx: crate::tp::TpContext, mut stream: Option<std::net::TcpStream>, engine: bool, chunk: usize, argv: &[String],
                  mut peers: Vec<std::net::TcpStream>) -> Result<()> {
    let (rank, world) = (ctx.rank as usize, ctx.world as usize);
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    // ENGINE mode (--probe-naive-tp-engine): the serve's own path — Engine prefill (chunked workspace) + the
    // graphed decode step with the GPU sampler — greedy tokens printed on every rank
    if engine {
        // the head's argv decides speculation (`--spec-source`, `--spec-block`, `--draft-dir` on the head)
        let spec = spec_cfg(argv, model_dir).filter(|_| rank == 0 || true);
        let tp = NaiveTp::attach(&dev, ctx, pack.cfg.hidden_size, chunk.max(MAX_ROWS))?;
        let mut eng = Engine::load_spec(dev.clone(), &pack, Shard { rank, world }, 32768, pack.cfg.num_layers, rank == 0, chunk, spec.as_ref())?;
        eng.model.tp = Some(tp);
        eng.arm()?;
        let _ctl = stream;
        let ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        eng.begin_request(&SampleParams::GREEDY, 1, &ids)?;
        let row = eng.prefill(&ids, 0, true)?.unwrap();
        let (mut t, _) = eng.dec.sample_row(&eng.model, row)?;
        let mut out = vec![t];
        let mut pos = ids.len();
        if eng.ver.is_some() {
            // speculative rounds (the head drafts; the node reads the exchange): greedy must match the plain path
            for r in 0..4 {
                let drafts: Option<(Vec<u32>, usize)> = if rank == 0 { let (d, c) = eng.draft(t, pos)?; let k = eng.pick_width(&c); Some((d, k)) } else { None };
                let (committed, a, _) = eng.spec_round(drafts.as_ref().map(|(d, k)| (d.as_slice(), *k)), t, pos).with_context(|| format!("spec round {r}"))?;
                println!("[naive-tp] rank {rank}: round {r}: drafts {:?} accepted {a} committed {:?}", drafts, committed);
                out.extend_from_slice(&committed); pos += a + 1; t = *committed.last().unwrap();
            }
        } else {
            for _ in 0..5 { let (nt, _) = eng.decode_step(t, pos)?; out.push(nt); t = nt; pos += 1; }
        }
        println!("[naive-tp] rank {rank}: ENGINE greedy tokens {out:?} (want [12095, 382, 59604, 374, ...] = ' Paris.\n\nParis is')");
        if rank == 0 { ensure!(out[0] == 12095, "NAIVE_TP_FAIL: engine path"); println!("NAIVE_TP_FULL_OK"); } else { println!("NAIVE_TP_NODE_DONE"); }
        return Ok(());
    }
    // Teacher-forced NLL on the SHARDED model (2026-10-20, the expert-TP quality gate): `--nll-file F --nll-out P`
    // (F and P paths on every rank). It runs the SERVED engine's prefill in MAX_ROWS-row chunks (a chunk that narrow
    // returns every row's logits): each rank writes, per row i, its vocabulary slice's (max, sum exp(l - max)) and the
    // target logit of token i+1 when the target is in its slice; the per-row NLL is combined offline (logsumexp over the
    // ranks' slices minus the target logit). (The first version forwarded all rows through `GpuModel::forward_rows`,
    // a probe-only path: under TP at 2,047 rows it gave ~12.7 nats per token on both expert deals, i.e. garbage.)
    if let (Some(f), Some(out)) = (crate::arg_value(argv, "--nll-file"), crate::arg_value(argv, "--nll-out")) {
        let text = std::fs::read_to_string(f).with_context(|| format!("--nll-file {f}"))?;
        let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", pack.dir)).context("tokenizer")?;
        let mut ids = tok.encode(&text, false)?;
        let n_max: usize = crate::arg_value(argv, "--nll-tokens").and_then(|s| s.parse().ok()).unwrap_or(1024);
        ids.truncate(n_max.min(4000));
        ensure!(ids.len() >= 2, "--nll-file: {} tokens", ids.len());
        // `--nll-rows R` (1..=16, default 1): rows per prefill call (1 = every row is the LAST row of its call, the
        // layout the serve reads; wider calls read a multi-row logits block the serve never reads row by row)
        let nll_rows: usize = crate::arg_value(argv, "--nll-rows").and_then(|s| s.parse().ok()).unwrap_or(1usize).clamp(1, MAX_ROWS);
        let tp = NaiveTp::attach(&dev, ctx, pack.cfg.hidden_size, MAX_ROWS)?;
        let mut eng = Engine::load(dev.clone(), &pack, Shard { rank, world }, 4096, pack.cfg.num_layers, rank == 0, MAX_ROWS)?;
        eng.model.tp = Some(tp);
        probe_barrier(&mut stream, &mut peers)?;
        let _ctl = stream;
        let (vl, v0) = (eng.model.vl, eng.model.v0);
        let mut lines = String::new();
        let mut i0 = 0usize;
        while i0 < ids.len() - 1 {
            let m = (ids.len() - 1 - i0).min(nll_rows);
            let last = eng.prefill_feed(&ids[i0..i0 + m], i0, true, false)?.context("NLL chunk without logits")?;
            eng.model.sync()?;
            let base = last - ((m - 1) * vl * 2) as u64;
            let lg: Vec<f32> = super::model::dtoh::<u16>(base, m * vl)?.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
            for r in 0..m {
                let i = i0 + r;
                let row = &lg[r * vl..(r + 1) * vl];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let se: f64 = row.iter().map(|x| ((*x as f64) - mx).exp()).sum();
                let t = ids[i + 1] as usize;
                let tl = if t >= v0 && t < v0 + vl { format!("{:.9}", row[t - v0]) } else { "null".into() };
                lines += &format!("{{\"i\":{i},\"m\":{mx:.9},\"s\":{se:.9},\"t\":{tl}}}\n");
            }
            i0 += m;
        }
        if let Some(tp) = &eng.model.tp { tp.check_link("at the end of the NLL pass")?; }
        let path = format!("{out}.rank{rank}.jsonl");
        std::fs::write(&path, lines)?;
        println!("[naive-tp] rank {rank}: NLL partials of {} rows (engine prefill) written to {path} ({nll_rows} row(s) per call)", ids.len() - 1);
        println!("{}", if rank == 0 { "NAIVE_TP_NLL_OK" } else { "NAIVE_TP_NODE_DONE" });
        return Ok(());
    }
    let tp = NaiveTp::attach(&dev, ctx, pack.cfg.hidden_size, MAX_ROWS)?;
    let t0 = std::time::Instant::now();
    let mut sh = GpuModel::load_slice_on(dev.clone(), &pack, pack.cfg.num_layers, 4096, SWA_RING, 1, Shard { rank, world })?;
    sh.tp = Some(tp);
    probe_barrier(&mut stream, &mut peers)?;
    println!("[naive-tp] rank {rank}/{world}: {} sharded layers loaded in {:.1} s", pack.cfg.num_layers, t0.elapsed().as_secs_f64());
    let _ctl = stream;
    let ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let pos: Vec<usize> = (0..ids.len()).collect();
    let (lg, taps) = sh.forward_rows(&ids, &pos)?;
    let vl = sh.vl;
    let last = &lg[(ids.len() - 1) * vl..ids.len() * vl];
    let mut idx: Vec<usize> = (0..vl).collect();
    idx.sort_by(|a, b| last[*b].partial_cmp(&last[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let rms = |v: &[f32]| (v.iter().map(|a| (*a as f64) * (*a as f64)).sum::<f64>() / v.len() as f64).sqrt();
    println!("[naive-tp] rank {rank}: residual rms after layer 0 {:.4}, 20 {:.4}, 47 {:.4}", rms(&taps[0]), rms(&taps[20]), rms(&taps[taps.len() - 1]));
    println!("[naive-tp] rank {rank}: top-5 of the local vocabulary half [{}..{}): {:?}", sh.v0, sh.v0 + vl,
             idx[..5].iter().map(|i| (sh.v0 + i, last[*i])).collect::<Vec<_>>());
    if rank == 0 {
        ensure!(sh.v0 + idx[0] == 12095, "NAIVE_TP_FAIL: rank 0 argmax {} is not ' Paris' (12095)", sh.v0 + idx[0]);
        println!("NAIVE_TP_FULL_OK");
    } else { println!("NAIVE_TP_NODE_DONE"); }
    Ok(())
}

/// `--probe-naive-spec --model-dir <pack> --draft-dir <dir> [--layers N] [--spec-block B] [--tokens T] [--spec-mode M] [--tp W]`:
/// the speculative engine path (prefill with taps, rounds, block verify, match acceptance) must produce EXACTLY
/// the plain greedy continuation. Modes: `drafter` (the DSpark drafter), `oracle` (drafts = the plain continuation:
/// every row of the verify block is accepted and checked every round), `miss` (oracle drafts with row (round mod n)
/// wrong: every partial-acceptance path), `mixed` (oracle, with a plain step between rounds: the guard's shape).
/// `--tp 2|4` runs W ranks as threads on ONE GPU over the loopback transport (eager kernels, no graph).
pub fn run_spec_probe(model_dir: &str, draft_dir: &str, n_layers: usize, block: usize, tokens: usize, mode: &str, world: usize, sampler: &str) -> Result<()> {
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    let prompt: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let n = block - 1;
    // plain greedy (one GPU, whole model slice): the reference, tokens + block extra for the oracle drafts
    let mut plain = Engine::load(dev.clone(), &pack, Shard::single(), 4096, n_layers, true, 64)?;
    plain.arm()?;
    plain.begin_request(&SampleParams::GREEDY, 1, &prompt)?;
    let row = plain.prefill(&prompt, 0, true)?.unwrap();
    let (mut t, _) = plain.dec.sample_row(&plain.model, row)?;
    let mut ref_toks = vec![t];
    let mut pos = prompt.len();
    while ref_toks.len() < tokens + block + 1 { let (nt, _) = plain.decode_step(t, pos)?; ref_toks.push(nt); t = nt; pos += 1; }
    drop(plain);
    let ref_toks = Arc::new(ref_toks);
    let sc = SpecCfg { draft_dir: Some(draft_dir.to_string()), block, conf: if mode == "drafter" || mode == "sampled" || mode == "chi" { SPEC_CONF } else { 0.0 }, guard_off: SPEC_GUARD_OFF, guard_on: SPEC_GUARD_ON, probe_every: SPEC_PROBE_EVERY, draft_temp: SPEC_DRAFT_TEMP, drafter: SPEC_DRAFTER.to_string(), sampler: sampler.to_string(), lane_ctx: true };
    // ---- gate (b) of work item A: distribution exactness, two-sample and reference-free -----------
    // The plain step and the FIRST token of a speculative round are both p's draw for the same
    // position: they differ only in the RNG stream and in the accept/residual machinery. So a
    // two-sample chi-square over the 50 most frequent ids plus an "other" bin rejects iff the spec
    // path is not drawing from p. A plain-vs-plain pair on disjoint seed ranges is run as the
    // control: it must NOT reject, and it prices the test's own false-positive rate.
    if mode == "chi" {
        fn gammq(a: f64, x: f64) -> f64 {
            // regularised upper incomplete gamma Q(a, x): series below a+1, continued fraction above
            if x <= 0.0 { return 1.0; }
            let gln = libm_lgamma(a);
            if x < a + 1.0 {
                let (mut ap, mut sum, mut del) = (a, 1.0 / a, 1.0 / a);
                for _ in 0..1000 { ap += 1.0; del *= x / ap; sum += del; if del.abs() < sum.abs() * 1e-15 { break; } }
                1.0 - sum * (-x + a * x.ln() - gln).exp()
            } else {
                let (tiny, mut b, mut c, mut d) = (1e-300f64, x + 1.0 - a, 1.0 / 1e-300f64, 1.0 / (x + 1.0 - a));
                let mut h = d;
                for i in 1..1000 {
                    let an = -(i as f64) * (i as f64 - a);
                    b += 2.0;
                    d = an * d + b; if d.abs() < tiny { d = tiny; }
                    c = b + an / c; if c.abs() < tiny { c = tiny; }
                    d = 1.0 / d;
                    let del = d * c;
                    h *= del;
                    if (del - 1.0).abs() < 1e-15 { break; }
                }
                (-x + a * x.ln() - gln).exp() * h
            }
        }
        fn libm_lgamma(x: f64) -> f64 {
            // Lanczos (g=7, n=9): enough for df/2 in [0.5, 60]
            const C: [f64; 9] = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
                                 -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                                 9.9843695780195716e-6, 1.5056327351493116e-7];
            if x < 0.5 { return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - libm_lgamma(1.0 - x); }
            let x = x - 1.0;
            let mut a = C[0];
            let t = x + 7.5;
            for (i, c) in C.iter().enumerate().skip(1) { a += c / (x + i as f64); }
            0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
        }
        /// two-sample chi-square over the `top` most frequent ids (union) plus one "other" bin
        fn chi2_two(a: &[u32], b: &[u32], top: usize) -> (f64, usize, Vec<(u32, usize, usize)>) {
            use std::collections::HashMap;
            let mut ca: HashMap<u32, usize> = HashMap::new();
            let mut cb: HashMap<u32, usize> = HashMap::new();
            for &t in a { *ca.entry(t).or_insert(0) += 1; }
            for &t in b { *cb.entry(t).or_insert(0) += 1; }
            let mut ids: Vec<u32> = ca.keys().chain(cb.keys()).copied().collect();
            ids.sort_unstable(); ids.dedup();
            ids.sort_by(|x, y| {
                let (ox, oy) = (ca.get(x).copied().unwrap_or(0) + cb.get(x).copied().unwrap_or(0),
                                ca.get(y).copied().unwrap_or(0) + cb.get(y).copied().unwrap_or(0));
                oy.cmp(&ox).then(x.cmp(y))
            });
            ids.truncate(top);
            let (na, nb) = (a.len() as f64, b.len() as f64);
            let mut x2 = 0.0f64;
            let mut rows = Vec::new();
            let (mut oa_o, mut ob_o) = (0usize, 0usize);
            for &id in &ids {
                let (oa, ob) = (ca.get(&id).copied().unwrap_or(0), cb.get(&id).copied().unwrap_or(0));
                oa_o += oa; ob_o += ob;
                if oa + ob < 5 { continue; }
                let (ea, eb) = ((oa + ob) as f64 * na / (na + nb), (oa + ob) as f64 * nb / (na + nb));
                x2 += (oa as f64 - ea).powi(2) / ea + (ob as f64 - eb).powi(2) / eb;
                rows.push((id, oa, ob));
            }
            // the "other" bin absorbs everything outside the listed ids
            let (oa, ob) = (a.len() - oa_o, b.len() - ob_o);
            let (ea, eb) = ((oa + ob) as f64 * na / (na + nb), (oa + ob) as f64 * nb / (na + nb));
            if ea > 0.0 && eb > 0.0 { x2 += (oa as f64 - ea).powi(2) / ea + (ob as f64 - eb).powi(2) / eb; }
            rows.push((u32::MAX, oa, ob));
            (x2, rows.len() - 1, rows)
        }
        let sp = SampleParams { temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 };
        let prefixes: [Vec<u32>; 3] = [vec![785, 6722, 315, 9625, 374], vec![9707, 11, 1879, 311, 279], vec![3838, 374, 220, 16, 13]];
        let n = tokens.max(4000);
        let width = block - 1;   // the verify width: a full block of drafts
        let mut eng = Engine::load_spec(dev.clone(), &pack, Shard::single(), 4096, n_layers, true, 64, Some(&sc))?;
        eng.arm()?;
        fn one(eng: &mut Engine, sp: &SampleParams, pre: &[u32], seed: u64, spec: bool, k: usize) -> Result<u32> {
            eng.begin_request(sp, seed, pre)?;
            let row = eng.prefill(pre, 0, true)?.unwrap();
            let (t0, _) = eng.dec.sample_row(&eng.model, row)?;
            let pos = pre.len();
            if !spec { let (nt, _) = eng.decode_step(t0, pos)?; return Ok(nt); }
            let (d, _c) = eng.draft(t0, pos)?;
            let (c, _, _) = eng.spec_round(Some((d.as_slice(), k)), t0, pos)?;
            Ok(c[0])
        }
        let (mut worst_p, mut fails) = (1.0f64, 0usize);
        println!("gate (b) distribution exactness: two-sample chi-square, top-50 + other, N = {n} draws/path");
        for (pi, pre) in prefixes.iter().enumerate() {
            let mut pl = Vec::with_capacity(n); let mut spc = Vec::with_capacity(n); let mut pl2 = Vec::with_capacity(n);
            for i in 0..n {
                let seed = 1000 + i as u64;
                pl.push(one(&mut eng, &sp, pre, seed, false, width)?);
                spc.push(one(&mut eng, &sp, pre, seed, true, width)?);
                pl2.push(one(&mut eng, &sp, pre, 1_000_000 + i as u64, false, width)?);
            }
            let (x2s, dfs, _) = chi2_two(&pl, &spc, 50);
            let (x2c, dfc, _) = chi2_two(&pl, &pl2, 50);
            let ps = gammq(dfs as f64 / 2.0, x2s / 2.0);
            let pc = gammq(dfc as f64 / 2.0, x2c / 2.0);
            println!("  prefix {pi}: spec vs plain chi2 {x2s:.1} df {dfs} p {ps:.4}  |  control plain vs plain chi2 {x2c:.1} df {dfc} p {pc:.4}");
            ensure!(pc >= 0.01, "NAIVE_CHI_CONTROL_FAIL: the plain-vs-plain control rejected at p = {pc:.4} (prefix {pi}) — the test itself is broken");
            if ps < 0.01 { println!("    SPEC REJECTS at p<0.01 (prefix {pi})"); fails += 1; worst_p = worst_p.min(ps); }
            else { worst_p = worst_p.min(ps); }
        }
        ensure!(fails == 0, "NAIVE_CHI_FAIL: the speculative path rejected distribution exactness on {fails}/3 prefixes (worst p {worst_p:.4})");
        println!("NAIVE_CHI_OK (spec indistinguishable from plain on 3 prefixes at p>=0.01; control clean)");
        return Ok(());
    }
    // one rank's run: returns its committed token stream and the acceptance stats
    fn run(dev: Arc<CudaDevice>, pack: Arc<NaivePack>, prompt: Vec<u32>, refs: Arc<Vec<u32>>, sc: SpecCfg, n_layers: usize,
           tokens: usize, mode: String, rank: usize, world: usize, tp: Option<NaiveTp>) -> Result<(Vec<u32>, usize, usize, f64)> {
        dev.bind_to_thread()?;
        let block = sc.block; let n = block - 1;
        let head = rank == 0;
        // work item A: `sampled` runs the request at the PRODUCTION sampler settings (the pack's
        // generation_config defaults: T 1.0 / top_p 0.95 / top_k 20) so the exact-acceptance path is
        // the one under test; every other mode stays greedy (its gates are token-exactness).
        let sp = if mode == "sampled" {
            SampleParams { temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 }
        } else { SampleParams::GREEDY };
        let mut eng = Engine::load_spec(dev.clone(), &pack, if world == 1 { Shard::single() } else { Shard { rank, world } }, 4096, n_layers, head, 64, Some(&sc))?;
        eng.model.tp = tp;
        if world == 1 { eng.arm()?; }
        eng.begin_request(&sp, 1, &prompt)?;
        let row = eng.prefill(&prompt, 0, true)?.unwrap();
        let (mut t, _) = eng.dec.sample_row(&eng.model, row)?;
        let mut out = vec![t];
        let mut pos = prompt.len();
        let (mut rounds, mut acc) = (0usize, 0usize);
        let mut hist = vec![0usize; n + 1];
        let t0 = std::time::Instant::now();
        let mut r = 0usize;
        while out.len() < tokens {
            if mode == "mixed" && r % 2 == 1 {
                let (nt, _) = eng.decode_step(t, pos)?; out.push(nt); pos += 1; t = nt; r += 1; continue;
            }
            // (drafts, verify width): `adapt` cycles the width 0..=n over oracle drafts (every verify step,
            // including the width-0 plain step, must give the plain continuation); `drafter` uses the
            // confidence head's width; the other modes verify the whole block.
            let drafts: Option<(Vec<u32>, usize)> = if !head { None } else if mode == "drafter" || mode == "sampled" {
                let (d, c) = eng.draft(t, pos)?; let k = eng.pick_width(&c); Some((d, k))
            } else {
                let mut d: Vec<u32> = refs[out.len()..out.len() + n].to_vec();
                if mode == "miss" { let j = r % n; d[j] = if d[j] == 0 { 1 } else { d[j] - 1 }; }
                let k = if mode == "adapt" { r % (n + 1) } else { n };
                Some((d, k))
            };
            let (committed, a, _) = eng.spec_round(drafts.as_ref().map(|(d, k)| (d.as_slice(), *k)), t, pos)?;
            if let Some((_, k)) = drafts.as_ref() {
                // work item A invariants, every round: never accept more rows than were verified,
                // and a round commits exactly a+1 tokens (the a drafts plus the one the target itself
                // drew — the residual when row a rejected, the bonus row when every row accepted).
                ensure!(a <= *k, "NAIVE_SPEC_FAIL: accepted {a} > verified rows {k}");
                ensure!(committed.len() == a + 1, "NAIVE_SPEC_FAIL: committed {} tokens for {a} accepted of {k}", committed.len());
                if mode == "sampled" { hist[a] += 1; }
            }
            rounds += 1; acc += a;
            out.extend_from_slice(&committed);
            pos += a + 1;
            t = *committed.last().unwrap();
            r += 1;
        }
        out.truncate(tokens);
        if head && mode == "sampled" {
            let full = *hist.last().unwrap_or(&0);
            println!("  accept histogram (accepted rows -> rounds): {hist:?}  ({full}/{rounds} rounds accepted every verified row)");
        }
        Ok((out, rounds, acc, t0.elapsed().as_secs_f64()))
    }
    let (spec_toks, rounds, acc, dt) = if world == 1 {
        run(dev.clone(), pack.clone(), prompt.clone(), ref_toks.clone(), sc.clone(), n_layers, tokens, mode.to_string(), 0, 1, None)?
    } else {
        let mut tps = NaiveTp::loopback_group(&dev, pack.cfg.hidden_size, 64.max(MAX_ROWS), world)?;
        let tp0 = tps.remove(0);
        let mut handles = Vec::new();
        for (i, tp) in tps.into_iter().enumerate() {
            let (d, p, pr, rf, s, m) = (dev.clone(), pack.clone(), prompt.clone(), ref_toks.clone(), sc.clone(), mode.to_string());
            handles.push(std::thread::spawn(move || run(d, p, pr, rf, s, n_layers, tokens, m, i + 1, world, Some(tp))));
        }
        let r0 = run(dev.clone(), pack.clone(), prompt.clone(), ref_toks.clone(), sc.clone(), n_layers, tokens, mode.to_string(), 0, world, Some(tp0))?;
        for (i, h) in handles.into_iter().enumerate() {
            let (o, _, _, _) = h.join().map_err(|_| anyhow::anyhow!("rank {} thread panicked", i + 1))??;
            println!("  rank {}: tokens identical to rank 0: {}", i + 1, o == r0.0);
            ensure!(o == r0.0, "NAIVE_SPEC_FAIL: rank {} committed a different stream", i + 1);
        }
        r0
    };
    println!("  plain greedy {:?}", &ref_toks[..tokens.min(16)]);
    println!("  spec  greedy {:?}", &spec_toks[..tokens.min(16)]);
    println!("  mode {mode} world {world}: {rounds} rounds, {:.2} accepted/round, {:.1} ms/round (block {block}, {n_layers} layers)", acc as f64 / rounds.max(1) as f64, dt / rounds.max(1) as f64 * 1e3);
    if mode == "sampled" {
        // A sampled round draws a token the plain path would not draw (different RNG streams), so
        // token-equality is NOT the assertion here. What this mode proves: the chain's structural
        // invariants held every round (checked in `run`), every committed id is in vocabulary, and the
        // path is deterministic for a fixed seed (a second run reproduces the stream). Exactness of
        // the DISTRIBUTION is gate (b): scripts/naive/spec_exact.py.
        ensure!(spec_toks.iter().all(|&t2| (t2 as usize) < pack.cfg.vocab_size), "NAIVE_SPEC_FAIL: sampled stream has an out-of-vocabulary id");
        // work item C's gate: the same seed under `--spec-sampler rows` and `batched` must print the same hash
        let h = spec_toks.iter().fold(0xcbf29ce484222325u64, |h, &t2| (h ^ t2 as u64).wrapping_mul(0x100000001b3));
        println!("  sampled stream hash {h:016x} ({} tokens, {rounds} rounds, {acc} accepted; --spec-sampler {sampler})", spec_toks.len());
        if world == 1 {
            let (b2, r2, a2, _) = run(dev.clone(), pack.clone(), prompt.clone(), ref_toks.clone(), sc.clone(), n_layers, tokens, mode.to_string(), 0, 1, None)?;
            ensure!(b2 == spec_toks && r2 == rounds && a2 == acc,
                    "NAIVE_SPEC_FAIL: sampled path is not deterministic for a fixed seed (rounds {r2} vs {rounds}, accepted {a2} vs {acc})");
            println!("  determinism: a second run with the same seed reproduced the stream ({rounds} rounds, {acc} accepted)");
        }
        println!("NAIVE_SPEC_OK (sampled: invariants + vocabulary + determinism; distribution exactness = spec_exact.py)");
        return Ok(());
    }
    if spec_toks != ref_toks[..tokens] {
        let first = spec_toks.iter().zip(ref_toks.iter()).position(|(a, b)| a != b).unwrap_or(tokens);
        bail!("NAIVE_SPEC_FAIL: speculative greedy differs from plain greedy at token {first} (spec {:?} vs plain {:?})",
              &spec_toks[first.saturating_sub(2)..(first + 3).min(tokens)], &ref_toks[first.saturating_sub(2)..(first + 3).min(tokens)]);
    }
    println!("NAIVE_SPEC_OK");
    Ok(())
}

#[cfg(test)]
mod prefix_tests {
    use super::*;

    #[test]
    fn ring_runs_cover_the_window_in_position_order() {
        assert_eq!(ring_runs(0, 128, 4096), vec![(0, 128)]);
        assert_eq!(ring_runs(7168 - 128, 128, 4096), vec![(7040 & 4095, 128)]);
        // a window that wraps the ring: the tail of the ring, then its head
        assert_eq!(ring_runs(4096 - 28, 128, 4096), vec![(4068, 28), (0, 100)]);
        assert_eq!(ring_runs(8192 + 4000, 128, 4096), vec![(4000, 96), (0, 32)]);
        for p0 in [0usize, 1, 4000, 4095, 12345] {
            let r = ring_runs(p0, 128, 4096);
            assert_eq!(r.iter().map(|x| x.1).sum::<usize>(), 128);
            assert_eq!(r[0].0, p0 & 4095);
        }
    }

    #[test]
    fn the_plan_resumes_at_the_held_chunk_boundary_and_snapshots_the_next() {
        let p: Vec<u32> = (0..7475u32).collect();
        // first request: nothing held; the snapshot is the last chunk boundary at or before prompt - margin
        assert_eq!(plan_prefix(None, &p, None, 1024, 128), (0, Some(7168)));
        // the template's message boundary wins over the margin
        assert_eq!(plan_prefix(None, &p, Some(5000), 1024, 128), (0, Some(4096)));
        // the next turn repeats the first 7168 tokens and goes on: resume there; its own snapshot further on
        let mut q = p.clone(); q.extend(9000..12000u32);
        assert_eq!(plan_prefix(Some(&p[..7168]), &q, None, 1024, 128), (7168, Some(10240)));
        // no boundary beyond the held one: resume, keep the held snapshot (None = no new one)
        let mut r = p[..7168].to_vec(); r.extend(1..500u32);
        assert_eq!(plan_prefix(Some(&p[..7168]), &r, None, 1024, 128), (7168, None));
        // a prompt that differs inside the held prefix, or only repeats it, prefills whole
        let mut d = p.clone(); d[100] = 999_999;
        assert_eq!(plan_prefix(Some(&p[..7168]), &d, None, 1024, 128).0, 0);
        assert_eq!(plan_prefix(Some(&p[..7168]), &p[..7168], None, 1024, 128).0, 0);
        // a short prompt: no snapshot (below one chunk), nothing to resume
        assert_eq!(plan_prefix(None, &p[..900], None, 1024, 128), (0, None));
    }

    #[test]
    fn lanes_follow_max_batch() {
        let a = |v: &[&str]| super::lanes_asked(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(a(&["--max-batch", "4"]), (4, 4));
        assert_eq!(a(&["--max-batch", "2"]), (2, 2));
        assert_eq!(a(&["--max-batch", "16"]), (4, 16)); // capped and announced, not refused
        assert_eq!(a(&[]), (4, 8)); // the generic --max-batch default
        assert_eq!(a(&["--max-batch", "4", "--naive-lanes", "1"]), (1, 1)); // the override wins
        assert_eq!(a(&["--max-batch", "3", "--naive-lanes", "0"]), (3, 3)); // 0 = follow --max-batch
    }

    #[test]
    fn an_admit_without_the_prefix_fields_still_decodes() {
        let j = r#"{"Admit":{"prompt":[1,2,3],"max_new":4,"seed":5,"sp":{"temperature":1.0,"top_p":0.95,"top_k":0,"min_p":0.0,"rep_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0},"min_new":0,"ignore_eos":false}}"#;
        match serde_json::from_str::<Ctl>(j).expect("decode") {
            Ctl::Admit { reuse, snap, .. } => { assert_eq!(reuse, 0); assert_eq!(snap, None); }
            other => panic!("{other:?}"),
        }
    }
}
