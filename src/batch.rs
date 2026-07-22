//! Continuous-batching scheduler for the live server.
//!
//! Owns the GPU. Incoming requests are prefilled into a free physical slot, then all active lanes
//! are decoded together via forward_decode (one shared weight read). Tokens stream back per request.
//! Each lane owns a physical slot (`Lane::phys`); the logical→physical map (`bufs.slot_ids_dev`) is
//! uploaded each step so stateful kernels index state by slot_ids[lane]. Finished lanes return their
//! slot to a free list — no state copying on compaction (slot indirection).

use tokio::sync::mpsc;

use crate::gpu::{BatchGpuState, B, CudaGraph, DecodeBuffers, GpuModel, Pool};
// DevicePtr provides `.device_ptr()` on CudaSlice<T>; needed to read raw device pointers for the
// MTP KV buffers (passed as u64 bases into the stateful batched kernels).
use cudarc::driver::DevicePtr;

/// A token streamed back to a request handler.
pub enum TokEvent {
    Tok(u32),
    Finish { reason: String },
}

/// A request submitted to the scheduler.
pub struct BatchRequest {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub rep_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub tx: mpsc::UnboundedSender<TokEvent>,
    pub seed: Option<u64>,
    /// Token index of the MESSAGE BOUNDARY — the prompt as rendered without the generation prompt.
    /// The scheduler snapshots the GDN recurrent state here, because this is the longest prefix of this
    /// prompt that the next turn will reproduce exactly. `None` = no checkpoint (raw / non-chat paths).
    pub ckpt_at: Option<usize>,
}

struct Lane {
    phys: usize,
    pos: usize,
    last_tok: u32,
    max_new: usize,
    generated: usize,
    greedy: bool,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    rep_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    history: Vec<u32>,
    tx: mpsc::UnboundedSender<TokEvent>,
    /// MTP KV cursor: the main-model position of the last committed token = next MTP write pos - 1.
    /// Only meaningful when this lane is served via the MTP path.
    mtp_pos: usize,
    /// True iff this lane's MTP KV was primed over the prompt at admit. A lane admitted while the
    /// MTP policy was inactive has no primed MTP KV and must never take the MTP path.
    mtp_primed: bool,
    /// Set the moment this lane takes a NON-MTP step. Its MTP KV is now missing entries for every
    /// token decoded since, so the head can never be trusted again for this request.
    ///
    /// Dropping MTP mid-lane is safe (nothing but the MTP path reads that KV). RESUMING it is not,
    /// and the policy really can flip back on -- MtpPolicy retries after MTP_RETRY_AFTER steps. That
    /// would have restarted drafting against a KV with holes in it: the head would attend over
    /// never-written (zero) K/V rows, which still get exp(0)=1 weight in the softmax and quietly
    /// poison every draft. Output stays correct -- the verify rejects the garbage -- so the only
    /// symptom is acceptance silently collapsing, which is exactly the failure mode AGENTS.md warns
    /// no correctness gate can see. Once stale, always stale.
    mtp_stale: bool,
    /// Per-lane PRNG seed for stochastic MTP (LCG: seed = seed*1664525 + 1013904223, matching the
    /// LCG in sample_b/sample_prob_b/spec_verify_b). Host-side accept decisions and device-side
    /// seeds all derive from this single seed, advanced once per random draw.
    seed: u64,
    /// TP=2 serving (item A): a cancel delivered over the per-step wire protocol. On the head it is
    /// set by `run_tp_head` the moment it observes `tx.is_closed()` (and shipped to the mirror as a
    /// `TpEvent::Cancel`); on the mirror it is set when that event arrives. In TP serving mode this
    /// is the ONLY cancel channel — `decode_step`'s sweep must not consult `is_closed()` there, or a
    /// disconnect landing between the head's detection and its sweep would finish the lane one step
    /// early on the head only, desyncing the lockstep. Always false in single-node serving.
    tp_cancelled: bool,
}

impl Lane {
    fn has_penalty(&self) -> bool {
        self.rep_penalty > 1.0 || self.presence_penalty > 0.0 || self.frequency_penalty > 0.0
    }
    /// A lane runs on the MTP path iff its MTP KV was primed at admit AND the policy is currently
    /// active. Whether it then verifies greedily (bitwise lossless) or stochastically
    /// (distribution-exact) follows from `self.greedy` alone — that is a property of the REQUEST
    /// (temperature), never a server setting.
    ///
    /// The priming flag matters because the policy can flip mid-flight: a lane admitted while MTP was
    /// off has no primed MTP KV and must never take the MTP path, and a lane admitted while it was on
    /// can simply stop (its MTP KV is only ever read by the MTP path, so abandoning it is safe).
    fn use_mtp(&self, active: bool) -> bool {
        active && self.mtp_primed && !self.mtp_stale
    }
}

/// Whether MTP pays for itself is a measurable question, not a configuration one.
///
/// A depth-`d` MTP step emits `1 + (accepted drafts)` tokens and costs `r` decode-steps, so
///
/// ```text
/// speedup = (tokens per step) / r        =>       MTP pays iff  tokens_per_step > r
/// ```
///
/// `r` is a pure cost ratio — it depends on the model's shape (above all on what fraction of the
/// weights the LM head is, since drafting must read it a second time to pick a draft token) and not
/// on the prompt. So it is measured once at load. Acceptance, by contrast, is workload-dependent
/// (code accepts differently from prose), so it is tracked live and the decision is revisited.
///
/// This is what replaces `RUST_INFER_MTP` / `RUST_INFER_MTP_STOCHASTIC`. Those env vars encoded a
/// judgement ("MTP is good on 9B, bad on 2B") that the engine can simply measure — and that was wrong
/// the moment either the model or the workload changed.
pub struct MtpPolicy {
    head_present: bool,
    /// Explicit override from `--mtp=on|off`; `None` = auto (the default).
    force: Option<bool>,
    /// Pinned depth from `--mtp-depth`; `None` = the policy chooses.
    pin_depth: Option<usize>,
    depth: usize,
    /// r(d) = MTP step cost / decode step cost, MEASURED per depth by `calibrate_mtp_r`.
    r_by_depth: Vec<(usize, f32)>,
    active: bool,
    // Rolling evaluation window.
    win_steps: u64,
    win_emitted: u64,
    win_drafts: u64,
    win_accepted: u64,
    /// Per-position CONDITIONAL acceptance: `hz[i]` = P(draft i accepted | drafts 1..i-1 accepted),
    /// as (accepted, offered) counts. This is the whole basis of the depth decision — see `yield_at`.
    hz: [(f64, f64); MAX_AUTO_DEPTH],
    decode_steps: u64,
    retry_at: u64,
}

use crate::gpu::MAX_AUTO_DEPTH;

/// MTP steps per policy re-evaluation.
/// Prefill window size. Prefill activation memory is O(window) (~1.2 MiB/token on 9B, more on 27B), so
/// a long prompt is prefilled in windows of this size to bound peak memory (a 256K single-shot prefill
/// would need ~400 GB on 27B). A prompt <= this is one window == the old single-shot path, byte-identical.
/// 8192 keeps typical prompts single-shot while bounding peak to ~13 GB/window on 27B.
const PREFILL_CHUNK: usize = 8192;

const MTP_EVAL_WINDOW: u64 = 128;
/// Decode steps to wait before re-probing a model whose acceptance had fallen below break-even
/// (the workload may have changed — acceptance is not a fixed property of the model).
const MTP_RETRY_AFTER: u64 = 4096;
/// A different depth must beat the current one by this factor to be worth switching to.
const MTP_DEPTH_MARGIN: f32 = 1.05;

/// Expected tokens emitted by one depth-`d` step: `1 + Σ_{i=1..d-1} Π_{j≤i} p_j`.
///
/// **The hazard is NOT constant across positions, and assuming it is was a real bug.** A single `p`
/// fitted at depth 2 is just the FIRST-position hazard — the easiest one — and `p^k` then wildly
/// over-predicts deep yield. Measured on 9B prose: p₁ ≈ 0.83, which predicts 3.94 tokens/step at
/// depth 6; the actual yield at depth 6 is 2.64. The policy believed the model, jumped to depth 6,
/// and sat there ~20% below the optimum (depth 4). Hazards decay because each draft is conditioned
/// on a chain of its own guesses.
///
/// So: measure `p_i` per position, and for positions deeper than anything observed, extrapolate with
/// the LAST observed hazard (the most pessimistic one seen) rather than the first. A depth we have
/// never run is a guess either way — it should be a conservative guess, because the cost of
/// over-reaching (a whole window at a bad depth) is real and the cost of under-reaching is that the
/// next window discovers the truth and goes deeper.
fn yield_at(hz: &[(f64, f64); MAX_AUTO_DEPTH], d: usize) -> f32 {
    const MIN_OBS: f64 = 8.0;                 // below this a hazard is noise, not a measurement
    let mut last = 0.5f64;                    // prior for positions never offered a draft
    let mut acc = 1.0f64;                     // the bonus token, always emitted
    let mut chain = 1.0f64;
    for i in 1..d {
        let p = match hz.get(i - 1) {
            Some(&(a, n)) if n >= MIN_OBS => { last = a / n; last }
            _ => last,                        // never observed this deep: carry the last known hazard
        };
        chain *= p;
        acc += chain;
    }
    acc as f32
}

/// Render the live accept-by-depth curve for the `[mtp]` stats line: one `pos:rate` per observed
/// position (rate = P(draft accepted | earlier drafts accepted); `?` where too few observations).
/// This is the §0 GO/NO-GO signal at a glance — a sharp drop from @1 to @2+ is exactly the gap the
/// chained head-finetune recovers.
fn fmt_accept_by_depth(mtp: &MtpPolicy) -> String {
    mtp.hazard_counts().iter().enumerate().map(|(i, &(a, n))| {
        if n >= 8 { format!("@{}:{:.0}%", i + 1, a as f64 / n as f64 * 100.0) }
        else { format!("@{}:?", i + 1) }
    }).collect::<Vec<_>>().join(" ")
}

impl MtpPolicy {
    pub fn new(head_present: bool, force: Option<bool>, pin_depth: Option<usize>,
               r_by_depth: Vec<(usize, f32)>) -> Self {
        // Start ON when auto: MTP is correctness-neutral either way (greedy is bitwise lossless,
        // stochastic is distribution-exact), so the worst case of guessing wrong is a slightly slow
        // first window, which the evaluation below then corrects.
        let active = head_present && force.unwrap_or(true);
        // Open in the MIDDLE of the range, not at 2. The policy learns a per-position hazard curve,
        // and at depth 2 it only ever observes position 1 — so it would have to extrapolate the whole
        // curve from its easiest point, which is exactly how it used to over-reach. Starting at 4
        // gives the first window three positions of real data to reason from.
        let depth = pin_depth.unwrap_or(4).clamp(2, MAX_AUTO_DEPTH);
        Self { head_present, force, pin_depth, depth, r_by_depth, active,
               win_steps: 0, win_emitted: 0, win_drafts: 0, win_accepted: 0,
               hz: [(0.0, 0.0); MAX_AUTO_DEPTH], decode_steps: 0, retry_at: 0 }
    }
    pub fn active(&self) -> bool { self.active }
    pub fn depth(&self) -> usize { self.depth }
    pub fn head_present(&self) -> bool { self.head_present }
    /// Cumulative per-position conditional acceptance counts (accepted, offered), truncated to the
    /// deepest position ever offered a draft. `hz[i]` = P(draft i+1 accepted | drafts 1..i accepted)
    /// — this IS the accept-by-depth curve the MTP head-finetune GO/NO-GO check needs. Never reset,
    /// so it integrates over the whole server run.
    pub fn hazard_counts(&self) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = self.hz.iter().map(|&(a, n)| (a as u64, n as u64)).collect();
        while v.last().map_or(false, |&(_, n)| n == 0) { v.pop(); }
        v
    }
    pub fn r(&self) -> f32 { self.r_at(self.depth) }
    fn r_at(&self, d: usize) -> f32 {
        self.r_by_depth.iter().find(|&&(x, _)| x == d).map(|&(_, r)| r).unwrap_or(f32::INFINITY)
    }
    /// Acceptance at which MTP breaks even, for reporting: tokens_per_step must exceed `r`.
    pub fn break_even_accept(&self) -> f32 {
        ((self.r() - 1.0) / (self.depth.max(2) - 1) as f32).clamp(0.0, 1.0)
    }

    /// Record one completed MTP step. `accepted` is the accepted PREFIX length: drafts 1..=accepted
    /// were taken and draft `accepted+1` (if there was one) was rejected. That is exactly the
    /// information a per-position hazard needs.
    fn record_step(&mut self, drafts: u64, accepted: u64, emitted: u64) {
        self.win_steps += 1;
        self.win_drafts += drafts;
        self.win_accepted += accepted;
        self.win_emitted += emitted;
        for i in 0..(accepted as usize).min(MAX_AUTO_DEPTH) {
            self.hz[i].0 += 1.0;
            self.hz[i].1 += 1.0;
        }
        // The first rejected position was offered and refused — that is the observation that makes
        // the hazard a hazard rather than an average.
        if (accepted as usize) < drafts as usize {
            if let Some(e) = self.hz.get_mut(accepted as usize) { e.1 += 1.0; }
        }
    }

    /// Called once per decode step. Re-evaluates the decision when a window completes, and re-probes
    /// a disabled model after a cooldown.
    fn tick(&mut self) {
        self.decode_steps += 1;
        if self.force.is_some() || !self.head_present { return; }

        if !self.active {
            if self.decode_steps >= self.retry_at {
                self.active = true;
                self.win_steps = 0; self.win_emitted = 0; self.win_drafts = 0; self.win_accepted = 0;
                eprintln!("[mtp] re-probing (workload may have changed)");
            }
            return;
        }
        if self.win_steps < MTP_EVAL_WINDOW { return; }

        let observed = self.win_emitted as f32 / self.win_steps as f32;
        let acc = self.win_accepted as f32 / self.win_drafts.max(1) as f32;
        self.win_steps = 0; self.win_emitted = 0; self.win_drafts = 0; self.win_accepted = 0;

        // Re-pick the depth. A step buys `yield_at(d)` tokens for `r(d)` decode steps, so maximise the
        // ratio. With a flat-in-N verify, r(d) grows only with the DRAFT chain — which is why the
        // optimum sits well past 2 and has to be chosen rather than configured.
        let cur = yield_at(&self.hz, self.depth) / self.r();
        let (mut best_d, mut best) = (self.depth, cur);
        if self.pin_depth.is_none() {
            for &(d, r) in &self.r_by_depth {
                if r <= 0.0 || !r.is_finite() || d == self.depth { continue; }
                let s = yield_at(&self.hz, d) / r;
                // Hysteresis: a challenger must beat the incumbent by a real margin. Adjacent depths
                // routinely score within a fraction of a percent of each other, and a switch costs a
                // window of relearning — without this the policy flaps 4->5->4 forever.
                if s > best * MTP_DEPTH_MARGIN { best = s; best_d = d; }
            }
        }

        // TP item E: log EVERY window evaluation, not just switches. Under TP=2 both ranks run
        // this policy on bit-identical token history with the head's shipped r(d) table, so the
        // lines must be byte-identical across ranks — and a gate that only diffed switch events
        // could pass vacuously on a no-decision run (AGENTS.md §4.12). All values are pure
        // functions of the deterministic hazard curve.
        eprintln!("[mtp] window: d={} yield {:.2} acc {:.1}% | cur {:.2}x best d={} {:.2}x",
                  self.depth, observed, acc * 100.0, cur, best_d, best);

        if best < 1.0 {
            self.active = false;
            self.retry_at = self.decode_steps + MTP_RETRY_AFTER;
            eprintln!("[mtp] DISABLED: acceptance {:.1}% gives {:.2} tok/step, and no depth beats a \
                       plain decode (best {:.2}x). Re-probing in {} steps.",
                      acc * 100.0, observed, best, MTP_RETRY_AFTER);
            return;
        }
        if best_d != self.depth {
            let hzs: Vec<String> = (0..self.depth.max(best_d) - 1)
                .map(|i| match self.hz.get(i) {
                    Some(&(a, n)) if n >= 8.0 => format!("{:.2}", a / n),
                    _ => "?".to_string(),
                }).collect();
            eprintln!("[mtp] depth {} -> {} ({:.2} -> {:.2} tok/step, r {:.2} -> {:.2}, {:.2}x -> {:.2}x) \
                       hazards [{}]",
                      self.depth, best_d, yield_at(&self.hz, self.depth), yield_at(&self.hz, best_d),
                      self.r(), self.r_at(best_d), cur, best, hzs.join(" "));
            self.depth = best_d;
        }
    }
}

/// Per-step RNG for stochastic MTP. The lane holds a 64-bit key; every draw is an independent
/// splitmix64 of (key, domain, index), and the key advances once per decode step.
///
/// Two properties this has to have, both learned the hard way:
///
/// 1. **Uniforms must actually land in [0,1).** The device kernels run a *32-bit* LCG and take
///    `r = (s >> 8) / 2^24`, which is a 24-bit mantissa. Carrying that state in a u64 and shifting
///    right by 8 leaves a full 32-bit value, so `r` lands in [0, 256) and an `r < ratio` test with
///    `ratio <= 1` succeeds only ~1/256 of the time — silently rejecting ~255/256 of perfectly
///    acceptable drafts (this is what pinned stochastic acceptance at ~1%).
///
/// 2. **Draws must not share a stream across consumers.** Each `spec_verify_b` column advances its
///    seed internally, so handing the columns *consecutive* LCG states makes column `j`'s draw come
///    from the state that seeds column `j+1` — fine only as long as every column draws exactly once,
///    and an outright collision the moment one draws twice. Domain separation removes the coupling
///    entirely, so the host accept decisions, the per-column residual samples, and the bonus sample
///    are independent by construction.
#[inline]
pub(crate) fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Domain separators keeping the host accept stream disjoint from the device column seeds.
pub(crate) const RNG_DOM_VERIFY: u64 = 0x5645_5249_4659_0001; // device seeds for spec_verify_b columns
pub(crate) const RNG_DOM_ACCEPT: u64 = 0x4143_4345_5054_0001; // host accept/reject uniforms
pub(crate) const RNG_DOM_SAMPLE: u64 = 0x5341_4D50_4C45_0001; // device seeds for sample_b (batched path)

/// One independent 32-bit draw from the lane's step key, keyed by domain + index.
#[inline]
pub(crate) fn rng_u32(key: u64, domain: u64, idx: usize) -> u32 {
    (splitmix64(key ^ domain ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 32) as u32
}

/// One independent uniform in [0,1), matching the device's 24-bit mantissa convention.
#[inline]
pub(crate) fn rng_uniform(key: u64, domain: u64, idx: usize) -> f32 {
    (rng_u32(key, domain, idx) >> 8) as f32 * (1.0f32 / 16777216.0)
}

/// Length of the longest common prefix of two token sequences.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Greedy tree accept-walk. Follow the target's argmax down the tree: at the current node, if any child
/// carries the token the target predicts (`preds[current]`), descend into it; else stop. Column 0 is the
/// committed token (always the start). This is the tree generalisation of the chain "accept longest
/// prefix", and greedy exactness is preserved by the same induction: every emitted token is the target's
/// argmax given its accepted prefix, so the output is byte-identical to plain autoregressive decode.
///
/// `parent[c]` is c's tree parent (parent[0] = -1). `tokens[c]` is the draft token at column c (tokens[0]
/// is the committed token). `preds[c]` is the target's argmax AFTER the path ending at c. Returns:
/// - `path`: the accepted node indices, root first (`path[0] == 0`).
/// - `emitted`: the tokens to emit = `preds` along the accepted path = each step's correction/accepted
///   token, ending with the bonus `preds[leaf]`.
///
/// A tie prefers the LOWEST child index (deterministic); duplicate sibling tokens are de-duped upstream.
fn tree_accept_walk(parent: &[i32], tokens: &[u32], preds: &[u32]) -> (Vec<usize>, Vec<u32>) {
    let n = parent.len();
    // children[c] in ascending index order (index order == DFS order here, so lowest = first-drafted).
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for c in 1..n { children[parent[c] as usize].push(c); }

    let mut path = vec![0usize];
    let mut emitted = Vec::new();
    let mut cur = 0usize;
    loop {
        let want = preds[cur];
        emitted.push(want); // the token the target actually chose after `cur` (accepted child or bonus)
        match children[cur].iter().copied().find(|&c| tokens[c] == want) {
            Some(c) => { path.push(c); cur = c; }
            None => break, // no child matched: `want` is the correction/bonus, walk ends
        }
    }
    (path, emitted)
}

pub struct BatchScheduler {
    gpu: GpuModel,
    pool: Pool,
    state: BatchGpuState,
    bufs: DecodeBuffers,
    graphs: std::collections::HashMap<usize, CudaGraph>,
    kv_stride: usize,
    /// EVERY token that ends an assistant turn. Not one token — a SET.
    ///
    /// Qwen3.5's config.json advertises `<|endoftext|>` while a chat turn actually ends with
    /// `<|im_end|>`. Stopping on only the advertised one let the model run past the end of its own
    /// turn and hallucinate the next one. See QwenTokenizer::stop_token_ids.
    eos: Vec<u32>,
    max_batch: usize,
    rx: mpsc::UnboundedReceiver<BatchRequest>,
    lanes: Vec<Option<Lane>>,
    /// Free physical slots (stack) available for new admissions. Each active lane owns a physical
    /// slot (`Lane::phys`) holding its persistent KV + GDN state; on finish the slot is returned here
    /// instead of copying state into a contiguous prefix (slot indirection via `bufs.slot_ids_dev`).
    free_slots: Vec<usize>,
    /// PREFIX CACHE, one entry per PHYSICAL slot: the exact token sequence this slot's KV cache, GDN
    /// recurrent state and conv1d state currently reflect — i.e. the state is the state AFTER these
    /// tokens. Empty when the slot holds nothing reusable.
    ///
    /// Why per-slot and not a shared radix tree: KV is position-addressable, so any prefix of it stands
    /// on its own; THE GDN RECURRENT STATE IS NOT. It exists only at the single point in the sequence
    /// where the scan was left. A prefix is therefore only reusable from the slot that was carried to
    /// exactly that token and no further — which is precisely the append-only shape a chat or an agent
    /// transcript has anyway.
    slot_cache: Vec<Vec<u32>>,
    /// The token sequence at which each slot's PROMPT CHECKPOINT was taken — i.e. the whole prompt of
    /// the request that last ran in that slot, with the GDN state snapshotted at its final token
    /// (before any generation moved it).
    ///
    /// `slot_cache` alone gives a hit only when the client replays our generated tokens VERBATIM.
    /// A tool-calling agent never does: it re-renders our assistant turn from structured JSON, without
    /// the `<think>` block we actually emitted, so the sequence diverges a few tokens into our own
    /// reply. Measured on tool-eval-bench: 0 hits in 93 requests, and 88% of ALL prefill tokens were
    /// ones we had already computed — the match point sat a median of 202 tokens BEFORE the only GDN
    /// state we held. The prompt boundary is exactly where those misses want to resume, so we keep a
    /// second, older state there. KV needs no snapshot: it is position-addressable and still valid.
    slot_ckpt_seq: Vec<Vec<u32>>,
    /// First of `max_batch` state slots holding the prompt checkpoints (lane i -> prompt_ckpt_slot + i).
    prompt_ckpt_slot: usize,
    /// Prompt-lookup draft n-gram order (0 = off). When the last `ngram_draft` tokens of a lane's
    /// context recur earlier, the follower is proposed as a draft instead of the 1-layer head's guess.
    /// Free (host-side) and lossless (the verify checks every draft); a big acceptance win on copyable
    /// tool/JSON text (60.3% -> 71.1% measured), a no-op on prose. See gpu.rs::bench_accept.
    ngram_draft: usize,
    /// Fork-then-chain tree drafting (opt-in). Lossless (verify checks every draft); the k=2 fork at
    /// position 1 rescues the chain-killing first-token miss. See mtp_tree_step.
    tree_draft: bool,
    /// Batched MTP verify across lanes (LANES design Step 3c): pack several concurrent lanes' draft
    /// chains into ONE forest verify. Opt-in; needs the full MAX_VERIFY checkpoint region.
    mtp_lanes: bool,
    /// Reuse a cached prefix instead of prefilling it again. OFF by default, and that is a deliberate
    /// correctness choice, not caution: reusing a prefix RE-CHUNKS the prefill, and prefill runs on
    /// cuBLAS, which picks a different kernel per shape (AGENTS.md §2.4). So a cached turn is not
    /// bit-identical to a cold one — the same conversation can word its answer slightly differently
    /// depending on what happened to be in the cache. Every engine with prefix caching has this
    /// property; ours states it. The snapshot/restore ITSELF is bit-exact (tests/prompt_ckpt_test.rs).
    prefix_cache: bool,
    /// GPU-side sampling (temp/top-k/top-p/multinomial in `sample_b`) rather than a full-logit dtoh
    /// plus a CPU sampler. Now the default — every run script already set it, and pulling a
    /// 248k-entry logit vector to the host per token is strictly worse. `RUST_INFER_CPU_SAMPLE=1`
    /// keeps the old path available as an escape hatch.
    gpu_sample: bool,
    /// Captured decode+sample graphs (per batch size) for the GPU-sampling path, when enabled.
    sample_graphs: std::collections::HashMap<usize, CudaGraph>,

    // ---- MTP (multi-token-prediction speculative decoding) ----
    /// Decides for itself whether MTP pays, from a measured cost ratio plus live acceptance.
    /// Greedy vs stochastic verify is NOT part of this — that follows from the request's temperature.
    mtp: MtpPolicy,
    /// Per-physical-slot MTP KV cache: `[nkv, kv_stride, hd]` bf16 each (packed `[nkv, kv_stride,
    /// hd/16×9]` bytes under the 4-bit KV cache). Indexed by `Lane::phys`. Empty when MTP is disabled.
    mtp_kc: Vec<B>,
    mtp_vc: Vec<B>,
    /// Per-physical-slot MTP hidden cursor `[h]` bf16: the pre-norm backbone hidden at the last
    /// committed position (seeds the next draft chain). Empty when MTP is disabled.
    mtp_h_prev: Vec<B>,
    /// Shared (single active MTP lane at a time) scratch buffers, allocated when MTP is enabled.
    mtp_h_save: Option<B>,      // `[h]`: snapshot of h_prev before a step (for post-accept re-prime)
    mtp_h_scratch: Option<B>,   // `[h]`: hidden-column extract scratch
    mtp_cur_hidden: Option<B>,  // `[h]`: draft-chain cursor hidden
    /// All-zero slot ids for the MTP attention (its KV is per-lane, so the slot is always 0).
    /// Sized for the widest MTP forward, NOT for max_batch: the MTP head runs at batch = depth
    /// Reserved physical slot (never assigned to a lane) used as the GDN-rollback snapshot target.
    /// `copy_gdn_slot(state, phys, snapshot_slot)` snapshots; the reverse restores on partial reject.
    mtp_snapshot_slot: usize,
    /// Persistent penalty buffers for the MTP verify path (depth positions). Filled per-step from
    /// the lane's committed history so greedy MTP lanes keep their repetition/presence/frequency
    /// penalty (no repetition). Stored on the compute stream's device.
    mtp_pen_tokens: Option<cudarc::driver::CudaSlice<i32>>,  // [depth * MAX_PEN_TOKENS]
    mtp_pen_counts: Option<cudarc::driver::CudaSlice<i16>>,  // [depth * MAX_PEN_TOKENS]
    mtp_pen_rep: Option<cudarc::driver::CudaSlice<f32>>,     // [depth]
    mtp_pen_presence: Option<cudarc::driver::CudaSlice<f32>>, // [depth]
    mtp_pen_freq: Option<cudarc::driver::CudaSlice<f32>>,    // [depth]
    /// Persistent penalty buffers for the MTP DRAFT path (1 column, for the MTP head before
    /// sampling). Like mtp_pen_* but sized for batch=1.
    mtp_draft_pen_tokens: Option<cudarc::driver::CudaSlice<i32>>,  // [MAX_PEN_TOKENS]
    mtp_draft_pen_counts: Option<cudarc::driver::CudaSlice<i16>>,  // [MAX_PEN_TOKENS]
    mtp_draft_pen_rep: Option<cudarc::driver::CudaSlice<f32>>,     // [1]
    mtp_draft_pen_presence: Option<cudarc::driver::CudaSlice<f32>>, // [1]
    mtp_draft_pen_freq: Option<cudarc::driver::CudaSlice<f32>>,    // [1]
    /// Cache key for the CONSTANT penalty values (rep, presence, freq bit patterns + width): the
    /// penalty VALUES are request-constant, so re-uploading them every MTP step is 3 blocking
    /// copies of pure waste. Re-upload only when the key changes (a request with different
    /// penalties arrives). The token/count history still uploads per step — it changes every step.
    pen_const_key: Option<(u32, u32, u32, usize)>,
    /// Whether the last batched_decode step uploaded penalty data. Guards the five penalty-array
    /// uploads: skipped entirely when no lane is penalized (the kernel skips -1 sentinels), with a
    /// one-shot clear when the last penalized lane departs so its values cannot linger.
    pen_had: bool,
    // ---- Live MTP acceptance telemetry (stderr, every N MTP lane-steps) ----
    mtp_stat_steps: u64,     // number of mtp_lane_step invocations
    mtp_stat_drafts: u64,    // total draft tokens proposed (depth-1 per step)
    mtp_stat_accepted: u64,  // total drafts accepted (matched verify argmax)
    mtp_stat_emitted: u64,   // total tokens emitted via the MTP path (accepted drafts + bonuses)
    mtp_stat_verify_fwds: u64, // total main-model verify+reverify forwards (cost)
    /// Env-gated (`MTP_DRAFT_LOG=path`) JSONL log of every chain-MTP step:
    /// `{step, lane, pos, committed, drafts, preds, nacc}` (preds = verify argmax per column; empty
    /// on the stochastic path where the verify samples instead). This is the engine-side reference
    /// the head-finetune runbook's B0 parity gate diffs the HF MTP module against. Log-only.
    mtp_draft_log: Option<std::fs::File>,
    /// Env-gated (`MTP_CURVE_FILE=path`) JSON dump of the cumulative accept-by-depth curve
    /// (`MtpPolicy::hazard_counts`), overwritten every 50 MTP steps — the runbook §0 baseline.
    mtp_curve_path: Option<String>,
    /// TP=2 serving (item A): set by `run_tp_head` / `run_tp_mirror`. Gates the cancel sweep in
    /// `decode_step` to wire-delivered cancels ONLY (`Lane::tp_cancelled`) — see that field.
    tp_serving: bool,
}

impl BatchScheduler {
    pub fn new(gpu: GpuModel, max_batch: usize, kv_stride: usize, eos: Vec<u32>,
               rx: mpsc::UnboundedReceiver<BatchRequest>,
               mtp: MtpPolicy, prefix_cache: bool, ngram_draft: usize, tree_draft: bool, mtp_lanes: bool) -> Self {
        // When MTP is on, reserve one extra physical slot as a shared GDN-rollback snapshot target
        // (MTP lanes run one at a time, so a single snapshot slot suffices). It is never assigned to
        // a lane; `copy_gdn_slot(state, phys, snapshot_slot)` snapshots before verify, the reverse
        // restores on partial reject.
        // Reserve the snapshot slot whenever the model HAS an MTP head, not merely when MTP is
        // currently active: the policy can switch MTP on later and the slot must already exist.
        let mtp_has_head = mtp.head_present();
        // Size for the DEEPEST depth the policy may choose, not the one it opens at — the policy
        // re-picks depth from live acceptance every window, and a buffer sized to the initial depth
        // would be silently overrun the moment it went deeper.
        let mtp_depth = crate::gpu::MAX_AUTO_DEPTH;
        // ONE checkpoint slot per verify column we might roll back to: nacc can be 0..depth-2, so
        // we need depth-1 of them, contiguous from `mtp_snapshot_slot`. The GDN kernels write
        // column t's post-state into slot (mtp_snapshot_slot + t) via a derived stride.
        //
        // A single slot is only correct at depth 2. With depth >= 3 it silently rolled the recurrent
        // state back past accepted drafts, which is why greedy MTP was not lossless above depth 2.
        let mtp_snapshot_slot = max_batch;
        let n_ckpt = if mtp_has_head { if tree_draft || mtp_lanes { crate::gpu::MAX_VERIFY } else { mtp_depth.saturating_sub(1).max(1) } } else { 0 };
        // One PROMPT checkpoint slot per lane, after the MTP snapshot slots. These hold the GDN state
        // as it stood at the END OF PREFILL — see `prompt_ckpt_slot`. They are pure state: no KV, and
        // none at all when prefix caching is off (51 MB/slot on 9B, 154 MB on 27B).
        let prompt_ckpt_slot = max_batch + n_ckpt;
        let n_state_slots = max_batch + n_ckpt + if prefix_cache { max_batch } else { 0 };
        let mut state = gpu.new_batch_state(max_batch, n_state_slots, kv_stride);
        gpu.dev().synchronize().unwrap(); // ensure state allocs visible to non-blocking stream
        let mut pool = Pool::new(gpu.dev().clone());
        let mut bufs = gpu.new_decode_buffers(max_batch);

        // Capture decode graphs for all batch sizes, IF every kernel in the captured region fits the
        // default 48 KB/block limit.
        //
        // THIS USED TO BE `(kv_stride + head_dim) * 4`, which described the PRE-split-K attention
        // kernel: it held a kv_stride-sized score array in shared memory, so smem grew with context and
        // graphs were skipped beyond ~12K positions. The old comment even said "split-K will fix this".
        // Split-K shipped — `gqa_attn_splitk` sizes its smem by WARP COUNT, not context — but the guard
        // was never updated, so every long-context deployment silently lost CUDA graphs: 129 KB computed
        // at seq 32768 (the 122B preset), and at the 256K target envelope graphs would never capture at
        // all, on any model.
        //
        // The real requirement is the MAX over the kernels inside `forward_decode_gpu`, all of which are
        // constant in context length:
        //   GDN delta_step_b  18.4 KB  (gdn_launch: kd*(GDN_C+1) + kd + 3*GDN_C + 2*kd floats)  <- max
        //   attention splitk   8.1 KB  ((nw*hd + 2*nw) * 4, nw = hd/32)
        //   argmax_b           8.0 KB
        //   rmsnorm_b          4.0 KB
        // Capture already degrades safely (`capture_decode_graph` returns None and we fall back), so this
        // guard is belt-and-braces — but it must not be the thing that disables graphs everywhere.
        let head_dim = 256;
        let c = gpu.cfg();
        let gdn_smem = crate::gpu::gdn_launch(c.lin_k_dim, c.lin_v_dim).1 as usize;
        let attn_smem = ((head_dim / 32) * head_dim + 2 * (head_dim / 32)) * 4;
        let smem_bytes = gdn_smem.max(attn_smem).max(1024 * 8);
        // A/B switch: GB10_NO_DECODE_GRAPHS=1 forces the non-graph path, so the value of capture can be
        // measured on any config without rebuilding (and gives an escape hatch if capture ever misbehaves).
        let smem_bytes = if std::env::var("GB10_NO_DECODE_GRAPHS").is_ok() { usize::MAX } else { smem_bytes };
        let mut graphs = std::collections::HashMap::new();
        if smem_bytes <= 48 * 1024 {
            print!("Attempting CUDA graph capture for batch sizes 1..={}... ", max_batch);
            let mut ok = true;
            for b in 1..=max_batch {
                match gpu.capture_decode_graph(&mut pool, &mut bufs, &mut state,
                                                 kv_stride, kv_stride, b) {
                    Some(g) => { graphs.insert(b, g); }
                    None => { ok = false; break; }
                }
            }
            // Graph capture (and its warmup) advanced the stateful GDN recurrent state — reset.
            gpu.zero_state(&mut state);
            gpu.dev().synchronize().unwrap(); // ensure zeroed state visible to non-blocking stream
            if ok {
                println!("captured (smem {} KB).", smem_bytes / 1024);
            } else {
                graphs.clear();
                println!("unsupported (legacy stream); using non-graph decode.");
            }
        } else {
            println!("Skipping CUDA graphs: attention smem {} KB > 48 KB. Using non-graph decode.",
                     smem_bytes / 1024);
        }

        // When GPU sampling is enabled, also capture a decode+sample graph per batch size so that
        // sampling requests get the same graph speedup greedy does. Falls back to the non-graph
        // sample path if capture is unsupported.
        let gpu_sample = std::env::var("RUST_INFER_CPU_SAMPLE").is_err();
        let mut sample_graphs: std::collections::HashMap<usize, CudaGraph> = std::collections::HashMap::new();
        if gpu_sample && !graphs.is_empty() {
            for b in 1..=max_batch {
                if let Some(g) = gpu.capture_decode_sample_graph(&mut pool, &mut bufs, &mut state,
                                                                  kv_stride, kv_stride, b) {
                    sample_graphs.insert(b, g);
                }
            }
            gpu.zero_state(&mut state);
            gpu.dev().synchronize().unwrap();
            println!("captured {} GPU-sample graph(s).", sample_graphs.len());
        }

        // Allocate per-slot MTP state when MTP is enabled. The MTP KV (`[nkv, kv_stride, hd]` bf16
        // per slot) must be zeroed: alloc_zeros is cuMemAllocAsync which does NOT zero, and stale
        // GPU garbage in unwritten MTP KV positions would yield nondeterministic drafts. Zero on the
        // COMPUTE stream (ordered with all later kernels), then sync.
        let (mtp_kc, mtp_vc, mtp_h_prev, mtp_h_save, mtp_h_scratch, mtp_cur_hidden,
             mtp_pen_tokens, mtp_pen_counts, mtp_pen_rep, mtp_pen_presence, mtp_pen_freq,
             mtp_draft_pen_tokens, mtp_draft_pen_counts, mtp_draft_pen_rep, mtp_draft_pen_presence,
             mtp_draft_pen_freq) =
            if mtp_has_head {
                let cfg = gpu.cfg();
                let h = cfg.hidden_size;
                let nkv = cfg.num_kv_heads;
                let hd = cfg.head_dim;
                let kv_bytes = nkv * kv_stride * hd * 2; // bf16 — the DRAFT cache stays bf16 even
                // under the 4-bit main KV cache (quantized draft KV costs real acceptance).
                let dev = gpu.dev().clone();
                let mut kc: Vec<B> = Vec::with_capacity(max_batch);
                let mut vc: Vec<B> = Vec::with_capacity(max_batch);
                let mut hp: Vec<B> = Vec::with_capacity(max_batch);
                for _ in 0..max_batch {
                    let k = dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
                    let v = dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
                    let p = dev.alloc_zeros::<half::bf16>(h).unwrap();
                    gpu.memset_compute_stream(*k.device_ptr(), kv_bytes);
                    gpu.memset_compute_stream(*v.device_ptr(), kv_bytes);
                    kc.push(k);
                    vc.push(v);
                    hp.push(p);
                }
                let save = dev.alloc_zeros::<half::bf16>(h).unwrap();
                let scr = dev.alloc_zeros::<half::bf16>(h).unwrap();
                let cur = dev.alloc_zeros::<half::bf16>(h).unwrap();
                gpu.sync_stream(); // ensure MTP KV zeroing visible before any primed lane reads it
                // Penalty buffers for the verify path (depth positions). bf16 lanes already keep
                // penalties via the normal path; MTP greedy lanes now keep theirs too.
                let mp = crate::gpu::MAX_PEN_TOKENS;
                // Sized to MAX_VERIFY (not just the chain depth): a FOREST verify has up to MAX_VERIFY
                // columns spanning several lanes, each column carrying ITS lane's penalty (per-column).
                let pcap = crate::gpu::MAX_VERIFY;
                let pen_tokens = dev.alloc_zeros::<i32>(pcap * mp).unwrap();
                let pen_counts = dev.alloc_zeros::<i16>(pcap * mp).unwrap();
                let pen_rep = dev.alloc_zeros::<f32>(pcap).unwrap();
                let pen_presence = dev.alloc_zeros::<f32>(pcap).unwrap();
                let pen_freq = dev.alloc_zeros::<f32>(pcap).unwrap();
                let dp_tokens = dev.alloc_zeros::<i32>(mp).unwrap();
                let dp_counts = dev.alloc_zeros::<i16>(mp).unwrap();
                let dp_rep = dev.alloc_zeros::<f32>(1usize).unwrap();
                let dp_presence = dev.alloc_zeros::<f32>(1usize).unwrap();
                let dp_freq = dev.alloc_zeros::<f32>(1usize).unwrap();
                (kc, vc, hp, Some(save), Some(scr), Some(cur),
                 Some(pen_tokens), Some(pen_counts), Some(pen_rep), Some(pen_presence), Some(pen_freq),
                 Some(dp_tokens), Some(dp_counts), Some(dp_rep), Some(dp_presence), Some(dp_freq))
            } else {
                // MTP absent: allocate MINIMAL 1-element dummy per-lane buffers so the unconditional
                // `mtp_kc[phys]`/`mtp_vc[phys]` indexing on the prefill/decode paths never panics. These
                // pointers are never dereferenced — every real MTP use is gated by `will_use_mtp` /
                // `self.mtp.active()`, both false without a head.
                let dev = gpu.dev().clone();
                let dummy = |n: usize| -> Vec<B> {
                    (0..n).map(|_| dev.alloc_zeros::<half::bf16>(1).unwrap()).collect()
                };
                (dummy(max_batch), dummy(max_batch), dummy(max_batch),
                 None, None, None, None, None, None, None, None,
                 None, None, None, None, None)
            };

        Self { gpu, pool, state, bufs, graphs, kv_stride, eos, max_batch, rx,
               lanes: (0..max_batch).map(|_| None).collect(),
               free_slots: (0..max_batch).rev().collect(),
               slot_cache: vec![Vec::new(); max_batch],
               slot_ckpt_seq: vec![Vec::new(); max_batch],
               prompt_ckpt_slot,
               prefix_cache,
               ngram_draft,
               tree_draft,
               mtp_lanes,
               gpu_sample,
               sample_graphs,
               mtp,
               mtp_kc,
               mtp_vc,
               mtp_h_prev,
               mtp_h_save,
               mtp_h_scratch,
               mtp_cur_hidden,
               mtp_snapshot_slot,
               mtp_pen_tokens,
               mtp_pen_counts,
               mtp_pen_rep,
               mtp_pen_presence,
               mtp_pen_freq,
               mtp_draft_pen_tokens,
               mtp_draft_pen_counts,
               mtp_draft_pen_rep,
               mtp_draft_pen_presence,
               mtp_draft_pen_freq,
               pen_const_key: None,
               pen_had: false,
               mtp_stat_steps: 0,
               mtp_stat_drafts: 0,
               mtp_stat_accepted: 0,
               mtp_stat_emitted: 0,
               mtp_stat_verify_fwds: 0,
               mtp_draft_log: std::env::var("MTP_DRAFT_LOG").ok().and_then(|p| {
                   match std::fs::File::create(&p) {
                       Ok(f) => { eprintln!("[mtp] draft log -> {}", p); Some(f) }
                       Err(e) => { eprintln!("[mtp] WARN: cannot open MTP_DRAFT_LOG {}: {}", p, e); None }
                   }
               }),
               mtp_curve_path: std::env::var("MTP_CURVE_FILE").ok(),
               tp_serving: false,
        }
    }

    /// Run the scheduler loop until the request channel closes and no lanes remain.
    pub async fn run(mut self) {
        loop {
            // Admit queued requests into free lanes (front-packed).
            while self.num_active() < self.max_batch
                  && self.lanes[self.num_active()].is_none() {
                match self.rx.try_recv() {
                    Ok(req) => self.admit(req),
                    Err(_) => break,
                }
            }
            let b = self.num_active();
            if b == 0 {
                match self.rx.recv().await {
                    Some(req) => { self.admit(req); continue; }
                    None => break,
                }
            }
            self.decode_step(b);
            // Yield to tokio so streaming handlers can flush SSE events between decode steps.
            tokio::task::yield_now().await;
        }
    }

    /// TP item D — per-step SPMD divergence guard. After every executed decode step, both ranks
    /// exchange `(step, cumulative emitted, state hash)` over the link's agreement channel
    /// (`net::agree`, 10 s deadline) and REFUSE to continue on mismatch or timeout. The hash folds
    /// the per-lane output state (`last_tok`, `generated`, `pos`, `mtp_pos`) plus the MTP policy
    /// decision (`active`, `depth`) — identical on both ranks by construction, so any acceptance
    /// or policy divergence flips it within one step. On mismatch: abort the link cooperatively
    /// (kernels no-op through the stream, I9) and return Err — the head's die-with-it guard exits
    /// the server and the mirror's supervisor re-arms, rather than serve one more divergent token.
    /// `GB10_TP_AGREE_DRILL=<step>` corrupts this rank's hash at one step (the forced-divergence
    /// drill gate; env read directly, test-only).
    fn tp_agree_step(&self, step: u64) -> anyhow::Result<()> {
        let mut h: u32 = 0x811c9dc5;                     // FNV-1a over the lane/policy state
        let mut mix = |x: u32| { h ^= x; h = h.wrapping_mul(0x0100_0193); };
        let mut total_generated: u64 = 0;
        for l in self.lanes.iter().flatten() {
            mix(l.last_tok);
            mix(l.generated as u32);
            mix(l.pos as u32);
            mix(l.mtp_pos as u32);
            total_generated += l.generated as u64;
        }
        mix(self.mtp.active() as u32);
        mix(self.mtp.depth() as u32);
        if let Ok(d) = std::env::var("GB10_TP_AGREE_DRILL") {
            if d.parse::<u64>().ok() == Some(step) {
                eprintln!("[tp-agree] DRILL: corrupting this rank's hash at step {step}");
                h ^= 0xDEAD;
            }
        }
        let count = (total_generated & 0xFF) as u8;
        let (pc, ph) = match crate::net::agree(step, count, h) {
            Some(x) => x,
            None => {
                eprintln!("[tp-agree] link aborted or peer timeout at step {step} — aborting");
                crate::net::abort_link();
                anyhow::bail!("TP agree: link aborted/timeout at step {step}");
            }
        };
        if pc != count || ph != h {
            eprintln!("[tp-agree] MISMATCH at step {step}: local (count {count}, hash {h:#010x}) \
                       vs peer (count {pc}, hash {ph:#010x}) — aborting rather than serving \
                       divergent output");
            crate::net::abort_link();
            anyhow::bail!("TP AGREE MISMATCH at step {step}: local ({count}, {h:#010x}) vs peer ({pc}, {ph:#010x})");
        }
        Ok(())
    }

    /// TP=2 serving head loop (item A): the same loop shape as `run()`, plus a per-step rendezvous
    /// with the node's mirror over the retained sync stream. Per step:
    ///   1. Drain admissions under the SAME capacity gate as `run()`, but DEFER the `admit()` calls
    ///      until after the Step message is sent — admission prefill runs SPMD all-reduces that the
    ///      mirror can only join once it has seen the admission, so shipping first is what keeps
    ///      the barrier sequences paired instead of deadlocked. Each drained request becomes a
    ///      `TpEvent::Admit`; `admit()` itself runs unmodified (its context-length reject is
    ///      deterministic from identical state, so the mirror rejects exactly the same requests).
    ///   2. Cancel detection: lanes whose client went away are marked `tp_cancelled` NOW (host
    ///      state, no race with decode_step's sweep) and shipped as `TpEvent::Cancel`. Indices are
    ///      the post-admission front-packed table; admissions only append past the active region,
    ///      so they cannot renumber these.
    ///   3. Send `ServingMsg::Step` — every executed step, even with an empty event list; it is the
    ///      rendezvous the mirror waits on.
    ///   4. Admit the pendings (in order), then one `decode_step` over the new front-packed table.
    /// On request-channel close (server shutdown): keep stepping until the live lanes drain, then
    /// send `ServingMsg::Shutdown` and return.
    pub async fn run_tp_head(mut self, mut stream: std::net::TcpStream) -> anyhow::Result<()> {
        self.tp_serving = true;
        let mut step: u64 = 0;
        let mut closed = false;
        loop {
            let mut events: Vec<crate::tp_serve::TpEvent> = Vec::new();
            let mut pending: Vec<BatchRequest> = Vec::new();
            if !closed {
                loop {
                    let projected = self.num_active() + pending.len();
                    if projected >= self.max_batch || self.lanes[projected].is_some() { break; }
                    match self.rx.try_recv() {
                        Ok(req) => {
                            events.push(crate::tp_serve::TpEvent::Admit((&req).into()));
                            pending.push(req);
                        }
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => { closed = true; break; }
                    }
                }
                // Idle: block for the next request, exactly like run() — the mirror blocks on the
                // session stream meanwhile, so both ranks sleep in step.
                if self.num_active() == 0 && pending.is_empty() && !closed {
                    match self.rx.recv().await {
                        Some(req) => {
                            events.push(crate::tp_serve::TpEvent::Admit((&req).into()));
                            pending.push(req);
                        }
                        None => closed = true,
                    }
                }
            }
            if closed && pending.is_empty() && self.num_active() == 0 {
                crate::tp_serve::send_serving(&mut stream, &crate::tp_serve::ServingMsg::Shutdown)?;
                return Ok(());
            }
            for i in 0..self.num_active() {
                if self.lanes[i].as_ref()
                    .map_or(false, |l| l.tx.is_closed() && !l.tp_cancelled) {
                    self.lanes[i].as_mut().unwrap().tp_cancelled = true;
                    events.push(crate::tp_serve::TpEvent::Cancel { lane: i });
                }
            }
            crate::tp_serve::send_serving(&mut stream, &crate::tp_serve::ServingMsg::Step(
                crate::tp_serve::StepEvents { step, events }))?;
            for req in pending { self.admit(req); }
            let b = self.num_active();
            if b > 0 { self.decode_step(b); self.tp_agree_step(step)?; }
            step += 1;
            // Yield to tokio so streaming handlers can flush SSE events between decode steps.
            tokio::task::yield_now().await;
        }
    }

    /// TP=2 serving mirror loop (node, rank 1): block on the head's per-step events, replay them in
    /// order, and run the identical `decode_step`. Scheduler state stays identical BY CONSTRUCTION —
    /// the forward's all-reduces keep both ranks bitwise in lockstep and every admission/cancel is
    /// replayed from the wire at the same step index. Tokens decode into a dummy channel whose
    /// receiver is held in `keepalive` forever, so a mirror lane's `tx.is_closed()` never fires on
    /// its own: cancels arrive exclusively as wire events (see `Lane::tp_cancelled`).
    pub async fn run_tp_mirror(mut self, mut stream: std::net::TcpStream) -> anyhow::Result<()> {
        self.tp_serving = true;
        let mut keepalive: Vec<mpsc::UnboundedReceiver<TokEvent>> = Vec::new();
        let mut step: u64 = 0;
        loop {
            let msg = match crate::tp_serve::recv_serving(&mut stream) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[tp-mirror] session stream ended ({e:#}) — mirror exit at step {step}");
                    return Ok(());
                }
            };
            match msg {
                crate::tp_serve::ServingMsg::Shutdown => {
                    eprintln!("[tp-mirror] head shut down at step {step} — mirror exit");
                    return Ok(());
                }
                crate::tp_serve::ServingMsg::Step(se) => {
                    anyhow::ensure!(se.step == step,
                        "TP step desync: head sent step {}, mirror is at step {step}", se.step);
                    for ev in se.events {
                        match ev {
                            crate::tp_serve::TpEvent::Admit(w) => {
                                let (tx, rx) = mpsc::unbounded_channel::<TokEvent>();
                                keepalive.push(rx);
                                self.admit(w.into_request(tx));
                            }
                            crate::tp_serve::TpEvent::Cancel { lane } => {
                                match self.lanes.get_mut(lane).and_then(|l| l.as_mut()) {
                                    Some(l) => l.tp_cancelled = true,
                                    None => anyhow::bail!(
                                        "TP cancel for empty lane {lane} at step {step} — state desync"),
                                }
                            }
                        }
                    }
                    let b = self.num_active();
                    if b > 0 { self.decode_step(b); self.tp_agree_step(step)?; }
                    step += 1;
                    tokio::task::yield_now().await;
                }
                other => anyhow::bail!("unexpected serving message on the step stream: {other:?}"),
            }
        }
    }

    fn num_active(&self) -> usize {
        self.lanes.iter().take_while(|l| l.is_some()).count()
    }

    /// Prefill `req` into a free physical slot; emits the first generated token to the client.
    ///
    /// PREFIX REUSE. OpenWebUI and opencode resend the ENTIRE conversation every turn, so without this a
    /// chat pays prefill of its whole history on every turn — per-turn TTFT grows linearly and a session
    /// costs O(T²) in total prefill. We pick the free slot whose cached sequence is the longest prefix of
    /// this prompt and prefill only the suffix.
    fn admit(&mut self, req: BatchRequest) {
        // Each free slot offers TWO points we could resume from, because we hold the GDN state at two
        // moments of its last request:
        //
        //   LIVE  — the slot's current state, at the end of the last GENERATION (`slot_cache`).
        //           Hits when the client replays our tokens verbatim (opencode does).
        //   CKPT  — the prompt checkpoint, at the end of the last PREFILL (`slot_ckpt_seq`).
        //           Hits when the client re-renders our turn instead (every tool-calling agent does),
        //           and when a new conversation reuses the same system/tool preamble.
        //
        // Either is usable only if its sequence is a STRICT prefix of this prompt: the recurrent state
        // sits immediately after its last token, so we need at least one token left to prefill and we
        // cannot resume from a point the state never occupied. Prefer the longer.
        #[derive(Clone, Copy, PartialEq)]
        enum From_ { Live, Ckpt }
        let best = if !self.prefix_cache { None } else { self.free_slots.iter().enumerate()
            .flat_map(|(i, &sl)| {
                let live = common_prefix_len(&self.slot_cache[sl], &req.prompt);
                let ckpt = common_prefix_len(&self.slot_ckpt_seq[sl], &req.prompt);
                [(i, sl, live, From_::Live, self.slot_cache[sl].len()),
                 (i, sl, ckpt, From_::Ckpt, self.slot_ckpt_seq[sl].len())]
            })
            .filter(|&(_, _, l, _, seq_len)| l > 0 && l == seq_len && l < req.prompt.len())
            .max_by_key(|&(_, _, l, _, _)| l) };

        let (phys, reuse, from) = match best {
            Some((idx, sl, l, f, _)) => { self.free_slots.remove(idx); (sl, l, Some(f)) }
            None => match self.free_slots.pop() {
                Some(s) => (s, 0, None),
                None => return, // no free physical slot (caller checks capacity)
            },
        };

        // Resuming from the checkpoint means winding the slot's GDN state BACK to the prompt boundary.
        // (KV is untouched: positions 0..reuse still hold this very prefix's keys, and the suffix
        // prefill overwrites everything after.)
        if from == Some(From_::Ckpt) {
            self.gpu.copy_gdn_slot(&self.state, self.prompt_ckpt_slot + phys, phys);
        }

        // On a miss, report the prefix we COULD have reused. This is the only way to see the size of
        // the opportunity we are leaving on the floor — and it is what exposed the 88% waste.
        if reuse == 0 {
            if let Some((best, cached)) = self.free_slots.iter().chain(std::iter::once(&phys))
                .flat_map(|&sl| [(common_prefix_len(&self.slot_cache[sl], &req.prompt), self.slot_cache[sl].len()),
                                 (common_prefix_len(&self.slot_ckpt_seq[sl], &req.prompt), self.slot_ckpt_seq[sl].len())])
                .max_by_key(|&(l, _)| l) {
                if best > 64 {
                    eprintln!("[req] prefix MISS: {} of {} prompt tokens match a cached sequence of {} \
                               — unusable, the GDN recurrent state only exists at token {}",
                              best, req.prompt.len(), cached, cached);
                }
            }
        }
        let greedy = req.temperature < 1e-6;
        let temperature = req.temperature;
        let top_p = req.top_p;
        let top_k = req.top_k;
        let tx = req.tx;
        let plen = req.prompt.len();
        let prompt = req.prompt;
        let max_new = req.max_new;
        let req_ckpt_at = req.ckpt_at;
        let rep_penalty = req.rep_penalty;
        let presence_penalty = req.presence_penalty;
        let frequency_penalty = req.frequency_penalty;
        let _has_penalty = rep_penalty > 1.0 || presence_penalty > 0.0 || frequency_penalty > 0.0;
        let will_use_mtp = self.mtp.active();

        let cfg = self.gpu.cfg().clone();
        let h = cfg.hidden_size;

        // The KV cache is exactly `kv_stride` positions deep. `write_kv_prefill` had no bound check, so
        // an over-long prompt wrote past the end of it and corrupted the neighbouring allocation. The
        // server rejects these before they get here; this is the backstop for every other caller (the
        // bench paths admit requests directly).
        if plen >= self.kv_stride {
            eprintln!("[req] REJECTED: prompt is {} tokens but the KV cache holds {} — raise --max-seq-len",
                      plen, self.kv_stride);
            // Return the physical slot — it was popped above and losing it here permanently shrinks
            // capacity (handoff 6.10). Two consistency rules: (1) the push-back is deterministic, so
            // both TP ranks make the same next pick; (2) if the CKPT restore already ran, the slot's
            // LIVE state was wound back to the prompt checkpoint — the live metadata must follow it,
            // or a later Live pick would resume from a state the slot no longer holds.
            if from == Some(From_::Ckpt) {
                self.slot_cache[phys] = self.slot_ckpt_seq[phys].clone();
            }
            self.free_slots.push(phys);
            let _ = tx.send(TokEvent::Finish { reason: "context_length_exceeded".to_string() });
            return;
        }
        let max_new = max_new.min(self.kv_stride - plen);

        // Bound the pool. Safe here: `trim` synchronizes before freeing, and this runs before any GPU
        // work for this request. Without it the pool grows forever — see Pool::trim.
        self.pool.trim();

        // On a cache MISS, wipe the slot. On a HIT we must NOT: the GDN recurrent state and the conv1d
        // tail are exactly what we are reusing, and they only exist at the point the last request left
        // them. (KV beyond `reuse` is stale but unreachable — attention never reads past `pos`.)
        if reuse == 0 {
            self.gpu.zero_slot_state(&mut self.state, phys, self.kv_stride);
        }
        let suffix = &prompt[reuse..];
        if reuse > 0 {
            eprintln!("[req] prefix hit ({}): {}/{} tokens cached, prefilling {} ({:.0}% skipped)",
                      if from == Some(From_::Ckpt) { "prompt checkpoint" } else { "live state" },
                      reuse, plen, suffix.len(), 100.0 * reuse as f32 / plen as f32);
        }

        // MTP prompt-prime: over main positions 0..plen-2, step t writes (h_t, embed(prompt[t+1]))
        // into the lane's MTP KV at position t. Then seed the cursor hidden h_prev = h at plen-1.
        if will_use_mtp && reuse == 0 {
            // Miss: a previously-finished lane may have left speculative KV here. alloc_zeros doesn't
            // zero, and stale KV → nondeterministic drafts. Compute-stream memset.
            let kv_bytes = cfg.num_kv_heads * self.kv_stride * cfg.head_dim * 2;
            self.gpu.memset_compute_stream(*self.mtp_kc[phys].device_ptr(), kv_bytes);
            self.gpu.memset_compute_stream(*self.mtp_vc[phys].device_ptr(), kv_bytes);
        }
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();

        // CHUNKED PREFILL. Prefill activation memory is O(prompt length) — ~1.2 MiB/token on 9B,
        // more on 27B — so a single-shot prefill of a long prompt OOMs (256K on 27B ≈ 400 GB). Process
        // the suffix in windows of PREFILL_CHUNK: each window's buffers are bounded by the chunk, and
        // the KV/GDN/conv state carries across windows via pos_start (the same mechanism prefix reuse
        // already uses). A prompt <= PREFILL_CHUNK is ONE window == the old single-shot path, so short
        // prompts are byte-identical to before; only long prompts (which would otherwise OOM) get
        // re-chunked, which perturbs their prefill hiddens by ulps — outside the batch-invariance
        // contract (prefill feeds decode and verify identically), so greedy MTP stays lossless.
        //
        // The message-boundary checkpoint (prefix cache) is honoured by forcing a window to END exactly
        // at `c`, then snapshotting the GDN state there before the next window moves it.
        let ckpt_at = req_ckpt_at.filter(|_| self.prefix_cache).filter(|&c| c > reuse && c < plen);
        let mut first_tok = 0u32;
        let mut w0 = reuse;
        while w0 < plen {
            let mut w1 = (w0 + PREFILL_CHUNK).min(plen);
            if let Some(c) = ckpt_at { if w0 < c && c < w1 { w1 = c; } }   // stop at the boundary

            let (tok, hw) = self.gpu.prefill_batch(
                &mut self.pool, &prompt[w0..w1], &mut self.state, phys, self.kv_stride, w0);
            first_tok = tok;   // only the LAST window's token (at plen-1) is the prompt's next token

            if will_use_mtp {
                // MTP prime pairs hidden[t] with token[t+1] for t in [w0, min(w1, plen-1)); position
                // plen-1 is never primed (no token plen to pair). hw's columns 0.. map to positions w0..
                let tok_end = w1.min(plen - 1);
                if tok_end > w0 {
                    self.gpu.mtp_prime_prompt(&mut self.pool, &hw, &prompt[w0 + 1..tok_end + 1],
                                              mtp_kc_ptr, mtp_vc_ptr, self.kv_stride, w0);
                }
                // Cursor hidden = pre-norm h at the LAST prompt position, i.e. last column of the
                // final window.
                if w1 == plen {
                    self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &hw, (w1 - w0) - 1);
                }
            }
            self.pool.release_bf16(hw, h * (w1 - w0));

            // Snapshot the GDN state at the message boundary (prefix cache), before the next window.
            if Some(w1) == ckpt_at {
                self.gpu.copy_gdn_slot(&self.state, phys, self.prompt_ckpt_slot + phys);
                self.slot_ckpt_seq[phys] = prompt[..w1].to_vec();
            }
            w0 = w1;
        }

        if self.prefix_cache && ckpt_at.is_none() {
            // No boundary inside the suffix (a raw/non-chat request, or one whose whole prompt was
            // already cached). Snapshot where we ended; it is still the prompt boundary for THIS prompt.
            self.gpu.copy_gdn_slot(&self.state, phys, self.prompt_ckpt_slot + phys);
            self.slot_ckpt_seq[phys] = prompt.clone();
        }

        // The slot's state now reflects the whole prompt. Decode extends this as tokens commit.
        self.slot_cache[phys] = prompt.clone();

        let slot = self.num_active();
        let _ = tx.send(TokEvent::Tok(first_tok));
        // Seed: use the explicit seed from the request, or derive from prompt hash + counter.
        let seed = req.seed.unwrap_or_else(|| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            prompt.hash(&mut h);
            h.finish()
        });

        self.lanes[slot] = Some(Lane {
            phys, pos: plen, last_tok: first_tok, max_new,
            generated: 1, greedy, temperature, top_p, top_k,
            rep_penalty, presence_penalty, frequency_penalty,
            history: vec![first_tok], tx,
            mtp_pos: plen.saturating_sub(1),
            mtp_primed: will_use_mtp,
            mtp_stale: false,
            seed,
            tp_cancelled: false,
        });
    }

    /// One scheduler decode step over the `b` active (front-packed) lanes. Two phases:
    ///   A. MTP-eligible lanes (greedy → mtp_lane_step; sampling → mtp_lane_step_sample when
    ///      stochastic MTP is on) are served one at a time — each emits 1+ tokens.
    ///   B. All remaining lanes are served by a single batched decode (one shared weight read),
    ///      each emitting exactly one token.
    fn decode_step(&mut self, b: usize) {
        let mut finished = vec![false; b];
        // Cancel lanes whose client is gone (disconnect, or the SSE generator dropped the receiver
        // on a stop-string hit): every send would silently fail and the lane would otherwise decode
        // to EOS/max_new, holding a batch slot and a share of every batched step. The compaction at
        // the end of the step does the teardown (frees the slot, attempts the Finish event).
        for i in 0..b {
            // TP serving takes its cancels from the wire (`tp_cancelled`), never from `is_closed()`
            // directly — a disconnect landing between the head's per-step detection and this sweep
            // would otherwise finish the lane a step early on the head only. Single-node is exactly
            // as before (`is_closed()` alone); the mirror's dummy txs never close on their own.
            if self.lanes[i].as_ref()
                .map_or(false, |l| l.tp_cancelled || (!self.tp_serving && l.tx.is_closed())) {
                finished[i] = true;
            }
        }

        // Phase A: per-lane MTP. Greedy lanes verify by argmax (bitwise lossless); sampling lanes
        // verify by speculative rejection sampling (distribution-exact). The split is decided by the
        // request's temperature, not by any server flag.
        //
        // MTP IS A SINGLE-LANE PATH -- this loop runs one lane at a time, each doing its own draft and
        // verify forwards. With two clients that is two full model passes per step, so concurrent
        // throughput was FLAT: measured 37.7 tok/s alone, 21.4 each with two clients, 10.7 each with
        // four, aggregate pinned at ~43 the whole way. Each client simply got 1/N of the machine.
        //
        // But BATCHING IS FREE HERE, and for the same reason speculation is: the quantized GEMM is a
        // fixed-shape kernel with N padded to 16, so a decode step is bound by the weight bytes, which
        // do not change with the batch. A 4-lane batched step (Phase B) costs what a 1-lane step costs.
        //
        // So above one lane, batching beats speculation and it is not close: 4 lanes at ~1 forward
        // total, versus 4 lanes at ~4 speculative forwards for ~2.5 tokens each. Speculation only wins
        // when there is nothing to batch WITH. Hence: MTP at b == 1, batched decode at b >= 2.
        //
        // (The real prize is batching the MTP VERIFY across lanes -- attn_dispatch already takes
        // per-column slot_ids and pos, so the attention supports mixed slots; the GDN recurrent state
        // and the per-lane accept/rollback bookkeeping are what make it a project rather than a patch.
        // That would give both: N lanes AND ~2.5 tokens per lane per step.)
        let policy_active = self.mtp.active();
        // Phase-B (batched-decode) eligibility: without lanes, MTP runs only at b==1 (one speculative
        // forward beats nothing to batch with); WITH lanes, MTP serves primed lanes at any b, so only
        // non-primed/stale lanes fall to Phase B.
        let phaseb_active = if self.mtp_lanes { policy_active } else { policy_active && b == 1 };
        if policy_active {
            if self.mtp_lanes {
                // FOREST: pack the greedy, primed, non-stale lanes (penalty carried per-column) into ONE
                // verify. Overflow beyond the column budget and any single leftover run the single-lane
                // MTP path, so NO eligible lane sits out (which strands it on plain decode via mtp_stale).
                let forest: Vec<usize> = (0..b).filter(|&i| {
                    let l = self.lanes[i].as_ref().unwrap();
                    l.greedy && l.use_mtp(true)
                }).collect();
                let take = forest.len().min(5);   // keeps per-lane depth >= 2 under the 16-column budget
                if take >= 2 {
                    let packed: Vec<usize> = forest[..take].to_vec();
                    for (i, fin) in self.mtp_forest_step(&packed) { if fin { finished[i] = true; } }
                    for &i in &forest[take..] { if self.mtp_lane_step(i) { finished[i] = true; } }
                } else {
                    for &i in &forest { if self.mtp_lane_step(i) { finished[i] = true; } }
                }
                // Sampling (non-greedy) primed lanes keep the single-lane stochastic path (v1).
                for i in 0..b {
                    let l = self.lanes[i].as_ref().unwrap();
                    if l.use_mtp(true) && !l.greedy {
                        if self.mtp_lane_step_sample(i) { finished[i] = true; }
                    }
                }
            } else if b == 1 {
                let lane = self.lanes[0].as_ref().unwrap();
                let is_greedy = lane.greedy;
                if lane.use_mtp(true) {
                    let done = if is_greedy { self.mtp_lane_step(0) } else { self.mtp_lane_step_sample(0) };
                    if done { finished[0] = true; }
                }
            }
        }
        self.mtp.tick();

        // Phase B: batched decode for the remaining lanes.
        let batch_idx: Vec<usize> = (0..b)
            .filter(|&i| !self.lanes[i].as_ref().unwrap().use_mtp(phaseb_active))
            .collect();
        if !batch_idx.is_empty() {
            let next_toks = self.batched_decode(&batch_idx);
            for (k, &i) in batch_idx.iter().enumerate() {
                let t = next_toks[k];
                let lane = self.lanes[i].as_mut().unwrap();
                // This lane just advanced WITHOUT writing MTP KV: the head now has a hole at this
                // position and can never be trusted again for this request. See Lane::mtp_stale.
                lane.mtp_stale = true;
                let _ = lane.tx.send(TokEvent::Tok(t));
                // THE CACHE RECORDS WHAT THE STATE HAS CONSUMED, NOT WHAT WE EMITTED. This step fed the
                // PREVIOUS token (`last_tok`) through the model at position `pos`; `t` is the model's
                // prediction and has not been fed yet. Caching `t` instead would put a token in the
                // cache that the state has never seen — and the next turn would reuse that state as if
                // it had, silently producing wrong output. Invariant: slot_cache.len() == lane.pos.
                let fed = lane.last_tok;
                let phys = lane.phys;
                lane.last_tok = t;
                lane.pos += 1;
                lane.generated += 1;
                lane.history.push(t);
                self.slot_cache[phys].push(fed);
                debug_assert_eq!(self.slot_cache[phys].len(), lane.pos);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
                if self.eos.contains(&t) || lane.generated >= lane.max_new {
                    finished[i] = true;
                }
            }
        }

        // compact: keep non-finished lanes front-packed. Each lane keeps its physical slot —
        // finished lanes return their slot to the free list; no state copying needed.
        let mut write = 0usize;
        for i in 0..b {
            if finished[i] {
                let lane = self.lanes[i].take().unwrap();
                self.free_slots.push(lane.phys);
                let reason = if lane.generated >= lane.max_new { "length" } else { "stop" };
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.to_string() });
            } else {
                self.lanes[write] = self.lanes[i].take();
                write += 1;
            }
        }
    }

    /// Build the per-position verify penalty from a lane's committed history (dedup, replicate to all
    /// MAX_AUTO_DEPTH positions). Shared by the chain and tree MTP paths. Returns None if no penalty.
    fn make_penalty(&mut self, history: &[u32], rep_pen: f32, presence_pen: f32, freq_pen: f32,
                    has_penalty: bool) -> Option<crate::gpu::VerifyPenalty> {
        if !has_penalty { return None; }
        let mp = crate::gpu::MAX_PEN_TOKENS;
        let cap = crate::gpu::MAX_VERIFY;   // buffers are MAX_VERIFY-sized (forest may span that many cols)
        let mut pen_tokens = vec![-1i32; cap * mp];
        let mut pen_counts = vec![0i16; cap * mp];
        let mut idx = 0usize;
        for &t in history.iter().rev().take(mp) {
            let ti = t as i32;
            match (0..idx).position(|j| pen_tokens[j] == ti) {
                Some(j) => { pen_counts[j] += 1; }
                None => { if idx < mp { pen_tokens[idx] = ti; pen_counts[idx] = 1; idx += 1; } }
            }
        }
        let head_t: Vec<i32> = pen_tokens[0..mp].to_vec();
        let head_c: Vec<i16> = pen_counts[0..mp].to_vec();
        for p in 1..cap {
            pen_tokens[p*mp..p*mp+mp].copy_from_slice(&head_t);
            pen_counts[p*mp..p*mp+mp].copy_from_slice(&head_c);
        }
        let (rv, pv, fv) = (vec![rep_pen; cap], vec![presence_pen; cap], vec![freq_pen; cap]);
        self.gpu.dev().htod_sync_copy_into(&pen_tokens, self.mtp_pen_tokens.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&pen_counts, self.mtp_pen_counts.as_mut().unwrap()).unwrap();
        // The VALUES are request-constant: upload them only when they (or the width) change, not
        // every MTP step. The history token/count arrays above still go up per step.
        let key = (rep_pen.to_bits(), presence_pen.to_bits(), freq_pen.to_bits(), cap);
        if self.pen_const_key != Some(key) {
            self.gpu.dev().htod_sync_copy_into(&rv, self.mtp_pen_rep.as_mut().unwrap()).unwrap();
            self.gpu.dev().htod_sync_copy_into(&pv, self.mtp_pen_presence.as_mut().unwrap()).unwrap();
            self.gpu.dev().htod_sync_copy_into(&fv, self.mtp_pen_freq.as_mut().unwrap()).unwrap();
            self.pen_const_key = Some(key);
        }
        // NOTE: no dev().synchronize() here — the copies are host-blocking on the NULL stream and
        // the compute stream is the blocking stream (invariant I1), so ordering is already
        // guaranteed; the sync was a full pipeline drain of pure waste per penalized MTP step.
        Some(crate::gpu::VerifyPenalty {
            tokens_ptr: *self.mtp_pen_tokens.as_ref().unwrap().device_ptr(),
            counts_ptr: *self.mtp_pen_counts.as_ref().unwrap().device_ptr(),
            rep_pen_ptr: *self.mtp_pen_rep.as_ref().unwrap().device_ptr(),
            presence_ptr: *self.mtp_pen_presence.as_ref().unwrap().device_ptr(),
            freq_ptr: *self.mtp_pen_freq.as_ref().unwrap().device_ptr(),
        })
    }

    /// FOREST per-column penalty: column c gets ITS lane's rep/presence/freq penalty (from that lane's
    /// deduped committed history). `lanes` is (start, len, rep, presence, freq, history) per packed lane;
    /// columns outside any lane and past the packed width are no-ops (rep=1, presence=freq=0, tokens=-1).
    /// Returns None if NO packed lane has a penalty. Buffers are MAX_VERIFY-sized.
    fn make_forest_penalty(&mut self, lanes: &[(usize, usize, f32, f32, f32, Vec<u32>)])
                           -> Option<crate::gpu::VerifyPenalty> {
        let any = lanes.iter().any(|(_, _, r, pr, f, _)| *r > 1.0 || *pr > 0.0 || *f > 0.0);
        if !any { return None; }
        let mp = crate::gpu::MAX_PEN_TOKENS;
        let cap = crate::gpu::MAX_VERIFY;
        let mut pen_tokens = vec![-1i32; cap * mp];
        let mut pen_counts = vec![0i16; cap * mp];
        let mut rep_v = vec![1.0f32; cap];
        let mut pres_v = vec![0.0f32; cap];
        let mut freq_v = vec![0.0f32; cap];
        for (start, len, rep, pres, freq, history) in lanes {
            // Dedup this lane's recent history into one [mp] block.
            let mut ht = vec![-1i32; mp];
            let mut hc = vec![0i16; mp];
            let mut idx = 0usize;
            for &t in history.iter().rev().take(mp) {
                let ti = t as i32;
                match (0..idx).position(|j| ht[j] == ti) {
                    Some(j) => { hc[j] += 1; }
                    None => { if idx < mp { ht[idx] = ti; hc[idx] = 1; idx += 1; } }
                }
            }
            for c in *start..(*start + *len).min(cap) {
                pen_tokens[c * mp..c * mp + mp].copy_from_slice(&ht);
                pen_counts[c * mp..c * mp + mp].copy_from_slice(&hc);
                rep_v[c] = *rep; pres_v[c] = *pres; freq_v[c] = *freq;
            }
        }
        self.gpu.dev().htod_sync_copy_into(&pen_tokens, self.mtp_pen_tokens.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&pen_counts, self.mtp_pen_counts.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&rep_v, self.mtp_pen_rep.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&pres_v, self.mtp_pen_presence.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&freq_v, self.mtp_pen_freq.as_mut().unwrap()).unwrap();
        // The forest just overwrote the per-column penalty VALUES with per-lane ones — invalidate
        // the const cache so the next chain make_penalty re-uploads instead of trusting stale data.
        self.pen_const_key = None;
        // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute stream
        // already order these before the verify kernels that read them (invariant I1).
        Some(crate::gpu::VerifyPenalty {
            tokens_ptr: *self.mtp_pen_tokens.as_ref().unwrap().device_ptr(),
            counts_ptr: *self.mtp_pen_counts.as_ref().unwrap().device_ptr(),
            rep_pen_ptr: *self.mtp_pen_rep.as_ref().unwrap().device_ptr(),
            presence_ptr: *self.mtp_pen_presence.as_ref().unwrap().device_ptr(),
            freq_ptr: *self.mtp_pen_freq.as_ref().unwrap().device_ptr(),
        })
    }

    /// One FORK-THEN-CHAIN tree MTP step (greedy). Drafts a k=2 fork, verifies the tree, walks the
    /// accepted path (target argmax), compacts its KV to contiguous slots, adopts the accepted leaf's
    /// GDN checkpoint, re-primes MTP over the accepted path, and emits. Lossless: every emitted token is
    /// the target's argmax given its accepted prefix (same as the chain). Returns true if finished.
    fn mtp_tree_step(&mut self, i: usize) -> bool {
        let h = self.gpu.cfg().hidden_size;
        let depth = self.mtp.depth();
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let ckpt = self.mtp_snapshot_slot;   // tree checkpoints base (slots ckpt..ckpt+n-2)
        let kv_stride = self.kv_stride;
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
        let h_save_ptr = *self.mtp_h_save.as_ref().unwrap().device_ptr();

        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let mtp_pos = self.lanes[i].as_ref().unwrap().mtp_pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();

        // h_save = pre-verify hidden (for re-prime column 0).
        self.gpu.copy_hidden_col(h_save_ptr, &self.mtp_h_prev[phys], 0);

        // n-gram context (the second branch's source, if enabled).
        let ngram = self.ngram_draft;
        let mut work: Vec<u32> = Vec::new();
        if ngram > 0 {
            work.extend_from_slice(&self.slot_cache[phys]);
            work.push(committed_tok);
        }

        // ---- Draft the fork-then-chain tree. ----
        let (parent, tokens) = self.gpu.mtp_fork_draft(
            &mut self.pool, &self.mtp_h_prev[phys], committed_tok as i32, mtp_pos, depth,
            mtp_kc_ptr, mtp_vc_ptr, kv_stride, &work, ngram);
        let n = tokens.len();

        // ---- Verify the tree. ----
        let penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);
        let topo = self.gpu.topo_from_parent(&parent, main_pos);
        let (preds, vout) = self.gpu.verify_forward_topo(
            &mut self.pool, &tokens, &mut self.state, phys, kv_stride, main_pos, Some(ckpt), penalty, Some(&topo));

        // ---- Accept walk: follow the target argmax down the tree. ----
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for c in 1..n { children[parent[c] as usize].push(c); }
        let mut path = vec![0usize]; let mut emitted = Vec::new(); let mut cur = 0usize;
        loop {
            let want = preds[cur]; emitted.push(want);
            match children[cur].iter().copied().find(|&c| tokens[c] == want) {
                Some(c) => { path.push(c); cur = c; } None => break,
            }
        }
        let nacc = path.len() - 1;               // accepted drafts (emitted[0..nacc]); emitted[nacc]=bonus
        let leaf = *path.last().unwrap();

        // ---- Commit: compact the accepted path's KV, adopt the leaf's GDN state, re-prime MTP. ----
        let src_pos: Vec<i32> = path.iter().map(|&p| p as i32).collect();
        self.gpu.compact_kv(&mut self.pool, &self.state, phys, main_pos, &src_pos, kv_stride);
        // Adopt the accepted leaf's GDN checkpoint. The DFS scan ends at column n-1, so if leaf==n-1 the
        // slot already holds the right state; else restore from its checkpoint slot (ckpt+leaf).
        if leaf != n - 1 { self.gpu.copy_gdn_slot(&self.state, ckpt + leaf, phys); }
        // h_prev = hidden at the accepted leaf.
        self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, leaf);
        // Re-prime MTP over the accepted path: column 0 uses h_save + committed; column k uses
        // vout[path[k-1]] + tokens[path[k]].
        let n_rp = nacc + 1;
        let rp_hidden = self.pool.get_bf16(h * n_rp);
        let rp_ptr = *rp_hidden.device_ptr();
        let mut rp_toks: Vec<u32> = vec![committed_tok];
        self.gpu.copy_hidden_col(rp_ptr, self.mtp_h_save.as_ref().unwrap(), 0);
        for k in 1..=nacc {
            self.gpu.copy_hidden_col(rp_ptr + (k * h * 2) as u64, &vout, path[k - 1]);
            rp_toks.push(tokens[path[k]]);
        }
        self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        self.pool.release_bf16(rp_hidden, h * n_rp);
        self.pool.release_bf16(vout, h * n);

        // ---- Emit (same EOS/max_new discipline as the chain). ----
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for k in 0..nacc {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(emitted[k]);
            if eos.contains(&emitted[k]) { hit_eos = true; break; }
        }
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(emitted[nacc]);   // bonus
            if eos.contains(&emitted[nacc]) { hit_eos = true; }
        }
        let emit_count = new_toks.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        // Cache what was FED: committed + accepted node tokens (emitted[0..nacc]).
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&emitted[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &new_toks {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            if !finished {
                lane.last_tok = emitted[nacc];
                lane.pos = main_pos + nacc + 1;
                lane.mtp_pos = main_pos + nacc;
            } else {
                lane.last_tok = *new_toks.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }
        self.mtp_stat_steps += 1;
        self.mtp_stat_drafts += (n - 1) as u64;
        self.mtp_stat_accepted += nacc as u64;
        self.mtp_stat_emitted += emit_count as u64;
        self.mtp_stat_verify_fwds += 1;
        self.mtp.record_step((n - 1) as u64, nacc as u64, emit_count as u64);
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 { self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0 } else { 0.0 };
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            eprintln!("[mtp/tree] steps={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (depth {} n {}) accept@k [{}]",
                      self.mtp_stat_steps, acc, self.mtp_stat_emitted, eff, depth, n, fmt_accept_by_depth(&self.mtp));
            self.dump_accept_curve();
        }
        finished
    }

    /// One MTP speculative-decoding step for lane `i` (greedy, penalty-free). Mirrors the validated
    /// `bench_mtp` loop body: draft (depth-1) → snapshot GDN → verify (depth) → accept longest
    /// prefix → rollback+reverify on partial reject → re-prime MTP over the accepted prefix with REAL
    /// verify hiddens. Emits the accepted drafts + bonus token, advancing the lane by nacc+1
    /// positions. Returns true if the lane finished (EOS or max_new reached).
    /// FOREST MTP step (LANES design Step 3c): draft each of `lanes`' chains, pack them into ONE forest
    /// verify (proven lossless by `--bench-lanes`), then per-lane accept / rollback / re-prime / emit.
    /// This is `mtp_lane_step` generalized to L lanes sharing one main-model forward — the concurrency
    /// throughput win. Greedy, no-penalty lanes only (v1). Returns (lane_index, finished) per lane.
    fn mtp_forest_step(&mut self, lanes: &[usize]) -> Vec<(usize, bool)> {
        let h = self.gpu.cfg().hidden_size;
        let ck = self.gpu.cfg().conv_kernel;
        let mv = crate::gpu::MAX_VERIFY;
        let kv_stride = self.kv_stride;
        let snapshot = self.mtp_snapshot_slot;
        let p = lanes.len();
        // v1 allocator: uniform per-lane depth so Σ(1+d) = p*(1+d) ≤ MAX_VERIFY, capped by the policy depth.
        let depth = ((mv / p).saturating_sub(1)).clamp(1, self.mtp.depth());

        // Per-lane locals + draft chains (drafting is sequential; the shared MTP scratch is reused, and
        // each lane's head KV is its own mtp_kc/mtp_vc slot). Each lane's chain = [committed, drafts...].
        struct L { i: usize, phys: usize, committed: u32, main_pos: usize, mtp_pos: usize,
                   generated: usize, max_new: usize, drafts: Vec<u32>, start: usize, n: usize,
                   rep: f32, pres: f32, freq: f32, history: Vec<u32> }
        let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();
        let mut ls: Vec<L> = Vec::with_capacity(p);
        let mut global = 0usize;
        for &i in lanes {
            let (phys, committed, main_pos, mtp_pos, generated, max_new, rep, pres, freq, history) = {
                let l = self.lanes[i].as_ref().unwrap();
                (l.phys, l.last_tok, l.pos, l.mtp_pos, l.generated, l.max_new,
                 l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.history.clone())
            };
            let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
            let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
            self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
            let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
            let mut cur_tok = committed as i32;
            let mut dpos = mtp_pos;
            for _ in 0..depth - 1 {
                let m = self.gpu.mtp_draft_step(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(),
                                                cur_tok, dpos, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                self.gpu.copy_hidden_col(cur_ptr, &m, 0);
                self.pool.release_bf16(m, h);
                cur_tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap()) as i32;
                drafts.push(cur_tok as u32);
                dpos += 1;
            }
            let n = 1 + drafts.len();
            ls.push(L { i, phys, committed, main_pos, mtp_pos, generated, max_new, drafts, start: global, n,
                        rep, pres, freq, history });
            global += n;
        }
        let ntot = global;

        // Build the FOREST topo + packed token stream (see bench_lanes / verify_forward_core_topo).
        let mut tokens: Vec<u32> = Vec::with_capacity(ntot);
        let mut parent = vec![-1i32; ntot];
        let mut slotv = vec![0i32; ntot];
        let mut rope = vec![0i32; ntot];
        let mut kv_pos = vec![0i32; ntot];
        let mut cps = vec![0i32; ntot];
        let mut path = vec![0u8; ntot * mv];
        let mut winsrc = vec![0i32; ntot * ck];
        for l in &ls {
            tokens.push(l.committed);
            tokens.extend_from_slice(&l.drafts);
            for r in 0..l.n {
                let c = l.start + r;
                parent[c] = if r == 0 { -1 } else { (c - 1) as i32 };
                slotv[c] = l.phys as i32;
                rope[c] = (l.main_pos + r) as i32;
                kv_pos[c] = (l.main_pos + r) as i32;
                cps[c] = l.main_pos as i32;
                for dd in 0..=r { path[c * mv + dd] = dd as u8; }
                for j in 0..ck {
                    let wd = r as i32 - (ck as i32 - 1) + j as i32;
                    winsrc[c * ck + j] = if wd < 0 { wd } else { l.start as i32 + wd };
                }
            }
        }
        let topo = crate::gpu::TreeTopo { rope, kv_pos, parent, path, winsrc,
                                          slot: Some(slotv), col_pos_start: Some(cps) };
        let pos_start_max = ls.iter().map(|l| l.main_pos).max().unwrap();

        // Per-lane verify penalty: each column carries ITS lane's rep/presence/freq penalty (buffers are
        // MAX_VERIFY-sized). None if no packed lane has a penalty.
        let ls_pen: Vec<(usize, usize, f32, f32, f32, Vec<u32>)> =
            ls.iter().map(|l| (l.start, l.n, l.rep, l.pres, l.freq, l.history.clone())).collect();
        let penalty = self.make_forest_penalty(&ls_pen);

        // ONE packed forest verify. ckpt writes column t's post-state to (snapshot + t); rollback reads
        // each lane's own accepted checkpoint.
        let (preds, vout) = self.gpu.verify_forward_topo(
            &mut self.pool, &tokens, &mut self.state, 0, kv_stride, pos_start_max, Some(snapshot), penalty, Some(&topo));

        let eos = self.eos.clone();
        let mut results: Vec<(usize, bool)> = Vec::with_capacity(p);
        for l in &ls {
            let draft_count = l.drafts.len();
            let mut nacc = 0usize;
            while nacc < draft_count && preds[l.start + nacc] == l.drafts[nacc] { nacc += 1; }
            let bonus = preds[l.start + nacc];
            // Rollback on partial reject: restore this lane's state as of its last accepted column.
            if nacc + 1 != l.n {
                self.gpu.copy_gdn_slot(&self.state, snapshot + l.start + nacc, l.phys);
            }
            // Re-prime this lane's MTP head over its accepted prefix (k=0 = pre-step h_prev, still intact;
            // k>=1 = the verify's real hidden vout[start+k-1]). Then advance h_prev to vout[start+nacc].
            let mtp_kc_ptr = *self.mtp_kc[l.phys].device_ptr();
            let mtp_vc_ptr = *self.mtp_vc[l.phys].device_ptr();
            let n_rp = nacc + 1;
            let rp_hidden = self.pool.get_bf16(h * n_rp);
            let rp_ptr = *rp_hidden.device_ptr();
            let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
            self.gpu.copy_hidden_col(rp_ptr, &self.mtp_h_prev[l.phys], 0);
            rp_toks.push(l.committed);
            // vout columns l.start..l.start+nacc-1 are CONTIGUOUS — one dtod for the whole prefix
            // (was nacc separate copy_hidden_col driver calls).
            self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, l.start, nacc);
            for k in 1..=nacc {
                rp_toks.push(l.drafts[k - 1]);
            }
            self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, l.main_pos - 1,
                                 mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            self.pool.release_bf16(rp_hidden, h * n_rp);
            self.gpu.copy_hidden_col(*self.mtp_h_prev[l.phys].device_ptr(), &vout, l.start + nacc);

            // Emit accepted drafts + bonus (budget on drafts too — see mtp_lane_step).
            let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
            let mut hit_eos = false;
            for &d in l.drafts.iter().take(nacc) {
                if l.generated + new_toks.len() >= l.max_new { break; }
                new_toks.push(d);
                if eos.contains(&d) { hit_eos = true; break; }
            }
            if !hit_eos && l.generated + new_toks.len() < l.max_new {
                new_toks.push(bonus);
                if eos.contains(&bonus) { hit_eos = true; }
            }
            let emit_count = new_toks.len();
            let finished = hit_eos || l.generated + emit_count >= l.max_new;
            {
                let cache = &mut self.slot_cache[l.phys];
                cache.push(l.committed);
                cache.extend_from_slice(&l.drafts[..nacc]);
            }
            {
                let lane = self.lanes[l.i].as_mut().unwrap();
                for &t in &new_toks {
                    let _ = lane.tx.send(TokEvent::Tok(t));
                    lane.history.push(t);
                    if lane.history.len() > 256 { lane.history.drain(0..128); }
                }
                lane.generated += emit_count;
                if !finished {
                    lane.last_tok = bonus;
                    lane.pos = l.main_pos + nacc + 1;
                    lane.mtp_pos = l.main_pos + nacc;
                } else {
                    lane.last_tok = *new_toks.last().unwrap_or(&l.committed);
                    lane.pos = l.main_pos + emit_count;
                }
            }
            self.mtp_stat_drafts += draft_count as u64;
            self.mtp_stat_accepted += nacc as u64;
            self.mtp_stat_emitted += emit_count as u64;
            results.push((l.i, finished));
        }
        self.pool.release_bf16(vout, h * ntot);
        self.mtp_stat_steps += p as u64;     // p lane-steps served by...
        self.mtp_stat_verify_fwds += 1;      // ...ONE main-model forward — the batching win
        results
    }

    /// Append one MTP step to the env-gated draft log (`MTP_DRAFT_LOG`), if open. JSONL, one object
    /// per line. `preds` is the verify's per-column argmax (greedy paths); pass `&[]` where the verify
    /// samples instead. Costs nothing when the log is closed. This is the reference the head-finetune
    /// B0 parity gate diffs the HF MTP module against, so it records exactly what was fed and predicted.
    fn log_draft_step(&mut self, lane: usize, pos: usize, committed: u32, drafts: &[u32], preds: &[u32], nacc: usize) {
        use std::io::Write;
        let Some(f) = self.mtp_draft_log.as_mut() else { return; };
        let arr = |xs: &[u32]| -> String {
            let mut s = String::from("[");
            for (k, x) in xs.iter().enumerate() { if k > 0 { s.push(','); } s.push_str(&x.to_string()); }
            s.push(']'); s
        };
        let _ = writeln!(f,
            "{{\"step\":{},\"lane\":{},\"pos\":{},\"committed\":{},\"drafts\":{},\"preds\":{},\"nacc\":{}}}",
            self.mtp_stat_steps, lane, pos, committed, arr(drafts), arr(preds), nacc);
    }

    /// Overwrite the accept-by-depth curve file (`MTP_CURVE_FILE`), if set, with the cumulative
    /// per-position conditional acceptance — the runbook §0 baseline curve. Called on the periodic
    /// stats boundary so a running server keeps a fresh snapshot on disk.
    fn dump_accept_curve(&self) {
        let Some(path) = self.mtp_curve_path.as_ref() else { return; };
        let hz = self.mtp.hazard_counts();
        let mut s = String::from("{\"depth\":");
        s.push_str(&self.mtp.depth().to_string());
        s.push_str(",\"steps\":");
        s.push_str(&self.mtp_stat_steps.to_string());
        s.push_str(",\"accept_by_depth\":[");
        // accept_by_depth[i] = {pos:i+1, accepted, offered, rate, cond_chain}. cond_chain = product of
        // rates up to and including pos i+1 = P(a draft chain reaches depth i+1) — the yield driver.
        let mut chain = 1.0f64;
        for (i, &(a, n)) in hz.iter().enumerate() {
            let rate = if n > 0 { a as f64 / n as f64 } else { 0.0 };
            chain *= rate;
            if i > 0 { s.push(','); }
            s.push_str(&format!(
                "{{\"pos\":{},\"accepted\":{},\"offered\":{},\"rate\":{:.4},\"cond_chain\":{:.4}}}",
                i + 1, a, n, rate, chain));
        }
        s.push_str("]}");
        let _ = std::fs::write(path, s);
    }

    fn mtp_lane_step(&mut self, i: usize) -> bool {
        // Fork-then-chain tree path (opt-in): rescues the chain-killing first-token miss.
        if self.tree_draft && self.mtp.depth() >= 3 {
            return self.mtp_tree_step(i);
        }
        let h = self.gpu.cfg().hidden_size;
        let depth = self.mtp.depth();
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
        let h_save_ptr = *self.mtp_h_save.as_ref().unwrap().device_ptr();
        let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();

        // Snapshot lane state into locals (avoids holding &mut self.lanes across GPU calls).
        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let mtp_pos = self.lanes[i].as_ref().unwrap().mtp_pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Penalty config + history (so the MTP verify keeps the lane's rep/presence/freq penalty).
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();

        // h_save = h_prev (hidden at main_pos-1); saved for the post-accept re-prime step k=0.
        self.gpu.copy_hidden_col(h_save_ptr, &self.mtp_h_prev[phys], 0);

        // Prompt-lookup context for n-gram drafting: the full realized sequence for this lane, which
        // slot_cache holds (prompt + every committed token), plus committed_tok (this step's first
        // token to verify, which slot_cache does not yet contain). Cheap host-side clone of u32s,
        // trivial next to a GPU forward. `ngram == 0` disables it.
        let ngram = self.ngram_draft;
        let mut work: Vec<u32> = Vec::new();
        if ngram > 0 {
            work.reserve(self.slot_cache[phys].len() + depth);
            work.extend_from_slice(&self.slot_cache[phys]);
            work.push(committed_tok);
        }

        // ---- Draft chain (depth-1 drafts). cur_hidden starts at h_prev; chains via MTP outputs. ----
        self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
        let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
        let mut cur_tok = committed_tok as i32;
        let mut dpos = mtp_pos;
        for _ in 0..depth - 1 {
            let m = self.gpu.mtp_draft_step(
                &mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), cur_tok, dpos,
                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            self.gpu.copy_hidden_col(cur_ptr, &m, 0);
            self.pool.release_bf16(m, h);
            cur_tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap()) as i32;
            // PROMPT-LOOKUP OVERRIDE (see mtp_lane_step's twin logic in gpu.rs::bench_accept). If the
            // last `ngram` tokens recur earlier in this lane's context, propose the token that followed
            // the most recent earlier occurrence — a free, exact copy that the 1-layer head cannot do.
            // Lossless by construction: the verify checks every draft; a wrong override is just rejected.
            if ngram > 0 && work.len() >= ngram {
                let tail_start = work.len() - ngram;
                for j in (0..tail_start).rev() {
                    if work[j..j + ngram] == work[tail_start..] {
                        if j + ngram < work.len() { cur_tok = work[j + ngram] as i32; }
                        break;
                    }
                }
                work.push(cur_tok as u32);
            }
            drafts.push(cur_tok as u32);
            dpos += 1;
        }

        // ---- Verify [committed_tok, drafts...] on the main model at positions main_pos.. ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        // Build the penalty for the verify: all `depth` positions share the lane's committed-history
        // penalty (the same one the normal decode path applies). This keeps greedy MTP lanes free of
        // repetition without slowing them. htod on the NULL stream, then sync before the compute-side
        // verify reads it.
        let penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);
        // Ping-pong GDN: the verify snapshots S1 (post committed-token state) into the snapshot slot
        // via the kernel checkpoint, so a rejected draft restores S1 with a dtod copy — no reverify.
        let (preds, vout) = self.gpu.verify_forward(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos, Some(snapshot), penalty);

        // ---- Accept longest prefix (greedy: drafts[i] accepted iff preds[i]==drafts[i]). ----
        let mut nacc = 0usize;
        while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
        let bonus = preds[nacc];

        // ---- GDN rollback on partial reject: restore S1 (the checkpoint — no second forward). ----
        // vout column nacc is the hidden at the last accepted position, valid in both cases.
        if nacc + 1 != depth {
            // Restore the state as of the LAST ACCEPTED column, not column 0. Checkpoint slots are
            // contiguous: slot (snapshot + t) holds the post-state of verify column t.
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }

        // h_prev = hidden at the last accepted position (vout column nacc).
        self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, nacc);

        // ---- Re-prime MTP over the accepted prefix with REAL hiddens (vout), in ONE forward. ----
        // Column k=0 uses h_save (the pre-verify hidden); k>=1 uses vout column k-1. Batching all
        // nacc+1 columns into a single MTP-layer forward reads the layer's weights once instead of
        // once per accepted position — the causal-append attention makes column k see the KV that
        // columns < k just wrote, exactly as in the multi-token verify.
        let n_rp = nacc + 1;
        let rp_hidden = self.pool.get_bf16(h * n_rp);
        let rp_ptr = *rp_hidden.device_ptr();
        let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
        self.gpu.copy_hidden_col(rp_ptr, self.mtp_h_save.as_ref().unwrap(), 0);
        rp_toks.push(committed_tok);
        // vout columns 0..nacc-1 are CONTIGUOUS — one dtod for the whole accepted prefix.
        self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, 0, nacc);
        for k in 1..=nacc {
            rp_toks.push(drafts[k - 1]);
        }
        self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1,
                             mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        self.pool.release_bf16(rp_hidden, h * n_rp);
        self.pool.release_bf16(vout, h * depth);

        // ---- Emit accepted drafts + bonus, honoring EOS and max_new. ----
        //
        // The BUDGET CHECK BELONGS ON THE DRAFTS TOO. It used to guard only the bonus, so a step that
        // accepted k drafts emitted all k regardless: a request for exactly max_tokens got back up to
        // max_tokens + depth - 1. Harmless-looking, and it is why a greedy MTP response and a greedy
        // plain response to the same prompt differed -- not in the tokens, which were identical, but
        // in how many of them came back. That cost real time to chase, because "MTP output != plain
        // output" reads as a losslessness failure when it is really an off-by-one in the stop rule.
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for &d in drafts.iter().take(nacc) {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(d);
            if eos.contains(&d) { hit_eos = true; break; }
        }
        // Bonus (the greedy next token after the last accepted position) — always progress.
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(bonus);
            if eos.contains(&bonus) { hit_eos = true; }
        }

        let emit_count = new_toks.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        // The verify FED [committed_tok] ++ drafts[..nacc] through the model, and the GDN state was
        // rolled back to exactly the last accepted column — so that, and only that, is what the slot's
        // state has consumed. The bonus token was PREDICTED, not fed. Cache what was consumed.
        // (If EOS truncated the emit, the cache still holds every fed token: it is then a longer
        // sequence than the client saw, so the next turn simply matches a shorter prefix. Correct,
        // just less reuse — which is the right way for this to fail.)
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }

        // Apply lane state. last_tok/pos/mtp_pos only matter if the lane continues, but keep them
        // consistent regardless. On full emit (not finished), advance the MTP cursor as in bench_mtp.
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &new_toks {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            // If the lane continues, it emitted all nacc drafts + bonus (emit_count == nacc+1).
            if !finished {
                lane.last_tok = bonus;
                lane.pos = main_pos + nacc + 1;
                lane.mtp_pos = main_pos + nacc;
            } else {
                lane.last_tok = *new_toks.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        // (committed_tok/main_pos/mtp_pos are all read above; no further bookkeeping needed.)

        // ---- Telemetry: accumulate per-step MTP stats and log a summary every 50 lane-steps. ----
        // Ping-pong GDN: every step is exactly ONE main-model verify forward (no reverify on reject).
        self.mtp_stat_steps += 1;
        self.mtp_stat_drafts += drafts.len() as u64;
        self.mtp_stat_accepted += nacc as u64;
        self.mtp_stat_emitted += emit_count as u64;
        self.mtp_stat_verify_fwds += 1;
        // Feed the auto-policy: it needs tokens-per-step to decide whether MTP is still paying.
        self.mtp.record_step(drafts.len() as u64, nacc as u64, emit_count as u64);
        self.log_draft_step(i, main_pos, committed_tok, &drafts, &preds, nacc);
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 {
                self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0
            } else { 0.0 };
            // Effective speedup ceiling = emitted tokens / verify forwards (how many output tokens
            // we get per main-model forward — 1.0 means no MTP benefit).
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            eprintln!("[mtp] steps={} drafts={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (depth {}) accept@k [{}]",
                      self.mtp_stat_steps, self.mtp_stat_drafts, acc,
                      self.mtp_stat_emitted, eff, depth, fmt_accept_by_depth(&self.mtp));
            self.dump_accept_curve();
        }
        finished
    }

    /// Stochastic MTP step for a sampling lane (temperature > 0). Mirrors mtp_lane_step but:
    /// 1. Drafts via mtp_draft_step_sample (samples from MTP head + records q(x))
    /// 2. Verifies via verify_forward_sample (returns p_of_draft + resid_tok + bonus_tok)
    /// 3. Accepts via speculative rejection sampling: accept draft with prob min(1, p(x)/q(x)),
    ///    else emit a token from the residual (p \ {draft}, renormalized).
    /// 4. Re-primes MTP with REAL verify hiddens for ACCEPTED positions only.
    /// Emits the accepted drafts + replacement/bonus token, advancing the lane.
    /// Returns true if the lane finished (EOS or max_new reached).
    fn mtp_lane_step_sample(&mut self, i: usize) -> bool {
        let h = self.gpu.cfg().hidden_size;
        let depth = self.mtp.depth();
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
        let h_save_ptr = *self.mtp_h_save.as_ref().unwrap().device_ptr();
        let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();

        // Snapshot lane state into locals.
        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let mtp_pos = self.lanes[i].as_ref().unwrap().mtp_pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        let temperature = self.lanes[i].as_ref().unwrap().temperature;
        let top_k = self.lanes[i].as_ref().unwrap().top_k;
        let top_p = self.lanes[i].as_ref().unwrap().top_p;
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();
        // All RNG for this step derives from one key (device column seeds + host accept draws,
        // domain-separated); the lane key advances exactly once per step.
        let step_key = self.lanes[i].as_ref().unwrap().seed;

        // h_save = h_prev (hidden at main_pos-1); saved for the post-accept re-prime step k=0.
        self.gpu.copy_hidden_col(h_save_ptr, &self.mtp_h_prev[phys], 0);

        // ---- Draft chain (depth-1 drafts) via greedy argmax from the MTP head. ----
        // Greedy drafting puts the draft token in the target model's high-probability region,
        // dramatically improving acceptance vs. sampling from the weak 1-layer MTP head.
        // Rejection sampling still corrects the output distribution to match non-MTP sampling.
        self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
        let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
        let mut qprobs: Vec<f32> = Vec::with_capacity(depth - 1);
        let mut cur_tok = committed_tok as i32;
        let mut dpos = mtp_pos;
        for _ in 0..depth - 1 {
            let m = self.gpu.mtp_draft_step(
                &mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), cur_tok, dpos,
                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            self.gpu.copy_hidden_col(cur_ptr, &m, 0);
            self.pool.release_bf16(m, h);
            let tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap());
            cur_tok = tok as i32;
            drafts.push(tok);
            qprobs.push(1.0); // greedy draft = point mass
            dpos += 1;
        }

        // ---- Build verify penalty (same as greedy). ----
        let verify_penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);

        // ---- Build verify input + seeds for spec_verify_b. ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        // Per-column device seeds, domain-separated from the host accept draws below.
        let verify_seeds: Vec<u32> =
            (0..depth).map(|j| rng_u32(step_key, RNG_DOM_VERIFY, j)).collect();

        // ---- Verify with stochastic output. ----
        let (vsample, vout) = self.gpu.verify_forward_sample(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), verify_penalty,
            &drafts, &qprobs, temperature, top_k, top_p, &verify_seeds);

        // ---- Speculative rejection sampling accept loop (Leviathan et al. 2023). ----
        // For each draft position j: accept with prob min(1, p_j(x_j) / q_j(x_j)).
        // On reject: emit resid_tok[j] and stop. If all accepted: emit bonus_tok.
        let mut nacc = 0usize;
        let mut emitted: Vec<u32> = Vec::with_capacity(depth);
        let mut rejected = false;
        let eps = 1e-12f32;
        for j in 0..drafts.len() {
            let ratio = if qprobs[j] < eps { 1.0 } else { (vsample.p_of_draft[j] / qprobs[j]).min(1.0) };
            let ru = rng_uniform(step_key, RNG_DOM_ACCEPT, j);
            if ru < ratio {
                emitted.push(drafts[j]);
                nacc += 1;
            } else {
                emitted.push(vsample.resid_tok[j]);
                rejected = true;
                break;
            }
        }
        if !rejected {
            emitted.push(vsample.bonus_tok);
        }

        // ---- GDN rollback on partial reject. ----
        if nacc + 1 != depth {
            // Restore the state as of the LAST ACCEPTED column, not column 0. Checkpoint slots are
            // contiguous: slot (snapshot + t) holds the post-state of verify column t.
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }

        // ---- h_prev = hidden at the last accepted position (vout column nacc). ----
        self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, nacc);

        // ---- Re-prime MTP with the REAL verify hiddens for the accepted positions, in ONE forward.
        // Only ACCEPTED positions are re-primed: `drafts[k-1]` for k<=nacc are by definition the
        // accepted drafts, so they are exactly the tokens that were emitted there. On a rejection the
        // replacement token at index nacc is NOT re-primed here — it becomes the next step's
        // committed token and is primed then.
        let n_rp = nacc + 1;
        let rp_hidden = self.pool.get_bf16(h * n_rp);
        let rp_ptr = *rp_hidden.device_ptr();
        let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
        self.gpu.copy_hidden_col(rp_ptr, self.mtp_h_save.as_ref().unwrap(), 0);
        rp_toks.push(committed_tok);
        // vout columns 0..nacc-1 are CONTIGUOUS — one dtod for the whole accepted prefix.
        self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, 0, nacc);
        for k in 1..=nacc {
            rp_toks.push(drafts[k - 1]);
        }
        self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1,
                             mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        self.pool.release_bf16(rp_hidden, h * n_rp);
        self.pool.release_bf16(vout, h * depth);

        // ---- Emit accepted tokens + replacement/bonus, honoring EOS and max_new. ----
        let mut hit_eos = false;
        let mut to_emit: Vec<u32> = Vec::with_capacity(emitted.len());
        for &t in &emitted {
            to_emit.push(t);
            if eos.contains(&t) { hit_eos = true; break; }
            if generated + to_emit.len() >= max_new { break; }
        }
        let emit_count = to_emit.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        // Same as the greedy path: the verify FED [committed_tok] ++ drafts[..nacc], and the GDN state
        // was rolled back to the last accepted column. On a rejection the REPLACEMENT token emitted at
        // index nacc was never fed — it becomes the next step's committed token and is fed then. So the
        // slot's state has consumed exactly this, and nothing more.
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }

        // Apply lane state.
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &to_emit {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            lane.seed = splitmix64(step_key);
            if !finished {
                lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + nacc + 1;
                lane.mtp_pos = main_pos + nacc;
            } else {
                lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        // ---- Telemetry. ----
        self.mtp_stat_steps += 1;
        self.mtp_stat_drafts += drafts.len() as u64;
        self.mtp_stat_accepted += nacc as u64;
        self.mtp_stat_emitted += emit_count as u64;
        self.mtp_stat_verify_fwds += 1;
        // Feed the auto-policy: it needs tokens-per-step to decide whether MTP is still paying.
        self.mtp.record_step(drafts.len() as u64, nacc as u64, emit_count as u64);
        // preds omitted: the stochastic verify SAMPLES rather than taking an argmax, so there is no
        // greedy per-column prediction to record (the parity gate uses the greedy path anyway).
        self.log_draft_step(i, main_pos, committed_tok, &drafts, &[], nacc);
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 {
                self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0
            } else { 0.0 };
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            eprintln!("[mtp] steps={} drafts={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (depth {}) accept@k [{}]",
                      self.mtp_stat_steps, self.mtp_stat_drafts, acc,
                      self.mtp_stat_emitted, eff, depth, fmt_accept_by_depth(&self.mtp));
            self.dump_accept_curve();
        }
        finished
    }

    /// One batched decode step over a subset of lanes (`batch_idx`). Builds the per-lane input
    /// arrays (tokens/positions/penalties/slot map), uploads them, runs the appropriate forward path
    /// (greedy graph / GPU-sample graph / CPU-sample / non-graph greedy), and returns the next token
    /// per lane (length = batch_idx.len()).
    fn batched_decode(&mut self, batch_idx: &[usize]) -> Vec<u32> {
        let s = batch_idx.len();
        let mb = self.max_batch;
        let mp = crate::gpu::MAX_PEN_TOKENS;
        let mut toks = vec![0i32; mb];
        let mut pos = vec![0i32; mb];
        let mut pen_tokens = vec![-1i32; mp * mb];
        let mut pen_counts = vec![0i16; mp * mb];
        let mut rep_pen = vec![1.0f32; mb];
        let mut presence_pen = vec![0.0f32; mb];
        let mut frequency_pen = vec![0.0f32; mb];
        for (k, &i) in batch_idx.iter().enumerate() {
            let lane = self.lanes[i].as_ref().unwrap();
            toks[k] = lane.last_tok as i32;
            pos[k] = lane.pos as i32;
            rep_pen[k] = lane.rep_penalty;
            presence_pen[k] = lane.presence_penalty;
            frequency_pen[k] = lane.frequency_penalty;
            // Fill this lane's unique recent tokens (with counts) only if it has any penalty;
            // lanes without penalty leave their slots as -1 sentinels (skipped by the kernel).
            if lane.has_penalty() {
                let base = k * mp;
                let mut idx = 0usize;
                for &t in lane.history.iter().rev().take(mp) {
                    let t_i = t as i32;
                    let found = (0..idx).position(|j| pen_tokens[base + j] == t_i);
                    match found {
                        Some(j) => { pen_counts[base + j] += 1; }
                        None => {
                            if idx < mp { pen_tokens[base + idx] = t_i; pen_counts[base + idx] = 1; idx += 1; }
                        }
                    }
                }
            }
        }
        let max_pc = batch_idx.iter()
            .map(|&i| self.lanes[i].as_ref().unwrap().pos + 1).max().unwrap_or(1);

        // Build the logical→physical slot map for this step's active lanes and upload it. The
        // stateful decode kernels index persistent KV/GDN state by slot_ids[lane], so active lanes
        // keep their assigned physical slots across compaction (no state copying).
        let mut slot_ids: Vec<i32> = (0..s)
            .map(|k| self.lanes[batch_idx[k]].as_ref().unwrap().phys as i32).collect();
        slot_ids.resize(mb, 0);

        self.gpu.dev().htod_sync_copy_into(&toks, &mut self.bufs.tokens_dev).unwrap();
        self.gpu.dev().htod_sync_copy_into(&pos, &mut self.bufs.pos_dev).unwrap();
        self.gpu.dev().htod_sync_copy_into(&slot_ids, &mut self.bufs.slot_ids_dev).unwrap();
        // The five penalty arrays are read by rep_penalty_b, which skips -1 sentinels — so they
        // only need uploading when some lane is actually penalized, plus ONE clear when the last
        // penalized lane departs (so its values cannot linger into an unpenalized successor).
        let any_pen = batch_idx.iter().any(|&i| self.lanes[i].as_ref().unwrap().has_penalty());
        if any_pen || self.pen_had {
            self.gpu.dev().htod_sync_copy_into(&pen_tokens, &mut self.bufs.penalty_tokens_dev).unwrap();
            self.gpu.dev().htod_sync_copy_into(&pen_counts, &mut self.bufs.penalty_counts_dev).unwrap();
            self.gpu.dev().htod_sync_copy_into(&rep_pen, &mut self.bufs.rep_pen_dev).unwrap();
            self.gpu.dev().htod_sync_copy_into(&presence_pen, &mut self.bufs.presence_dev).unwrap();
            self.gpu.dev().htod_sync_copy_into(&frequency_pen, &mut self.bufs.frequency_dev).unwrap();
            self.pen_had = any_pen;
        }
        // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute stream
        // already order these before the kernels/graph replay that read them (invariant I1).

        // Greedy lanes use the captured graph (penalties are now graph-compatible — rep_penalty_b
        // always scans MAX_PEN_TOKENS skipping -1 sentinels). Sampling lanes use the non-graph path.
        let any_sampling = batch_idx.iter().any(|&i| !self.lanes[i].as_ref().unwrap().greedy);
        let can_graph = !any_sampling && self.graphs.contains_key(&s);

        if can_graph {
            let graph = self.graphs.get(&s).unwrap();
            self.gpu.replay_decode(&self.bufs, graph, s)
        } else if any_sampling {
            let temps: Vec<f32> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().temperature).collect();
            let tks: Vec<usize> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().top_k).collect();
            let tps: Vec<f32> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().top_p).collect();
            if self.gpu_sample {
                // htod sampling params + fresh seeds into bufs (NULL stream), then dispatch to the
                // captured decode+sample graph when available, else the non-graph core.
                let mut t = temps.clone(); t.resize(mb, 1.0);
                let mut ki: Vec<i32> = tks.iter().map(|&x| x as i32).collect(); ki.resize(mb, 1);
                let mut p = tps.clone(); p.resize(mb, 1.0);
                // Seed sample_b from each lane's own PRNG, not rand::random(). The lane already
                // carries a seed (from the request's `seed` field), and the MTP path honours it —
                // drawing a fresh OS-random seed here meant an explicit `"seed": 42` was silently
                // ignored on the plain-sampler path, so identical requests were irreproducible.
                // Advance each lane's key once per decode step, exactly as the MTP path does.
                let mut sd: Vec<u32> = Vec::with_capacity(mb);
                for k in 0..s {
                    let lane = self.lanes[batch_idx[k]].as_mut().unwrap();
                    sd.push(rng_u32(lane.seed, RNG_DOM_SAMPLE, 0));
                    lane.seed = splitmix64(lane.seed);
                }
                sd.resize(mb, 0);
                self.gpu.dev().htod_sync_copy_into(&t, &mut self.bufs.temps_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&ki, &mut self.bufs.topk_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&p, &mut self.bufs.topp_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&sd, &mut self.bufs.seeds_dev).unwrap();
                // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute
                // stream already order these before the sample kernels/replay (invariant I1).
                if let Some(g) = self.sample_graphs.get(&s) {
                    self.gpu.replay_decode_sample(&self.bufs, g, s)
                } else {
                    self.gpu.forward_decode_sample_gpu(
                        &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s)
                }
            } else {
                self.gpu.forward_decode_sample(
                    &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s,
                    &temps, &tks, &tps)
            }
        } else {
            self.gpu.forward_decode(
                &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s)
        }
    }
}

#[cfg(test)]
mod tree_accept_tests {
    use super::tree_accept_walk;

    // A tree degenerates to a chain (parent[c]=c-1). The walk must reproduce "accept longest prefix".
    #[test]
    fn chain_is_accept_longest_prefix() {
        // committed=100; drafts=[10,20,30]; target preds=[10,20,99,..] => accept 10,20, correct 30->99.
        let parent = [-1, 0, 1, 2];
        let tokens = [100u32, 10, 20, 30];
        let preds  = [10u32, 20, 99, 7];         // preds[2]=99 != tokens[3]=30 -> stop, bonus 99
        let (path, emitted) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0, 1, 2]);          // accepted committed, 10, 20
        assert_eq!(emitted, vec![10, 20, 99]);    // 10, 20 accepted, 99 the correction
    }

    // Fork at position 1: child A (col 1) wrong, child B (col 2) right -> B rescued, then chains.
    #[test]
    fn fork_rescues_second_branch() {
        //        0(committed=100)
        //       / \
        //   1(A=10) 2(B=20)---3(B2=30)
        let parent = [-1, 0, 0, 2];
        let tokens = [100u32, 10, 20, 30];
        let preds  = [20u32, 5, 30, 88];  // after committed target wants 20 (=B), then 30 (=B2), then 88
        let (path, emitted) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0, 2, 3]);          // walked root -> B -> B2
        assert_eq!(emitted, vec![20, 30, 88]);    // B, B2 accepted, 88 the bonus
    }

    // Neither child matches at the root -> emit just the correction, accept nothing past committed.
    #[test]
    fn no_child_matches() {
        let parent = [-1, 0, 0];
        let tokens = [100u32, 10, 20];
        let preds  = [77u32, 1, 2];               // target wants 77, neither child is 77
        let (path, emitted) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0]);
        assert_eq!(emitted, vec![77]);
    }

    // Tie: two children carry the target's token -> prefer the lowest index (deterministic).
    #[test]
    fn tie_prefers_lowest_child() {
        let parent = [-1, 0, 0];
        let tokens = [100u32, 42, 42];
        let preds  = [42u32, 9, 9];
        let (path, _) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0, 1]);             // col 1, not col 2
    }
}
