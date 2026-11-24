//! The seam between the serve shell and the model: the `Forward` trait the lane loop drives, and the CPU
//! `StubForward` the shell is tested (and smoke-served) with.
//!
//! What the GPU forward (legs TP / CE / SL / integration) has to provide, in the shell's terms:
//! * `prefill(lane, from, tokens)`: process `tokens` at absolute positions `from..from+len` in `lane`, whose state
//!   before `from` is the lane's own (fresh = 0, or restored from a snapshot); return the LAST row's output;
//! * `step(rows)`: one decode step, one input token per row at its position; one `RowOut` per row, in order;
//! * `snapshot / restore / free`: the prefix state of a lane at a position: KV + SWA rings + compressor frontiers
//!   (no Engram n-gram state: it is rebuilt from the token ids), see `prefix.rs`;
//! * `truncate(lane, to)`: forget every position at or after `to` (the sink rejection's rollback; the real forward
//!   implements it with the rejected-draft rollback it needs for speculation anyway).
//! A `RowOut` carries the full-vocabulary logits (FP32, as DeepSeek's reference) on the host. The GPU sampler with
//! top-k candidates replaces the transfer later (roadmap P7); this is the contract's reference form.

use anyhow::{bail, Result};
use std::any::Any;

/// One decode row of a step.
#[derive(Clone, Copy, Debug)]
pub struct StepRow { pub lane: usize, pub pos: usize, pub token: u32 }

/// One lane's segment in a batched multi-lane prefill (`Forward::prefill_batch`, leg P7.3).
#[derive(Clone, Copy, Debug)]
pub struct PrefillItem<'a> { pub lane: usize, pub from: usize, pub tokens: &'a [u32] }

/// One lane's parameters in a batched multi-lane start (`Forward::start_lanes`, leg P7.3).
#[derive(Clone, Copy, Debug)]
pub struct StartItem<'a> { pub lane: usize, pub sampling: &'a super::sampler::Sampling, pub seed: u64, pub suppress_eos: bool }

/// What one row of a forward produced.
#[derive(Clone, Debug, Default)]
pub struct RowOut {
    pub logits: Vec<f32>,
    /// |residual[sink column]| of this row (max over the mHC streams after the last layer), when the forward was asked
    /// for it (`--dsv41-sink-reject`); None = not measured.
    pub sink: Option<f32>,
}

/// A lane's prefix state at `pos`: opaque to the shell. `bytes` is what the pool charges for it.
/// `min_suffix` (leg CE): the rows a prefill resumed from this state must recompute at least, for the resume to be
/// exact. A FULL state (every layer's ring valid at `pos`: an exact prefill, the prompt end of a CED prefill, a decode
/// position) needs only the pool's floor (window 128 + 6 backtracking = 134; 0 here = that floor). An ENCODER-only
/// state (a mid-prompt state of a CED prefill: the decoder layers never ran there) needs the whole decoder cone
/// (2,541 rows for this model: `ced::schedule::Schedule::min_suffix`). The pool never hands out a state whose
/// `min_suffix` the new prompt does not leave.
///
/// `tail` (leg CE3): a FULL state taken at the very end of a prompt that also carries what a resumed prefill would otherwise
/// recompute the prompt's last rows for (the TP forward: the DSpark taps of the last 128 rows). Such a state also resumes a
/// prompt that leaves fewer than 134 rows after it: the same prompt again (rolled back up to 6 rows: `PrefixPool::TAIL_BACK`),
/// a prompt whose last few tokens changed, or a short extension of it.
pub struct Snapshot { pub pos: usize, pub bytes: u64, pub data: Box<dyn Any + Send>, pub min_suffix: usize, pub tail: bool }

pub trait Forward {
    fn vocab(&self) -> usize;
    /// Lanes the forward holds state for.
    fn lanes(&self) -> usize;
    /// Longest sequence (prompt + generation) a lane holds.
    fn max_seq_len(&self) -> usize;
    /// Ask the forward to measure the sink column on every row (`--dsv41-sink-reject`).
    fn want_sink(&mut self, _on: bool) {}
    fn prefill(&mut self, lane: usize, from: usize, tokens: &[u32]) -> Result<RowOut>;
    fn step(&mut self, rows: &[StepRow]) -> Result<Vec<RowOut>>;
    fn snapshot(&mut self, lane: usize, pos: usize) -> Result<Snapshot>;
    /// Make `lane` hold exactly the state `snap` captured (positions `0..snap.pos`).
    fn restore(&mut self, lane: usize, snap: &Snapshot) -> Result<()>;
    fn truncate(&mut self, lane: usize, to: usize) -> Result<()>;
    /// Rows a mid-prompt snapshot of this forward must leave to a resumed prefill (`Snapshot::min_suffix` of the
    /// states the prefill takes on its grid): the engine places its grid snapshots at or below `prompt - this`, so a
    /// follow-up of the same length can use them. The TP CED forward answers 134: since leg CE2 its grid states are FULL
    /// (`prefill_snapshots`), so the pool's own floor is all they need.
    fn mid_prompt_keep(&self) -> usize { super::prefix::MIN_KEEP }
    /// The lane's prompt of `total` tokens is about to be prefilled, in calls of `prefill` that continue each other (leg SE2: the
    /// lane loop interleaves a long prompt's chunks with the other lanes' decode rounds).
    fn begin_prefill(&mut self, _lane: usize, _total: usize) {}
    /// Leg VIS1: set per-token VL metadata (`token_types` over `0..prompt_len`) and Rank-0 image patches before prefill.
    /// `cached` is the prefix-cache resume position (`0` on cold prefill): images with `img.end <= cached` are already
    /// baked into the restored prefix state and skip ViT re-encoding.
    fn set_vl_inputs(
        &mut self,
        _lane: usize,
        _cached: usize,
        _token_types: Vec<i8>,
        _images: Vec<crate::dsv41::vision::preproc::PreparedImage>,
    ) -> Result<()> {
        Ok(())
    }
    /// Leg CE2: the prefill just begun (resumed at `from`, `total` tokens) will be snapshotted at the positions `at` (the shell's
    /// `prefix::snapshot_plan`), so the forward can make the states it takes there resumable (the CED forward makes every
    /// mid-prompt one FULL: `ced::schedule::Schedule::new_with`). Called after `begin_prefill`, only with the prefix cache on.
    fn prefill_snapshots(&mut self, _lane: usize, _from: usize, _total: usize, _at: &[usize]) {}
    /// Leg CE2: can a prefill of a `prompt_len`-token prompt resume exactly from `snap`? The shell asks this BEFORE it restores a
    /// kept state into a lane; a refusal makes it fall back to the next usable kept state or a cold prefill (one line, counted
    /// on `/metrics`), never a failed request or a dead forward. The default accepts every state the pool's labels admit.
    /// `at` = where the shell would resume (the state's own position, or up to `PrefixPool::TAIL_BACK` rows before it for a
    /// `tail` state, which the shell then rolls back with `truncate`).
    fn can_resume(&self, _snap: &Snapshot, _at: usize, _prompt_len: usize) -> Result<()> { Ok(()) }
    /// Leg CE3: the rows a mid-prompt state of this forward's prefill must leave to a resumed prompt (its decoder cone: the
    /// TP CED forward's mid-prompt states are encoder-only). `Some(c)`: the shell snapshots a cold prompt only where its prefill
    /// calls end anyway (no extra call) and only below `prompt - c`; `None` (default): the 128-token grid near the end.
    fn encoder_cone(&self) -> Option<usize> { None }
    /// Rows the LAST prefill call of a prompt must carry at least (a forward that reads the prompt tail after the call, such as the
    /// DSpark taps of the last 128 rows, cannot have another lane's round run between two calls that both write that tail).
    fn prefill_tail(&self) -> usize { 0 }
    /// Leg PP1: the tokens of the lane's NEXT prefill call (the prompt rows right after the call about to be made), given just
    /// before that call; a forward may start reading what they need (the TP forward: their Engram rows) while the call computes.
    /// Advisory only: the next call's rows are what its own `prefill` passes. The default ignores it.
    fn prefill_lookahead(&mut self, _lane: usize, _next: &[u32]) {}
    /// A rounds forward: the lane's last round lost its last `drop` tokens (the pending one included, `drop >= 1`) and its pending
    /// token becomes `tok` (`thinking_token_budget` forcing the end of the reasoning block).
    fn force(&mut self, _lane: usize, _drop: usize, _tok: u32) -> Result<()> { bail!("the {} forward cannot force a token", self.name()) }
    /// Score mode (GA's G1; `POST /v1/completions` with `echo: true, max_tokens: 0, logprobs`): an EXACT prefill of
    /// `tokens` in `lane` (every layer on every row; CED has no logits for most rows) and the log-probability of
    /// every token given the ones before it: `out[i] = log p(tokens[i + 1] | tokens[..= i])`, `tokens.len() - 1`
    /// values. The lane holds no request afterwards (it is released).
    fn score(&mut self, _lane: usize, _tokens: &[u32]) -> Result<Vec<f32>> {
        bail!("prompt scoring is not implemented by the {} forward", self.name())
    }
    /// The lane is free; its state may be dropped.
    fn release(&mut self, lane: usize);
    // ---- the round-based forward (leg SE): speculation draws its own tokens ----
    /// True when the forward samples inside its own rounds (DSpark under TP): the lane loop then calls `start` once
    /// per request and `round` per iteration instead of `step` + the host sampler.
    fn rounds(&self) -> bool { false }
    /// Draw the request's first token from the prompt just prefilled in `lane` (rounds forwards only).
    fn start(&mut self, _lane: usize, _s: &super::sampler::Sampling, _seed: u64) -> Result<u32> { bail!("the {} forward does not run rounds", self.name()) }
    /// [`Self::start`] with optional EOS suppression on the first token (`min_tokens > 0 || ignore_eos`, leg P7.2).
    fn start_ex(&mut self, lane: usize, s: &super::sampler::Sampling, seed: u64, _suppress_eos: bool) -> Result<u32> {
        self.start(lane, s, seed)
    }
    /// Batched multi-lane prefill of short/tail prompts (`items.len() >= 2`, leg P7.3).
    fn prefill_batch(&mut self, items: &[PrefillItem<'_>]) -> Result<Vec<RowOut>> {
        items.iter().map(|it| self.prefill(it.lane, it.from, it.tokens)).collect()
    }
    /// Batched multi-lane first-token start (`items.len() >= 2`, leg P7.3).
    fn start_lanes(&mut self, items: &[StartItem<'_>]) -> Result<Vec<u32>> {
        items.iter().map(|it| self.start_ex(it.lane, it.sampling, it.seed, it.suppress_eos)).collect()
    }
    /// One round over `lanes` (all decoding): the tokens each emitted, in order (1 + accepted drafts).
    fn round(&mut self, _lanes: &[usize]) -> Result<Vec<Vec<u32>>> { bail!("the {} forward does not run rounds", self.name()) }
    /// [`Self::round`] carrying per-lane `suppress_eos_rows` (`0..=6`: rows `0..n` mask model stop_ids, leg P7.2).
    fn round_ex(&mut self, lanes_sup: &[(usize, usize)]) -> Result<Vec<Vec<u32>>> {
        let ls: Vec<usize> = lanes_sup.iter().map(|&(l, _)| l).collect();
        self.round(&ls)
    }
    /// Positions the forward holds for `lane` (rounds forwards).
    fn lane_pos(&self, _lane: usize) -> Option<usize> { None }
    /// Called by the lane loop while it has nothing to do (the TP forward keeps its nodes awake).
    fn idle(&mut self) -> Result<()> { Ok(()) }
    /// Leg MEM1: what the prefix pool may hold now, given the `pool_bytes` it holds and its configured `budget` (the TP forward
    /// follows the box's free memory against the memory watchdog's floor); None = no change this time.
    fn pool_allowance(&mut self, _pool_bytes: u64, _budget: u64) -> Option<u64> { None }
    /// Leg MEM1: the device bytes a prefix state at `pos` positions takes (None = unknown: the engine takes it and the pool
    /// decides); the engine does not copy a state the pool has no room for.
    fn state_bytes(&self, _pos: usize) -> Option<u64> { None }
    /// Leg PFX2: whether prompt-end states captured by `snapshot` carry the prompt tail (`Snapshot::tail == true`).
    fn has_tail_snaps(&self) -> bool { false }
    /// Leg PFX2: device bytes of a prefix state at `pos` (`is_tail`: includes any prompt-end tail buffer).
    fn state_bytes_for(&self, pos: usize, _is_tail: bool) -> Option<u64> { self.state_bytes(pos) }
    /// Leg PFX2: flush any queued device snapshot frees (and trim the device pool when budget is tight) before allocating.
    fn flush_frees(&mut self) -> Result<()> { Ok(()) }
    /// A forward that failed in a way that leaves it unusable says so: the lane loop stops.
    fn fatal(&self) -> bool { false }
    /// The lane loop is ending.
    fn shutdown(&mut self) {}
    /// A short name for the boot line and metrics.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------------------------------------------------
// The stub
// ---------------------------------------------------------------------------------------------------------------------

/// What the stub says for a prompt: the reply tokens (the stub then emits EOS).
pub type ReplyFn = Box<dyn Fn(&[u32]) -> Vec<u32> + Send>;
/// Whether a row is a sink: given the lane's tokens so far (prompt + fed, including the row's input token).
pub type SinkFn = Box<dyn Fn(&[u32]) -> bool + Send>;

/// A CPU forward with no model: lane `l` holds the tokens fed so far; its next-token logits are one-hot on the
/// scripted reply (`reply_fn(prompt)`), with a second candidate (`second`) just below it so alternates exist.
pub struct StubForward {
    vocab: usize,
    eos: u32,
    second: u32,
    max_seq: usize,
    reply_fn: ReplyFn,
    sink_fn: Option<SinkFn>,
    want_sink: bool,
    prefill_delay: std::time::Duration,
    lanes: Vec<StubLane>,
    /// every (lane, pos, token) row ever stepped, for tests
    pub trace: Vec<(usize, usize, u32)>,
    pub prefills: Vec<(usize, usize, usize)>,
    pub restores: usize,
    pub truncates: Vec<(usize, usize)>,
}

#[derive(Default)]
struct StubLane { tokens: Vec<u32>, prompt_len: usize, reply: Vec<u32> }

impl StubForward {
    pub fn new(vocab: usize, eos: u32, lanes: usize, max_seq: usize, reply_fn: ReplyFn) -> Self {
        StubForward {
            vocab, eos, second: 2.min(vocab as u32 - 1), max_seq, reply_fn, sink_fn: None, want_sink: false, prefill_delay: std::time::Duration::ZERO,
            lanes: (0..lanes).map(|_| StubLane::default()).collect(), trace: Vec::new(), prefills: Vec::new(),
            restores: 0, truncates: Vec::new(),
        }
    }
    pub fn with_sink(mut self, f: SinkFn) -> Self { self.sink_fn = Some(f); self }
    pub fn with_prefill_delay(mut self, d: std::time::Duration) -> Self { self.prefill_delay = d; self }
    pub fn with_second(mut self, t: u32) -> Self { self.second = t; self }

    fn out_for(&self, lane: usize) -> RowOut {
        let l = &self.lanes[lane];
        let idx = l.tokens.len() - l.prompt_len;
        let next = l.reply.get(idx).copied().unwrap_or(self.eos);
        let mut logits = vec![-10.0f32; self.vocab];
        if self.second != next { logits[self.second as usize] = 10.0; }
        logits[next as usize] = 20.0;
        let sink = if self.want_sink { Some(if self.sink_fn.as_ref().map_or(false, |f| f(&l.tokens)) { 120_000.0 } else { 200.0 }) } else { None };
        RowOut { logits, sink }
    }
}

impl Forward for StubForward {
    fn vocab(&self) -> usize { self.vocab }
    fn lanes(&self) -> usize { self.lanes.len() }
    fn max_seq_len(&self) -> usize { self.max_seq }
    fn want_sink(&mut self, on: bool) { self.want_sink = on; }
    fn name(&self) -> &'static str { "stub" }

    fn prefill(&mut self, lane: usize, from: usize, tokens: &[u32]) -> Result<RowOut> {
        if !self.prefill_delay.is_zero() { std::thread::sleep(self.prefill_delay); }
        let l = &mut self.lanes[lane];
        if l.tokens.len() != from { bail!("stub prefill: lane {lane} holds {} positions, asked to resume at {from}", l.tokens.len()); }
        l.tokens.extend_from_slice(tokens);
        self.prefills.push((lane, from, tokens.len()));
        // a prefill that completes a prompt fixes the reply (the stub is told the prompt by its first call and
        // extended by later chunks: the reply is a function of the whole prompt, so it is recomputed per chunk)
        let l = &mut self.lanes[lane];
        l.prompt_len = l.tokens.len();
        l.reply = (self.reply_fn)(&l.tokens);
        Ok(self.out_for(lane))
    }

    fn step(&mut self, rows: &[StepRow]) -> Result<Vec<RowOut>> {
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let l = &mut self.lanes[r.lane];
            if l.tokens.len() != r.pos { bail!("stub step: lane {} holds {} positions, row is at {}", r.lane, l.tokens.len(), r.pos); }
            l.tokens.push(r.token);
            self.trace.push((r.lane, r.pos, r.token));
            out.push(self.out_for(r.lane));
        }
        Ok(out)
    }

    fn snapshot(&mut self, lane: usize, pos: usize) -> Result<Snapshot> {
        let l = &self.lanes[lane];
        if pos > l.tokens.len() { bail!("stub snapshot at {pos} beyond the lane's {}", l.tokens.len()); }
        Ok(Snapshot { pos, bytes: 4096 + pos as u64 * 64, data: Box::new(l.tokens[..pos].to_vec()), min_suffix: 0, tail: false })
    }

    fn restore(&mut self, lane: usize, snap: &Snapshot) -> Result<()> {
        let toks = snap.data.downcast_ref::<Vec<u32>>().ok_or_else(|| anyhow::anyhow!("stub restore: foreign snapshot"))?;
        let l = &mut self.lanes[lane];
        l.tokens = toks.clone();
        l.prompt_len = toks.len();
        self.restores += 1;
        Ok(())
    }

    fn truncate(&mut self, lane: usize, to: usize) -> Result<()> {
        self.lanes[lane].tokens.truncate(to);
        self.truncates.push((lane, to));
        Ok(())
    }

    fn release(&mut self, lane: usize) { self.lanes[lane] = StubLane::default(); }

    /// The stub's scores: a fixed function of the token (the API path and its plumbing are what the tests check).
    fn score(&mut self, lane: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() { bail!("stub score: empty prompt"); }
        self.lanes[lane] = StubLane::default();
        Ok(tokens[1..].iter().map(|&t| stub_score(t)).collect())
    }
}

/// The stub's log-prob of a token (tests compare against it).
pub fn stub_score(t: u32) -> f32 { -0.25 * ((t % 17) as f32 + 1.0) }
