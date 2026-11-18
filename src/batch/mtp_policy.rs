//! `MtpPolicy` behaviour: the windowed depth/enable controller (`tick`, `record_step`, the yield model).
//! The struct and `SpecSource` stay in `mod.rs`; decisions use the MEASURED r(d) per depth (AGENTS.md section 6).

use super::*;

pub(super) const MTP_EVAL_WINDOW: u64 = 128;
/// First window after activation (and after each re-probe): catch a losing head quickly.
pub(super) const MTP_EVAL_FIRST: u64 = 32;
/// Decode steps to wait before re-probing a model whose acceptance had fallen below break-even
/// (the workload may have changed — acceptance is not a fixed property of the model).
pub(super) const MTP_RETRY_AFTER: u64 = 4096;
/// A different depth must beat the current one by this factor to be worth switching to.
pub(super) const MTP_DEPTH_MARGIN: f32 = 1.05;

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
pub(super) fn yield_at(hz: &[(f64, f64); MAX_AUTO_DEPTH], d: usize) -> f32 {
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
pub(super) fn fmt_accept_by_depth(mtp: &MtpPolicy) -> String {
    mtp.hazard_counts().iter().enumerate().map(|(i, &(a, n))| {
        if n >= 8 { format!("@{}:{:.0}%", i + 1, a as f64 / n as f64 * 100.0) }
        else { format!("@{}:?", i + 1) }
    }).collect::<Vec<_>>().join(" ")
}

impl MtpPolicy {
    pub fn new(head_present: bool, force: Option<bool>, pin_depth: Option<usize>,
               r_ctx: Vec<(usize, Vec<(usize, f32)>)>) -> Self {
        MtpPolicy::with_source(head_present, force, pin_depth, r_ctx, SpecSource::Mtp)
    }

    /// S5F: `new` + an explicit speculation source (`--spec-source`). Default routing stays MTP;
    /// the source is selectable per process (S6F owns per-domain routing).
    pub fn with_source(head_present: bool, force: Option<bool>, pin_depth: Option<usize>,
                       r_ctx: Vec<(usize, Vec<(usize, f32)>)>, spec_source: SpecSource) -> Self {
        // Start ON when auto: MTP is correctness-neutral either way (greedy is bitwise lossless,
        // stochastic is distribution-exact), so the worst case of guessing wrong is a slightly slow
        // first window, which the evaluation below then corrects.
        let active = head_present && force.unwrap_or(true);
        // Open in the MIDDLE of the range, not at 2. The policy learns a per-position hazard curve,
        // and at depth 2 it only ever observes position 1 — so it would have to extrapolate the whole
        // curve from its easiest point, which is exactly how it used to over-reach. Starting at 4
        // gives the first window three positions of real data to reason from.
        let depth = pin_depth.unwrap_or(4).clamp(2, MAX_AUTO_DEPTH);
        Self { head_present, spec_source, force, pin_depth, depth, r_ctx, last_ctx: 0, active,
               win_steps: 0, win_emitted: 0, win_drafts: 0, win_accepted: 0,
               hz: [(0.0, 0.0); MAX_AUTO_DEPTH], decode_steps: 0, retry_at: 0,
               first_short: true, explore_every: 0, windows: 0 }
    }
    /// COPY27B: arm the depth exploration (see `explore_every`); 0 = off.
    pub(crate) fn set_explore(&mut self, every: u64) { self.explore_every = every; }
    /// COPY27B: open at the deepest depth instead of 4 (unless pinned). With the copy arm on, copy
    /// rounds serve most rounds of copy-heavy requests and the policy only learns from the drafter
    /// rounds in between: opening at 4 kept a boot at d4/d5 through its first requests (measured:
    /// an agent turn after two code edits ran 207 of 251 drafter rounds at d4/d5, 36.9 against
    /// 43.5 tok/s at d8 with the flag off). Opening deep measures every position in the first
    /// window; a weak head still steps down at its first evaluation.
    pub(crate) fn open_deep(&mut self) {
        if self.pin_depth.is_none() { self.depth = MAX_AUTO_DEPTH; }
    }
    pub fn active(&self) -> bool { self.active }
    /// B8/G3: runtime switch (per lane-admission or policy window). Records the source and
    /// returns whether the DRAFT pass should run the block drafter this step. The serving loop
    /// consults this at admission + every eval window; the decision joins the agree hash.
    pub fn set_spec_source(&mut self, src: SpecSource) {
        self.spec_source = src;
    }
    pub fn spec_source(&self) -> SpecSource { self.spec_source }
    /// True when the block drafter should own this step's draft. ALWAYS falls back to the MTP
    /// head when the DSpark model is not resident (the loader sets `dspark_present` false until
    /// G4's artifact is bound); the switch is exercised but never serves from nothing.
    pub fn use_dspark(&self, dspark_present: bool) -> bool {
        self.spec_source == SpecSource::Dspark && dspark_present
    }
    /// S5F: true when the DFlash2 round should own this step's draft. ALWAYS falls back to the
    /// MTP path when the DFlash2 round is not resident (`df2_present` false — absent or failed
    /// artifact: the standing directive's fallback, never a hard failure). b==1 only (the decode
    /// loop enforces it; AGENTS §4).
    pub fn use_dflash2(&self, df2_present: bool) -> bool {
        is_df2_src(self.spec_source) && df2_present
    }
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
    pub fn r(&self) -> f32 { self.r_at(self.depth, self.last_ctx) }
    /// The r(d) table for context `ctx`: the first bucket whose point×2 covers it, else the top
    /// (asymptotic) bucket. Empty table (calibration failed) → every r reads as INFINITY → the
    /// policy disables itself, exactly as before.
    pub(super) fn bucket_table(&self, ctx: usize) -> &[(usize, f32)] {
        if self.r_ctx.is_empty() { return &[]; }
        let i = self.r_ctx.iter().position(|&(p, _)| ctx <= p * 2)
            .unwrap_or(self.r_ctx.len() - 1);
        &self.r_ctx[i].1
    }
    pub(super) fn r_at(&self, d: usize, ctx: usize) -> f32 {
        self.bucket_table(ctx).iter().find(|&&(x, _)| x == d).map(|&(_, r)| r).unwrap_or(f32::INFINITY)
    }
    /// Acceptance at which MTP breaks even, for reporting: tokens_per_step must exceed `r`.
    pub fn break_even_accept(&self) -> f32 {
        ((self.r() - 1.0) / (self.depth.max(2) - 1) as f32).clamp(0.0, 1.0)
    }

    /// Record one completed MTP step. `accepted` is the accepted PREFIX length: drafts 1..=accepted
    /// were taken and draft `accepted+1` (if there was one) was rejected. That is exactly the
    /// information a per-position hazard needs.
    pub(super) fn record_step(&mut self, drafts: u64, accepted: u64, emitted: u64) {
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

    /// Called once per decode step with the deepest live context (max lane pos). Re-evaluates the
    /// decision when a window completes, and re-probes a disabled model after a cooldown.
    pub(super) fn tick(&mut self, ctx: usize) {
        self.decode_steps += 1;
        self.last_ctx = ctx;
        if self.force.is_some() || !self.head_present {
            if mtp_dispatch_trace() && self.decode_steps <= 3 {
                eprintln!("[b31-trace] tick: no-eval (force={:?} head={} depth={} ctx={})",
                          self.force, self.head_present, self.depth, ctx);
            }
            return;
        }

        if !self.active {
            if self.decode_steps >= self.retry_at {
                self.active = true;
                self.first_short = true;
                self.win_steps = 0; self.win_emitted = 0; self.win_drafts = 0; self.win_accepted = 0;
                eprintln!("[mtp] re-probing (workload may have changed)");
            }
            return;
        }
        let need = if self.first_short { MTP_EVAL_FIRST } else if self.explore_every > 0 { COPY_MTP_EVAL_WINDOW } else { MTP_EVAL_WINDOW };
        if self.win_steps < need { return; }
        self.first_short = false;

        let observed = self.win_emitted as f32 / self.win_steps as f32;
        let acc = self.win_accepted as f32 / self.win_drafts.max(1) as f32;
        self.win_steps = 0; self.win_emitted = 0; self.win_drafts = 0; self.win_accepted = 0;

        // Re-pick the depth FROM THE CURRENT CONTEXT'S BUCKET (E17): a step buys `yield_at(d)`
        // tokens for r(d) decode steps, so maximise the ratio. r(d) grows with context (the verify's
        // KV bytes), so the same acceptance curve yields a shallower optimum at ≥16K than at 2K.
        let cur = yield_at(&self.hz, self.depth) / self.r_at(self.depth, ctx);
        let (mut best_d, mut best) = (self.depth, cur);
        if self.pin_depth.is_none() {
            let table = self.bucket_table(ctx).to_vec();
            for &(d, r) in &table {
                if r <= 0.0 || !r.is_finite() || d == self.depth { continue; }
                let s = yield_at(&self.hz, d) / r;
                // Hysteresis, ASYMMETRIC (PLAN/10 #12, 2026-08-17): an UP-switch still needs the
                // real margin — adjacent depths score within a fraction of a percent and flapping
                // 4->5->4 costs a window of relearning each way. But a DOWN-switch is CHEAP: the
                // hazard observations from the deeper run are all still valid at the shallower
                // depth (hz[i] for i < new depth were measured under the deep policy and are
                // exactly the same conditional events), so the next window can go straight back
                // up if it was wrong. The P2 canonical run showed the symmetric margin eating a
                // real win: hazards [@1:44% @2:31% @3:26% @4:24%] (accept@4 = 0.54 × accept@1)
                // with r(3)/r(4) = 1.26/1.39 made d3 strictly better, and the policy stayed at
                // d4/d5 — the 5% margin outweighed the ~1% score gap. Down-margin 1.0.
                let margin = if d < self.depth { 1.0 } else { MTP_DEPTH_MARGIN };
                if s > best * margin { best = s; best_d = d; }
            }
        }

        // TP item E: log EVERY window evaluation, not just switches. Under TP=2 both ranks run
        // this policy on bit-identical token history with the head's shipped r(d) tables, so the
        // lines must be byte-identical across ranks — and a gate that only diffed switch events
        // could pass vacuously on a no-decision run (AGENTS.md §4.12). All values are pure
        // functions of the deterministic hazard curve and the shipped table.
        crate::reprintln!("[mtp] window: d={} yield {:.2} acc {:.1}% | ctx {} | cur {:.2}x best d={} {:.2}x",
                  self.depth, observed, acc * 100.0, ctx, cur, best_d, best);

        if best < 1.0 {
            self.active = false;
            self.retry_at = self.decode_steps + MTP_RETRY_AFTER;
            crate::reprintln!("[mtp] DISABLED: acceptance {:.1}% gives {:.2} tok/step, and no depth beats a \
                       plain decode (best {:.2}x). Re-probing in {} steps.",
                      acc * 100.0, observed, best, MTP_RETRY_AFTER);
            return;
        }
        self.windows += 1;
        if self.explore_every > 0 && self.pin_depth.is_none() && self.windows % self.explore_every == 0
            && best_d < MAX_AUTO_DEPTH && self.r_at(MAX_AUTO_DEPTH, ctx).is_finite() {
            crate::reprintln!("[mtp] explore: one window at depth {MAX_AUTO_DEPTH} (copy-span on: the head's deep \
                       hazards are refreshed on the text it drafts; best was d={best_d})");
            best_d = MAX_AUTO_DEPTH;
            best = yield_at(&self.hz, best_d) / self.r_at(best_d, ctx);
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
                      self.r(), self.r_at(best_d, ctx), cur, best, hzs.join(" "));
            self.depth = best_d;
        }
    }
}
