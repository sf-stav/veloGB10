//! Per-lane helpers: `token_logprobs`, the JSON-schema mask feed (`SchemaStep`, `schema_feed`), `think_emit`, and `impl Lane`
//! (stats, cancel and loop logging, sampler rows, thinking-budget filter). The `Lane` type itself is defined in `mod.rs`.

use super::*;

/// Log-softmax of one f16 logits row: the natural-log probability of `tok` and the `n`
/// most likely tokens (logprob descending, ties by id ascending; only finite logits — a token a
/// mask removed has probability 0 and is never listed). None when `tok` is outside the row or
/// its logit, or the row's maximum, is not finite.
///
/// One counting pass over the row's f16 bit patterns, then the normalizer as a sum over the
/// distinct values present (exp once per value, accumulated in f64) and a second pass that
/// collects only the ids at or above the n-th largest value — no per-call f32 copy or index
/// vector of the whole vocabulary.
pub(super) fn token_logprobs(row: &[u16], tok: u32, n: usize) -> Option<(f32, Vec<(u32, f32)>)> {
    let lg = |b: u16| half::f16::from_bits(b).to_f32();
    let xt = lg(*row.get(tok as usize)?);
    if !xt.is_finite() { return None; }
    let mut cnt = vec![0u32; 1 << 16];
    for &b in row { cnt[b as usize] += 1; }
    // finite values, largest first: +65504 .. +0 are 0x7BFF down to 0x0000, -0 .. -65504 are
    // 0x8000 up to 0xFBFF (infinities and NaN patterns lie outside both ranges)
    let desc = || (0x0000u32..=0x7BFF).rev().chain(0x8000..=0xFBFF).filter(|&p| cnt[p as usize] > 0);
    let lmax = lg(desc().next()? as u16) as f64;
    let sum: f64 = desc().map(|p| cnt[p as usize] as f64 * (lg(p as u16) as f64 - lmax).exp()).sum();
    let lse = lmax + sum.ln();
    let lp = |x: f32| (x as f64 - lse) as f32;
    let mut top: Vec<(u32, f32)> = Vec::new();
    if n > 0 {
        // the n-th largest value present (ties included)
        let mut acc = 0usize;
        let mut vth = f32::NEG_INFINITY;
        for p in desc() {
            acc += cnt[p as usize] as usize;
            vth = lg(p as u16);
            if acc >= n { break; }
        }
        let mut cand: Vec<(u32, f32)> = row.iter().enumerate()
            .filter_map(|(i, &b)| { let x = lg(b); (x.is_finite() && x >= vth).then_some((i as u32, x)) })
            .collect();
        // value descending, then id ascending (finite values: partial_cmp is total here)
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        cand.truncate(n);
        top = cand.into_iter().map(|(i, x)| (i, lp(x))).collect();
    }
    Some((lp(xt), top))
}

/// W2-EXL3: what one emitted token does to a schema lane.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum SchemaStep {
    /// keep decoding (constraint not armed yet, or the document is still open)
    Continue,
    /// the JSON document is complete: the turn ends here ("stop")
    Complete,
    /// the token is not valid in the machine's state, or leaves no valid continuation: the
    /// turn ends with an error finish (`SCHEMA_DEAD_END`) and the event is logged
    Invalid,
}

/// The finish reason of a schema turn that reached a state no token can continue (or emitted a
/// token the state refuses): an error, so the client never reads the partial document as a
/// completed one.
pub(super) const SCHEMA_DEAD_END: &str = "error: json-schema constrained decoding reached a state with no valid continuation";

/// W2-EXL3: advance a schema lane's FSM by one emitted token (pure; unit-tested).
pub(super) fn schema_feed(sm: &crate::json_schema::SchemaMask, state: &mut u32, armed: &mut bool, tok: u32) -> SchemaStep {
    if !*armed {
        if sm.defer_until() == Some(tok) { *armed = true; }
        return SchemaStep::Continue;
    }
    match sm.step(*state, tok) {
        None => SchemaStep::Invalid,
        Some(ns) => {
            *state = ns;
            if sm.is_complete(ns) {
                SchemaStep::Complete
            } else if sm.allowed(ns).iter().all(|&w| w == 0) {
                // a state no token can continue (the machine should never reach one): end the
                // turn instead of letting a fully masked row pick id 0
                SchemaStep::Invalid
            } else {
                SchemaStep::Continue
            }
        }
    }
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

/// THINKBUDGET1: the token to emit for the model's token `t` — emission `j` of the `n` tokens of its
/// round — under the request's thinking budget (no budget = `t`). Counts and logs a forced end.
pub(super) fn think_emit(tb: Option<&mut crate::think_budget::ThinkBudget>, slot: usize, t: u32, j: usize, n: usize) -> u32 {
    let Some(tb) = tb else { return t };
    let ended_before = tb.forced;
    let (out, forced) = tb.filter(t);
    if forced && (j > 0 || n > 1) {
        // Unreachable by construction: round_depth clamps every round at the budget's room, so a
        // forced token is always the only token of a plain step. If it ever fires, the tokens after
        // it in this round were predicted from the replaced token, not from the marker.
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::reprintln!("[exl3-serve] thinking budget: the end marker was forced at token {j} of a \
{n}-token round (slot {slot}) — a scheduling bug (the round should have been clamped); please report it"));
    }
    if tb.forced > ended_before {
        crate::metrics::think_budget_forced();
        crate::dreprintln!("[exl3-serve] thinking budget slot={slot}: the reasoning reached {} tokens (budget {}); \
the server emitted the end-of-thinking marker", tb.forced_at.unwrap_or(0), tb.budget());
    }
    out
}

impl Lane {
    /// THINKBUDGET1: tokens this lane may emit before its thinking budget forces one (None = no limit).
    pub(super) fn think_room(&self) -> Option<usize> {
        self.think.as_ref().and_then(|t| t.room())
    }

    /// THINKBUDGET1: see think_emit.
    pub(super) fn think_filter(&mut self, slot: usize, t: u32, j: usize, n: usize) -> u32 {
        think_emit(self.think.as_mut(), slot, t, j, n)
    }

    pub(super) fn log_stats(&self, slot: usize, reason: &str) {
        if self.st_rounds == 0 && self.st_plain == 0 { return; }
        let r = self.st_rounds.max(1) as f64;
        crate::tel::add_spec(self.st_rounds as u64, self.st_drafted as u64, self.st_accepted as u64);
        crate::tel::add_mtp_reprobe(self.reprobe.off_events as u64, self.reprobe.reprobes as u64,
                                    self.reprobe.reprobe_ok as u64);
        self.log_wp23(slot);
        // S-B9: off_events/reprobes/reprobe_ok APPENDED (existing key order is frozen —
        // scripts parse it).
        // S-B9-CF-PACK4S: packed= APPENDED (lane rounds a packed verify served, of `rounds`); with
        // --round-trace also the request's seed and its per-round arm trace (lane_pack::ArmTrace:
        // the provenance of each emitted token — the sampled-lane distribution gate reads it).
        let trace = if round_trace_on() {
            format!(" seed={} arms={} toks={}", self.seed, self.arms.text(),
                    self.first_toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","))
        } else { String::new() };
        crate::dreprintln!("[mtp-stats] slot={slot} finish={reason} gen={} rounds={} drafted={} accepted={} ({:.1}%) \
                   tok/round={:.2} ms/round={:.1} plain_steps={} temp={} ctrl={} off_events={} reprobes={} reprobe_ok={} packed={}{trace}",
                  self.generated, self.st_rounds, self.st_drafted, self.st_accepted,
                  100.0 * self.st_accepted as f64 / self.st_drafted.max(1) as f64,
                  (self.st_accepted as f64 + self.st_rounds as f64) / r, self.st_ms / r,
                  self.st_plain, self.temperature, self.st_ctrl,
                  self.reprobe.off_events, self.reprobe.reprobes, self.reprobe.reprobe_ok, self.arms.packed);
        // WIDEVERIFY1: the copy rounds of this request (already inside rounds/drafted/accepted above)
        if self.copy.is_some() {
            let (g, c) = { let cs = self.copy.as_ref().unwrap(); (*cs.gate(), *cs.counters()) };
            let sw = cs_start_w_str(self.copy.as_ref().unwrap().cfg().start_w);
            crate::dreprintln!("[copy-span] slot={slot} finish={reason} copy_rounds={} copy_accepted={} copy_emitted={} \
                       neural_rounds={} gate_ec={:.2} gate_en={:.2} gate_nc={} gate_nn={} gate_denied={} gate_probes={} gate_suspended={} \
                       start_w={sw} narrow_search={} narrow_cursor={} zero_accept={}",
                       self.st_copy_rounds, self.st_copy_accepted, self.st_copy_emitted,
                       self.st_rounds - self.st_copy_rounds,
                       g.ec as f64 / crate::copy_span::YG_ONE as f64, g.en as f64 / crate::copy_span::YG_ONE as f64,
                       g.nc, g.nn, c.gate_denied, c.gate_probes, g.suspended() as u8,
                       c.narrow_search, c.narrow_cursor, c.rounds_zero_accepted);
        }
        // COPYPF1: the prefill-width copy verifies of this request (already inside rounds/drafted/accepted above)
        if self.st_cpv_steps > 0 {
            crate::dreprintln!("[copy-span-prefill] slot={slot} finish={reason} steps={} offered={} accepted={} emitted={} \
                       rerun_rows={} next_rung={}",
                       self.st_cpv_steps, self.st_cpv_offered, self.st_cpv_accepted, self.st_cpv_emitted,
                       self.st_cpv_rerun, self.cpv_next.0);
        }
    }

    /// WP02: the lane's client is gone (receiver dropped: disconnect, or a stop-string hit on the
    /// HTTP side) — its slot is freed now; always leaves a [mtp-stats] finish=cancelled line.
    pub(super) fn log_cancel(&self, slot: usize) {
        if self.st_rounds == 0 && self.st_plain == 0 {
            crate::dreprintln!("[mtp-stats] slot={slot} finish=cancelled gen={} rounds=0 plain_steps=0 temp={}",
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
        crate::dreprintln!("[dds] slot={slot} widths={:?} c_hat={:.2} ms/draft (fit rounds {}) target={:.3}",
                  g.widths, g.fit.slope(), g.fit.n, g.target);
    }

    /// Sampler params for `n` consecutive rows starting at the lane's counter (None = greedy).
    pub(super) fn samp_rows(&self, n: usize) -> Vec<Option<crate::exl3_forward::SampParams>> {
        (0..n).map(|i| crate::exl3_forward::SampParams::sampled(
            self.temperature, self.top_p, self.top_k, self.min_p, self.seed,
            self.ctr.wrapping_add(i as u32))).collect()
    }

    /// WP08: feed one emitted token (one that did not already end the request) to the loop
    /// detector — the reference implementation's job.py:1010-1013 order. True = end the response ("loop_detected").
    pub(super) fn loop_feed(&mut self, t: u32) -> bool {
        self.loop_det.as_mut().map_or(false, |d| d.feed(t))
    }

    /// WP08: one line per detection, with what a replay needs (a loop can be an engine bug:
    /// replay greedy + plain before blaming the model).
    pub(super) fn log_loop(&self, slot: usize, tok: &QwenTokenizer) {
        let Some(d) = self.loop_det.as_ref() else { return };
        let tail_ids = &self.history[self.history.len().saturating_sub(96)..];
        let text = tok.decode(tail_ids, false).unwrap_or_default();
        let n = text.chars().count();
        let tail: String = text.chars().skip(n.saturating_sub(200)).collect();
        crate::dreprintln!("[loop] detected slot={slot} period={} (detector {}) gen={} pos={} seed={} temp={} \
                   top_p={} top_k={} min_p={} pen={:?} mtp={} prompt_hash={:016x} tail={:?}",
                  d.fundamental_period().map_or("-".into(), |p| p.to_string()),
                  d.period().map_or("-".into(), |p| p.to_string()),
                  self.generated, self.pos, self.seed, self.temperature, self.top_p, self.top_k,
                  self.min_p, self.pen, self.mtp.is_some(), self.prompt_hash, tail);
    }
}
