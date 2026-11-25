//! The speculative round of one lane: `step_mtp` / `step_mtp_ex` (draft, verify, accept), `round_depth*`, the round accounting
//! (`round_account`), emission of accepted tokens (`round_emit*`, log-probs, stop info) and `learn_shared_ms`. Another `impl Exl3Scheduler` block.
//! Binding invariants: MTP semantics of AGENTS.md section 2 (item 6) and greedy MTP staying bitwise lossless (item 7).

use super::*;

impl Exl3Scheduler {
    /// The depth of a speculative round for a lane at position `p` with `remaining` tokens of
    /// budget: (after the end-of-window and remaining-budget clamps, after the sparse-switch
    /// clamp as well). The second is the round's depth; first > second = the round is clamped
    /// at the switch. One definition for `step_mtp` and the packed arm's plan.
    pub(super) fn round_depth(&self, p: usize, remaining: usize, think_room: Option<usize>) -> (usize, usize) {
        self.round_depth_for(self.mtp_k, p, remaining, think_room)
    }

    /// `round_depth` for a depth ceiling `depth` (the neural rounds: `self.mtp_k`; WIDEVERIFY1
    /// refinement 2: a wide copy round clamps COPY_MAX_K the same way — window, budget, thinking
    /// budget, then the sparse switch).
    pub(super) fn round_depth_for(&self, depth: usize, p: usize, remaining: usize, think_room: Option<usize>) -> (usize, usize) {
        let mut k_eff = depth.min(self.model.max_pos().saturating_sub(p + 2));
        if crate::exl3_forward::a_flags().rem_clamp {
            k_eff = k_eff.min(remaining.saturating_sub(1));
        }
        // THINKBUDGET1: a round never emits past the thinking budget (room tokens -> at most room-1
        // drafts + the correction); room 0 -> k_eff 0, a plain step, whose token the filter forces.
        // Unconditional (independent of --a-remclamp): forcing mid-round would desync the state.
        if let Some(r) = think_room {
            k_eff = k_eff.min(r.saturating_sub(1));
        }
        (k_eff, self.model.regime_depth(p, k_eff))
    }

    /// WP24: this lane's real-q draft sampler for its next MTP round — Some only for a SAMPLED
    /// lane (greedy stays on the bitwise match path) under --spec-sampling ratio (mode 2), or
    /// with --wp24-dump set (mode 1: today's drafts, top-32 lists recorded for rung 0).
    pub(super) fn rq_stage(&self, s: usize) -> Option<crate::exl3_forward::RqStage> {
        let lane = self.lanes[s].as_ref()?;
        crate::exl3_forward::RqStage::for_lane(self.spec_ratio, self.draft_temp, lane.temperature,
                                               lane.top_p, lane.top_k, lane.min_p, lane.seed, lane.dctr)
    }

    /// One chain-verify round for one lane, with the §6 profitability control:
    /// every 128th round runs PLAIN as the control and the MTP path auto-disables
    /// (loudly) if its per-token cost loses to plain.
    /// ISSUES[94] DF2: `solo` = exactly ONE lane was active in this scheduler step (the serial
    /// arms pass slots.len() == 1; the packed/pair/shared-refresh arms pass false). Only a
    /// solo round may run the DFlash2 chain — the DF2 ring follows ONE token stream, so a
    /// multi-lane step keeps the pack-MTP machinery — and only when the ring is current for
    /// this lane (`nprev == p`): the prefill primed it and every DF2 round advanced it, but a
    /// lane that prefilled while another lane held the ring is stale until its next prefill;
    /// those rounds fall back to the pack MTP and count on `exl3_df2_fallback_mtp_total`.
    pub(super) fn step_mtp(&mut self, s: usize, solo: bool) -> Result<()> {
        self.step_mtp_ex(s, solo, None)
    }

    /// COPYLANES: `step_mtp` with `busy_copy` = Some(the batch's rate, tokens/ms) for a lane `cl_step` took out of a busy
    /// step for its copy work: the lone path's copy proposal and stretch pick run for it as for a lone lane (`copy_ok`),
    /// a stretch must also pay for the batch; everything that keys on a lone STREAM (the DFlash2 ring) keeps `solo`.
    pub(super) fn step_mtp_ex(&mut self, s: usize, solo: bool, busy_copy: Option<f64>) -> Result<()> {
        let copy_ok = solo || busy_copy.is_some();
        crate::metrics::lane_pack_round(1); // arm = serial (a lone speculative lane round)
        let k = self.mtp_k;
        let af = crate::exl3_forward::a_flags();
        let head = self.model.mtp.as_ref().unwrap();
        let (b, p, tap_row, rows, pen, remaining, think_room, lane_cost) = {
            let lane = self.lanes[s].as_ref().unwrap();
            (lane.last_tok as i32, lane.pos, lane.mtp.unwrap(), lane.samp_rows(k + 1), lane.pen,
             lane.max_new.saturating_sub(lane.generated), lane.think_room(),
             if lane.ema_tok > 0.0 { lane.ema_ms / lane.ema_tok } else { 0.0 })
        };
        if crate::exl3_forward::DF2_TRACE.load(std::sync::atomic::Ordering::Relaxed) {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                let nv = self.df2.as_ref().map(|r| r.nprev());
                eprintln!("[df2trace] step_mtp s={s} solo={solo} p={p} nprev={nv:?} draft_conf={} df2_k={}",
                          self.draft_conf, self.df2_k);
            }
        }
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
        // (S-B9-CF-PACK4: the two clamps and the regime clamp below are `round_depth` — one
        // definition for this round and the packed arm's plan.)
        let (mut k_eff, kd) = self.round_depth(p, remaining, think_room);
        // S-B9-REL-QSAREGIME: a verify (and its draft chain) runs ONE attention regime, chosen from
        // its last row, so a round that straddles the sparse switch would put its rows at/below the
        // switch on the sparse kernels — not the bits a plain decode of those positions has (AGENTS
        // 2.4: column 0 of a verify is the decode's). Like the two clamps above, the round stops at
        // the switch: rows p..=qsa_limit only; at p == qsa_limit that is k_eff 0, a plain step. A
        // few short rounds once per request; a function of p alone (lockstep state under TP).
        {
            if kd < k_eff {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| crate::drprintln!("[exl3-serve] per-row attention regime: a speculative round at position {p} stops at the \
sparse switch (depth {k_eff} -> {kd}; --qsa-row-regime 0 = rounds may straddle it)"));
            }
            k_eff = kd;
        }
        // S-B9-REL-MTPREPROBE: a pending confirm forces ONE control round now (a fresh
        // plain sample at the CURRENT context) before an off may fire.
        let force_ctrl = {
            let l = self.lanes[s].as_mut().unwrap();
            std::mem::take(&mut l.force_control)
        };
        self.ctrl_round += 1;
        // WP03 cadence: 1/1024 while this request's MTP cost is clearly below plain (< 0.8x), else
        // 1/128. ema_plain stays context-local: it is refreshed only by control rounds at the
        // current context, never by a boot-time plain cost (under-priced at long context).
        let period = crate::lane_pack::ctrl_period(af.ctrl == 0, self.ema_plain, lane_cost);
        // (--a-ctrl off, gate diagnostics: no cadence control; see exl3_forward::AFlags)
        let control = (af.ctrl != 3 && self.ctrl_round % period == 0) || force_ctrl;
        let plain_step = control || k_eff == 0;
        // COPYPF1 (--copy-span-prefill, default off): an ESTABLISHED copy of a lone lane is verified by one chunked
        // prefill instead of a copy round (`cpv_on` false: nothing is called and every line below runs as before).
        // Control rounds and plain steps keep priority (cpv_pick returns None for them).
        if self.cpv_on {
            if let Some((n, rung)) = self.cpv_pick(s, copy_ok, plain_step, p, remaining, think_room, busy_copy) {
                return self.step_cpv(s, n, rung, b, p, tap_row);
            }
        }
        // WIDEVERIFY1: a solo speculative round first asks the lane's copy-span proposer (None when
        // --copy-span is off: no call, every line below runs as before). Control rounds and plain steps
        // keep priority (no proposal is asked for them).
        // WIDEVERIFY1 refinement 2: with the wide copy verify armed (and the scratch built for it) a copy
        // round's depth is COPY_MAX_K clamped exactly like k_eff (window, budget, thinking budget, the
        // sparse switch); otherwise it is k_eff, as before.
        let k_copy = match self.copy_arm {
            Some(arm) if copy_span_wide(arm.max_w) && self.sc.vrows >= crate::exl3_forward::COPY_VROWS && !plain_step && copy_ok =>
                self.round_depth_for(crate::exl3_forward::COPY_MAX_K, p, remaining, think_room).1,
            _ => k_eff,
        };
        let copy_prop = if !plain_step && copy_ok {
            Self::copy_propose(self.copy_arm, self.lanes[s].as_mut(), k_copy, self.record.as_deref_mut())
        } else { None };
        // G2: stage this lane's sampler rows (verify rows 0..k, or row 0 of a plain step).
        // WP24: a sampled lane under --spec-sampling ratio stages real-q rows + the draft sampler
        // (plain steps stay plain); None = exactly set_sampling (the match rule, today).
        // WIDEVERIFY1: a copy round stages the match rule (None) — a copied draft has no draft law.
        let rq = if plain_step || copy_prop.is_some() { None } else { self.rq_stage(s) };
        let rq_ratio = rq.map_or(false, |r| r.mode == 2);
        // FLASHNEXT-DF2-B 2026-10-23: the greedy-lane divert to pack MTP that lived here
        // (interim for ISSUES work[112]: a graphed greedy DF2 draft overflowed the band
        // kernel's dynamic smem on replay) is GONE — the real fix sizes the smem to the
        // window bound (dflash2::band_smem); greedy lanes take the DF2 round like every other.
        if let Some(cp) = copy_prop.as_ref() {
            // WIDEVERIFY1: sampler AND penalty rows at the copy round's own verify width (the DF2 arm's
            // pattern): row j's draw depends on (seed, ctr + j) only, so a narrower staging changes no row.
            let rows_c = self.lanes[s].as_ref().unwrap().samp_rows(cp.tokens.len() + 1);
            self.model.set_spec_sampling(&mut self.sc, &rows_c, None)?;
            self.model.set_penalties(&mut self.sc, &vec![pen; cp.tokens.len() + 1])?;
        } else {
        self.model.set_spec_sampling(&mut self.sc, if plain_step { &rows[..1] } else { &rows }, rq)?;
        // WP15: the same penalty params on every row of the round (and the draft mirror's row 0).
        self.model.set_penalties(&mut self.sc, &vec![pen; if plain_step { 1 } else { k + 1 }])?;
        }
        let t0 = std::time::Instant::now();
        let g0 = self.model.graph_count(); // WP23: a round that captures a graph is not cost-fit
        let mut df2_served = false;
        // EXL27D-reanchor (hoisted): a second request on the same server may start at a
        // position BEHIND the DF2 ring's frontier (its prefill came from the PREFIX
        // CACHE, so df2_prime_window never ran; df2_reprime only ADVANCES). Left alone,
        // every later request falls back to pack MTP forever (measured 2026-10-21:
        // request 2 = 0 DF2 rounds, 843 MTP drafts). Rewind the frontier to just PAST
        // this lane's position — `p`, not `p-1` — so the DF2 gate still misses THIS
        // round: the adaptive round that serves it refreshes the tap sink at the lane's
        // positions, and the Fix2 reprime advances the ring from there. The next round
        // is a DF2 round with fresh conditioning. Ring rows over a shared prefix are
        // exact (request 1 wrote them); a different prompt self-heals within a few
        // rounds. Counted on `exl3_df2_reanchor_total`.
        if !plain_step && solo && p >= 1
            && self.df2.as_ref().map_or(false, |r| r.nprev() + 1 > p)
        {
            // ISSUES 126: rewind to `p` so the gate (nprev == p, the trunk's invariant) matches THIS
            // round. Ring rows below p are the previous request's (exact over a shared prefix,
            // stale otherwise — ISSUES 114): that costs acceptance, never correctness (the verify
            // commits the target's own tokens). (The old one-behind convention rewound to p-1 and
            // admitted nprev+1 == p: the block then started one position early and the seam's tap
            // was never in context.)
            if let Some(r) = self.df2.as_mut() { r.rewind(p - 1); }
            df2_count_reanchor();
        }
        let (out, plain_ms, confs) = if plain_step {
            // plain round: consume b at p like a plain decode step. WP03: no draft pass runs, so
            // first write the head's row p KV-only (tap that predicted b, b), and after the step
            // snapshot its tap into row 0 of the lane's OWN window (sc.resid is shared: another
            // lane's round overwrites it before this lane's next round). --a-ctrl=old = neither.
            let ctrl_new = af.ctrl != 2;
            let tc = if ctrl_new {
                self.model.mtp_plain_prime(&mut self.sc, head, s, b, p, tap_row)?;
                self.model.dsync()?; // the control's plain-cost timing stays the step alone (NIGHT1 follow-up, PR #51: through host_wait)
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
        } else if let Some(cp) = copy_prop.as_ref() {
            // WIDEVERIFY1: the copy round — no draft pass; the copied ids ride the same chain verify
            // (exl3_forward::copy_round: head row p prime, verify, head re-prime of the accepted rows).
            // Width = the proposal's length <= copy_width(k_eff, --copy-span-max-w).
            let drafts: Vec<i32> = cp.tokens.iter().map(|&x| x as i32).collect();
            // TP: every rank computed this proposal locally from lockstep-agreed inputs; agree the width
            // and the copied ids NOW, before any launch of the round (the head prime has no collective;
            // the verify's width selects its barrier sequence). 0 head passes launched: a rank that ran a
            // neural round here instead (launched >= 1) can never pass this agree. The previous lockstep
            // op (tp_round fence B / step-go) is the rendezvous this exchange needs (pre_verify's shape).
            if let Some(t) = self.tp.as_mut() {
                if tp_preverify_on() {
                    t.pre_verify(drafts.len(), &drafts, 0)?;
                    CS_TP_AGREED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            (self.model.copy_round(&mut self.sc, head, s, b, p, tap_row, drafts, t0)?, 0.0, Vec::new())
        } else if solo && self.df2.as_ref().is_some_and(|r| r.nprev() + 1 == p) {
            // ISSUES[94] DF2 (Fix1, 2026-10-16): the DFlash2 chain round owns the solo
            // speculative step whenever it is armed and its ring is fresh — INCLUDING under
            // `draft_conf > 0`: the WP23 adaptive draft-stop is pack-MTP-head machinery (it
            // reads the MTP head's per-position confidences through a keep hook the DF2 block
            // drafter has no analog of), so the adaptive branch keeps serving only the lanes
            // DF2 cannot (multi-lane rounds, stale ring). The DF2 round keeps the §6
            // profitability control (plain control rounds still measure it) and returns the
            // same MtpRoundOut, so `round_account` / `round_emit` are the MTP path's. The
            // plain CONTROL round above is untouched — the DF2 economics are still measured
            // against it.
            // ISSUES 127: ONE width for the whole DF2 round — the scheduler's k_eff (remaining budget, end of the KV
            // window, the QSA dense/sparse switch: a round never straddles it) caps --df2-k like it caps the MTP depth.
            // k_eff >= 1 here (k_eff == 0 is a plain step).
            let k_df2 = self.df2_k.min(crate::exl3_forward::MTP_MAX_K).min(k_eff);
            // EXL27D-RQ (item 1): a sampled lane re-stages its sampler rows at the DF2
            // verify width (k_df2+1; the shared staging above sized the MTP depth k_eff —
            // a mismatch would stage fewer ratio rows than the verify judges) and drafts
            // through the sampled selector walk. sel_seeds = rng(seed, DF2_SEL, j) per
            // position — the trunk lane's exact convention (batch.rs df2_lane_step_sample_rq);
            // tau = the staged draft temperature (--draft-temperature or the request's).
            let rq2 = if rq_ratio { self.rq_stage(s) } else { None };
            // ISSUES 127: sampler AND penalty rows at the DF2 verify width for greedy lanes too (they were staged at the
            // MTP depth k+1 above, so a wider DF2 verify judged its extra rows with stale rows).
            let (rows_df2, seed) = {
                let l = self.lanes[s].as_ref().unwrap();
                (l.samp_rows(k_df2 + 1), l.seed)
            };
            self.model.set_spec_sampling(&mut self.sc, &rows_df2, if rq_ratio { rq2 } else { rq })?;
            self.model.set_penalties(&mut self.sc, &vec![pen; k_df2 + 1])?;
            if rq_ratio {
                let seeds: Vec<u32> = (0..crate::dflash2::levels())
                    .map(|j| crate::batch::rng_u32(seed, crate::batch::RNG_DOM_DF2_SEL, j))
                    .collect();
                let tau = rq2.map(|r| r.tau).unwrap_or(1.0);
                // FLASHNEXT-DF2-117 (ISSUES 117): ONE sampled draft per round on every rank. Only
                // the head (or a single box) runs the selector walk; under TP the head ships its
                // tokens + candidate table to every node and the nodes adopt them instead of
                // drawing their own — a replicated walk over residuals that differ across ranks
                // in the low bits drafted differently and tripped the lockstep agree (TP=2 within
                // ~600 rounds, TP=4 after ~10k tokens on the owner's traffic). Greedy drafts never
                // needed this (a graph replay whose top-1 agreed over thousands of rounds).
                let draws_here = self.tp.as_ref().map_or(true, |t| t.rank == 0 || !t.has_ctl());
                let mine = if draws_here {
                    Some(self.model.df2_draft_sample(self.df2.as_mut().unwrap(), b as u32, &seeds, tau)?)
                } else { None };
                let sample = match self.tp.as_mut() {
                    Some(t) => match t.df2_draft(s, p, mine.as_ref())? {
                        Some(theirs) => theirs,
                        None => mine.expect("the head keeps the draft it shipped"),
                    },
                    None => mine.expect("single-box: the local draw"),
                };
                {
                    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if let Some(t) = self.tp.as_ref() {
                        if t.has_ctl() && !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            crate::reprintln!("[exl3-serve] df2: DF2 drafts are head-authoritative under TP={} (greedy and sampled: the head's draft ships to every node per round; exl3_df2_shipped_drafts_total counts them — ISSUES 117)", t.world);
                        }
                    }
                }
                let o = self.model.df2_round(&mut self.sc, self.df2.as_mut().unwrap(), s, k_df2, b, p,
                                             Some((seeds.as_slice(), tau)), Some(sample),
                                             [t0, t0, t0])?;
                df2_count_round(o.a);
                df2_served = true;
                (o, 0.0, Vec::new())
            } else {
                // FLASHNEXT-DF2-117 (greedy twin): the greedy top-1 flips across ranks on near-ties as
                // well (TP=2 proof: one greedy round in ~1,500 after the sampled rounds were fixed), so
                // the greedy draft is head-authoritative too — tokens only, no candidate table.
                let draws_here = self.tp.as_ref().map_or(true, |t| t.rank == 0 || !t.has_ctl());
                let mine = if draws_here {
                    Some(crate::dflash2::round::Df2SampleOut {
                        tokens: self.model.df2_draft_greedy(self.df2.as_mut().unwrap(), b as u32)?,
                        q_rows: Vec::new(), cand_tok: Vec::new(), cand_q: Vec::new(),
                    })
                } else { None };
                let sample = match self.tp.as_mut() {
                    Some(t) => match t.df2_draft(s, p, mine.as_ref())? {
                        Some(theirs) => theirs,
                        None => mine.expect("the head keeps the draft it shipped"),
                    },
                    None => mine.expect("single-box: the local draw"),
                };
                let o = self.model.df2_round(&mut self.sc, self.df2.as_mut().unwrap(), s, k_df2, b, p,
                                             None, Some(sample), [t0, t0, t0])?;
                df2_count_round(o.a);
                df2_served = true;
                (o, 0.0, Vec::new())
            }
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
            if solo && self.df2.is_some() {
                // DF2 engaged but the ring is stale for this lane (`nprev != p`: the lane
                // prefilled while another lane held the chain, or the artifact never primed
                // this window): the pack MTP serves the round and the fallback is COUNTED —
                // `exl3_df2_fallback_mtp_total` — never a silent skip.
                df2_count_fallback_mtp();
            }
            (self.model.mtp_round(&mut self.sc, head, s, k_eff, b, p, tap_row)?, 0.0, Vec::new())
        };
        // EXL27C-DF2-Fix2: after any NON-DF2 solo round (plain/control, adaptive or MTP
        // fallback), re-prime the DF2 ring from the taps that round's target forward captured
        // (every solo forward writes the staging sink), so the next solo round is DF2 again
        // instead of falling back forever. `df2_round` advanced the ring for its own rounds.
        if solo && self.df2.is_some() && !df2_served {
            if let Some(r) = self.df2.as_mut() {
                if let Err(e) = self.model.df2_reprime(r, out.a + 1) {
                    eprintln!("[exl3-serve] df2 reprime after fallback round failed: {e}");
                }
            }
        }
        // WP24: the dump / XCHECK host diagnostics are not round cost (they would trip the §6
        // per-request auto-off on a diagnostics run); 0 when off.
        let ms = (t0.elapsed().as_secs_f64() * 1e3 - out.diag_ms).max(0.0);
        // TP-C: lockstep — agree on (step, accept, verify width, emitted + drafts + confidences) and
        // take the HEAD's ms / plain ms / capture flag (identity at TP=1: no TP state).
        let captured_local = self.model.graph_count() != g0;
        // MTPAGREE1: --round-trace prints every speculative round's view on EVERY rank (and at TP=1), so two
        // runs or two ranks can be diffed field by field: tokens vs the per-position confidences (the f32
        // bits the TP mtp-round digest also hashes).
        if round_trace_on() {
            let r = self.tp.as_ref().map_or(0, |t| t.rank);
            eprintln!("[round-trace] rank={r} s={s} p={p} plain={plain_step} k_eff={k_eff} a={} drafts={:?} emitted={:?} confs_bits={:?}",
                      out.a, out.drafts, out.emitted, confs.iter().map(|c| format!("{:08x}", c.to_bits())).collect::<Vec<_>>());
        }
        let (ms, plain_ms, captured) = if self.tp.is_some() {
            let mut w: Vec<u32> = out.emitted.iter().map(|&x| x as u32).collect();
            w.extend(out.drafts.iter().map(|&x| x as u32));
            w.extend(confs.iter().map(|c| c.to_bits()));
            let m = if plain_step { 1 } else { out.drafts.len() + 1 };
            let what = if plain_step { "plain-round" } else if copy_prop.is_some() { "copy-round" } else { "mtp-round" };
            match self.tp_round(what, out.a, m, &w, ms, plain_ms, captured_local, false) {
                Ok(t) => t,
                Err(e) => {
                    // P7-TP2 bisect: both ranks print their round view at the mismatch; diff the
                    // logs to see whether the chain diverges at draft[0] (input/ring) or mid-chain.
                    eprintln!("[df2-tp2] p={:?} plain={} accept={} drafts={:?} emitted={:?} confs={:?}",
                              p, plain_step, out.a, out.drafts, out.emitted, confs);
                    // MTPAGREE1 (public v0.7.6 TP=2 report): the lane's sampler state next to the round
                    // view, so a rank diff names the subsystem (seed/counters = RNG stream; pen = the
                    // penalty rows; history tail = the committed stream the two ranks hold).
                    if let Some(lane) = self.lanes[s].as_ref() {
                        let ht: Vec<u32> = lane.history.iter().rev().take(8).copied().collect();
                        eprintln!("[df2-tp2] lane s={s} seed={} ctr={} dctr={} temp={} top_p={} top_k={} min_p={} pen={:?} \
                                   spec_ratio={} k_eff={} tap_row={} b={} pos={} generated={} history_tail(rev)={:?}",
                                  lane.seed, lane.ctr, lane.dctr, lane.temperature, lane.top_p, lane.top_k,
                                  lane.min_p, lane.pen, self.spec_ratio, k_eff, tap_row, b, lane.pos, lane.generated, ht);
                    }
                    return Err(e);
                }
            }
        } else {
            (ms, plain_ms, captured_local)
        };
        // S-B9-CF-PACK4: the round's bookkeeping and its emit loop are `round_account` /
        // `round_emit` (moved there unchanged, in order) so the packed arm runs the same code
        // per request.
        self.round_account(s, &out, ms, plain_ms, captured, control, plain_step, rq_ratio, xcheck, false,
                           copy_prop.is_some())?;
        self.round_emit(s, &out, None)?;
        // WIDEVERIFY1 yield gate: a NEURAL speculative round (not a plain/control step, not a copy round)
        // reports its emitted count to the lane's proposer — the ONE call site of copy_note_neural; copy
        // rounds are counted by the proposer's commit below. `out.emitted` is lockstep-agreed under TP
        // (the tp_round digest above), so every rank's gate holds the same state.
        if !plain_step && copy_prop.is_none() {
            if let Some(lane) = self.lanes[s].as_mut() { Self::copy_note_neural(&mut lane.copy, out.emitted.len()); }
        }
        // WIDEVERIFY1: commit the copy round to its proposer with exactly the tokens round_emit emitted
        // (a finished lane is gone: nothing to commit). Other rounds commit lazily (copy_propose).
        if let Some(cp) = copy_prop.as_ref() {
            // COPYLANES: every copy round prices the busy-step copy work (ms = the head's under TP; accept counts agreed)
            if self.cl_on && !plain_step {
                let wide = cp.tokens.len() > crate::exl3_forward::MTP_MAX_K;
                self.cl_price.note_copy_round(ms, cp.tokens.len(), out.a, cp.source == crate::copy_span::Source::Cursor, wide);
                self.cl_price.work.note(out.emitted.len(), ms);
                if let Some(lane) = self.lanes[s].as_mut() { lane.cl_work.note(out.emitted.len(), ms); }
            }
            // ECHOPRICE: a fully accepted copy round prices the copy path (ms = the head's under TP)
            if self.cpv_on && !plain_step && !cp.tokens.is_empty() && out.a >= cp.tokens.len() {
                self.cpv_price.note_copy_round(ms, cp.max_w);
            }
            if let Some(lane) = self.lanes[s].as_mut() {
                if let Some(cs) = lane.copy.as_mut() {
                    let th = std::time::Instant::now();
                    let before = *cs.counters();
                    cs.commit(Some(cp), out.a, &lane.copy_new);
                    copy_span_count_delta(&before, cs.counters());
                    lane.copy_new.clear();
                    CS_HOST_US.fetch_add(th.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }

    /// One lane's round bookkeeping, exactly as the serial arm does it after its lockstep point:
    /// the tap cross-check record, the sampler counters, the per-request profitability control
    /// (cost EMAs, re-probe policy, control samples) and the WP23 width counts + cost fit.
    /// `ms` = this lane's round time (a packed round passes its SHARE, lane_pack::lane_share_ms).
    /// `packed` = a packed verify served the round (the request's arm trace records it).
    /// WIDEVERIFY1 `copy` = a copy-span round (drafts copied from the context, no draft pass). It
    /// touches: the wall-latency EMA (ema_wall_ms: real round-to-round latency), the tap cross-check,
    /// the arm trace ('S', a serial speculative round), the sampler counter `ctr`, and the request's
    /// round anatomy st_rounds / st_drafted / st_accepted / st_ms (the generic velogb10_spec_* series,
    /// "all speculative rounds") plus its own st_copy_* and velogb10_copy_span_* counters. It does NOT
    /// touch the neural drafter's statistics: the serial-arm delivery rate (arm_acc), the WP24 gate
    /// tally, mtp_rounds, the cost EMAs ema_ms / ema_tok / ema_n, the re-probe / §6 auto-off policy,
    /// the WP23 width counts and cost fit, the DDS calibrator (step_mtp feeds it from the adaptive arm
    /// only) or ema_plain (control rounds only) — a copy round's cost and yield would distort the
    /// depth / on-off decisions about the neural drafter.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn round_account(&mut self, s: usize, out: &crate::exl3_forward::MtpRoundOut, ms: f64, plain_ms: f64,
                     captured: bool, control: bool, plain_step: bool, rq_ratio: bool, xcheck: bool, packed: bool,
                     copy: bool) -> Result<()> {
        let k = self.mtp_k;
        // P18 SPEC8K: sample the real round-to-round latency of the lane (queueing included) for the
        // load-adaptive choice of arm. Once per lane per round, here, whichever arm the round ran
        // (serial speculation, a packed pair, a shared plain step): this is the single point that every
        // arm reaches with the ms of its own round.
        if !captured {
            let now = std::time::Instant::now();
            let lane = self.lanes[s].as_mut().unwrap();
            if let Some(prev) = lane.last_wall {
                let wall = (now - prev).as_secs_f64() * 1e3;
                if wall > 0.0 && wall <= WALL_STALL_X * ms.max(1.0) {
                    lane.ema_wall_ms = if lane.ema_wall_ms <= 0.0 { wall } else {
                        WALL_EMA_W * lane.ema_wall_ms + (1.0 - WALL_EMA_W) * wall
                    };
                }
            }
            lane.last_wall = Some(now);
        }
        // S-C4XOVER: every round_account call is a serial-arm round (a lone speculative
        // round, a packed pair's lane share, or its plain control): fold its delivery
        // (emitted tokens, round ms) into the arm's measured rate.
        if !copy { self.arm_acc(ARM_SERIAL, out.emitted.len() as f64, ms); }
        if xcheck {
            self.model.tap_xcheck_record(&self.sc, s, out.tap_row)?;
        }
        // G2: advance the lane's sampler counter past every row this round consumed.
        {
            let lane = self.lanes[s].as_mut().unwrap();
            lane.arms.push(if packed { 'P' } else if control { 'c' } else if plain_step { 'p' } else { 'S' }, out.emitted.len());
            // WIDEVERIFY1 refinement 2: a wide copy round consumes drafts+1 > k+1 rows; never hand a
            // row counter it used to the next round (max is a no-op for every round of <= k drafts)
            lane.ctr = lane.ctr.wrapping_add(if plain_step { 1 } else { (k + 1).max(out.drafts.len() + 1) as u32 });
            // WP24: past every draft counter this round could use (passes 0..k_eff-1 incl. a
            // HOST speculative pass the DDS then stopped before; k_eff <= k)
            if rq_ratio { lane.dctr = lane.dctr.wrapping_add(k as u32); }
            // WP24 gate bookkeeping (None in the server): round anatomy per arm + the request's
            // first-round draft d_0 (the negative control: d_0 ~ q', a law != p)
            if let Some(g) = self.gate.as_mut().filter(|_| !copy) {
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
            if let Some(lane) = self.lanes[s].as_mut() {
                lane.st_ctrl += 1;
                // S-B9: a pending confirm collects this fresh sample (current context)
                if lane.reprobe.confirm_pending {
                    lane.reprobe.confirm_sample(plain_ms);
                }
            }
        } else if plain_step {
            // WP02: the window/budget-clamped last step (k_eff == 0) — a plain step, not a round
            if let Some(lane) = self.lanes[s].as_mut() { lane.st_plain += 1; }
        } else if copy {
            // WIDEVERIFY1: a copy round — the request's round anatomy and the copy counters only
            let lane = self.lanes[s].as_mut().unwrap();
            lane.st_rounds += 1;
            lane.st_drafted += out.drafts.len();
            lane.st_accepted += out.a;
            lane.st_ms += ms;
            lane.st_copy_rounds += 1;
            lane.st_copy_accepted += out.a;
            lane.st_copy_emitted += out.emitted.len();
            copy_span_count_round(out.drafts.len(), out.a, out.emitted.len(), ms);
        } else {
            let ema_plain = self.ema_plain;
            let reprobe_on = self.reprobe_on;
            let cfg = self.reprobe_cfg;
            // legacy mode keeps today's estimator weight so --mtp-reprobe=off is exact
            let ema_w = if reprobe_on { cfg.ema_w } else { 1.0 / 16.0 };
            let lane = self.lanes[s].as_mut().unwrap();
            lane.mtp_rounds += 1;
            lane.st_rounds += 1;
            lane.st_drafted += out.drafts.len();
            lane.st_accepted += out.a;
            lane.st_ms += ms;
            // The first `warmup` rounds of a slot can include graph capture; keep them
            // out of the cost. --mtp-reprobe=off keeps the exact legacy estimator
            // (one-round seed + 1/16 EMA, hardcoded 4-round guard: bit-for-bit the
            // pre-branch behaviour). The re-probe policy blends via
            // mtp_reprobe::blend_cost_ema (W1.1): a RUNNING MEAN over the first 64
            // post-reset rounds and capture rounds excluded entirely — one
            // capture-poisoned or 1-token round must not steer a probe verdict
            // (at 1/64 a seed still carries 0.79-0.62 of itself at rounds 16-31).
            if reprobe_on {
                if lane.mtp_rounds > cfg.warmup {
                    crate::mtp_reprobe::blend_cost_ema(
                        &mut lane.ema_ms, &mut lane.ema_tok, &mut lane.ema_n,
                        ms, toks, captured, (cfg.ema_w.recip()) as usize, cfg.ema_w);
                }
            } else if lane.mtp_rounds > 4 {
                if lane.ema_tok <= 0.0 {
                    lane.ema_ms = ms;
                    lane.ema_tok = toks;
                } else {
                    lane.ema_ms = (1.0 - ema_w) * lane.ema_ms + ema_w * ms;
                    lane.ema_tok = (1.0 - ema_w) * lane.ema_tok + ema_w * toks;
                }
            }
            let cost = if lane.ema_tok > 0.0 { lane.ema_ms / lane.ema_tok } else { 0.0 };
            if !reprobe_on {
                // --mtp-reprobe=off: the previous one-way rule, exactly (2% margin, no confirm,
                // no re-arm — the frozen pre-fix behaviour for A/B and for users who want it).
                if lane.mtp_rounds >= 64 && ema_plain > 0.0 && cost > ema_plain * 1.02 {
                    crate::dreprintln!("[exl3-serve] MTP off for this request (slot {s}): {:.2} ms/tok ({:.1} ms/round, \
                           {:.2} tok/round) vs plain {:.2} after {} rounds (AGENTS §6, per-request)",
                           cost, lane.ema_ms, lane.ema_tok, ema_plain, lane.mtp_rounds);
                    // the lane is at a clean seam (round committed, last_tok unconsumed at
                    // lane.pos — the same state a control round leaves), so plain can take over.
                    lane.mtp = None;
                }
            } else {
                // S-B9-REL-MTPREPROBE: the §6 decision is the pure policy's (margin 10%, a
                // 1/64-round window that spans local troughs, and a plain baseline that must
                // survive a fresh confirm at the CURRENT context — see src/mtp_reprobe.rs).
                // The round left the lane at a clean seam (last_tok unconsumed at lane.pos,
                // the round's tap saved in out.tap_row), so an off can prime from here.
                use crate::mtp_reprobe::RoundVerdict as RV;
                let plain_ref = lane.reprobe.plain_reference(ema_plain);
                match lane.reprobe.observe_round(cost, ema_plain, &cfg) {
                    RV::Continue => {}
                    RV::ConfirmControl => {
                        // crossed the margin: demand ONE fresh plain sample before deciding
                        lane.force_control = true;
                    }
                    RV::Off => {
                        crate::dreprintln!("[exl3-serve] MTP off for this request (slot {s}): {:.2} ms/tok ({:.1} ms/round, \
{:.2} tok/round) vs plain {:.2} after {} rounds (AGENTS §6; re-probe in {} plain steps)",
                                   cost, lane.ema_ms, lane.ema_tok, plain_ref, lane.mtp_rounds, cfg.backoff_base);
                        lane.mtp = None;
                        lane.reprobe.went_off(out.tap_row, false, &cfg);
                    }
                    RV::ProbeFailed => {
                        let next_b = (lane.reprobe.backoff.saturating_mul(2)).min(cfg.backoff_cap).max(cfg.backoff_base);
                        crate::reprintln!("[exl3-serve] MTP re-probe FAILED (slot {s}): {:.2} ms/tok vs plain {:.2} after {} \
probe rounds; off again (next probe in {next_b} plain steps)", cost, plain_ref, lane.reprobe.probe_rounds);
                        lane.mtp = None;
                        lane.reprobe.went_off(out.tap_row, true, &cfg);
                    }
                    RV::ProbeSucceeded => {
                        crate::dreprintln!("[exl3-serve] MTP re-probe OK (slot {s}): stayed on {} probe rounds at {:.2} \
ms/tok vs plain {:.2}; backoff reset to {}", cfg.probe_window, cost, plain_ref, cfg.backoff_base);
                        lane.reprobe.probe_ok(&cfg);
                    }
                }
            }
        }
        // WP23: rounds by verified draft width, and the cost fit (ms on drafts) — MTP rounds of a
        // DDS lane past the EMAs' 4-round warm-up, minus any round that captured a graph (a lazy
        // first-use capture) or ran > 3x the lane's mean round (a host stall): outliers, not cost.
        if !plain_step && !copy {
            let dds = self.draft_conf > 0.0;
            if let Some(lane) = self.lanes[s].as_mut() {
                let w = out.drafts.len();
                if let Some(c) = lane.wp23.widths.get_mut(w) { *c += 1; }
                if dds && lane.mtp_rounds > 4 && !captured && (lane.ema_ms <= 0.0 || ms <= 3.0 * lane.ema_ms) {
                    lane.wp23.fit.add(w as f64, ms);
                }
            }
        }
        Ok(())
    }

    /// Emit one round's tokens for the lane in slot `s` (stop at EOS / max_new / a loop / a gone
    /// client mid-list; a finishing lane frees its slot). `packed_row0` = Some(the lane's first
    /// row in the packed verify) when the round's logits rows live in the packed scratch.
    pub(super) fn round_emit(&mut self, s: usize, out: &crate::exl3_forward::MtpRoundOut, packed_row0: Option<usize>) -> Result<()> {
        self.round_emit_rows(s, out, packed_row0, true)
    }

    /// `round_emit` with `row_info` = the round's logits rows are still in the scratch (every verify and plain
    /// step). COPYPF1: a prefill-width copy verify passes false — its rows were judged in groups and are gone, so
    /// it reads none back (no stop_p line; logprob lanes never take that step).
    pub(super) fn round_emit_rows(&mut self, s: usize, out: &crate::exl3_forward::MtpRoundOut, packed_row0: Option<usize>,
                       row_info: bool) -> Result<()> {
        // PR #30 + speculation (owner 2026-11-02): the log probabilities of a speculative round's tokens. Emitted token j
        // is the choice of logits row r0 + j (item 5(a) below: still in sc.logits — the re-prime after a verify is
        // KV-only), so the round's rows are read back ONCE and reduced in parallel before the tokens go out. TP=1 only
        // (under TP no rank holds a full greedy row; the HTTP layer serves such requests with logprobs null).
        let r0 = packed_row0.unwrap_or(0);
        let lp_n = self.lanes[s].as_ref().and_then(|l| l.logprobs).filter(|_| row_info && self.lp_reader());
        let lps = lp_n.map(|n| self.round_logprobs(packed_row0.is_some(), r0, &out.emitted, n));
        // ---- emit the round's tokens (stop at EOS / max_new mid-list)
        for (j, &t32) in out.emitted.iter().enumerate() {
            // THINKBUDGET1: the clamp makes a forced token the first (and only) token of its round
            let t = self.lanes[s].as_mut().unwrap().think_filter(s, t32 as u32, j, out.emitted.len());
            if let (Some(n), Some(lp)) = (lp_n, lps.as_ref()) {
                let tx = self.lanes[s].as_ref().unwrap().tx.clone();
                if t == t32 as u32 {
                    if let Some(Some((logprob, top))) = lp.get(j) {
                        let _ = tx.send(crate::batch::TokEvent::Logprob { logprob: *logprob, top: top.clone() });
                    }
                } else if self.lp_reader() {
                    // logprobs of a token the thinking budget forced (not the row's own choice): its probability under that row
                    self.send_logprob(&tx, packed_row0.is_some(), r0 + j, t, n);
                }
            }
            let (finish, reason, generated) = {
                let lane = self.lanes[s].as_mut().unwrap();
                lane.last_tok = t;
                lane.pos += 1;
                lane.generated += 1;
                crate::lane_wide::note_token();
                lane.history.push(t);
                if lane.copy.is_some() { lane.copy_new.push(t); } // WIDEVERIFY1
                if let Some(r) = lane.rec_reply.as_mut() { r.push(t); } // COPYREC1
                if lane.first_toks.len() < FIRST_TOKS { lane.first_toks.push(t); }
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
                if let Some(lane) = self.take_lane(s, false) {
                    lane.log_cancel(s);
                }
                return Ok(());
            }
            if finish {
                self.publish_kv(); // issue #24: the lane's final position (the peaks include the round that ends the request)
                // item 5(a): emitted[j] is the choice of the round's logits row j (verify row j, or
                // row 0 of a plain step) — still in sc.logits: the re-prime after a verify is KV-only
                if reason == "stop" && row_info { self.send_stop_info(s, packed_row0.map_or(j, |r0| r0 + j), t, packed_row0.is_some()); }
                if let Some(lane) = self.take_lane(s, crate::lane_reserve::normal_finish(&reason)) {
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

    /// A speculative round's log probabilities: rows `r0 .. r0 + toks.len()` read back in one copy, each emitted token's
    /// row reduced on its own thread (token_logprobs is ~0.9 ms per 248K-entry row). None per token when the readback
    /// fails or a row is not finite (the handler then has no entry for that token) — it never stops the request.
    fn round_logprobs(&self, packed: bool, r0: usize, toks: &[i32], n: u8) -> Vec<Option<(f32, Vec<(u32, f32)>)>> {
        let v = self.model.cfg.vocab_size;
        // a packed round's rows are in the packed scratch (row0 + j), like send_stop_info's
        let sc = match (packed, self.scp.as_ref()) { (true, Some(scp)) => scp, _ => &self.sc };
        let rows = match self.model.logits_rows(sc, r0, toks.len()) {
            Ok(r) => r,
            Err(e) => {
                crate::reprintln!("[exl3-serve] logprob readback failed (rows {r0}..{}): {e:#}", r0 + toks.len());
                return vec![None; toks.len()];
            }
        };
        crate::metrics::logprobs_spec_rows(toks.len() as u64);
        std::thread::scope(|sc| {
            let hs: Vec<_> = toks.iter().enumerate()
                .map(|(j, &t)| { let r = &rows[j * v..(j + 1) * v]; sc.spawn(move || token_logprobs(r, t as u32, n as usize)) })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or(None)).collect()
        })
    }

    /// The log probability of `tok` (the choice of the plain step's logits row `row` in
    /// `self.sc`) and the `n` most likely alternatives, sent as a `TokEvent::Logprob` just before
    /// the token itself. A failed readback sends nothing (the handler then has no entry for that
    /// token and reports the gap) — it never stops the request.
    pub(super) fn send_logprob(&self, tx: &crate::server::TokTx, packed: bool, row: usize, tok: u32, n: u8) {
        let sc = match (packed, self.scp.as_ref()) { (true, Some(scp)) => scp, _ => &self.sc };
        match self.model.logits_row(sc, row) {
            Ok(r) => match token_logprobs(&r, tok, n as usize) {
                Some((logprob, top)) => { let _ = tx.send(TokEvent::Logprob { logprob, top }); }
                None => crate::dreprintln!("[exl3-serve] logprob unavailable (row {row}, token {tok}): non-finite logits"),
            },
            Err(e) => crate::reprintln!("[exl3-serve] logprob readback failed (row {row}): {e:#}"),
        }
    }

    /// S-B9-REL-SMALLFIXES item 5(a): the lane in slot `s` is finishing on the stop token `tok`,
    /// the choice of logits row `row` of the step that just ran. Reads that ONE row back and sends
    /// the token's probability and rank ahead of the Finish (`stop_p` / `stop_rank` of the HTTP
    /// side's `[gen]` line). Cost: one host readback of vocab f16 values plus one host pass over
    /// them, on a request's FINISHING step and only when it ends on a stop token; no step of a
    /// running request pays anything. TP=1 only: under TP the greedy head is vocab-parallel and no
    /// rank holds a full logits row (the line says `na`). A greedy row checks itself — its token
    /// must be the row's top logit (`stop_info_checked`); if it is not, the row read is not the row
    /// that chose the token, nothing is sent and one line per boot says so.
    pub(super) fn send_stop_info(&self, s: usize, row: usize, tok: u32, packed: bool) {
        if self.tp.is_some() { return; }
        let Some(lane) = self.lanes[s].as_ref() else { return };
        let samp = crate::exl3_forward::SampParams::sampled(lane.temperature, lane.top_p, lane.top_k, lane.min_p, 0, 0);
        // S-B9-CF-PACK4: a packed round's logits rows are in the packed scratch (row = row0 + j)
        let sc = match (packed, self.scp.as_ref()) { (true, Some(scp)) => scp, _ => &self.sc };
        let stats = match self.model.logits_row(sc, row) {
            Ok(r) => crate::exl3_forward::stop_token_stats(&r, tok as usize, samp.as_ref()),
            Err(_) => None,
        };
        match stop_info_checked(stats, samp.is_none()) {
            Some((p, rank)) => { let _ = lane.tx.send(TokEvent::StopInfo { p: p as f32, rank, sampled: samp.is_some() }); }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| crate::drprintln!("[exl3-serve] stop_p unavailable for a request (slot {s}, logits row {row}, token {tok}, {:?}): \
the [gen] line says stop_p=na", stats));
            }
        }
    }

    /// Plain batched decode for the given slots (the pre-f-d step() body).
    /// CF-P1e: fold a measured shared-round time (ms, lockstep-adopted under TP) into the model's learned scale.
    /// The first rounds of a width can include a graph capture, so an outlier above 3x the model is ignored.
    pub(super) fn learn_shared_ms(&mut self, n: usize, ms: f64) {
        if n < 2 { return; }
        let r = ms / shared_step_prior_ms(n);
        if !(0.05..3.0).contains(&r) { return; }
        self.shared_scale = if self.shared_streak <= 1 && self.shared_scale == 1.0 { r } else { 0.9 * self.shared_scale + 0.1 * r };
    }
}
