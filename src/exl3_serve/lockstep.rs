//! The TP lockstep layer of the EXL3 server: `WaitProf` and `TpSync` (fence A / exchange / fence B, the world > 2 merged head-hub op),
//! the agree-field packing and verdicts, the lane-width and max-batch resolvers, `fnv32`, and the fatal-error helpers.
//! Load-bearing: the per-step agree guard and the watchdog must not be bypassed, and every verify width must stay distinguishable in the agree word (AGENTS.md section 2, item 10).

use super::*;

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

impl WaitProf {
    #[inline]
    pub(super) fn rec(&mut self, site: usize, t0: std::time::Instant, s0: u64) {
        let ns = t0.elapsed().as_nanos() as u64;
        self.n[site] += 1;
        self.sum_ns[site] += ns;
        self.max_ns[site] = self.max_ns[site].max(ns);
        let us = ns / 1000;
        let b = WP_EDGES_US.iter().position(|&e| us < e).unwrap_or(WP_EDGES_US.len());
        self.hist[site][b] += 1;
        self.sleeps[site] += crate::net::wait_sleeps().saturating_sub(s0);
    }

    pub(super) fn line(&self, rank: i32) -> String {
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

impl TpSync {
    pub(crate) fn new(rank: i32, world: i32) -> Self {
        let deadline_ms = if world > 2 { lockstep_deadline_ms() } else { 0 };
        // S-B9-CF-AGREE16B (W3): the drill is armed (or declared inert) HERE, once, with a boot line
        // either way; `net::drill_plan` is the pure decision.
        let plan = crate::net::drill_plan(crate::opts::get(crate::opt!("tp-agree-drill")).as_deref(),
                                          crate::opts::get(crate::opt!("tp-width-drill")).as_deref(), true, Some((rank, world)));
        for l in &plan.lines { crate::dreprintln!("{l}"); }
        TpSync { rank, world, deadline_ms, epoch_warns: 0, step: 0, ident: false, ident_bufs: 0, ident_bad: 0, agrees: 0,
                 pre_agrees: 0, ctl: Vec::new(), wp: WaitProf::default(), drill_width: plan.width, drill_hits: 0 }
    }

    /// The verify width this rank REPORTS in a round's lockstep words: the real one, except ONCE under the
    /// `--tp-width-drill N` test drill, where the first round of width N is reported as `N - 1 - rank` on
    /// the box that carries the flag (on the head, N = 16: "this rank believes 15, the peer 16"). The
    /// forward itself is untouched: only the reported word lies, so a guard that cannot tell the two widths
    /// apart lets the drill pass in silence and one that can aborts the link at that round.
    ///
    /// S-B9-CF-AGREE16B: it was `--tp-agree-drill N`, it lied on EVERY N-row round, and every rank said
    /// N - 1. Two armed ranks then agreed on the lie, and a real 15-against-16 disagreement stayed masked
    /// for the whole boot. One shot bounds that to one round, and the rank term makes two armed ranks
    /// report different widths (`net::drill_plan` arms only where `N > rank`, so this cannot underflow).
    pub(super) fn width_said(&mut self, m: usize) -> usize {
        match self.drill_width {
            Some(n) if n == m && self.drill_hits == 0 && m > self.rank as usize => {
                self.drill_hits = 1;
                let said = m - 1 - self.rank as usize;
                crate::deprintln!("[tp-agree] DRILL (one-shot, now spent): rank {} reports verify width {said} for a {m}-row round \
                           (lockstep step {}); from here on it reports the truth", self.rank, self.step);
                said
            }
            _ => m,
        }
    }

    /// TP-D: a boolean the HEAD decides inside a step (a prefill chunk boundary's "client gone"),
    /// shipped over the control stream; every rank returns the head's value at the same program
    /// point. No control plane (the harness) = false on every rank (nothing is cancellable there).
    pub(super) fn head_flag(&mut self, v: bool) -> Result<bool> {
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

    /// FLASHNEXT-DF2-117 (ISSUES 117): ONE sampled DFlash2 draft per round for every rank. The head
    /// ships ITS walk's result (`mine`) to every node and keeps using it (returns None); a node
    /// blocks here at the same program point, adopts the head's draft (returns Some) and never
    /// draws its own. No control plane (the harness, TP=1) = None: the caller uses its own draw.
    /// `lane`/`p` ride the frame as the out-of-step tripwire.
    pub(crate) fn df2_draft(&mut self, lane: usize, p: usize,
                            mine: Option<&crate::dflash2::round::Df2SampleOut>)
                            -> Result<Option<crate::dflash2::round::Df2SampleOut>> {
        use crate::tp_serve::ServingMsg;
        if self.ctl.is_empty() { return Ok(None); }
        if self.rank == 0 {
            let m = mine.ok_or_else(|| anyhow::anyhow!("df2 draft ship: the head has no sampled draft to ship"))?;
            let msg = ServingMsg::Df2Draft {
                lane: lane as u32, p: p as u64, tokens: m.tokens.clone(),
                q_rows: m.q_rows.iter().map(|x| x.to_bits()).collect(), cand_tok: m.cand_tok.clone(),
                cand_q: m.cand_q.iter().map(|x| x.to_bits()).collect(),
            };
            for s in self.ctl.iter_mut() { crate::tp_serve::send_serving(s, &msg)?; }
            DF2_SHIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(None)
        } else {
            match crate::tp_serve::recv_serving(&mut self.ctl[0])? {
                ServingMsg::Df2Draft { lane: l, p: pp, tokens, q_rows, cand_tok, cand_q } => {
                    anyhow::ensure!(l as usize == lane && pp as usize == p,
                        "TP control plane out of step: the head's DF2 draft is for lane {l} at p {pp}; this rank is at lane {lane} p {p}");
                    DF2_SHIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(Some(crate::dflash2::round::Df2SampleOut {
                        tokens, q_rows: q_rows.into_iter().map(f32::from_bits).collect(), cand_tok,
                        cand_q: cand_q.into_iter().map(f32::from_bits).collect(),
                    }))
                }
                other => anyhow::bail!("TP control plane out of step: expected Df2Draft, got {other:?}"),
            }
        }
    }

    /// Whether this rank sits on a serve control plane (false in the harness / at TP=1).
    pub(crate) fn has_ctl(&self) -> bool { !self.ctl.is_empty() }

    /// `spill_agree` at world > 2: the head hub, generic over the transport so the four-rank mock runs the real
    /// frame + verdict (ISSUES 121: the verdict had never passed on hardware and no test reached it).
    pub(crate) fn spill_agree_hub<X: crate::net::HubXport>(&mut self, x: &mut X, ok: bool) -> Result<bool> {
        use crate::tp_lockstep as ls;
        let all = crate::net::hub_all(x, &ls::spill_frame(ok), ls::SPILL_WIRE)
            .map_err(|e| self.hub_fail("spill agree", &format!("{e:#}")))?;
        ls::spill_verdict(&all, self.world as usize).map_err(|why| self.hub_fail("spill agree", &why))
    }

    /// LANERES3: the spill tier's lockstep bit — the symmetric sibling of `head_flag`.
    /// Every rank contributes whether its local disk op succeeded (capture write /
    /// restore read) and the verdict is the AND of all ranks. Called from the
    /// admission-phase spill hooks, which every rank executes at the same program point
    /// of the same shipped admit sequence, so the rendezvous pairs up the same way
    /// HeadFlag frames pair up inside a prefill. This keeps the replicated
    /// `kv_spill::Policy` identical on every rank: a key no rank-local disk state can
    /// back on ALL ranks is dropped everywhere (counted), never acted on by one rank
    /// alone — a disk that diverged can never desync the pair.
    pub(super) fn spill_agree(&mut self, ok: bool) -> Result<bool> {
        use crate::tp_serve::ServingMsg;
        if self.world > 2 {
            let mut x = crate::net::link_hub(self.deadline_ms)?;
            return self.spill_agree_hub(&mut x, ok);
        }
        if self.ctl.is_empty() {
            return Ok(ok); // no control plane (harness paths): local truth, ranks symmetric by construction
        }
        if self.rank == 0 {
            crate::tp_serve::send_serving(&mut self.ctl[0], &ServingMsg::SpillFlag { ok })?;
            match crate::tp_serve::recv_serving(&mut self.ctl[0])? {
                ServingMsg::SpillFlag { ok: peer } => Ok(ok && peer),
                other => anyhow::bail!("TP control plane out of step: expected SpillFlag, got {other:?}"),
            }
        } else {
            match crate::tp_serve::recv_serving(&mut self.ctl[0])? {
                ServingMsg::SpillFlag { ok: peer } => {
                    crate::tp_serve::send_serving(&mut self.ctl[0], &ServingMsg::SpillFlag { ok })?;
                    Ok(ok && peer)
                }
                other => anyhow::bail!("TP control plane out of step: expected SpillFlag, got {other:?}"),
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
    pub(super) fn step_go(&mut self, step_no: u64, n_events: usize) -> Result<(u64, usize)> {
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
        let r = tp_agree_eq(agree_field(self.step, AGREE_GO), 0, 0, go_fence_word(hs, hn));
        self.wp.rec(7, t0, s0);
        if let Err(why) = r {
            crate::net::abort_link();
            anyhow::bail!("TP step-go fence FAILED at control step {hs}: {why}");
        }
        Ok((hs, hn))
    }

    /// S-B9-CF-PACK5: the PACKED round's pre-verify lockstep point — ONE exchange + ONE agree for a
    /// pair (the serial arm does one per lane), after BOTH lanes drafted and before the packed verify
    /// graph (whose R-keyed barrier sequence every rank must run alike) launches.
    ///
    /// `r` = the total verify rows R = m_0 + m_1 (the explicit number the barrier sequence depends
    /// on), `m0` = the first segment's rows, `words` = `lane_pack::pre_words` (per segment: rows,
    /// slot, position, bonus token, drafts — the design's OQ1 combined hash: self-delimiting, so a
    /// single FNV over the concatenation catches every disagreement a per-request vector would),
    /// `launched` = the SUM of both lanes' head draft passes issued on this rank.
    ///
    /// Same shape and slot discipline as `pre_verify` (exchange = rendezvous, then agree = fence;
    /// the wire sizes are unchanged: [tag, R, hash, launched] + the 16-byte tail guard). The width
    /// the exchange and the agree carry is R (<= 16; AGREE16 made the token's 5-bit field exact
    /// up to 31), the accept byte carries m0; `--tp-width-drill N` lies about a pack of R = N rows
    /// HERE (before the verify), so the drill proves the guard fires before the verify.
    pub(super) fn pre_verify_pack(&mut self, r: usize, m0: usize, words: &[i32], launched: usize) -> Result<()> {
        let r_said = self.width_said(r);
        if self.world > 2 {
            let mut x = crate::net::link_hub(self.deadline_ms)?;
            return self.pre_verify_hub(&mut x, r_said, words, launched);
        }
        let step = agree_field(self.step + 1, AGREE_PV);
        // a domain of its own (tag 0x9F.., seed 0x9AC5_0001): a rank that planned a PACK where its peer ran
        // a lone round (whose pre-verify uses 0x9E.. / 0x5EED) is caught by the exchange word itself,
        // exactly, not by two hashes over different word lists happening to differ
        let h = fnv32(0x9AC5_0001, words.iter().map(|&d| d as u32));
        let tag = 0x9F00_0000u32 | (step as u32 & 0xFFFF);
        let mine = [tag, r_said as u32, h, launched as u32];
        // the race probe's deterministic alignment, exactly as `pre_verify` applies it (the pack's
        // exchange has the same slot-aliasing exposure, so the REL-TP2RACE drills cover it)
        let e_last = crate::net::traced_device_epoch();
        let g0 = crate::net::traced_xchg_gen();
        let g = if crate::net::race_probe_hold_ms() > 0 {
            let g1 = crate::net::traced_gen_add(crate::net::probe_gen_delta(g0, e_last & 7));
            crate::dreprintln!("[tp-race-probe] rank {}: pack pre-verify aligned xchg gen {g0} -> {g1} onto epoch {e_last} \
                       (device_epoch&7 = {}, gen&7 = {}), hold {} ms on rank 1",
                      self.rank, e_last & 7, g1 & 7, crate::net::race_probe_hold_ms());
            g1
        } else { g0 };
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, 4 + 4)?; // + the 16-byte tail guard
        self.wp.rec(3, t0, s0);
        if peer != mine {
            crate::net::abort_link();
            anyhow::bail!("TP pack pre-verify FAILED (round {}): this rank R = {r_said} rows, pack hash {h:08x} (combined over every \
                           segment's rows, slot, position, bonus, sampler and drafts), head passes launched {launched}; peer {:?} \
                           [tag, R, hash, launched] — device_epoch&7 = {}, xchg gen&7 = {} — ranks chose different packs (row total, \
                           slots, drafts or speculative head-pass launches); this rank's pack: {}; link aborted before the packed verify",
                          self.step + 1, peer, e_last & 7, g & 7, crate::lane_pack::describe_pre_words(words));
        }
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let rr = tp_agree_eq(step, m0, r_said, h);
        self.wp.rec(4, t0, s0);
        if let Err(why) = rr {
            crate::net::abort_link();
            anyhow::bail!("TP pack pre-verify agree FAILED (round {}, R = {r_said} rows, first segment {m0} rows, pack hash {h:08x}: \
                           {why}) — ranks chose different packs; this rank's pack: {}; link aborted before the packed verify",
                          self.step + 1, crate::lane_pack::describe_pre_words(words));
        }
        self.pre_agrees += 1;
        Ok(())
    }

    /// LANEWIDE-TP: the wide round's ONE pre-verify lockstep point — `pre_verify_pack`'s exchange (tag domain 0x9D..,
    /// seed 0x9AC5_0002) over the segments of every lane, with the lane count in the agree token (a wide round's row
    /// total, up to --lane-wide-rows-max, does not fit its 5-bit width; it rides the exchange word exactly).
    pub(super) fn pre_verify_wide(&mut self, r: usize, n: usize, words: &[i32], launched: usize) -> Result<()> {
        let r_said = self.width_said(r);
        if self.world > 2 {
            let mut x = crate::net::link_hub(self.deadline_ms)?;
            return self.pre_verify_hub(&mut x, r_said, words, launched);
        }
        let step = agree_field(self.step + 1, AGREE_PV);
        // a domain of its own (tag 0x9F.., seed 0x9AC5_0001): a rank that planned a PACK where its peer ran
        // a lone round (whose pre-verify uses 0x9E.. / 0x5EED) is caught by the exchange word itself,
        // exactly, not by two hashes over different word lists happening to differ
        let h = fnv32(0x9AC5_0002, words.iter().map(|&d| d as u32));
        let tag = 0x9D00_0000u32 | (step as u32 & 0xFFFF);
        let mine = [tag, r_said as u32, h, launched as u32];
        // the race probe's deterministic alignment, exactly as `pre_verify` applies it (the pack's
        // exchange has the same slot-aliasing exposure, so the REL-TP2RACE drills cover it)
        let e_last = crate::net::traced_device_epoch();
        let g0 = crate::net::traced_xchg_gen();
        let g = if crate::net::race_probe_hold_ms() > 0 {
            let g1 = crate::net::traced_gen_add(crate::net::probe_gen_delta(g0, e_last & 7));
            crate::dreprintln!("[tp-race-probe] rank {}: pack pre-verify aligned xchg gen {g0} -> {g1} onto epoch {e_last} \
                       (device_epoch&7 = {}, gen&7 = {}), hold {} ms on rank 1",
                      self.rank, e_last & 7, g1 & 7, crate::net::race_probe_hold_ms());
            g1
        } else { g0 };
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, 4 + 4)?; // + the 16-byte tail guard
        self.wp.rec(3, t0, s0);
        if peer != mine {
            crate::net::abort_link();
            anyhow::bail!("TP wide pre-verify FAILED (round {}): this rank R = {r_said} rows, pack hash {h:08x} (combined over every \
                           segment's rows, slot, position, bonus, sampler and drafts), head passes launched {launched}; peer {:?} \
                           [tag, R, hash, launched] — device_epoch&7 = {}, xchg gen&7 = {} — ranks chose different packs (row total, \
                           slots, drafts or speculative head-pass launches); this rank's pack: {}; link aborted before the wide verify",
                          self.step + 1, peer, e_last & 7, g & 7, crate::lane_pack::describe_pre_words(words));
        }
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let rr = tp_agree_eq(step, n, n, h); // the row total rides the exchange word (it may exceed the token's 5-bit width)
        self.wp.rec(4, t0, s0);
        if let Err(why) = rr {
            crate::net::abort_link();
            anyhow::bail!("TP wide pre-verify agree FAILED (round {}, R = {r_said} rows, {n} lanes, pack hash {h:08x}: \
                           {why}) — ranks chose different packs; this rank's pack: {}; link aborted before the wide verify",
                          self.step + 1, crate::lane_pack::describe_pre_words(words));
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
        // S-B9-CF-AGREE16: the frame words are whole u32s; the real accept count and width ride them (the
        // saturation at 255 / at 15 inherited from the pairwise token made 15, 16 and 17 rows one word here too)
        let frame = ls::rnd_frame(&ls::RoundIn { step, field, accept: a, width: self.width_said(m), hash: h,
                                                 epoch, ms, plain_ms, captured });
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
                crate::deprintln!("[tp-ident] FIRST MISMATCH at {what} (lockstep step {step}, width {m}): {bad} of {} digests differ \
                           (logits rows 0..{m}, resid, taps; some rank differs from the head)", d.len());
            }
        }
        self.agrees += 1;
        if !out.epoch_skew.is_empty() {
            self.epoch_warns += 1;
            if self.epoch_warns <= 5 || self.epoch_warns % 1000 == 0 {
                crate::deprintln!("[tp-round] EPOCH-PROBE at {what} (lockstep step {step}, #{}): device epoch differs from the head's on \
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
pub(super) const AGREE_A: u64 = 0;

pub(super) const AGREE_PV: u64 = 1;

pub(super) const AGREE_B: u64 = 2;

pub(super) const AGREE_GO: u64 = 3;

pub(super) fn agree_field(counter: u64, kind: u64) -> u64 { (counter & 0x3F_FFFF) | (kind << 22) }

/// TP-E: the lockstep agree WITH the comparison. `net::agree_ext` only rendezvouses on the step field
/// and returns the PEER's (accept, hash); until TP-E the EXL3 callers checked `is_none()` alone, so a
/// divergent accept count / emitted-ids hash / boot config would have sailed through every "agree ok"
/// (only the timeout could fire). The NVFP4 path (batch.rs) always compared. Ok(()) = both ranks
/// published the identical token; Err = timeout/abort or a mismatch (message says which).
pub(crate) fn tp_agree_eq(step: u64, accept: usize, k_verify: usize, hash: u32) -> std::result::Result<(), String> {
    let (a, k) = tp_agree_fields(accept, k_verify)?;
    tp_agree_decide(a, k, hash, crate::net::agree_ext(step, a, k, hash))
}

/// The (accept, width) pair as the pairwise token carries it: accept in a byte, width in the five bits of
/// `net::agree_word`. S-B9-CF-AGREE16: a value that does not fit is REFUSED (the caller aborts the link),
/// never saturated — a saturation at 15 here made a 15-row and a 16-row verify one and the same word.
pub(crate) fn tp_agree_fields(accept: usize, k_verify: usize) -> std::result::Result<(u8, u8), String> {
    if accept > u8::MAX as usize || k_verify > crate::net::AGREE_WIDTH_MAX {
        return Err(format!("REFUSED: accept count {accept} / verify width {k_verify} do not fit the lockstep token (accept <= 255, \
                            width <= {}); refusing instead of sending a value the peer could mistake for another",
                           crate::net::AGREE_WIDTH_MAX));
    }
    Ok((accept as u8, k_verify as u8))
}

/// The step-go fence's digest of the (control step, event count) pair both ranks hold after the exchange.
/// S-B9-CF-AGREE16: it was `(hs as u32) ^ ((hn as u32) << 24) ^ 0x60`, where the event count shared bits
/// 24..31 with the step (`(hs ^ 1 << 24, hn ^ 1)` gave the same word) and an event count of 256 or more
/// shifted out. An FNV of the two whole words has neither alias, and it is a digest, which is what
/// `net::agree_word` folds.
pub(crate) fn go_fence_word(hs: u64, hn: usize) -> u32 {
    fnv32(0x60, [hs as u32, (hs >> 32) as u32, hn as u32, (hn as u64 >> 32) as u32].into_iter())
}

/// Boot check (TP=2 only): the pairwise lockstep token names a round's width in five bits, and a shared
/// plain step is as wide as the busy lanes, so `--max-batch` above 31 could reach a round the token
/// cannot name. Refuse at boot instead of aborting the link mid-serve at the first such round. World > 2
/// frames carry whole u32 words and TP=1 has no lockstep: both are unbounded here.
pub(crate) fn tp_lane_width_fits(world: u32, max_batch: usize) -> std::result::Result<(), String> {
    if world == 2 && max_batch > crate::net::AGREE_WIDTH_MAX {
        return Err(format!("--max-batch {max_batch} at TP=2: a plain step shared by more than {} lanes cannot be named in the \
                            pairwise lockstep token (5-bit verify-width field); use --max-batch <= {}",
                           crate::net::AGREE_WIDTH_MAX, crate::net::AGREE_WIDTH_MAX));
    }
    Ok(())
}

/// F2 (owner 2026-10-10: "it should output a message and clearly show the cap at 16 lanes"): the EXL3
/// server serves at most 16 requests at the same time — the PLE prefill kernels cap a plain step at
/// m <= 16 (a 17th lane used to error every in-flight request and go DEAD under TP), the packed
/// verify widths are 4..=16, and the old TP=2 rule capped at `AGREE_WIDTH_MAX`. A `--max-batch`
/// above 16 now BOOTS WITH 16: one plain line at the boot, `velogb10_max_batch` reports 16, the KV
/// is sized for 16, and the head ships 16 to the TP nodes (the same resolver on the same argv).
/// Further requests wait in the queue — `--max-waiting` semantics are unchanged.
/// PREHARD1 [ISSUES 104]: ONE authority — this is `batch::SERVE_MAX_LANES`, the cap the trunk
/// (NVFP4) serve path now shares; the two can no longer diverge.
pub(crate) const EXL3_MAX_LANES: usize = crate::batch::SERVE_MAX_LANES;

/// `--max-batch` as the serve resolves it from an argv: the ONE parser (`build_serve` and the
/// pre-network refusal below), so the two cannot come to read different defaults.
pub(crate) fn max_batch_arg(args: &[String]) -> usize {
    arg(args, "--max-batch").and_then(|s| s.parse().ok()).unwrap_or(8)
}

/// F2: the resolved lane count and the value the argv asked for. Every consumer of `--max-batch`
/// (`build_serve`, the pre-network refusal, the boot line and the metrics) reads the resolved half,
/// so the ranks cannot come to different lane counts.
pub(crate) fn resolved_max_batch(args: &[String]) -> (usize, usize) {
    let raw = max_batch_arg(args);
    (raw.min(EXL3_MAX_LANES), raw)
}

/// S-B9-CF-AGREE16B (W4): `tp_lane_width_fits` from the argv and the world alone, so it can run BEFORE
/// any network step. It used to run only inside `build_serve`, after the pack sync (minutes, on a cold
/// node) and the RDMA attach: a `--max-batch 32` TP=2 launch synced, linked, and was refused then. Call
/// sites, in boot order: the head before discovery and the pack sync (`run_exl3_tp_serve_head`), the
/// node before the link (`exl3_tp_node_prelink`, reported to the head), both ranks before the attach
/// (`run_tp_head_serve` / `run_tp_node_serve`, next to the otel refusal), and `build_serve` as the
/// backstop for a caller that reaches it another way.
/// TP2BIT5 #26: head-side pack-width preflight. Opens the pack header-only and refuses —
/// in SECONDS, before discovery and the 57.5-GiB first sync — any module whose k2 has no
/// kernel in this build (the same `width_supported` predicate the loader itself enforces,
/// so head and node can never disagree). MTP-head-only unsupported widths are NOT a refusal:
/// Fix 4 skips the draft head and serves plain, so this preflight stays silent about them
/// (open() prints the skip line once, at real load).
pub fn tp_pack_width_preflight(dir: &str) -> Result<()> {
    // The loader's own open() IS the check: header-only (cheap, no weight IO), refuses any
    // non-MTP module at an unsupported k2 with the plain-word bail, and prints the Fix-4 skip
    // line for MTP-head-only widths instead of refusing. Prefix the refusal so the operator
    // sees it was caught before the ship.
    // V0710: an FP4-Local pack (ModelOpt MXFP8 + NVFP4 on the EXL3 trunk) has no quantization_config.json and no
    // trellis widths: Exl3Pack::open refused it here, so v0.7.9 could not start FP4-Local at TP=2 / TP=4 at all. Its own
    // format check (the one the ship plan runs, cluster.rs) is the preflight for it.
    if crate::fp4l::pack::is_fp4l_dir(std::path::Path::new(dir)) {
        return crate::fp4l::pack::preflight(std::path::Path::new(dir))
            .map_err(|e| anyhow::anyhow!("pack preflight (before the node sync; nothing was shipped): {e:#}"));
    }
    crate::exl3::Exl3Pack::open(dir)
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("pack width preflight (before the node sync; nothing was shipped): {e:#}"))
}

pub fn tp_lane_width_refusal(args: &[String], world: u32) -> Result<()> {
    // F2: the 16-lane cap is applied FIRST, so this can no longer refuse anything — 16 always fits
    // the pairwise agree width (`AGREE_WIDTH_MAX` 31). The function stays as the backstop that keeps
    // a future caller that bypasses the cap from booting a width the agree token cannot name.
    let (width, _) = resolved_max_batch(args);
    tp_lane_width_fits(world, width).map_err(|e| anyhow::anyhow!(e))
}

/// The comparison half of `tp_agree_eq`, pure: `peer` is what `net::agree_ext` returned (None = link
/// aborted / peer timeout, Some = the peer's (accept, agree_word) for the same step field).
pub(crate) fn tp_agree_decide(accept: u8, k_verify: u8, hash: u32, peer: Option<(u8, u32)>) -> std::result::Result<(), String> {
    let word = crate::net::agree_word(k_verify, hash);
    match peer {
        None => Err("link aborted or peer timeout".into()),
        Some((pa, pw)) if pa == accept && pw == word => Ok(()),
        Some((pa, pw)) => Err(format!("MISMATCH: this rank (accept {accept}, width {k_verify}, word {word:08x}) vs peer (accept {pa}, \
                                       width {}, word {pw:08x}) [word = width in bits 27..31 | 27-bit digest]", pw >> 27)),
    }
}

pub(super) fn fnv32(h: u32, words: impl Iterator<Item = u32>) -> u32 {
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
pub(super) fn sticky_cuda_error(e: &anyhow::Error) -> bool {
    let s = format!("{e:?}");
    ["ILLEGAL_ADDRESS", "LAUNCH_FAILED", "MISALIGNED_ADDRESS", "ILLEGAL_INSTRUCTION", "ECC_UNCORRECTABLE",
     "HARDWARE_STACK_ERROR", "INVALID_PC", "INVALID_ADDRESS_SPACE", "CUDA_ERROR_ASSERT", "LAUNCH_TIMEOUT",
     "CONTEXT_IS_DESTROYED"].iter().any(|t| s.contains(t))
}
