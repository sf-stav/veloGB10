//! Round drivers: `mtp_round_clamped` (the serve's clamp rule), `mtp_round`, `copy_round`, `mtp_round_adaptive`,
//! `mtp_verify`, and the WP24 / WP05 verify-stage helpers. A source-text test pins `mtp_round_clamped` as the item
//! that precedes `mtp_round`. See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// S-B9-REL-SMALLFIXES item 2: the served round for callers OUTSIDE the scheduler (--bench-mtp,
    /// the draft-quality bench, the KV-NLL acceptance probe): `mtp_round` behind the scheduler's
    /// clamp at the sparse switch (`regime_depth`, exactly `step_mtp`'s). A verify and its draft
    /// chain run ONE attention regime chosen from the last row, so a round straddling qsa_limit()
    /// puts its rows at/below the switch on the sparse kernels — not the plain decode's bits
    /// (AGENTS 2.4) — and a LOSSLESS run crossing the switch could report a mismatch the serve no
    /// longer has. Here the round ends on the switch; where no draft fits (p == qsa_limit) it is the
    /// scheduler's head-synced plain step (`step_mtp`, k_eff == 0): the head's row p written KV-only
    /// from the saved tap, one forward_step, its tap snapshotted as the next round's row 0.
    /// Identity for every round that does not reach the switch, and with --qsa-row-regime 0.
    pub fn mtp_round_clamped(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, k: usize,
                             b: i32, p: usize, tap_row: usize) -> Result<MtpRoundOut> {
        let lim = if self.qsa_live(usize::MAX) && row_regime_on() { Some(self.qsa_limit()) } else { None };
        match clamped_round_plan(p, k, lim) {
            Some(d) => {
                if d < k {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| crate::drprintln!("[exl3] per-row attention regime: a bench/probe round at position {p} stops at the \
sparse switch (depth {k} -> {d}; --qsa-row-regime 0 = rounds may straddle it)"));
                }
                self.mtp_round(sc, head, slot, d, b, p, tap_row)
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| crate::drprintln!("[exl3] per-row attention regime: a bench/probe round at position {p} is a plain \
step (no draft fits before the sparse switch)"));
                self.mtp_plain_prime(sc, head, slot, b, p, tap_row)?;
                let ids = self.forward_step(sc, &[b], &[p], &[slot], None)?;
                self.tap_snapshot(sc, slot)?;
                Ok(MtpRoundOut { a: 0, emitted: vec![ids[0]], pos_next: p + 1, last_tok: ids[0], tap_row: 0,
                                 drafts: Vec::new(), diag_ms: 0.0 })
            }
        }
    }

    pub fn mtp_round(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, k: usize,
                     b: i32, p: usize, tap_row: usize) -> Result<MtpRoundOut> {
        // S-B9-CF-PACK4S: the draft half is `mtp_draft_fixed` (its statements moved there
        // unchanged, in order) so the packed arm's fixed-depth chain is this very code.
        let (drafts, t) = self.mtp_draft_fixed(sc, head, slot, k, b, p, tap_row)?;
        let out = self.mtp_verify(sc, slot, k, b, p, drafts, t)?;
        self.head_reprime(sc, head, slot, p, &out.drafts[..out.a])?;
        Ok(out)
    }

    /// WIDEVERIFY1 copy-span round: verify host-proposed draft ids (tokens copied from the lane's own
    /// context, `crate::copy_span`) with NO draft pass, through the same chain verify as `mtp_round`.
    /// `drafts` = 1..=MTP_MAX_K token ids, or up to COPY_MAX_K on a scratch built with the wide copy
    /// verify (`Scratch::vrows == COPY_VROWS`, WIDEVERIFY1 refinement 2: same kernels, the ring /
    /// taps planes and the accept / REPRIME words sized for 16 rows, a 15-row re-prime graph); the
    /// verify width is drafts.len() + 1. Sampling: the caller
    /// stages the match rule (`set_spec_sampling(rows, None)`, rows and penalties at that width): a
    /// copied draft has no draft probabilities, and the match rule is exact for any draft that does not
    /// depend on this round's uniforms. Each call below replaces what the skipped draft chain did:
    /// 1. `mtp_plain_prime` writes the draft head's KV row p from (the saved tap that predicted b, b),
    ///    the row a normal round's pass 0 writes (the control round's prime, WP03): without it the next
    ///    neural round attends over whatever an older round left at p.
    /// 2. `draft_meta` = {b, p, slot, dctr}: the device re-prime (`xq_reprime_stage`) reads b and p from
    ///    it (the draft chain uploads it each round; stale values would re-prime the wrong positions).
    /// 3. `d_dev` = the drafts: `xq_accept` compares the verify rows against `d_dev`, which the draft
    ///    kernels fill on device in a normal round (`df2_round` uploads its drafts the same way).
    /// 4. `mtp_verify`, unchanged: verify, accept, commit, taps of every verified row.
    /// 5. `head_reprime` with the accepted drafts: rewrites the head's rows p+1..p+a from the target's
    ///    verify taps, so the neural drafter resumes on real target states after a copy (a copy that
    ///    commits many tokens would otherwise leave those rows unwritten).
    /// Emitted tokens are the verify's own choices, so greedy output equals plain decoding bitwise.
    pub fn copy_round(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, b: i32, p: usize,
                      tap_row: usize, mut drafts: Vec<i32>, t0: std::time::Instant) -> Result<MtpRoundOut> {
        let k = drafts.len();
        let kmax = if sc.vrows >= COPY_VROWS { COPY_MAX_K } else { MTP_MAX_K };
        anyhow::ensure!(k >= 1 && k <= kmax && k < sc.vrows && k < sc.toks.len() && p + k < self.max_pos,
                        "copy_round: {k} drafts at p={p} (max {kmax} on this scratch: verify rows {}, width {}; KV window {})",
                        sc.vrows, sc.toks.len(), self.max_pos);
        anyhow::ensure!(sc.d_dev.len() >= k, "copy_round: d_dev narrower than {k}");
        // the draft chain's clamp (an id outside the vocabulary becomes 0; a context id never is)
        let v = self.cfg.vocab_size as i32;
        for d in drafts.iter_mut() {
            if *d < 0 || *d >= v { *d = 0; }
        }
        self.mtp_plain_prime(sc, head, slot, b, p, tap_row)?;
        self.dev.htod_copy_into(vec![b, p as i32, slot as i32, sc.rq_dctr], &mut sc.draft_meta)?;
        let mut d_w = drafts.clone();
        d_w.resize(sc.d_dev.len(), 0);
        self.dev.htod_copy_into(d_w, &mut sc.d_dev)?;
        let t1 = std::time::Instant::now();
        let out = self.mtp_verify(sc, slot, k, b, p, drafts, [t0, t1, t1])?;
        self.head_reprime(sc, head, slot, p, &out.drafts[..out.a])?;
        Ok(out)
    }

    /// API-parity DDS: an MTP round whose draft chain STOPS EARLY. Drafting runs one graphed
    /// pass at a time; after pass i the host reads its confidence (the max draft logit) and
    /// `keep(i, conf)` decides whether to draft pass i+1. The verify then runs at the chosen
    /// width m = w+1 (graphs per width). Emitted tokens obey the same accept/commit contract
    /// as `mtp_round` (greedy stays bitwise lossless: only WHICH drafts are proposed changes).
    /// Returns the round and the per-pass confidences of the w verified drafts.
    pub fn mtp_round_adaptive(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, kmax: usize,
                              b: i32, p: usize, tap_row: usize,
                              keep: &mut dyn FnMut(usize, f32) -> bool,
                              pre_verify: Option<&mut dyn FnMut(usize, &[i32], usize) -> Result<()>>)
                              -> Result<(MtpRoundOut, Vec<f32>)> {
        // S-B9-CF-PACK4: the draft half is `mtp_draft_adaptive` (its statements moved there
        // unchanged, in order) so the packed arm drafts each request with this very code.
        let DraftHalf { drafts, confs, w, launched, wasted, spec, tp0, tp1 } =
            self.mtp_draft_adaptive(sc, head, slot, kmax, b, p, tap_row, keep)?;
        let tp2 = tp1;
        // TP-I (latent TP-H defect, found while collecting routing histograms): with the sharded
        // draft-head screen every pass is a collective, and a WASTED speculative pass (launched, then
        // stopped by the DDS) is never synchronized by the loop above — its bmax doorbell epoch can
        // still be in flight when the host pre-verify exchange below runs. The host exchanges stage
        // in send slot 0 and receive in recv slot (gen & 7) of the SAME rings the doorbell epochs use,
        // so the pass's payload landed in the exchange frame's slot (node: "recv payload tail for gen
        // 137 never landed (tail = two fp32 block maxima 11.04 / 10.11)", code 2, engine DEAD at round
        // 90). Drain first: device sync (our K2 consumed the peer's epoch) + proxy retired every
        // epoch we published. Only in wasted-pass rounds (~4.5%); the verify waited behind the
        // pass on the stream anyway.
        // TP-4D: world 2 only. The aliasing above is a world-2 property. At world 4 the screen IS sharded
        // (the enable_shard call in this file admits world 2 | 4), so a wasted pass there IS a collective with epochs possibly in
        // flight, but the pre-verify lockstep is a head-hub exchange over the DEDICATED control slots
        // (net_exchange_one: own staging slot, own recv rings, send completions counted by the proxy through
        // xchg_send_seq), which share no memory with the doorbell rings — nothing can land in an exchange frame's
        // slot, so there is nothing to drain and the device sync would be a pure stall. This rests on the
        // disjoint-region argument only (PLAN/TP-4D_REPORT.md section 3); the first hardware evidence is MTP decode
        // at world 4 (PLAN/TP-4D_GATES.md rung 3), where wasted-pass rounds occur in a few percent of rounds.
        if wasted.is_some() && pre_verify.is_some() && self.tp.as_ref().map_or(false, |t| t.world == 2) && xtp::dh_shard_on() {
            self.dsync()?;
            crate::net::drain_sends(std::time::Duration::from_millis(200))?;
        }
        // TP-D: the caller's pre-verify hook (the TP width/drafts agree) — after every draft pass,
        // before the verify graph (whose width selects the barrier sequence) launches
        if let Some(f) = pre_verify { f(w, &drafts, launched)?; }
        let out = self.mtp_verify(sc, slot, w, b, p, drafts, [tp0, tp1, tp2])?;
        if spec {
            // HOST: where this lane's next round starts (a round elsewhere = a new request)
            if let Some(np) = sc.wp05.as_mut().and_then(|s| s.spec.as_mut()).and_then(|s| s.next_p.get_mut(slot)) {
                *np = out.pos_next;
            }
        }
        self.head_reprime(sc, head, slot, p, &out.drafts[..out.a])?;
        Ok((out, confs))
    }

    /// The verify half of an MTP round (shared by `mtp_round` and `mtp_round_adaptive`):
    /// uploads, PLE host half, the graphed verify block at m = k+1, accept, commit, readback.
    pub(super) fn mtp_verify(&self, sc: &mut Scratch, slot: usize, k: usize, b: i32, p: usize,
                  drafts: Vec<i32>, t: [std::time::Instant; 3]) -> Result<MtpRoundOut> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let m = k + 1;
        // WP02: verify rows p..p+k write trunk K/V, indexer keys and read cos_tab — all max_pos rows
        anyhow::ensure!(p + k < self.max_pos,
                        "mtp_verify: rows {p}..={} past the KV window (max_pos {})", p + k, self.max_pos);
        let prof = round_prof_on();
        let [tp0, tp1, tp2] = t;
        // ---- verify: uploads (toks/pos/slots), PLE host half, shadows, kernels
        let mut toks_v = vec![b];
        toks_v.extend_from_slice(&drafts);
        // WP05: ONE pinned staging fill (toks/pos/slots/slot_ids/ple_emb — the same full-width
        // contents as below, PLE rows the DDS passes already built reused) + async copies.
        let hist_pre = if wp05_on("dds") && wp05_stage_fits(sc, m) {
            self.wp05_verify_stage(sc, slot, &toks_v, p)?
        } else {
        let mut toks_w = toks_v.clone();
        toks_w.resize(sc.toks.len(), 0);
        self.dev.htod_copy_into(toks_w, &mut sc.toks)?;
        let mut pos_v: Vec<i32> = (p..p + m).map(|x| x as i32).collect();
        pos_v.resize(sc.pos.len(), 0);
        self.dev.htod_copy_into(pos_v, &mut sc.pos)?;
        let mut slotpos: Vec<i32> = vec![0; sc.slots.len()];
        anyhow::ensure!(
            sc.slots.len() >= m * 2 && sc.toks.len() >= m,
            "mtp_round: scratch width {} < verify width {} — size the scratch >= k+1",
            sc.toks.len(), m
        );
        for r in 0..m {
            slotpos[r * 2] = slot as i32;
            slotpos[r * 2 + 1] = (p + r) as i32;
        }
        self.dev.htod_copy_into(slotpos, &mut sc.slots)?;
        // S-A3-i: the bf16 selection kernels read a DEINTERLEAVED slot table — stage it
        // here too (the verify rows are this round's lanes; rows >= m are never read).
        self.dev.htod_copy_into(vec![slot as i32; sc.slot_ids.len()], &mut sc.slot_ids)?;
        let hist_pre = {
            // PLE host half (advances the host ring by m; restored post-accept)
            let mut emb_bits = vec![0u16; sc.ple_emb.len()];
            let hp = {
                let hist = self.ple_hist.lock().unwrap();
                (hist[slot * 2], hist[slot * 2 + 1])
            };
            self.ple_build_embed(m, &toks_v, &vec![slot; m], &mut emb_bits)?;
            self.dev.htod_copy_into(emb_bits, &mut sc.ple_emb)?;
            hp
        };
        hist_pre
        };
        // S-A3-m: the verify block (shadow -> kernels -> taps -> accept -> commit) is
        // launches only with every per-round value in device buffers, so it replays
        // as ONE CUDA graph per (m, regime, slot) — first round eager, then captured
        // (the forward_step pattern). Kills the eager path's launch gaps (~6 ms/round
        // at m=6, S-A3-m budget).
        let tp3 = std::time::Instant::now();
        let qsa = self.qsa_live(p + k);
        let (qg, qb) = self.qsa_grid_bucket_key(p + k);
        sc.qsa_grid_nblk = qg;
        let vkey = vkey_of(slot, m, qsa, qb) | sk(sc) // TP-C: + the scratch id
            | ((sc.pen_live as usize) << 40) // WP15: penalized verify = its own graph
            | rq_verify_key(sc) // WP24: the real-q verify sampler = its own graph
            | self.vp_key(sc, m); // TP-H #4: the vocab-parallel greedy tail = its own graph
        let want_graph = crate::opts::var(crate::opt!("exl3-no-graph")).is_err()
            && crate::opts::var(crate::opt!("mtp-dump")).map_or(true, |s| s.is_empty())
            && gdn_ring_mode() != 2
            && !wp12_xcheck();
        let graphed = want_graph && self.graphs.lock().unwrap().contains_key(&vkey);
        // TUNE T1 (§5.2): the --autotune harness's verify shadow replays (m' <= m, after the
        // uploads, before the canonical verify: canonical-last); never armed in serving.
        let sr_at = autotune::SrAt { slot, k, m, p, tap_row: 0, qsa, bucket: qb };
        if autotune::armed() { autotune::sr_before_verify(self, sc, &sr_at)?; }
        if graphed {
            self.graphs.lock().unwrap().get(&vkey).unwrap().launch();
        } else {
            self.verify_block(&l, sc, slot, m, k, qsa)?;
        }
        if autotune::armed() { autotune::sr_after_verify(self, sc, &sr_at)?; }

        // ---- single readback
        let tp4 = std::time::Instant::now();
        // REPRIME (w2): with the device re-prime live, the readback is an async copy + event and the
        // fixed-width re-prime is enqueued right behind the verify (head_reprime only finishes it).
        let rp_live = self.reprime_dev_on(sc);
        let out = if rp_live { self.reprime_dev_enqueue(&l, sc, slot, p, k)? } else {
        self.dsync()?;
        self.dtoh(&sc.accept_out)?
        };
        if let Some(st) = sc.wp05.as_mut() { st.inflight = false; } // WP05: stage copies retired
        if prof {
            let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
            round_prof_add([ms(tp0, tp1), ms(tp1, tp2), ms(tp2, tp3), ms(tp3, tp4), ms(tp4, std::time::Instant::now())]);
        }
        if want_graph && !graphed {
            if rp_live { self.dsync()?; } // REPRIME: capture with the re-prime drained
            self.capture_verify_graph(&l, sc, slot, m, k, qsa, vkey);
        }
        let a = out[k + 2] as usize;
        anyhow::ensure!(a <= k, "mtp_round: accept out of range");
        let emitted: Vec<i32> = out[..=a].to_vec();
        let last = emitted[a];
        // host ring restore: build_embed advanced by m; only a+1 tokens committed.
        {
            let mut hist = self.ple_hist.lock().unwrap();
            if a == 0 {
                // committed = [b]: the pair is (token before b, b)
                hist[slot * 2] = hist_pre.1;
                hist[slot * 2 + 1] = b;
            } else {
                hist[slot * 2] = toks_v[a - 1];
                hist[slot * 2 + 1] = toks_v[a];
            }
        }
        // WP24 diagnostics (off unless --wp24-dump / --wp24-xcheck): the rung-0 dump and the
        // real-q cross-check read this verify's logits rows / ids / lists back (after a sync; the
        // device re-prime enqueued behind the verify touches none of those buffers).
        let mut diag_ms = 0.0;
        if sc.rq_mode > 0 && (crate::wp24::dump_on() || crate::wp24::xcheck_on()) {
            let td = std::time::Instant::now();
            self.wp24_after_verify(sc, slot, p, k, &drafts, a)?;
            diag_ms = td.elapsed().as_secs_f64() * 1e3;
        }
        Ok(MtpRoundOut {
            a,
            emitted,
            pos_next: p + a + 1,
            last_tok: last,
            tap_row: a,
            drafts,
            diag_ms,
        })
    }

    /// WP24: host half of the rung-0 dump / the real-q cross-check for the verify that just
    /// completed (rows 0..=k of sc.logits / sc.argmax, sc.samp, the passes' lists).
    fn wp24_after_verify(&self, sc: &mut Scratch, slot: usize, p: usize, k: usize, drafts: &[i32],
                         a: usize) -> Result<()> {
        let (m, v) = (k + 1, self.cfg.vocab_size);
        self.dsync()?;
        let logits = self.dtoh(&sc.logits)?;
        let emitted = self.dtoh(&sc.argmax)?;
        let samp = self.dtoh(&sc.samp)?;
        let list = self.dtoh(&sc.rq_list)?;
        anyhow::ensure!(logits.len() >= m * v && emitted.len() >= m && samp.len() >= m * 8
                        && list.len() >= k * crate::wp24::RQ_ROW, "wp24: scratch narrower than the verify");
        let r = crate::wp24::Round {
            slot, pos: p, w: k, a, mode: sc.rq_mode, v,
            logits: &logits[..m * v], emitted: &emitted[..m], samp: &samp[..m * 8],
            list: &list[..k * crate::wp24::RQ_ROW], drafts,
            dsamp: sc.rq_dsamp_host, dctr: sc.rq_dctr as u32,
        };
        if crate::wp24::dump_on() { crate::wp24::dump_round(&r)?; }
        if crate::wp24::xcheck_on() && sc.rq_mode == 2 { crate::wp24::xcheck_round(&r); }
        Ok(())
    }

    /// WP05: verify PLE row q into the pinned stage: tokens t = toks (t_0 = b, t_i = d_{i-1}),
    /// ring pair hp before t_0 — row q's n-gram history is [t_{q-2}, t_{q-1}, t_q] with
    /// t_{-2}, t_{-1} = hp, exactly what ple_build_embed's advancing ring yields for row q.
    pub(super) fn wp05_ple_row(&self, st: &mut Wp05Stage, hp: (i32, i32), toks: &[i32], q: usize) -> Result<()> {
        let ple = match self.ple.as_ref() {
            Some(p) => p,
            None => return Ok(()),
        };
        let t = |j: usize| -> i32 { if j >= 2 { toks[j - 2] } else if j == 1 { hp.1 } else { hp.0 } };
        // t(j) = t_{j-2}; row q history = [t_{q-2}, t_{q-1}, t_q]
        let h3 = [t(q), t(q + 1), toks[q]];
        let uids = ple.ngram_uids(&h3);
        ple.prefetch_rows(&uids); // PERFFIX1: the row's 16 reads in flight together (no-op for the RAM table)
        ple.embed_row(&uids, st.ple_row(q))
    }

    /// WP05: mtp_verify's uploads through the pinned stage. Writes the SAME full-width contents
    /// the pageable path uploads — toks [b, d.., 0 pad], pos [p.., 0 pad], slots [(slot, p+r)..,
    /// 0 pad], slot_ids [slot; w], ple_emb [rows 0..m, zero rows m..w] — into one pinned buffer
    /// (PLE rows already built by the DDS passes for this (slot, ring pair, tokens) are reused,
    /// the rest built here) and enqueues five async NULL-stream copies (the old path's
    /// htod_copy_into stream; ordering vs the compute stream unchanged). Advances the host
    /// n-gram ring by m exactly like ple_build_embed (mtp_verify restores it post-accept).
    /// Returns the pre-round ring pair (the old path's hist_pre).
    fn wp05_verify_stage(&self, sc: &mut Scratch, slot: usize, toks_v: &[i32], p: usize) -> Result<(i32, i32)> {
        let m = toks_v.len();
        let (d_toks, d_pos) = (*sc.toks.device_ptr() as u64, *sc.pos.device_ptr() as u64);
        let (d_slots, d_sids) = (*sc.slots.device_ptr() as u64, *sc.slot_ids.device_ptr() as u64);
        let d_ple = *sc.ple_emb.device_ptr() as u64;
        let hp = {
            let hist = self.ple_hist.lock().unwrap();
            (hist[slot * 2], hist[slot * 2 + 1])
        };
        let st = sc.wp05.as_mut().unwrap();
        if st.inflight { self.dsync()?; st.inflight = false; }
        let w = st.w;
        anyhow::ensure!(m <= w, "wp05 verify stage: m {m} > width {w}");
        let pre = st.ple_pre.take();
        let op = st.off_ple();
        let have = match pre {
            Some((s, h, t)) if s == slot && h == hp && t.len() <= m && t[..] == toks_v[..t.len()] => t.len(),
            _ => 0,
        };
        if let Some(ple) = self.ple.as_ref() {
            // PERFFIX1: queue every SSD row of this verify at once (no-op for the RAM table)
            let t = |j: usize| -> i32 { if j >= 2 { toks_v[j - 2] } else if j == 1 { hp.1 } else { hp.0 } };
            ple.prefetch_h3s(&(have..m).map(|q| [t(q), t(q + 1), toks_v[q]]).collect::<Vec<_>>());
            for q in have..m {
                self.wp05_ple_row(st, hp, toks_v, q)?;
            }
            st.pin.u16s(op + m * 2560 * 2, (w - m) * 2560).fill(0);
            // the ring advance ple_build_embed performs (restored post-accept by mtp_verify)
            let mut hist = self.ple_hist.lock().unwrap();
            hist[slot * 2] = if m >= 2 { toks_v[m - 2] } else { hp.1 };
            hist[slot * 2 + 1] = toks_v[m - 1];
        } else {
            st.pin.u16s(op, w * 2560).fill(0);
        }
        {
            let t = st.pin.i32s(st.off_toks(), w);
            t.fill(0);
            t[..m].copy_from_slice(toks_v);
        }
        {
            let ps = st.pin.i32s(st.off_pos(), w);
            ps.fill(0);
            for (r, x) in ps[..m].iter_mut().enumerate() { *x = (p + r) as i32; }
        }
        {
            let sp = st.pin.i32s(st.off_slots(), 2 * w);
            sp.fill(0);
            for r in 0..m { sp[r * 2] = slot as i32; sp[r * 2 + 1] = (p + r) as i32; }
        }
        st.pin.i32s(st.off_sids(), w).fill(slot as i32);
        if ple_xcheck_on() {
            // the pageable path's exact vectors, PLE rows through the pre-WP05 decode
            let mut toks_w = toks_v.to_vec();
            toks_w.resize(w, 0);
            let mut pos_v: Vec<i32> = (p..p + m).map(|x| x as i32).collect();
            pos_v.resize(w, 0);
            let mut slotpos: Vec<i32> = vec![0; 2 * w];
            for r in 0..m { slotpos[r * 2] = slot as i32; slotpos[r * 2 + 1] = (p + r) as i32; }
            let mut emb = vec![0u16; w * 2560];
            if let Some(ple) = self.ple.as_ref() {
                let mut h3s = Vec::with_capacity(m);
                let (mut a, mut b) = hp;
                for &tk in toks_v { h3s.push([a, b, tk]); a = b; b = tk; }
                ple.rows_par(&h3s, &mut emb, true)?;
            }
            let (ot, opos, osl, osi) = (st.off_toks(), st.off_pos(), st.off_slots(), st.off_sids());
            let tab_bad = (st.pin.i32s(ot, w) != &toks_w[..]) as usize + (st.pin.i32s(opos, w) != &pos_v[..]) as usize
                + (st.pin.i32s(osl, 2 * w) != &slotpos[..]) as usize
                + st.pin.i32s(osi, w).iter().filter(|&&x| x != slot as i32).count().min(1);
            let (n, first) = wp05_diff_u16(st.pin.u16s(op, w * 2560), &emb);
            crate::dprintln!("PLE_XCHECK verify slot {slot} m {m} (rows prebuilt during the passes: {have}): ple {n} mismatching f16 of {} (first (idx, new, old) {first:?}); tables differing {tab_bad} of 4",
                     w * 2560);
        }
        st.pin.htod(st.off_toks(), d_toks, w * 4)?;
        st.pin.htod(st.off_pos(), d_pos, w * 4)?;
        st.pin.htod(st.off_slots(), d_slots, w * 8)?;
        st.pin.htod(st.off_sids(), d_sids, w * 4)?;
        st.pin.htod(op, d_ple, w * 2560 * 2)?;
        st.inflight = true;
        Ok(hp)
    }
}
