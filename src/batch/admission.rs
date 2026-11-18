//! Request admission and prefill: `admit` and the chunked prefill cursor (`pf_advance`, `pf_finish`).
//! Another `impl BatchScheduler` block (file map in `mod.rs`); the methods keep their exact text.
//! Binding invariants: greedy MTP stays bitwise lossless; spec gating is per source (AGENTS.md section 4).

use super::*;

impl BatchScheduler {
    /// This scheduler does not compute log probabilities (the EXL3 serve path does). PR #30 port, owner rule
    /// (fall back, never refuse): the request is served without them. The HTTP layer already decides this
    /// (AppState logprobs_ok = false: `logprobs: null`, counted); this is the backstop for direct callers.
    pub(super) fn refuse_logprobs(mut req: BatchRequest) -> Option<BatchRequest> {
        if req.logprobs.take().is_some() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::reprintln!("[serve] logprobs: this scheduler computes none; such requests are served \
without them (this line prints once per boot)"));
        }
        Some(req)
    }

    pub(super) fn admit(&mut self, req: BatchRequest) {
        crate::think_budget::note_unsupported(req.thinking_budget, "NVFP4"); // THINKBUDGET1: the EXL3 server enforces it
        let Some(req) = Self::refuse_logprobs(req) else { return };
        // W2: the request's compiled schema (moved into the lane below).
        let req_schema = req.schema.clone();
        // R9: register the live gpu+state once so net::agree's mismatch path can dump GDN state.
        if crate::opts::var(crate::opt!("tp-diag")).is_ok() { r9_register_state(&self.gpu, &self.state); }
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
        enum From_ { Live, Ckpt, Ring(usize) }
        // Image-content identities for this request (empty for text-only). A prefix candidate whose
        // reused range includes a DIFFERENT image is discarded below, so the visual content of the
        // shared prefix is never replayed from a stale image.
        let req_imgs = request_image_identities(req.image_embeds.as_ref(), &req.image_spans);
        // Part-B (load invariance): reuse ONLY at ring-grid points (absolute-position multiples of
        // RING_CKPT_STRIDE). A Live/Ckpt hit resumes at an arbitrary length, which puts the first
        // prefill window's start mid-grid — its width differs run-to-run with the cache state, and
        // window width perturbs the prefill hiddens by ulps (the documented chunked-prefill caveat).
        // That made the SAME request's token stream depend on WHO ELSE had used the box: the
        // load-invariance failure. Ring-only reuse makes the window decomposition a pure function
        // of plen — [0,512),[512,1024),...,[512k,plen) — identical for cold, warm, solo, and
        // loaded runs, because the ring snapshot at 512k was itself produced by that same
        // window sequence (snapshots are bit-exact). Cost: reuse precision drops from token-exact
        // to the 512-token grid — a hit re-prefills up to RING_CKPT_STRIDE-1 extra tokens
        // (<=0.75 s at ~690 tok/s TP2 prefill; typical ~0.35 s), and verbatim-replay clients
        // (opencode) re-prefill the round-down remainder instead of skipping 100%.
        // --prefix-grid-reuse=0 restores token-exact Live/Ckpt reuse (diagnostics-only A/B).
        let grid_reuse = crate::opts::var(crate::opt!("prefix-grid-reuse")).map_or(true, |v| v != "0");
        let best = if !self.prefix_cache { None } else { self.free_slots.iter().enumerate()
            .flat_map(|(i, &sl)| {
                let live = common_prefix_len(&self.slot_cache[sl], &req.prompt);
                let ckpt = common_prefix_len(&self.slot_ckpt_seq[sl], &req.prompt);
                let mut cands = Vec::new();
                if !grid_reuse {
                    cands.push((i, sl, live, From_::Live, self.slot_cache[sl].len(), live));
                    cands.push((i, sl, ckpt, From_::Ckpt, self.slot_ckpt_seq[sl].len(), live));
                }
                // A6 ring checkpoints: entry j holds the GDN state at absolute length
                // slot_ring_len[sl][j] — reusable when that whole prefix matches (lcp ≥ len).
                // The live LCP against slot_cache bounds every ring candidate at once.
                if live > 0 {
                    for (j, &rlen) in self.slot_ring_len[sl].iter().enumerate() {
                        if rlen > 0 && rlen <= live && rlen < req.prompt.len() {
                            cands.push((i, sl, rlen, From_::Ring(j), rlen, live));
                        }
                    }
                }
                cands
            })
            .filter(|&(_, sl, l, from, seq_len, _live)|
                l > 0 && match from {
                    // live/ckpt: the state sits at the END of the cached sequence — the cache
                    // must be a strict prefix (at least one token left to prefill).
                    From_::Live | From_::Ckpt => l == seq_len && l < req.prompt.len(),
                    // ring: the state sits at rlen — already filtered to rlen ≤ lcp above.
                    From_::Ring(_) => true,
                }
                && match from {
                    From_::Live => images_compatible(&self.slot_cache_images[sl], &req_imgs, l),
                    From_::Ckpt => images_compatible(&self.slot_ckpt_images[sl], &req_imgs, l),
                    From_::Ring(_) => images_compatible(&self.slot_cache_images[sl], &req_imgs, l),
                })
            .max_by_key(|&(_, _, l, _, _, _)| l) };

        let (phys, reuse, from, live_at_admit) = match best {
            Some((idx, sl, l, f, _, live)) => { self.free_slots.remove(idx); (sl, l, Some(f), live) }
            None => match self.free_slots.pop() {
                Some(s) => (s, 0, None, common_prefix_len(&self.slot_cache[s], &req.prompt)),
                None => {
                    // NO SILENT DROPS (P10 lesson): dropping a BatchRequest drops its `tx`, which
                    // closes the client's SSE stream with no tokens and no error — the same
                    // client-visible symptom as the pf8 OOB class. Capacity is the CALLER's job, so
                    // reaching here means the caller's accounting and the slot table disagree;
                    // say so, and tell the client instead of hanging it.
                    crate::reprintln!("[req] DROPPED: no free physical slot (free_slots=0) — caller capacity                                accounting disagrees with the slot table");
                    let _ = req.tx.send(TokEvent::Finish { reason: "capacity".to_string() });
                    return;
                }
            },
        };

        // Resuming from the checkpoint (or a ring entry) means winding the slot's GDN state BACK
        // to that point. (KV is untouched: positions 0..reuse still hold this very prefix's keys,
        // and the suffix prefill overwrites everything after.)
        match from {
            Some(From_::Ckpt) => {
                self.gpu.copy_gdn_slot(&self.state, self.prompt_ckpt_slot + phys, phys);
            }
            Some(From_::Ring(j)) => {
                self.gpu.copy_gdn_slot(&self.state, self.ring_ckpt_slot + phys * RING_CKPT_K + j, phys);
                if crate::opts::var(crate::opt!("dump-pfhash")).is_ok() {
                    // Phase-8 [pfhash]: state ACTUALLY sitting in the slot right after the ring
                    // restore — compare against [ring-hash] printed at snapshot time (cold run).
                    eprintln!("[pre-hash] reuse={}{}", reuse, self.gpu.pf_hash(&self.state, phys, None));
                }
            }
            _ => {}
        }
        // Ring entries past the reuse point are invalidated: their positions get re-occupied by
        // the new prefill's snapshots. Entries at lengths ≤ reuse stay valid (state is a pure
        // function of the token prefix, which did not change).
        let src_early = self.mtp.spec_source();
        for rlen in self.slot_ring_len[phys].iter_mut() {
            if *rlen > reuse { *rlen = 0; }
        }
        // A6: the DSpark ring rows [0, reuse) are exactly valid when the previous dspark lane in
        // THIS slot covered them (single-tenant b==1 ring; rows are a pure function of the token
        // prefix). Carry the ring over (rewind nprev) instead of degrading the lane to MTP.
        // IDENTITY (PLAN/DSPARK_RING_IDENTITY_SPEC.md): length alone is NOT sufficient — the round
        // is ONE shared buffer while this bookkeeping is per slot, so an interleaved request on
        // another slot can leave the ring holding a foreign prefix. Require the claim to name THIS
        // slot; otherwise `rewind` would preserve rows that are not this prefix's and the lane
        // would draft from them (a tau loss, invisible to every losslessness gate).
        let dspark_ring_slot = self.dspark.as_ref().and_then(|d| d.ring_slot());
        // Diagnostics-only negative control (PLAN/DSPARK_RING_IDENTITY_SPEC.md §4 gate 2): drop
        // ONLY the identity term, reproducing the pre-guard length-only carry so the hazard can be
        // made to bite (the twin of DF2's --df2-carry-len-only). Never set outside a gate run.
        let dspark_ring_blind = crate::opts::var(crate::opt!("dspark-ring-blind")).is_ok();
        let dspark_carry = reuse > 0 && src_early == SpecSource::Dspark
            && self.slot_dspark_len[phys] >= reuse
            && (dspark_ring_slot == Some(phys) || dspark_ring_blind);
        if !dspark_carry {
            self.slot_dspark_len[phys] = 0;
            // The claim must be dropped alongside the length — every path that today zeroes the
            // length invalidates the identity too (mirrors `Df2Round::invalidate_ring`).
            if let Some(d) = self.dspark.as_mut() { d.invalidate_ring(); }
        }

        // On a miss, report the prefix we COULD have reused. This is the only way to see the size of
        // the opportunity we are leaving on the floor — and it is what exposed the 88% waste.
        if reuse == 0 {            if let Some((best, cached)) = self.free_slots.iter().chain(std::iter::once(&phys))
                .flat_map(|&sl| [(common_prefix_len(&self.slot_cache[sl], &req.prompt), self.slot_cache[sl].len()),
                                 (common_prefix_len(&self.slot_ckpt_seq[sl], &req.prompt), self.slot_ckpt_seq[sl].len())])
                .max_by_key(|&(l, _)| l) {
                if best > 64 {
                    crate::reprintln!("[req] prefix MISS: {} of {} prompt tokens match a cached sequence of {} \
                               — unusable, the GDN recurrent state only exists at token {}",
                              best, req.prompt.len(), cached, cached);
                }
            }
        }
        let greedy = req.temperature < 1e-6;
        let temperature = req.temperature;
        let top_p = req.top_p;
        let top_k = req.top_k;
        // TOPKM1 (PR #29 follow-up): on the batched GPU samplers a request's top_k 0 ("no top-k
        // truncation", also what the server maps a negative top_k to) and any top_k above their
        // candidate limit are served as top-64 (crate::sampler::gpu_top_k). Count it and say so once
        // per boot, so a client that asked for the full law can see what it got. The CPU sampler
        // (--cpu-sample) applies no limit; greedy and top_k 1 never reach a sampler.
        if self.gpu_sample && !greedy && top_k != 1 && crate::sampler::gpu_top_k_capped(top_k) {
            crate::metrics::top_k_capped();
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                crate::reprintln!("[req] top_k {} requested ({}): this model family's GPU sampler keeps the {} most likely tokens \
                                   per step (its candidate limit), so this request and every later one like it is served with \
                                   top_k {}; the CPU sampler (--cpu-sample) applies no limit. Counted in \
                                   velogb10_top_k_capped_total; this line prints once per boot.",
                                  top_k, if top_k == 0 { "no top-k truncation" } else { "above the limit" },
                                  crate::sampler::GPU_SAMPLE_K_MAX, crate::sampler::GPU_SAMPLE_K_MAX);
            }
        }
        let domain = req.domain;
        let tx = req.tx;
        let plen = req.prompt.len();
        let prompt = req.prompt;
        let max_new = req.max_new;
        let min_new = req.min_new;
        let ignore_eos = req.ignore_eos;
        let req_ckpt_at = req.ckpt_at;
        let rep_penalty = req.rep_penalty;
        let presence_penalty = req.presence_penalty;
        let frequency_penalty = req.frequency_penalty;
        let _has_penalty = rep_penalty > 1.0 || presence_penalty > 0.0 || frequency_penalty > 0.0;
        let will_use_mtp = self.mtp.active();
        if mtp_dispatch_trace() {
            crate::reprintln!("[b31-trace] admit: active={} depth={} src={:?} reuse={} plen={}",
                      will_use_mtp, self.mtp.depth(), self.mtp.spec_source(), reuse, plen);
        }
        // S5F: does THIS lane take the DFlash2 path? Requires the source + a resident round + a
        // full prompt prime (reuse == 0 — a prefix-hit lane's ring cannot be trusted for the
        // reused prefix, so it falls back to MTP/batched; prefix-cache + DFlash2 is out of scope
        // for S5F and documented as such).
        let src = self.mtp.spec_source();
        // DF2_CARRY: the DFlash2 twin of the A6 prefix-cache carve-out. The exact conditions —
        // and why a length-only copy of DSpark's check would be wrong (the DF2 ring is MODULO,
        // DSpark's is LINEAR) — are documented on `df2_carry_ok` and unit-tested in `mod df2_carry`.
        // Capture the ring state BEFORE any invalidation below: the log must report the guard's
        // INPUTS, not the state it leaves behind (reading it after made a refused carry print as
        // if the ring had never been bound).
        let (ring_slot_in, ring_len_in) = self.df2.as_ref()
            .map(|d| (d.ring_slot(), d.ring_len())).unwrap_or((None, 0));
        let df2_carry = self.df2.is_some() && df2_carry_ok(
            self.df2_carry_enabled, is_df2_src(src), reuse, plen, phys,
            ring_slot_in, ring_len_in, crate::opts::var(crate::opt!("df2-carry-len-only")).is_ok());
        // Does the ring's claim on THIS slot survive this admit?
        //
        // The claim is "rows [N-RING, N) hold the k/v of tokens [0, N) of this slot's cached
        // sequence". This admit replaces that cached sequence with the new prompt, so the claim
        // holds afterwards exactly when the new prompt still agrees with the sequence the ring was
        // built from over the whole range the claim covers — [0, min(N, plen)). `live_at_admit` is
        // the match length against the OLD cached sequence, so the test is `live >= min(N, plen)`.
        //
        // Dropping the claim whenever we simply do not carry is WRONG, and measurably so: a
        // shorter follow-up turn (plen < N — e.g. a retry, or a sub-agent re-sending only the
        // shared preamble) fails the frontier test for its own legitimate reason, and the blunt
        // rule then killed the claim for the rest of the session, pinning every later turn to MTP.
        // The refusal is per-request; the claim is about identity, and identity usually survives.
        let ring_claim_survives = ring_len_in > 0 && live_at_admit >= ring_len_in.min(plen);
        if !df2_carry && !ring_claim_survives
            && self.df2.as_ref().is_some_and(|d| d.ring_slot() == Some(phys)) {
            if let Some(d) = self.df2.as_mut() { d.invalidate_ring(); }
        }
        if reuse > 0 {
            let kind = match from {
                Some(From_::Live) => "live",
                Some(From_::Ckpt) => "ckpt",
                Some(From_::Ring(j)) => "ring",
                None => unreachable!(),
            };
            // DF2_CARRY telemetry (spec §6.7): print the guard's own INPUTS, so a refusal is
            // diagnosable rather than inferred. `ring_len` is the frontier N that `N <= plen`
            // bounds. All four booleans are what `df2_carry_ok` actually evaluated.
            crate::reprintln!("[req] prefix HIT: reuse={reuse} of plen={plen} ({kind}, dspark_carry={dspark_carry}, blind={dspark_ring_blind}, \
                       df2_carry={df2_carry} [enabled={} df2_src={} ring_slot={ring_slot_in:?} phys={phys} \
                       ring_len={ring_len_in} reuse<=len={} len<=plen={} live={live_at_admit} claim_kept={}])",
                      self.df2_carry_enabled, is_df2_src(src),
                      ring_len_in >= reuse, ring_len_in <= plen, ring_claim_survives);
            // S2 (REL_V0_7_3): tell the HANDLER too, not just the log — `cached_tokens` was
            // always 0 on the NVFP4 server because this event was never sent from this HIT
            // path (EXL3 sends it at its admit; see exl3_serve.rs `Admitted`). Once, right
            // here at admission, before any token is generated.
            let _ = tx.send(TokEvent::Admitted { cached_tokens: reuse as u32 });
        }
        // W2: the DFlash2 round is not schema-masked in this release — a schema request keeps the
        // plain decode path instead of an unmasked draft/verify round.
        let will_use_df2 = will_use_mtp && is_df2_src(src)
            && self.df2.is_some() && self.df2_prime.is_some()
            && (reuse == 0 || df2_carry) && req_schema.is_none();
        // WI1: the DSpark twin of will_use_df2. A6 softens the prefix carve-out: a prefix-hit
        // lane whose slot carries a valid DSpark ring for the reused range (dspark_carry) keeps
        // the DSpark path — the ring rows [0, reuse) are exactly valid and only the suffix gets
        // primed. Without a carried ring the old rule stands (the ring cannot be rebuilt from
        // stored trunk state: the drafter consumes trunk tap hiddens, which are not retained).
        // P14: the v1 lane runs only on a lane whose ctx feature covers the WHOLE prompt (the
        // drafter conditions on every committed position, and a prefix-cache hit skips the prefill
        // that fills it) and only for greedy requests (v1 is greedy-only by construction).
        let will_use_dflash = will_use_mtp && src == SpecSource::DFlash && self.dflash.is_some()
            && reuse == 0 && greedy && req_schema.is_none();
        self.dflash_prime_done = false;
        self.dflash_prime_armed = will_use_dflash;
        if src == SpecSource::DFlash && self.dflash.is_none() && !self.dflash_fallback_logged {
            eprintln!("[dflash] SpecSource=dflash but no v1 lane is resident — serving via MTP \
                       (install the artifact with --draft-dir <dir>, or --spec-source without a drafter to serve MTP)");
            self.dflash_fallback_logged = true;
        }
        if src == SpecSource::DFlash && self.dflash.is_some() && !will_use_dflash
            && !self.dflash_carry_logged {
            eprintln!("[dflash] v1 lane NOT primed for this request (prefix reuse {reuse}, greedy \
                       {greedy}): serving it with MTP — the v1 lane never runs on a partial ctx");
            self.dflash_carry_logged = true;
        }
        let will_use_dspark = will_use_mtp && src == SpecSource::Dspark
            && self.dspark.is_some() && self.dspark_prime.is_some() && (reuse == 0 || dspark_carry)
            && req_schema.is_none();
        if src == SpecSource::Dspark && self.dspark.is_none() && !self.dspark_fallback_logged {
            crate::reprintln!("[dspark] SpecSource=dspark but the DSpark round is NOT resident — serving via \
                       MTP per the standing-directive fallback (absent/failed artifact is never a \
                       hard failure)");
            self.dspark_fallback_logged = true;
        }
        if is_df2_src(src) && self.df2.is_none() && !self.df2_fallback_logged {
            crate::reprintln!("[df2] SpecSource=DFlash2 but the DFlash2 round is NOT resident — serving via \
                       MTP per the standing-directive fallback (absent/failed artifact is never a \
                       hard failure)");
            self.df2_fallback_logged = true;
        }
        // TTFT fix 0 attribution (--prefill-trace): admission-phase wall times. The window's
        // prefill_batch and mtp_prime_prompt each end with a sync, so their timers are honest
        // without extra syncs; `other` is the residue (pool trim, slot zeroing, lane bookkeeping).
        let received_at = req.received_at;
        let trace_pf = crate::opts::var(crate::opt!("prefill-trace")).ok().is_some();
        let admit_t0 = std::time::Instant::now();
        let mut pf_mark = admit_t0;
        let mut t_memsets = 0.0f64;
        let mut t_prefill = 0.0f64;
        let mut t_prime = 0.0f64;

        let cfg = self.gpu.cfg().clone();
        let h = cfg.hidden_size;

        // The KV cache is exactly `kv_stride` positions deep. `write_kv_prefill` had no bound check, so
        // an over-long prompt wrote past the end of it and corrupted the neighbouring allocation. The
        // server rejects these before they get here; this is the backstop for every other caller (the
        // bench paths admit requests directly).
        // B8 blocker B: the MTP draft/verify/re-prime step writes rows up to plen + max_new + depth,
        // so the PLAIN bound (max_new <= kv_stride - plen) is short by `depth` — a request that exactly
        // fills the context OOBs the MTP KV (the mtp_draft_step assert fires = a worker PANIC). Reserve
        // depth + 8 (the τ floor's slop) when MTP is active, using the policy MAX depth so a later
        // depth-upswitch can't cross the line mid-request. Plain decode needs no such headroom (1 slot
        // of slop). All inputs are replicated state, so both TP ranks reject identically.
        // (PR #4 expresses this as `decode_headroom(will_use_mtp)`; the depth it reserved is the
        // same MAX_AUTO_DEPTH + 8 — dev's clamp-before-reject refinement below is kept.)
        let mtp_headroom = decode_headroom(will_use_mtp);
        // BUG2 fix (2026-09-06, PLAN/FIX_8192_CHAT_TRUNCATION): CLAMP before rejecting. The server
        // grants room = max_seq_len - plen and cannot see this spec headroom, so with
        // --max-tokens >= max_seq_len every request that omitted max_tokens asked for the full
        // room and was REJECTED outright: completion_tokens=0, finish mapped to "length" — a
        // dead server for every default client (reproduced on master @ 286fdbd: plen 17 +
        // max_new 65519 + depth 8 + 8 > 65536). A request whose budget doesn't fit must still
        // generate what the KV cache holds (finish_reason: "length" is the contract for running
        // short); reject only when NOTHING fits (prompt + headroom alone fill the cache). The
        // bound this clamp enforces is exactly the one the reject used to enforce, so the
        // OOB-corruption protection is unchanged.
        let asked_max_new = max_new;
        let max_new = max_new.min(self.kv_stride.saturating_sub(plen + mtp_headroom));
        if max_new < asked_max_new {
            crate::reprintln!("[req] max_tokens clamped {} -> {} (KV stride {} − plen {plen} − spec headroom \
                       {mtp_headroom}; finish=length if generation runs out)",
                      asked_max_new, max_new, self.kv_stride);
        }
        let reject_msg = if plen >= self.kv_stride {
            Some(format!("prompt is {plen} tokens but the KV cache holds {} — raise --max-seq-len",
                         self.kv_stride))
        } else if max_new == 0 {
            Some(format!("plen {plen} + spec headroom {mtp_headroom} leaves no generation room in the \
                          KV stride {} (asked {asked_max_new}) — raise --max-seq-len",
                         self.kv_stride))
        } else {
            None
        };
        if let Some(msg) = reject_msg {
            crate::reprintln!("[req] REJECTED: {msg}");
            // Return the physical slot — it was popped above and losing it here permanently shrinks
            // capacity (handoff 6.10). Two consistency rules: (1) the push-back is deterministic, so
            // both TP ranks make the same next pick; (2) if the CKPT restore already ran, the slot's
            // LIVE state was wound back to the prompt checkpoint — the live metadata must follow it,
            // or a later Live pick would resume from a state the slot no longer holds.
            if from == Some(From_::Ckpt) {
                self.slot_cache[phys] = self.slot_ckpt_seq[phys].clone();
                self.slot_cache_images[phys] = self.slot_ckpt_images[phys].clone();
            }
            self.free_slots.push(phys);
            let _ = tx.send(TokEvent::Finish { reason: "context_length_exceeded".to_string() });
            return;
        }
        // (max_new was already clamped to the KV budget above the reject — BUG2 fix, 2026-09-06.)

        // Bound the pool. Safe here: `trim` synchronizes before freeing, and this runs before any GPU
        // work for this request. Without it the pool grows forever — see Pool::trim.
        self.pool.trim();

        // On a cache MISS, wipe the slot. On a HIT we must NOT: the GDN recurrent state and the conv1d
        // tail are exactly what we are reusing, and they only exist at the point the last request left
        // them. (KV beyond `reuse` is stale but unreachable — attention never reads past `pos`.)
        if reuse == 0 {
            self.gpu.zero_slot_state(&mut self.state, phys, self.kv_stride);
        } else {
            // E2 Fix 2: the hit leaves the slot's q4 dequant mirror intact and valid for [0, reuse),
            // but its rows PAST reuse are the previous request's — clamp the watermarks so this
            // request's windows re-dequant their own tail (KV past reuse gets overwritten).
            self.gpu.clamp_kv_mirrors(&mut self.state, phys, reuse);
        }
        let suffix = &prompt[reuse..];
        if reuse > 0 {
            crate::reprintln!("[req] prefix hit ({}): {}/{} tokens cached, prefilling {} ({:.0}% skipped)",
                      match from { Some(From_::Ckpt) => "prompt checkpoint",
                                   Some(From_::Ring(_)) => "ring grid",
                                   _ => "live state" },
                      reuse, plen, suffix.len(), 100.0 * reuse as f32 / plen as f32);
        }

        // MTP prompt-prime: over main positions 0..plen-2, step t writes (h_t, embed(prompt[t+1]))
        // into the lane's MTP KV at position t. Then seed the cursor hidden h_prev = h at plen-1.
        if will_use_mtp && reuse == 0 {
            // Miss: a previously-finished lane may have left speculative KV here. alloc_zeros doesn't
            // zero, and stale KV → nondeterministic drafts. Compute-stream memset.
            // (b3, EXPERT_TTFT_PREFILL_RESPONSE): zero only the positions the prime will write,
            // [0, plen) — positions >= plen are written by drafting before it attends. The cache is
            // head-major ([nkv, kv_stride, hd] bf16, head h at h·kv_stride·hd·2), so a contiguous
            // `nkv·plen·...` memset would only reach head 0 — one memset per head. The old full-cache
            // memset was 1.07 GB ≈ 4.2 ms of every MTP-on admission's TTFT; the full extent stays on
            // any path with `reuse > 0` (the CKPT-restore case, where the safe bound is less obvious).
            if trace_pf { pf_mark = std::time::Instant::now(); }
            let pos_bytes = plen * cfg.head_dim * 2;
            let head_stride = self.kv_stride * cfg.head_dim * 2;
            let kc_ptr = *self.mtp_kc[phys].device_ptr();
            let vc_ptr = *self.mtp_vc[phys].device_ptr();
            // §4.1b ROOT-CAUSE FIX: iterate the SHARD-AWARE head count. Under --tp-shard-mtp the
            // draft cache holds mtp_kv_heads() heads (1 at world=4), but this loop ran
            // cfg.num_kv_heads (4) — memsets at +8/+16/+24 MB past the buffer, corrupting whatever
            // followed in the VA layout (TP flags/rings/pools). That corruption WAS the entire
            // "first-MTP-step transport stall": proxies wedged, payloads arrived empty, K2' tails
            // timed out, and the memsets themselves faulted at ...aa000 when they crossed unmapped
            // VA (GPU coredumps: memset32, 8704 B = plen·hd·2, deterministic grid). Every alloc
            // site must use mtp_kv_heads() — this was the one loop that didn't.
            let nkv = self.gpu.mtp_kv_heads();
            for h in 0..nkv {
                self.gpu.memset_compute_stream(kc_ptr + (h * head_stride) as u64, pos_bytes);
                self.gpu.memset_compute_stream(vc_ptr + (h * head_stride) as u64, pos_bytes);
            }
            if trace_pf { t_memsets += pf_mark.elapsed().as_secs_f64(); }
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
        // The message-boundary checkpoint (prefix cache) is honoured by snapshotting the GDN state
        // at `c` when a window ends there. Part-B (load invariance): under grid reuse a window is
        // NEVER SPLIT at `c` — window ends stay a pure function of absolute position (the next
        // RING_CKPT_STRIDE multiple, then plen), never of cache state. The snapshot still fires
        // when `c` coincides with a window end; otherwise slot_ckpt_seq is refreshed at prompt end
        // (it is diagnostics-only now — Ckpt no longer resumes anything under grid reuse).
        let ckpt_at = req_ckpt_at.filter(|_| self.prefix_cache).filter(|&c| c > reuse && c < plen);
        let mut first_sent = false;
        let mut w0 = reuse;
        // window by window (the prefill captures them into the wide prime sink; the round consumes
        // failure).
        // Community report 2026-10-02 (27B + DFlash2, --max-batch 2: "df2 ring nprev 512 != lane
        // pos 713", scheduler panic, exit 70): the round drafters (DFlash2 / DSpark / DFlash v1) own
        // ONE shared ring, and the reset/rewind below hands it to THIS request. A live lane still
        // drafting from it would read this request's prime as its own frontier; it goes stale here
        // and takes the MTP fallback (the standing degrade path, never a panic).
        if will_use_df2 || will_use_dspark || will_use_dflash {
            let mut taken = 0usize;
            for l in self.lanes.iter_mut().flatten() {
                if l.df2_primed && !l.df2_stale { l.df2_stale = true; taken += 1; }
            }
            if taken > 0 {
                crate::reprintln!("[spec] round-drafter ring handed to the new request; {taken} live lane(s) continue on MTP");
            }
        }
        if will_use_df2 {
            if let Some(df2) = self.df2.as_mut() {
                // DF2_CARRY: on a carry, move the frontier to the reuse point and KEEP the ring
                // rows [0, reuse) — the suffix prime below only writes [reuse, plen), and its
                // ring rows are the complementary residues of the draft's read band, so the two
                // halves tile the band exactly. Cold start: full reset (which also drops the
                // identity claim). The window loop below already starts at w0 = reuse, and
                // prime_window sets nprev absolutely, so nothing else needs to change.
                if df2_carry { df2.rewind(reuse); } else { df2.reset(); }
            }
            if let Some(ps) = self.df2_prime.as_ref() {
                self.gpu.set_df2_prime_sink(ps.clone());
            }
        }
        if will_use_dspark {
            if let Some(ds) = self.dspark.as_mut() {
                // A6 carry: rewind to the reuse point (rows [0, reuse) are this very prefix's
                // drafter KV); cold start: full reset.
                if dspark_carry {
                    ds.rewind(reuse);
                } else {
                    ds.reset();
                }
            }
            if let Some(ps) = self.dspark_prime.as_ref() {
                self.gpu.set_dspark_prime_sink(ps.clone());
            }
        }
        // ---------------------------------------------------------------------------------------
        // PLAN/SCHEDULER_2LANE_WORKDOC.md §3.1 — hand the WINDOW LOOP to a per-lane cursor.
        //
        // Everything above this line is the DECISION half of admission and stays here: capacity,
        // the context-length reject, the prefix-cache/reuse pick, `will_use_*`/carry, the slot
        // wipe/zero, the MTP memsets, the DF2/DSpark reset-or-rewind and the prime-sink arming.
        // Everything below moves into `PfCursor` and is advanced by the step loop, so a long
        // prompt's prefill no longer runs to completion inside one scheduler iteration.
        //
        // ORDERING NOTE (the reason arming stays here): at most ONE cursor exists at a time
        // (`BatchScheduler::pf`), so admission can never overlap an in-flight prime — the shared
        // `df2`/`dspark` rounds are reset and armed exactly where they always were relative to
        // their own lane's windows.
        // ---------------------------------------------------------------------------------------
        //
        // V3 vision: upload the merged image embeddings. The splice is armed per WINDOW (in
        // `pf_advance`), never for the whole prefill: `state.vision_embeds` is a GLOBAL field, and
        // a co-resident lane's window or decode step must never see this request's splice (§3.4).
        let vision = match req.image_embeds {
            Some(emb) if !req.image_spans.is_empty() && !emb.is_empty() => {
                let hb: Vec<half::bf16> = emb.iter().map(|&x| half::bf16::from_f32(x)).collect();
                let buf = self.gpu.dev().htod_sync_copy(&hb).expect("vision htod");
                Some((buf, req.image_spans.clone()))
            }
            _ => None,
        };
        // Seed: the request's (a seedless request got a random one at `intake`), else — only under
        // --seedless-seed prompt-hash — a hash of the prompt (identical request => identical text).
        // (Kept at admission so the cursor owns the whole lane-install payload.)
        let seed = req.seed.unwrap_or_else(|| {
            use std::hash::{Hash, Hasher};
            let mut sh = std::collections::hash_map::DefaultHasher::new();
            prompt.hash(&mut sh);
            sh.finish()
        });
        debug_assert!(self.pf.is_none(),
                      "one prefill cursor at a time — see BatchScheduler::pf (the shared df2/dspark \
                       rounds are primed window-by-window and cannot be interleaved)");
        self.pf = Some(Box::new(PfCursor {
            phys, plen, reuse, w0: reuse, ckpt_at, grid_reuse, req_imgs,
            schema: req_schema.clone(),
            will_use_mtp, will_use_df2, will_use_dspark, df2_carry, dspark_carry,
            first_tok: 0, first_sent: false, pf_hash_str: String::new(),
            df2_primed_ok: true, dspark_primed_ok: true,
            tx, greedy, domain, temperature, top_p, top_k, max_new, min_new, ignore_eos,
            rep_penalty, presence_penalty, frequency_penalty, seed, received_at,
            trace_pf, admit_t0, t_memsets, t_prefill, t_prime, vision,
            prompt,
        }));
        // P11 W5a (--prefill-sched inline, owner-blessed escape): drain the whole prefill inside
        // this admission — the pre-P10 schedule. Decode lanes wait the FULL prompt (max gap = the
        // entire prefill, the posture the cursor replaced). A/B + emergency use; the cursor stays
        // the default. The window formula/order is untouched — schedule-only.
        if self.pf_inline {
            while self.pf.is_some() { self.pf_advance(); }
        }
    }

    /// Advance the in-flight prefill cursor by ONE window (SCHEDULER_2LANE §3.1/§3.3).
    ///
    /// This is the `while w0 < plen` body of `admit()` moved VERBATIM (§3.2 — the window formula is
    /// a NUMERIC contract, not a schedule: the pf8 padding family fails only when the pool's
    /// power-of-two capacity lands in a width gap, so "same windows, different order" is the whole
    /// safety argument and a re-derived formula would silently re-open the §0.2 counter-case #2
    /// class). No-op when nothing is prefilling.
    ///
    /// Budget policy (§3.3 option 1): ONE window per loop iteration, no knob. Deterministic — a
    /// pure function of scheduler state — which is what `run_tp_mirror`'s replay-by-wire contract
    /// requires (a wall-clock or memory-pressure budget would desync the ranks).
    pub(super) fn pf_advance(&mut self) {
        let Some(mut c) = self.pf.take() else { return };
        let phys = c.phys;
        let (plen, w0) = (c.plen, c.w0);
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        if w0 >= plen {
            // reuse == plen (a full prefix hit): today's inline loop never ran either, so the
            // post-loop tail must still execute (it streams the first token and installs the lane).
            self.pf_finish(*c);
            return;
        }
        let mut w1 = (w0 + PREFILL_CHUNK).min(plen);
        if !c.grid_reuse {
            // Token-exact reuse (diagnostics): resume points are arbitrary, so honour the
            // message boundary by splitting a window at `c` (the pre-grid-reuse behavior).
            if let Some(ck) = c.ckpt_at { if w0 < ck && ck < w1 { w1 = ck; } }   // stop at the boundary
        }
        // A6: stop at the next ring-checkpoint boundary so the GDN state passes exactly
        // through it and can be snapshotted below (prefix-cache re-chunking is the
        // documented caveat; the ring slack is bounded by RING_CKPT_STRIDE).
        if self.prefix_cache {
            let rb = ((w0 / RING_CKPT_STRIDE) + 1) * RING_CKPT_STRIDE;
            if w0 < rb && rb < w1 { w1 = rb; }
        }
        // PLAN/TRIPWIRE_SPEC.md §3.2 + SCHEDULER_2LANE §5 gate 4: the WINDOW CENSUS. `--pool-census`
        // only, and NEVER on for a timing run. This is the artifact the scheduler's P4 gate diffs:
        // the multiset of `n = w1 - w0` values `prefill_batch` is presented for a fixed prompt mix,
        // plus the pf8 pad arithmetic each one implies (the pool-luck class of §0.2 counter-case #2).
        crate::gpu::pf_census_window(w0, w1, h);

        // V3 vision: arm the splice for THIS window only. `state.vision_embeds` is global (the
        // splice is consumed host-side inside `prefill_batch`), so it is armed immediately before
        // and disarmed immediately after — a co-resident lane's window must never inherit it.
        match c.vision.as_ref() {
            Some((buf, spans)) => {
                self.state.vision_embeds = Some(buf.clone());
                self.state.vision_spans = spans.clone();
            }
            None => { self.state.vision_embeds = None; self.state.vision_spans.clear(); }
        }

        let mut pf_mark = std::time::Instant::now();
        // P14: the v1 lane's ctx feature is filled by the trunk's own tap capture during the REAL
        // prefill (a position-addressed D2D per tapped layer per window, inside
        // `prefill_batch_range`) — no token-by-token prefill anywhere. NOTE: that capture site was
        // MISSING until the P14 repair; this comment described the design while the code primed the
        // lane from a surface no forward had written (see the capture site's comment in gpu.rs).
        if self.dflash_prime_armed { self.gpu.set_dflash_wide_pos(w0); }
        let (tok, hw) = self.gpu.prefill_batch(
            &mut self.pool, &c.prompt[w0..w1], &mut self.state, phys, self.kv_stride, w0);
        self.state.vision_embeds = None;
        self.state.vision_spans.clear();
        if c.trace_pf { c.t_prefill += pf_mark.elapsed().as_secs_f64(); }
        c.first_tok = tok;   // only the LAST window's token (at plen-1) is the prompt's next token
        if w1 == plen && self.dflash_prime_armed {
            // The prompt's columns [0, plen) are captured: hand them to the drafter (its KV append
            // happens lazily in the first draft step) and mark the lane primed.
            self.dflash_prime(w1);
            // The lane struct is created later in the admit path, so record the prime on the
            // scheduler and let the dispatch accept either flag (the v1 lane is b == 1 only).
            if let Some(l) = self.lanes[0].as_mut() { l.df2_primed = true; }
            self.dflash_prime_done = true;
            self.dflash_prime_armed = false;
        }
        if w1 == plen && crate::opts::var(crate::opt!("dump-pfhash")).is_ok() {
            // Phase-8 [pfhash]: warm-vs-cold prefill equality probe (diagnostics).
            c.pf_hash_str = self.gpu.pf_hash(&self.state, phys, Some((&hw, h, w1 - w0)));
        }

        // TTFT (b2, EXPERT_TTFT_PREFILL_RESPONSE): stream the first token the moment the last
        // window's prefill produced it — BEFORE the MTP prime + cursor copy below. The prime
        // only gates DRAFTING, which runs in decode_step after admission completes, so ordering
        // is preserved and semantics are unchanged; the client just sees its first chunk
        // ~6-9 ms earlier on MTP-on servers (P10 removed from the TTFT window).
        if w1 == plen && !c.first_sent {
            // W2 (Phase 13): a schema-constrained lane does NOT emit the prefill's own first
            // token. The prefill selection has no schema mask (it is a different code path), so
            // that token is unconstrained — observed as a leading '```' in the first W2 boot.
            // It becomes the FEED token of the first masked decode step instead, which means
            // every token the client sees came from a masked selection.
            if c.schema.is_some() {
                c.first_sent = true;
            } else {
                let _ = c.tx.send(TokEvent::Tok(c.first_tok));
                c.first_sent = true;
            }
            // The mirror replays the same admit and prints its own line to the node log;
            // the head's line is the measurement's signal. (tp_serving is true on BOTH ranks
            // in TP serving mode, so it cannot gate this — only rank selection could, and
            // the two lines are harmless in separate logs.)
            crate::reprintln!("[req] ttft={:.1}ms plen={}",
                      c.received_at.elapsed().as_secs_f64() * 1000.0, plen);
        }

        if c.will_use_mtp {
            // MTP prime pairs hidden[t] with token[t+1] for t in [w0, min(w1, plen-1)); position
            // plen-1 is never primed (no token plen to pair). hw's columns 0.. map to positions w0..
            let tok_end = w1.min(plen - 1);
            if tok_end > w0 {
                let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
                let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
                if c.trace_pf { pf_mark = std::time::Instant::now(); }
                self.gpu.mtp_prime_prompt(&mut self.pool, &hw, &c.prompt[w0 + 1..tok_end + 1],
                                          mtp_kc_ptr, mtp_vc_ptr, self.kv_stride, w0);
                if c.trace_pf { c.t_prime += pf_mark.elapsed().as_secs_f64(); }
            }
            // Cursor hidden = pre-norm h at the LAST prompt position, i.e. last column of the
            // final window.
            if w1 == plen {
                self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &hw, (w1 - w0) - 1);
            }
        }
        if c.will_use_dspark {
            // WI1: the DSpark prime twin — same window taps (dspark::TAP_LAYERS captures),
            // same large-M gemm_tiled path, same fail-soft contract.
            if let (Some(ds), Some(ps)) = (self.dspark.as_mut(), self.dspark_prime.as_ref()) {
                if let Err(e) = ds.prime_window(&ps.taps, w1 - w0, w0) {
                    crate::reprintln!("[dspark] prompt prime window {w0}..{w1} FAILED ({e:#}) — this lane \
                               will NOT take the DSpark path (falls back to MTP/batched)");
                    c.dspark_primed_ok = false;
                }
            }
        }
        if c.will_use_df2 {
            // S5F: prime the DFlash2 ring with THIS window's taps. prefill_batch synced at its
            // tail, so the window's capture D2Ds are complete before the round reads them.
            if let (Some(df2), Some(ps)) = (self.df2.as_mut(), self.df2_prime.as_ref()) {
                if let Err(e) = df2.prime_window(&ps.taps, w1 - w0, w0) {
                    crate::reprintln!("[df2] prompt prime window {w0}..{w1} FAILED ({e:#}) — this lane \
                               will NOT take the DFlash2 path (falls back to MTP/batched)");
                    c.df2_primed_ok = false;
                    // DF2_CARRY: a failed prime can leave the ring half-overwritten, so the
                    // identity claim is no longer true — drop it rather than let a later
                    // request carry a partially primed ring.
                    df2.invalidate_ring();
                }
            }
        }
        self.pool.release_bf16(hw, h * (w1 - w0));

        // Snapshot the GDN state at the message boundary (prefix cache), before the next window.
        if Some(w1) == c.ckpt_at {
            self.gpu.copy_gdn_slot(&self.state, phys, self.prompt_ckpt_slot + phys);
            self.slot_ckpt_seq[phys] = c.prompt[..w1].to_vec();
            self.slot_ckpt_images[phys] = c.req_imgs.iter()
                .filter(|x| x.start < w1).copied().collect();
        }
        // A6: ring checkpoint at the boundary — round-robin over the K most-recent entries.
        // w1 < plen keeps the prompt-end state in the live/ckpt slots (it must stay the
        // generation-resume state, not a ring candidate).
        if self.prefix_cache && w1 < plen && w1 % RING_CKPT_STRIDE == 0 {
            let j = self.ring_cursor[phys];
            self.ring_cursor[phys] = (j + 1) % RING_CKPT_K;
            self.gpu.copy_gdn_slot(&self.state, phys,
                self.ring_ckpt_slot + phys * RING_CKPT_K + j);
            self.slot_ring_len[phys][j] = w1;
            if crate::opts::var(crate::opt!("dump-pfhash")).is_ok() && (w1 % 1024 == 0 || w1 >= plen - 512) {
                // Phase-8 [pfhash]: what the ring entry ACTUALLY holds at snapshot time, plus
                // the live slot state it was copied from — both hashed after the copy lands.
                crate::reprintln!("[ring-hash] w1={} j={} src{} ring{}",
                          w1, j, self.gpu.pf_hash(&self.state, phys, None),
                          self.gpu.pf_hash(&self.state, self.ring_ckpt_slot + phys * RING_CKPT_K + j, None));
            }
        }
        crate::metrics::sched_touch(); // H3: prefill chunk COMPLETE (long prefill stays live)
        c.w0 = w1;
        if c.w0 >= plen {   // the last window completed: the lane becomes visible NOW (§3.4)
            self.pf_finish(*c);
        } else {
            self.pf = Some(c);
        }
    }

    /// Complete an admission whose last window has run: `admit()`'s post-loop tail, moved verbatim.
    ///
    /// This is the ONLY place a partially-prefilled request becomes visible to the rest of the
    /// scheduler (§3.4): the prefix-cache entry (`slot_cache`/`slot_ckpt_seq`) is written here, not
    /// per window, so the matcher can never see a half-prefilled length; and the lane itself is
    /// installed here, so `num_active()`/`b`, the speculation gates and the cancel sweep all treat
    /// a mid-prefill request as if it did not exist.
    fn pf_finish(&mut self, mut c: PfCursor) {
        let phys = c.phys;
        self.state.vision_embeds = None;
        self.state.vision_spans.clear();

        if !c.pf_hash_str.is_empty() {
            crate::reprintln!("[pfhash] plen={} reuse={} phys={} a0={}{}",
                      c.plen, c.reuse, phys, c.first_tok, c.pf_hash_str);
        }

        if self.prefix_cache && (c.ckpt_at.is_none() || c.grid_reuse) {
            // No boundary snapshot fired inside the suffix (a raw/non-chat request, one whose whole
            // prompt was already cached, or — grid reuse — a boundary that is not a window end
            // anymore). Snapshot where we ended; it is still the prompt boundary for THIS prompt.
            self.gpu.copy_gdn_slot(&self.state, phys, self.prompt_ckpt_slot + phys);
            self.slot_ckpt_seq[phys] = c.prompt.clone();
            self.slot_ckpt_images[phys] = c.req_imgs.clone();
        }

        // The slot's state now reflects the whole prompt. Decode extends this as tokens commit.
        self.slot_cache[phys] = c.prompt.clone();
        self.slot_cache_images[phys] = c.req_imgs.clone();
        // S5F: the DFlash2 prime is done — the prefill capture must not run for later lanes or
        // requests (it is a per-admit arm, disarmed here regardless of success).
        if c.will_use_df2 { self.gpu.set_df2_prime_off(); }

        let slot = self.num_active();
        if !c.first_sent {
            // No window ran (reuse == plen full-hit edge) — the loop never produced a token.
            if c.schema.is_some() {
                // W2: same rule as the w1 == plen site — the token stays internal (feed), the
                // client's first token comes from the masked decode path.
                c.first_sent = true;
            } else {
                let _ = c.tx.send(TokEvent::Tok(c.first_tok));
            }
            crate::reprintln!("[req] ttft={:.1}ms plen={}",
                      c.received_at.elapsed().as_secs_f64() * 1000.0, c.plen);
        }
        if c.trace_pf {
            let other = c.admit_t0.elapsed().as_secs_f64() - c.t_memsets - c.t_prefill - c.t_prime;
            crate::reprintln!("[pf-admit] plen={} memsets={:.2}ms prefill={:.2}ms prime={:.2}ms other={:.2}ms",
                      c.plen, c.t_memsets * 1000.0, c.t_prefill * 1000.0, c.t_prime * 1000.0,
                      other * 1000.0);
        }

        // PLAN/25 Phase 0: serving-side job marker. Only opens when no job is open (the bench
        // harness emits its own markers; a job left open by a mid-admit reject self-heals at the
        // next lane finish). Records interleave if lanes overlap — max-batch 1 is the clean case.
        if let Some(d) = self.step_dump.as_mut() {
            if !d.job_open() {
                self.cov_req_n += 1;
                let tag = format!("req{:04}", self.cov_req_n);
                d.job_start(&tag, &c.prompt, self.gpu.dev());
            }
        }
        // LOOPFIX: a penalized lane starts its slot's whole-output penalty state from scratch —
        // zero output counts, and the prompt's tokens marked when its repetition penalty is on
        // (vLLM applies repetition to prompt + output). Unpenalized lanes never touch the state.
        let pen_on = c.rep_penalty > 1.0 || c.presence_penalty > 0.0 || c.frequency_penalty > 0.0;
        if pen_on {
            let prompt: &[u32] = if c.rep_penalty > 1.0 { &c.prompt } else { &[] };
            self.gpu.pen_slot_init(&mut self.bufs, phys, prompt);
        }
        self.lanes[slot] = Some(Lane {
            phys, pos: c.plen, last_tok: c.first_tok, max_new: c.max_new,
            // W2: a schema lane emitted nothing yet (its prefill token is internal) — count from 0.
            generated: if c.schema.is_some() { 0 } else { 1 },
            greedy: c.greedy, domain: c.domain, temperature: c.temperature,
            top_p: c.top_p, top_k: c.top_k,
            rep_penalty: c.rep_penalty, presence_penalty: c.presence_penalty,
            frequency_penalty: c.frequency_penalty, min_new: c.min_new, ignore_eos: c.ignore_eos,
            history: vec![c.first_tok], tx: c.tx,
            // The prefill token is the first output token; a penalized lane counts it at its first sync.
            pen_total: 1, pen_synced: if pen_on { 0 } else { 1 },
            schema_state: c.schema.as_ref().map(|sm| sm.start_state()).unwrap_or(0),
            schema: c.schema.clone(),
            mtp_pos: c.plen.saturating_sub(1),
            mtp_primed: c.will_use_mtp,
            mtp_stale: false,
            df2_primed: (c.will_use_df2 && c.df2_primed_ok) || (c.will_use_dspark && c.dspark_primed_ok),
            df2_stale: false,
            seed: c.seed,
            tp_cancelled: false,
            copy: None,
            in_reason: think_state_after(think_state_after(false, &c.prompt), &[c.first_tok]),
            acct: [[(0, 0, 0); 3]; 2],
            acct_depth: [[0; MAX_AUTO_DEPTH + 1]; 2],
        });
        if mtp_dispatch_trace() {
            crate::reprintln!("[b31-trace] lane install: slot={} pos={} mtp_primed={} df2_primed={} greedy={} schema={}",
                      slot, c.plen, c.will_use_mtp,
                      (c.will_use_df2 && c.df2_primed_ok) || (c.will_use_dspark && c.dspark_primed_ok),
                      c.greedy, c.schema.is_some());
        }
        // Device-resident loop: the lane composition changed — the next batched step must
        // re-upload the full buffer set (tokens/pos/slot_ids/ring/keys).
        self.resident_dirty = true;
        // S-B16: (re)arm the audit cursor for this request from the lane's INSTALL state —
        // gt_last/gt_pos must be the prefill token and its position, captured BEFORE any MTP step
        // advances the lane. The shadow's zero+prime decode is deferred to the first audited step
        // (the cursor is only consumed by lanes that actually take the MTP path).
        if self.audit_on {
            // The shadow must start from a REAL prefill of this request's prompt. Zeroed KV rows
            // are NOT softmax-neutral (exp(0)=1 per unwritten key): the v2 auditor's zero-only
            // shadow predicted token 0 at every request's first token (false divergence, caught
            // in the first v2 run). This mirrors bench_mtp_steps' slot-1 ground-truth prefill.
            self.gpu.zero_slot_state(&mut self.state, self.audit_slot, self.kv_stride);
            self.gpu.zero_slot_kv(&mut self.state, self.audit_slot, self.kv_stride);
            let (a0s, hs) = self.gpu.prefill_batch(&mut self.pool, &c.prompt, &mut self.state,
                                                   self.audit_slot, self.kv_stride, 0);
            let mw = self.gpu.mtp_hidden_width();
            self.pool.release_bf16(hs, mw * c.plen);
            if a0s != c.first_tok {
                crate::reprintln!("[lossless-audit] WARN req={} shadow prefill a0={} != lane a0={}",
                          self.audit_req_n, a0s, c.first_tok);
            }
            // The cursor's FIRST prediction must be real: with pred left at 0 the request's first
            // emission compares against token 0 (run4's false "seq=0" divergence at emitted#0).
            let pred0 = self.audit_shadow_decode(c.first_tok, c.plen, self.kv_stride);
            self.audit_cur[slot] = Some(MtpAuditCursor {
                gt_last: c.first_tok, gt_pos: c.plen, pred: pred0, idx: 0,
                tag: self.audit_req_n, diverged: false, primed: true,
            });
            self.audit_req_n += 1;
        }
    }
}
