//! Chunked prefill: the interleave and pack-prefill helpers, `prefill_range` / `prefill_plan` / `prefill_cached` / `prefill_finish`,
//! the turn-resume background pass (`turn_*`), `wp16_xcheck` and `mtp_prime`. Another `impl Exl3Scheduler` block.
//! Under TP the per-step agree guard stays load-bearing (AGENTS.md section 2, item 10).

use super::*;

impl Exl3Scheduler {
    /// Ranks of this scheduler's TP group for the world-aware DDS prior: 1 = no TP (an impossible tp-without-model
    /// pairing keeps the pre-TP-4F value, world 2).
    pub(super) fn tp_world(&self) -> usize {
        if self.tp.is_none() { 1 } else { self.model.tp_rank().map_or(2, |(_, w)| w as usize) }
    }
}

/// `--prefill-interleave` value from the command line: absent / empty = 0 (off).
pub(super) fn parse_prefill_interleave(v: Option<&str>) -> Result<usize> {
    match v.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(0),
        Some(v) => v.parse::<usize>().map_err(|e| anyhow::anyhow!("--prefill-interleave: {e} (decode steps per prefill chunk, 0 = off)")),
    }
}

/// The interleave the scheduler runs with, and the boot line that says so (None = flag off, no
/// line: default boot logs are unchanged). PERFFIX1: under TP too — every rank reads the value from
/// the head's shipped registry (`prefill-interleave` is SPMD, in the boot agree's digest) and runs the
/// SAME steps at the SAME chunk boundaries (the boundary is the per-chunk HeadFlag exchange; a step
/// is lockstep work on every rank, its cancels come from the head's outer loop), so no rank steps alone.
pub(super) fn resolve_prefill_interleave(n: usize, tp: bool, chunk: usize) -> (usize, Option<String>) {
    match (n, tp) {
        (0, _) => (0, None),
        (n, true) => (n, Some(format!("[exl3-serve] prefill interleave ON under TP (--prefill-interleave {n}): every rank runs up to {n} \
                                       scheduler step(s) for the lanes already decoding after each non-final prefill chunk (chunk {chunk} \
                                       tokens), in lockstep; the chunk grid and the joining request's bytes are unchanged, its first token is later"))),
        (n, false) => (n, Some(format!("[exl3-serve] prefill interleave ON (--prefill-interleave {n}): up to {n} scheduler step(s) for the \
                                        lanes already decoding after each non-final prefill chunk (chunk {chunk} tokens); \
                                        the chunk grid and the joining request's bytes are unchanged, its first token is later"))),
    }
}

/// Public issue #24: one lane's (positions used, positions cached) as /metrics reports them.
/// A running request uses `pos` positions (prompt + generated) and a prefill in flight the rows it
/// has written (a resumed prefix included); neither reads as cached, because the prefix snapshot is
/// part of what the request holds. A free lane uses nothing and keeps its snapshot's positions.
pub(super) fn kv_lane_occupancy(active_pos: Option<usize>, prefill_pos: Option<usize>, cache_len: Option<usize>) -> (usize, usize) {
    match active_pos.or(prefill_pos) {
        Some(p) => (p, 0),
        None => (0, cache_len.unwrap_or(0)),
    }
}

/// Does the hook run after the chunk that ended at `end` of a prefill to `n`? Never after the final
/// chunk (the seam step and the head prime follow it directly), never for a non-cancellable
/// internal prefill (`cancellable` false: the WP16 XCHECK's reference run). Every topology (PERFFIX1).
pub(super) fn interleave_point(interleave: usize, cancellable: bool, end: usize, n: usize) -> bool {
    interleave > 0 && cancellable && end < n
}

/// C1 v2: prefill scratch rows — the merged-chunk cap with tail checkpoints on (2C - 1; under TP
/// min(2C - 1, 4,095) so `--prefill-chunk 4095` fits the all-reduce partial buffer, TP-4X1), else C.
/// PACKPF (from GitHub andersc's PACK-PF): --prefill-pack on|off.
pub(super) fn packpf_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(crate::opts::var(crate::opt!("prefill-pack")).as_deref(), Ok("on") | Ok("1")))
}

/// PACKPF: --prefill-pack-rows N — the most rows one packed pass carries (0 = the prefill scratch's
/// rows: 2C - 1 with the tail checkpoints on, no extra memory; larger grows the scratch).
pub(super) fn packpf_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("prefill-pack-rows")).ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// PACKPF: --prefill-pack-min K — with K or more requests waiting and fewer than K free lanes while
/// lanes are decoding, admission waits for lanes (a decode round) so the next packed pass carries K.
pub(super) fn packpf_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("prefill-pack-min")).ok().and_then(|v| v.parse().ok()).unwrap_or(2))
}

/// PACKPF: --prefill-pack-wait on — while lanes decode, also hold a queue shorter than
/// --prefill-pack-min so passes fill up (throughput mode).
pub(super) fn packpf_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(crate::opts::var(crate::opt!("prefill-pack-wait")).as_deref(), Ok("on") | Ok("1")))
}

/// PACKPF: --prefill-pack-wait-ms — the longest (since arrival) a request is held for a fuller pack.
pub(super) fn packpf_wait_ms() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("prefill-pack-wait-ms")).ok().and_then(|v| v.parse().ok()).unwrap_or(1000))
}

/// The prefill scratch rows to ALLOCATE: psc_rows (what the chunk grid may merge up to), grown to the
/// packed pass budget when packing is on. The grid keeps using psc_rows.
pub(super) fn psc_alloc_rows(c: usize, tp: bool) -> usize {
    let r = psc_rows(c, tp);
    if packpf_on() && !tp { r.max(packpf_rows()) } else { r }
}

pub(super) fn psc_rows(c: usize, tp: bool) -> usize {
    if crate::exl3_forward::tail_ckpt_on() && !crate::exl3_forward::wp16_off() {
        crate::exl3_forward::merge_cap(c, tp).max(c)
    } else { c }
}

/// FNV-1a over the prompt ids — a replay key for the [loop] log (not a security hash).
pub(super) fn prompt_hash(p: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in p {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl Exl3Scheduler {
    /// KVPOOL: free the position pages of ONE idle lane (no request on it): the SMALLEST first, the longest last. Its cached
    /// conversation goes to the SSD spill tier first (when on) — the same capture an admission takeover does — and its
    /// prefix bookkeeping is dropped, so a later prompt of that conversation restores from disk or prefills again.
    /// `keep`: lanes that must not be picked. Ok(false) = nothing to free. Single box only (TP ranks would have to agree).
    pub(super) fn vmm_evict_idle(&mut self, keep: &[usize]) -> Result<bool> {
        if !self.model.vmm_lazy() || self.tp.is_some() { return Ok(false); }
        let Some(&(rows, v)) = self.vmm_idle_victims(keep).first() else { return Ok(false) };
        let t0 = std::time::Instant::now();
        let cached = self.cache[v].take();
        if let Some(old) = cached.as_ref() {
            if self.spill.is_some() { self.spill_capture_slot(v, old)?; }
        }
        if let Some(st) = self.wp16.as_mut() { st.clear_slot(v); }
        self.segs_set(v, None, 0);
        self.pf_done.remove(&v); // PACKPF: a finished packed pass claims this lane's rows 0..d.end
        if let Some(a) = self.turn.as_mut() { a.toks[v] = None; a.pending[v] = None; } // TURNRESUME: ditto for the kept next-turn state
        self.model.vmm_truncate(v, 0)?;
        crate::kv_vmm::EVICTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::drprintln!("[kv-vmm] pool full: evicted idle slot {v} ({rows} rows mapped, {} cached tokens{}) in {:.2} s",
                          cached.as_ref().map_or(0, |c| c.len()),
                          if cached.is_some() && self.spill.is_some() { ", offered to the SSD spill tier" } else { "" },
                          t0.elapsed().as_secs_f64());
        self.publish_kv();
        Ok(true)
    }

    /// KVPOOL: the idle lanes `vmm_evict_idle` may take, in the order it takes them: (rows mapped, slot). Read-only,
    /// so TPPOOL's capacity gate (design v3 step 4) credits victims in exactly the eviction's order.
    /// PRIORITY (owner 2026-11-17): the longest idle session goes LAST. One long main conversation and short
    /// agent / sub-agent sessions share the pool; evicting the long one costs a multi-GB SSD round trip when its
    /// user comes back. So: smallest idle lane first, and the largest idle lane is kept while anything else
    /// idle can be freed. When it is the only idle lane left, it goes too (the alternative is failing a request).
    pub(super) fn vmm_idle_victims(&self, keep: &[usize]) -> Vec<(usize, usize)> {
        let base = crate::kv_vmm::rows_grid(crate::exl3_forward::VMM_FLOOR_ROWS + crate::exl3_forward::VMM_MARGIN_ROWS,
                                            self.model.max_pos());
        let pf = self.kv_pf.map(|p| p.0);
        let mut idle: Vec<(usize, usize)> = (0..self.lanes.len())
            .filter(|s| self.lanes[*s].is_none() && !keep.contains(s) && pf != Some(*s) && !self.pf_busy(*s))
            .map(|s| (self.model.vmm_rows_mapped(s), s))
            .filter(|x| x.0 > base)
            .collect();
        idle.sort();
        if idle.len() > 1 { idle.pop(); }
        idle
    }

    /// KVPOOL: make sure lane `slot` has its position rows `< rows` mapped, evicting idle lanes while the pool is full.
    /// Err = the pool is exhausted and nothing idle is left (the request fails; the engine and the other lanes go on).
    fn vmm_room(&mut self, slot: usize, rows: usize) -> Result<()> {
        if !self.model.vmm_lazy() { return Ok(()); }
        loop {
            match self.model.vmm_ensure(slot, rows) {
                Ok(()) => return Ok(()),
                Err(e) if crate::kv_vmm::is_exhausted(&e) => { if !self.vmm_evict_idle(&[slot])? { return Err(e); } }
                Err(e) => return Err(e),
            }
        }
    }

    /// Prefill one lane's prompt into its slot, chunked (S-A3-f-b): C tokens per
    /// forward sweep through the wide-M path. The final sweep's last token is
    /// NOT processed here — see the S-A3-h seam contract below.
    /// NOTE: the sequential path primed the M1 draft head per prompt token; the
    /// chunked prefill skips that work entirely (pure TTFT win) and the head is
    /// primed once after the seam step (PLAN/S_A3_F_B_CHUNKED_PREFILL.md).
    /// S-A3-h item 1 seam contract (bench byte-parity): chunked prefill covers
    /// prompt[..plen-1] ONLY — the LAST prompt token runs through `forward_step`,
    /// the graphed/fused decode step, in `admit` (the exact bench `--prefill-chunk`
    /// contract, exl3_forward::run). The f-g prefill row and the f-f decode row for
    /// the same token differ in low bits (N.6 class): using the prefill row at the
    /// prompt/decode seam broke bench-vs-serve byte parity AND primed the M1 tap
    /// from a drifted hidden (bug5's self-consistent-but-wrong chains). One
    /// composition per position on both sides, by construction.
    /// WP02: returns Ok(false) when the client went away between chunks (no Finish is sent; the
    /// caller leaves cache[slot] = None, so the partial slot state is never reused). `tx` None =
    /// not cancellable (the WP16 XCHECK's fresh reference prefill).
    /// WP16: `aligned` = this run's chunk grid starts at 0 or at an aligned checkpoint (the caller
    /// did CkptStore::begin_aligned): its chunks extend the slot's checkpointed prefix and the
    /// chunk ends `wp16_due` picks become checkpoints. Never set for a prompt-end prefix-hit resume.
    /// C1 (`tail` Some only with prefix.tail_ckpt on): the grid is `run_grid(from, n, C, realign,
    /// tail)` — realigned to the C grid and split at the message boundary `tail`, which becomes a
    /// checkpoint of this (tracked) run. Knob off: realign = false, tail = None = today's grid.
    /// `key`: the prompt as the prefix cache / WP16 checkpoints see it (VIS-2: image rows carry
    /// pixel-hash ids so two different same-size images never share a cached prefix); == `prompt`
    /// for text.
    fn prefill_range(&mut self, slot: usize, prompt: &[u32], key: &[u32], from: usize, to: usize,
                     tx: Option<&crate::server::TokTx>, aligned: bool,
                     tail: Option<usize>, realign: bool) -> Result<bool> {
        let tp = std::time::Instant::now();
        let c = self.chunk.max(1);
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_alloc_rows(c, self.tp.is_some()))?);
        }
        let mut pos = from;
        let n = to.min(prompt.len().saturating_sub(1)); // the seam token is a decode step
        if tx.is_some() {
            // issue #24: the request owns its slot from here; a resumed prefix (rows 0..from) counts
            self.kv_pf = Some((slot, from));
            self.publish_kv();
        }
        if pos >= n { return Ok(true); }
        self.vmm_room(slot, n + 1)?; // KVPOOL: the whole prompt's rows up front (idle lanes give way)
        let mut taken: Vec<usize> = Vec::new();
        let (mut il_points, mut il_steps, mut il_ms) = (0usize, 0usize, 0.0f64);
        // C1 v2 (knob on == realign): merge away the chunk the split / realign would add (psc holds
        // 2C - 1 rows then, see run())
        let absorb = crate::exl3_forward::absorb_rmin();
        let grid = crate::exl3_forward::plan_grid(from, n, c, realign, tail, psc_rows(c, self.tp.is_some()), absorb, aligned);
        let nchunks = grid.len();
        if absorb.is_some() || c > 2048 {
            // TP-4X1 receipts: the grid actually run (only with the new options on; default logs unchanged)
            let mut rows = Vec::with_capacity(grid.len());
            let mut s = from;
            for &e in &grid { rows.push(e - s); s = e; }
            let shown = if rows.len() > 16 {
                format!("{:?} .. {:?}", &rows[..4], &rows[rows.len() - 6..])
            } else { format!("{rows:?}") };
            crate::drprintln!("[exl3-serve] prefill grid slot={slot} n={} (pos {from}..{n}) C={c}: {nchunks} chunk(s), rows {shown}{}",
                     n - from, if absorb.is_some() { " (absorb-tail)" } else { "" });
        }
        for (gi, end) in grid.iter().copied().enumerate() {
            self.pool_emergency(); // PFXPOOL2: a long prefill must not run the box into the memory watchdog
            // FP4L-SPEED1: the next chunk's n-gram rows start loading from the drive while this chunk runs
            if let Some(&next_end) = grid.get(gi + 1) { self.model.ple_warm_ahead(prompt, end, next_end); }
            // (re-borrowed per chunk: the interleave hook at the end of the body needs all of self)
            let psc = self.psc.as_mut().unwrap();
            if pos > from {
                let mut gone = tx.map_or(false, |t| t.is_closed());
                // TP-D: the HEAD decides (its client); every rank stops at this same chunk boundary
                if let (Some(t), true) = (self.tp.as_mut(), tx.is_some()) {
                    gone = t.head_flag(gone)?;
                }
                if gone {
                    crate::drprintln!("[exl3-serve] prefill slot={slot} cancelled at pos {pos} of {n} (client gone)");
                    return Ok(false);
                }
            }
            let toks: Vec<i32> = prompt[pos..end].iter().map(|&t| t as i32).collect();
            let t_chunk = std::time::Instant::now();
            self.model.prefill_chunk(psc, &toks, pos, slot, false, None)?;
            if sched_trace_on() {
                crate::rprintln!("[sched-trace] t={:.3} prefill slot={slot} rows={} end={end} of {n} ms={:.2}", wall_now(),
                                 end - pos, t_chunk.elapsed().as_secs_f64() * 1e3);
            }
            crate::metrics::sched_touch(); // H3: prefill chunk COMPLETE (long prefill stays live)
            // ISSUES[94] DF2: prime the draft ring with this chunk's window — the prime sink
            // captured the backbone hiddens of rows pos..end during the sweep that just
            // finished (cols = what this chunk actually ran), so the ring is current for the
            // lane's first speculative round after the prefill.
            if let Some(round) = self.df2.as_mut() {
                self.model.df2_prime_window(round, pos, end - pos)?;
            }
            if aligned {
                if let Some(st) = self.wp16.as_mut() {
                    st.extend(slot, &key[pos..end]);
                    if crate::exl3_forward::wp16_due(end, n, c) || tail == Some(end) {
                        let model = &self.model;
                        // TP-4D (audit C2): at world > 2 the allocation outcome is agreed across ranks (a rank-local
                        // failure would desynchronise the checkpoint stores); world <= 2: the plain local alloc.
                        let n_before = st.n_alloc;
                        let mut tp_alloc = self.tp.as_mut().filter(|t| t.world > 2);
                        let mut hub_err: Option<anyhow::Error> = None;
                        let got = st.acquire(|| {
                            let r = model.ckpt_alloc();
                            match tp_alloc.as_mut() { Some(t) => t.wp16_alloc(n_before, r, &mut hub_err), None => r }
                        });
                        if let Some(e) = hub_err { return Err(e); }
                        if let Some(mut b) = got {
                            match model.ckpt_save(slot, psc, &mut b) {
                                Ok(()) => { st.insert(slot, end, b); taken.push(end); }
                                Err(e) => { st.release(b); return Err(e); }
                            }
                        }
                    }
                }
            }
            pos = end;
            if tx.is_some() {
                // issue #24: a request's prefill holds rows 0..end of its slot
                self.kv_pf = Some((slot, end));
                self.publish_kv();
            }
            // --prefill-interleave: this chunk's rows, checkpoint and head-fill carry are complete
            // (interleave_steps drains the device first), so the lanes that were decoding before
            // this prompt arrived get their steps now.
            if interleave_point(self.interleave, tx.is_some(), end, n) {
                let t_il = std::time::Instant::now();
                let ran = self.interleave_steps(slot)?;
                il_points += usize::from(ran > 0);
                il_steps += ran;
                if ran > 0 { il_ms += t_il.elapsed().as_secs_f64() * 1e3; }
            }
        }
        if il_steps > 0 {
            crate::metrics::prefill_interleave(il_points as u64, il_steps as u64);
            crate::drprintln!("[exl3-serve] prefill-interleave slot={slot}: {il_steps} scheduler step(s) at {il_points} of {} chunk boundaries, \
                              {il_ms:.1} ms of decode (not counted in the prefill line's ms / tok/s)", nchunks.saturating_sub(1));
        }
        // S-A3-f-d Item 2: serve-path prefill telemetry (TTFT's main term). With --prefill-interleave
        // the decode steps run between the chunks are taken out, so the line stays the prefill's own rate.
        let ms = tp.elapsed().as_secs_f64() * 1e3 - il_ms;
        let done = n - from;
        crate::drprintln!("[exl3-serve] prefill slot={} n={} (pos {}..{}) in {:.1} ms ({:.0} tok/s, C={})",
                 slot, done, from, n, ms, if ms > 0.0 { done as f64 * 1e3 / ms } else { 0.0 }, c);
        if let Some(t) = tail {
            crate::drprintln!("[exl3-serve] C1 tail slot={slot}: grid split + checkpoint at {t} (message boundary - {}; {nchunks} chunk(s){})",
                     crate::exl3_forward::TAIL_BACKOFF,
                     if realign && from % c != 0 { ", realigned to the C grid" } else { "" });
        } else if realign && from % c != 0 {
            crate::drprintln!("[exl3-serve] C1 slot={slot}: run from {from} realigned to the C grid ({nchunks} chunk(s))");
        }
        if !taken.is_empty() {
            if let Some(st) = self.wp16.as_ref() {
                let per = self.model.ckpt_layout().bytes() as f64;
                crate::drprintln!("[exl3-serve] WP16 slot={slot}: checkpoints taken at {taken:?}; slot holds {:?}; \
                          live {} x {:.0} MB = {:.2} GiB (allocated {} of cap {})",
                         st.positions(slot), st.live(), per / 1e6, st.live() as f64 * per / (1u64 << 30) as f64,
                         st.n_alloc, st.cap);
            }
        }
        Ok(true)
    }

    // ---------------------------------------------------------- TURNRESUME
    /// Admission: run the inline diagnostic pass (--next-turn-prefill inline), then decide whether this prompt resumes
    /// from the lane's kept next-turn state (returns its depth; 0 = no). Either way the kept state and the pending hint
    /// of the lane are spent: the rows above the resume point are rewritten by this request.
    fn turn_admit(&mut self, slot: usize, key: &[u32], end: usize, snap: usize) -> Result<usize> {
        use crate::turn_resume as tr;
        use std::sync::atomic::Ordering::Relaxed;
        let Some(nt) = self.turn.as_ref().map(|a| a.nt) else { return Ok(0) };
        if nt == tr::NtMode::Inline {
            self.turn_drain();
            let g = self.turn.as_mut().unwrap().pending[slot].take();
            if let Some(g) = g.filter(|g| g.len() <= end && key[..g.len()] == g[..]) {
                if let Err(e) = self.turn_pass(slot, &g, None) {
                    tr::BG_FAILED.fetch_add(1, Relaxed);
                    self.turn.as_mut().unwrap().toks[slot] = None;
                    crate::reprintln!("[exl3-serve] next-turn prefill (inline) failed slot={slot}: {e:#}");
                    if sticky_cuda_error(&e) { return Err(e); }
                }
            }
        }
        let arm = self.turn.as_mut().unwrap();
        let p = tr::pick(arm.toks[slot].as_deref(), key, end, snap);
        match p {
            tr::Pick::Use(n) => { tr::HITS.fetch_add(1, Relaxed); tr::HIT_TOKENS.fetch_add((n - snap) as u64, Relaxed); }
            tr::Pick::None => { if snap > 0 { tr::MISS_NONE.fetch_add(1, Relaxed); } }
            // the same conversation (its prompt resumed the lane's snapshot) sent something else than the kept reply;
            // else another conversation took the lane
            tr::Pick::Diverged => { if snap > 0 { tr::MISS_DIVERGED.fetch_add(1, Relaxed); } else { tr::MISS_OTHER.fetch_add(1, Relaxed); } }
            tr::Pick::NotDeeper => { tr::MISS_NOTDEEPER.fetch_add(1, Relaxed); }
        }
        arm.pending[slot] = None;
        if let tr::Pick::Use(n) = p {
            return Ok(n); // the caller restores bufs[slot], then clears toks[slot]
        }
        if arm.toks[slot].is_some() {
            crate::drprintln!("[exl3-serve] turn-resume miss slot={slot}: {p:?} (kept {} tokens, prompt-end snapshot {snap})",
                              arm.toks[slot].as_ref().map_or(0, |t| t.len()));
        }
        arm.toks[slot] = None;
        Ok(0)
    }

    /// `--prefill-interleave N` (public PR #23): between two prefill chunks of the prompt joining
    /// `slot`, run up to N scheduler steps for the lanes that are already decoding, so a long
    /// prompt no longer freezes every running stream for its whole prefill. Returns the steps run.
    ///
    /// Why the nested `step()` cannot see or disturb the prefill in flight:
    /// - `lanes[slot]` is None until `admit` finishes, and `step()` (cancel sweep, re-probe arming,
    ///   shared-plain choice, fcfs front, every round) only walks the lanes that are Some. Nothing
    ///   in a step admits a request, so the joiner's slot cannot be handed to anyone else; a lane
    ///   that finishes or is cancelled inside a step frees a DIFFERENT index (it was Some when this
    ///   slot was picked) and `run` admits the next waiting request there after this admit returns.
    /// - the prefill carries its state across chunks in `self.psc` (head-fill carry, WP05 stage,
    ///   image splice) and in `slot`'s own rows of the model (GDN / conv state, PLE ring, KV and
    ///   QSA planes, head KV). A step works in `self.sc` and the stepped lanes' slots; it never
    ///   reads `self.psc`, `self.cache` or `self.wp16` (those are touched by admit / prefill only).
    /// - `admit` stages the joiner's sampler, penalties and seam step in `self.sc` only AFTER the
    ///   prefill returns, so nothing a nested step stages there is read by the joiner.
    /// A failing step fails every live lane (`fail_step`) and the error ends this prefill too.
    pub(super) fn interleave_steps(&mut self, slot: usize) -> Result<usize> {
        anyhow::ensure!(self.lanes[slot].is_none(), "prefill-interleave: slot {slot} already holds a lane");
        if self.fatal.is_some() || self.lanes.iter().all(|l| l.is_none()) { return Ok(0); }
        // prefill_chunk ends synced, but a checkpoint taken at this chunk end queued its copies
        // after that: drain them so the first nested round's measured time (lane EMAs, the cost
        // guard, the shared-step scale) is the round's own.
        self.model.dev().synchronize()?;
        let mut ran = 0usize;
        // S-B9-CF-PACK4: a prompt is mid-prefill for these steps — the decoding lanes do not pack
        self.in_prefill = true;
        for _ in 0..self.interleave {
            if self.fatal.is_some() || self.lanes.iter().all(|l| l.is_none()) { break; }
            if let Err(e) = self.step() {
                self.in_prefill = false;
                self.fail_step(&e);
                // The joiner fails too (the PR's choice: after a failed step nothing is admitted on
                // top of it). Its client sees `{e}` of this error, so the root cause goes in the text.
                anyhow::bail!("a decode step between prefill chunks failed (--prefill-interleave): {e:#}");
            }
            ran += 1;
        }
        self.in_prefill = false;
        if let Some(f) = self.fatal.as_ref() {
            anyhow::bail!("engine stopped during a prefill-interleave step: {f}");
        }
        Ok(ran)
    }

    /// A failed decode step (from `run`, or from a prefill-interleave step): fail every live lane
    /// loudly (never emit garbage as if it were text) and mark the engine dead on a sticky CUDA error.
    pub(super) fn fail_step(&mut self, e: &anyhow::Error) {
        eprintln!("[exl3-serve] decode step failed: {e:#}");
        for s in 0..self.lanes.len() {
            if let Some(lane) = self.take_lane(s, false) {
                let _ = lane.tx.send(TokEvent::Finish { reason: format!("error: {e}") });
            }
        }
        if sticky_cuda_error(e) {
            self.fatal = Some(format!("{e:#}"));
        }
    }

    /// Prefill prompt[..plen-1] into `slot`, resuming from a cached prefix when possible, and
    /// snapshot the recurrent state at the END of the chunked prefill (prompt[..plen-1]; the last
    /// token is the seam decode step). No extra split: a fresh prompt prefills bit-identically to
    /// --prefix-cache off (a message-boundary split cost a ~175 ms small chunk and moved chunk
    /// boundaries). The next chat turn re-renders this prompt verbatim up to plen-1 in both
    /// thinking modes (verified via /v1/tokenize: thinking on differs only at the final token).
    /// WP16: when the prompt-end snapshot does not match (a new conversation on a shared system
    /// prompt, a new question on the same document), resume from the deepest intermediate
    /// checkpoint <= min(LCP(prompt, the checkpointed prefix), plen-1) instead of prefilling from 0
    /// — the same chunk grid as a fresh prefill, so bit-identical to it. The prompt-end hit path
    /// is unchanged (whenever it matches it is the deeper resume and it still wins).
    /// C1 (prefix.tail_ckpt on): `ckpt_at` (the message boundary: this prompt without its
    /// generation prompt, server.rs) becomes a tail checkpoint of this run — the resume point of
    /// the next chat turn, which diverges right there — and every run is tracked (rows extend the
    /// checkpointed prefix) with its grid realigned to C. See wp16.rs T_PREFIX_TAIL_CKPT.
    pub(super) fn prefill_cached(&mut self, slot: usize, prompt: &[u32], key: &[u32], ckpt_at: Option<usize>,
                      tx: &crate::server::TokTx) -> Result<bool> {
        anyhow::ensure!(key.len() == prompt.len(), "prefix key length {} != prompt length {}", key.len(), prompt.len());
        // PACKPF: this slot's prompt was already prefilled by a packed pass (pack_prefill_waiting)
        if let Some(r) = self.pf_take_done(slot, key) { return r; }
        let pl = self.prefill_plan(slot, prompt, key, ckpt_at, tx)?;
        if !pl.snapshot {
            return self.prefill_range(slot, prompt, key, pl.from, pl.end, Some(tx), false, None, false);
        }
        match self.prefill_range(slot, prompt, key, pl.from, pl.end, Some(tx), pl.tracked, pl.tail, pl.realign) {
            Ok(true) => {}
            // WP02: cancelled mid-prefill — no snapshot, cache stays None. WP16: the checkpoints
            // taken so far stay (they describe prompt[..pos], rows this run did write).
            Ok(false) => return Ok(false),
            Err(e) => {
                if let Some(st) = self.wp16.as_mut() { st.clear_slot(slot); } // slot state unknown
                return Err(e);
            }
        }
        self.prefill_finish(slot, prompt, key, &pl)
    }

    /// prefill_cached's tail after a complete prefill run: the WP16 xcheck, the prompt-end snapshot,
    /// the cache entry. PACKPF: a packed run calls it per prompt with that slot's carry selected.
    pub(super) fn prefill_finish(&mut self, slot: usize, prompt: &[u32], key: &[u32], pl: &PfPlan) -> Result<bool> {
        if pl.q > 0 && self.wp16_xcheck && key == prompt { // VIS-2: the xcheck's scratch prefill is text-only
            self.wp16_xcheck(slot, prompt, pl.q)?;
        }
        self.model.snapshot_slot(slot, self.psc.as_ref().unwrap())?;
        self.cache[slot] = Some(key[..pl.end].to_vec());
        Ok(true)
    }

    /// prefill_cached's head: every decision and restore before the prefill run (the victim capture,
    /// the prefix-cache / pool / spill / WP16 resume, the slot reset, the C1 tail point). The run is
    /// then prefill_range(slot, prompt, key, from, end, tx, tracked, tail, realign) — or, for a
    /// PACKPF packed prompt, its chunks inside packed passes.
    pub(super) fn prefill_plan(&mut self, slot: usize, prompt: &[u32], key: &[u32], ckpt_at: Option<usize>,
                    tx: &crate::server::TokTx) -> Result<PfPlan> {
        let plen = prompt.len();
        let end = plen.saturating_sub(1);
        self.last_resume = (0, 0);
        if !self.prefix_on || !self.model.prefix_snapshots() {
            // PFX1 (b): the chunked prefill below head-fills rows 0..plen-2 (no head-KV memset)
            self.model.reset_slot_for_prefill(slot)?;
            self.cache[slot] = None;
            return Ok(PfPlan { from: 0, end, tracked: false, tail: None, realign: false, q: 0, snapshot: false });
        }
        let tail_on = self.wp16.is_some() && crate::exl3_forward::tail_ckpt_on();
        // LANERES2: this slot is being taken over by a different conversation when its cache
        // does not prefix the incoming prompt — capture the victim's snapshot to the SSD spill
        // tier BEFORE anything below resets/overwrites it. The D2H is the only in-round cost
        // (~0.1 s per 0.5-GiB snapshot); the disk write is the async part.
        // PFXPOOL: the prefix pool takes it first (a device copy into memory); the SSD tier only gets what the pool
        // did not keep.
        // PFXPOOL2: agree the pool back into one state (entries a rank freed under memory pressure) and give memory
        // back while any rank is under the floor — before the capture and the lookup below
        self.pool_upkeep()?;
        self.vmm_room(slot, plen)?; // KVPOOL: before any restore writes this lane's rows
        if self.pool.on() || self.spill.is_some() {
            if let Some(old) = self.cache[slot].clone() {
                let own = old.len() <= end && key[..old.len()] == old[..];
                if !own {
                    let held = self.pool_capture_slot(slot, &old)?;
                    if !held && self.spill.is_some() {
                        self.spill_capture_slot(slot, &old)?;
                    }
                }
            }
        }
        let mut reuse = match &self.cache[slot] {
            Some(t) if t.len() <= end && key[..t.len()] == t[..] => t.len(),
            _ => 0,
        };
        // TURNRESUME: the lane's kept next-turn state, when this prompt starts with it deeper than the snapshot
        let turn_hit = self.turn_admit(slot, key, end, reuse)?;
        if turn_hit > 0 { reuse = turn_hit; }
        // PFXPOOL / LANERES2: set when `reuse` came from OUTSIDE this lane (a pool entry or an SSD snapshot): the
        // lane's WP16 bookkeeping then describes the conversation that was here before and is cleared below.
        let mut external = false;
        // PFXPOOL: a full pool restore whose lane's aligned prefix was adopted keeps the C1 / WP16 tracking of an own hit
        let mut adopted = false;
        // PFXPOOL: own-snapshot miss — restore the deepest pool entry, unless the lane's own WP16 checkpoint is as deep.
        if reuse == 0 && self.pool.on() {
            let own_q = self.wp16.as_ref().map_or(0, |st| st.resume_point(slot, key, end, self.chunk.max(1)).0);
            match self.pool_restore_slot(slot, key, end, own_q)? {
                Some(PoolHit::Full(n, a)) => { reuse = n; external = true; adopted = a; }
                Some(PoolHit::Ckpt(_)) => {} // installed as the lane's WP16 checkpoint: the WP16 resume below takes it
                None => {}
            }
        }
        // LANERES2 / SPILLTRUNC: own-snapshot miss — try the SSD spill tier. A Full hit restores
        // rows 0..n + snapshot state (`reuse > 0` branch below); a Trunc hit restores rows 0..q and
        // installs the carried WP16 checkpoint into `self.wp16` (leaving `reuse = 0` so the WP16
        // resume branch runs `ckpt_restore`, identical to `PoolHit::Ckpt`).
        if reuse == 0 && self.spill.is_some() {
            match self.spill_restore_slot(slot, key, end)? {
                Some(SpillHit::Full(n)) => {
                    reuse = n;
                    external = true;
                    adopted = false;
                    self.segs_set(slot, None, n); // PFXPOOL2: rows from disk: a lineage of their own
                }
                Some(SpillHit::Trunc(_)) => {} // installed as the lane's WP16 checkpoint: the WP16 resume below takes it
                None => {}
            }
        }
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_alloc_rows(self.chunk.max(1), self.tp.is_some()))?);
        }
        let c = self.chunk.max(1);
        // WP16: only on a prompt-end miss (a hit is always the deeper, unchanged resume).
        let (q, lcp) = match (&self.wp16, reuse) {
            (Some(st), 0) => st.resume_point(slot, key, end, c),
            _ => (0, 0),
        };
        let mut tracked = reuse == 0 && self.wp16.is_some();
        if turn_hit > 0 {
            // TURNRESUME: rows below turn_hit were written by the lane's own background pass over the rendered reply
            let arm = self.turn.as_mut().unwrap();
            self.model.ckpt_restore(slot, self.psc.as_mut().unwrap(), &arm.bufs[slot])?;
            arm.toks[slot] = None; // rows >= turn_hit are rewritten from here on
            crate::drprintln!("[exl3-serve] turn-resume hit slot={slot}: reuse {reuse} of {plen} prompt tokens (kept next-turn state)");
            if let Some(st) = self.wp16.as_mut() { st.begin_unaligned(slot, reuse); }
        } else if reuse > 0 {
            self.model.restore_slot(slot, self.psc.as_mut().unwrap())?;
            crate::drprintln!("[exl3-serve] prefix-cache hit slot={slot}: reuse {reuse} of {plen} prompt tokens");
            // an unaligned grid from `reuse`: no checkpoint is taken in it, none above it survives.
            // C1: with tail checkpoints the run is tracked instead (realigned grid, tail split) when
            // the slot's checkpointed prefix is exactly prompt[..reuse].
            if let Some(st) = self.wp16.as_mut() {
                if external && !adopted {
                    // PFXPOOL: rows 0..reuse now hold the restored conversation, but the lane's WP16 list and token
                    // prefix still describe the one that was here before: a later prompt sharing THAT prefix would
                    // resume an old checkpoint over the new rows. Forget it (the spill restore had the same hole).
                    st.clear_slot(slot);
                } else if tail_on && st.prefix_ok(slot, key, reuse, c) {
                    st.begin_aligned(slot, reuse, c);
                    tracked = true;
                } else {
                    st.begin_unaligned(slot, reuse);
                }
            }
        } else if q > 0 {
            let st = self.wp16.as_mut().unwrap();
            let dropped = st.begin_aligned(slot, q, c);
            st.touch(slot, q);
            let ck = st.get(slot, q).context("WP16: resume checkpoint vanished")?;
            // replaces reset_slot: recurrent state + PLE ring/hist + head-fill carry at q; every
            // positional row < q (trunk/head KV, QSA planes) is the checkpointed prefix's own
            self.model.ckpt_restore(slot, self.psc.as_mut().unwrap(), ck)?;
            self.model.ckpt_poison_head_kv_from(slot, q)?; // PFX1 (b) poison gate only (no-op otherwise)
            crate::drprintln!("[exl3-serve] WP16 checkpoint resume slot={slot}: resume at {q} of {plen} prompt tokens \
                      (LCP {lcp}, C={c}; {dropped} checkpoint(s) above dropped) — prefill {q}..{end}");
        } else {
            self.model.reset_slot_for_prefill(slot)?; // PFX1 (b): prefill_range from 0 head-fills
            if let Some(st) = self.wp16.as_mut() { st.begin_aligned(slot, 0, c); }
        }
        self.cache[slot] = None;
        self.last_resume = (reuse, q);
        let from = reuse.max(q);
        if self.turn.is_some() && from > 0 {
            crate::turn_resume::FOLLOWUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::turn_resume::FOLLOWUP_PREFILLED.fetch_add((plen - from) as u64, std::sync::atomic::Ordering::Relaxed);
        }
        self.segs_prefill(slot, from, end); // PFXPOOL2: rows from..end are this run's
        // v0.7.3 (#8.2): report the cache-served prefix (prefill skipped) to the request
        // handler ONCE, before the first token: usage.prompt_tokens_details.cached_tokens
        // and the /metrics counter. from == 0 (fresh prefill) stays silent — the
        // handler's default is 0, so engines without a prefix cache report 0.
        if from > 0 {
            let _ = tx.send(TokEvent::Admitted { cached_tokens: from as u32 });
        }
        let tail = if tail_on && tracked { crate::exl3_forward::tail_point(ckpt_at, from, end, c, crate::exl3_forward::tail_ckpt_extra_chunk()) } else { None };
        Ok(PfPlan { from, end, tracked, tail, realign: tail_on && tracked, q, snapshot: true })
    }

    /// Hand the server's rendered next-turn prefixes to their lanes: the free lane whose prompt-end key is the longest
    /// prefix of the rendering. A hint no lane holds is dropped (counted).
    fn turn_drain(&mut self) {
        use crate::turn_resume as tr;
        use std::sync::atomic::Ordering::Relaxed;
        if self.turn.is_none() { return; }
        for (h, at) in tr::take() {
            let keys: Vec<Option<&[u32]>> = (0..self.lanes.len())
                .map(|i| if self.lanes[i].is_none() { self.cache[i].as_deref() } else { None }).collect();
            match tr::lane_for(&h, &keys) {
                Some(i) => {
                    let a = self.turn.as_mut().unwrap();
                    a.pending[i] = Some(h);
                    a.pending_at[i] = at;
                }
                None => {
                    // the same conversation (a lane key sharing at least half of itself with the rendering and leaving
                    // it before either ends) = the rendering does not start with the prompt; else no idle lane holds it
                    // (another conversation took the lane, or the conversation already moved on: its lane's prompt
                    // extends the rendering)
                    let near = keys.iter().flatten().any(|k| {
                        let l = k.iter().zip(&h).take_while(|(a, b)| a == b).count();
                        !k.is_empty() && l * 2 >= k.len() && l < k.len().min(h.len())
                    });
                    if near { tr::HINT_MISMATCH.fetch_add(1, Relaxed); } else { tr::HINT_NOLANE.fetch_add(1, Relaxed); }
                    crate::drprintln!("[exl3-serve] next-turn hint ({} tokens) dropped: {}", h.len(),
                                      if near { "the rendering does not start with the lane's prompt" } else { "no idle lane holds that conversation" });
                }
            }
        }
    }

    /// One background (or inline) pass: prefill the rendered next-turn prefix `g` on lane `slot`, from the deepest state
    /// that `g` starts with (the lane's kept state, else its prompt-end snapshot), in --next-turn-chunk row chunks. With
    /// `rx` (background), an arriving request stops the pass after the chunk in flight; the rows read so far are kept
    /// (the hint stays pending, the pass goes on at the next idle point) and the request is returned. The prompt-end
    /// snapshot is never written.
    fn turn_pass(&mut self, slot: usize, g: &[u32],
                 mut rx: Option<&mut tokio::sync::mpsc::UnboundedReceiver<BatchRequest>>) -> Result<Option<BatchRequest>> {
        use crate::turn_resume as tr;
        use std::sync::atomic::Ordering::Relaxed;
        let s = match &self.cache[slot] {
            Some(k) if !k.is_empty() && k.len() <= g.len() && g[..k.len()] == k[..] => k.len(),
            _ => { self.turn.as_mut().unwrap().pending[slot] = None; return Ok(None); }
        };
        let (base, from_kept) = {
            let arm = self.turn.as_ref().unwrap();
            match arm.toks[slot].as_deref() {
                Some(k) if k.len() > s && k.len() <= g.len() && g[..k.len()] == k[..] => (k.len(), true),
                _ => (s, false),
            }
        };
        let to = g.len();
        if base >= to {
            self.turn.as_mut().unwrap().pending[slot] = None;
            return Ok(None);
        }
        // an arrival before the first chunk: the pass has not started, nothing waited
        if let Some(r) = rx.as_deref_mut() {
            if let Ok(req) = r.try_recv() { return Ok(Some(req)); }
        }
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_alloc_rows(self.chunk.max(1), self.tp.is_some()))?);
        }
        let t0 = std::time::Instant::now();
        {
            let arm = self.turn.as_mut().unwrap();
            if from_kept {
                self.model.ckpt_restore(slot, self.psc.as_mut().unwrap(), &arm.bufs[slot])?;
            } else {
                self.model.restore_slot(slot, self.psc.as_mut().unwrap())?;
            }
            arm.toks[slot] = None; // rows >= base are being written
        }
        let chunk = self.turn.as_ref().unwrap().chunk;
        let grid = tr::bg_grid(base, to, chunk);
        let mut pos = base;
        let mut arrived: Option<BatchRequest> = None;
        for (gi, &e) in grid.iter().enumerate() {
            if gi > 0 {
                if let Some(r) = rx.as_deref_mut() {
                    if let Ok(req) = r.try_recv() {
                        // it waited (at most) for the chunk that just ended
                        tr::wait_sample(req.received_at.elapsed().as_micros() as u64);
                        arrived = Some(req);
                        break;
                    }
                }
            }
            if let Some(&next) = grid.get(gi + 1) { self.model.ple_warm_ahead(g, e, next); }
            let toks: Vec<i32> = g[pos..e].iter().map(|&t| t as i32).collect();
            let psc = self.psc.as_mut().unwrap();
            self.model.prefill_chunk(psc, &toks, pos, slot, false, None)?;
            crate::metrics::sched_touch();
            tr::BG_CHUNKS.fetch_add(1, Relaxed);
            pos = e;
        }
        // the last chunk may have ended with a request already queued: it waited for that chunk
        if arrived.is_none() {
            if let Some(r) = rx.as_deref_mut() {
                if let Ok(req) = r.try_recv() {
                    tr::wait_sample(req.received_at.elapsed().as_micros() as u64);
                    arrived = Some(req);
                }
            }
        }
        {
            let arm = self.turn.as_mut().unwrap();
            self.model.ckpt_save(slot, self.psc.as_ref().unwrap(), &mut arm.bufs[slot])?;
            arm.toks[slot] = Some(g[..pos].to_vec());
            if pos >= to { arm.pending[slot] = None; }
        }
        self.segs_prefill(slot, base, pos); // PFXPOOL2: rows base..pos are this pass's
        let us = t0.elapsed().as_micros() as u64;
        tr::BG_PASSES.fetch_add(1, Relaxed);
        tr::BG_TOKENS.fetch_add((pos - base) as u64, Relaxed);
        tr::BG_US.fetch_add(us, Relaxed);
        if pos < to { tr::BG_ABORTED.fetch_add(1, Relaxed); }
        crate::drprintln!("[exl3-serve] next-turn prefill slot={slot}: rows {base}..{pos} of {to} ({} from the {}) in {:.1} ms{}",
                          pos - base, if from_kept { "kept state" } else { "prompt-end snapshot" }, us as f64 / 1e3,
                          if pos < to { " — yielded to an arriving request" } else { "" });
        Ok(arrived)
    }

    /// --wp16-xcheck=1 (diagnostic): after a checkpoint resume of `slot` (state R live), prefill
    /// the same prompt FRESH from 0 into a scratch slot — a free lane's slot, or (--max-batch 1)
    /// the resumed slot itself — and bit-compare GDN S / conv / PLE ring / PLE hist / head-fill
    /// carry and the seam step's first-token logits. The resumed state is then restored, so the
    /// served output stays the resumed path's (scratch mode: exactly; same-slot mode: the KV rows
    /// are the fresh run's rewrite of the same positions).
    fn wp16_xcheck(&mut self, slot: usize, prompt: &[u32], q: usize) -> Result<()> {
        let t0 = std::time::Instant::now();
        let plen = prompt.len();
        let end = plen - 1;
        let last = prompt[end] as i32;
        let model = self.model.clone();
        let v = model.cfg.vocab_size;
        let x = (0..self.lanes.len()).find(|&i| i != slot && self.lanes[i].is_none()).unwrap_or(slot);
        let mut r = model.ckpt_alloc()?;
        model.ckpt_save(slot, self.psc.as_ref().unwrap(), &mut r)?;
        // raw logits: greedy, unpenalized (admit re-stages both before the real seam step)
        model.set_sampling(&mut self.sc, &[None])?;
        model.set_penalties(&mut self.sc, &[None])?;
        let id_r = model.forward_step(&mut self.sc, &[last], &[end], &[slot], None)?[0];
        let lg_r: Vec<u16> = model.logits_host(&self.sc)?[..v].to_vec();
        if x != slot {
            self.cache[x] = None; // the scratch slot's rows get overwritten
            if let Some(a) = self.turn.as_mut() { a.toks[x] = None; a.pending[x] = None; }
            if let Some(st) = self.wp16.as_mut() { st.clear_slot(x); }
        }
        crate::dprintln!("[wp16-xcheck] slot={slot} resumed at {q}: fresh reference prefill 0..{end} into {} slot {x}",
                 if x != slot { "scratch" } else { "the same (no free lane)" });
        model.reset_slot_for_prefill(x)?;
        self.prefill_range(x, prompt, prompt, 0, end, None, false, None, false)?;
        self.segs_set(x, None, end); // PFXPOOL2: rewritten by a fresh prefill
        let mut f = model.ckpt_alloc()?;
        model.ckpt_save(x, self.psc.as_ref().unwrap(), &mut f)?;
        let d = model.ckpt_diff(&r, &f)?;
        let id_f = model.forward_step(&mut self.sc, &[last], &[end], &[x], None)?[0];
        let lg_f: Vec<u16> = model.logits_host(&self.sc)?[..v].to_vec();
        let nl = lg_r.iter().zip(&lg_f).filter(|(a, b)| a != b).count();
        // serve the resumed state: recurrent state + PLE ring/hist + carry back to R (the seam step
        // above advanced them; the real seam step rewrites KV row plen-1 identically)
        model.ckpt_restore(slot, self.psc.as_mut().unwrap(), &r)?;
        if x != slot { model.reset_slot(x)?; }
        let exact = d.exact() && nl == 0 && id_r == id_f;
        crate::dprintln!("[wp16-xcheck] slot={slot} resume {q} vs fresh (slot {x}) at plen {plen}: {}; first-token logits \
                  differ {nl}/{v} (argmax {id_r} vs {id_f}) -> {} ({:.0} ms)",
                 d.line(), if exact { "EXACT" } else { "MISMATCH" }, t0.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    /// The idle point with --next-turn-prefill on: run pending passes (yielding to requests), else wait in short ticks
    /// so a hint posted after a reply wakes the scheduler.
    pub(super) fn turn_idle(&mut self, rx: &mut tokio::sync::mpsc::UnboundedReceiver<BatchRequest>) -> TurnWake {
        use crate::turn_resume as tr;
        self.turn_drain();
        // a pass starts once its hint is --next-turn-delay-ms old (an agent that sends its next request at once never
        // waits behind a pass); a pass that yielded goes on at once at the next idle point
        let next = self.turn.as_ref().and_then(|a| (0..a.pending.len())
            .find(|&i| a.pending[i].is_some() && (a.toks[i].is_some() || a.pending_at[i].elapsed() >= a.delay)));
        if let Some(slot) = next {
            let g = self.turn.as_ref().unwrap().pending[slot].clone().unwrap();
            return match self.turn_pass(slot, &g, Some(rx)) {
                Ok(Some(req)) => TurnWake::Req(req),
                Ok(None) => TurnWake::Again,
                Err(e) => {
                    tr::BG_FAILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    crate::reprintln!("[exl3-serve] next-turn prefill failed slot={slot}: {e:#} (the lane keeps its prompt-end snapshot)");
                    if let Some(a) = self.turn.as_mut() { a.toks[slot] = None; a.pending[slot] = None; }
                    if sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                    TurnWake::Again
                }
            };
        }
        if tr::pending() { return TurnWake::Again; }
        // nothing to do (or only hints younger than the delay): wait for a request or a hint, 20 ms ticks
        // (5 ms sleeps: a hint or a request is seen within 5 ms; the thread sleeps between checks)
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(20);
        let got = loop {
            match rx.try_recv() {
                Ok(r) => break Some(Some(r)),
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break Some(None),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {}
            }
            if tr::pending() || std::time::Instant::now() >= deadline { break None; }
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        match got {
            Some(Some(r)) => TurnWake::Req(r),
            Some(None) => TurnWake::Closed,
            None => {
                let up = self.turn.as_ref().map_or(false, |a| a.last_upkeep.elapsed() >= std::time::Duration::from_secs(1));
                if up && self.pool.on() {
                    self.pool_emergency();
                    if let Some(a) = self.turn.as_mut() { a.last_upkeep = std::time::Instant::now(); }
                }
                TurnWake::Again
            }
        }
    }

    /// S-A3-h: head prime after the seam decode step — snapshot the tap the seam
    /// step left in sc.resid (the f-f decode composition, byte-exact: the same tap
    /// class as the bench-mtp prime and every round's re-prime; the f-d
    /// tap_snapshot_from_prefill workaround fed round 1's verify restore a
    /// ≤1e-3-drifted prefill tap, whose chain was self-consistent but wrong), then
    /// feed the head the last prompt token with that tap (round 1's draft-extend
    /// reuses the snapshot via tap_row 0).
    /// WP03: the slot's own taps window (row 0) + a KV-only head row plen-1 paired with the stream
    /// after plen-2 (psc's head-fill carry) — FwdModel::mtp_seam_prime (--a-seam = old arm).
    pub(super) fn mtp_prime(&mut self, slot: usize, last_prompt_tok: u32, plen: usize) -> Result<()> {
        let head = self.model.mtp.as_ref()
            .context("mtp_prime called without a draft head")?;
        self.model.mtp_seam_prime(&mut self.sc, self.psc.as_ref(), head, last_prompt_tok as i32, plen, slot)
    }
}
