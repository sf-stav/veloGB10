//! Packed prefill of waiting requests: slot picking and the job list (`pf_*`), `pack_prefill_waiting`, `pack_run` and `own_snapshot_slot`.
//! Another `impl Exl3Scheduler` block; the job and plan types are defined in `mod.rs`, the chunk helpers in `prefill.rs`.

use super::*;

impl Exl3Scheduler {
    /// WP16 resume depth per slot for THIS prompt (0 = no grid / no usable checkpoint) — the
    /// second branch of the owner-change victim order.
    pub(super) fn wp16_depths(&self, prompt: &[u32]) -> Vec<usize> {
        let end = prompt.len().saturating_sub(1);
        (0..self.lanes.len()).map(|i| self.wp16.as_ref()
            .map_or(0, |st| st.resume_point(i, prompt, end, self.chunk.max(1)).0)).collect()
    }

    /// The pick block's prompt-end hit test over FREE lanes (same test, same (len, slot) max):
    /// the "own slot" a waiting request is admitted into first under --lane-reserve.
    /// PACKPF: the free lane admit_req would pick for `prompt` (its unscoped rule: the longest prompt-end
    /// snapshot hit, the deepest WP16 resume, an empty lane, the first free lane), lanes in `taken`
    /// excluded (earlier prompts of the same packed run, admitted before this one).
    fn pf_pick_slot(&self, prompt: &[u32], taken: &[usize]) -> Option<usize> {
        let free: Vec<usize> = (0..self.lanes.len()).filter(|&i| self.lanes[i].is_none() && !taken.contains(&i)).collect();
        let hit = free.iter().copied().filter_map(|i| match &self.cache[i] {
            Some(t) if t.len() < prompt.len() && prompt[..t.len()] == t[..] => Some((t.len(), i)),
            _ => None,
        }).max();
        hit.map(|(_, i)| i)
            .or_else(|| {
                let st = self.wp16.as_ref()?;
                let end = prompt.len().saturating_sub(1);
                free.iter().copied()
                    .map(|i| (st.resume_point(i, prompt, end, self.chunk.max(1)).0, i))
                    .filter(|&(q, _)| q > 0).max().map(|(_, i)| i)
            })
            .or_else(|| free.iter().copied().find(|&i| self.cache[i].is_none()))
            .or_else(|| self.lru_free(&free)) // Issue #46 (LRUPICK): same rule as pick_lane
            .or_else(|| free.first().copied())
    }

    /// PACKPF: why packing is off for this scheduler right now (None = it can pack). Counted per
    /// reason on /metrics when a pack was possible by queue but skipped.
    fn pf_skip_reason(&self) -> Option<&'static str> {
        if self.tp.is_some() { return Some("tp"); }
        if self.df2.is_some() { return Some("df2"); }
        if self.reserve.is_some() { return Some("lane_reserve"); }
        if self.wp16_xcheck { return Some("wp16_xcheck"); }
        if self.model.pack_pf_unsupported().is_some() { return Some("model"); }
        None
    }

    /// PACKPF: may this waiting request ride a packed prefill? (text-only, client still there)
    fn pf_eligible(r: &BatchRequest) -> bool {
        r.image_spans.is_empty() && r.prompt.len() >= 2 && !r.tx.is_closed()
    }

    /// PACKPF (--prefill-pack-min > 2 / --prefill-pack-wait): hold admission for a decode round so a
    /// fuller pack forms — only while lanes decode, only for an eligible, not yet packed front, and
    /// never past --prefill-pack-wait-ms since its arrival.
    pub(super) fn pf_hold(&self, waiting: &std::collections::VecDeque<BatchRequest>) -> bool {
        if packpf_min() <= 2 && !packpf_wait() { return false; }
        if self.pf_skip_reason().is_some() || !self.pf_jobs.is_empty() { return false; } // never stall a run in flight
        let Some(f) = waiting.front() else { return false };
        if self.pf_done.values().any(|d| d.key == f.prompt) { return false; }
        let running = self.lanes.iter().any(|l| l.is_some());
        let nfree = self.lanes.iter().filter(|l| l.is_none()).count();
        let front_ok = Self::pf_eligible(f) && f.received_at.elapsed().as_millis() < packpf_wait_ms() as u128;
        let short_queue = packpf_wait() && waiting.len() < packpf_min();
        let hold = running && front_ok && (short_queue || (waiting.len() >= packpf_min() && nfree < packpf_min()));
        if hold { crate::metrics::prefill_pack_hold(); }
        hold
    }

    /// PACKPF: the admit side of a packed prefill — Some(Ok(true)) when `slot` holds this prompt from a
    /// packed pass (its head-fill carry is put back for the seam prime), None otherwise.
    pub(super) fn pf_take_done(&mut self, slot: usize, key: &[u32]) -> Option<Result<bool>> {
        let d = self.pf_done.remove(&slot)?;
        if d.key != key { return None; } // another prompt took the lane: the normal path below
        let r = (|| -> Result<bool> {
            let psc = self.psc.as_mut().context("PACKPF: no prefill scratch")?;
            self.model.pf_carry_from_slot(psc, slot)?;
            self.last_resume = d.resume;
            self.kv_pf = Some((slot, d.end));
            self.publish_kv();
            Ok(true)
        })();
        Some(r)
    }

    /// PACKPF (idea and first implementation: GitHub andersc, veloGB10 fork, PACK-PF): prefill the
    /// FIFO front of the waiting queue — up to one prompt per free lane — in shared passes. Each
    /// prompt gets exactly the lane, the resume (prefix snapshot, pool, spill, WP16 checkpoint) and
    /// the chunk grid its own admit would give it (prefill_plan, plan_grid); a pass carries the next
    /// chunk of the front prompt plus the next chunks of the prompts behind it that still fit
    /// --prefill-pack-rows (a chunk below pf_min_seg_rows runs alone). Passes run only until the
    /// FRONT prompt is complete, so its first token is not held back for the prompts behind it; the
    /// others stay in flight (self.pf_jobs, their lanes reserved) and continue at the next call.
    /// Checkpoints, the snapshot and the cache entry are taken as prefill_range / prefill_cached take
    /// them; the admit then picks that lane and skips the prefill (pf_done). Any error: every lane
    /// in flight is left unsnapshotted and those requests are admitted one by one as before (said,
    /// counted); a sticky CUDA error stops the engine.
    pub(super) fn pack_prefill_waiting(&mut self, waiting: &std::collections::VecDeque<BatchRequest>) {
        let same = |r: &BatchRequest, at: std::time::Instant, key: &[u32]| r.received_at == at && r.prompt == key;
        self.pf_done.retain(|_, d| waiting.iter().any(|r| r.prompt == d.key));
        if let Some(f) = waiting.front() {
            if self.pf_done.values().any(|d| d.key == f.prompt) { return; } // the front is packed: admit it
        }
        // in-flight prompts whose request left the queue (client gone) give their lane back
        let gone: Vec<usize> = self.pf_jobs.iter().filter(|j| !waiting.iter().any(|r| same(r, j.at, &j.key) && !r.tx.is_closed()))
            .map(|j| j.slot).collect();
        if !gone.is_empty() {
            for &sl in &gone {
                crate::drprintln!("[exl3-serve] PACKPF: slot={sl} dropped mid-prefill (client gone)");
                self.pf_abandon_slot(sl);
            }
            self.pf_jobs.retain(|j| !gone.contains(&j.slot));
        }
        // the front must be the first in-flight prompt, or start a new run
        let front_ok = waiting.front().map_or(false, |f| self.pf_jobs.first().map_or(false, |j| same(f, j.at, &j.key)));
        if !front_ok && !self.pf_jobs.is_empty() {
            crate::drprintln!("[exl3-serve] PACKPF: the queue front is not the first prompt in flight; {} in-flight prompt(s) re-prefilled one by one",
                              self.pf_jobs.len());
            for sl in self.pf_jobs.iter().map(|j| j.slot).collect::<Vec<_>>() { self.pf_abandon_slot(sl); }
            self.pf_jobs.clear();
        }
        if self.pf_jobs.is_empty() {
            self.pf_done.clear(); // a new run is a FIFO prefix: behind an unpacked front a marker is unreachable
            let nfree = self.lanes.iter().filter(|l| l.is_none()).count();
            let neligible = waiting.iter().take(nfree).take_while(|r| Self::pf_eligible(r)).count();
            if neligible < 2 { return; }
            if let Some(why) = self.pf_skip_reason() {
                crate::metrics::prefill_pack_skip(why);
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| crate::reprintln!("[exl3-serve] --prefill-pack on: prompts are prefilled one by one here ({why}{}); \
counted in velogb10_prefill_pack_skipped_total", if why == "model" { format!(": {}", self.model.pack_pf_unsupported().unwrap_or("")) } else { String::new() }));
                return;
            }
        }
        if let Err(e) = self.pack_run(waiting) {
            crate::reprintln!("[exl3-serve] PACKPF: packed prefill failed, admitting one by one: {e:#}");
            crate::metrics::prefill_pack_skip("error");
            for sl in self.pf_jobs.iter().map(|j| j.slot).collect::<Vec<_>>() { self.pf_abandon_slot(sl); }
            self.pf_jobs.clear();
            if sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
        }
        self.kv_pf = None;
        self.publish_kv();
    }

    /// PACKPF: a lane whose packed prefill will not complete — no snapshot, no checkpoints (its admit,
    /// if the request is still there, prefills it the normal way).
    fn pf_abandon_slot(&mut self, slot: usize) {
        self.cache[slot] = None;
        if let Some(st) = self.wp16.as_mut() { st.clear_slot(slot); }
    }

    /// PACKPF: the lanes a packed prefill holds (in flight): no other admit may pick them.
    pub(super) fn pf_busy(&self, slot: usize) -> bool { self.pf_jobs.iter().any(|j| j.slot == slot) }

    fn pack_run(&mut self, waiting: &std::collections::VecDeque<BatchRequest>) -> Result<()> {
        let t0 = std::time::Instant::now();
        let c = self.chunk.max(1);
        let tp = self.tp.is_some();
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_alloc_rows(c, tp))?);
        }
        let cap = self.model.pf_rows_cap(self.psc.as_ref().unwrap());
        let budget = if packpf_rows() > 0 { packpf_rows().min(cap) } else { cap };
        let min_seg = crate::exl3_forward::pf_min_seg_rows();
        let absorb = crate::exl3_forward::absorb_rmin();
        // 1. join: the waiting requests right behind the prompts in flight (FIFO, contiguous, eligible),
        //    each planned in admission order (lane pick, reset or resume), one per free lane
        let max_jobs = crate::exl3_forward::PACK_PF_MAX;
        loop {
            if self.pf_jobs.len() >= max_jobs { break; }
            // the first waiting request that is neither in flight nor already complete (a short prompt
            // behind the front can complete first): everything before it is packed, so FIFO holds
            let Some(wi) = waiting.iter().position(|r| {
                !self.pf_jobs.iter().any(|j| r.received_at == j.at && r.prompt == j.key)
                    && !self.pf_done.values().any(|d| d.key == r.prompt)
            }) else { break };
            let r = &waiting[wi];
            if !Self::pf_eligible(r) { break; }
            let taken: Vec<usize> = self.pf_jobs.iter().map(|j| j.slot).chain(self.pf_done.keys().copied()).collect();
            let Some(slot) = self.pf_pick_slot(&r.prompt, &taken) else { break };
            self.model.clear_slot_vision(slot)?;
            let plan = self.prefill_plan(slot, &r.prompt, &r.prompt, r.ckpt_at, &r.tx)?;
            if self.psc.is_none() {
                self.psc = Some(self.model.prefill_scratch(psc_alloc_rows(c, tp))?);
            }
            // a resume restored this slot's head-fill carry into psc: keep it in the slot's store
            self.model.pf_carry_to_slot(self.psc.as_mut().unwrap(), slot)?;
            let grid = crate::exl3_forward::plan_grid(plan.from, plan.end, c, plan.realign, plan.tail, psc_rows(c, tp), absorb, plan.tracked);
            self.pf_jobs.push(PfJob { slot, at: r.received_at, key: r.prompt.clone(), pos: plan.from, plan, grid, gi: 0,
                                      taken: Vec::new(), resume: self.last_resume, passes: 0, shared: 0, t0: std::time::Instant::now() });
        }
        if self.pf_jobs.is_empty() { return Ok(()); }
        // 2. passes until the front prompt is complete (prompts that complete on the way are finished too)
        let (mut il_steps, mut il_ms) = (0usize, 0.0f64);
        let mut npass = 0usize;
        // until THIS front is complete (once it is removed, pf_jobs.first() is the next prompt: stop there)
        let front_slot = self.pf_jobs[0].slot;
        while self.pf_jobs.iter().any(|j| j.slot == front_slot && j.gi < j.grid.len()) {
            let mut segs: Vec<crate::exl3_forward::PfSeg> = Vec::new();
            let mut members: Vec<usize> = Vec::new();
            let mut toks: Vec<i32> = Vec::new();
            for (ji, j) in self.pf_jobs.iter().enumerate() {
                if j.gi >= j.grid.len() { continue; }
                let e = j.grid[j.gi];
                let rows = e - j.pos;
                let small = rows < min_seg;
                // the front's chunk always goes; a chunk below the segment minimum runs alone
                if !segs.is_empty() && (small || toks.len() + rows > budget || segs.len() >= max_jobs) { continue; }
                segs.push(crate::exl3_forward::PfSeg { r0: toks.len(), n: rows, pos0: j.pos, slot: j.slot });
                toks.extend(j.key[j.pos..e].iter().map(|&t| t as i32));
                members.push(ji);
                if small { break; }
            }
            self.pool_emergency();
            for &ji in &members {
                let j = &self.pf_jobs[ji];
                if let Some(&ne) = j.grid.get(j.gi + 1) { self.model.ple_warm_ahead(&j.key, j.grid[j.gi], ne); }
            }
            let psc = self.psc.as_mut().unwrap();
            self.model.prefill_segs(psc, &toks, &segs)?;
            crate::metrics::sched_touch();
            npass += 1;
            crate::metrics::prefill_pack_pass(segs.len() as u64, toks.len() as u64);
            // per prompt: WP16 extend + checkpoints, exactly as prefill_range after its chunk
            for &ji in &members {
                let (slot, pos, end, aligned, tail, n_end) = {
                    let j = &self.pf_jobs[ji];
                    (j.slot, j.pos, j.grid[j.gi], j.plan.tracked, j.plan.tail, j.plan.end)
                };
                if aligned {
                    if let Some(st) = self.wp16.as_mut() {
                        st.extend(slot, &self.pf_jobs[ji].key[pos..end]);
                        if crate::exl3_forward::wp16_due(end, n_end, c) || tail == Some(end) {
                            let model = &self.model;
                            // veloGB10 #44 (latent note): the same world > 2 allocation agreement as prefill_range
                            // (TP-4D, audit C2), so a rank-local failure can never desynchronise the checkpoint stores
                            // if packing is ever enabled under TP (pf_skip_reason refuses packing under TP today).
                            let n_before = st.n_alloc;
                            let mut tp_alloc = self.tp.as_mut().filter(|t| t.world > 2);
                            let mut hub_err: Option<anyhow::Error> = None;
                            let got = st.acquire(|| {
                                let r = model.ckpt_alloc();
                                match tp_alloc.as_mut() { Some(t) => t.wp16_alloc(n_before, r, &mut hub_err), None => r }
                            });
                            if let Some(e) = hub_err { return Err(e); }
                            if let Some(mut b) = got {
                                let psc = self.psc.as_mut().unwrap();
                                let r = model.pf_carry_from_slot(psc, slot).and_then(|_| model.ckpt_save(slot, psc, &mut b));
                                match r {
                                    Ok(()) => { st.insert(slot, end, b); self.pf_jobs[ji].taken.push(end); }
                                    Err(e) => { st.release(b); return Err(e); }
                                }
                            }
                        }
                    }
                }
                let j = &mut self.pf_jobs[ji];
                j.pos = end;
                j.gi += 1;
                j.passes += 1;
                if segs.len() > 1 { j.shared += 1; }
            }
            // prompts complete after this pass: snapshot + cache entry + the admit marker
            let mut ji = 0;
            while ji < self.pf_jobs.len() {
                if self.pf_jobs[ji].gi >= self.pf_jobs[ji].grid.len() {
                    let j = self.pf_jobs.remove(ji);
                    self.pf_finish_job(j)?;
                } else { ji += 1; }
            }
            // --prefill-interleave: the decoding lanes get their steps while the front still prefills
            if let Some(s0) = self.pf_jobs.first().filter(|j| j.gi < j.grid.len() && j.gi > 0).map(|j| j.slot) {
                if self.interleave > 0 {
                    let t_il = std::time::Instant::now();
                    let ran = self.interleave_steps(s0)?;
                    il_steps += ran;
                    if ran > 0 { il_ms += t_il.elapsed().as_secs_f64() * 1e3; crate::metrics::prefill_interleave(1, ran as u64); }
                }
            }
        }
        // a front with nothing to prefill (a full prompt-end hit) completes here
        while self.pf_jobs.first().map_or(false, |j| j.gi >= j.grid.len()) {
            let j = self.pf_jobs.remove(0);
            self.pf_finish_job(j)?;
        }
        if npass > 0 {
            crate::drprintln!("[exl3-serve] PACKPF: {npass} pass(es) in {:.1} ms (budget {budget} rows, C={c}); {} prompt(s) still in flight{}",
                              t0.elapsed().as_secs_f64() * 1e3 - il_ms, self.pf_jobs.len(),
                              if il_steps > 0 { format!("; {il_steps} decode step(s) between passes ({il_ms:.1} ms, not counted)") } else { String::new() });
        }
        Ok(())
    }

    /// PACKPF: a prompt whose packed prefill is complete — its carry back into psc, the prompt-end
    /// snapshot + cache entry (prefill_finish), then the marker its admit takes (pf_take_done).
    fn pf_finish_job(&mut self, j: PfJob) -> Result<()> {
        {
            let psc = self.psc.as_mut().unwrap();
            self.model.pf_carry_from_slot(psc, j.slot)?;
        }
        if j.plan.snapshot { self.prefill_finish(j.slot, &j.key, &j.key, &j.plan)?; }
        let ms = j.t0.elapsed().as_secs_f64() * 1e3;
        let rows = j.plan.end - j.plan.from;
        crate::metrics::prefill_pack_prompts(1);
        crate::drprintln!("[exl3-serve] prefill slot={} n={rows} (pos {}..{}) in {ms:.1} ms ({:.0} tok/s, C={}): packed, {} pass(es), {} shared{}",
                          j.slot, j.plan.from, j.plan.end, if ms > 0.0 { rows as f64 * 1e3 / ms } else { 0.0 }, self.chunk.max(1),
                          j.passes, j.shared,
                          if j.taken.is_empty() { String::new() } else { format!("; WP16 checkpoints at {:?}", j.taken) });
        self.pf_done.insert(j.slot, PfDone { key: j.key, resume: j.resume, end: j.plan.end });
        Ok(())
    }

    pub(super) fn own_snapshot_slot(&self, prompt: &[u32]) -> Option<usize> {
        (0..self.lanes.len()).filter(|&i| self.lanes[i].is_none())
            .filter_map(|i| match &self.cache[i] {
                Some(t) if t.len() < prompt.len() && prompt[..t.len()] == t[..] => Some((t.len(), i)),
                _ => None,
            }).max().map(|(_, i)| i)
    }
}
