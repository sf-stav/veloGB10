//! The prefix-state pool and the KV spill: capture and restore of a slot's prompt state (`pool_*`, `segs_*`, `spill_*`), emergency eviction
//! and the pool upkeep. Another `impl Exl3Scheduler` block; the `PoolBuf` / `PoolHit` types and the pool constants live in `mod.rs`.

use super::*;

impl Exl3Scheduler {
    // ---------------------------------------------------------- PFXPOOL: the prefix pool
    fn pool_publish(&self) {
        crate::metrics::pfx_pool_state(self.pool.cap_bytes(), self.pool.bytes(), self.pool.len() as u64);
    }

    /// The pool's lockstep bit under TP (the AND of every rank's local outcome; the spill tier's SpillFlag
    /// rendezvous, run at the same admission point on every rank); local truth on a single box.
    fn pool_agree(&mut self, ok: bool) -> Result<bool> {
        match self.tp.as_mut() {
            Some(t) => t.spill_agree(ok),
            None => Ok(ok),
        }
    }

    /// Would `bytes` more device memory keep this box at or above the pool's MemAvailable floor?
    fn pool_mem_ok(&self, bytes: u64) -> bool {
        crate::memwatch::mem_available_bytes().map_or(true, |a| a >= bytes.saturating_add(self.pool_floor))
    }

    /// PFXPOOL2: lane `slot`'s row lineage (empty until the lane has prefilled under the pool).
    fn segs_of(&mut self, slot: usize) -> &mut crate::prefix_pool::Segs {
        if self.slot_segs.len() <= slot { self.slot_segs.resize(slot + 1, Vec::new()); }
        &mut self.slot_segs[slot]
    }

    /// PFXPOOL2: rows 0..from of `slot` are kept, rows from..end are written by a new prefill computation.
    pub(super) fn segs_prefill(&mut self, slot: usize, from: usize, end: usize) {
        self.row_gen += 1;
        let g = self.row_gen;
        let s = self.segs_of(slot);
        let mut kept = crate::prefix_pool::clip(s, from).unwrap_or_default();
        if kept.last().map_or(0, |x| x.0) < from {
            // rows below `from` of unknown provenance (should not happen): label them with this computation alone
            kept.clear();
        }
        if end > kept.last().map_or(0, |x| x.0) { kept.push((end, g)); }
        *s = kept;
    }

    /// PFXPOOL2: rows 0..n of `slot` now hold a copy of rows with lineage `src` (a pool restore), or (None) rows of
    /// unknown lineage (an SSD restore): a fresh label.
    pub(super) fn segs_set(&mut self, slot: usize, src: Option<crate::prefix_pool::Segs>, n: usize) {
        let v = match src.and_then(|s| crate::prefix_pool::clip(&s, n)) {
            Some(v) => v,
            None => { self.row_gen += 1; if n > 0 { vec![(n, self.row_gen)] } else { Vec::new() } }
        };
        *self.segs_of(slot) = v;
    }

    /// PFXPOOL: `slot` is being taken over by a different conversation — copy its prompt state (`old`: the
    /// tokens its prompt-end snapshot holds) into the pool before anything overwrites it. Returns true when
    /// the pool now holds that state (captured now, or already held). Never fails the round on its own:
    /// over the cap, under the memory floor or a failed copy on ANY rank = not captured (counted). Err only
    /// when a TP agree itself fails (the link is gone; the admission must stop with that message).
    /// PFXPOOL2: when the lane's leading rows are an older entry's (same lineage), only the rows after them are copied
    /// (the entry reads the rest through that one). Under memory pressure it never empties the pool for nothing: it
    /// first agrees that shifting entries out CAN make room.
    pub(super) fn pool_capture_slot(&mut self, slot: usize, old: &[u32]) -> Result<bool> {
        if !self.pool.on() { return Ok(false); }
        // PFXPOOL4: image-bearing prompts ARE capturable. Main-attention RoPE is baked into K rows at write,
        // and the per-slot vision arm is request-scoped (`vision_setup` / `clear_slot_vision` runs before any
        // restore). Image rows key on `image_row_key(pixel_hash, j)`, so any moved/changed image mismatches at
        // its first row and truncates to the deepest provable prefix.
        // short prompts are cheaper to read again than to keep
        if old.len() < POOL_MIN_TOKENS {
            crate::metrics::pfx_pool_skip();
            return Ok(false);
        }
        if self.pool.contains(old) {
            crate::metrics::pfx_pool_skip();
            return Ok(true);
        }
        let n = old.len();
        let segs = match crate::prefix_pool::clip(self.segs_of(slot), n) {
            Some(v) => v,
            None => { self.segs_set(slot, None, n); self.segs_of(slot).clone() }
        };
        // the lane's WP16 checkpoints over this prompt travel with it (copies: the incoming prompt may still resume
        // from them in this lane)
        let (ck_c, ck_pos, aligned): (usize, Vec<usize>, usize) = match self.wp16.as_ref() {
            Some(st) => {
                let (c, v) = st.ckpts_under(slot, old, n);
                (c, v.into_iter().map(|(p, _)| p).collect(), st.aligned_len(slot, old).1)
            }
            None => (0, Vec::new(), 0),
        };
        let snapb = self.model.pool_snap_bytes()?;
        let ckb = self.model.ckpt_bytes();
        let ckt = ck_pos.len() as u64 * ckb;
        let r1 = self.model.pool_rows_for(n);
        let grid = self.model.pool_row_grid();
        // the entry whose rows this lane's leading rows are (same lineage): only the rows after them are copied. Its
        // chain is touched now, so the cap below shifts it out last.
        let mut parent: Option<(u64, usize)> = self.pool.best_parent(&segs, POOL_MAX_DEPTH).and_then(|(i, k)| {
            let sh = (k / grid * grid).min(r1);
            (sh >= POOL_SHARE_MIN).then(|| (self.pool.get(i).unwrap().id, sh))
        });
        if let Some((id, _)) = parent { if let Some(i) = self.pool.index_of(id) { self.pool.touch(i); } }
        let need = |m: &FwdModel, p: &Option<(u64, usize)>| -> Result<u64> {
            Ok(m.pool_rows_bytes(p.map_or(0, |x| x.1), r1)? + snapb + ckt)
        };
        let mut bytes = need(&self.model, &parent)?;
        if !self.pool.fits(bytes) {
            crate::metrics::pfx_pool_refused();
            crate::drprintln!("[exl3-serve] prefix pool: {n} tokens ({} MiB) exceed the cap ({} MiB); not kept",
                              bytes >> 20, self.pool.cap_bytes() >> 20);
            return Ok(false);
        }
        let t0 = std::time::Instant::now();
        // the cap: shift out ascending by re-prefill cost (PFXPOOL3); refuse if room would displace deeper entries
        let mut nev = 0u64;
        loop {
            let ev = match self.pool.make_room(bytes, n) {
                Some(ev) => ev,
                None => {
                    crate::metrics::pfx_pool_refused();
                    self.pool_publish();
                    crate::drprintln!("[exl3-serve] prefix pool: {n} tokens not kept (room here would cost a deeper \
                                       entry than it: {} entries, {} of {} MiB untouched)",
                                      self.pool.len(), self.pool.bytes() >> 20, self.pool.cap_bytes() >> 20);
                    return Ok(false);
                }
            };
            nev += ev.len() as u64;
            for e in ev { if let Some(b) = e.buf.buf { self.pool_spare.push(b); } }
            if parent.map_or(false, |(id, _)| self.pool.index_of(id).is_none()) {
                parent = None; // the parent itself had to go: this capture keeps all its rows
                bytes = need(&self.model, &parent)?;
                if !self.pool.fits(bytes) { break; }
                continue;
            }
            break;
        }
        crate::metrics::pfx_pool_evict(nev);
        let chain: Vec<u64> = parent.and_then(|(id, _)| self.pool.index_of(id)).map_or(Vec::new(), |i| self.pool.chain(i));
        // a slab: a recycled one when one is big enough (no new memory), else a fresh allocation under the memory floor,
        // agreed across ranks. Feasibility first (every rank: could freeing the spares and every entry except the chain
        // this capture needs make room?), so a capture that cannot fit does not shift the pool out for nothing.
        // The spare list is rank-local (`pool_emergency` frees it on one rank alone), so every rank runs the same agree
        // sequence whether or not it found a spare.
        let mut slab = self.pool_take_spare(bytes);
        {
            let keep: u64 = chain.iter().filter_map(|id| self.pool.index_of(*id)).map(|i| self.pool.get(i).unwrap().bytes).sum();
            let spare_b: u64 = self.pool_spare.iter().map(|b| b.len() as u64).sum();
            let reclaim = spare_b + self.pool.bytes().saturating_sub(keep);
            let feasible = slab.is_some() || crate::memwatch::mem_available_bytes()
                .map_or(true, |a| a.saturating_add(reclaim) >= bytes.saturating_add(self.pool_floor));
            if !self.pool_agree(feasible)? {
                if let Some(b) = slab.take() { self.pool_spare.push(b); }
                crate::metrics::pfx_pool_refused();
                self.pool_publish();
                crate::drprintln!("[exl3-serve] prefix pool: memory floor ({} GiB free kept) — {n} tokens not kept (the pool \
                                   could not make room; nothing shifted out)", self.pool_floor >> 30);
                return Ok(false);
            }
            let mut k = 0u64;
            let mut cleared = false;
            loop {
                let ok = slab.is_some() || self.pool_mem_ok(bytes);
                if self.pool_agree(ok)? { break; }
                if !cleared {
                    cleared = true;
                    self.pool_spare.clear();
                } else {
                    // the LRU entry is replicated state: every rank takes the same branch here
                    if self.pool.lru().map_or(true, |id| chain.contains(&id)) {
                        crate::metrics::pfx_pool_evict(k);
                        crate::metrics::pfx_pool_refused();
                        if let Some(b) = slab.take() { self.pool_spare.push(b); }
                        self.pool_publish();
                        crate::drprintln!("[exl3-serve] prefix pool: memory floor ({} GiB free kept) — {n} tokens not kept",
                                          self.pool_floor >> 30);
                        return Ok(false);
                    }
                    for e in self.pool.pop_lru() {
                        k += 1;
                        if let Some(b) = e.buf.buf {
                            if slab.is_none() && b.len() as u64 >= bytes { slab = Some(b); }
                        }
                    }
                }
                self.model.dev().synchronize()?; // the stream-ordered frees return the memory at a sync
            }
            crate::metrics::pfx_pool_evict(k);
        }
        let fresh = slab.is_none();
        let t_slab = t0.elapsed().as_secs_f64() * 1e3;
        let r0 = parent.map_or(0, |x| x.1);
        let cap = (|| -> Result<PoolBuf> {
            let buf = match slab { Some(b) => b, None => self.model.pool_alloc(bytes as usize)? };
            let base = *buf.device_ptr() as u64;
            let snap_off = self.model.pool_capture_rows(slot, r0, r1, base)?;
            let hist = self.model.pool_capture_snap_to(slot, base + snap_off)?;
            let ck_off = snap_off + snapb;
            let mut ckpts = Vec::with_capacity(ck_pos.len());
            if let Some(st) = self.wp16.as_ref() {
                for (k, &p) in ck_pos.iter().enumerate() {
                    let src = st.get(slot, p).context("WP16 checkpoint vanished")?;
                    let h = self.model.ckpt_save_to(src, base + ck_off + k as u64 * ckb)?;
                    ckpts.push((p, h));
                }
            }
            Ok(PoolBuf { buf: Some(buf), hist, r0, r1, snap_off, ck_off, ck_bytes: ckb, ckpts, ckpt_c: ck_c, aligned })
        })();
        let t_copy = t0.elapsed().as_secs_f64() * 1e3;
        if !self.pool_agree(cap.is_ok())? {
            if let Err(e) = &cap { crate::dreprintln!("[exl3-serve] prefix pool: capture failed slot={slot} ({e:#}); not kept"); }
            if let Ok(pb) = cap { if let Some(b) = pb.buf { self.pool_spare.push(b); } }
            crate::metrics::pfx_pool_refused();
            self.pool_publish();
            return Ok(false);
        }
        let pb = cap?;
        self.pool.insert_marked(old.to_vec(), bytes, pb, ck_pos.clone(), parent, segs);
        // keep at most two spare slabs (the largest: the likeliest to fit the next long prompt)
        self.pool_spare.sort_by_key(|b| std::cmp::Reverse(b.len()));
        self.pool_spare.truncate(2);
        crate::metrics::pfx_pool_capture(t0.elapsed().as_micros() as u64);
        self.pool_publish();
        if crate::boot_report::debug() {
            self.model.dev().synchronize()?; // debug only: when the copies have actually landed
            crate::rprintln!("[exl3-serve] prefix pool: kept {n} tokens from slot={slot} ({}) with {} checkpoint(s) ({} MiB, {} slab; \
                              pool {} entries, {} of {} MiB; {:.1} GiB free) — slab {:.1} ms, copies queued {:.1} ms, landed {:.1} ms",
                             if r0 > 0 { format!("rows 0..{r0} shared with an older entry") } else { "all rows".to_string() },
                             ck_pos.len(), bytes >> 20, if fresh { "new" } else { "recycled" }, self.pool.len(),
                             self.pool.bytes() >> 20, self.pool.cap_bytes() >> 20,
                             crate::memwatch::mem_available_bytes().map_or(0.0, |b| b as f64 / (1u64 << 30) as f64),
                             t_slab, t_copy, t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(true)
    }

    /// PFXPOOL2: write rows 0..`rows` of pool entry `i` into `slot` through its parent chain.
    fn pool_restore_chain(&self, slot: usize, i: usize, rows: usize) -> Result<()> {
        let parts = self.pool.row_parts(i, rows).context("prefix pool: an ancestor vanished")?;
        for (j, lo, hi) in parts {
            let e = self.pool.get(j).context("prefix pool: entry vanished")?;
            let b = e.buf.buf.as_ref().context("prefix pool: entry freed")?;
            self.model.pool_restore_rows_part(slot, lo, hi, e.buf.r0, e.buf.r1, *b.device_ptr() as u64)?;
        }
        Ok(())
    }

    /// PFXPOOL2: memory pressure while serving. The pool's floor was only checked when an entry was kept, so other
    /// growth (the head box's other processes, a build) could push a box to the memory watchdog with the pool still
    /// full (owner hand test 2026-11-10: exit at 4.0 GB free while VeloBench grew 36 -> 45 GB). Two halves:
    /// `pool_emergency` (any time, rank-local) frees slabs in LRU order the moment its box is under the floor;
    /// `pool_upkeep` (every admission, an SPMD point) agrees the policy back into one state: entries freed on any rank
    /// are dropped everywhere, then while any rank is under the floor the LRU entry is shifted out everywhere.
    pub(super) fn pool_upkeep(&mut self) -> Result<()> {
        if !self.pool.on() { return Ok(()); }
        let (mut k, mut freed, mut cleared) = (0u64, 0u64, false);
        // entries a rank freed on its own: one agree when none is (the common case); else one per entry, in id order
        // (replicated), dropping exactly the ones freed somewhere (with their descendants)
        let alive = |s: &Self, i: usize| s.pool.get(i).map_or(true, |e| e.buf.buf.is_some());
        let all = (0..self.pool.len()).all(|i| alive(self, i));
        if !self.pool_agree(all)? {
            for id in self.pool.ids() {
                let Some(i) = self.pool.index_of(id) else { continue }; // went with an ancestor
                if !self.pool_agree(alive(self, i))? {
                    for e in self.pool.remove_id(id) { k += 1; freed += e.bytes; }
                }
            }
        }
        loop {
            if self.pool_agree(self.pool_mem_ok(0))? { break; }
            if !cleared {
                cleared = true;
                freed += self.pool_spare.iter().map(|b| b.len() as u64).sum::<u64>();
                self.pool_spare.clear();
            } else if self.pool.is_empty() {
                break;
            } else {
                for e in self.pool.pop_lru() { k += 1; freed += e.bytes; }
            }
            self.model.dev().synchronize()?;
        }
        if k > 0 || freed > 0 {
            crate::metrics::pfx_pool_evict(k);
            self.pool_publish();
            crate::drprintln!("[exl3-serve] prefix pool: memory floor — shifted out {k} entr(ies), {} MiB ({} entries kept)",
                              freed >> 20, self.pool.len());
        }
        Ok(())
    }

    /// PFXPOOL2: rank-local emergency release, called between prefill chunks and scheduler iterations (at most every
    /// 200 ms). Under the floor it frees the spare slabs, then entry slabs in the order `pool_upkeep` will shift the
    /// entries out (LRU, with descendants), until the box is back above the floor. No rendezvous: the policy keeps the
    /// entries (marked freed) until the next admission agrees them away; nothing reads a freed entry before that.
    pub(super) fn pool_emergency(&mut self) {
        if !self.pool.on() || (self.pool.is_empty() && self.pool_spare.is_empty()) { return; }
        let now = std::time::Instant::now();
        if self.pool_emerg_at.map_or(false, |t| now.duration_since(t) < std::time::Duration::from_millis(200)) { return; }
        self.pool_emerg_at = Some(now);
        if self.pool_mem_ok(0) { return; }
        let mut freed = 0u64;
        if !self.pool_spare.is_empty() {
            freed += self.pool_spare.iter().map(|b| b.len() as u64).sum::<u64>();
            self.pool_spare.clear();
            let _ = self.model.dev().synchronize();
        }
        let mut n = 0usize;
        for i in self.pool.lru_order() {
            if self.pool_mem_ok(0) { break; }
            for j in self.pool.family_of(i) {
                if let Some(e) = self.pool.get_mut(j) {
                    if let Some(b) = e.buf.buf.take() { freed += b.len() as u64; n += 1; drop(b); }
                }
            }
            let _ = self.model.dev().synchronize();
        }
        crate::metrics::pfx_pool_emergency(n as u64);
        crate::rprintln!("[exl3-serve] prefix pool: under the memory floor ({} GiB) — freed {n} entr(ies) and spares on this box \
                          ({} MiB; {:.1} GiB free now); the pool drops them at the next request",
                         self.pool_floor >> 30, freed >> 20,
                         crate::memwatch::mem_available_bytes().map_or(0.0, |b| b as f64 / (1u64 << 30) as f64));
    }

    /// PFXPOOL: the smallest spare slab that holds `bytes` (removed from the list), or None.
    fn pool_take_spare(&mut self, bytes: u64) -> Option<cudarc::driver::CudaSlice<u8>> {
        let i = self.pool_spare.iter().enumerate().filter(|(_, b)| b.len() as u64 >= bytes)
            .min_by_key(|(_, b)| b.len()).map(|(i, _)| i)?;
        Some(self.pool_spare.swap_remove(i))
    }

    /// PFXPOOL: the lane's own snapshot missed — restore the deepest pool entry that prefixes `key` (holding at most
    /// `end` tokens) into `slot`, unless the lane's own WP16 checkpoint (`own_q`) is at least as deep (that resume
    /// needs no copy). Returns the restored token count (the new `reuse`). Lookup and choice are replicated state, the
    /// copies are local device copies; a failed copy is fatal under TP (the ranks' KV would differ) and a re-prefill
    /// on a single box. PFXPOOL2: rows come through the entry's parent chain, the recurrent state from the entry; the
    /// lane's row lineage becomes the entry's.
    pub(super) fn pool_restore_slot(&mut self, slot: usize, key: &[u32], end: usize, own_q: usize) -> Result<Option<PoolHit>> {
        // PFXPOOL4: image-bearing keys restore like text ones (`vision_setup` already armed the slot for this prompt)
        if !self.pool.on() || self.pool.is_empty() {
            return Ok(None);
        }
        let c = self.chunk.max(1);
        // a carried checkpoint needs the WP16 store (and the chunk width it was taken at); a full entry does not
        let cand = self.pool.best(key, end).filter(|r| r.full || (self.wp16.is_some()
            && self.pool.get(r.i).map_or(false, |e| e.buf.ckpt_c == c)));
        let Some(r) = cand.filter(|r| r.at > own_q) else {
            crate::metrics::pfx_pool_miss();
            return Ok(None);
        };
        let t0 = std::time::Instant::now();
        let segs = self.pool.get(r.i).context("prefix pool: entry vanished")?.segs.clone();
        if r.full {
            let res = (|| -> Result<()> {
                self.pool_restore_chain(slot, r.i, self.model.pool_rows_for(r.at))?;
                let e = self.pool.get(r.i).context("prefix pool: entry vanished")?;
                let b = e.buf.buf.as_ref().context("prefix pool: entry freed")?;
                self.model.pool_restore_snap_from(slot, *b.device_ptr() as u64 + e.buf.snap_off, e.buf.hist)
            })();
            if let Err(e) = res {
                if self.tp.is_some() {
                    anyhow::bail!("prefix pool: restore failed slot={slot} ({e:#}) — the ranks' KV would differ");
                }
                crate::dreprintln!("[exl3-serve] prefix pool: restore failed slot={slot} ({e:#}); re-prefill");
                crate::metrics::pfx_pool_miss();
                self.segs_set(slot, None, 0);
                return Ok(None);
            }
            self.segs_set(slot, Some(segs), r.at);
            // C1 parity with an own-lane hit: the lane adopts the entry's aligned prefix (when it covers the restore
            // point at this chunk width), so the run below is tracked exactly as it would be in the entry's own lane.
            let adopted = match (self.wp16.as_mut(), self.pool.get(r.i)) {
                (Some(st), Some(e)) if e.buf.ckpt_c == c && e.buf.aligned >= r.at => {
                    st.adopt(slot, c, &e.toks[..e.buf.aligned]);
                    true
                }
                _ => false,
            };
            self.pool.touch(r.i);
            crate::metrics::pfx_pool_hit(r.at as u64, t0.elapsed().as_micros() as u64);
            crate::drprintln!("[exl3-serve] prefix pool hit slot={slot}: restored {} of {} prompt tokens{}", r.at, end + 1,
                              if adopted { " (aligned prefix adopted)" } else { "" });
            return Ok(Some(PoolHit::Full(r.at, adopted)));
        }
        // a carried checkpoint at q: a WP16 buffer for the lane (agreed: an allocation that fails on any rank skips the
        // resume everywhere), rows 0..q, then the checkpoint installed as the lane's own
        let q = r.at;
        let model = self.model.clone();
        let got = self.wp16.as_mut().unwrap().acquire(|| model.ckpt_alloc());
        if !self.pool_agree(got.is_some())? {
            if let Some(b) = got { self.wp16.as_mut().unwrap().release(b); }
            crate::metrics::pfx_pool_miss();
            return Ok(None);
        }
        let mut dst = got.unwrap();
        let res = (|| -> Result<()> {
            self.pool_restore_chain(slot, r.i, self.model.pool_rows_for(q))?;
            let e = self.pool.get(r.i).context("prefix pool: entry vanished")?;
            let k = e.buf.ckpts.iter().position(|(p, _)| *p == q).context("prefix pool: checkpoint vanished")?;
            let b = e.buf.buf.as_ref().context("prefix pool: entry freed")?;
            let at = *b.device_ptr() as u64 + e.buf.ck_off + k as u64 * e.buf.ck_bytes;
            self.model.ckpt_load_from(&mut dst, at, e.buf.ckpts[k].1)
        })();
        if let Err(e) = res {
            if self.tp.is_some() {
                anyhow::bail!("prefix pool: checkpoint restore failed slot={slot} ({e:#}) — the ranks' KV would differ");
            }
            self.wp16.as_mut().unwrap().release(dst);
            crate::dreprintln!("[exl3-serve] prefix pool: checkpoint restore failed slot={slot} ({e:#}); re-prefill");
            crate::metrics::pfx_pool_miss();
            self.segs_set(slot, None, 0);
            return Ok(None);
        }
        self.segs_set(slot, Some(segs), q);
        self.wp16.as_mut().unwrap().install(slot, c, &key[..q], q, dst);
        self.pool.touch(r.i);
        crate::metrics::pfx_pool_hit(q as u64, t0.elapsed().as_micros() as u64);
        crate::drprintln!("[exl3-serve] prefix pool hit slot={slot}: carried checkpoint at {q} of {} prompt tokens", end + 1);
        Ok(Some(PoolHit::Ckpt(q)))
    }

    // ---------------------------------------------------------- LANERES2: --kv-spill
    /// Capture `slot`'s snapshot (the EVICTED conversation's — the caller checked the incoming
    /// prompt does not extend it) into the spill store. Never fails the round: a capture or
    /// store error counts as a recompute-miss and the snapshot is simply lost (today's
    /// behavior). Key = FNV-1a of the cached tokens, so a later request whose prompt extends
    /// them finds it by prefix scan. The disk write is async (the store's writer thread).
    /// Err only under TP, when the cross-rank agree itself failed (the link is aborted by then): the admission must stop
    /// there with this message (ISSUES 121: it used to be parked in `self.fatal` while the admission ran on, and the
    /// next lockstep's 'transport already aborted' message replaced it).
    pub(super) fn spill_capture_slot(&mut self, slot: usize, old: &[u32]) -> Result<()> {
        let Some(st) = self.spill.clone() else { return Ok(()) };
        let key = crate::kv_spill::conv_key(old);
        let tokens = old.len() as u64;
        // SPILLTRUNC: the lane's checkpoints for THIS conversation ride the file (spill_capture) so a later
        // prompt that is a truncated prefix of `old` can restore rows + one carried checkpoint.
        let (c0, ckrefs) = self.wp16
            .as_ref()
            .map_or((0usize, Vec::new()), |st| st.ckpts_under(slot, old, old.len()));
        let poss: Vec<u64> = ckrefs.iter().map(|(p, _)| *p as u64).collect();
        let keys = crate::kv_spill::prefix_keys(old, &poss);
        let cks: Vec<(u64, u64, &crate::exl3_forward::RecurCkpt)> =
            ckrefs.iter().zip(&keys).map(|((p, b), k)| (*p as u64, *k, *b)).collect();
        // LANERES3 (TP>=2) + veloGB10 #52: compute the exact serialized v6 sizes (base and
        // full-with-WP16) BEFORE allocating or doing D2H. Every decision under TP must come
        // out IDENTICALLY on all ranks — the policy state it feeds decides future restores in
        // lockstep. So under TP we agree (via SpillFlag) whether all ranks can fit `full`
        // (policy + host MemAvailable headroom), falling back to an agreed `base` (0 WP16
        // checkpoints) when checkpoints exist, or an agreed pre-D2H skip. The write itself is
        // synchronous at this lockstep point (no async completion to race), and a disk or
        // capture failure on ANY rank drops the key EVERYWHERE (counted).
        let sizes = self.model.spill_capture_sizes(slot, old.len(), &cks);
        if self.tp.is_some() {
            let available = crate::memwatch::mem_available_bytes().unwrap_or(0);
            let budget = crate::kv_spill::host_capture_budget(
                available, crate::memwatch::watchdog_floor_bytes());
            let (base, full) = match sizes {
                Ok(s) => s,
                Err(ref e) => {
                    crate::dreprintln!("[kv-spill] capture size unavailable slot={slot} ({e:#})");
                    (0, 0)
                }
            };
            let sizes_ok = sizes.is_ok();
            let fits = |bytes| sizes_ok && bytes <= budget && self.spill_pol.would_admit(key, bytes, tokens);
            let has_cks = !cks.is_empty();
            let all_full = match self.tp.as_mut().unwrap().spill_agree(fits(full)) {
                Ok(v) => v,
                Err(e) => anyhow::bail!("kv spill: pre-capture agree failed slot={slot}: {e:#}"),
            };
            let (c0, cks) = if all_full {
                (c0, cks.as_slice())
            } else if has_cks {
                let all_base = match self.tp.as_mut().unwrap().spill_agree(fits(base)) {
                    Ok(v) => v,
                    Err(e) => anyhow::bail!("kv spill: pre-capture base agree failed slot={slot}: {e:#}"),
                };
                if all_base {
                    crate::drprintln!("[kv-spill] capture without WP16 (agreed): tokens={tokens} full={full} base={base} budget={budget} available={available}");
                    (0, &[][..])
                } else {
                    crate::drprintln!("[kv-spill] capture refused before D2H (agreed): tokens={tokens} full={full} base={base} budget={budget} available={available}");
                    crate::metrics::kv_spill_miss_recompute();
                    return Ok(());
                }
            } else {
                crate::drprintln!("[kv-spill] capture refused before D2H (agreed): tokens={tokens} full={full} base={base} budget={budget} available={available}");
                crate::metrics::kv_spill_miss_recompute();
                return Ok(());
            };
            let mut buf = Vec::new();
            let fixed = match self.model.spill_capture(slot, old.len(), &mut buf, c0 as u32, cks) {
                Ok(f) => f,
                Err(e) => {
                    crate::metrics::kv_spill_miss_recompute();
                    crate::dreprintln!("[exl3-serve] kv spill: capture failed slot={slot} ({e}); snapshot dropped");
                    // veloGB10 #44: a rank-local capture failure still casts its vote so peers stay paired.
                    if let Some(t) = self.tp.as_mut() {
                        if let Err(e2) = t.spill_agree(false) {
                            anyhow::bail!("kv spill: capture agree failed slot={slot}: {e2:#}");
                        }
                    }
                    return Ok(());
                }
            };
            crate::kv_spill::finalize_header(&mut buf, tokens);
            let wrote = st.put_sync(key, self.spill_rank, &buf);
            let all_ok = match self.tp.as_mut().map(|t| t.spill_agree(wrote.is_ok())) {
                Some(Ok(v)) => v,
                Some(Err(e)) => anyhow::bail!("kv spill: capture agree failed slot={slot}: {e:#}"),
                None => wrote.is_ok(),
            };
            if !all_ok {
                st.remove(key, self.spill_rank);
                crate::metrics::kv_spill_capture_peer_drop();
                crate::dreprintln!("[kv-spill] capture dropped: a rank's write failed (agreed across ranks; re-prefill on return)");
                return Ok(());
            }
            let bytes = wrote.unwrap_or(0);
            let snap = crate::kv_spill::Snap {
                conv: key,
                bytes,
                tokens,
                fixed,
                ck_c: c0,
                ck: cks.iter().map(|(p, k, _)| (*p, *k)).collect(),
            };
            for v in self.spill_pol.put_snap(snap) {
                st.remove(v, self.spill_rank);
            }
            if !self.spill_pol.contains(key) {
                // SPILLDUR2: actual serialized size could not displace deeper entries
                st.remove(key, self.spill_rank);
                crate::drprintln!("[kv-spill] capture not kept: admitting {tokens} tokens ({} MiB) would evict deeper snapshots", bytes >> 20);
                return Ok(());
            }
            crate::metrics::kv_spill_write(bytes, tokens);
            return Ok(());
        }
        // Single-rank (TP=1): pre-check queue room, host MemAvailable headroom above the
        // memwatch floor, and policy admission against the exact serialized v6 sizes BEFORE
        // allocating or copying from device memory. If the full snapshot (with WP16) does
        // not fit, fall back to a base v6 snapshot (0 WP16 checkpoints) before skipping.
        let (base, full) = match sizes {
            Ok(s) => s,
            Err(e) => {
                crate::metrics::kv_spill_miss_recompute();
                crate::drprintln!("[kv-spill] capture size unavailable slot={slot} ({e:#})");
                return Ok(());
            }
        };
        let pending = st.pending.load(std::sync::atomic::Ordering::Relaxed);
        let available = crate::memwatch::mem_available_bytes().unwrap_or(0);
        let budget = crate::kv_spill::capture_byte_budget(
            pending, st.max_pending, available, crate::memwatch::watchdog_floor_bytes());
        let fits = |bytes| bytes <= budget && self.spill_pol.would_admit(key, bytes, tokens);
        let (c0, cks) = if fits(full) {
            (c0, cks.as_slice())
        } else if full != base && fits(base) {
            crate::drprintln!("[kv-spill] capture without WP16: tokens={tokens} full={full} base={base} budget={budget} pending={pending} available={available}");
            (0, &[][..])
        } else {
            crate::drprintln!("[kv-spill] capture refused before D2H: tokens={tokens} full={full} base={base} budget={budget} pending={pending} available={available}");
            crate::metrics::kv_spill_miss_recompute();
            return Ok(());
        };
        let mut buf = Vec::new();
        let fixed = match self.model.spill_capture(slot, old.len(), &mut buf, c0 as u32, cks) {
            Ok(f) => f,
            Err(e) => {
                crate::metrics::kv_spill_miss_recompute();
                crate::drprintln!("[exl3-serve] kv spill: capture failed slot={slot} ({e}); snapshot dropped");
                return Ok(());
            }
        };
        crate::kv_spill::finalize_header(&mut buf, tokens);
        match st.put_prebuilt(key, self.spill_rank, buf) {
            Ok(bytes) => {
                let snap = crate::kv_spill::Snap {
                    conv: key,
                    bytes,
                    tokens,
                    fixed,
                    ck_c: c0,
                    ck: cks.iter().map(|(p, k, _)| (*p, *k)).collect(),
                };
                for v in self.spill_pol.put_snap(snap) {
                    st.remove(v, self.spill_rank);
                }
                if !self.spill_pol.contains(key) {
                    st.remove(key, self.spill_rank);
                    crate::drprintln!("[kv-spill] capture not kept: admitting {tokens} tokens ({} MiB) would evict deeper snapshots", bytes >> 20);
                    return Ok(());
                }
                crate::metrics::kv_spill_write(bytes, tokens);
            }
            Err(e) => {
                crate::metrics::kv_spill_miss_recompute();
                crate::drprintln!("[exl3-serve] kv spill: store put failed ({e}); snapshot dropped");
            }
        }
        Ok(())
    }

    /// Own-snapshot miss: try the spill store. Returns the restored token count (the new
    /// `reuse`) or None (miss / cost-model drop / any error — re-prefill, counted). The read
    /// is synchronous in the admission path, like the prefill it replaces, only ~50x shorter.
    /// Err only under TP, when the cross-rank agree failed or a restore failed AFTER every rank agreed to it (the ranks'
    /// KV would differ): the admission stops there with this message (ISSUES 121, as `spill_capture_slot`).
    pub(super) fn spill_restore_slot(&mut self, slot: usize, key: &[u32], end: usize) -> Result<Option<SpillHit>> {
        if self.spill.is_none() || self.spill_pol.is_empty() || end == 0 {
            return Ok(None);
        }
        let st = self.spill.clone().unwrap();
        let c = if self.wp16.is_some() { self.chunk.max(1) } else { 0 };
        let prefix = &key[..end.min(key.len())];
        // SPILLTRUNC: lookup picks Full (n <= end) or Trunc (q <= end) from replicated policy state only (TP-identical).
        let Some(hit) = self.spill_pol.lookup_at(prefix, c, crate::kv_spill::NVME_READ_BPS, crate::kv_spill::PREFILL_TPS) else {
            crate::metrics::kv_spill_miss_recompute();
            return Ok(None);
        };
        let conv = hit.conv();
        // SPILLDUR: on single-box, if the async capture for `conv` is still queued in the writer thread,
        // join it (condvar wait bounded by cost-model headroom, capped at SPILL_WAIT_MAX_SECS) before reading.
        // Skipped under TP where captures use `put_sync`.
        if self.tp.is_none() && st.is_pending(conv) {
            let (priced, saved) = match hit {
                crate::kv_spill::Hit::Full { n, bytes, .. } =>
                    (bytes.max((n as u64) * crate::kv_spill::KV_BYTES_PER_TOKEN), n as u64),
                crate::kv_spill::Hit::Trunc { q, read_bytes, .. } => (read_bytes, q as u64),
            };
            let budget = self.spill_pol.wait_budget_secs(priced, saved, crate::kv_spill::NVME_READ_BPS,
                                                         crate::kv_spill::PREFILL_TPS);
            let t0 = std::time::Instant::now();
            let joined = st.await_durable(conv, budget);
            let us = t0.elapsed().as_micros() as u64;
            crate::metrics::kv_spill_durability_wait(us, joined);
            crate::drprintln!("[kv-spill] capture still queued (conv {conv:016x}, {priced} bytes): waited {} ms, \
                               joined={joined} (budget {budget:.2} s)", us / 1000);
        }
        if let crate::kv_spill::Hit::Trunc { n, .. } = hit {
            let want = prefix.len();
            let rdres = self.model.spill_trunc_restore(slot, prefix, c, &mut |off: u64, len: usize| -> Result<Vec<u8>> {
                st.read_at(conv, self.spill_rank, 36 + off, len)
                    .map_err(|e| anyhow::anyhow!("span read at {off}+{len}: {e}"))
            });
            let all_ok = match self.tp.as_mut().map(|t| t.spill_agree(rdres.is_ok())) {
                Some(Ok(v)) => v,
                Some(Err(e)) => anyhow::bail!("kv spill: truncated-restore agree failed slot={slot}: {e:#}"),
                None => rdres.is_ok(),
            };
            if !all_ok {
                let why = rdres.err().map(|e| format!("{e:#}")).unwrap_or_default();
                crate::dreprintln!("[kv-spill] truncated restore dropped: conv {conv:016x} slot={slot} \
                                    rank={} ({why}) - dropped on all ranks, re-prefill", self.spill_rank);
                self.spill_pol.remove(conv);
                st.remove(conv, self.spill_rank);
                crate::metrics::kv_spill_restore_peer_miss();
                return Ok(None);
            }
            let (q, ck) = match rdres {
                Ok(v) => v,
                Err(e) => anyhow::bail!("kv spill: truncated restore failed slot={slot} ({e:#}) after the \
                                         cross-rank agree — ranks would diverge"),
            };
            if let Some(s16) = self.wp16.as_mut() {
                s16.install(slot, c, &key[..q], q, ck);
            }
            self.segs_set(slot, None, q);
            self.spill_pol.remove(conv);
            st.remove(conv, self.spill_rank);
            crate::metrics::kv_spill_hit(q as u64);
            crate::metrics::kv_spill_truncated_restore();
            crate::drprintln!("[exl3-serve] kv spill: truncated restore slot={slot}: rows 0..{q} + carried \
                               checkpoint of a {n}-token snapshot ({want}-token resumable prefix; priced {} MiB)",
                              hit.read_bytes() >> 20);
            return Ok(Some(SpillHit::Trunc(q)));
        }
        // LANERES3 (TP>=2): the lookup and cost model are pure replicated state (identical
        // on every rank); the local read's outcome is agreed with one SpillFlag, so a
        // snapshot only ONE rank can read is dropped on BOTH (counted) and both re-prefill
        // — the rounds stay identical, never a one-sided restore.
        let read = st.get(conv, self.spill_rank);
        // LANERES3: agree the PARSE outcome, not just the read — the payload must pass
        // `spill_restore_check` (magic/version/shape vs THIS model) on BOTH ranks before
        // either restores. A payload any rank cannot read OR parse (e.g. a file from a
        // pre-LANERES3 capture format) is dropped everywhere and both re-prefill: the
        // policy stays replicated-identical and neither rank acts alone.
        if self.tp.is_some() {
            let local_ok = matches!(&read, Ok((_, p)) if self.model.spill_restore_check(slot, p).is_ok());
            let all_ok = match self.tp.as_mut().map(|t| t.spill_agree(local_ok)) {
                Some(Ok(v)) => v,
                Some(Err(e)) => anyhow::bail!("kv spill: restore agree failed slot={slot}: {e:#}"),
                None => local_ok,
            };
            if !all_ok {
                // One rank could not read or parse it: drop everywhere (symmetric), re-prefill.
                let rerr = match &read {
                    Ok(_) => String::new(),
                    Err(e) => format!("read: {e}"),
                };
                let cerr = match &read {
                    Ok((_, p)) => match self.model.spill_restore_check(slot, p) {
                        Ok(_) => String::new(),
                        Err(e) => format!("check: {e:#}"),
                    },
                    Err(_) => String::new(),
                };
                crate::dreprintln!(
                    "[kv-spill] restore dropped: conv {conv:016x} slot={slot} rank={} \
                     local read=({}) local check=({}) - dropped on all ranks, re-prefill",
                    self.spill_rank, rerr, cerr);
                self.spill_pol.remove(conv);
                st.remove(conv, self.spill_rank);
                crate::metrics::kv_spill_restore_peer_miss();
                return Ok(None);
            }
        }
        let payload = match read {
            Ok((_, p)) => p,
            Err(_) => {
                self.spill_pol.remove(conv);
                st.remove(conv, self.spill_rank);
                crate::metrics::kv_spill_miss_recompute();
                return Ok(None);
            }
        };
        let n = match self.model.spill_restore(slot, &payload) {
            Ok(n) => n,
            Err(e) => {
                // LANERES3 (TP>=2): after the cross-rank agree a one-sided failure means
                // the ranks' KV would differ — fatal-grade, reported loudly, never a
                // silent re-prefill (single-box keeps the graceful re-prefill).
                if self.tp.is_some() {
                    anyhow::bail!("kv spill: restore failed slot={slot} ({e}) after the cross-rank agree — ranks would diverge");
                }
                crate::metrics::kv_spill_miss_recompute();
                crate::drprintln!("[exl3-serve] kv spill: restore failed slot={slot} ({e}); re-prefill");
                return Ok(None);
            }
        };
        self.spill_pol.remove(conv);
        st.remove(conv, self.spill_rank);
        crate::metrics::kv_spill_hit(n as u64);
        crate::drprintln!("[exl3-serve] kv spill: restored slot={slot}: {n} of {end} prompt tokens (read {} MiB)", payload.len() >> 20);
        Ok(Some(SpillHit::Full(n)))
    }

    // ---------------------------------------------------------- LANERES1: --lane-reserve
    /// The SlotViews the policy reads, built fresh at every plan/pick: busy, snapshot present,
    /// release time + normal-ness of the last release.
    pub(super) fn slot_views(&self) -> Vec<crate::lane_reserve::SlotView> {
        (0..self.lanes.len()).map(|i| crate::lane_reserve::SlotView {
            busy: self.lanes[i].is_some(),
            has_snapshot: self.cache[i].is_some(),
            released_at: self.rel[i].map(|(t, _)| t),
            normal_finish: self.rel[i].map_or(false, |(_, n)| n),
        }).collect()
    }
}
