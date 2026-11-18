//! `impl Engine`: load (plain, speculative, lanes), prefix snapshots, speculative rounds, request begin,
//! lane steps, prefill and decode. Siblings: `loops.rs` drives these per request; `mod.rs` has the type and the map.

use super::*;

impl Engine {
    /// `n_layers` < the model's = a SMOKE configuration (garbage text, the full serve machinery): the
    /// `--layers N` flag, shipped to the node in the head's argv.
    pub fn load(dev: Arc<CudaDevice>, pack: &NaivePack, shard: Shard, max_pos: usize, n_layers: usize, head: bool, chunk: usize) -> Result<Engine> {
        Self::load_spec(dev, pack, shard, max_pos, n_layers, head, chunk, None)
    }

    /// `spec` = Some: the verify block step on every rank; the drafter + full lm_head on the head only.
    #[allow(clippy::too_many_arguments)]
    pub fn load_spec(dev: Arc<CudaDevice>, pack: &NaivePack, shard: Shard, max_pos: usize, n_layers: usize, head: bool, chunk: usize,
                     spec: Option<&SpecCfg>) -> Result<Engine> {
        Self::load_spec_lanes(dev, pack, shard, max_pos, n_layers, head, chunk, spec, 1)
    }

    /// `load_spec` with G1 lanes (`--naive-lanes N`): N KV slots, the shared lane table, the m = 2..=N lane steps.
    #[allow(clippy::too_many_arguments)]
    pub fn load_spec_lanes(dev: Arc<CudaDevice>, pack: &NaivePack, shard: Shard, max_pos: usize, n_layers: usize, head: bool, chunk: usize,
                           spec: Option<&SpecCfg>, lanes: usize) -> Result<Engine> {
        ensure!((1..=4).contains(&lanes), "--naive-lanes {lanes} (1..=4)");
        crate::metrics::NAIVE_LANES.store(lanes as u64, std::sync::atomic::Ordering::Relaxed);
        // the lane count is this server's capacity: running vs waiting and the uptime clock come from it (was never set: "up 0 min")
        crate::metrics::set_max_batch(lanes);
        let t0 = std::time::Instant::now();
        ensure!(chunk >= 1 && chunk <= super::model::MAX_PREFILL_ROWS, "--prefill-chunk {chunk} (1..{})", super::model::MAX_PREFILL_ROWS);
        // every row of a chunk writes its K/V before any row attends: the SWA ring must hold the chunk + the window
        ensure!(chunk + pack.cfg.sliding_window <= SWA_RING, "--prefill-chunk {chunk} + window {} exceeds the SWA ring {SWA_RING}", pack.cfg.sliding_window);
        ensure!(n_layers >= 1 && n_layers <= pack.cfg.num_layers, "--layers {n_layers}");
        if n_layers < pack.cfg.num_layers { println!("[naive-serve] SMOKE configuration: {n_layers} of {} layers — the text is NOT the model's", pack.cfg.num_layers); }
        let model = GpuModel::load_slice_on(dev.clone(), pack, n_layers, max_pos, SWA_RING, lanes, shard)?;
        crate::dprintln!("[naive-serve] rank {}/{} shard loaded ({n_layers} layers, max_pos {max_pos}) in {:.1} s", shard.rank, shard.world,
                         t0.elapsed().as_secs_f64());
        let pf_ws = Ws::new(&dev, ws_bytes(&pack.cfg, chunk, max_pos, 0))?;
        let (mut ver, mut hs, mut pf_taps, mut block, mut conf) = (None, None, None, 1usize, 0f32);
        let mut draft_temp: f32 = SPEC_DRAFT_TEMP;
        let mut sharded = false;
        let mut batched = false;
        let mut guard = (SPEC_GUARD_OFF, SPEC_GUARD_ON, SPEC_PROBE_EVERY);
        let mut vers = Vec::new();
        // the plain step carries the drafter's taps too when speculation is on: a plain step's row is
        // context the drafter must see (a hole in its context after every plain step cost acceptance)
        let mut dec_taps: Vec<usize> = Vec::new();
        if let Some(sc) = spec {
            ensure!((2..=MAX_ROWS).contains(&sc.block), "--spec-block {} (2..=16)", sc.block);
            block = sc.block;
            conf = sc.conf;
            draft_temp = sc.draft_temp;
            guard = (sc.guard_off, sc.guard_on, sc.probe_every);
            let h = pack.cfg.hidden_size;
            let mut taps: Vec<usize> = Vec::new();
            // work item B: where the drafter runs (graceful fallback, never a refusal: a combination the sharded
            // drafter cannot serve runs head-only and says so)
            let world = model.shard.world;
            let want_sharded = match sc.drafter.as_str() { "sharded" => true, "head" => false, _ => world > 1 };
            sharded = want_sharded && world > 1 && sc.draft_temp <= 0.0;
            if want_sharded && world > 1 && !sharded {
                println!("[naive-serve] --spec-drafter {}: sampled drafts (--spec-draft-temp {}) need the whole vocabulary's q on one rank; running the drafter HEAD-ONLY instead", sc.drafter, sc.draft_temp);
            }
            if head || sharded {
                let dd = sc.draft_dir.as_ref().context("the head needs --draft-dir (or the drafter next to the pack)")?;
                super::drafter_gpu::load_module(&dev)?;
                let drafter = if sharded { GpuDrafter::load_sharded(&dev, dd, "model.safetensors", model.shard, model.v0, model.vl)? }
                              else { GpuDrafter::load(&dev, dd, "model.safetensors")? };
                ensure!(drafter.cfg.hidden == h && drafter.cfg.vocab == pack.cfg.vocab_size, "drafter geometry does not match the target");
                taps = drafter.cfg.taps.clone();
                let lm_full = if sharded { None } else { Some(super::gpu::upload_quad(&dev, pack, pack.quad("lm_head")?)?) };
                let ws = Ws::new(&dev, 512 << 20)?;
                pf_taps = Some(dev.alloc_zeros::<f32>(taps.len() * chunk * h)?);
                crate::dprintln!("[naive-serve] S8 drafter loaded from {dd}: {} layers, taps {:?}, block {block}, {} (rank {}/{}: {} q heads, {} kv heads, inter {}, fc rows {}, vocab [{}, {}))",
                                 drafter.cfg.layers, taps, if sharded { "SHARDED" } else { "head-only" }, model.shard.rank, world,
                                 drafter.nh_l, drafter.kv_l, drafter.inter_l, drafter.h_l, drafter.v0, drafter.v0 + drafter.vl);
                hs = Some(HeadSpec { drafter, lm_full, ws,
                                     qbuf: dev.alloc_zeros::<f32>(16 * pack.cfg.vocab_size + 2 * 1024)?,
                                     qstat: dev.alloc_zeros::<f32>(2 * (block - 1))? });
            }
            for m in 2..block { vers.push(DecodeStep::new_with_taps(&model, m, head, taps.clone())?); }
            ver = Some(DecodeStep::new_with_taps(&model, block, head, taps.clone())?);
            // work item C: the verify sampler's exchange pattern, identical on every rank (the doorbell sequence must match)
            batched = match sc.sampler.as_str() { "batched" => true, "rows" => false, _ => world > 1 };
            for v in vers.iter_mut() { v.set_batched(batched); }
            if let Some(v) = ver.as_mut() { v.set_batched(batched); }
            if head { println!("[naive-serve] verify sampler: {} (--spec-sampler {})", if batched { "BATCHED exchange (one per phase for all rows)" } else { "per-row exchange" }, sc.sampler); }
            // the verify graphs bake the q address at capture: assign it now, before `arm()`
            if let Some(h) = hs.as_ref() {
                let (qp, qz) = (dptr(&h.qbuf), dptr(&h.qstat));
                for v in vers.iter_mut() { v.smp_q = qp; v.smp_qstat = qz; v.smp_stride = pack.cfg.vocab_size; }
                if let Some(v) = ver.as_mut() { v.smp_q = qp; v.smp_qstat = qz; v.smp_stride = pack.cfg.vocab_size; }
            }
            dec_taps = taps;
        }
        let mut dec = DecodeStep::new_with_taps(&model, 1, head, dec_taps.clone())?;
        // G1: one sampler table for every step, and the multi-request steps (each with the drafter's taps, so the
        // request that owns the drafter context keeps feeding it while it shares a step)
        let (mut lane_tab, mut lsteps) = (None, Vec::new());
        if lanes > 1 {
            let tab = super::decode::LaneTab::new(&model, lanes)?;
            dec.use_lane_tab(&tab);
            for v in vers.iter_mut() { v.use_lane_tab(&tab); }
            if let Some(v) = ver.as_mut() { v.use_lane_tab(&tab); }
            for m in 2..=lanes {
                let mut st = DecodeStep::new_with_taps(&model, m, head, dec_taps.clone())?;
                st.use_lane_tab(&tab);
                st.set_multilane(true);
                lsteps.push(st);
            }
            lane_tab = Some(tab);
        }
        let xd = dev.alloc_zeros::<f32>(32)?;
        // G1 + speculation: a context per lane (a lane's ring set is the drafter's own geometry)
        let mut lane_ctx = Vec::new();
        if lanes > 1 && spec.is_some_and(|sc| sc.lane_ctx) {
            if let Some(h) = hs.as_ref() {
                for _ in 0..lanes {
                    let mut set = Vec::new();
                    for l in &h.drafter.layers {
                        set.push((dev.alloc_zeros::<u16>(cudarc::driver::DeviceSlice::len(&l.kc))?, dev.alloc_zeros::<u16>(cudarc::driver::DeviceSlice::len(&l.vc))?));
                    }
                    lane_ctx.push(set);
                }
            }
        }
        crate::metrics::NAIVE_LANE_CTX.store(!lane_ctx.is_empty(), std::sync::atomic::Ordering::Relaxed);
        dev.synchronize()?;
        Ok(Engine { model, max_pos, pf_ws, dec, chunk, ver, vers, spec: hs, pf_taps, xd, block, conf, draft_temp, last_width: 0, guard,
                    sp_cur: SampleParams::GREEDY, seed_cur: 0, smp_round: 0, smp_fallbacks: 0, smp_nolist: 0, sharded, batched, node_drafts: None, draft_secs: 0.0, draft_onehot: true,
                    lanes, lane_tab, lsteps, pfx: Vec::new(), lane_ctx, ctx_live: None, spec_id: u64::MAX })
    }

    /// Prefix reuse: allocate the snapshot buffers (`--prefix-cache on`), one set per KV slot (one per lane).
    pub fn enable_prefix(&mut self) -> Result<()> {
        let dev = self.model.dev.clone();
        let (w, hd, vd) = (self.model.cfg.sliding_window, self.model.cfg.head_dim, self.model.cfg.v_head_dim);
        let mut sets = Vec::new();
        for _ in 0..self.model.n_slots {
            let mut swa = Vec::new();
            for (i, l) in self.model.layers.iter().enumerate() {
                if l.kind != super::config::AttnKind::Swa { continue; }
                ensure!(w <= l.slots, "window {w} > ring {}", l.slots);
                swa.push((i, dev.alloc_zeros::<u16>(l.kv_l * w * hd)?, dev.alloc_zeros::<u16>(l.kv_l * w * vd)?));
            }
            let mut dr = Vec::new();
            if let Some(hs) = self.spec.as_ref() {
                for l in &hs.drafter.layers {
                    dr.push((dev.alloc_zeros::<u16>(cudarc::driver::DeviceSlice::len(&l.kc))?, dev.alloc_zeros::<u16>(cudarc::driver::DeviceSlice::len(&l.vc))?));
                }
            }
            sets.push(PrefixSnap { toks: None, dr_ok: false, swa, dr });
        }
        dev.synchronize()?;
        self.pfx = sets;
        Ok(())
    }

    pub fn pfx_on(&self) -> bool { !self.pfx.is_empty() }

    /// G1 + prefix reuse: the free KV slot a new request should take: the one whose snapshot it resumes furthest into,
    /// else one holding no snapshot (keep the others' conversations), else the most recently freed. Returns (index into
    /// `free`, resume point, snapshot point).
    pub fn prefix_pick(&self, free: &[usize], prompt: &[u32], ckpt_at: Option<usize>, feeds: bool) -> (usize, usize, Option<usize>) {
        let last = free.len() - 1;
        if !self.pfx_on() { return (last, 0, None); }
        let mut best = (last, 0usize, self.prefix_plan(free[last], prompt, ckpt_at, feeds).1);
        let mut empty: Option<usize> = None;
        for (i, &sl) in free.iter().enumerate() {
            let (r, sn) = self.prefix_plan(sl, prompt, ckpt_at, feeds);
            if r > best.1 { best = (i, r, sn); }
            if self.pfx[sl].toks.is_none() && empty.is_none() { empty = Some(i); }
        }
        if best.1 == 0 { if let Some(i) = empty { return (i, 0, self.prefix_plan(free[i], prompt, ckpt_at, feeds).1); } }
        best
    }

    /// Prefix reuse, the head's decision for a prompt: (the position its prefill resumes at, 0 = from the start; the
    /// chunk boundary its own snapshot goes to). `ckpt_at` = the template's message boundary (what the next turn
    /// replays verbatim), else the prompt less `PREFIX_MARGIN`; the snapshot is the last chunk boundary at or before it.
    /// `feeds`: this prefill feeds the drafter, so a snapshot without the drafter's context cannot be resumed.
    pub fn prefix_plan(&self, slot: usize, prompt: &[u32], ckpt_at: Option<usize>, feeds: bool) -> (usize, Option<usize>) {
        if !self.pfx_on() { return (0, None); }
        let p = &self.pfx[slot];
        let held = if feeds && self.spec.is_some() && !p.dr_ok { None } else { p.toks.as_deref() };
        plan_prefix(held, prompt, ckpt_at, self.chunk, self.model.cfg.sliding_window)
    }

    /// `to_snap`: the sliding-window state at position `at` -> the snapshot (else back into the rings). Every SWA
    /// layer's entries of [at - window, at) (per kv head, one or two runs of the ring) and the drafter's whole rings.
    fn prefix_copy(&self, slot: usize, at: usize, to_snap: bool, drafter: bool) -> Result<()> {
        let s = self.model.stream.stream;
        let (w, hd, vd) = (self.model.cfg.sliding_window, self.model.cfg.head_dim, self.model.cfg.v_head_dim);
        ensure!(at >= w, "prefix snapshot at {at} < window {w}");
        let cp = |dst: u64, src: u64, bytes: usize| -> Result<()> {
            unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, s) }.map_err(|e| anyhow::anyhow!("prefix snapshot copy: {e:?}"))
        };
        let ps = &self.pfx[slot];
        for (li, ks, vs) in &ps.swa {
            let l = &self.model.layers[*li];
            for (cache, snap, d) in [(dptr(&l.kc), dptr(ks), hd), (dptr(&l.vc), dptr(vs), vd)] {
                // the KV slot's block of the cache: [n_slots][kv][S][d]
                let cache = cache + ((slot * l.kv_l * l.slots * d) * 2) as u64;
                for kh in 0..l.kv_l {
                    let mut off = 0usize;
                    for (rs, n) in ring_runs(at - w, w, l.slots) {
                        let r = cache + (((kh * l.slots + rs) * d) * 2) as u64;
                        let q = snap + (((kh * w + off) * d) * 2) as u64;
                        if to_snap { cp(q, r, n * d * 2)?; } else { cp(r, q, n * d * 2)?; }
                        off += n;
                    }
                }
            }
        }
        if let Some(hs) = self.spec.as_ref().filter(|_| drafter) {
            for (l, (ks, vs)) in hs.drafter.layers.iter().zip(&ps.dr) {
                for (cache, snap, len) in [(dptr(&l.kc), dptr(ks), cudarc::driver::DeviceSlice::len(&l.kc)), (dptr(&l.vc), dptr(vs), cudarc::driver::DeviceSlice::len(&l.vc))] {
                    if to_snap { cp(snap, cache, len * 2)?; } else { cp(cache, snap, len * 2)?; }
                }
            }
        }
        Ok(())
    }

    /// The prompt's prefill with prefix reuse: resume at `reuse` (the slot must hold exactly that prefix: every rank
    /// checks its own), take this prompt's snapshot at `snap`. Same chunks as a whole prefill, so the same state.
    /// `slot` = the KV slot the prompt is prefilled into (lanes 1: 0); `feed` = the prefill feeds the drafter (its context
    /// is then part of the snapshot; with per-lane contexts the caller made the lane's context live first).
    pub fn prefill_resume(&mut self, slot: usize, prompt: &[u32], reuse: usize, snap: Option<usize>, feed: bool) -> Result<Option<u64>> {
        use std::sync::atomic::Ordering::Relaxed;
        let dr = feed && self.spec.is_some();
        if reuse > 0 {
            ensure!(self.pfx_on() && self.pfx[slot].toks.as_ref().is_some_and(|t| t.len() == reuse && prompt.len() > reuse && prompt[..reuse] == t[..])
                    && (!dr || self.pfx[slot].dr_ok),
                    "prefix reuse at {reuse} in slot {slot}: this rank's slot does not hold that prefix (the ranks' prefix state diverged)");
            self.prefix_copy(slot, reuse, false, dr)?;
            crate::metrics::NAIVE_PREFIX_HITS.fetch_add(1, Relaxed);
            crate::metrics::NAIVE_PREFIX_REUSED.fetch_add(reuse as u64, Relaxed);
        } else if self.pfx_on() {
            crate::metrics::NAIVE_PREFIX_MISSES.fetch_add(1, Relaxed);
            self.pfx[slot].toks = None; // the prefill below rewrites the slot from position 0
        }
        if let Some(c) = snap.filter(|&c| self.pfx_on() && c > reuse && c < prompt.len()) {
            self.prefill_feed(&prompt[reuse..c], reuse, false, feed)?;
            self.model.sync()?;
            self.prefix_copy(slot, c, true, dr)?;
            self.pfx[slot].toks = Some(prompt[..c].to_vec());
            self.pfx[slot].dr_ok = dr;
            return self.prefill_feed(&prompt[c..], c, true, feed);
        }
        self.prefill_feed(&prompt[reuse..], reuse, true, feed)
    }

    /// The verify width for this round's draft confidences: the longest prefix clearing the threshold.
    pub fn pick_width(&self, conf: &[f32]) -> usize {
        let n = self.block - 1;
        if self.conf <= 0.0 { return n.min(conf.len()); }
        conf.iter().take(n).take_while(|&&c| c >= self.conf).count()
    }

    /// True when the last speculative round ran as a plain step (verify width 0): its token was counted by
    /// the plain step's own note kernel, so the commit bookkeeping must not count it into `dec` again.
    pub fn last_round_plain(&self) -> bool { self.last_width == 0 }

    /// Every verify step sees the head's cancel flag for the next round.
    pub fn set_cancel_all(&mut self, cancel: bool) -> Result<()> {
        self.dec.set_cancel(&self.model, cancel)?;
        for v in self.vers.iter_mut() { v.set_cancel(&self.model, cancel)?; }
        if let Some(v) = self.ver.as_mut() { v.set_cancel(&self.model, cancel)?; }
        Ok(())
    }

    /// Committed tokens into every step's penalty table. A plain step already counted its own token into
    /// `dec` (the step's note kernel), so `from_plain` skips `dec`; a verify round counted nothing itself.
    pub fn note_commit_all(&mut self, toks: &[u32], from_plain: bool) -> Result<()> {
        if !from_plain { self.dec.note_commit(&self.model, toks)?; }
        for v in self.vers.iter_mut() { v.note_commit(&self.model, toks)?; }
        if let Some(v) = self.ver.as_mut() { v.note_commit(&self.model, toks)?; }
        Ok(())
    }

    /// A plain step inside a speculative request: the sampled token, and its row appended to the drafter's
    /// context (the head), so the next draft sees it.
    pub fn plain_step_spec(&mut self, t: u32, pos: usize) -> Result<(u32, bool)> {
        let r = self.dec.step(&self.model, &[t], &[pos])?;
        if let Some(hs) = self.spec.as_mut() {
            let taps = self.dec.tap_view(&self.model);
            hs.ws.reset();
            hs.drafter.append_context(&self.model.stream, &mut hs.ws, &taps, 1, pos, self.model.tp.as_ref())?;
        }
        Ok(r)
    }

    /// One speculative round on every rank. The head passes its drafts (block - 1 tokens); the node passes
    /// None and reads them from the slot exchange. Returns (committed tokens, rows accepted a, cancel flag):
    /// committed = drafts[..a] + the bonus sample of row a (a + 1 tokens, positions pos+1 ..). The target
    /// rows [t_last, d_1..d_{B-1}] at positions pos.. are verified in one graphed step; row j's sample is
    /// accepted while it equals d_{j+1} ("match": the samples are exact draws from the target's sampled
    /// distribution, so the output distribution is the plain one at every temperature; greedy is exact).
    pub fn spec_round(&mut self, drafts: Option<(&[u32], usize)>, t_last: u32, pos: usize) -> Result<(Vec<u32>, usize, bool)> {
        let b = self.block;
        let n = b - 1;
        // the drafts exchange: head writes slot 0..n and the verify width k in slot 31, node zeros; the
        // all-reduce gathers (adding zeros), so every rank runs the same width on the same drafts
        let dev = self.model.dev.clone();
        let mut slots = vec![0f32; 32];
        if let Some((d, k)) = drafts {
            ensure!(d.len() == n, "drafts {} != block-1 {n}", d.len());
            ensure!(k <= n, "verify width {k} > {n} drafts");
            for (i, t) in d.iter().enumerate() { slots[i] = *t as f32; }
            slots[31] = k as f32;
        }
        self.model.sync()?;
        dev.htod_sync_copy_into(&slots, &mut self.xd)?;
        if let Some(tp) = &self.model.tp { tp.all_reduce(&self.model.stream, &self.xd, 32)?; self.model.sync()?; }
        let got = dev.dtoh_sync_copy(&self.xd)?;
        let k = got[31] as usize;
        ensure!(k <= n, "exchanged verify width {k} > {n}");
        if let Some((nd, nk)) = self.node_drafts.take() {
            // sharded drafter: this node drafted the same block itself; the exchanged (head) drafts must match
            let ex: Vec<u32> = got[..n].iter().map(|x| *x as u32).collect();
            ensure!(ex == nd && k == nk, "SHARDED DRAFTER DIVERGED: node {}/{} drafts {nd:?} width {nk}, head {ex:?} width {k} (pos {pos})",
                    self.model.shard.rank, self.model.shard.world);
        }
        self.last_width = k;
        if k == 0 {
            // nothing confident enough to verify: a plain step (its row still feeds the drafter's context)
            let (tok, cancel) = self.plain_step_spec(t_last, pos)?;
            return Ok((vec![tok], 0, cancel));
        }
        let drafts: Vec<u32> = got[..k].iter().map(|x| *x as u32).collect();
        // the verify block: t_last + the k drafts
        let mut ids = Vec::with_capacity(k + 1); ids.push(t_last); ids.extend_from_slice(&drafts);
        let posv: Vec<usize> = (pos..pos + k + 1).collect();
        ensure!(pos + k + 1 <= self.max_pos, "speculative block past the context");
        // the verify step for k drafts (m = k + 1): `ver` is m = block, `vers[k - 1]` the narrower ones
        // (inline so the borrow is of these two fields only, not of `self.model`)
        let ver: &mut DecodeStep = if k + 1 == self.block { self.ver.as_mut().context("no verify step")? } else { &mut self.vers[k - 1] };
        // Work item A: a SAMPLED request takes the exact accept path (ratio test + residual); greedy
        // keeps the token-equality path bit-for-bit. NOTE: a node rank has no drafter (`spec` is None)
        // but MUST take this branch too — its accept kernel no-ops (smp_on stays 0) and its committed
        // list arrives in the ship below; taking the greedy branch there would pair the equality rule
        // with sampled drafts and never receive the head's tokens.
        let sampled = self.sp_cur.temperature > 0.0;
        self.smp_round += 1;
        let (xs, cancel) = if sampled {
            ver.step_block_smp(&self.model, &ids, &posv, Some((&drafts, self.seed_cur ^ (self.smp_round << 32), self.smp_round, self.draft_onehot)))?
        } else {
            ver.step_block(&self.model, &ids, &posv)?
        };
        let mut a = 0usize;
        let mut committed: Vec<u32>;
        if sampled {
            // rows accept independently (each has its own p, q and uniform); the block stops at the
            // first rejection, whose committed token is the residual draw
            let (acc, rtok) = ver.smp_accepts(&self.model, k)?;
            while a < k && acc[a] == 1 { a += 1; }
            committed = drafts[..a].to_vec();
            committed.push(if a < k {
                if rtok[a] < 0 {
                    self.smp_fallbacks += 1;
                    if rtok[a] == -1 { self.smp_nolist += 1; crate::metrics::NAIVE_SPEC_FALLBACK_NOLIST.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                    else { crate::metrics::NAIVE_SPEC_FALLBACK_EMPTY.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                    xs[a]
                } else { rtok[a] as u32 }
            } else { xs[k] });
            if self.model.shard.rank == 0 {
                crate::metrics::NAIVE_SPEC_SAMPLED_ROUNDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::metrics::NAIVE_SPEC_SAMPLED_ACCEPTED.fetch_add(a as u64, std::sync::atomic::Ordering::Relaxed);
            }
            ensure!(committed.iter().all(|&t| (t as usize) < self.model.cfg.vocab_size),
                    "spec round: committed an out-of-vocabulary id {:?} (a {a} of {k}, rtok {:?}, xs {:?})", committed, &rtok[..k], &xs[..xs.len().min(8)]);
        } else {
            while a < k && xs[a] == drafts[a] { a += 1; }
            committed = drafts[..a].to_vec();
            committed.push(xs[a]);
        }
        // TP: the chain is decided on the head (q lives only there); ship the committed list so every
        // rank commits the same tokens and writes the same KV.
        if sampled {
            if let Some(tp) = self.model.tp.as_ref() {
                let mut slots = vec![0f32; 32];
                if self.model.shard.rank == 0 {
                    for (i, t) in committed.iter().enumerate().take(31) { slots[i] = *t as f32; }
                    slots[31] = committed.len() as f32;
                }
                self.model.sync()?;
                dev.htod_sync_copy_into(&slots, &mut self.xd)?;
                tp.all_reduce(&self.model.stream, &self.xd, 32)?;
                self.model.sync()?;
                let got2 = dev.dtoh_sync_copy(&self.xd)?;
                let cl = got2[31] as usize;
                ensure!(cl >= 1 && cl <= 31, "spec ship: committed length {cl}");
                committed = got2[..cl].iter().map(|x| *x as u32).collect();
                a = cl - 1;
            }
        }
        // the drafter's context: rows 0..=a of the block (t_last and the accepted drafts) at positions pos..
        let taps = ver.tap_view(&self.model);
        if let Some(hs) = self.spec.as_mut() {
            hs.ws.reset();
            hs.drafter.append_context(&self.model.stream, &mut hs.ws, &taps, a + 1, pos, self.model.tp.as_ref())?;
        }
        Ok((committed, a, cancel))
    }

    /// A request begins on this rank: the sampler state of the plain step AND the verify block.
    pub fn begin_request(&mut self, sp: &SampleParams, seed: u64, prompt: &[u32]) -> Result<()> {
        self.begin_request_lane(sp, seed, prompt, 0)
    }

    /// G1: as `begin_request`, but the request takes lane `l` of BOTH the decode step and the verify
    /// block, so a second concurrent request cannot overwrite this one's knobs, RNG or penalty
    /// tables. `begin_request` is lane 0, which is what every single-stream caller wants unchanged.
    pub fn begin_request_lane(&mut self, sp: &SampleParams, seed: u64, prompt: &[u32], l: usize) -> Result<()> {
        // work item A: the drafter draws from the REQUEST's temperature and the accept draws are keyed
        // on (request seed, round)
        self.sp_cur = *sp;
        self.seed_cur = seed;
        self.smp_round = 0;
        self.smp_fallbacks = 0;
        self.smp_nolist = 0;
        self.draft_secs = 0.0;
        self.node_drafts = None;
        self.dec.begin_request_lane(&self.model, sp, seed, prompt, l)?;
        for (i, v) in self.vers.iter_mut().enumerate() { v.begin_request_lane(&self.model, sp, seed ^ (0x5EC0_0000_0000_0010 + i as u64), prompt, l)?; }
        if let Some(v) = self.ver.as_mut() { v.begin_request_lane(&self.model, sp, seed ^ 0x5EC0_0000_0000_0001, prompt, l)?; }
        Ok(())
    }

    /// The head's drafts for the next round: the anchor `t_last` at position `pos`, block - 1 tokens.
    pub fn draft(&mut self, t_last: u32, pos: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        let (t, c, _) = self.draft_q(t_last, pos)?;
        Ok((t, c))
    }

    /// Work item A: as `draft`, but a SAMPLED request draws d_j ~ q_j (Gumbel-max over the corrected,
    /// T-scaled logits) and materialises the full-vocabulary q the verify side needs for the ratio
    /// test. The third value says whether q is valid for this round (greedy, or no head = false).
    pub fn draft_q(&mut self, t_last: u32, pos: usize) -> Result<(Vec<u32>, Vec<f32>, bool)> {
        let n = self.block - 1;
        let sampled = self.sp_cur.temperature > 0.0;
        let (qp, qz) = match self.spec.as_ref() { Some(h) => (dptr(&h.qbuf), dptr(&h.qstat)), None => (0u64, 0u64) };
        let seed = self.seed_cur ^ ((pos as u64) << 24);
        let t0 = std::time::Instant::now();
        let draft_temp = self.draft_temp;
        let sp_t = self.sp_cur.temperature;
        let hs = self.spec.as_mut().context("no drafter on this rank")?;
        hs.ws.reset();
        let model = &self.model;
        // head-only: the full lm_head; sharded: this rank's vocabulary slice (the Markov stage gathers the argmax)
        let lm_q: &super::gpu::GpuQuad = hs.lm_full.as_ref().unwrap_or(&model.lm_head);
        let mut lm = |ws: &mut Ws, x16: u64, m: usize| -> Result<u64> { model.chain_ws(lm_q, x16, m, ws) };
        // argmax drafts unless the draft temperature is > 0 (F = 0 is the default: see SPEC_DRAFT_TEMP)
        let smp = if sampled && qp != 0 && draft_temp > 0.0 && hs.lm_full.is_some() {
            Some(super::drafter_gpu::SmpDraft { t: (sp_t * draft_temp).max(1e-3), rng: seed, smb: qp, qstat: qz })
        } else { None };
        let r = hs.drafter.round(&model.stream, &mut hs.ws, pos, t_last, n, dptr(&model.embed), &mut lm, smp, model.tp.as_ref())?;
        let ok = smp.is_some();
        self.draft_onehot = smp.is_none();
        self.draft_secs += t0.elapsed().as_secs_f64();
        Ok((r.tokens, r.confidence, ok))
    }

    /// Record the decode graph (after the TP transport is attached: the reduces are in the graph).
    pub fn arm(&mut self) -> Result<()> {
        let t0 = std::time::Instant::now();
        self.dec.capture(&self.model)?;
        for v in self.vers.iter_mut() { v.capture(&self.model)?; }
        if let Some(v) = self.ver.as_mut() { v.capture(&self.model)?; }
        for v in self.lsteps.iter_mut() { v.capture(&self.model)?; }
        crate::dprintln!("[naive-serve] decode graph captured in {:.2} s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Prefill `ids` at positions start.. in chunks of MAX_ROWS rows (eager; S6b replaces the 16-row
    /// chunks with the wide prefill path). Returns the LAST row's logits (f16 [V], a device address
    /// inside the prefill workspace, valid until the next prefill) when asked.
    pub fn prefill(&mut self, ids: &[u32], start: usize, want_logits: bool) -> Result<Option<u64>> {
        self.prefill_feed(ids, start, want_logits, true)
    }

    /// G1: the KV slot every row of the next step(s) reads and writes (the verify block, a prefill chunk: one request).
    pub fn set_slot_all(&mut self, slot: usize) -> Result<()> {
        if self.lanes <= 1 { return Ok(()); }
        self.model.sync()?;
        let n = cudarc::driver::DeviceSlice::len(&self.model.sids_buf);
        self.model.dev.htod_sync_copy_into(&vec![slot as i32; n], &mut self.model.sids_buf)?;
        Ok(())
    }

    /// G1: per-row KV slots of a multi-request step (row j = lane j's slot; the rest of the buffer keeps the last one).
    pub fn set_slots(&mut self, slots: &[usize]) -> Result<()> {
        self.model.sync()?;
        let n = cudarc::driver::DeviceSlice::len(&self.model.sids_buf);
        let mut v: Vec<i32> = slots.iter().map(|s| *s as i32).collect();
        let last = *v.last().unwrap_or(&0);
        v.resize(n, last);
        self.model.dev.htod_sync_copy_into(&v, &mut self.model.sids_buf)?;
        Ok(())
    }

    /// G1: one multi-request step over `ids`/`pos` (row j = lane j, KV slot `slots[j]`) -> each lane's token.
    /// `owner`: the lane whose request owns the drafter context; its row's taps are appended to that context
    /// (every rank that runs the drafter does the same), so the request can speculate again when it is alone.
    pub fn step_lanes(&mut self, ids: &[u32], pos: &[usize], slots: &[usize], owner: Option<usize>) -> Result<Vec<u32>> {
        let m = ids.len();
        ensure!(m >= 1 && m <= self.lanes && pos.len() == m && slots.len() == m, "lane step: {m} rows of {} lanes", self.lanes);
        self.set_slots(slots)?;
        let toks = if m == 1 { vec![self.dec.step(&self.model, ids, pos)?.0] } else { self.lsteps[m - 2].step_lanes(&self.model, ids, pos)? };
        if self.per_lane_ctx() && self.spec.is_some() {
            // every lane's row into ITS context (the same appends on every rank that runs the drafter)
            let h = self.model.cfg.hidden_size;
            for j in 0..m {
                self.ctx_make_live(j)?;
                let st: &DecodeStep = if m == 1 { &self.dec } else { &self.lsteps[m - 2] };
                let taps: Vec<(u64, usize)> = st.tap_view(&self.model).iter().map(|(a, p)| (a + (j * h * 4) as u64, *p)).collect();
                let hs = self.spec.as_mut().unwrap();
                hs.ws.reset();
                hs.drafter.append_context(&self.model.stream, &mut hs.ws, &taps, 1, pos[j], self.model.tp.as_ref())?;
            }
            return Ok(toks);
        }
        if let (Some(j), Some(hs)) = (owner, self.spec.as_mut()) {
            let st: &DecodeStep = if m == 1 { &self.dec } else { &self.lsteps[m - 2] };
            let h = self.model.cfg.hidden_size;
            let taps: Vec<(u64, usize)> = st.tap_view(&self.model).iter().map(|(a, p)| (a + (j * h * 4) as u64, *p)).collect();
            hs.ws.reset();
            hs.drafter.append_context(&self.model.stream, &mut hs.ws, &taps, 1, pos[j], self.model.tp.as_ref())?;
        }
        Ok(toks)
    }

    /// G1: every lane keeps its own drafter context (`lane_ctx`).
    pub fn per_lane_ctx(&self) -> bool { !self.lane_ctx.is_empty() }

    fn ctx_swap(&mut self, j: usize) {
        if let Some(hs) = self.spec.as_mut() {
            for (l, (k, v)) in hs.drafter.layers.iter_mut().zip(self.lane_ctx[j].iter_mut()) {
                std::mem::swap(&mut l.kc, k);
                std::mem::swap(&mut l.vc, v);
            }
        }
    }

    /// G1: lane `j`'s context into the drafter (the live one parked first). No-op without per-lane contexts.
    pub fn ctx_make_live(&mut self, j: usize) -> Result<()> {
        if self.lane_ctx.is_empty() || self.ctx_live == Some(j) { return Ok(()); }
        // no sync: the kernels already queued keep the raw addresses they were launched with, every buffer stays
        // allocated, and the next append runs on the same stream after them
        if let Some(k) = self.ctx_live.take() { self.ctx_swap(k); }
        self.ctx_swap(j);
        self.ctx_live = Some(j);
        Ok(())
    }

    /// G1 compaction: lane `from`'s sampler state into lane `to` (a finished request's lane is refilled by the last one),
    /// and its drafter context with it.
    pub fn copy_lane(&mut self, from: usize, to: usize) -> Result<()> {
        if let Some(t) = &self.lane_tab { t.copy_lane(&self.model, from, to)?; self.model.sync()?; }
        if !self.lane_ctx.is_empty() {
            if let Some(k) = self.ctx_live.take() { self.ctx_swap(k); }
            self.lane_ctx.swap(from, to);
        }
        Ok(())
    }

    /// `prefill`, optionally WITHOUT feeding the drafter's context (G1: a request admitted next to the one that owns it).
    pub fn prefill_feed(&mut self, ids: &[u32], start: usize, want_logits: bool, feed_drafter: bool) -> Result<Option<u64>> {
        ensure!(!ids.is_empty(), "empty prefill");
        ensure!(start + ids.len() <= self.max_pos, "prompt of {} tokens at {start} exceeds the context ({})", ids.len(), self.max_pos);
        let v = self.model.vl; // the logits are this rank's vocabulary SLICE (vocab-parallel head)
        let dev = self.model.dev.clone();
        let mut out = None;
        let n = ids.len();
        let mut i = 0;
        while i < n {
            let m = (n - i).min(self.chunk);
            let last = i + m == n;
            self.model.sync()?;
            let ids_d = dev.htod_sync_copy(&ids[i..i + m].iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
            let pos_d = dev.htod_sync_copy(&(start + i..start + i + m).map(|x| x as i32).collect::<Vec<i32>>())?;
            let sel: Option<(Vec<usize>, u64)> = match (&self.spec, &self.pf_taps) { (Some(hs), Some(pt)) => Some((hs.drafter.cfg.taps.clone(), dptr(pt))), _ => None };
            let sel_ref: Option<(&[usize], u64)> = sel.as_ref().map(|(l, d)| (l.as_slice(), *d));
            let lg = self.model.step_kernels_sel(&mut self.pf_ws, dptr(&ids_d), dptr(&pos_d), m, want_logits && last, None, sel_ref)?;
            self.model.sync()?; // the chunk's ids/pos buffers die here
            if let Some(tp) = &self.model.tp { tp.check_link("after a prefill chunk")?; }
            if let (true, Some(hs), Some(pt)) = (feed_drafter, self.spec.as_mut(), &self.pf_taps) {
                // the drafter's context for the prompt rows (head only)
                let h = self.model.cfg.hidden_size;
                let taps: Vec<(u64, usize)> = (0..hs.drafter.cfg.taps.len()).map(|j| (dptr(pt) + (j * m * h * 4) as u64, h)).collect();
                hs.ws.reset();
                hs.drafter.append_context(&self.model.stream, &mut hs.ws, &taps, m, start + i, self.model.tp.as_ref())?;
                self.model.sync()?;
            }
            // a 1-row logits buffer for a wide chunk, the last of m rows for a small one
            if let Some(p) = lg { out = Some(if m > MAX_ROWS { p } else { p + ((m - 1) * v * 2) as u64 }); }
            i += m;
        }
        Ok(out)
    }

    /// One graphed decode step: token `t` at `pos` -> (the next sampled token, the head's cancel flag),
    /// identical on every rank.
    pub fn decode_step(&mut self, t: u32, pos: usize) -> Result<(u32, bool)> {
        ensure!(pos < self.max_pos, "position {pos} beyond the context ({})", self.max_pos);
        self.dec.step(&self.model, &[t], &[pos])
    }
}
