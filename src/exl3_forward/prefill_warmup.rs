//! `prefill_prompt`, the warmup family, `lt_prewarm_start` (its thread must never destroy a library handle:
//! a source-text test pins the function body together with the item that follows it), `precapture_decode_graphs`
//! and `prefill_argmax`. See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// FP4L-SPEED1: start reading the n-gram (PLE) rows of prompt positions [from, to) into the page cache in the
    /// background (the serve loop calls it with the NEXT chunk before running the current one). Advisory: changes
    /// no output; no-op without a PLE table or with the table in RAM.
    pub fn ple_warm_ahead(&self, prompt: &[u32], from: usize, to: usize) {
        if let Some(ple) = self.ple.as_ref() { ple.warm_ahead(prompt, from, to); }
    }

    /// S-A3-t: prompt prefill for the MTP harnesses. None = token-by-token forward_step over the
    /// whole prompt; Some(C) = the served posture — chunked sweeps over prompt[..n-1], then the
    /// last token through forward_step (its argmax and tap feed round 1). sc.argmax holds the
    /// first generated token afterwards in both postures.
    pub fn prefill_prompt(&self, sc: &mut Scratch, prompt: &[u32], chunk: Option<usize>,
                          psc: &mut Option<PrefillScratch>) -> Result<()> {
        let n = prompt.len();
        anyhow::ensure!(n > 0, "empty prompt");
        match chunk.filter(|&c| c > 0) {
            Some(cw) => {
                if psc.as_ref().map_or(true, |p| p.c < cw) { *psc = Some(self.prefill_scratch(cw)?); }
                let ps = psc.as_mut().unwrap();
                let mut pos = 0usize;
                while pos < n - 1 {
                    let end = (pos + cw).min(n - 1);
                    let toks: Vec<i32> = prompt[pos..end].iter().map(|&t| t as i32).collect();
                    self.prefill_chunk(ps, &toks, pos, 0, false, None)?;
                    pos = end;
                }
                self.forward_step(sc, &[prompt[n - 1] as i32], &[n - 1], &[0], None)?;
            }
            None => {
                for (i, t) in prompt.iter().enumerate() {
                    self.forward_step(sc, &[*t as i32], &[i], &[0], None)?;
                }
            }
        }
        Ok(())
    }

    /// S-A3-u: startup warm-up — dummy prefill chunks at the given widths on `slot`, then
    /// `reset_slot`. cuBLASLt loads its kernels lazily and plans heuristics per (k, n, m) on first
    /// use; on a fresh process that was ~0.42 s of the FIRST prefill (0.33 + 0.09 s gaps before
    /// the first nvjet launches). Standard serve practice: pay it at startup, not on request 1.
    /// State-neutral: every per-slot plane is restored by reset_slot, and K/V + indexer keys are
    /// rewritten before any later read of those positions.
    pub fn prefill_warmup(&self, psc: &mut PrefillScratch, slot: usize, widths: &[usize]) -> Result<()> {
        let t0 = std::time::Instant::now();
        // TP-4S2: where SP is live (world 2 or 4), RAGGED split chunks join the list (see sp_warmup_extra).
        // The decision reads only state every rank agreed at boot (flag, transport, world), never a per-rank
        // probe, so every rank runs the identical list and the exchanges stay SPMD.
        let mut all: Vec<usize> = widths.to_vec();
        if self.sp_warmup_live() { all.extend(Self::sp_warmup_extra(widths)); }
        let mut per_ms: Vec<u32> = Vec::with_capacity(all.len());
        for (i, &w) in all.iter().enumerate() {
            let tw = std::time::Instant::now();
            let w = w.min(psc.c).max(1);
            // PFX1 (e): the full-width chunk records the prefill's cuBLASLt shape families
            let rec = i == 0 && w == psc.c && pfx1_on(PFX1_LTPLANS);
            if rec {
                LT_REC.lock().unwrap().clear();
                LT_REC_ON.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            let toks = vec![1i32; w];
            let r = self.prefill_chunk(psc, &toks, 0, slot, false, None);
            if rec {
                LT_REC_ON.store(false, std::sync::atomic::Ordering::Relaxed);
                let recs = std::mem::take(&mut *LT_REC.lock().unwrap());
                let fams: Vec<(usize, usize, bool, usize, usize)> = recs.iter()
                    .filter(|r| r.2 % w == 0)
                    .map(|&(k, n, m, ta, ws)| (k, n, ta, m / w, ws))
                    .collect();
                *LT_FAMILIES.lock().unwrap() = Some((fams, w));
            }
            r?;
            self.reset_slot(slot)?;
            self.dsync()?;
            per_ms.push(tw.elapsed().as_millis().min(u32::MAX as u128) as u32);
        }
        self.dsync()?;
        crate::dprintln!("EXL3 prefill warm-up: widths {:?} in {:.2} s (cuBLASLt kernels + plans loaded before the first request; \
                  per-chunk ms {:?})", all, t0.elapsed().as_secs_f32(), per_ms);
        Ok(())
    }

    /// TP-4S2: the sequence-parallel split is live for this model (world 2 or 4, --tp-seq-parallel on, the
    /// pipelined transport up, not composed with the overlap fold) — the conditions of `sp_rows_for` that are
    /// agreed across ranks at boot. Deliberately NOT the per-width pinned-plan probe (rank-local).
    fn sp_warmup_live(&self) -> bool {
        self.tp.as_ref().map_or(false, |tp| xtp::seq_parallel_on() && tp.sp_ready() && !xtp::pf_overlap_fold())
    }

    /// TP-4S2: extra warm-up widths that make the first SERVED split chunk's one-time costs land at boot.
    /// The base list (full width, 512, 160, ...) is all multiples of 4, so every split chunk it runs has
    /// EQUAL row blocks and a chunk width that is a clean multiple of the quarter. A served chunk is the
    /// prompt length minus the seam token: RAGGED blocks (c = 2018 splits 505/505/505/503), with its own
    /// c-keyed state (cuBLASLt reference + pinned hc plans at quarter m, exchange programs, kernel picks by
    /// M class) that the clean widths never touch. Four widths, none a multiple of 4, each in the split
    /// range [SP_MIN_ROWS, min(C, SP_MAX_ROWS)] and not already in the list: C - 30 (the ~2K-token request
    /// class, c = 2018 at C = 2048), about C/2, about C/7 + 8, and SP_MIN_ROWS + 3. Residues mod 4 are mixed
    /// (2, 1, 3, 3) so world 4 sees ragged quarters and world 2 sees odd halves. Pure: depends only on the
    /// base list (C = its largest entry). Costs about one extra 2K chunk of boot (~1 s at C = 2048).
    pub fn sp_warmup_extra(base: &[usize]) -> Vec<usize> {
        let c = base.iter().copied().max().unwrap_or(0);
        // the largest w' <= w with w' % 4 == r
        let ragged = |w: usize, r: usize| w.saturating_sub((w + 4 - r) % 4);
        let (lo, hi) = (xtp::SP_MIN_ROWS, c.min(xtp::SP_MAX_ROWS));
        let mut out: Vec<usize> = Vec::new();
        for (w, r) in [(c.saturating_sub(30), 2usize), (c / 2, 1), (c / 7 + 8, 3), (xtp::SP_MIN_ROWS + 3, 3)] {
            let w = ragged(w, r);
            if (lo..=hi).contains(&w) && w % 4 != 0 && !base.contains(&w) && !out.contains(&w) { out.push(w); }
        }
        out
    }

    /// Warm-up widths for a chunk width C: the full chunk plus representative tails (cuBLASLt picks
    /// kernels by M range; tails of arbitrary length take other kernels than the full chunk).
    /// PFX1 (f): + width 16 — the c <= 16 tail class (split-K decode-arm chains, xq_gemm_f16
    /// hc/ab, and the head fc chains at 4c = 64 rows through cuBLASLt) loads before request 1.
    pub fn prefill_warmup_widths(c: usize) -> Vec<usize> {
        let mut v = vec![c];
        for w in [512usize, 160, 32] { if w < c { v.push(w); } }
        if pfx1_on(PFX1_W16) && 16 < c { v.push(16); }
        v
    }

    /// PFX1 (e): pre-populate the cuBLASLt plan cache for EVERY chunk row count 1..=C of every
    /// shape family the warm-up's full-width chunk recorded (m = mult * c: the m = c chains, the
    /// hc / PLE / a,b projections, and the head fc chains at m = 4c), wherever m >= the recon
    /// threshold — ascending c, so the short prompts (the TTFT-critical ones) are ready first.
    /// A first-seen prompt length paid +2.5..+10 ms of heuristics on its tail chunk (B3: 258.4
    /// vs 248.5 ms at m = 72). BITWISE: a plan is a pure function of (k, n, m, ta, workspace), so
    /// the pre-built plan is the one the lazy path would build (the lazy path stays as the
    /// fallback for anything not yet built). Runs on its own thread with its own cuBLASLt handle:
    /// host-side descriptor + heuristic calls only (no stream work, no device allocation); the
    /// plan-cache mutex is taken only to probe and insert, never across a heuristic. Returns the
    /// thread (the bench joins it; the serve detaches it and logs completion).
    pub fn lt_prewarm_start(&self) -> Option<std::thread::JoinHandle<()>> {
        if !pfx1_on(PFX1_LTPLANS) { return None; }
        let (fams, cmax) = LT_FAMILIES.lock().unwrap().take()?;
        let rmin = recon_min_rows();
        if fams.is_empty() || rmin == 0 { return None; }
        let dev = self.dev.clone();
        let nf = fams.len();
        let spawned = std::thread::Builder::new().name("pfx1-ltplans".into()).spawn(move || {
            let t0 = std::time::Instant::now();
            if let Err(e) = dev.bind_to_thread() {
                println!("PFX1: ltplans: bind_to_thread failed ({e:?}) — prefill plans stay lazy");
                return;
            }
            let handle = match lt::create_handle() {
                Ok(h) => h,
                Err(e) => { println!("PFX1: ltplans: cublasLtCreate failed ({e:?}) — prefill plans stay lazy"); return; }
            };
            let (mut built, mut had, mut lost) = (0usize, 0usize, 0usize);
            'rows: for c in 1..=cmax {
                for &(k, n, ta, mult, ws) in &fams {
                    let m = mult * c;
                    if m < rmin { continue; }
                    let key: LtKey = (k, n, m, ta as u8);
                    if lt_plans().lock().unwrap().contains_key(&key) { had += 1; continue; }
                    match lt_plan_build(handle, k, n, m, ta, ws) {
                        Ok(p) => {
                            let mut map = lt_plans().lock().unwrap();
                            if let std::collections::hash_map::Entry::Vacant(v) = map.entry(key) {
                                v.insert(p);
                                built += 1;
                            } else {
                                drop(map);
                                lt_plan_drop(p);
                                lost += 1;
                            }
                        }
                        Err(s) => {
                            crate::dprintln!("PFX1: ltplans stopped at c={c}: {s} — the remaining shapes stay lazy");
                            break 'rows;
                        }
                    }
                }
            }
            // S-B9-REL-VISIONFIX2 V2a: the handle is NOT destroyed. cublasLtDestroy is not refused, but it
            // invalidates a graph capture that the scheduler thread has open at that moment (measured,
            // every capture mode: tests/vision_capture_race_test.rs
            // `destroying_a_library_handle_invalidates_another_threads_capture`), and this thread
            // finishes at an arbitrary time after boot while the scheduler captures lazily. One handle
            // per boot stays allocated for the life of the process (host objects only).
            let _ = handle;
            crate::dprintln!("PFX1: cuBLASLt prefill plans pre-populated in {:.1} s (background): {built} built, \
                      {had} already cached, {lost} raced the prefill; {nf} shape families x chunk rows 1..={cmax} (m >= {rmin})",
                     t0.elapsed().as_secs_f32());
        });
        match spawned {
            Ok(h) => Some(h),
            Err(e) => { println!("PFX1: ltplans thread spawn failed ({e}) — prefill plans stay lazy"); None }
        }
    }

    /// WP04 (PLAN/SURPASS_PLAN_2026-09-26.md): capture EVERY decode graph the served config can
    /// reach at boot, so no request pays a lazy capture (the first request after a restart paid
    /// ~10 instantiations; an 11K run captured 21 graphs mid-generation at the old power-of-two
    /// scorer buckets). Per regime (dense; sparse when reachable and the scorer is bucket-free —
    /// otherwise its keys carry a bucket and stay lazy), for every lane slot: the decode step at
    /// m = 1..=lanes, verify m = 2..=k+1 (DDS; the fixed chain only verifies at k+1), and the draft
    /// passes 0..k (DDS) or the chain (fixed).
    ///
    /// CAPTURE-ONLY, NOTHING EXECUTES (WP04-v2). v1 ran one EAGER warm step + MTP round per regime
    /// on slot 0 first and "undid" it (planes re-zeroed, reset_slot); on hardware that shifted every
    /// later request's DDS trajectory (python drafted 1043 vs 1041, prose 1255 vs 1245 — hashes
    /// equal, trajectory gate FAILED), and the sparse warm round alone moved dense-only requests
    /// (dense-only vs dense+sparse warm differed although both leave identical planes) — the warm
    /// writes survive in the SCHEDULER SCRATCH, which the first request then reads where the
    /// baseline read fresh buffers. A captured graph only records its launches, so this version
    /// leaves every device byte (scratch, KV/QSA planes, recurrent state, PLE ring) exactly as a
    /// --graph-precapture off boot has it; the only one-time work the eager run used to trigger —
    /// the raw exl3_bench module the fused router launches through, the scorer's SM-count query —
    /// is done host-side first (a module load must not happen inside a capture). Each graph is
    /// then cuGraphUpload-ed (device-side instantiation, no kernel runs) so the first replay does
    /// not pay the upload either. DraftCal / ema_plain live in the scheduler and are not touched.
    /// Prints the graph count, seconds and the cuMemGetInfo delta; stops above 1 GB.
    /// The server's --graph-precapture off skips it (lazy capture as before).
    pub fn precapture_decode_graphs(&self, sc: &mut Scratch, lanes: usize, mtp_k: usize, dds: bool) -> Result<()> {
        if crate::opts::var(crate::opt!("exl3-no-graph")).is_ok()
            || crate::opts::var(crate::opt!("mtp-dump")).map_or(false, |s| !s.is_empty()) {
            crate::dprintln!("EXL3_GRAPH: precapture skipped (graphs off: --exl3-no-graph / --mtp-dump)");
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let free0 = cudarc::driver::result::mem_get_info().map(|x| x.0).unwrap_or(0);
        let n0 = self.graphs.lock().unwrap().len();
        PRECAP_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
        let r = self.precapture_inner(sc, lanes, mtp_k, dds, free0);
        PRECAP_ACTIVE.store(false, std::sync::atomic::Ordering::Relaxed);
        // the uploads were queued on the compute stream; no kernel was launched
        let synced = self.dsync().map_err(anyhow::Error::from);
        let free1 = cudarc::driver::result::mem_get_info().map(|x| x.0).unwrap_or(free0);
        let n = self.graphs.lock().unwrap().len().saturating_sub(n0);
        let mb = (free0 as f64 - free1 as f64) / 1e6;
        let up = PRECAP_UPLOAD_FAIL.load(std::sync::atomic::Ordering::Relaxed);
        let upn = if up > 0 { format!(" ({up} cuGraphUpload failed: their first replay uploads)") } else { String::new() };
        match &r {
            Ok(note) => println!("EXL3_GRAPH: precaptured {n} decode graphs at boot in {:.2} s (device mem {mb:+.0} MB){note}{upn}",
                                 t0.elapsed().as_secs_f64()),
            Err(e) => println!("EXL3_GRAPH: precapture FAILED after {n} graphs in {:.2} s (device mem {mb:+.0} MB): {e:#}",
                               t0.elapsed().as_secs_f64()),
        }
        r?;
        synced
    }

    fn precapture_inner(&self, sc: &mut Scratch, lanes: usize, mtp_k: usize, dds: bool, free0: usize) -> Result<String> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let lanes = lanes.clamp(1, self.width);
        let k = mtp_k.clamp(1, MTP_MAX_K);
        let head = if mtp_disabled_by_opt() { None } else { self.mtp.as_ref() };
        // the replay sites' own graph conditions (forward_step / mtp_verify / mtp_round*)
        // (p2 merge: + WP12's xcheck gate, which p1's forward_step / mtp_verify replay sites carry)
        let step_ok = crate::opts::var(crate::opt!("exl3-route-check")).is_err() && !wp12_xcheck();
        let verify_ok = gdn_ring_mode() != 2 && !wp12_xcheck();
        let draft_ok = tune::get_cur(&T_DRAFT_GRAPH) != 0
            && crate::opts::var(crate::opt!("hc-int8-check")).is_err()
            && !dhead::dhead_xcheck(); // (p2 merge) DHEAD's XCHECK turns the draft graphs off at the replay sites
        let bf = self.qsa_bucket_free();
        if [crate::opt!("exl3-score-xcheck"), crate::opt!("exl3-topk-xcheck"), crate::opt!("exl3-gather-xcheck")].iter().any(|v| wp14_xcheck(*v)) {
            crate::dprintln!("EXL3_GRAPH: note — WP04/WP14 XCHECK modes check EAGER calls only; graphed replays are \
                      not checked (run with --exl3-no-graph=1 for full coverage)");
        }
        // Host-side one-time init the eager first launch would otherwise do INSIDE a capture: the
        // fused router's raw-module load (cuModuleLoadDataEx) and the scorer's SM-count query.
        let _ = xq_raw_fn("xq_router_topk_route");
        // (p2 merge, WP04 x WP10/WP20: the decode-path raw entries WP04's base did not have —
        // WP10's routing fold and WP20's epilogues — resolved here too, so no cuModuleGetFunction
        // (and no lazy kernel load) happens inside a capture.)
        for f in ["xq_router_fold", "xq_router_fold_m0", "xq_router_fold_c", "xq_router_fold_c_ep", 
                  "xq_ks_combine_f32_k1l", "xq_moe_combine_ep_k1l", "xq_cvt_f16_f32_k1l", "xq_tp_wait_add_dec", "xq_tp_wait_add_dec_single", "xq_argmax_rows_vp", "xq_tp_wait_keys", "xq_vp_gather_k1", "xq_vp_gather_k2", "xq_vp_gather4_k1", "xq_vp_gather4_k2", "xq_dh_screen_k1", "xq_moe_gu_epi", "xq_moe_gu_epi_sh",
                  WP20_GU_FOLD_FN, "xq_moe_dn_epi_k8", "xq_moe_dn_epi_sh_k8", "xq_moe_dn_epi_k10", "xq_moe_dn_epi_sh_k10",
                  "xq_moe_gu_epi_c2", "xq_moe_gu_epi_c4", "xq_moe_gu_epi_b4", "xq_moe_gu_epi_b4_c2", "xq_moe_gu_epi_b4_c4",
                  "xq_moe_dn_epi_b4_k10"] {
            let _ = xq_raw_fn(f);
        }
        let _ = qsa_mr_ctas();
        // (sparse, a position of that regime): the key only needs the regime (the bucket is 0 when
        // the scorer is bucket-free) and qsa_grid_nblk only feeds the bucketed scorers.
        let mut regimes: Vec<(bool, usize)> = vec![(false, 0)];
        let mut note = format!(" [capture-only, lanes {lanes}, k {k}, {}, regimes: dense",
                               if dds { "DDS passes" } else { "fixed chain" });
        if self.cfg.has_indexer() && !qsa_disabled() {
            let ps = self.qsa_limit() + 1; // the first sparse position
            if !bf {
                note += ", sparse LAZY (bucketed scorer: --exl3-score-mr=0 or no pooled planes)";
            } else if ps < self.max_pos && self.qsa_live(ps) {
                regimes.push((true, ps));
                note += ", sparse (bucket-free)";
            } else {
                note += ", sparse unreachable at this window";
            }
        }
        note += "]";
        let over = || cudarc::driver::result::mem_get_info()
            .map(|(f, _)| free0.saturating_sub(f) > (1usize << 30)).unwrap_or(false);
        const CAP: &str = " [stopped at the 1 GB precapture cap]";
        for &(sparse, p) in &regimes {
            let (qg, qb) = self.qsa_grid_bucket_key(p);
            if step_ok {
                // decode step: m = 1 (seam steps, control rounds) up to every lane (batched plain lanes)
                for m in 1..=lanes {
                    let key = gkey_of(m, sparse, qb) | sk(sc) | self.vp_key(sc, m);
                    if self.graphs.lock().unwrap().contains_key(&key) { continue; }
                    if over() { note += CAP; return Ok(note); }
                    sc.qsa_grid_nblk = qg;
                    self.capture_step_graph(&l, sc, m, key, sparse, 0); // mixed-regime steps capture lazily, per (m, md) seen
                }
            }
            let Some(h) = head else { continue };
            for s in 0..lanes {
                if verify_ok {
                    let widths: Vec<usize> = if dds { (2..=k + 1).collect() } else { vec![k + 1] };
                    for m in widths {
                        let key = vkey_of(s, m, sparse, qb) | sk(sc) | self.vp_key(sc, m);
                        if self.graphs.lock().unwrap().contains_key(&key) { continue; }
                        if over() { note += CAP; return Ok(note); }
                        sc.qsa_grid_nblk = qg;
                        self.capture_verify_graph(&l, sc, s, m, m - 1, sparse, key);
                    }
                }
                if draft_ok {
                    if dds {
                        for i in 0..k {
                            let key = pkey_of(s, i, sparse, qb) | sk(sc);
                            if self.graphs.lock().unwrap().contains_key(&key) { continue; }
                            if over() { note += CAP; return Ok(note); }
                            sc.qsa_grid_nblk = qg;
                            self.capture_pass_graph(&l, sc, h, i, sparse, key);
                        }
                    } else {
                        let key = dkey_of(s, k, sparse, qb) | sk(sc);
                        if !self.graphs.lock().unwrap().contains_key(&key) {
                            if over() { note += CAP; return Ok(note); }
                            sc.qsa_grid_nblk = qg;
                            self.capture_draft_graph(&l, sc, h, k, sparse, key);
                        }
                    }
                }
            }
        }
        Ok(note)
    }

    pub fn prefill_argmax(&self, psc: &PrefillScratch) -> Result<u32> {
        let ids = self.dtoh(&psc.argmax)?;
        Ok(ids[0] as u32)
    }

}
