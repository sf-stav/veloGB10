//! The MTP draft chain: `draft_block`, `draft_pass`, the draft graph captures, `mtp_cover`, `mtp_draft_fixed` and
//! `mtp_draft_adaptive`. The head KV is single-slot (AGENTS 2.5). See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// S-A3-p: the k-pass draft chain as launches only (graph-capturable): per pass, stage
    /// {tok (pass 0), pos = p + i, slot row} from sc.draft_meta on device, run the head, and
    /// route its argmax to d_dev[i] and next pass's toks[0]. Same kernels, same inputs as the
    /// eager per-pass head_forward (whose host uploads forced a NULL-stream drain per pass).
    pub(super) fn draft_block(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, k: usize, qsa: bool) -> Result<()> {
        let _tc = tune::enter(tune::Ctx::fam(tune::Fam::DraftChain, 1).regime(qsa)); // TUNE family scope
        for i in 0..k {
            self.draft_pass(l, sc, head, i, qsa)?;
        }
        Ok(())
    }

    /// One draft pass i (launches only, graph-capturable): stage {tok, pos, slot}, run the
    /// head at m=1, keep the argmax on device (d_dev[i], next pass's toks[0]) and record the
    /// pass confidence (max draft logit) into dconf[i] for the dynamic-draft stop (DDS).
    fn draft_pass(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, i: usize, qsa: bool) -> Result<()> {
        // TUNE family scope: a pass inside draft_block keeps the chain's family
        let _tc = (tune::current().fam != tune::Fam::DraftChain)
            .then(|| tune::enter(tune::Ctx::fam(tune::Fam::DraftPass, 1).regime(qsa)));
        xqlaunch!(l, "xq_draft_setup", (1, 1, 1), (32, 1, 1), 0,
                  (&mut sc.toks, &mut sc.pos, &mut sc.slots, &sc.draft_meta, i as i32))?;
        // DHEAD (w2): the head body, then the screen + exact-rescore tail (writes argmax, dconf[i],
        // d_dev[i], toks[0] itself). DHEADP (w3): a penalized draft (WP15 mirror) with monotone
        // params takes the same screen + a bound-expanded, penalized rescore; a non-monotone one
        // keeps the old tail (full draft logits + xq_pen_draft). The choice is part of the graph
        // key (pen_draft_key). --dhead-off=1 = the old tail always; --dheadp-off=1 = the
        // old tail for penalized drafts.
        // WP24: a real-q pass (rq_mode 2) takes the old tail (draft_tail) unless A5-L4's
        // spec.ratio_dhead routes it to DHEAD (DheadRq); the dump rung (rq_mode 1) keeps the served
        // tail and adds the list (rq_dump_list / head_launches).
        if let (Some(dh), Some(qd)) = (self.dhead.as_ref(), self.lm_head_draft.as_ref()) {
            match self.draft_tail(sc) {
                DraftTail::Dhead => {
                    self.head_launches(l, sc, head, 1, qsa, false)?;
                    self.dhead_pass(l, sc, dh, qd, i)?;
                    return self.rq_dump_list(l, sc, qd, i);
                }
                DraftTail::DheadP => {
                    self.head_launches(l, sc, head, 1, qsa, false)?;
                    self.dheadp_pass(l, sc, dh, qd, i)?;
                    return self.rq_dump_list(l, sc, qd, i);
                }
                DraftTail::DheadRq => {
                    self.head_launches(l, sc, head, 1, qsa, false)?;
                    return self.dhead_rq_pass(l, sc, dh, qd, i);
                }
                DraftTail::Old => {}
            }
        }
        sc.pen_draft_i = if pen_draft_on() { Some(i) } else { None };
        sc.rq_draft_i = if sc.rq_mode > 0 { Some(i) } else { None }; // WP24
        let r = self.head_launches(l, sc, head, 1, qsa, true);
        sc.pen_draft_i = None;
        sc.rq_draft_i = None;
        r?;
        let v = self.lm_head_draft.as_ref().map(|q| q.n as usize).unwrap_or(self.lm_head.n as usize);
        xqlaunch!(l, "xq_rowmax_f16", (1, 1, 1), (1024, 1, 1), 0,
                  (&mut sc.dconf, i as i64, &sc.logits, 0i64, v as i32))?;
        xqlaunch!(l, "xq_copy1_i32", (1, 1, 1), (1, 1, 1), 0,
                  (&mut sc.d_dev, i as i64, &sc.argmax))?;
        xqlaunch!(l, "xq_copy1_i32", (1, 1, 1), (1, 1, 1), 0,
                  (&mut sc.toks, 0i64, &sc.argmax))?;
        Ok(())
    }

    /// Record draft pass i into a CUDA graph (after an eager run of the same pass, or capture-only
    /// at boot — recording executes nothing).
    pub(super) fn capture_pass_graph(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, i: usize, qsa: bool, key: usize) {
        use cudarc::driver::sys;
        let raw = self.stream.stream;
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(raw, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            crate::metrics::exl3_frac_capture_reset(); // K35: this capture's fractional-expert round tally starts at 0
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::rprintln!("EXL3_GRAPH: draft-pass BeginCapture failed ({r:?}) at i={i} — staying eager");
                crate::metrics::graph_capture_lost("draft-pass", &format!("i={i}, qsa={qsa}"), &format!("{r:?}"));
                return;
            }
        }
        let cap = self.draft_pass(l, sc, head, i, qsa);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r2 = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
        if r2 != sys::CUresult::CUDA_SUCCESS || cap.is_err() {
            crate::rprintln!("EXL3_GRAPH: draft-pass capture failed (end={r2:?}, body={:?}) at i={i} — staying eager", cap.err());
            crate::metrics::graph_capture_lost("draft-pass", &format!("i={i}, qsa={qsa}"), &capture_lost_status(r2));
            return;
        }
        crate::exl3_wp27::finish(graph, "draft-pass", i); // WP27: node priorities + DAG receipt (m = pass index)
        if !self.tp_spine_audit(graph, "draft-pass", i) { unsafe { sys::cuGraphDestroy(graph); } return; } // TP-C: barriers on the spine
        unsafe {
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            let r3 = crate::exl3_wp27::instantiate(&mut exec, graph); // WP27 (= cuGraphInstantiate_v2 unless --wp27-prio=1)
            sys::cuGraphDestroy(graph);
            if r3 == sys::CUresult::CUDA_SUCCESS {
                precap_upload(exec, raw);
                crate::metrics::set_graph_entries({ let mut g = self.graphs.lock().unwrap(); g.insert(key, crate::gpu::CudaGraph::from_exec(exec, raw).with_frac_rounds(crate::metrics::exl3_frac_capture_take())); g.len() as u64 });
            } else {
                crate::rprintln!("EXL3_GRAPH: draft-pass instantiate failed ({r3:?}) at i={i} — staying eager");
                crate::metrics::graph_capture_lost("draft-pass", &format!("i={i}, qsa={qsa}"), &format!("instantiate {r3:?}"));
            }
        }
    }

    pub(super) fn capture_draft_graph(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, k: usize, qsa: bool, dkey: usize) {
        use cudarc::driver::sys;
        let raw = self.stream.stream;
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(raw, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            crate::metrics::exl3_frac_capture_reset(); // K35: this capture's fractional-expert round tally starts at 0
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::rprintln!("EXL3_GRAPH: draft BeginCapture failed ({r:?}) at k={k} — staying eager");
                crate::metrics::graph_capture_lost("draft-chain", &format!("k={k}, qsa={qsa}"), &format!("{r:?}"));
                return;
            }
        }
        let cap = self.draft_block(l, sc, head, k, qsa);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r2 = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
        if r2 != sys::CUresult::CUDA_SUCCESS || cap.is_err() {
            crate::rprintln!("EXL3_GRAPH: draft capture failed (end={r2:?}, body={:?}) at k={k} — staying eager", cap.err());
            crate::metrics::graph_capture_lost("draft-chain", &format!("k={k}, qsa={qsa}"), &capture_lost_status(r2));
            return;
        }
        crate::exl3_wp27::finish(graph, "draft-chain", k); // WP27: node priorities + DAG receipt (m = k)
        if !self.tp_spine_audit(graph, "draft-chain", k) { unsafe { sys::cuGraphDestroy(graph); } return; } // TP-C: barriers on the spine
        unsafe {
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            let r3 = crate::exl3_wp27::instantiate(&mut exec, graph); // WP27 (= cuGraphInstantiate_v2 unless --wp27-prio=1)
            sys::cuGraphDestroy(graph);
            if r3 == sys::CUresult::CUDA_SUCCESS {
                precap_upload(exec, raw);
                crate::metrics::set_graph_entries({ let mut g = self.graphs.lock().unwrap(); g.insert(dkey, crate::gpu::CudaGraph::from_exec(exec, raw).with_frac_rounds(crate::metrics::exl3_frac_capture_take())); g.len() as u64 });
                if graph_log_on() { crate::rprintln!("EXL3_GRAPH: captured draft chain at k={k}"); }
            } else {
                crate::rprintln!("EXL3_GRAPH: draft instantiate failed ({r3:?}) at k={k} — staying eager");
                crate::metrics::graph_capture_lost("draft-chain", &format!("k={k}, qsa={qsa}"), &format!("instantiate {r3:?}"));
            }
        }
    }

    /// S-A3-x (--mtp-cover): TEACHER-FORCED draft coverage at a round start. Restores the same
    /// tap `mtp_round` would, then runs `targets.len()` head passes feeding the TRUE greedy tokens
    /// (not the head's argmax) and returns, per level, the rank of the true next token in the
    /// head's draft logits (0 = top-1; usize::MAX = outside the pruned draft slice). This is the
    /// head's input along the one branch a draft tree must contain — each shape's acceptance
    /// economics. Head KV rows p.. are rewritten by the real round before it reads them.
    pub fn mtp_cover(&self, sc: &mut Scratch, head: &DraftHead, b: i32, p: usize, tap_row: usize,
                     targets: &[i32]) -> Result<Vec<usize>> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let tb = self.tap_base(sc, 0)?; // the cover probe drafts slot 0 (draft_meta below)
        if tap_row != usize::MAX {
            xqlaunch!(l, "xq_copy_f32", ((((he as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut sc.resid, &sc.taps_keep, he as i64, 0i64, (tb + tap_row * he) as i64))?;
        }
        self.dev.htod_copy_into(vec![b, p as i32, 0, 0], &mut sc.draft_meta)?;
        let v = self.lm_head_draft.as_ref().map(|q| q.n as usize).unwrap_or(self.lm_head.n as usize);
        let mut ranks = Vec::with_capacity(targets.len());
        for (i, &g) in targets.iter().enumerate() {
            xqlaunch!(l, "xq_draft_setup", (1, 1, 1), (32, 1, 1), 0,
                      (&mut sc.toks, &mut sc.pos, &mut sc.slots, &sc.draft_meta, i as i32))?;
            sc.qsa_grid_nblk = self.qsa_grid_bucket(p + i).0;
            self.head_launches(&l, sc, head, 1, self.qsa_live(p + i), true)?;
            let lg = self.dtoh(&sc.logits)?;
            let row = &lg[..v];
            ranks.push(if g < 0 || g as usize >= v { usize::MAX } else {
                let gv = half::f16::from_bits(row[g as usize]).to_f32();
                // argmax tie-break = lowest index: equal logits at lower ids rank ahead
                row.iter().enumerate().filter(|(j, x)| {
                    let xv = half::f16::from_bits(**x).to_f32();
                    xv > gv || (xv == gv && *j < g as usize)
                }).count()
            });
            let mut t = vec![0i32; sc.toks.len()];
            t[0] = g;
            self.dev.htod_copy_into(t, &mut sc.toks)?;
        }
        // S-A3-z: the REAL chain (head fed its own argmax, exactly as served) — per level, the draft
        // head's top-1 softmax probability over the draft logits. Appended to `ranks` as
        // round(p * 1e6) so the CSV carries both (ranks first, then confidences).
        if tap_row != usize::MAX {
            xqlaunch!(l, "xq_copy_f32", ((((he as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut sc.resid, &sc.taps_keep, he as i64, 0i64, (tb + tap_row * he) as i64))?;
        }
        self.dev.htod_copy_into(vec![b, p as i32, 0, 0], &mut sc.draft_meta)?;
        for i in 0..targets.len() {
            xqlaunch!(l, "xq_draft_setup", (1, 1, 1), (32, 1, 1), 0,
                      (&mut sc.toks, &mut sc.pos, &mut sc.slots, &sc.draft_meta, i as i32))?;
            sc.qsa_grid_nblk = self.qsa_grid_bucket(p + i).0;
            self.head_launches(&l, sc, head, 1, self.qsa_live(p + i), true)?;
            let lg = self.dtoh(&sc.logits)?;
            let row: Vec<f32> = lg[..v].iter().map(|&x| half::f16::from_bits(x).to_f32()).collect();
            let (mut am, mut mx) = (0usize, f32::NEG_INFINITY);
            for (j, &x) in row.iter().enumerate() { if x > mx { mx = x; am = j; } }
            let z: f64 = row.iter().map(|&x| ((x - mx) as f64).exp()).sum();
            ranks.push(((1.0 / z) * 1e6).round() as usize);
            let mut t = vec![0i32; sc.toks.len()];
            t[0] = am as i32;
            self.dev.htod_copy_into(t, &mut sc.toks)?;
        }
        Ok(ranks)
    }

    /// The DRAFT half of the fixed-depth `mtp_round` (S-B9-CF-PACK4S: split out unchanged so the
    /// lone round and the packed arm's `--draft-confidence 0` chain run one copy): tap restore,
    /// the k-pass chain (graphed per (k, slot)), the drafts readback and clamp. Returns the k
    /// drafts and the round's three phase stamps for `mtp_verify`.
    pub(super) fn mtp_draft_fixed(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, k: usize,
                       b: i32, p: usize, tap_row: usize) -> Result<(Vec<i32>, [std::time::Instant; 3])> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let hcn = self.cfg.hc_count.max(1);
        let m = k + 1;
        let he = hcn * self.cfg.hidden_size;
        let prof = round_prof_on();
        let tp0 = std::time::Instant::now();
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);

        // WP02: draft rows p..p+k-1 and verify rows p..p+k must stay inside the KV window
        // (the serve clamps k at the context end; this refuses anything that slipped past).
        anyhow::ensure!(k >= 1 && p + k < self.max_pos,
                        "mtp_round: window p={p} k={k} past the KV window (max_pos {})", self.max_pos);
        // ---- tap restore: the saved verify tap that predicted b. usize::MAX =
        // control-round marker: sc.resid already holds the true tap, no restore.
        if tap_row != usize::MAX {
            let tb = self.tap_base(sc, slot)?;
            xqlaunch!(l, "xq_copy_f32", grid(he as i64), (256, 1, 1), 0,
                      (&mut sc.resid, &sc.taps_keep, he as i64, 0i64, (tb + tap_row * he) as i64))?;
        }

        // ---- draft chain (device-resident; argmax D2D between steps). S-A3-p: ONE upload of
        // {b, p, slot} per round, then the k passes run as launches only — graphed per
        // (k, slot) after the first eager round (--exl3-draft-graph=0 = eager A/B).
        // [3] = the WP24 draft RNG counter base (0 unless real-q: only xq_rq_draft reads it).
        self.dev.htod_copy_into(vec![b, p as i32, slot as i32, sc.rq_dctr], &mut sc.draft_meta)?;
        // head QSA: regime + grid bucket of the chain's last pass; the selection kernels read
        // the deinterleaved slot table (xq_draft_setup stages only toks/pos/slots).
        let dqsa = self.qsa_live(p + k - 1);
        let (dg, db) = self.qsa_grid_bucket_key(p + k - 1);
        sc.qsa_grid_nblk = dg;
        // ISSUES 128: staged in EVERY regime — the head's raw-key / pooled-plane writes address slot_ids[0] in the dense
        // regime too; staging it only for a sparse last pass let a dense round write into the previous op's slot (another lane).
        self.dev.htod_copy_into(vec![slot as i32; sc.slot_ids.len()], &mut sc.slot_ids)?;
        let dkey = dkey_of(slot, k, dqsa, db) | self.pen_draft_key(sc) | sk(sc); // WP15 mirror + DHEADP bits + TP-C scratch id
        let want_dgraph = crate::opts::var(crate::opt!("exl3-no-graph")).is_err()
            && tune::get_cur(&T_DRAFT_GRAPH) != 0
            && crate::opts::var(crate::opt!("mtp-dump")).map_or(true, |s| s.is_empty())
            && crate::opts::var(crate::opt!("hc-int8-check")).is_err()
            && !dhead::dhead_xcheck(); // DHEAD XCHECK compares on eager passes only
        let dgraphed = want_dgraph && self.graphs.lock().unwrap().contains_key(&dkey);
        // TUNE T1 (§5.2): the --autotune harness's shadow replays of the draft unit run BEFORE the
        // canonical chain (canonical-last); never armed in serving.
        let sr_at = autotune::SrAt { slot, k, m: k + 1, p, tap_row, qsa: dqsa, bucket: db };
        if autotune::armed() { autotune::sr_before_draft(self, sc, head, &sr_at)?; }
        if dgraphed {
            self.graphs.lock().unwrap().get(&dkey).unwrap().launch();
        } else {
            self.draft_block(&l, sc, head, k, dqsa)?;
        }
        if autotune::armed() { autotune::sr_after_draft(self, sc, head, &sr_at)?; }

        // ---- drafts readback (iteration 1: host PLE hash + verify upload)
        let tp1 = std::time::Instant::now();
        self.dsync()?;
        let tp2 = std::time::Instant::now();
        let d_buf = self.dtoh(&sc.d_dev)?;
        if want_dgraph && !dgraphed {
            self.capture_draft_graph(&l, sc, head, k, dqsa, dkey);
        }
        let mut drafts: Vec<i32> = d_buf[..k].to_vec();
        // A draft is a GUESS: the full-head verify decides every emitted token, so a bad draft
        // must cost a rejection, never the request (the old `d <= 0` bail killed a live request
        // on draft id 0 — a real token, and the argmax of a non-finite row). Out-of-range ids
        // are clamped to 0 (verify rejects them unless 0 really is the target) and counted.
        let v = self.cfg.vocab_size as i32;
        for d in drafts.iter_mut() {
            if *d < 0 || *d >= v {
                static BAD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let n = BAD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 8 || n.is_power_of_two() {
                    crate::deprintln!("[exl3-mtp] draft id {} out of range (vocab {v}) — clamped, verify rejects (#{})", *d, n + 1);
                }
                *d = 0;
            }
        }

        Ok((drafts, [tp0, tp1, tp2]))
    }

    /// The DRAFT half of `mtp_round_adaptive` (S-B9-CF-PACK4: split out unchanged so the lone
    /// round and the packed arm run one copy): tap restore, the DDS pass loop with its pinned
    /// stage and speculative next pass, and the clamped drafts. Leaves the head's rows p..p+w-1
    /// written from its own chain and `sc.d_dev[..w]` on the device.
    pub(super) fn mtp_draft_adaptive(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, kmax: usize,
                          b: i32, p: usize, tap_row: usize,
                          keep: &mut dyn FnMut(usize, f32) -> bool) -> Result<DraftHalf> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let tp0 = std::time::Instant::now();
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        // WP02: draft rows p..p+kmax-1 and verify rows p..p+w (w <= kmax) inside the KV window
        anyhow::ensure!(kmax >= 1 && p + kmax < self.max_pos,
                        "mtp_round_adaptive: window p={p} k={kmax} past the KV window (max_pos {})", self.max_pos);
        if tap_row != usize::MAX {
            let tb = self.tap_base(sc, slot)?; // WP03: this lane's window
            xqlaunch!(l, "xq_copy_f32", grid(he as i64), (256, 1, 1), 0,
                      (&mut sc.resid, &sc.taps_keep, he as i64, 0i64, (tb + tap_row * he) as i64))?;
        }
        // WP05: `stage` = the pinned path — {b, p, slot} and the slot table go up from the pinned
        // stage, each pass enqueues its {dconf_i, d_i} readback BEFORE the one per-pass
        // synchronize (the old path synced, then dtoh_sync_copy'd dconf: a second sync plus a
        // pageable allocation, and the drafts came back in a third readback after the loop),
        // and PLE row i (token b, then d_{i-1}) is built on the host while pass i runs.
        let stage = wp05_on("dds") && wp05_stage_fits(sc, kmax + 1);
        let hp = { let h = self.ple_hist.lock().unwrap(); (h[slot * 2], h[slot * 2 + 1]) };
        let mut tv_run: Vec<i32> = Vec::with_capacity(kmax + 1);
        tv_run.push(b);
        let (d_meta, d_sids) = (*sc.draft_meta.device_ptr() as u64, *sc.slot_ids.device_ptr() as u64);
        let (d_conf, d_d) = (*sc.dconf.device_ptr() as u64, *sc.d_dev.device_ptr() as u64);
        let dctr = sc.rq_dctr; // WP24: draft RNG counter base -> draft_meta[3] (0 unless real-q)
        if stage {
            let st = sc.wp05.as_mut().unwrap();
            if st.inflight { self.dsync()?; st.inflight = false; }
            st.ple_pre = None; // the ple region is about to be rewritten row by row
            let om = st.off_meta();
            st.pin.i32s(om, 4).copy_from_slice(&[b, p as i32, slot as i32, dctr]);
            st.pin.htod(om, d_meta, 16)?;
            st.inflight = true;
        } else {
        self.dev.htod_copy_into(vec![b, p as i32, slot as i32, dctr], &mut sc.draft_meta)?;
        }
        let want_graph = crate::opts::var(crate::opt!("exl3-no-graph")).is_err()
            && tune::get_cur(&T_DRAFT_GRAPH) != 0
            && crate::opts::var(crate::opt!("mtp-dump")).map_or(true, |s| s.is_empty())
            && crate::opts::var(crate::opt!("hc-int8-check")).is_err()
            && !dhead::dhead_xcheck(); // DHEAD XCHECK compares on eager passes only
        let mut confs: Vec<f32> = Vec::with_capacity(kmax);
        let mut w = kmax;
        // ISSUES 128: staged in EVERY regime (was: only when the chain's last pass is sparse) — the head's raw-key /
        // pooled-plane writes address slot_ids[0] in the dense regime too, so a dense round of a multi-lane server wrote
        // the head's QSA rows into the previous op's slot (another lane's): its drafts degraded once it went sparse.
        {
            if stage {
                let st = sc.wp05.as_mut().unwrap();
                let (os, sw) = (st.off_sids(), st.w);
                st.pin.i32s(os, sw).fill(slot as i32);
                st.pin.htod(os, d_sids, sw * 4)?;
            } else {
            self.dev.htod_copy_into(vec![slot as i32; sc.slot_ids.len()], &mut sc.slot_ids)?;
            }
        }
        // HOST / RO-5(a): speculative next pass (SpecPass) — only on the pinned, graphed path.
        let spec = stage && want_graph && self.spec_pass_begin(sc, slot, p)?;
        let mut ahead: Option<usize> = None; // graph key of pass i when iteration i-1 launched it
        let mut wasted: Option<usize> = None; // position of a launched pass the DDS then stopped before
        // TP-H: head draft passes this rank ISSUED this round (loop launches + speculative launches).
        // A sharded head puts collectives in every pass, so the count is part of the pre-verify
        // agreement: a rank-local speculative-launch decision that diverged would otherwise pair
        // the verify's first barrier with a wasted pass's.
        let mut launched = 0usize;
        for i in 0..kmax {
            // head QSA: per-pass regime + grid bucket (pass i attends from position p+i)
            let pq = self.qsa_live(p + i);
            let (pg, _) = self.qsa_grid_bucket_key(p + i);
            sc.qsa_grid_nblk = pg;
            let key = self.dds_pass_key(sc, slot, p, i); // HOST: one key formula (loop + speculative launch)
            let graphed = want_graph && self.graphs.lock().unwrap().contains_key(&key);
            if let Some(k2) = ahead {
                // HOST: launched (with its readback + event) by iteration i-1 — same graph, same inputs
                anyhow::ensure!(k2 == key && graphed,
                    "spec pass: pass {i} was launched as graph {k2:#x} but the loop key is {key:#x}");
            } else if graphed {
                self.graphs.lock().unwrap().get(&key).unwrap().launch();
                launched += 1;
            } else {
                self.draft_pass(&l, sc, head, i, pq)?;
                launched += 1;
            }
            // HOST: pinned readback slot i & 1 when speculating (pass i+1 lands in the other one)
            let rbs = if spec { 8 * (i & 1) } else { 0 };
            if stage && ahead.is_none() {
                // WP05: queue this pass's {dconf_i, d_i} readback into pinned memory, then build
                // verify PLE row i (token t_i: b, then d_{i-1}) while the pass runs on the GPU.
                let st = sc.wp05.as_mut().unwrap();
                let rb = st.off_rb() + rbs;
                st.pin.dtoh(rb, d_conf + 4 * i as u64, 4)?;
                st.pin.dtoh(rb + 4, d_d + 4 * i as u64, 4)?;
                if spec { st.spec.as_ref().unwrap().record(i)?; }
            }
            // HOST: launch pass i+1 BEFORE pass i's confidence is read, when it will very likely run
            // TP: pass i itself must be an already-seen key too, or one rank could skip speculating where its peer
            // (that precaptured this key) would not — the rank-local `graphed` alone is not symmetric.
            let sym_ok = self.tp.is_none()
                || sc.wp05.as_ref().and_then(|s| s.spec.as_ref()).map_or(false, |sp| sp.seen.contains(&key));
            let next = if spec && graphed && sym_ok && i + 1 < kmax { self.spec_pass_next(&l, sc, head, slot, p, i)? } else { None };
            if next.is_some() { launched += 1; }
            if stage {
                let st = sc.wp05.as_mut().unwrap();
                self.wp05_ple_row(st, hp, &tv_run, i)?;
            }
            if spec {
                // pass i's readback only — never waits for a speculative pass i+1 behind it
                sc.wp05.as_ref().unwrap().spec.as_ref().unwrap().wait(i)?;
            } else {
            self.dsync()?;
            }
            if want_graph && !graphed {
                self.capture_pass_graph(&l, sc, head, i, pq, key);
            }
            if self.tp.is_some() {
                if let Some(sp) = sc.wp05.as_mut().and_then(|s| s.spec.as_mut()) { sp.seen.insert(key); }
            }
            let c = if stage {
                let st = sc.wp05.as_mut().unwrap();
                // the synchronize above retired every stage copy (HOST: the event wait retired every
                // copy queued before pass i's readback; a speculative pass's readback may still be
                // landing in the OTHER rb slot, which no host code writes — it reads it only after
                // that slot's own event)
                st.inflight = false;
                let rb = st.off_rb() + rbs;
                let (cb, d) = (st.pin.read_u32(rb), st.pin.read_u32(rb + 4) as i32);
                // same clamp the old path applies after the loop (verify rejects a bad draft)
                let v = self.cfg.vocab_size as i32;
                tv_run.push(if d < 0 || d >= v { 0 } else { d });
                if ple_xcheck_on() {
                    let (oc, od) = (self.dtoh(&sc.dconf)?[i], self.dtoh(&sc.d_dev)?[i]);
                    if oc.to_bits() != cb || od != d {
                        crate::dprintln!("PLE_XCHECK dds pass {i}: readback MISMATCH conf {:08x} vs {:08x}, d {d} vs {od}", cb, oc.to_bits());
                    }
                }
                f32::from_bits(cb)
            } else {
            self.dtoh(&sc.dconf)?[i] // 6 floats; one tiny readback per pass
            };
            confs.push(c);
            if i + 1 < kmax && !keep(i, c) {
                w = i + 1;
                if next.is_some() { wasted = Some(p + i + 1); } // HOST: pass i+1 ran for nothing
                break;
            }
            ahead = next;
        }
        if spec { self.spec_pass_end(&l, sc, head, slot, kmax, w, wasted)?; }
        let tp1 = std::time::Instant::now();
        let drafts: Vec<i32> = if stage {
            // WP05: tv_run = [b, d_0 .. d_{w-1}] (clamped) came back pass by pass; one verify PLE
            // row was built per executed pass (rows 0..w, tokens tv_run[..w]); mtp_verify builds
            // row w (token d_{w-1}) and reuses the rest.
            let st = sc.wp05.as_mut().unwrap();
            let rows = tv_run.len() - 1;
            st.ple_pre = if self.ple.is_some() && rows > 0 { Some((slot, hp, tv_run[..rows].to_vec())) } else { None };
            tv_run[1..].to_vec()
        } else {
        let d_buf = self.dtoh(&sc.d_dev)?;
        let mut drafts: Vec<i32> = d_buf[..w].to_vec();
        let v = self.cfg.vocab_size as i32;
        for d in drafts.iter_mut() {
            if *d < 0 || *d >= v { *d = 0; }
        }
        drafts
        };
        Ok(DraftHalf { drafts, confs, w, launched, wasted, spec, tp0, tp1 })
    }

}
