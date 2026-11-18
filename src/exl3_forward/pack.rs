//! Packed verify: the `PackVerifyMs`, `PackDraft`, `DraftHalf`, `MtpRoundOut` types, the verify graph capture
//! for packed rounds and the `pack_*` methods (precapture, family keys, verify, finish, list hand-off).
//! Batch-invariant verify rows (AGENTS 2.4). See `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------
// S-A3-f-d — MTP k-chain verify round (the multiplier).
//
// Round structure (reference anatomy answer key, k drafts / width-(k+1) verify):
//   draft-extend  head(b, T_pred(b))                -> d0   [argmax kept on device]
//   draft x k-1   head(d_{i-1}, p+i)                -> d_i  [device-resident chain]
//   VERIFY        trunk rows (b, d0..d_{k-1}) at p..p+k  -> e_0..e_k
//   accept        a = longest prefix with d_i == e_i     [one device kernel]
//   emit          d_0..d_{a-1} + bonus e_a           (a+1 tokens; E = 1+p+..+p^k)
//   commit        replay rows 0..a of every stateful op from the LIVE state
//                 (bit-identical; rows a+1..k never commit)  [device-keyed]
//   re-prime      the head consumed the confirmed tokens via the NEXT round's
//                 draft chain; row a's verify tap becomes that draft-extend tap
//
// Greedy chain-verify is bitwise-lossless by construction: every emitted token
// equals the trunk's own argmax at its position with the true prefix (accepted
// drafts d_i == e_i by the accept check; the bonus IS e_a), the trunk commits
// only valid-row states, and the head never feeds back into the trunk.
// ---------------------------------------------------------------------------

/// S-B9-CF-PACK4: host wall ms of the three parts of one packed verify (attribution only).
#[derive(Default, Clone, Copy, Debug)]
pub struct PackVerifyMs {
    /// plan + PLE host half + the metadata uploads + the PLE-ring shadow copies
    pub stage: f64,
    /// graph replay + the taps hand-off + the device synchronize (the GPU time of the verify)
    pub graph: f64,
    /// accept readback + the per-segment parse + the host n-gram ring restore
    pub read: f64,
    /// PACK4S: host time of the --wp24-xcheck re-derivation (0 unless that diagnostic is on);
    /// not round cost — the serve subtracts it, as it does a lone round's `diag_ms`
    pub diag: f64,
}

/// S-B9-CF-PACK4: the once-per-boot LIVE line of the served packed verify (AGENTS 4: a new
/// dispatch path names the entry that served it — identical hashes are not a liveness receipt).
fn pack_live_note(r: usize, qsa: bool) {
    static LIVE: std::sync::Once = std::sync::Once::new();
    LIVE.call_once(|| crate::drprintln!("[exl3] lane-pack LIVE: a packed speculative verify replayed ONE graph for two requests (the readback and the per-segment tripwires follow; first at R = {r} rows, {} regime; seg kernels xq_gdn_step_chunk_seg / xq_conv1d_chunk_seg_ns / xq_ple_conv_chunk_seg, accept xq_accept_seg, commit xq_conv_commit_seg / xq_gdn_commit_ring_seg / xq_ple_ring_commit_seg)",
                                       if qsa { "sparse" } else { "dense" }));
}

/// S-B9-CF-PACK4S: the once-per-boot LIVE lines of SAMPLED rows in a packed verify — one for
/// the real-q family (`xq_sample_rows_rq` in the replayed graph) and one for match-mode rows
/// (`xq_sample_rows` of the greedy family), each naming the staging of the pair that hit it.
fn pack_sampled_live_note(r: usize, rq: bool, kinds: [&str; 2]) {
    static RQ: std::sync::Once = std::sync::Once::new();
    static MATCH: std::sync::Once = std::sync::Once::new();
    if rq {
        RQ.call_once(|| crate::drprintln!("[exl3] lane-pack LIVE (sampled, real-q): a packed verify of R = {r} rows replayed the rq graph family (xq_sample_rows_rq; segment 0 = a {} lane, segment 1 = a {} lane; per-row sampler blocks with each lane's own seed and counters, draft lists at each segment's row offset)", kinds[0], kinds[1]));
    } else {
        MATCH.call_once(|| crate::drprintln!("[exl3] lane-pack LIVE (sampled, match rule): a packed verify of R = {r} rows sampled its rows with xq_sample_rows (segment 0 = a {} lane, segment 1 = a {} lane; the greedy graph family)", kinds[0], kinds[1]));
    }
}

pub struct MtpRoundOut {
    pub a: usize,            // accepted drafts
    pub emitted: Vec<i32>,   // a+1 tokens: d_0..d_{a-1}, e_a
    pub pos_next: usize,     // position of the next round's bonus
    pub last_tok: i32,       // the bonus e_a (next round's b)
    pub tap_row: usize,      // taps_keep row holding the next draft-extend tap
    pub drafts: Vec<i32>,    // the full drafted chain (k) — for post-hoc scoring
    // WP24: host time (ms) this round spent in the --wp24-dump / --wp24-xcheck diagnostics
    // (readbacks + host re-derivation); the serve subtracts it from the round's cost (0 when off).
    pub diag_ms: f64,
}

impl FwdModel {
    /// Record verify_block into a CUDA graph (after the eager round's readback, so the
    /// capture only RECORDS). Failure is loud and leaves verify eager (never wrong).
    pub(super) fn capture_verify_graph(&self, l: &Launcher, sc: &mut Scratch, slot: usize, m: usize,
                            k: usize, qsa: bool, vkey: usize) {
        use cudarc::driver::sys;
        let raw = self.stream.stream;
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(raw, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            crate::metrics::exl3_frac_capture_reset(); // K35: this capture's fractional-expert round tally starts at 0
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::rprintln!("EXL3_GRAPH: verify BeginCapture failed ({r:?}) at m={m} — staying eager");
                crate::metrics::graph_capture_lost("verify", &format!("m={m}, slot={slot}, qsa={qsa}"), &format!("{r:?}"));
                return;
            }
        }
        let cap = self.verify_block(l, sc, slot, m, k, qsa);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r2 = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
        if r2 != sys::CUresult::CUDA_SUCCESS || cap.is_err() {
            crate::rprintln!("EXL3_GRAPH: verify capture failed (end={r2:?}, body={:?}) at m={m} — staying eager", cap.err());
            crate::metrics::graph_capture_lost("verify", &format!("m={m}, slot={slot}, qsa={qsa}"), &capture_lost_status(r2));
            return;
        }
        crate::exl3_wp27::finish(graph, "verify", m); // WP27: node priorities + DAG receipt
        if !self.tp_spine_audit(graph, "verify", m) { unsafe { sys::cuGraphDestroy(graph); } return; } // TP-C: barriers on the spine
        unsafe {
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            let r3 = crate::exl3_wp27::instantiate(&mut exec, graph); // WP27 (= cuGraphInstantiate_v2 unless --wp27-prio=1)
            sys::cuGraphDestroy(graph);
            if r3 == sys::CUresult::CUDA_SUCCESS {
                precap_upload(exec, raw);
                crate::metrics::set_graph_entries({ let mut g = self.graphs.lock().unwrap(); g.insert(vkey, crate::gpu::CudaGraph::from_exec(exec, raw).with_frac_rounds(crate::metrics::exl3_frac_capture_take())); g.len() as u64 });
                if graph_log_on() { crate::rprintln!("EXL3_GRAPH: captured verify block at m={m} (qsa={qsa}, slot={slot})"); }
            } else {
                crate::rprintln!("EXL3_GRAPH: verify instantiate failed ({r3:?}) at m={m} — staying eager");
                crate::metrics::graph_capture_lost("verify", &format!("m={m}, slot={slot}, qsa={qsa}"), &format!("instantiate {r3:?}"));
            }
        }
    }

    // ==================== PACK3 LEG 3 (S-B9-CF-PACK3) ====================
    // Probe-only CUDA-graph path for the packed verify. The graphable body is
    // the seg trunk + accept_seg + commit_seg: every per-staging value those
    // consume (slot, row0, len per segment) is read ON DEVICE from sc.seg, so
    // one graph serves ANY k_seg=2 staging with the same R_pad (5+5, 6+4, ...).
    // The two spots whose launch offsets bake slot/row0/len — the PLE ring
    // shadow (slot-plane src) and the per-segment taps snapshot (slot-window
    // dst + row0/len src) — stay OUT of the graph and run as a 4-launch eager
    // tail around the replay (`pack_eager_tail_*`; same launches, same args as
    // verify_block_packed_full's equivalents). The served boot never captures
    // this family (probe-only, key space 4<<30, own scratch gid via sk()).

    fn verify_block_packed_graphable(&self, l: &Launcher, sc: &mut Scratch, rpad: usize,
                                     k_seg: usize, qsa: bool) -> Result<()> {
        let _tc = tune::enter(tune::Ctx::fam(tune::Fam::Verify, rpad).regime(qsa));
        self.verify_kernels(l, sc, rpad, qsa, 0, k_seg)?;
        xqlaunch!(l, "xq_accept_seg", (k_seg as u32, 1, 1), (32, 1, 1), 0,
                  (&sc.d_dev, &sc.argmax, &mut sc.accept_out, &mut sc.acc_seg, &sc.seg))?;
        self.verify_commit_seg(l, sc, k_seg)
    }

    /// The eager tail around a packed graph replay: the PLE ring shadow (BEFORE
    /// the replay) and the per-segment taps snapshot (AFTER — it reads the
    /// trunk's resid). Same launches, same args as in verify_block_packed_full.
    pub(super) fn pack_eager_tail_pre(&self, l: &Launcher, sc: &mut Scratch, segs: &[[i32; 4]]) -> Result<()> {
        self.verify_shadow_seg(l, sc, segs)
    }

    pub(super) fn pack_eager_tail_post(&self, l: &Launcher, sc: &mut Scratch, segs: &[[i32; 4]]) -> Result<()> {
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        for sg in segs.iter() {
            let (slot, row0, len) = (sg[0] as usize, sg[1] as usize, sg[2] as usize);
            let tb = self.tap_base(sc, slot)?;
            xqlaunch!(l, "xq_copy_f32", grid((he * len) as i64), (256, 1, 1), 0,
                      (&mut sc.taps_keep, &sc.resid, (he * len) as i64, tb as i64,
                       (row0 * he) as i64))?;
        }
        Ok(())
    }

    /// PACK3: capture the packed verify body under `pkey` (pkeyv_of | sk).
    /// Mirrors capture_verify_graph: THREAD_LOCAL capture on the compute
    /// stream, WP27 finish + spine audit, instantiate, into the graph map.
    /// Stream capture RECORDS only — nothing executes.
    pub(super) fn capture_verify_graph_packed(&self, l: &Launcher, sc: &mut Scratch, rpad: usize,
                                   k_seg: usize, qsa: bool, pkey: usize) {
        use cudarc::driver::sys;
        let raw = self.stream.stream;
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(raw, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            crate::metrics::exl3_frac_capture_reset(); // K35: this capture's fractional-expert round tally starts at 0
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::rprintln!("EXL3_GRAPH: packed BeginCapture failed ({r:?}) at rpad={rpad} — staying eager");
                crate::metrics::graph_capture_lost("packed-verify", &format!("rpad={rpad}, k_seg={k_seg}, qsa={qsa}"), &format!("{r:?}"));
                return;
            }
        }
        let cap = self.verify_block_packed_graphable(l, sc, rpad, k_seg, qsa);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r2 = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
        if r2 != sys::CUresult::CUDA_SUCCESS || cap.is_err() {
            crate::rprintln!("EXL3_GRAPH: packed capture failed (end={r2:?}, body={:?}) at rpad={rpad} — staying eager", cap.err());
            crate::metrics::graph_capture_lost("packed-verify", &format!("rpad={rpad}, k_seg={k_seg}, qsa={qsa}"), &capture_lost_status(r2));
            return;
        }
        crate::exl3_wp27::finish(graph, "packed-verify", rpad);
        if !self.tp_spine_audit(graph, "packed-verify", rpad) { unsafe { sys::cuGraphDestroy(graph); } return; }
        unsafe {
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            let r3 = crate::exl3_wp27::instantiate(&mut exec, graph);
            sys::cuGraphDestroy(graph);
            if r3 == sys::CUresult::CUDA_SUCCESS {
                precap_upload(exec, raw);
                crate::metrics::set_graph_entries({ let mut g = self.graphs.lock().unwrap(); g.insert(pkey, crate::gpu::CudaGraph::from_exec(exec, raw).with_frac_rounds(crate::metrics::exl3_frac_capture_take())); g.len() as u64 });
                if graph_log_on() { crate::rprintln!("EXL3_GRAPH: captured packed verify at rpad={rpad} (k_seg={k_seg}, qsa={qsa})"); }
            } else {
                crate::rprintln!("EXL3_GRAPH: packed instantiate failed ({r3:?}) at rpad={rpad} — staying eager");
                crate::metrics::graph_capture_lost("packed-verify", &format!("rpad={rpad}, k_seg={k_seg}, qsa={qsa}"), &format!("instantiate {r3:?}"));
            }
        }
    }

    // ==================== PACK4 LEG 4 (S-B9-CF-PACK4): the SERVED packed round ====================
    // `--lane-pack on`: two requests share one packed verify (PACK4: greedy; PACK4S: sampled
    // and mixed pairs too). Drafting and the head
    // re-prime stay per request on the SERVED scratch (`mtp_draft_adaptive`, `head_reprime`);
    // the verify runs on its own 16-row scratch through the PACK3 graph family, keyed on the
    // exact row total R. The one cross-scratch hand-off is the taps: the verify's resid rows
    // go into each slot's window of the SERVED scratch, where the next draft and the
    // re-prime read them exactly as after a lone verify.

    /// Can this boot run the packed verify at all? The segmented kernels exist for the ring
    /// commit, the WP12 nostore conv and kd = vd = 128 only (verify_kernels refuses the rest),
    /// and the served arm replays GRAPHS only: it never runs an eager packed block and never
    /// captures while serving.
    pub fn pack_supported(&self) -> Result<()> {
        let cfg = &self.tc;
        anyhow::ensure!(self.mtp.is_some() && !mtp_disabled_by_opt(), "--lane-pack on needs the MTP draft head (this boot has none, or MTP is off)");
        // TP=1, TP=2 and TP=4 (owner 2026-10-10: the TP=4 refusal is lifted); a larger world is
        // refused before the network (xtp::world_gt2_flag_refusals) and, structurally, here
        anyhow::ensure!(self.tp.as_ref().map_or(true, |t| t.world as u32 <= crate::lane_pack::LANE_PACK_MAX_WORLD),
                        "--lane-pack runs at TP=1, TP=2 and TP=4 (this boot is TP={})", self.tp.as_ref().map_or(1, |t| t.world));
        anyhow::ensure!(cfg.lin_k_dim == 128 && cfg.lin_v_dim == 128,
                        "--lane-pack on: the segmented GDN kernels need kd = vd = 128 (this pack: {} / {})", cfg.lin_k_dim, cfg.lin_v_dim);
        anyhow::ensure!(gdn_commit_mode(cfg.lin_k_dim, cfg.lin_v_dim) == 1,
                        "--lane-pack on: the per-segment commit is the ring commit (--exl3-gdn-ring 0 / --exl3-gdn-ring-check are set)");
        anyhow::ensure!(crate::opts::var(crate::opt!("exl3-no-graph")).is_err()
                        && crate::opts::var(crate::opt!("mtp-dump")).map_or(true, |s| s.is_empty())
                        && !wp12_xcheck(),
                        "--lane-pack on replays captured graphs only (--exl3-no-graph / --mtp-dump / the WP12 cross-check turn the verify graphs off)");
        anyhow::ensure!(!a_flags().taps_shared, "--lane-pack on needs the per-slot taps windows (--a-taps=shared is set)");
        Ok(())
    }

    /// The packed verify's own scratch: 16 rows, a greedy sampler block (real zeros until a
    /// sampled pair stages it, `set_pack_sampling`) and no penalties (never staged: the packed
    /// arm serves unpenalized lanes only), with its own graph-key id (`sk`), so no served graph
    /// can replay on it. PACK4S: its real-q list holds one row per verify row (16).
    pub fn pack_scratch(&self) -> Result<Scratch> {
        let scp = Self::scratch(&self.dev, &self.cfg, crate::lane_pack::PACK_ROWS_MAX, self.max_pos)?;
        anyhow::ensure!(!scp.samp_live && !scp.pen_live && !scp.rq_live && scp.rq_mode == 0,
                        "lane-pack: a fresh scratch must be greedy and unpenalized");
        anyhow::ensure!(scp.rq_list.len() >= crate::lane_pack::PACK_ROWS_MAX * crate::wp24::RQ_ROW
                        && scp.samp.len() == crate::lane_pack::PACK_ROWS_MAX * 8,
                        "lane-pack: the packed sampler block / real-q list are narrower than {} rows", crate::lane_pack::PACK_ROWS_MAX);
        anyhow::ensure!(scp.d_dev.len() >= crate::lane_pack::PACK_ROWS_MAX && scp.accept_out.len() >= crate::lane_pack::PACK_ROWS_MAX
                        && scp.seg.len() >= crate::lane_pack::PACK_K * 4 && scp.acc_seg.len() >= crate::lane_pack::PACK_K * 2,
                        "lane-pack: packed staging buffers narrower than {} rows", crate::lane_pack::PACK_ROWS_MAX);
        Ok(scp)
    }

    /// The regimes a packed verify can be asked for on this boot: dense always; sparse when the
    /// pack has a live indexer, the scorer is bucket-free (one graph per regime serves every
    /// context — the served precapture's own rule) and the window reaches the switch.
    fn pack_regimes(&self) -> Vec<(bool, usize)> {
        let mut regimes = vec![(false, 0usize)];
        if self.cfg.has_indexer() && !qsa_disabled() && self.qsa_bucket_free() {
            let ps = self.qsa_limit() + 1;
            if ps < self.max_pos && self.qsa_live(ps) { regimes.push((true, ps)); }
        }
        regimes
    }

    /// The packed verify graph key of one (family, regime, width): PACK3's family key, the real-q
    /// bit (PACK4S), the vocabulary-parallel tail bits the family's sampler state selects (TP:
    /// S-B9-CF-PACK5; zero at TP=1, where there is no vocab-parallel head, so every TP=1 key is
    /// PACK4S's bit for bit) and the packed scratch's id.
    fn pack_fam_key(&self, scp: &Scratch, r: usize, sparse: bool, qb: usize, fam: crate::lane_pack::PackFam) -> usize {
        let (samp, rq) = fam.state();
        pack_key(r, sparse, qb, rq) | self.vp_key_st(samp, false, rq, r) | sk(scp)
    }

    /// Boot precapture of the packed families on `scp`: every exact width R in 4..=16 for each
    /// reachable regime and each family in `fams` (capture-only: nothing executes, no device byte
    /// changes). The served arm NEVER captures a packed graph while serving, so a width that
    /// failed to capture fails the boot loudly instead of leaving a lazy path behind.
    ///
    /// PACK4S: `fams` carries the real-q family under `--spec-sampling ratio`. Each family is
    /// captured with the scratch's sampler shadows set to the state its graphs replay under
    /// (`PackFam::state`, restored after): the TP verify tail is chosen from that state. At TP=1
    /// the match-rule family has the greedy family's key and is not captured twice.
    ///
    /// PACK4-REVIEW B2: the 1 GB cap that read a `cuMemGetInfo` free-memory delta is GONE. That
    /// figure moved by up to +724 MB for the same captures (unified memory, concurrent
    /// allocations) and could refuse a healthy boot; a real out-of-memory shows up as a capture
    /// that did not land in the graph map, which is checked per graph below and is the exact
    /// condition the cap was a noisy proxy for. The memory delta is still returned for the boot
    /// line (informational, labelled noisy there).
    ///
    /// TP: the loop's order and contents are a function of resolved flags and the pack config only
    /// (regimes, widths, `fams`), so both ranks capture the same keys in the same order; the boot
    /// agree compares the resulting set (`pack_graph_set`) before the first request.
    /// Returns (graphs captured, device-memory delta in MB, the regimes line).
    pub fn pack_precapture(&self, scp: &mut Scratch, fams: &[crate::lane_pack::PackFam]) -> Result<(usize, f64, String)> {
        self.pack_supported()?;
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let free0 = cudarc::driver::result::mem_get_info().map(|x| x.0).unwrap_or(0);
        let regimes = self.pack_regimes();
        let mut n = 0usize;
        // Host-side one-time init the first launch of a 9..16-row entry would otherwise do INSIDE
        // a capture: the raw-module handles of the R16 router fold and hc fuse (and the 10..16-row
        // expert epilogues; the served precapture resolves the rest of its list).
        for f in ["xq_router_fold_c_r16", "xq_hc_fuse_i8_rbk_r16", "xq_moe_gu_epi_c2", "xq_moe_gu_epi_c4",
                  "xq_moe_gu_epi_b4_c2", "xq_moe_gu_epi_b4_c4", "xq_moe_dn_epi_k10", "xq_moe_dn_epi_b4_k10"] {
            let _ = xq_raw_fn(f);
        }
        // `--lane-pack-drill-skip-sparse` (diagnostic, one-sided per box): this rank holds no sparse-regime
        // packed graphs, so the TP boot agree has a rank-local graph-set difference to refuse
        let drill_skip_sparse = crate::opts::var(crate::opt!("lane-pack-drill-skip-sparse")).map_or(false, |v| !matches!(v.trim(), "" | "0" | "off"));
        if drill_skip_sparse {
            crate::rprintln!("[exl3-serve] lane-pack DRILL (--lane-pack-drill-skip-sparse): this rank captures NO sparse-regime packed verify graphs; \
its packed graph set will differ from its peer's — the TP boot agree must refuse the boot");
        }
        PRECAP_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
        let r = (|| -> Result<()> {
            for &fam in fams {
                // the capture records the row sampler and the vp tail the family replays
                // (verify_kernels reads the shadows)
                let (samp, rq) = fam.state();
                scp.samp_live = samp;
                scp.rq_live = rq;
                for &(sparse, p) in &regimes {
                    if sparse && drill_skip_sparse { continue; }
                    let (qg, qb) = self.qsa_grid_bucket_key(p);
                    for r in crate::lane_pack::PACK_WIDTHS {
                        let key = self.pack_fam_key(scp, r, sparse, qb, fam);
                        if self.graphs.lock().unwrap().contains_key(&key) { continue; }
                        scp.qsa_grid_nblk = qg;
                        self.capture_verify_graph_packed(&l, scp, r, crate::lane_pack::PACK_K, sparse, key);
                        anyhow::ensure!(self.graphs.lock().unwrap().contains_key(&key),
                                        "lane-pack: the packed verify graph at R = {r} ({} regime, {} family) did not capture (see the EXL3_GRAPH line above) — boot with --lane-pack off",
                                        if sparse { "sparse" } else { "dense" }, fam.label());
                        n += 1;
                    }
                }
            }
            Ok(())
        })();
        // a fresh packed scratch is greedy until a sampled pair stages it
        scp.samp_live = false;
        scp.rq_live = false;
        PRECAP_ACTIVE.store(false, std::sync::atomic::Ordering::Relaxed);
        self.dsync()?; // the uploads were queued on the compute stream; no kernel ran
        r?;
        let free1 = cudarc::driver::result::mem_get_info().map(|x| x.0).unwrap_or(free0);
        let note = format!("regimes: dense{}", match (regimes.len() > 1, self.cfg.has_indexer() && !qsa_disabled()) {
            (true, _) => ", sparse (bucket-free)",
            (false, true) => " only (sparse lanes are not packed: bucketed scorer or the window ends before the switch)",
            (false, false) => " only (no sparse attention on this boot)",
        });
        Ok((n, (free0 as f64 - free1 as f64) / 1e6, note))
    }

    /// Is the whole packed family `fam` present for a round whose rows are on the `sparse` side?
    /// (the arm asks BEFORE drafting: once two lanes have drafted, their verify must be able to run.)
    pub fn pack_family_ready(&self, scp: &Scratch, sparse: bool, fam: crate::lane_pack::PackFam) -> bool {
        let Some(&(_, p)) = self.pack_regimes().iter().find(|(s, _)| *s == sparse) else { return false };
        let (_, qb) = self.qsa_grid_bucket_key(p);
        let g = self.graphs.lock().unwrap();
        crate::lane_pack::PACK_WIDTHS.clone().all(|r| g.contains_key(&self.pack_fam_key(scp, r, sparse, qb, fam)))
    }

    /// S-B9-CF-PACK5: the packed graph set this rank holds, as (graphs present, FNV digest of the
    /// (family, regime, width, present) table in the capture loop's own order, a description).
    /// The digest and count go into the TP boot agree: a pack whose ranks hold DIFFERENT sets (the
    /// receipt of the CF-P1e TP=2 `--max-batch 16` death was exactly such a rank-local graph
    /// presence read by a scheduling decision) refuses to boot instead of deciding rank-locally.
    pub fn pack_graph_set(&self, scp: &Scratch, fams: &[crate::lane_pack::PackFam]) -> (usize, u32, String) {
        let regimes = self.pack_regimes();
        let g = self.graphs.lock().unwrap();
        let (mut n, mut h) = (0usize, 0x811c_9dc5u32);
        let mut parts: Vec<String> = Vec::new();
        for (fi, &fam) in fams.iter().enumerate() {
            for &(sparse, p) in &regimes {
                let (_, qb) = self.qsa_grid_bucket_key(p);
                let mut have = 0usize;
                for r in crate::lane_pack::PACK_WIDTHS {
                    let present = g.contains_key(&self.pack_fam_key(scp, r, sparse, qb, fam));
                    have += present as usize;
                    for b in [fi as u8, sparse as u8, r as u8, present as u8] {
                        h ^= b as u32;
                        h = h.wrapping_mul(0x0100_0193);
                    }
                }
                n += have;
                parts.push(format!("{} {} R={}..{} x{}", fam.label(), if sparse { "sparse" } else { "dense" },
                                   crate::lane_pack::PACK_WIDTHS.start(), crate::lane_pack::PACK_WIDTHS.end(), have));
            }
        }
        (n, h, parts.join(", "))
    }

    /// One request's draft chain for a packed round: exactly the lone round's draft half, on the
    /// served scratch. `keep` = Some: the DDS chain (`mtp_draft_adaptive`: tap restore, DDS
    /// passes, speculative next pass, clamped drafts); None (`--draft-confidence 0`): the
    /// fixed-depth chain of exactly `kmax` drafts (`mtp_draft_fixed`, the lone `mtp_round`'s
    /// own). The caller staged the lane's draft sampler on `sc` first (`set_spec_sampling`), as
    /// the lone round does; `samp` / `rq` record what it staged for the packed verify.
    #[allow(clippy::too_many_arguments)]
    pub fn pack_draft(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, kmax: usize, b: i32, p: usize,
                      tap_row: usize, keep: Option<&mut dyn FnMut(usize, f32) -> bool>,
                      samp: crate::lane_pack::SegSamp, rq: Option<RqStage>) -> Result<PackDraft> {
        anyhow::ensure!(samp.ratio == rq.map_or(false, |r| r.mode == 2) && sc.rq_live == samp.ratio && sc.rq_mode == rq.map_or(0, |r| r.mode),
                        "pack_draft: slot {slot} drafts with a sampler staging that is not its own (ratio {}, staged mode {})", samp.ratio, sc.rq_mode);
        let (drafts, confs, spec, launched, wasted) = match keep {
            Some(keep) => {
                let d = self.mtp_draft_adaptive(sc, head, slot, kmax, b, p, tap_row, keep)?;
                (d.drafts, d.confs, d.spec, d.launched, d.wasted.is_some())
            }
            // the fixed-depth chain: exactly `kmax` passes, no speculative one
            None => (self.mtp_draft_fixed(sc, head, slot, kmax, b, p, tap_row)?.0, Vec::new(), false, kmax, false),
        };
        Ok(PackDraft { slot, b, p, drafts, confs, spec, samp, rq, launched, wasted })
    }

    /// S-B9-CF-PACK5: TP-I's drain for the packed round. A wasted speculative pass of EITHER lane
    /// is a collective whose doorbell epoch may still be in flight; the pack's one pre-verify
    /// exchange stages in send slot 0 and lands in recv slot `gen & 7` of the same rings, so it
    /// must not run before the device has consumed the peer's epoch and the proxy has retired
    /// ours (the lone round does exactly this before ITS exchange, `mtp_round_adaptive`). Same
    /// condition: world 2 with the sharded draft head, only when a pass was wasted (~4.5% of
    /// lane rounds); one device sync covers both lanes' passes (they run on one stream, in order).
    pub fn pack_drain_wasted(&self, ds: &[PackDraft]) -> Result<()> {
        if ds.iter().any(|d| d.wasted) && self.tp.as_ref().map_or(false, |t| t.world == 2) && xtp::dh_shard_on() {
            self.dsync()?;
            crate::net::drain_sends(std::time::Duration::from_millis(200))?;
        }
        Ok(())
    }

    /// PACK4S: hand a drafted real-q lane's lists to the packed scratch. The lane's chain wrote
    /// list rows 0..w of the SERVED scratch (one per draft pass); its verify rows start at row
    /// `row0` of the packed verify, where `xq_sample_rows_rq` reads list row `row0 + i`. A D2D
    /// copy enqueued on the COMPUTE stream (AGENTS 2.3), i.e. behind the chain that wrote the
    /// rows and ahead of everything issued later: the caller issues it BEFORE the next lane
    /// drafts, because that chain writes the same served rows. No-op for a greedy or match lane.
    pub fn pack_list_handoff(&self, sc: &Scratch, scp: &mut Scratch, row0: usize, d: &PackDraft) -> Result<()> {
        let Some((doff, words)) = crate::lane_pack::list_handoff(row0, d.drafts.len(), d.samp.ratio, scp.rq_list.len())? else { return Ok(()) };
        anyhow::ensure!(words <= sc.rq_list.len(), "pack_list_handoff: {words} list words > the served list ({})", sc.rq_list.len());
        use cudarc::driver::sys;
        self.dev.bind_to_thread()?;
        let (dst, src) = (*scp.rq_list.device_ptr() + (doff * 4) as u64, *sc.rq_list.device_ptr());
        let r = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, words * 4, self.stream.stream) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "pack_list_handoff: list copy failed ({r:?})");
        Ok(())
    }

    /// PACK4S: stage the packed verify's per-row sampler block (`lane_pack::pack_samp`) and the
    /// two shadows the graph key and the kernels follow. An all-greedy pair after an all-greedy
    /// state uploads nothing and leaves the scratch exactly as PACK4 had it.
    fn set_pack_sampling(&self, scp: &mut Scratch, ps: &crate::lane_pack::PackSamp) -> Result<()> {
        anyhow::ensure!(ps.words.len() == scp.samp.len(), "set_pack_sampling: {} words for a {}-word block", ps.words.len(), scp.samp.len());
        if ps.samp_live || scp.samp_live {
            self.dev.htod_copy_into(ps.words.clone(), &mut scp.samp)?;
        }
        scp.samp_live = ps.samp_live;
        scp.rq_live = ps.rq_live;
        Ok(())
    }

    /// PACK4S (`--wp24-xcheck`): re-derive every real-q segment of the packed verify that just
    /// completed on the host (wp24::xcheck_round: the draft draw, q', the accept test, the
    /// residual support, the bonus row) from the PACKED scratch's own rows — the list rows at
    /// the segment's row offset, its sampler blocks and its drafts. Diagnostics; syncs.
    fn pack_xcheck(&self, scp: &Scratch, meta: &crate::lane_pack::PackMeta, ds: [&PackDraft; 2], acc: [usize; 2]) -> Result<()> {
        static SEGS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let v = self.cfg.vocab_size;
        self.dsync()?;
        let logits = self.dtoh(&scp.logits)?;
        let emitted = self.dtoh(&scp.argmax)?;
        let samp = self.dtoh(&scp.samp)?;
        let list = self.dtoh(&scp.rq_list)?;
        for (j, d) in ds.into_iter().enumerate() {
            let (Some(rq), true) = (d.rq, d.samp.ratio) else { continue };
            let (row0, m) = (meta.segs[j][1] as usize, meta.segs[j][2] as usize);
            let w = m - 1;
            anyhow::ensure!(logits.len() >= (row0 + m) * v && emitted.len() >= row0 + m && samp.len() >= (row0 + m) * 8
                            && list.len() >= (row0 + w) * crate::wp24::RQ_ROW, "pack_xcheck: packed scratch narrower than segment {j}");
            crate::wp24::xcheck_round(&crate::wp24::Round {
                slot: d.slot, pos: d.p, w, a: acc[j], mode: 2, v,
                logits: &logits[row0 * v..(row0 + m) * v], emitted: &emitted[row0..row0 + m],
                samp: &samp[row0 * 8..(row0 + m) * 8],
                list: &list[row0 * crate::wp24::RQ_ROW..(row0 + w) * crate::wp24::RQ_ROW], drafts: &d.drafts,
                dsamp: rq.pack(), dctr: rq.dctr,
            });
            let n = SEGS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n == 1 || n % 64 == 0 {
                crate::drprintln!("WP24_XCHECK lane-pack: {n} real-q segments of PACKED verifies re-derived on the host so far (latest: slot {} rows {row0}..{} of R = {})",
                         d.slot, row0 + m - 1, meta.r);
            }
        }
        Ok(())
    }

    /// ONE packed verify for two drafted requests: stage both segments on the packed scratch,
    /// replay the R-keyed graph (seg trunk + per-segment accept + per-segment commit) between
    /// the eager PLE-ring shadow and the taps hand-off into the SERVED scratch's per-slot
    /// windows, read back once, and restore each slot's host n-gram ring to its accepted
    /// prefix. Each returned round is what `mtp_verify` returns for that request alone.
    /// PACK4S: the rows carry each lane's own sampler block (`set_pack_sampling`); a real-q
    /// lane's draft lists were already copied to its row offset (`pack_list_handoff`), and a
    /// pair holding one replays the rq family.
    pub fn pack_verify(&self, sc: &mut Scratch, scp: &mut Scratch, d0: &PackDraft, d1: &PackDraft) -> Result<([MtpRoundOut; 2], PackVerifyMs)> {
        use crate::lane_pack::{pack_meta, seg_result, SegIn};
        let tv0 = std::time::Instant::now();
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        for d in [d0, d1] {
            // WP02: verify rows p..p+k write trunk K/V, indexer keys and read cos_tab — all max_pos rows
            anyhow::ensure!(d.p + d.drafts.len() < self.max_pos,
                            "pack_verify: slot {} rows {}..={} past the KV window (max_pos {})", d.slot, d.p, d.p + d.drafts.len(), self.max_pos);
        }
        let meta = pack_meta(&SegIn { slot: d0.slot, b: d0.b, p: d0.p, drafts: &d0.drafts },
                             &SegIn { slot: d1.slot, b: d1.b, p: d1.p, drafts: &d1.drafts }, scp.d_dev.len())?;
        // one regime for the whole verify, from each request's LAST row (the lone verify's rule)
        let (q0, q1) = (self.qsa_live(d0.p + d0.drafts.len()), self.qsa_live(d1.p + d1.drafts.len()));
        anyhow::ensure!(q0 == q1, "pack_verify: slots {} and {} are on different sides of the sparse switch", d0.slot, d1.slot);
        let (qg, qb) = self.qsa_grid_bucket_key((d0.p + d0.drafts.len()).max(d1.p + d1.drafts.len()));
        // PACK4S: per-row sampler blocks (segment j row i = lane j's params, counter ctr_j + i);
        // an all-greedy pair stages nothing and keeps the greedy family
        let ps = crate::lane_pack::pack_samp(&meta.segs, &[d0.samp, d1.samp], scp.samp.len() / 8)?;
        self.set_pack_sampling(scp, &ps)?;
        // the family the staged state selects: PACK3's key + the real-q bit + (TP) the vp tail bits
        let pkey = pack_key(meta.r, q0, qb, scp.rq_live) | self.vp_key_st(scp.samp_live, false, scp.rq_live, meta.r) | sk(scp);
        anyhow::ensure!(self.graphs.lock().unwrap().contains_key(&pkey),
                        "pack_verify: no packed graph at R = {} (qsa {q0}, {} family) — the arm must not have packed this pair",
                        meta.r, crate::lane_pack::fam_of(scp.samp_live, scp.rq_live).label());
        // the host n-gram ring pair of each slot BEFORE the staging advances it
        let hp: Vec<(i32, i32)> = {
            let h = self.ple_hist.lock().unwrap();
            [d0, d1].iter().map(|d| (h[d.slot * 2], h[d.slot * 2 + 1])).collect()
        };
        self.pack2_stage(scp, &meta.rows)?;
        self.dev.htod_copy_into(meta.d_dev.clone(), &mut scp.d_dev)?;
        let mut seg_flat: Vec<i32> = meta.segs.iter().flat_map(|s| s.iter().copied()).collect();
        seg_flat.resize(scp.seg.len(), 0);
        self.dev.htod_copy_into(seg_flat, &mut scp.seg)?;
        scp.qsa_grid_nblk = qg;
        self.pack_eager_tail_pre(&l, scp, &meta.segs)?;
        let tv1 = std::time::Instant::now();
        self.graphs.lock().unwrap().get(&pkey).unwrap().launch();
        pack_live_note(meta.r, q0);
        if ps.samp_live {
            let kind = |d: &PackDraft| if d.samp.samp.is_none() { "greedy" } else if d.samp.ratio { "real-q sampled" } else { "match-rule sampled" };
            pack_sampled_live_note(meta.r, ps.rq_live, [kind(d0), kind(d1)]);
        }
        // taps: segment rows row0..row0+len of the packed resid -> rows 0..len of the slot's
        // window in the SERVED scratch (the lone verify's snapshot, per segment)
        for sg in meta.segs.iter() {
            let (slot, row0, len) = (sg[0] as usize, sg[1] as usize, sg[2] as usize);
            let tb = self.tap_base(sc, slot)?;
            xqlaunch!(l, "xq_copy_f32", grid((he * len) as i64), (256, 1, 1), 0,
                      (&mut sc.taps_keep, &scp.resid, (he * len) as i64, tb as i64, (row0 * he) as i64))?;
        }
        self.dsync()?;
        let tv2 = std::time::Instant::now();
        let out = self.dtoh(&scp.accept_out)?;
        let acc = self.dtoh(&scp.acc_seg)?;
        if let Some(st) = sc.wp05.as_mut() { st.inflight = false; } // WP05: the draft stage's copies retired
        let mut res: Vec<MtpRoundOut> = Vec::with_capacity(2);
        for (j, d) in [d0, d1].into_iter().enumerate() {
            let (a, emitted) = seg_result(&out, &acc, j, &meta.segs[j])?;
            anyhow::ensure!(a <= d.drafts.len(), "pack_verify: slot {} accept {a} past its {} drafts", d.slot, d.drafts.len());
            // host ring restore (mtp_verify's rule): the staging advanced it by m; a + 1 tokens committed
            {
                let mut hist = self.ple_hist.lock().unwrap();
                if a == 0 {
                    hist[d.slot * 2] = hp[j].1;
                    hist[d.slot * 2 + 1] = d.b;
                } else {
                    let tok = |i: usize| if i == 0 { d.b } else { d.drafts[i - 1] };
                    hist[d.slot * 2] = tok(a - 1);
                    hist[d.slot * 2 + 1] = tok(a);
                }
            }
            let last = emitted[a];
            res.push(MtpRoundOut { a, emitted, pos_next: d.p + a + 1, last_tok: last, tap_row: a,
                                   drafts: d.drafts.clone(), diag_ms: 0.0 });
        }
        let r1 = res.pop().unwrap();
        let r0 = res.pop().unwrap();
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        let tv3 = std::time::Instant::now();
        // PACK4S diagnostics (off unless --wp24-xcheck): the host re-derivation of the real-q segments
        if ps.rq_live && crate::wp24::xcheck_on() {
            self.pack_xcheck(scp, &meta, [d0, d1], [r0.a, r1.a])?;
        }
        Ok(([r0, r1], PackVerifyMs { stage: ms(tv0, tv1), graph: ms(tv1, tv2), read: ms(tv2, tv3), diag: tv3.elapsed().as_secs_f64() * 1e3 }))
    }

    /// The per-request tail of a packed round: where the speculative next pass expects this
    /// lane's next round, then the head re-prime from the taps the verify left in the lane's
    /// window (the eager a-row re-prime: the device re-prime's bits for rows < a).
    pub fn pack_finish(&self, sc: &mut Scratch, head: &DraftHead, d: &PackDraft, out: &MtpRoundOut) -> Result<()> {
        // PACK4-REVIEW A2: a packed round re-primes with the EAGER a-row path; the lone round's
        // device re-prime is never enqueued here. A hand-off a lone round left behind (it failed
        // between `reprime_dev_enqueue` and its own `head_reprime`; the engine keeps serving after
        // such an error) would otherwise be taken by `head_reprime` below, fail
        // `reprime_dev_finish`'s slot / position check and kill both lanes of the pair again. A lone
        // verify clears it itself (`rp.pending = None` in `reprime_dev_enqueue`).
        if let Some(rp) = sc.rp.as_mut() { rp.pending = None; }
        if d.spec {
            if let Some(np) = sc.wp05.as_mut().and_then(|s| s.spec.as_mut()).and_then(|s| s.next_p.get_mut(d.slot)) {
                *np = out.pos_next;
            }
        }
        self.head_reprime(sc, head, d.slot, d.p, &out.drafts[..out.a])
    }

}
