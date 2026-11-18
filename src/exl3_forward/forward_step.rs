//! The decode / verify-width step: `forward_step`, `forward_step_kernels` (the launch sequence), the step-graph
//! capture, `head_launches` and `head_forward`. Compute stream is a blocking stream (AGENTS 2.1): never fork_default_stream; D2D copies run on it.
//! See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// WP09: greedy argmax of sc.logits rows 0..m (row stride = width = v) -> sc.argmax[0..m].
    /// Default: ONE xq_argmax_rows launch (grid nb x m, last-block reduce, self re-arming
    /// counters) — the same ids as xq_argmax bit for bit (the comparator's order is total).
    /// --wp09-off=1 / --wp09-argmax=0: the m single-block xq_argmax launches.
    pub(super) fn argmax_rows(&self, l: &Launcher, sc: &mut Scratch, m: usize, v: usize) -> Result<()> {
        if !wp09_argmax_on() {
            for i in 0..m {
                xqlaunch!(l, "xq_argmax", (1, 1, 1), (1024, 1, 1), (1024 * 12) as u32,
                         (&mut sc.argmax, i as i32, &sc.logits, (i * v) as i64, v as i32))?;
            }
            return Ok(());
        }
        anyhow::ensure!(m >= 1 && m <= sc.am_cnt.len(), "argmax_rows: m {m} > scratch width {}", sc.am_cnt.len());
        let nb = wp09_am_nb(m);
        xqlaunch!(l, "xq_argmax_rows", (nb as u32, m as u32, 1), (256, 1, 1), 0,
                 (&mut sc.argmax, 0i32, &sc.logits, 0i64, v as i64, v as i32, &mut sc.am_part, &mut sc.am_cnt))?;
        if wp09_xcheck() && !self.stream_capturing() {
            let mut old = self.dev.alloc_zeros::<i32>(m)?;
            for i in 0..m {
                xqlaunch!(l, "xq_argmax", (1, 1, 1), (1024, 1, 1), (1024 * 12) as u32,
                         (&mut old, i as i32, &sc.logits, (i * v) as i64, v as i32))?;
            }
            self.dsync()?;
            let a = self.dtoh(&sc.argmax)?;
            let b = self.dtoh(&old)?;
            let mism: Vec<(usize, i32, i32)> = (0..m).filter(|&i| a[i] != b[i]).map(|i| (i, a[i], b[i])).collect();
            crate::dprintln!("WP09_XCHECK argmax m={m} v={v} nb={nb} mism={} first={:?} ids={:?}",
                     mism.len(), mism.first(), &a[..m]);
        }
        Ok(())
    }

    /// WP09 (XCHECK only): true while self.stream is being captured into a graph — the
    /// diagnostic syncs the device, which is illegal inside a capture. (Also WP04/WP14's
    /// eager-only XCHECKs.)
    pub(super) fn stream_capturing(&self) -> bool {
        use cudarc::driver::sys;
        let mut st = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let r = unsafe { sys::cuStreamIsCapturing(self.stream.stream, &mut st) };
        r != sys::CUresult::CUDA_SUCCESS || st != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    }

    /// The `x` the combine kernels dot with `moe.sg`: sc.x, or — FP4L stored-bf16 gate (sg_bf) — the gate channels
    /// xf_sg_gate wrote this step (moe.sg then holds the channel constants). Every combine site reads it here.
    pub(crate) fn gate_x(&self, sc: &Scratch, moe: &MoeLayer) -> u64 {
        match (moe.sg_bf.as_ref(), sc.fxs.as_ref()) {
            (Some(_), Some(f)) => *f.xg.device_ptr() as u64,
            _ => *sc.x.device_ptr() as u64,
        }
    }

    /// The prefill twin of gate_x (psc.fxg after xf_sg_gate over the chunk's rows).
    pub(crate) fn gate_x_rows(&self, psc: &PrefillScratch, moe: &MoeLayer) -> u64 {
        match (moe.sg_bf.as_ref(), psc.fxg.as_ref()) {
            (Some(_), Some(g)) => *g.device_ptr() as u64,
            _ => *psc.x.device_ptr() as u64,
        }
    }

    pub fn forward_step(&self, sc: &mut Scratch, toks: &[i32], pos_host: &[usize], slots: &[usize],
                        mut ladder: Option<&mut LadderCtx>) -> Result<Vec<i32>> {
        let m = toks.len();
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let cfg = &self.tc; // TP-B: the trunk's per-rank geometry
        let h = cfg.hidden_size;
        let hcn = cfg.hc_count.max(1);
        let kd = cfg.lin_k_dim;
        let vd = cfg.lin_v_dim;
        let nh = cfg.lin_num_v_heads;
        let nk = cfg.lin_num_k_heads;
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let ck = cfg.conv_kernel;

        // S-A3-e FIX: htod_copy_into asserts HOST.len == DST.len and the serve path runs
        // m < width (one request prefills in its slot on a width-8 scratch). Stage with
        // fresh exact-size allocations instead - every kernel grid is m-sized and all
        // state addressing is slot-based for b < m, so nothing reads beyond m.
        // S-A3-f Item 1: persistent width-sized staging buffers (htod_copy_into wants
        // exact host len); lanes >= m stay zero — every consumer grid is m-sized and all
        // state addressing is slot-based, so padded lanes are never read.
        let w = sc.toks.len();
        // WP02: a row past the window writes K/V (and reads cos_tab) out of bounds — refuse it.
        anyhow::ensure!(pos_host.iter().all(|&p| p < self.max_pos),
                        "forward_step: position {:?} outside the KV window (max_pos {})", pos_host, self.max_pos);
        let mut toks_v = toks.to_vec();
        toks_v.resize(w, 0);
        self.dev.htod_copy_into(toks_v, &mut sc.toks)?;
        let pos_h: Vec<i32> = pos_host.iter().map(|p| *p as i32).collect();
        let mut pos_v = pos_h.clone();
        pos_v.resize(w, 0);
        self.dev.htod_copy_into(pos_v, &mut sc.pos)?;
        let mut slotpos: Vec<i32> = vec![0; w * 2];
        let mut sids_v: Vec<i32> = vec![0; w];
        for (i, pp) in pos_host.iter().enumerate() {
            slotpos[i * 2] = slots[i] as i32;
            slotpos[i * 2 + 1] = *pp as i32;
            sids_v[i] = slots[i] as i32;
        }
        self.dev.htod_copy_into(slotpos, &mut sc.slots)?;
        self.dev.htod_copy_into(sids_v, &mut sc.slot_ids)?;
        // S-A3-i: the graph is keyed by (width, regime) — the dense/sparse branch is a
        // capture-time property; each regime captures once per width on its first step.
        let qsa = self.qsa_live(pos_host.iter().copied().max().unwrap_or(0));
        let (qg, qb) = self.qsa_grid_bucket_key(pos_host.iter().copied().max().unwrap_or(0));
        sc.qsa_grid_nblk = qg;
        // S-B9-REL-QSAREGIME: the regime of a ROW is a function of its own position. In a sparse
        // step the leading `md` rows at/below qsa_limit keep the dense kernels (what they run alone
        // or next to other short rows), so a lane's bits no longer depend on a longer lane sharing
        // its step. The serve orders short lanes first (regime_order); a short row behind a long
        // one stays on the sparse kernels and is reported once.
        let md = if qsa && row_regime_on() {
            // the step's rows are the first m positions (a probe may pass a longer position list)
            let rows = &pos_host[..m.min(pos_host.len())];
            let (md, late) = dense_prefix(rows, self.qsa_limit());
            row_regime_note(md, rows.len() - md - late, late);
            md
        } else { 0 };
        // WP15: a penalized step is its own graph (bit 40), so unpenalized steps replay the
        // pre-WP15 graphs untouched. WP04: the base key is gkey_of (shared with the precapture,
        // which runs unpenalized: pen_live is false at boot). QSAREGIME: a mixed step is its own
        // graph per dense-row count (mixed_key_bits; 0 = the pre-existing keys, bit for bit).
        let gkey = gkey_of(m, qsa, qb) | mixed_key_bits(md) | ((sc.pen_live as usize) << 40) | sk(sc) // TP-C: + the scratch id
            | self.vp_key(sc, m) // TP-H #4: the vocab-parallel greedy tail = its own graph
            | vmask_key(sc); // W2-EXL3: a schema-masked step = its own graph (bit 47)

        // RoPE: gather this step's rows from the load-time table. The table was built
        // with the SAME host powf/cos/sin code the old per-step loop ran, so the bits
        // are identical; the gather is a plain row copy.
        let rdim = cfg.rotary_dim;
        let cs_stride = (2 * rdim) as i32;
        { let _ = cs_stride; // VIS-2: the mrope-aware gather (text-only slots: cos_tab[pos], unchanged)
        let slots_src = SlotSrc::Pairs(*sc.slots.device_ptr() as u64);
        self.cos_gather(&l, &mut sc.cos, *sc.pos.device_ptr() as u64, slots_src, m)?; }
        // S-A3-f Item 1b: PLE host half (ring + embedding bits) — pure host work plus a
        // padded upload into the persistent buffer; OUTSIDE the graphed kernel region.
        if self.ple.is_some() && crate::opts::var(crate::opt!("exl3-no-ple")).is_err() {
            let mut emb_bits = vec![0u16; w * 2560];
            self.ple_build_embed(m, toks, slots, &mut emb_bits)?;
            self.dev.htod_copy_into(emb_bits, &mut sc.ple_emb)?;
        }


        // TP-C (rung 4): graphs under TP. K1/K2 take only the ctx pointer and derive the epoch on
        // device, so a replay advances the protocol without re-capture; every doorbell sits on the
        // serial spine after its fork's join (design §2.3; audit in PLAN/notes_2026-09-28/TP-C.md).
        // Capture executes nothing (no barrier runs during a capture), eager and replay run the
        // identical barrier sequence, so a rank that fails a capture stays in lockstep eagerly.
        let want_graph = ladder.is_none()
            && crate::opts::var(crate::opt!("exl3-route-check")).is_err()
            && crate::opts::var(crate::opt!("exl3-no-graph")).is_err()
            && !wp12_xcheck();
        let dump_active = !want_graph
            && crate::opts::var(crate::opt!("mtp-dump")).as_deref().map_or(false, |s| !s.is_empty());
        let graphed = want_graph && self.graphs.lock().unwrap().contains_key(&gkey);
        if autotune::armed() && autotune::li_step(self, sc, m, qsa, qb)? {
            // TUNE T1 (§5.3): the armed --autotune harness ran this step from its LI arm's graph
            // (bitwise twins: the trajectory is the same whichever arm runs)
        } else if graphed {
            self.graphs.lock().unwrap().get(&gkey).unwrap().launch();
        } else {
            self.forward_step_kernels(&l, sc, m, ladder.as_deref_mut(), dump_active, qsa, md)?;
        }
        self.dsync()?;
        let ids = self.dtoh(&sc.argmax)?;
        if let Some(ld) = ladder.as_deref_mut() {
            if ld.active() {
                ld.dump_u16(&self.dev, &sc.logits, "logits", self.cfg.vocab_size)?;
            }
            ld.step += 1;
        }
        // MTP_DIAG (plain path, graph-safe): with --mtp-dump set, run the DUMP
        // hook pass after the step; never inside a graph capture (host file IO).
        if crate::opts::var(crate::opt!("mtp-dump")).as_deref().map_or(false, |s| !s.is_empty()) {
            if !want_graph {
                self.diag_plain_dump(&l, sc, m)?;
            }
        }
        // S-A3-f Item 1b: first (warmup) step at this width ran eager and its outputs are
        // already read back above — record the same kernel sequence into a graph for the
        // next step. Capture failure is loud and leaves this engine eager (never wrong).
        if want_graph && !graphed {
            self.capture_step_graph(&l, sc, m, gkey, qsa, md);
        }
        // W2-EXL3: the mask is staged per step (one-shot). Cleared only AFTER the capture above:
        // the graph recorded under a bit-47 key must contain the mask node it is keyed for.
        sc.vmask_live = false;
        Ok(ids)
    }

    /// S-A3-f Item 1b: the decode-step kernel sequence (embed gather → layers → final
    /// mixer → lm_head → argmax). Launches ONLY — no uploads, no host reads, no
    /// allocations — so it is legal to run eager, or record into a CUDA graph.
    /// `md` (S-B9-REL-QSAREGIME): in a sparse step, the leading rows at/below qsa_limit that keep
    /// the dense attention kernels (0 = none; always 0 in a dense step).
    pub(super) fn forward_step_kernels(&self, l: &Launcher, sc: &mut Scratch, m: usize,
                            mut ladder: Option<&mut LadderCtx>, diag: bool, qsa: bool, md: usize) -> Result<()> {
        let _tc = tune::enter(tune::Ctx::fam(tune::Fam::Step, m).regime(qsa)); // TUNE family scope
        let cfg = &self.tc; // TP-B: the trunk's per-rank geometry
        let h = cfg.hidden_size;
        let hcn = cfg.hc_count.max(1);
        let kd = cfg.lin_k_dim;
        let vd = cfg.lin_v_dim;
        let nh = cfg.lin_num_v_heads;
        let nk = cfg.lin_num_k_heads;
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let ck = cfg.conv_kernel;
        // WP12 XCHECK: eager only (never inside a capture — it syncs and reads back)
        let wp12_xchk = wp12_xcheck() && !self.wp12_capturing();
        // WP27: graph class of this body + the early PLE prefix (see ple_prefix).
        sc.wp27_cls = crate::exl3_wp27::CLS_STEP;
        let ple_early = self.wp27_ple_early();
        let mut wp27_ple = crate::exl3_wp27::Cap::none();
        let mut wp27_ple_tail = crate::exl3_wp27::Tail::default();
        self.embed_gather(l, *sc.resid.device_ptr() as u64, *sc.toks.device_ptr() as u64, m)?;
        if let Some(ld) = ladder.as_deref_mut() {
            if ld.active() {
                ld.dump(&self.dev, &sc.resid, "embed", m)?;
            }
        }
        // S-A3-f-d Item 1b (MTP_DIAG plain side): when --mtp-dump is set and this
        // step runs EAGER (diag=true — never during graph capture), mirror the
        // verify_kernels dump set at the SAME buffer+point so the A/B diff is 1:1
        // with no offsets. Pure dtoh + file IO between launches; col 0 = row-0
        // slice (w = len/m).
        let pd = if diag {
            crate::opts::var(crate::opt!("mtp-dump")).ok().filter(|s| !s.is_empty())
                .map(std::path::PathBuf::from)
        } else {
            None
        };
        let pd_f32 = |d: &std::path::PathBuf, buf: &CudaSlice<f32>, name: &str| -> Result<()> {
            let host = self.dtoh(buf)?;
            let w = buf.len() / m;
            let mut f = std::fs::File::create(format!("{}/{}_p.f32", d.display(), name))?;
            for &v in &host[..w.min(host.len())] { f.write_all(&v.to_le_bytes())?; }
            Ok(())
        };
        let pd_u16 = |d: &std::path::PathBuf, buf: &CudaSlice<u16>, name: &str| -> Result<()> {
            let host = self.dtoh(buf)?;
            let w = buf.len() / m;
            let mut f = std::fs::File::create(format!("{}/{}_p.f32", d.display(), name))?;
            for &b in &host[..w.min(host.len())] {
                f.write_all(&f16::from_bits(b).to_f32().to_le_bytes())?;
            }
            Ok(())
        };
        if let Some(d) = pd.as_ref() {
            self.dsync()?;
            pd_f32(d, &sc.resid, "stagingCTX")?;
            // MTP_DIAG: the live GDN/conv state row this step starts from (layer 0).
            let host = self.dtoh(&self.s_state[0])?;
            let mut f = std::fs::File::create(format!("{}/g0sin_p.f32", d.display()))?;
            for &v in &host { f.write_all(&v.to_le_bytes())?; }
            let hostc = self.dtoh(&self.conv_state[0])?;
            let mut f2 = std::fs::File::create(format!("{}/g0cin_p.f32", d.display()))?;
            for &v in &hostc { f2.write_all(&v.to_le_bytes())?; }
        }

        let mut gdn_i = 0usize;
        let mut att_i = 0usize;
        // HC (w4): deferred injects (xq_hc_inj_norm) unless a ladder/dump/poison diagnostic touches
        // sc.resid or sc.inj between an hc_post and the next hc_pre; flushed before the PLE chain.
        let hc_def = self.hc_defer() && ladder.is_none() && pd.is_none()
            && crate::opts::var(crate::opt!("exl3-poison")).is_err();
        let mut hc_pend: Option<u64> = None;
        for (li, layer) in self.layers.iter().enumerate() {
            if li == 1 && self.ple.is_some() { self.hc_flush(&l, sc, m, &mut hc_pend)?; }
            // MTP_DIAG: same point as verify_kernels (loop top, pre-hc_pre) — 1:1 stage.
            if let Some(d) = pd.as_ref() {
                self.dsync()?;
                pd_u16(d, &sc.x, &format!("xpre{li}"))?;
            }
            if li == 1 && self.ple.is_some() {
                // S-A3-e NaN-poison probe (--exl3-poison): fill pure-activation scratch with
            // fp16 NaN so any read of a row >= m (stale data from a smaller previous batch)
            // surfaces as NaN logits -> deterministic argmax 0, instead of plausible drift.
            // Accumulators (chain_yraw, y*_raw) are skipped: they are read-modify-write.
            if crate::opts::var(crate::opt!("exl3-poison")).is_ok() {
                let g16 = |n: usize| (((n + 255) / 256).max(1) as u32, 1, 1);
                for buf in [
                    &sc.x, &sc.qkv, &sc.qg, &sc.zbuf, &sc.a_out, &sc.b_out, &sc.core,
                    &sc.mixer_out, &sc.moe_out, &sc.ysh, &sc.yd, &sc.xh_e,
                ] {
                    let n = buf.len();
                    xqlaunch!(l, "xq_memset_u16", g16(n), (256, 1, 1), 0,
                              (buf, n as i64, 0x7E00i32))?;
                }
                {
                    let n = sc.inj.len();
                    xqlaunch!(l, "xq_memset_f32", g16(n), (256, 1, 1), 0,
                              (&mut sc.inj, 0i64, n as i64))?;
                }
            }
                if crate::opts::var(crate::opt!("exl3-no-ple")).is_err() {
                    wp27_ple.join(&wp27_ple_tail)?; // WP27: the early PLE prefix branch joins here
                    self.ple_kernels(&l, sc, m, ple_early)?;
                }
            }
            self.hc_pre_p(&l, sc, &layer.hc_attn, m, hc_pend.take())?;
            self.tp_pf_hint(&layer.hc_mlp, m);   // TP-4Z2: the mixer-out reduce's K2 prefetches the mlp-site hc weights
            if let Some(ld) = ladder.as_deref_mut() {
                if ld.want(li) {
                    self.dsync()?;
                    ld.dump_u16(&self.dev, &sc.x, &format!("xpre{li}"), (m * cfg.hidden_size))?; // sc.x = mixed GEMM input [m,h]
                }
            }
            match &layer.mixer {
                Mixer::Gdn(g) => {
                    // WP27 GDN fork: the in_z chain (its own scratch) runs beside conv -> scan
                    // (it starts after in_qkv, off its bandwidth window) and joins before the
                    // gate (zbuf); a|b runs beside in_qkv and joins before the scan (b_out/a_out).
                    let zdup = crate::exl3_wp27::fork_on(crate::exl3_wp27::FORK_GDN) && sc.wp27_z_fits(&g.in_z, m);
                    let mut wp27 = if zdup {
                        crate::exl3_wp27::Cap::open(l.stream.stream, sc.wp27_cls, crate::exl3_wp27::FORK_GDN)
                    } else { crate::exl3_wp27::Cap::none() };
                    exl3_chain(&l, &g.in_qkv, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.qkv, m, &sc.chain_ws)?;
                    let wp27_q = wp27.mark()?; // in_qkv tail: the z chain and the conv both hang off it
                    wp27.side_begin()?;
                    if zdup {
                        exl3_chain(&l, &g.in_z, &sc.x, &mut sc.z_xh, &mut sc.z_yraw, &mut sc.zbuf, m, &sc.z_ws)?;
                    } else {
                    exl3_chain(&l, &g.in_z, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.zbuf, m, &sc.chain_ws)?;
                    }
                    wp27.side_end()?;
                    let wp27_z = wp27.cut()?; // a|b restarts from the fork base
                    // S-A3-f-f ITEM 1: a/b projections — warp-per-output vectorized GEMV.
                    // Was ONE block of 64 threads for K=2560 (72 launches @ 7.1 GB/s).
                    // WP12 ab: both in ONE launch (bitwise per output; --wp12-off=ab = two).
                    if let Some(ab) = g.ab_fx.as_deref() {
                        fp4l_fwd::fx_ab(&l, ab, &sc.x, &mut sc.a_out, &mut sc.b_out, m)?; // FP4L: exact MXFP8
                    } else if wp12_on(WP12_AB) {
                        xqlaunch!(l, "xq_gemm_f16_v_ab", (12u32, m as u32, 1), (256, 1, 1), 0,
                                 (&mut sc.a_out, &mut sc.b_out, &g.a_w, &g.b_w, &sc.x, m as i32, nh as i32, h as i32))?;
                        if wp12_xchk { self.wp12_xcheck_ab(&l, sc, g, m, "step a|b")?; }
                    } else {
                    xqlaunch!(l, "xq_gemm_f16_v", (6u32, m as u32, 1), (256, 1, 1), 0,
                             (&mut sc.a_out, &g.a_w, &sc.x, m as i32, nh as i32, h as i32))?;
                    xqlaunch!(l, "xq_gemm_f16_v", (6u32, m as u32, 1), (256, 1, 1), 0,
                             (&mut sc.b_out, &g.b_w, &sc.x, m as i32, nh as i32, h as i32))?;
                    }
                    let wp27_ab = wp27.cut_to(&wp27_q)?; // WP27: resume from the in_qkv tail
                    if let Some(ld) = ladder.as_deref_mut() {
                        if ld.want_steps(li) {
                            self.dsync()?;
                            ld.dump_u16(&self.dev, &sc.qkv, &format!("g{li}_qkv"), m * conv_dim)?;
                            ld.dump_u16(&self.dev, &sc.b_out, &format!("g{li}_b"), m * nh)?;
                            ld.dump_u16(&self.dev, &sc.a_out, &format!("g{li}_a"), m * nh)?;
                        }
                    }
                    xqlaunch!(l, "xq_conv1d", (((conv_dim * m) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                             (&mut sc.qkv, (*self.conv_state[gdn_i].device_ptr()) as u64, &g.conv_w,
                              conv_dim as i32, ck as i32, m as i32, &sc.slots, conv_dim as i32))?;
                    wp27.join(&wp27_ab)?; // WP27: the scan reads b_out/a_out
                    // MTP_DIAG layer-0 bisect: pre/post-conv planes (live state).
                    if li == 0 {
                        if let Some(d) = pd.as_ref() {
                            self.dsync()?;
                            pd_u16(d, &sc.qkv, "l0conv")?;
                        }
                    }
                    if let Some(ld) = ladder.as_deref_mut() {
                        if ld.want_steps(li) {
                            self.dsync()?;
                            ld.dump_u16(&self.dev, &sc.qkv, &format!("g{li}_conv"), m * conv_dim)?;
                            ld.dump_u16(&self.dev, &sc.zbuf, &format!("g{li}_z"), m * vd)?;
                        }
                    }
                    let nchunk = vd / 32;
                    let smem = (kd * 33 + kd + 32 * 3 + kd * 2) * 4;
                    if wp12_on(WP12_STEP) && kd == 128 && vd == 128 {
                        // WP12 step: the verify chunk's register-resident device body at C = 1
                        // (AGENTS §2.8), grid m*nh x 128 threads; bitwise == xq_gdn_step.
                        // XCHECK: snapshot the live plane first, replay the old kernel on it after.
                        let twin = if wp12_xchk { Some(self.wp12_state_twin(&l, gdn_i)?) } else { None };
                        xqlaunch!(l, "xq_gdn_step_r", ((m * nh) as u32, 1, 1), (128, 1, 1), 0,
                                 (&mut sc.core, &sc.qkv, (*self.s_state[gdn_i].device_ptr()) as u64,
                                  &sc.b_out, &sc.a_out, nh as i32, nk as i32, kd as i32, vd as i32,
                                  &g.a_log, &g.dt_bias, &sc.slots))?;
                        if let Some(st2) = twin { self.wp12_xcheck_decode_step(&l, sc, g, gdn_i, m, st2)?; }
                    } else {
                    xqlaunch!(l, "xq_gdn_step", ((m * nh * nchunk) as u32, 1, 1), (kd as u32, 1, 1), smem as u32,
                             (&mut sc.core, &sc.qkv, (*self.s_state[gdn_i].device_ptr()) as u64,
                              &sc.b_out, &sc.a_out, nh as i32, nk as i32, kd as i32, vd as i32,
                              &g.a_log, &g.dt_bias, &sc.slots))?;
                    }
                    if li == 0 {
                        if let Some(d) = pd.as_ref() {
                            self.dsync()?;
                            pd_u16(d, &sc.core, "l0core")?;
                        }
                    }
                    if let Some(ld) = ladder.as_deref_mut() {
                        if ld.want_steps(li) {
                            self.dsync()?;
                            ld.dump_u16(&self.dev, &sc.core, &format!("g{li}_core"), m * vd)?;
                        }
                    }
                    wp27.join(&wp27_z)?; // WP27: the gate reads zbuf
                    if cfg.gdn_gate_sigmoid {
                    xqlaunch!(l, "xq_gdn_gate", ((m * nh) as u32, 1, 1), (vd as u32, 1, 1), (vd * 4) as u32,
                             (&mut sc.normed, &sc.core, &sc.zbuf, &g.norm, vd as i32, nh as i32, m as i32, cfg.rms_eps))?;
                    } else { // EXL27B: Qwen3.5-class gated norm, silu(z) (a kernel copy)
                    xqlaunch!(l, "xq_gdn_gate_silu", ((m * nh) as u32, 1, 1), (vd as u32, 1, 1), (vd * 4) as u32,
                             (&mut sc.normed, &sc.core, &sc.zbuf, &g.norm, vd as i32, nh as i32, m as i32, cfg.rms_eps))?;
                    }
                    if li == 0 {
                        if let Some(d) = pd.as_ref() {
                            self.dsync()?;
                            pd_u16(d, &sc.normed, "l0gate")?;
                        }
                    }
                    if let Some(ld) = ladder.as_deref_mut() {
                        if ld.want_steps(li) {
                            self.dsync()?;
                            ld.dump_u16(&self.dev, &sc.normed, &format!("g{li}_gate"), m * vd)?;
                        }
                    }
                    self.chain_o(l, &g.out_proj, &sc.normed, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.mixer_out, m, &sc.chain_ws)?; // TP-B: row-parallel -> all-reduce
                    if li == 0 {
                        if let Some(d) = pd.as_ref() {
                            self.dsync()?;
                            pd_u16(d, &sc.mixer_out, "l0yatt")?;
                        }
                    }
                    gdn_i += 1;
                }
                Mixer::Attn(a) => {
                    // q, k|v (+ the indexer qk) — W4/DENSE: one launch; else the old chains
                    let idx_pre = self.attn_in_chains(l, sc, a, m, true)?;
                    let nh_p = (cfg.num_heads as i32) | ((cfg.num_kv_heads as i32) << 16);
                    let hd_p = (cfg.head_dim as i32) | ((cfg.rotary_dim as i32) << 16);
                    // S-A3-i: past the dense threshold the model's own attention is the
                    // QSA selection; below it this dense kernel IS the spec (bit-identical
                    // to the pre-port engine, graphs included).
                    if qsa && a.idx.is_some() {
                        self.qsa_attn_step(l, sc, a, att_i, m, idx_pre, false)?;   // rows = lanes (own slots)
                        if md > 0 {
                            // S-B9-REL-QSAREGIME: rows 0..md are at/below qsa_limit — their attention
                            // is the dense kernels', exactly as in a dense step. The launches above
                            // already appended every row's indexer keys and wrote every row's K/V
                            // words (the dense prep rewrites the same words for these rows). The dense
                            // kernels read sc.qg / kp / vp / cos / slots, which the sparse launches
                            // leave untouched; of what a later launch reads they replace rows 0..md
                            // of sc.normed only (their own sc.qstage rows and the dv3 score planes
                            // are scratch nothing after this attention reads; the next layer re-preps).
                            self.dense_attn_step(l, sc, a, att_i, md, nh_p, hd_p)?;
                        }
                    } else {
                    self.qsa_keys_append_dense(l, sc, a, att_i, m, idx_pre)?;
                    self.dense_attn_step(l, sc, a, att_i, m, nh_p, hd_p)?;
                    }
                    self.chain_o(l, &a.o_proj, &sc.normed, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.mixer_out, m, &sc.chain_ws)?; // TP-B: row-parallel -> all-reduce
                    att_i += 1;
                }
            }
            let yatt_p = *sc.mixer_out.device_ptr() as u64;
            self.hc_post_d(&l, sc, yatt_p, m, hc_def, &mut hc_pend)?; // attn site: inject mixer_out
            if li == 0 {
                if let Some(d) = pd.as_ref() {
                    self.dsync()?;
                    pd_f32(d, &sc.resid, "l0rattn")?;
                }
            }
            if let Some(ld) = ladder.as_deref_mut() {
                if ld.want_steps(li) {
                    self.dsync()?;
                    ld.dump(&self.dev, &sc.resid, &format!("rmid{li}"), m)?;
                    ld.dump_u16(&self.dev, &sc.mixer_out, &format!("yatt{li}"), m * h)?;
                }
            }
            self.hc_pre_p(&l, sc, &layer.hc_mlp, m, hc_pend.take())?; // re-mix from the UPDATED residual for the mlp site
            if let Some(ld) = ladder.as_deref_mut() {
                if ld.want_steps(li) {
                    self.dsync()?;
                    ld.dump_u16(&self.dev, &sc.x, &format!("xpre_mlp{li}"), m * h)?;
                    ld.dump(&self.dev, &sc.inj, &format!("inj{li}"), m)?;
                }
            }
            if li == 0 && ple_early {
                // WP27: the PLE prefix (reads only sc.ple_emb) as a sibling branch of layer 0's
                // MoE window; joined at the layer-1 PLE boundary (no hc barrier in between).
                wp27_ple = crate::exl3_wp27::Cap::open(l.stream.stream, sc.wp27_cls, crate::exl3_wp27::FORK_PLE);
                wp27_ple.side_begin()?;
                self.ple_prefix(&l, sc, m)?;
                wp27_ple.side_end()?;
                wp27_ple_tail = wp27_ple.cut()?;
            }
            self.tp_pf_hint(self.layers.get(li + 1).map_or(&self.hc_mixer, |nl| &nl.hc_attn), m);   // TP-4Z2: the MoE reduce's K2 prefetches the next hc mixer's weights
            self.moe(&l, sc, &layer.moe, m)?;
            if li == 0 {
                if let Some(d) = pd.as_ref() {
                    self.dsync()?;
                    pd_u16(d, &sc.moe_out, "l0ymlp")?;
                }
            }
            // TP-A (G-T1-b): the replicated router logits + the post-all-reduce MoE output.
            if let Some(ld) = ladder.as_deref_mut() {
                if ld.want(li) && ld.mem.is_some() {
                    self.dsync()?;
                    ld.dump(&self.dev, &sc.logits_r, &format!("rlog{li}"), m)?;
                    ld.dump_u16(&self.dev, &sc.moe_out, &format!("ymlp{li}"), m * h)?;
                    // TP-B: the mixer output after its row-parallel all-reduce (replicated)
                    ld.dump_u16(&self.dev, &sc.mixer_out, &format!("yatt{li}"), m * h)?;
                }
            }
            if let Some(ld) = ladder.as_deref_mut() {
                if ld.want_steps(li) {
                    self.dsync()?;
                    ld.dump_u16(&self.dev, &sc.moe_out, &format!("ymlp{li}"), m * h)?;
                    ld.dump_u16(&self.dev, &sc.ysh, &format!("ysh{li}"), m * h)?;
                    ld.dump_u16(&self.dev, &sc.yd, &format!("yd{li}"), m * self.cfg.num_experts_per_tok * h)?;
                }
            }
            let ymlp_p = *sc.moe_out.device_ptr() as u64;
            self.hc_post_d(&l, sc, ymlp_p, m, hc_def, &mut hc_pend)?; // mlp site: inject moe_out (was stale mixer_out — root cause)
            // EXL27C-DF2 (fwd): the DFlash2 tap - hidden_states[li+1] (the post-MLP-inject
            // residual, sc.resid's stream-0 slice) into the round's 8-column sink, mirroring
            // the NVFP4 trunk's S4F hook (gpu.rs forward_batch_dev: `n <= block()` gate,
            // TAP_LAYERS position, cols [0, m)). The 27B never defers the inject (hc_defer()
            // needs hidden <= 4096) but flush anyway so the tap reads the materialized
            // residual under any config - the flush is bitwise the fused path (w4 note).
            // Graph-safe: the sink staging and sc.resid addresses are fixed per graph key.
            // Zero-cost when the sink is unarmed. Under TP every rank holds the full token
            // rows, so the taps are rank-identical (the trunk captures on all ranks too).
            if self.df2_sink.is_some() && m <= crate::dflash2::block()
                && self.cfg.hidden_size == crate::dflash2::geom().hidden {
                if let Some(kk) = crate::dflash2::geom().tap_layers.iter().position(|&x| x == li) {
                    if hc_pend.is_some() { self.hc_flush(&l, sc, m, &mut hc_pend)?; }
                    crate::dflash2::capture::capture_cols_f32(
                        &self.dev, l.stream, self.df2_sink.as_ref().unwrap(),
                        *sc.resid.device_ptr() as u64,
                        self.cfg.hc_count.max(1) * self.cfg.hidden_size,
                        self.cfg.hidden_size, self.cfg.hc_count.max(1), m, kk)?;
                }
            }
            if li == 0 {
                if let Some(d) = pd.as_ref() {
                    self.dsync()?;
                    pd_f32(d, &sc.resid, "l0rmlp")?;
                }
            }
            if let Some(ld) = ladder.as_deref_mut() {
                if ld.want(li) {
                    self.dsync()?;
                    ld.dump(&self.dev, &sc.resid, &format!("layer{li}"), m)?;
                }
            }
            // LEG-B 166 debug (2026-10-30): per-layer stream dump for the wheel cross-check.
            // Same gate as the other pd dumps (eager + --mtp-dump); files are overwritten per
            // step, so a multi-step run leaves the LAST step's streams — compare row m-1 vs the
            // wheel taps. resid = hc streams (wheel TransformerBlock output class).
            if let Some(d) = pd.as_ref() {
                self.dsync()?;
                pd_f32(d, &sc.resid, &format!("lay{li:03}_resid"))?;
            }
        }

        if let Some(ld) = ladder.as_deref_mut() {
            if ld.active() {
                self.dsync()?;
                ld.dump(&self.dev, &sc.resid, "precollapse", m)?;
            }
        }
        // final mixer: streams -> hidden (no inject); sc.x holds the hidden after hc_pre
        self.hc_pre_p(&l, sc, &self.hc_mixer, m, hc_pend.take())?;
        // MTP_DIAG: same buffer+point as verify's "finalmix" (sc.resid AFTER the final
        // hc_pre, f32) — 1:1 comparable with finalmix_s000.
        if let Some(d) = pd.as_ref() {
            self.dsync()?;
            pd_f32(d, &sc.resid, "finalmix")?;
        }
        if let Some(ld) = ladder.as_deref_mut() {
            if ld.active() {
                let hh = h;
                ld.dump_u16(&self.dev, &sc.x, "finalmix", hh)?;
            }
        }
        let v = cfg.vocab_size;
        if self.vp_greedy(sc, m) && ladder.is_none() && !diag && !sc.vmask_live {
            // TP-H #4: vocab-parallel greedy tail (see verify_kernels). The ladder / MTP_DUMP
            // diagnostics read the full logits row, so they keep the replicated head (never graphed).
            self.vp_greedy_tail(&l, sc, m)?;
            return Ok(());
        }
        if self.vp_gather(sc, m) && ladder.is_none() && !diag {
            // TP-H2 (--tp-vp-sampled): shard GEMM + row all-gather -> sc.logits rows 0..m = the replicated
            // head's rows bit for bit; the penalty / argmax / sampler below run unchanged on them.
            self.vp_gather_tail(&l, sc, m)?;
        } else {
        exl3_chain(&l, &self.lm_head, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.logits, m, &sc.chain_ws)?;
        }
        // WP15: penalties before BOTH the argmax and the sampler (only in pen_live graphs).
        self.pen_rows(l, sc, m)?;
        // W2-EXL3: the JSON-schema mask after the penalties, before BOTH the argmax and the
        // sampler (only in vmask_live graphs; unmasked steps launch nothing here).
        self.vmask_rows(l, sc, m)?;
        // WP09: one multi-block launch for all m rows (bitwise the same ids; escape inside).
        self.argmax_rows(l, sc, m, v)?;
        // API-parity G2: sampled rows overwrite their id (zero-param rows exit at once).
        xqlaunch!(l, "xq_sample_rows", (m as u32, 1, 1), (1024, 1, 1), 0,
                 (&mut sc.argmax, &sc.logits, v as i32, &sc.samp))?;
        Ok(())
    }

    /// WP15: `xq_pen_rows` over the m logits rows (decode lanes or verify rows; the row -> slot /
    /// position / input-token tables are sc.slots / sc.toks). No launch unless penalties are
    /// staged, so unpenalized steps are the pre-WP15 kernel sequence exactly.
    pub(super) fn pen_rows(&self, l: &Launcher, sc: &mut Scratch, m: usize) -> Result<()> {
        if !sc.pen_live {
            return Ok(());
        }
        let v = self.cfg.vocab_size;
        xqlaunch!(l, "xq_pen_rows", (v.div_ceil(PEN_SPAN) as u32, m as u32, 1), (256, 1, 1), 0,
                  (&mut sc.logits, v as i32, &sc.pen, &sc.toks, &sc.slots, &mut sc.pen_hist,
                   PEN_RING_LOG2 as i32))?;
        Ok(())
    }

    /// W2-EXL3: `xq_vmask_rows` over the m logits rows: a row whose flag is set keeps only the
    /// tokens whose bit is set in its allowed bitset (every other logit becomes -inf). No launch
    /// unless a mask is staged for this step.
    fn vmask_rows(&self, l: &Launcher, sc: &mut Scratch, m: usize) -> Result<()> {
        if !sc.vmask_live {
            return Ok(());
        }
        let v = self.cfg.vocab_size;
        let nw = vmask_words(v);
        xqlaunch!(l, "xq_vmask_rows", (v.div_ceil(VMASK_SPAN) as u32, m as u32, 1), (256, 1, 1), 0,
                  (&mut sc.logits, v as i32, &sc.vmask, nw as i32, &sc.vmask_flags))?;
        Ok(())
    }

    /// S-A3-f Item 1b: record forward_step_kernels at width m into a CUDA graph on the
    /// model's blocking stream (gpu.rs CudaGraph wrapper — the DSV4 capture mechanism).
    pub(super) fn capture_step_graph(&self, l: &Launcher, sc: &mut Scratch, m: usize, gkey: usize, qsa: bool, md: usize) {
        use cudarc::driver::sys;
        let raw = self.stream.stream;
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(raw, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            crate::metrics::exl3_frac_capture_reset(); // K35: this capture's fractional-expert round tally starts at 0
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::drprintln!("EXL3_GRAPH: BeginCapture failed ({r:?}) at width {m} — staying eager");
                crate::metrics::graph_capture_lost("step", &format!("width {m}, qsa={qsa}, mixed rows={md}"), &format!("{r:?}"));
                return;
            }
        }
        let cap = self.forward_step_kernels(l, sc, m, None, false, qsa, md);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r2 = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
        if r2 != sys::CUresult::CUDA_SUCCESS || cap.is_err() {
            crate::drprintln!("EXL3_GRAPH: capture failed (end={r2:?}, body={:?}) at width {m} — staying eager", cap.err());
            crate::metrics::graph_capture_lost("step", &format!("width {m}, qsa={qsa}, mixed rows={md}"), &capture_lost_status(r2));
            return;
        }
        crate::exl3_wp27::finish(graph, "step", m); // WP27: node priorities + DAG receipt
        if !self.tp_spine_audit(graph, "step", m) { unsafe { sys::cuGraphDestroy(graph); } return; } // TP-C: barriers on the spine
        unsafe {
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            let r3 = crate::exl3_wp27::instantiate(&mut exec, graph); // WP27 (= cuGraphInstantiate_v2 unless --wp27-prio=1)
            sys::cuGraphDestroy(graph);
            if r3 == sys::CUresult::CUDA_SUCCESS {
                precap_upload(exec, raw);
                crate::metrics::set_graph_entries({ let mut g = self.graphs.lock().unwrap(); g.insert(gkey, crate::gpu::CudaGraph::from_exec(exec, raw).with_frac_rounds(crate::metrics::exl3_frac_capture_take())); g.len() as u64 });
                // S-B9-REL-SMALLFIXES 9a(iii): the mixed-step graphs are lazy and uncapped — count
                // them (gauge + the scheduler's idle line) and say so once, at the first capture.
                if md > 0 && crate::metrics::mixed_step_graph_captured() == 1 {
                    crate::drprintln!("[exl3] per-row attention regime: first mixed-step graph captured (width {m}, {md} dense row(s)). \
One is captured lazily per step shape seen (width, dense rows, sparse grid bucket); they are not capped. Count: \
velogb10_graph_cache_mixed_step_entries on /metrics and the scheduler's idle line");
                }
                if graph_log_on() { crate::drprintln!("EXL3_GRAPH: captured decode step at width {m}{}",
                                                      if md > 0 { format!(" (mixed regime: {md} dense rows)") } else { String::new() }); }
            } else {
                crate::drprintln!("EXL3_GRAPH: instantiate failed ({r3:?}) at width {m} — staying eager");
                crate::metrics::graph_capture_lost("step", &format!("width {m}, qsa={qsa}, mixed rows={md}"), &format!("instantiate {r3:?}"));
            }
        }
    }
}


impl FwdModel {
    /// S-A3-e — one MTP draft-head pass for m lanes (M1, EXL3 weight path).
    ///
    /// INPUT contract: `sc.resid` holds the per-lane stream tap [m, hc, h] f32 — the trunk's
    /// pre-collapse stack after the last committed step (or the head's own previous output
    /// when chaining drafts). `toks` = the token whose embedding enters (the committed token
    /// for the first draft step, then the previously drafted id). `pos_host` = per-lane
    /// ABSOLUTE main position of that token (drives both RoPE and the head-KV write slot).
    ///
    /// S-A3-f-d chain modes (reference patch 0003 structure):
    /// - `toks_device`: skip the toks upload — sc.toks already holds the ids (the caller
    ///   D2D-copied the previous step's argmax there). `toks` content ignored.
    /// - `keep_dev`: run the draft lm_head + argmax but SKIP the dtoh sync — the id stays
    ///   in sc.argmax (device) for the next chain step / the accept kernel.
    ///
    /// OUTPUT: sc.resid = the head's updated stream stack (the next chain step's tap);
    /// with `want_logits` the per-lane draft argmax (host-returned unless keep_dev).
    #[allow(clippy::too_many_arguments)]
    /// S-A3-p: the device half of head_forward — launches only (graph-capturable). Reads
    /// toks/pos/slots from the persistent device staging buffers.
    pub(super) fn head_launches(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, m: usize, qsa: bool,
                     want_logits: bool) -> Result<()> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let hcn = cfg.hc_count.max(1);
        let rdim = cfg.rotary_dim;
        let cs_stride = (2 * rdim) as i32;
        sc.wp27_cls = crate::exl3_wp27::CLS_DRAFT; // WP27: graph class of this body (MoE fork)
        { let _ = cs_stride; // VIS-2: the mrope-aware gather (text-only slots: cos_tab[pos], unchanged)
        let slots_src = SlotSrc::Pairs(*sc.slots.device_ptr() as u64);
        self.cos_gather(&l, &mut sc.cos, *sc.pos.device_ptr() as u64, slots_src, m)?; }

        // ---- entry (their Qwen4ExpMTPInputLayer, stream_tap=True, out_dtype=f32) ----
        // grouped_norm over the tap with pre_fc_norm_hidden [hc*h]; fc_hidden per stream.
        self.mtp_pre_norm(l, head, 0, *sc.hn.device_ptr() as u64, *sc.resid.device_ptr() as u64, m)?;
        exl3_chain(l, &head.fc_hidden, &sc.hn, &mut sc.hcx, &mut sc.hcy, &mut sc.hf, m * hcn, &sc.chain_ws)?;
        // embedding branch: gather (f32, broadcast rows) -> norm(pre_fc_norm_embedding) -> fc_embedding.
        self.embed_gather(l, *sc.he32.device_ptr() as u64, *sc.toks.device_ptr() as u64, m)?;
        self.mtp_pre_norm(l, head, 1, *sc.hn.device_ptr() as u64, *sc.he32.device_ptr() as u64, m)?;
        exl3_chain(l, &head.fc_embed, &sc.hn, &mut sc.hcx, &mut sc.hcy, &mut sc.yd, m * hcn, &sc.chain_ws)?;
        // streams = h + emb, exact float add of halves, f32 out (their out_dtype).
        xqlaunch!(l, "xq_add_f16_f32", (((h * hcn * m) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                 (&mut sc.resid, &sc.hf, &sc.yd, ((h * hcn * m) as i64)))?;

        // ---- the head's hyper-connected attn+MoE block (same device sequence as a trunk layer) ----
        self.hc_pre(l, sc, &head.hc_attn, m)?;
        let a = &head.attn;
        // q, k|v (+ the indexer qk) — W4/DENSE: one launch; else the old chains
        let idx_pre = self.attn_in_chains(l, sc, a, m, true)?;
        let nh_p = (cfg.num_heads as i32) | ((cfg.num_kv_heads as i32) << 16);
        let hd_p = (cfg.head_dim as i32) | ((cfg.rotary_dim as i32) << 16);
        if qsa && a.idx.is_some() && !qsa_disabled() && !head_qsa_off() {
            // Head QSA: past qsa_limit the draft attends its own indexer's selection (spec).
            self.qsa_attn_step(l, sc, a, HEAD_ATT, m, idx_pre, false)?;
        } else if draft_attn_legacy() || cfg.head_dim != 256 {
            self.qsa_keys_append_dense(l, sc, a, HEAD_ATT, m, idx_pre)?;
            xqlaunch!(l, "xq_attn_decode", ((m as u32) * (cfg.num_heads as u32), 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                      (&mut sc.normed, &sc.qg, &sc.kp, &sc.vp,
                       (*head.kv.device_ptr()) as u64,
                       &a.qnkn, &sc.cos, &sc.slots,
                       nh_p, hd_p, self.mpf(), cfg.rms_eps))?;
        } else {
            // the head's raw indexer keys are spec state on every forward (S-A3-r rule)
            self.qsa_keys_append_dense(l, sc, a, HEAD_ATT, m, idx_pre)?;
            // W4S (draft): dense_prep (grid m*nh) + splitk4_w4 + combine_w4, bitwise to the three
            // launches below (--w4s-off=draft = those).
            if !self.w4s_draft_attn(l, sc, head, a, m)? {
            // A5 D5: prep (K/V cache write + q staging) -> dense split-K (GQA-shared K/V
            // reads, range from the row's position on device) -> ascending split combine.
            xqlaunch!(l, "xq_attn_prep", ((m * cfg.num_kv_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                      (&mut sc.qstage, (*head.kv.device_ptr()) as u64,
                       &sc.qg, &sc.kp, &sc.vp, &a.qnkn, &sc.cos, &sc.slots,
                       nh_p, hd_p, self.mpf(), cfg.rms_eps))?;
            let geom: u64 = (DRAFT_ATTN_SPLITS as u64) | ((self.mpf() as u32 as u64) << 32);
            xqlaunch!(l, "xq_attn_dense_splitk4", ((m * cfg.num_kv_heads * DRAFT_ATTN_SPLITS) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut sc.dpm, &mut sc.dpl, &mut sc.dpacc, &sc.qstage,
                       (*head.kv.device_ptr()) as u64, &sc.slots, nh_p, geom as i64))?;
            xqlaunch!(l, "xq_attn_sel_combine", ((m * cfg.num_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                      (&mut sc.normed, &sc.dpm, &sc.dpl, &sc.dpacc, &sc.qg,
                       nh_p, DRAFT_ATTN_SPLITS as i32, self.rows_kvf(m)))?;
            } // !w4s_draft_attn
        }
        exl3_chain(l, &a.o_proj, &sc.normed, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.mixer_out, m, &sc.chain_ws)?;
        let yatt_p = *sc.mixer_out.device_ptr() as u64;
        // HC (w4): both head injects ride the next hc_pre norm (nothing reads sc.resid between).
        let (hc_def, mut hc_pend) = (self.hc_defer(), None);
        self.hc_post_d(l, sc, yatt_p, m, hc_def, &mut hc_pend)?;
        self.hc_pre_p(l, sc, &head.hc_mlp, m, hc_pend.take())?;
        self.moe(l, sc, &head.moe, m)?;
        let ymlp_p = *sc.moe_out.device_ptr() as u64;
        self.hc_post_d(l, sc, ymlp_p, m, hc_def, &mut hc_pend)?;

        // ---- exit: the head's own mixer collapses streams -> hidden; trunk lm_head drafts. ----
        self.hc_pre_p(l, sc, &head.mixer, m, hc_pend.take())?;
        if want_logits {
            // S-A3-f-d: the draft chain uses the PRUNED head slice when present
            // (reference patch 0002). Draft-only; out-of-slice argmax = verify rejection;
            // the FULL head always verifies, so accepted bits never change.
            let q = match self.lm_head_draft.as_ref() {
                Some(qd) if m == 1 => qd,
                _ => &self.lm_head,
            };
            match (self.fx_dhs.as_ref(), m == 1 && self.lm_head_draft.is_some()) {
                // FP4L --fp4l-draft-head screen: int4 screen -> candidate blocks -> exact rescore (others -inf)
                (Some(dhs), true) => fp4l_fwd::dhs_logits(&l, dhs, q, *sc.x.device_ptr() as u64, *sc.logits.device_ptr() as u64)?,
                _ => exl3_chain(&l, q, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.logits, m, &sc.chain_ws)?,
            }
            let v = q.n as usize;
            // WP15 draft mirror (draft-only): a graphed draft pass proposes from the same
            // penalized distribution the verify will judge (in-chain drafts count as past).
            if let (true, 1, Some(di)) = (sc.pen_live, m, sc.pen_draft_i) {
                xqlaunch!(l, "xq_pen_draft", (v.div_ceil(PEN_SPAN) as u32, 1, 1), (256, 1, 1), 0,
                          (&mut sc.logits, v as i32, &sc.pen, &sc.draft_meta, &sc.d_dev, di as i32,
                           &sc.pen_hist, PEN_RING_LOG2 as i32))?;
            }
            // WP24: a draft pass under --spec-sampling ratio SAMPLES its draft from the top-32
            // proposal q' (xq_rq_draft mode 2, replaces the argmax); the dump rung (mode 1) keeps
            // the argmax draft and only records the pass's top-32. Default (mode 0) = unchanged.
            match (m, sc.rq_draft_i, sc.rq_mode) {
                (1, Some(di), 2) => self.rq_draft(l, sc, v, di, 2)?,
                (1, Some(di), 1) => {
                    self.argmax_rows(l, sc, m, v)?;
                    self.rq_draft(l, sc, v, di, 1)?;
                }
                // WP09: one multi-block launch (bitwise the same draft ids; escape inside).
                _ => self.argmax_rows(l, sc, m, v)?,
            }
        }
        Ok(())
    }

    /// WP24: xq_rq_draft over draft pass `i`'s logits row (sc.logits[..v], the pruned head after
    /// the WP15 draft mirror): its top-32 -> sc.rq_list row i, and (mode 2) the draft sampled from
    /// q' -> sc.argmax[0] (draft_pass then routes it to d_dev[i] / the next pass's token exactly
    /// like the argmax). Counter = draft_meta[3] + i; params = sc.rq_dsamp. Launches only
    /// (graph-capturable; the pass index is baked per pass graph like the other tail launches).
    fn rq_draft(&self, l: &Launcher, sc: &mut Scratch, v: usize, i: usize, mode: i32) -> Result<()> {
        let nb = crate::wp24::draft_nb(v);
        anyhow::ensure!(crate::wp24::draft_fits(v) && i <= MTP_MAX_K,
                        "rq_draft: vocab {v} / pass {i} outside the xq_rq_draft envelope");
        // i0 = i: list row i AND counter draft_meta[3] + i (row r = 0 of a one-row grid)
        xqlaunch!(l, "xq_rq_draft", (nb as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.argmax, &sc.logits, v as i32, &mut sc.rq_part, &mut sc.rq_cnt,
                   &mut sc.rq_list, &sc.rq_dsamp, &sc.draft_meta, i as i32, mode))?;
        Ok(())
    }

    /// WP24 rung 0 on the DHEAD / DHEADP tails (--wp24-dump, rq_mode 1): the served tail has
    /// already written the pass's draft (argmax, dconf, d_dev, toks — untouched here, so the
    /// trajectory is the served one); recompute the slice logits from the same sc.x (the old
    /// tail's exl3_chain, + the WP15 draft mirror when penalties are live — DHEAD's XCHECK does the
    /// same) and record their top-32 into rq_list row i. Launches only (graph-capturable).
    pub(super) fn rq_dump_list(&self, l: &Launcher, sc: &mut Scratch, q: &Quad, i: usize) -> Result<()> {
        if sc.rq_mode != 1 {
            return Ok(());
        }
        exl3_chain(l, q, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.logits, 1, &sc.chain_ws)?;
        let v = q.n as usize;
        if sc.pen_live && pen_draft_on() {
            xqlaunch!(l, "xq_pen_draft", (v.div_ceil(PEN_SPAN) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut sc.logits, v as i32, &sc.pen, &sc.draft_meta, &sc.d_dev, i as i32,
                       &sc.pen_hist, PEN_RING_LOG2 as i32))?;
        }
        self.rq_draft(l, sc, v, i, 1)
    }

    pub fn head_forward(&self, sc: &mut Scratch, head: &DraftHead, toks: &[i32],
                        pos_host: &[usize], slots: &[usize], want_logits: bool,
                        toks_device: bool, keep_dev: bool) -> Result<Vec<i32>> {
        let m = toks.len();
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let cfg = &self.cfg;

        // S-A3-f Item 1: persistent width-sized staging buffers + table-gather RoPE
        // (same contract as forward_step; see the note there).
        let w = sc.toks.len();
        if !toks_device {
            let mut toks_v = toks.to_vec();
            toks_v.resize(w, 0);
            self.dev.htod_copy_into(toks_v, &mut sc.toks)?;
        }
        let pos_h: Vec<i32> = pos_host.iter().map(|p| *p as i32).collect();
        let mut pos_v = pos_h.clone();
        pos_v.resize(w, 0);
        self.dev.htod_copy_into(pos_v, &mut sc.pos)?;
        let mut slotpos: Vec<i32> = vec![0; w * 2];
        for (i, pp) in pos_host.iter().enumerate() {
            slotpos[i * 2] = slots[i] as i32;
            slotpos[i * 2 + 1] = *pp as i32;
        }
        self.dev.htod_copy_into(slotpos, &mut sc.slots)?;
        // head QSA: the selection kernels read the deinterleaved slot table + the grid bucket
        let mut sids = vec![0i32; sc.slot_ids.len()];
        for (i, &s_) in slots.iter().enumerate() { if i < sids.len() { sids[i] = s_ as i32; } }
        self.dev.htod_copy_into(sids, &mut sc.slot_ids)?;
        let pmax = pos_host.iter().copied().max().unwrap_or(0);
        sc.qsa_grid_nblk = self.qsa_grid_bucket(pmax).0;
        let qsa = self.qsa_live(pmax);
        self.head_launches(&l, sc, head, m, qsa, want_logits)?;
        if want_logits {
            if keep_dev {
                Ok(Vec::new())
            } else {
                self.dsync()?;
                Ok(self.dtoh(&sc.argmax)?)
            }
        } else {
            Ok(Vec::new())
        }
    }
}
