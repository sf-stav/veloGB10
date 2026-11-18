//! Probe-only: `probe_pack3`, the pack2 stage / snapshot / lone / packed helpers and `probe_pack2`.
//! See `mod.rs` for the file map and `pack.rs` for the production packed path.

use super::*;

impl FwdModel {
    /// PACK3 LEG 3 STEP A (S-B9-CF-PACK3): the GRAPHED round-pair A/B.
    /// V5 = the served-style lone verify graph (slot 0, m=5) captured through
    /// the SERVED capture_verify_graph machinery; P10 = the packed verify
    /// graph (R_pad=10, k_seg=2; trunk + accept_seg + commit_seg in the
    /// graph, PLE shadow + taps as a 4-launch eager tail). Asserts
    /// graph-vs-eager bit identity (outputs + full post-commit state, 3 reps)
    /// and the 6+4 SAME-graph replay, then times everything with CUDA events
    /// around the replay and prints the per-stage table + round_pair_ratio.
    pub fn probe_pack3(&self) -> Result<()> {
        anyhow::ensure!(gdn_commit_mode(self.tc.lin_k_dim, self.tc.lin_v_dim) == 1,
                        "pack3: needs the ring commit mode (gdn_commit_mode == 1)");
        let head = match self.mtp.as_ref() {
            Some(h) => h,
            None => anyhow::bail!("pack has no mtp.* draft head — pack3 probe unavailable"),
        };
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let k = 4usize;
        let m = k + 1;
        let rpad = 2 * m;
        let m0 = 6usize; // the 6+4 split's first segment length
        let mut sc = Scratch::new(&self.dev, &self.cfg, 16, self.cfg.rotary_dim, self.max_pos)?;
        anyhow::ensure!(sc.d_dev.len() >= 2 * (k + 1), "pack3: d_dev narrower than 2 x (k+1)");
        crate::dprintln!("PACK3A: graphed round-pair A/B (k={k}, m={m}, R_pad={rpad}; graph = seg trunk + accept_seg + commit_seg; \
                  eager tail = PLE shadow + per-segment taps)");
        // ---- prime two slots (the PACKSTATE probe pattern) ----
        let synth = |n: usize, salt: usize| -> Vec<u32> {
            (0..n).map(|i| ((i * 7919 + salt * 104729 + 12345) % (self.cfg.vocab_size - 2000) + 1000) as u32).collect()
        };
        let pa = synth(97, 1);
        let pb = synth(89, 2);
        self.reset_slot(0)?;
        self.reset_slot(1)?;
        let mut psc: Option<PrefillScratch> = None;
        let prime = |slot: usize, pr: &[u32], sc: &mut Scratch,
                     psc: &mut Option<PrefillScratch>| -> Result<(i32, usize)> {
            let n = pr.len();
            let mut pos = 0usize;
            while pos + 1 < n {
                let end = (pos + 1024).min(n - 1);
                let toks: Vec<i32> = pr[pos..end].iter().map(|&t| t as i32).collect();
                if psc.is_none() { *psc = Some(self.prefill_scratch(1024)?); }
                self.prefill_chunk(psc.as_mut().unwrap(), &toks, pos, slot, false, None)?;
                pos = end;
            }
            self.forward_step(sc, &[pr[n - 1] as i32], &[n - 1], &[slot], None)?;
            let b = self.dtoh(&sc.argmax)?[0];
            self.tap_snapshot(sc, slot)?;
            Ok((b, n))
        };
        let (b0, p0) = prime(0, &pa, &mut sc, &mut psc)?;
        let (b1, p1) = prime(1, &pb, &mut sc, &mut psc)?;
        // real draft chains (k+1 drafts each: the 6+4 arm needs 5 and 3)
        self.packstate_tap_restore(&l, &mut sc, 0, 0)?;
        let d0 = self.packstate_draft(&l, &mut sc, head, 0, k + 1, b0, p0)?;
        self.packstate_tap_restore(&l, &mut sc, 1, 0)?;
        let d1 = self.packstate_draft(&l, &mut sc, head, 1, k + 1, b1, p1)?;
        let pmax = p0.max(p1) + k;
        let qsa = self.qsa_live(pmax);
        let (qg, qb) = self.qsa_grid_bucket_key(pmax);
        sc.qsa_grid_nblk = qg;
        crate::dprintln!("PACK3A: regime qsa={qsa} qb={qb} (frozen for every arm); b=({b0},{b1}) p=({p0},{p1})");
        // ---- staging helpers ----
        let stage_lone = |sc: &mut Scratch, rows: &[(usize, i32, usize)], dv: &[i32]| -> Result<()> {
            self.pack2_stage(sc, rows)?;
            let mut d = vec![0i32; sc.d_dev.len()];
            let n = dv.len().min(d.len());
            d[..n].copy_from_slice(&dv[..n]);
            self.dev.htod_copy_into(d, &mut sc.d_dev)?;
            Ok(())
        };
        let stage_packed = |sc: &mut Scratch, segs: &[[i32; 4]], rows: &[(usize, i32, usize)],
                            dv: &[i32]| -> Result<()> {
            self.pack2_stage(sc, rows)?;
            let mut d = vec![0i32; sc.d_dev.len()];
            let n = dv.len().min(d.len());
            d[..n].copy_from_slice(&dv[..n]);
            self.dev.htod_copy_into(d, &mut sc.d_dev)?;
            let mut sf: Vec<i32> = segs.iter().flat_map(|s| s.iter().cloned()).collect();
            sf.resize(PACK_SEG_MAX * 4, 0);
            self.dev.htod_copy_into(sf, &mut sc.seg)?;
            Ok(())
        };
        let rows55: Vec<(usize, i32, usize)> = {
            let mut r: Vec<(usize, i32, usize)> =
                (0..m).map(|i| (0usize, if i == 0 { b0 } else { d0[i - 1] }, p0 + i)).collect();
            r.extend((0..m).map(|i| (1usize, if i == 0 { b1 } else { d1[i - 1] }, p1 + i)));
            r
        };
        let segs55: [[i32; 4]; 2] = [[0, 0, m as i32, 1], [1, m as i32, m as i32, 1]];
        let rows64: Vec<(usize, i32, usize)> = {
            let mut r: Vec<(usize, i32, usize)> =
                (0..m0).map(|i| (0usize, if i == 0 { b0 } else { d0[i - 1] }, p0 + i)).collect();
            r.extend((0..(rpad - m0)).map(|i| (1usize, if i == 0 { b1 } else { d1[i - 1] }, p1 + i)));
            r
        };
        let segs64: [[i32; 4]; 2] = [[0, 0, m0 as i32, 1], [1, m0 as i32, (rpad - m0) as i32, 1]];
        let dv55: Vec<i32> = {
            let mut d = vec![0i32; 2 * m];
            d[..k].copy_from_slice(&d0[..k]);
            d[m..m + k].copy_from_slice(&d1[..k]);
            d
        };
        let dv64: Vec<i32> = {
            let mut d = vec![0i32; 2 * m];
            d[..m0 - 1].copy_from_slice(&d0[..m0 - 1]);
            d[m0..m0 + (rpad - m0 - 1)].copy_from_slice(&d1[..rpad - m0 - 1]);
            d
        };
        // ---- output collectors (bitwise f32 compare) ----
        // core/x are bf16 planes (u16 words) — bitwise u16 compare is exact
        let collect = |sc: &Scratch, rows: usize, packed: bool|
                -> Result<(Vec<u16>, Vec<u16>, Vec<i32>, Vec<i32>, Vec<i32>)> {
            Ok((
                self.dtoh(&sc.core)?[..rows * self.core_row()].to_vec(),
                self.dtoh(&sc.x)?[..rows * self.cfg.hidden_size].to_vec(),
                self.dtoh(&sc.argmax)?[..rows].to_vec(),
                self.dtoh(&sc.accept_out)?.to_vec(),
                if packed { self.dtoh(&sc.acc_seg)?.to_vec() }
                else { self.dtoh(&sc.acc2)?.to_vec() },
            ))
        };
        let outs_eq = |a: &(Vec<u16>, Vec<u16>, Vec<i32>, Vec<i32>, Vec<i32>),
                       b: &(Vec<u16>, Vec<u16>, Vec<i32>, Vec<i32>, Vec<i32>)|
                -> bool { a.0 == b.0 && a.1 == b.1 && a.2 == b.2 && a.3 == b.3 && a.4 == b.4 };
        let cap = p0.max(p1) + 64;
        // ============ P10: eager reference + graph identity (5+5) ============
        stage_packed(&mut sc, &segs55, &rows55, &dv55)?;
        let shadow = self.packstate_save(&l, &sc, &[0, 1], cap)?;
        self.verify_block_packed_full(&l, &mut sc, &segs55, rpad, 2, qsa)?;
        self.dsync()?;
        let e_out = collect(&sc, rpad, true)?;
        let e_state = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
        self.packstate_restore(&l, &shadow)?;
        self.dsync()?;
        let pkey = pkeyv_of(rpad, qsa, qb) | sk(&sc);
        self.capture_verify_graph_packed(&l, &mut sc, rpad, 2, qsa, pkey);
        anyhow::ensure!(self.graphs.lock().unwrap().contains_key(&pkey),
                        "pack3: packed graph capture failed (see EXL3_GRAPH lines)");
        for rep in 0..3usize {
            self.packstate_restore(&l, &shadow)?;
            self.pack_eager_tail_pre(&l, &mut sc, &segs55)?;
            self.graphs.lock().unwrap().get(&pkey).unwrap().launch();
            self.pack_eager_tail_post(&l, &mut sc, &segs55)?;
            self.dsync()?;
            let g_out = collect(&sc, rpad, true)?;
            let g_state = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
            anyhow::ensure!(outs_eq(&g_out, &e_out),
                            "pack3: packed graph outputs differ from eager (5+5, rep {rep})");
            anyhow::ensure!(g_state == e_state,
                            "pack3: packed graph post-commit state differs from eager (5+5, rep {rep})");
        }
        crate::dprintln!("PACK3A: P10 graph-vs-eager identity (5+5): PASS x3 (outputs + full post-commit state)");
        // ============ V5: served-style lone graph on slot 0 ============
        let rows0: Vec<(usize, i32, usize)> =
            (0..m).map(|r| (0usize, if r == 0 { b0 } else { d0[r - 1] }, p0 + r)).collect();
        let dv0: Vec<i32> = d0[..k].to_vec();
        stage_lone(&mut sc, &rows0, &dv0)?;
        let shadow0 = self.packstate_save(&l, &sc, &[0, 1], cap)?;
        self.verify_block(&l, &mut sc, 0, m, k, qsa)?;
        self.dsync()?;
        let ev_out = collect(&sc, m, false)?;
        let ev_state = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
        let vkey = vkey_of(0, m, qsa, qb) | sk(&sc)
            | ((sc.pen_live as usize) << 40)
            | rq_verify_key(&sc)
            | self.vp_key(&sc, m);
        if !self.graphs.lock().unwrap().contains_key(&vkey) {
            self.dsync()?;
            self.capture_verify_graph(&l, &mut sc, 0, m, k, qsa, vkey);
        }
        anyhow::ensure!(self.graphs.lock().unwrap().contains_key(&vkey),
                        "pack3: lone graph capture failed (see EXL3_GRAPH lines)");
        for rep in 0..3usize {
            self.packstate_restore(&l, &shadow0)?;
            self.graphs.lock().unwrap().get(&vkey).unwrap().launch();
            self.dsync()?;
            let g_out = collect(&sc, m, false)?;
            let g_state = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
            anyhow::ensure!(outs_eq(&g_out, &ev_out),
                            "pack3: lone graph outputs differ from eager (rep {rep})");
            anyhow::ensure!(g_state == ev_state,
                            "pack3: lone graph post-commit state differs from eager (rep {rep})");
        }
        crate::dprintln!("PACK3A: V5 graph-vs-eager identity: PASS x3");
        // ============ P(6+4): SAME graph, different split ============
        stage_packed(&mut sc, &segs64, &rows64, &dv64)?;
        let shadow64 = self.packstate_save(&l, &sc, &[0, 1], cap)?;
        self.verify_block_packed_full(&l, &mut sc, &segs64, rpad, 2, qsa)?;
        self.dsync()?;
        let e64 = collect(&sc, rpad, true)?;
        let e64s = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
        self.packstate_restore(&l, &shadow64)?;
        self.pack_eager_tail_pre(&l, &mut sc, &segs64)?;
        self.graphs.lock().unwrap().get(&pkey).unwrap().launch();
        self.pack_eager_tail_post(&l, &mut sc, &segs64)?;
        self.dsync()?;
        let g64 = collect(&sc, rpad, true)?;
        let g64s = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
        anyhow::ensure!(outs_eq(&g64, &e64), "pack3: 6+4 replay outputs differ from eager");
        anyhow::ensure!(g64s == e64s, "pack3: 6+4 replay post-state differs from eager");
        crate::dprintln!("PACK3A: P(6+4) SAME-graph replay (key is R_pad-only): PASS (outputs + post-commit state)");
        // ============ timing: CUDA events around the replay ============
        let mut ev: [cudarc::driver::sys::CUevent; 2] = [std::ptr::null_mut(); 2];
        for e in ev.iter_mut() {
            let r = unsafe { cudarc::driver::sys::cuEventCreate(e, 0) };
            anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                            "pack3: cuEventCreate ({r:?})");
        }
        let (warm, reps) = (3usize, 20usize);
        let series = |title: &str, f: &mut dyn FnMut()| -> Result<f64> {
            use cudarc::driver::sys;
            for _ in 0..warm { f(); }
            self.dsync()?;
            let mut v = Vec::with_capacity(reps);
            for _ in 0..reps {
                let stream = self.stream.stream;
                unsafe { sys::cuEventRecord(ev[0], stream); }
                f();
                unsafe { sys::cuEventRecord(ev[1], stream); }
                let r = unsafe { sys::cuEventSynchronize(ev[1]) };
                anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "pack3: evsync ({r:?})");
                let mut ms = 0f32;
                let r = unsafe { sys::cuEventElapsedTime(&mut ms, ev[0], ev[1]) };
                anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "pack3: elapsed ({r:?})");
                v.push(ms as f64);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            crate::dprintln!("PACK3A TIME {title}: median {:.3} ms (min {:.3}, max {:.3})",
                     v[reps / 2], v[0], v[reps - 1]);
            Ok(v[reps / 2])
        };
        let host_ms = |title: &str, f: &mut dyn FnMut()| -> Result<f64> {
            for _ in 0..warm { f(); }
            self.dsync()?;
            let mut v = Vec::new();
            for _ in 0..reps {
                let t0 = std::time::Instant::now();
                f();
                v.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            crate::dprintln!("PACK3A TIME {title}: median {:.3} ms (host wall, min {:.3}, max {:.3})",
                     v[reps / 2], v[0], v[reps - 1]);
            Ok(v[reps / 2])
        };
        // V5 arms (staging: lone slot 0)
        stage_lone(&mut sc, &rows0, &dv0)?;
        self.dsync()?;
        let v5_g = series("V5_graph (lone m=5, served-style graph)", &mut || {
            self.graphs.lock().unwrap().get(&vkey).unwrap().launch();
        })?;
        let v5_e = series("V5_eager (lone m=5, verify_block)", &mut || {
            let _ = self.verify_block(&l, &mut sc, 0, m, k, qsa);
        })?;
        // P10 arms (staging: packed 5+5)
        stage_packed(&mut sc, &segs55, &rows55, &dv55)?;
        self.dsync()?;
        let p10_g = series("P10_graph (packed R_pad=10, graph only)", &mut || {
            self.graphs.lock().unwrap().get(&pkey).unwrap().launch();
        })?;
        let p10_eff = series("P10_eff (graph + 4-launch eager tail)", &mut || {
            self.pack_eager_tail_pre(&l, &mut sc, &segs55).unwrap();
            self.graphs.lock().unwrap().get(&pkey).unwrap().launch();
            self.pack_eager_tail_post(&l, &mut sc, &segs55).unwrap();
        })?;
        let p10_e = series("P10_eager (verify_block_packed_full)", &mut || {
            let _ = self.verify_block_packed_full(&l, &mut sc, &segs55, rpad, 2, qsa);
        })?;
        // P(6+4): SAME graph, different split
        stage_packed(&mut sc, &segs64, &rows64, &dv64)?;
        self.dsync()?;
        let p64_eff = series("P64_eff (SAME R_pad graph, 6+4 split, + tail)", &mut || {
            self.pack_eager_tail_pre(&l, &mut sc, &segs64).unwrap();
            self.graphs.lock().unwrap().get(&pkey).unwrap().launch();
            self.pack_eager_tail_post(&l, &mut sc, &segs64).unwrap();
        })?;
        // per-stage eager costs
        stage_packed(&mut sc, &segs55, &rows55, &dv55)?;
        self.dsync()?;
        let shadow_ms = series("stage shadow_pre (2 launches)", &mut || {
            let _ = self.pack_eager_tail_pre(&l, &mut sc, &segs55);
        })?;
        let taps_ms = series("stage taps_post (2 launches)", &mut || {
            let _ = self.pack_eager_tail_post(&l, &mut sc, &segs55);
        })?;
        let tapr_ms = series("stage tap_restore (1 launch)", &mut || {
            let _ = self.packstate_tap_restore(&l, &mut sc, 0, 0);
        })?;
        let draft_ms = host_ms("stage draft_chain (per request, k=4, probe path incl sync+readback)", &mut || {
            let _ = self.packstate_draft(&l, &mut sc, head, 0, k, b0, p0);
        })?;
        let reprime_ms = host_ms("stage head_reprime (per request)", &mut || {
            let _ = self.head_reprime(&mut sc, head, 0, p0, &d0[..2]);
        })?;
        let readb_ms = host_ms("stage readback (accept_out + acc_seg dtoh)", &mut || {
            let _ = self.dtoh(&sc.accept_out);
            let _ = self.dtoh(&sc.acc_seg);
        })?;
        let staging_ms = host_ms("stage pack staging (pack2_stage + d_dev + seg htod)", &mut || {
            let _ = stage_packed(&mut sc, &segs55, &rows55, &dv55);
        })?;
        // ---- the round model (equal acceptance assumed; commits inside both graphs) ----
        let verify_ratio = (2.0 * v5_g) / p10_eff;
        let serial_pair = 2.0 * (draft_ms + v5_g + tapr_ms + reprime_ms + readb_ms);
        let packed_round = 2.0 * (draft_ms + tapr_ms) + p10_eff + 2.0 * reprime_ms + readb_ms;
        let rpr = serial_pair / packed_round;
        crate::dprintln!("PACK3A: verify_ratio = 2*V5_graph / P10_eff = {verify_ratio:.3}");
        crate::dprintln!("PACK3A: round_pair_ratio (probe-eager stages) = {rpr:.3} \
                  (serial {serial_pair:.2} ms vs packed {packed_round:.2} ms; staging {staging_ms:.3} ms extra, not counted)");
        // served-graphed draft/reprime sensitivity: 0.77 ms per draft pass, 0.24 ms re-prime
        let draft_g = 0.77 * k as f64;
        let reprime_g = 0.24f64;
        let serial_g = 2.0 * (draft_g + v5_g + reprime_g + tapr_ms + readb_ms);
        let packed_g = 2.0 * (draft_g + tapr_ms) + p10_eff + 2.0 * reprime_g + readb_ms;
        crate::dprintln!("PACK3A: round_pair_ratio (served-graphed draft {draft_g:.2} ms/req, re-prime {reprime_g:.2} ms) = {:.3} \
                  (serial {serial_g:.2} ms vs packed {packed_g:.2} ms)",
                 serial_g / packed_g);
        let d_v5 = v5_e - v5_g;
        let d_p10 = p10_e - p10_g;
        let tail_cost = p10_eff - p10_g;
        crate::dprintln!("PACK3A: launch-gap recovery: V5 {v5_e:.2} -> {v5_g:.2} ms ({d_v5:.2} ms saved); \
                  P10 {p10_e:.2} -> {p10_g:.2} ms ({d_p10:.2} ms saved); tail cost {tail_cost:.3} ms; \
                  shadow {shadow_ms:.3} taps {taps_ms:.3} tapr {tapr_ms:.3} draft {draft_ms:.3} \
                  reprime {reprime_ms:.3} readback {readb_ms:.3} staging {staging_ms:.3}");
        // ============ STEP B: the R_pad graph family (--pack3-family) ============
        // Enumerate R_pad {2..16} x k_seg=2 at the live regime: lazy capture,
        // graph-vs-eager bit identity x3 per R_pad (outputs + post-commit
        // state), cuMemGetInfo accounting with the 1 GB cap.
        if crate::opts::on(crate::opt!("pack3-family")) {
            use cudarc::driver::sys;
            crate::dprintln!("PACK3B: family capture begins (R_pad {{2..16}} x k_seg=2, qsa={qsa} qb={qb})");
            let k7 = 7usize; // MTP_MAX_K: drafts for the widest family member
            self.packstate_tap_restore(&l, &mut sc, 0, 0)?;
            let d0f = self.packstate_draft(&l, &mut sc, head, 0, k7, b0, p0)?;
            self.packstate_tap_restore(&l, &mut sc, 1, 0)?;
            let d1f = self.packstate_draft(&l, &mut sc, head, 1, k7, b1, p1)?;
            let (mut free0, _total) = (0usize, 0usize);
            unsafe {
                let mut fr: usize = 0; let mut tot: usize = 0;
                let r = sys::cuMemGetInfo_v2(&mut fr, &mut tot);
                anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "pack3b: cuMemGetInfo ({r:?})");
                free0 = fr;
            }
            crate::dprintln!("PACK3B: free0 = {:.3} GiB", free0 as f64 / (1 << 30) as f64);
            let mut cap_bytes = 0usize;
            let mut capped_at: Option<usize> = None;
            for rpad_f in [2usize, 4, 6, 8, 10, 12, 14, 16] {
                if cap_bytes > (1 << 30) {
                    capped_at = Some(rpad_f);
                    crate::dprintln!("PACK3B: stopped at the 1 GB cap before R_pad={rpad_f} (captured {cap_bytes} bytes)");
                    break;
                }
                let mm = rpad_f / 2;
                let rows_f: Vec<(usize, i32, usize)> = {
                    let mut r: Vec<(usize, i32, usize)> =
                        (0..mm).map(|i| (0usize, if i == 0 { b0 } else { d0f[i - 1] }, p0 + i)).collect();
                    r.extend((0..mm).map(|i| (1usize, if i == 0 { b1 } else { d1f[i - 1] }, p1 + i)));
                    r
                };
                let segs_f: [[i32; 4]; 2] =
                    [[0, 0, mm as i32, 1], [1, mm as i32, mm as i32, 1]];
                let dv_f: Vec<i32> = {
                    let mut d = vec![0i32; rpad_f];
                    d[..mm - 1].copy_from_slice(&d0f[..mm - 1]);
                    d[mm..rpad_f - 1].copy_from_slice(&d1f[..mm - 1]);
                    d
                };
                stage_packed(&mut sc, &segs_f, &rows_f, &dv_f)?;
                let sh_f = self.packstate_save(&l, &sc, &[0, 1], cap)?;
                self.verify_block_packed_full(&l, &mut sc, &segs_f, rpad_f, 2, qsa)?;
                self.dsync()?;
                let ef = collect(&sc, rpad_f, true)?;
                let efs = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
                self.packstate_restore(&l, &sh_f)?;
                self.dsync()?;
                let key_f = pkeyv_of(rpad_f, qsa, qb) | sk(&sc);
                self.capture_verify_graph_packed(&l, &mut sc, rpad_f, 2, qsa, key_f);
                anyhow::ensure!(self.graphs.lock().unwrap().contains_key(&key_f),
                                "pack3b: family capture failed at R_pad={rpad_f}");
                for rep in 0..3usize {
                    self.packstate_restore(&l, &sh_f)?;
                    self.pack_eager_tail_pre(&l, &mut sc, &segs_f)?;
                    self.graphs.lock().unwrap().get(&key_f).unwrap().launch();
                    self.pack_eager_tail_post(&l, &mut sc, &segs_f)?;
                    self.dsync()?;
                    let gf = collect(&sc, rpad_f, true)?;
                    let gfs = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
                    anyhow::ensure!(outs_eq(&gf, &ef), "pack3b: R_pad={rpad_f} outputs differ (rep {rep})");
                    anyhow::ensure!(gfs == efs, "pack3b: R_pad={rpad_f} post-state differs (rep {rep})");
                }
                let mut fr: usize = 0; let mut tot: usize = 0;
                unsafe {
                    let r = sys::cuMemGetInfo_v2(&mut fr, &mut tot);
                    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "pack3b: cuMemGetInfo ({r:?})");
                }
                let used = free0.saturating_sub(fr).saturating_sub(cap_bytes);
                cap_bytes = free0.saturating_sub(fr);
                crate::dprintln!("PACK3B: R_pad={rpad_f:2} identity x3 PASS, graph ~{:.1} MiB (family total {:.1} MiB)",
                         used as f64 / (1 << 20) as f64, cap_bytes as f64 / (1 << 20) as f64);
            }
            let n_family = self.graphs.lock().unwrap().keys()
                .filter(|k| (**k >> 30) & 0x7 == 4).count();
            crate::dprintln!("PACK3B: family verdict: {n_family} packed graphs, {:.3} GiB captured{}",
                     cap_bytes as f64 / (1 << 30) as f64,
                     if let Some(at) = capped_at { format!(" (CAPPED before R_pad={at})") } else { String::new() });
            // the qsa=true axis at the shipping width (R_pad=10), best-effort:
            // eager reference first; on ANY error, skip the regime with a note.
            let mut qsa_note = String::from("not attempted");
            match (|| -> Result<()> {
                let key_q = pkeyv_of(rpad, true, qb) | sk(&sc);
                stage_packed(&mut sc, &segs55, &rows55, &dv55)?;
                let sh_q = self.packstate_save(&l, &sc, &[0, 1], cap)?;
                self.verify_block_packed_full(&l, &mut sc, &segs55, rpad, 2, true)?;
                self.dsync()?;
                let eq_ = collect(&sc, rpad, true)?;
                let eqs = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
                self.packstate_restore(&l, &sh_q)?;
                self.dsync()?;
                self.capture_verify_graph_packed(&l, &mut sc, rpad, 2, true, key_q);
                anyhow::ensure!(self.graphs.lock().unwrap().contains_key(&key_q),
                                "qsa capture failed");
                for rep in 0..3usize {
                    self.packstate_restore(&l, &sh_q)?;
                    self.pack_eager_tail_pre(&l, &mut sc, &segs55)?;
                    self.graphs.lock().unwrap().get(&key_q).unwrap().launch();
                    self.pack_eager_tail_post(&l, &mut sc, &segs55)?;
                    self.dsync()?;
                    let gq = collect(&sc, rpad, true)?;
                    let gqs = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
                    anyhow::ensure!(outs_eq(&gq, &eq_), "qsa outputs differ (rep {rep})");
                    anyhow::ensure!(gqs == eqs, "qsa post-state differs (rep {rep})");
                }
                Ok(())
            })() {
                Ok(()) => qsa_note = format!("qsa=true at R_pad={rpad}: identity x3 PASS"),
                Err(e) => qsa_note = format!("qsa=true at R_pad={rpad}: SKIPPED ({e:#})"),
            }
            crate::dprintln!("PACK3B: QSA axis: {qsa_note} (full 8x2 enumeration belongs to LEG 4 boot precapture; capture is staging-value-independent)");
        }
        // ---- nsys receipt window: 5 isolated packed replays ----
        stage_packed(&mut sc, &segs55, &rows55, &dv55)?;
        self.dsync()?;
        crate::dprintln!("PACK3A: NSYS-WINDOW-BEGIN (5 packed graph replays + tails)");
        for _ in 0..5usize {
            self.pack_eager_tail_pre(&l, &mut sc, &segs55)?;
            self.graphs.lock().unwrap().get(&pkey).unwrap().launch();
            self.pack_eager_tail_post(&l, &mut sc, &segs55)?;
        }
        self.dsync()?;
        crate::dprintln!("PACK3A: NSYS-WINDOW-END");
        for e in ev.iter() { unsafe { cudarc::driver::sys::cuEventDestroy_v2(*e); } }
        crate::dprintln!("PACK3A: done (V5 + P10 identity x3, 6+4 same-graph PASS; timings above)");
        Ok(())
    }

    /// PACK1 probe staging: per-row (slot, tok, pos) uploads + the PLE host half.
    /// Advances ple_hist (per slot) — the callers save/restore around it.
    pub(super) fn pack2_stage(&self, sc: &mut Scratch, rows: &[(usize, i32, usize)]) -> Result<()> {
        let m = rows.len();
        let mut toks: Vec<i32> = rows.iter().map(|r| r.1).collect();
        toks.resize(sc.toks.len(), 0);
        self.dev.htod_copy_into(toks, &mut sc.toks)?;
        let mut pos: Vec<i32> = rows.iter().map(|r| r.2 as i32).collect();
        pos.resize(sc.pos.len(), 0);
        self.dev.htod_copy_into(pos, &mut sc.pos)?;
        let mut slotpos = vec![0i32; sc.slots.len()];
        for (r, row) in rows.iter().enumerate() {
            slotpos[r * 2] = row.0 as i32;
            slotpos[r * 2 + 1] = row.2 as i32;
        }
        self.dev.htod_copy_into(slotpos, &mut sc.slots)?;
        let mut sids = vec![0i32; sc.slot_ids.len()];
        for (r, row) in rows.iter().enumerate() {
            sids[r] = row.0 as i32;
        }
        self.dev.htod_copy_into(sids, &mut sc.slot_ids)?;
        let mut emb = vec![0u16; sc.ple_emb.len()];
        let slots_v: Vec<usize> = rows.iter().map(|r| r.0).collect();
        let tokv: Vec<i32> = rows.iter().map(|r| r.1).collect();
        self.ple_build_embed(m, &tokv, &slots_v, &mut emb)?;
        self.dev.htod_copy_into(emb, &mut sc.ple_emb)?;
        Ok(())
    }

    /// G3 helper: serialize slots' live GDN/conv/PLE rows (one row per layer) deterministically.
    fn pack2_snap_state(&self, l: &Launcher, slots: &[usize]) -> Result<Vec<u64>> {
        let cfg = &self.tc;
        let s_e = (cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim) as i64;
        let c_e = ((cfg.key_dim() * 2 + cfg.value_dim()) * cfg.conv_kernel) as i64;
        let p_e = (10240 * 9) as i64;
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        let mut s_tmp = self.dev.alloc_zeros::<f32>(s_e as usize)?;
        let mut c_tmp = self.dev.alloc_zeros::<f32>(c_e as usize)?;
        let mut p_tmp = self.dev.alloc_zeros::<u16>(p_e as usize)?;
        let mut out = Vec::new();
        for &slot in slots {
            for li in 0..self.s_state.len() {
                xqlaunch!(l, "xq_copy_f32", grid(s_e), (256, 1, 1), 0,
                          (&mut s_tmp, &self.s_state[li], s_e, 0i64, (slot as i64) * s_e))?;
                for f in self.dtoh(&s_tmp)? {
                    out.push(f.to_bits() as u64);
                }
                xqlaunch!(l, "xq_copy_f32", grid(c_e), (256, 1, 1), 0,
                          (&mut c_tmp, &self.conv_state[li], c_e, 0i64, (slot as i64) * c_e))?;
                for f in self.dtoh(&c_tmp)? {
                    out.push(f.to_bits() as u64);
                }
            }
            xqlaunch!(l, "xq_copy_u16", grid(p_e), (256, 1, 1), 0,
                      (&mut p_tmp, &self.ple_state, p_e, 0i64, (slot as i64) * p_e))?;
            for u in self.dtoh(&p_tmp)? {
                out.push(u as u64);
            }
        }
        Ok(out)
    }

    fn pack2_lone(&self, l: &Launcher, sc: &mut Scratch, slot: usize, b: i32, p: usize,
                  drafts: &[i32], k: usize, qsa: bool) -> Result<Pack2Seg> {
        let m = k + 1;
        let hist_save = {
            let h = self.ple_hist.lock().unwrap();
            (h[slot * 2], h[slot * 2 + 1])
        };
        self.dsync()?;
        let t0 = std::time::Instant::now();
        let rows: Vec<(usize, i32, usize)> =
            (0..m).map(|r| (slot, if r == 0 { b } else { drafts[r - 1] }, p + r)).collect();
        self.pack2_stage(sc, &rows)?;
        let mut dv = vec![0i32; sc.d_dev.len()];
        dv[..k].copy_from_slice(&drafts[..k]);
        self.dev.htod_copy_into(dv, &mut sc.d_dev)?;
        self.verify_block_nostore(l, sc, slot, m, k, qsa)?;
        self.dsync()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        {
            let mut h = self.ple_hist.lock().unwrap();
            h[slot * 2] = hist_save.0;
            h[slot * 2 + 1] = hist_save.1;
        }
        let out = self.dtoh(&sc.accept_out)?;
        let acc = self.dtoh(&sc.acc2)?;
        let a = out[k + 2] as usize;
        Ok(Pack2Seg {
            core: self.dtoh(&sc.core)?[..m * self.core_row()].to_vec(),
            x: self.dtoh(&sc.x)?[..m * self.cfg.hidden_size].to_vec(),
            logits: self.dtoh(&sc.logits)?[..m * self.cfg.vocab_size].to_vec(),
            argmax: self.dtoh(&sc.argmax)?[..m].to_vec(),
            out,
            acc,
            a,
            ms,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn pack2_packed(&self, l: &Launcher, sc: &mut Scratch, b0: i32, p0: usize, d0: &[i32],
                    b1: i32, p1: usize, d1: &[i32], k: usize, qsa: bool) -> Result<Pack2Packed> {
        let m = k + 1;
        let r = 2 * m;
        // {slot, row_start, len, flags}; flags bit0 = nostore (the packed verify is
        // nostore: the LEG 2 commit will replay from live — never store here).
        let segs: [[i32; 4]; 2] =
            [[0, 0, m as i32, 1], [1, m as i32, m as i32, 1]];
        let hist_save = {
            let h = self.ple_hist.lock().unwrap();
            (h[0], h[1], h[2], h[3])
        };
        self.dsync()?;
        let t0 = std::time::Instant::now();
        let mut rows: Vec<(usize, i32, usize)> =
            (0..m).map(|i| (0usize, if i == 0 { b0 } else { d0[i - 1] }, p0 + i)).collect();
        rows.extend((0..m).map(|i| (1usize, if i == 0 { b1 } else { d1[i - 1] }, p1 + i)));
        self.pack2_stage(sc, &rows)?;
        let mut dv = vec![0i32; sc.d_dev.len()];
        for (j, &d) in d0.iter().enumerate() { dv[j] = d; }
        for (j, &d) in d1.iter().enumerate() { dv[m + j] = d; }
        self.dev.htod_copy_into(dv, &mut sc.d_dev)?;
        // Pad to PACK_SEG_MAX segments: cudarc's copy asserts exact length, and the
        // seg kernels only read blockIdx.y < k_seg rows (the padding is inert).
        let mut seg_flat: Vec<i32> = segs.iter().flat_map(|s| s.iter().cloned()).collect();
        seg_flat.resize(PACK_SEG_MAX * 4, 0);
        assert_eq!(seg_flat.len(), sc.seg.len());
        self.dev.htod_copy_into(seg_flat, &mut sc.seg)?;
        self.verify_block_packed(l, sc, &segs, r, 2, qsa)?;
        self.dsync()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        {
            let mut h = self.ple_hist.lock().unwrap();
            h[0] = hist_save.0;
            h[1] = hist_save.1;
            h[2] = hist_save.2;
            h[3] = hist_save.3;
        }
        let xf = self.dtoh(&sc.x)?;
        let lgf = self.dtoh(&sc.logits)?;
        let cf = self.dtoh(&sc.core)?;
        let amf = self.dtoh(&sc.argmax)?;
        let outf = self.dtoh(&sc.accept_out)?;
        let accf = self.dtoh(&sc.acc_seg)?;
        let hrow = self.cfg.hidden_size;
        let vrow = self.cfg.vocab_size;
        let crow = self.core_row();
        Ok(Pack2Packed {
            core_a: cf[..m * crow].to_vec(),
            core_b: cf[m * crow..r * crow].to_vec(),
            x_a: xf[..m * hrow].to_vec(),
            x_b: xf[m * hrow..r * hrow].to_vec(),
            logits_a: lgf[..m * vrow].to_vec(),
            logits_b: lgf[m * vrow..r * vrow].to_vec(),
            argmax_a: amf[..m].to_vec(),
            argmax_b: amf[m..r].to_vec(),
            out_a: outf[..m].to_vec(),
            out_b: outf[m..r].to_vec(),
            acc_a: (accf[0], accf[1]),
            acc_b: (accf[2], accf[3]),
            a_a: accf[1] as usize,
            a_b: accf[3] as usize,
            ms,
        })
    }

    fn core_row(&self) -> usize {
        let cfg = &self.tc;
        cfg.lin_num_v_heads * cfg.lin_v_dim
    }

    /// PACK1 LEG 1: the eager packed-verify probe (--probe-exl3-pack2).
    /// Pairs of synthetic prompts prime slots 0/1; then per rep: (a) two lone 5-row
    /// verifies, (b) ONE packed 10-row verify, (c) the lones re-run (ABAB drift check).
    /// Per-segment outputs (GDN core rows, final hidden, logits, argmax, accept_out,
    /// accept keys) are compared BITWISE against the lone verify of the same rep.
    pub fn probe_pack2(&self) -> Result<()> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let k = 4usize;
        let m = k + 1;
        let v = self.cfg.vocab_size;
        let mut sc = Scratch::new(&self.dev, &self.cfg, 16, self.cfg.rotary_dim, self.max_pos)?;
        crate::dprintln!("PACK2: eager packed-verify probe (2 requests x {m} rows, scratch w=16, nostore, eager)");
        let pairs = [(97usize, 89usize), (513usize, 523usize), (2201usize, 2201usize)];
        let mut ratios: Vec<f64> = Vec::new();
        let mut all_id_ok = true;
        let mut lone_b_debug: (Vec<i32>, usize) = (vec![], 0);
        for (pi, &(la, lb)) in pairs.iter().enumerate() {
            let synth = |n: usize, salt: usize| -> Vec<u32> {
                (0..n).map(|i| ((i * 7919 + salt * 104729 + 12345) % (v - 2000) + 1000) as u32).collect()
            };
            let pa = synth(la, 1);
            let pb = synth(lb, 2);
            self.reset_slot(0)?;
            self.reset_slot(1)?;
            let mut psc: Option<PrefillScratch> = None;
            let mut prime = |slot: usize, pr: &[u32], sc: &mut Scratch,
                             psc: &mut Option<PrefillScratch>| -> Result<(i32, usize)> {
                let n = pr.len();
                let mut pos = 0usize;
                while pos + 1 < n {
                    let end = (pos + 1024).min(n - 1);
                    let toks: Vec<i32> = pr[pos..end].iter().map(|&t| t as i32).collect();
                    if psc.is_none() { *psc = Some(self.prefill_scratch(1024)?); }
                    self.prefill_chunk(psc.as_mut().unwrap(), &toks, pos, slot, false, None)?;
                    pos = end;
                }
                self.forward_step(sc, &[pr[n - 1] as i32], &[n - 1], &[slot], None)?;
                let b = self.dtoh(&sc.argmax)?[0];
                Ok((b, n))
            };
            let (b0, p0) = prime(0, &pa, &mut sc, &mut psc)?;
            let (b1, p1) = prime(1, &pb, &mut sc, &mut psc)?;
            let qa = self.qsa_live(p0 + k);
            let qb = self.qsa_live(p1 + k);
            crate::dprintln!("PACK2 PAIR {pi}: len {la}/{lb} pos {p0}/{p1} b {b0}/{b1} qsa {qa}/{qb} agree={}",
                     qa == qb);
            anyhow::ensure!(qa == qb, "PACK2: mixed QSA regimes in pair {pi} (design: fall back to serial)");
            let drafts0: Vec<i32> =
                (0..k).map(|j| (((b0 as usize) * 31 + j * 7 + 101) % (v - 16)) as i32).collect();
            let drafts1: Vec<i32> =
                (0..k).map(|j| (((b1 as usize) * 31 + j * 7 + 103) % (v - 16)) as i32).collect();
            let snap_pre = self.pack2_snap_state(&l, &[0, 1])?;
            for rep in 0..3usize {
                // (a) two serial lone verifies
                let la_out = self.pack2_lone(&l, &mut sc, 0, b0, p0, &drafts0, k, qa)?;
                let lb_out = self.pack2_lone(&l, &mut sc, 1, b1, p1, &drafts1, k, qb)?;
                let a_ms = la_out.ms + lb_out.ms;
                // (b) ONE packed 10-row verify
                let pk = self.pack2_packed(&l, &mut sc, b0, p0, &drafts0, b1, p1, &drafts1, k, qa)?;
                let b_ms = pk.ms;
                // (c) the lones re-run (drift check)
                let lc_out = self.pack2_lone(&l, &mut sc, 0, b0, p0, &drafts0, k, qa)?;
                let ld_out = self.pack2_lone(&l, &mut sc, 1, b1, p1, &drafts1, k, qb)?;
                let c_ms = lc_out.ms + ld_out.ms;
                lone_b_debug = (ld_out.out.clone(), ld_out.a);
                // ---- identity: packed segments vs this rep's lone verifies ----
                let seg_a_ok = la_out.core == pk.core_a && la_out.x == pk.x_a
                    && la_out.logits == pk.logits_a && la_out.argmax == pk.argmax_a
                    && la_out.acc == [pk.acc_a.0, pk.acc_a.1] && la_out.out[..=la_out.a] == pk.out_a[..=pk.a_a];
                let seg_b_ok = lb_out.core == pk.core_b && lb_out.x == pk.x_b
                    && lb_out.logits == pk.logits_b && lb_out.argmax == pk.argmax_b
                    && lb_out.acc == [pk.acc_b.0, pk.acc_b.1] && lb_out.out[..=lb_out.a] == pk.out_b[..=pk.a_b];
                let drift = la_out.x == lc_out.x && lb_out.x == ld_out.x
                    && la_out.logits == lc_out.logits && lb_out.logits == ld_out.logits;
                all_id_ok &= seg_a_ok && seg_b_ok && drift;
                let b_core = lb_out.core == pk.core_b;
                let b_x = lb_out.x == pk.x_b;
                let b_lg = lb_out.logits == pk.logits_b;
                let b_am = lb_out.argmax == pk.argmax_b;
                let b_out = lb_out.out[..=lb_out.a] == pk.out_b[..=pk.a_b];
                let a_lg = la_out.logits == pk.logits_a;
                let a_am = la_out.argmax == pk.argmax_a;
                crate::dprintln!("PACK2 IDENT pair {pi} rep {rep}: segA {seg_a_ok} segB {seg_b_ok} drift {drift} | \
segB bufs core {b_core} x {b_x} lg {b_lg} am {b_am} out {b_out} | segA lg {a_lg} am {a_am} | \
A(core {:016x} x {:016x} lg {:016x} am {:016x}) B(core {:016x} x {:016x} lg {:016x} am {:016x}) | \
P(coreA {:016x} coreB {:016x} xA {:016x} xB {:016x} lgA {:016x} lgB {:016x} amB {:016x}) | \
acc A {:?} B {:?} packedA {:?} packedB {:?}",
                         fnv64_u16(&la_out.core), fnv64_u16(&la_out.x), fnv64_u16(&la_out.logits),
                         fnv64_i32(&la_out.argmax), fnv64_u16(&lb_out.core), fnv64_u16(&lb_out.x),
                         fnv64_u16(&lb_out.logits), fnv64_i32(&lb_out.argmax),
                         fnv64_u16(&pk.core_a), fnv64_u16(&pk.core_b), fnv64_u16(&pk.x_a),
                         fnv64_u16(&pk.x_b), fnv64_u16(&pk.logits_a), fnv64_u16(&pk.logits_b),
                         fnv64_i32(&pk.argmax_b), la_out.acc, lb_out.acc, pk.acc_a, pk.acc_b);
                let r = a_ms / b_ms;
                ratios.push(r);
                crate::dprintln!("PACK2 TIME pair {pi} rep {rep}: a {a_ms:.2} ms  b {b_ms:.2} ms  c {c_ms:.2} ms  r {r:.3}");
                // G3: live state must be untouched after a + b + c
                let snap_post = self.pack2_snap_state(&l, &[0, 1])?;
                let diffs = snap_pre.iter().zip(snap_post.iter()).filter(|(x, y)| x != y).count();
                if rep == 0 {
                    println!("PACK2 G3 pair {pi}: live-state words differing after a+b+c = {diffs} (must be 0)");
                }
                anyhow::ensure!(diffs == 0, "PACK2 G3 FAIL: live state moved (nostore violated)");
            }
        }
        // ---- pad-route synthetic self-test ----
        let topk = 10usize;
        let ids_v: Vec<i32> = (0..5 * topk).map(|i| ((i / topk) * 100 + (i % topk)) as i32).collect();
        let wts_v: Vec<f32> = (0..5 * topk).map(|i| (i % topk) as f32 * 0.5).collect();
        let mut ids = self.dev.htod_sync_copy(&ids_v)?;
        let mut wts = self.dev.htod_sync_copy(&wts_v)?;
        let pgrid = (((2 * topk + 255) / 256).max(1) as u32, 1, 1);
        xqlaunch!(l, "xq_pad_route", pgrid, (256, 1, 1), 0,
                  (&mut ids, &mut wts, topk as i32, 3i32, 2i32))?;
        let ids_h = self.dtoh(&ids)?;
        let r2 = &ids_h[2 * topk..3 * topk];
        let r3 = &ids_h[3 * topk..4 * topk];
        let r4 = &ids_h[4 * topk..5 * topk];
        let eq32 = r3 == r2;
        let eq42 = r4 == r2;
        let pad_ok = eq32 && eq42;
        crate::dprintln!("PACK2 PADROUTE: pas={}(rows 3,4 must equal row 2) eq32 {eq32} eq42 {eq42}", pad_ok);
        crate::dprintln!("PACK2 PADR2 {r2:?}");
        crate::dprintln!("PACK2 PADR3 {r3:?}");
        crate::dprintln!("PACK2 PADR4 {r4:?}");
        let raw_out = self.dtoh(&sc.accept_out)?;
        crate::dprintln!("PACK2 OUTCHK: last packed accept_out = {:?} (lone B had {:?})", &raw_out[..10], lone_b_debug);
        let r_mean = ratios.iter().sum::<f64>() / ratios.len().max(1) as f64;
        let r_min = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("PACK2 SUMMARY: r_mean {r_mean:.3} r_min {r_min:.3} over {} reps \
(decision: >=1.30 GO, 1.20-1.30 borderline, <1.20 STOP)", ratios.len());
        anyhow::ensure!(pad_ok, "PACK2: pad_route self-test failed");
        anyhow::ensure!(all_id_ok, "PACK2: per-segment bit-identity failed");
        Ok(())
    }

}
