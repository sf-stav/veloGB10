//! Probe-only: the `packstate_*` snapshot / restore / dump helpers and `probe_packstate`. See `mod.rs` for the
//! file map and `pack.rs` for the production packed path.

use super::*;

impl FwdModel {
    /// LEG 2 state snapshot: EVERYTHING the next round consumes, serialized
    /// deterministically — trunk GDN/conv rows per layer, PLE row, trunk KV rows
    /// 0..rows per attention layer, trunk QSA raw+pooled planes (rows-h prefix),
    /// taps windows (whole buffer), host ple_hist, head KV + head QSA planes.
    pub(super) fn packstate_snap(&self, l: &Launcher, sc: &Scratch, slots: &[usize], rows: usize) -> Result<Vec<u64>> {
        let cfg = &self.tc;
        let he = cfg.hc_count.max(1) * cfg.hidden_size;
        let s_e = (cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim) as i64;
        let c_e = ((cfg.key_dim() * 2 + cfg.value_dim()) * cfg.conv_kernel) as i64;
        let p_e = (10240 * 9) as i64;
        let nkv = cfg.num_kv_heads;
        let rb = kv_rowbytes(self.kv_fmt, cfg.head_dim);
        let hdx = cfg.indexer_head_dim;
        let ratio = cfg.indexer_compress_ratio.max(1);
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        let rows = rows.min(self.max_pos) as i64;
        let kv_w = (rows as usize * rb / 2) as i64;      // u16 words of one slot's KV prefix
        let qa_w = if hdx > 0 { (rows as usize * hdx) as i64 } else { 0 };
        let tmp_w = kv_w.max(qa_w).max(1);
        let mut s_tmp = self.dev.alloc_zeros::<f32>(s_e as usize)?;
        let mut c_tmp = self.dev.alloc_zeros::<f32>(c_e as usize)?;
        let mut p_tmp = self.dev.alloc_zeros::<u16>(p_e as usize)?;
        let mut kv_tmp = self.dev.alloc_zeros::<u16>(tmp_w as usize)?;
        let mut out = Vec::new();
        out.push(rows as u64);
        {
            let nkv_layers = self.k_cache.len();
            crate::deprintln!(
                "SNAPMAP rows {rows} s_e {s_e} c_e {c_e} ple {p_e} taps {} kv_layers {nkv_layers} kv_w {kv_w} slots {}",
                sc.vrows * he, slots.len());
        }
        for &slot in slots {
            for li in 0..self.s_state.len() {
                xqlaunch!(l, "xq_copy_f32", grid(s_e), (256, 1, 1), 0,
                          (&mut s_tmp, &self.s_state[li], s_e, 0i64, (slot as i64) * s_e))?;
                for f in self.dtoh(&s_tmp)? { out.push(f.to_bits() as u64); }
                xqlaunch!(l, "xq_copy_f32", grid(c_e), (256, 1, 1), 0,
                          (&mut c_tmp, &self.conv_state[li], c_e, 0i64, (slot as i64) * c_e))?;
                for f in self.dtoh(&c_tmp)? { out.push(f.to_bits() as u64); }
            }
            xqlaunch!(l, "xq_copy_u16", grid(p_e), (256, 1, 1), 0,
                      (&mut p_tmp, &self.ple_state, p_e, 0i64, (slot as i64) * p_e))?;
            for u in self.dtoh(&p_tmp)? { out.push(u as u64); }
            // this slot's taps window (rows 0..MTP_MAX_K+1): round 0's tap comes
            // from the prime, later rounds' from the previous verify — persistent.
            {
                let he64 = he as i64;
                let vr = sc.vrows as i64;
                let mut t_tmp = self.dev.alloc_zeros::<f32>(sc.vrows * he)?;
                xqlaunch!(l, "xq_copy_f32", grid(vr * he64), (256, 1, 1), 0,
                          (&mut t_tmp, &sc.taps_keep, vr * he64, 0i64,
                           (slot as i64) * (vr * he64)))?;
                for f in self.dtoh(&t_tmp)? { out.push(f.to_bits() as u64); }
            }
            // trunk KV prefix per attention layer (row bytes are % 16 -> u16 copies)
            let kv_src = ((slot as i64) * 2 * (nkv as i64) * (self.max_pos as i64)) * (rb as i64) / 2;
            for kc in self.k_cache.iter() {
                xqlaunch!(l, "xq_copy_u16", grid(kv_w), (256, 1, 1), 0,
                          (&mut kv_tmp, (*kc.device_ptr()) as u64, kv_w, 0i64, kv_src))?;
                for u in self.dtoh(&kv_tmp)?.iter().take(kv_w as usize) {
                    out.push(*u as u64);
                }
            }
            // trunk QSA planes (raw keys rows*hdx, pooled rows/ratio*hdx)
            if hdx > 0 {
                for att in 0..self.k_cache.len() {
                    if let Some(k) = self.qsa_keys.get(att) {
                        let n = (rows as usize * hdx) as i64;
                        let src = (slot as i64) * (self.max_pos as i64) * (hdx as i64);
                        xqlaunch!(l, "xq_copy_u16", grid(n), (256, 1, 1), 0,
                                  (&mut kv_tmp, (*k.device_ptr()) as u64, n, 0i64, src))?;
                        for u in self.dtoh(&kv_tmp)?.iter().take(n as usize) {
                            out.push(*u as u64);
                        }
                    }
                    if let Some(k) = self.qsa_pool.get(att) {
                        let n = ((rows as usize / ratio) * hdx) as i64;
                        let src = (slot as i64) * ((self.max_pos as i64) / (ratio as i64)) * (hdx as i64);
                        xqlaunch!(l, "xq_copy_u16", grid(n), (256, 1, 1), 0,
                                  (&mut kv_tmp, (*k.device_ptr()) as u64, n, 0i64, src))?;
                        for u in self.dtoh(&kv_tmp)?.iter().take(n as usize) {
                            out.push(*u as u64);
                        }
                    }
                }
            }
        }
        // head KV + head QSA (the draft side's persistent state)
        if let Some(h) = self.mtp.as_ref() {
            for &slot in slots {
                let kv_src = ((slot as i64) * 2 * (nkv as i64) * (self.max_pos as i64)) * (rb as i64) / 2;
                xqlaunch!(l, "xq_copy_u16", grid(kv_w), (256, 1, 1), 0,
                          (&mut kv_tmp, (*h.kv.device_ptr()) as u64, kv_w, 0i64, kv_src))?;
                for u in self.dtoh(&kv_tmp)?.iter().take(kv_w as usize) {
                    out.push(*u as u64);
                }
                if hdx > 0 {
                    if let Some(k) = h.qsa_keys.as_ref() {
                        let n = (rows as usize * hdx) as i64;
                        let src = (slot as i64) * (self.max_pos as i64) * (hdx as i64);
                        xqlaunch!(l, "xq_copy_u16", grid(n), (256, 1, 1), 0,
                                  (&mut kv_tmp, (*k.device_ptr()) as u64, n, 0i64, src))?;
                        for u in self.dtoh(&kv_tmp)?.iter().take(n as usize) {
                            out.push(*u as u64);
                        }
                    }
                    if let Some(k) = h.qsa_pool.as_ref() {
                        let n = ((rows as usize / ratio) * hdx) as i64;
                        let src = (slot as i64) * ((self.max_pos as i64) / (ratio as i64)) * (hdx as i64);
                        xqlaunch!(l, "xq_copy_u16", grid(n), (256, 1, 1), 0,
                                  (&mut kv_tmp, (*k.device_ptr()) as u64, n, 0i64, src))?;
                        for u in self.dtoh(&kv_tmp)?.iter().take(n as usize) {
                            out.push(*u as u64);
                        }
                    }
                }
            }
        }
        // host PLE ring pair (both arms must hold it equal). taps_keep is
        // deliberately EXCLUDED: every round's verify fully rewrites rows 0..m of
        // each window before any read (tap_row <= k < m is that same round's row),
        // so stale window contents cannot affect a round — token+accept+state
        // equality over 20 rounds proves the taps stayed in sync.
        let hist = self.ple_hist.lock().unwrap();
        for j in 0..self.width * 2 { out.push(hist[j] as i64 as u64); }
        Ok(out)
    }

    /// One request's draft chain (the served draft_block with mtp_round's meta +
    /// QSA staging), from the tap already restored into sc.resid row 0. Returns
    /// the clamped drafts (the served clamp: out-of-range -> 0, verify rejects).
    pub(super) fn packstate_draft(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead,
                       slot: usize, k: usize, b: i32, p: usize) -> Result<Vec<i32>> {
        self.dev.htod_copy_into(vec![b, p as i32, slot as i32, sc.rq_dctr], &mut sc.draft_meta)?;
        let dqsa = self.qsa_live(p + k - 1);
        let (dg, _) = self.qsa_grid_bucket_key(p + k - 1);
        sc.qsa_grid_nblk = dg;
        // This PROBE stages slot_ids for the chain UNCONDITIONALLY (the head key write targets
        // slot_ids[b]'s plane even in the dense regime). Since ISSUES 128 the SERVED drafts
        // (`mtp_draft_adaptive`, the fixed chain) do the same (they used to stage them only for a
        // sparse last pass, writing a dense round's head QSA rows at the previous op's slot).
        self.dev.htod_copy_into(vec![slot as i32; sc.slot_ids.len()], &mut sc.slot_ids)?;
        self.draft_block(l, sc, head, k, dqsa)?;
        self.dsync()?;
        let d_buf = self.dtoh(&sc.d_dev)?;
        let v = self.cfg.vocab_size as i32;
        let mut drafts: Vec<i32> = d_buf[..k].to_vec();
        for d in drafts.iter_mut() { if *d < 0 || *d >= v { *d = 0; } }
        Ok(drafts)
    }

    /// Restore the round's tap (mtp_round's step 1): sc.resid row 0 <- taps_keep[slot].tap_row.
    pub(super) fn packstate_tap_restore(&self, l: &Launcher, sc: &mut Scratch, slot: usize, tap_row: usize) -> Result<()> {
        if tap_row == usize::MAX { return Ok(()); } // control-round marker: resid already holds it
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let tb = self.tap_base(sc, slot)?;
        xqlaunch!(l, "xq_copy_f32", ((((he as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.resid, &sc.taps_keep, he as i64, 0i64, (tb + tap_row * he) as i64))?;
        Ok(())
    }

    /// One SERIAL lone round for one request: the served verify_block (taps
    /// snapshot -> accept -> the W4S/per-layer commit). Returns accept_out.
    fn packstate_verify_lone(&self, l: &Launcher, sc: &mut Scratch, slot: usize, k: usize,
                             b: i32, p: usize, drafts: &[i32]) -> Result<Vec<i32>> {
        let m = k + 1;
        let rows: Vec<(usize, i32, usize)> =
            (0..m).map(|r| (slot, if r == 0 { b } else { drafts[r - 1] }, p + r)).collect();
        self.pack2_stage(sc, &rows)?;
        let mut dv = vec![0i32; sc.d_dev.len()];
        dv[..k].copy_from_slice(drafts);
        self.dev.htod_copy_into(dv, &mut sc.d_dev)?;
        let qsa = self.qsa_live(p + k);
        let (qg, _) = self.qsa_grid_bucket_key(p + k);
        sc.qsa_grid_nblk = qg;
        self.verify_block(l, sc, slot, m, k, qsa)?;
        self.dsync()?;
        Ok(self.dtoh(&sc.accept_out)?)
    }

    /// ONE packed round for two requests: staging for both, ONE R_pad verify with
    /// per-segment taps + accept + commit. Returns (accept_out, acc_seg).
    fn packstate_verify_packed(&self, l: &Launcher, sc: &mut Scratch,
                               rows: &[(usize, i32, usize)], d0: &[i32], d1: &[i32],
                               k: usize, pmax: usize) -> Result<(Vec<i32>, Vec<i32>)> {
        let m = k + 1;
        let r = rows.len();
        anyhow::ensure!(r == 2 * m, "packstate: packed rows {} != 2 x {m}", r);
        let segs: [[i32; 4]; 2] =
            [[rows[0].0 as i32, 0, m as i32, 1], [rows[m].0 as i32, m as i32, m as i32, 1]];
        self.pack2_stage(sc, rows)?;
        let mut dv = vec![0i32; sc.d_dev.len()];
        anyhow::ensure!(dv.len() >= m + k, "packstate: d_dev narrower than 2 x k");
        dv[..k].copy_from_slice(d0);
        dv[m..m + k].copy_from_slice(d1);
        self.dev.htod_copy_into(dv, &mut sc.d_dev)?;
        let mut seg_flat: Vec<i32> = segs.iter().flat_map(|s| s.iter().cloned()).collect();
        seg_flat.resize(PACK_SEG_MAX * 4, 0);
        self.dev.htod_copy_into(seg_flat, &mut sc.seg)?;
        let qsa = self.qsa_live(pmax + k);
        let (qg, qb) = self.qsa_grid_bucket_key(pmax + k);
        sc.qsa_grid_nblk = qg;
        // PACK3 STEP B (--pack3-graphs): replay the packed verify through the
        // R_pad graph family (lazy capture on first use; the graphable body is
        // verify_kernels(seg)+accept_seg+commit_seg, the slot/row0/len-baked
        // PLE shadow + taps run as the eager tail). Default flag off = the
        // PACK2 eager block, byte for byte.
        if crate::opts::on(crate::opt!("pack3-graphs")) {
            let pkey = pkeyv_of(r, qsa, qb) | sk(sc);
            if !self.graphs.lock().unwrap().contains_key(&pkey) {
                self.dsync()?;
                self.capture_verify_graph_packed(l, sc, r, 2, qsa, pkey);
            }
            if self.graphs.lock().unwrap().contains_key(&pkey) {
                self.pack_eager_tail_pre(l, sc, &segs)?;
                self.graphs.lock().unwrap().get(&pkey).unwrap().launch();
                self.pack_eager_tail_post(l, sc, &segs)?;
                self.dsync()?;
                return Ok((self.dtoh(&sc.accept_out)?, self.dtoh(&sc.acc_seg)?));
            }
            // rel/cutb (v0.7.3): per-round anomaly marker moved to logq — a direct println! on the
            // scheduler thread blocks on a full stdout pipe (GATES1/FIXC direct-print rule).
            crate::drprintln!("EXL3_GRAPH: packed graph missing at rpad={r} — eager fallback this round");
        }
        self.verify_block_packed_full(l, sc, &segs, r, 2, qsa)?;
        self.dsync()?;
        Ok((self.dtoh(&sc.accept_out)?, self.dtoh(&sc.acc_seg)?))
    }

    /// The served host ring restore (mtp_verify's rule) for one request.
    fn packstate_hist_restore(&self, slot: usize, hist_pre: (i32, i32), toks_v: &[i32],
                              a: usize, b: i32) {
        let mut hist = self.ple_hist.lock().unwrap();
        if a == 0 {
            hist[slot * 2] = hist_pre.1;
            hist[slot * 2 + 1] = b;
        } else {
            hist[slot * 2] = toks_v[a - 1];
            hist[slot * 2 + 1] = toks_v[a];
        }
    }

    /// Save the shadow: d2d copies of every persistent region (GDN/conv slot
    /// rows, PLE row, trunk KV + QSA prefix [0, cap), head KV + QSA prefix) plus
    /// the host ple_hist. Restoring replays the copies back (bitwise).
    pub(super) fn packstate_save(&self, l: &Launcher, sc: &Scratch, slots: &[usize], cap: usize) -> Result<PackShadow> {
        let cfg = &self.tc;
        let he = cfg.hc_count.max(1) * cfg.hidden_size;
        let sc_taps = &sc.taps_keep;
        let s_e = (cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim) as i64;
        let c_e = ((cfg.key_dim() * 2 + cfg.value_dim()) * cfg.conv_kernel) as i64;
        let nkv = cfg.num_kv_heads;
        let rb = kv_rowbytes(self.kv_fmt, cfg.head_dim) as i64;
        let hdx = cfg.indexer_head_dim as i64;
        let ratio = cfg.indexer_compress_ratio.max(1) as i64;
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        let cap = cap.min(self.max_pos) as i64;
        let mut specs: Vec<(u64, i64, i64)> = Vec::new(); // (dst, n u16, src_off)
        for &slot in slots {
            let sl = slot as i64;
            for ss in &self.s_state {
                specs.push(((*ss.device_ptr()) as u64, s_e * 2, sl * s_e * 2));
            }
            for cs in &self.conv_state {
                specs.push(((*cs.device_ptr()) as u64, c_e * 2, sl * c_e * 2));
            }
            specs.push(((*self.ple_state.device_ptr()) as u64, 10240 * 9, sl * (10240 * 9)));
            specs.push(((*sc_taps.device_ptr()) as u64, (sc.vrows as i64) * (he as i64) * 2,
                        sl * ((sc.vrows as i64) * (he as i64) * 2)));
            let kv_n = cap * rb / 2;
            let kv_base = (sl * 2 * (nkv as i64) * (self.max_pos as i64)) * rb / 2;
            for kc in self.k_cache.iter() {
                specs.push(((*kc.device_ptr()) as u64, kv_n, kv_base));
            }
            if hdx > 0 {
                let kn = cap * hdx;
                let kb = sl * (self.max_pos as i64) * hdx;
                for k in self.qsa_keys.iter() {
                    specs.push(((*k.device_ptr()) as u64, kn, kb));
                }
                let pn = (cap / ratio) * hdx;
                let pb = sl * ((self.max_pos as i64) / ratio) * hdx;
                for k in self.qsa_pool.iter() {
                    specs.push(((*k.device_ptr()) as u64, pn, pb));
                }
            }
        }
        if let Some(h) = self.mtp.as_ref() {
            for &slot in slots {
                let sl = slot as i64;
                let kv_n = cap * rb / 2;
                let kv_base = (sl * 2 * (nkv as i64) * (self.max_pos as i64)) * rb / 2;
                specs.push(((*h.kv.device_ptr()) as u64, kv_n, kv_base));
                if hdx > 0 {
                    let kn = cap * hdx;
                    let kb = sl * (self.max_pos as i64) * hdx;
                    if let Some(k) = h.qsa_keys.as_ref() {
                        specs.push(((*k.device_ptr()) as u64, kn, kb));
                    }
                    let pn = (cap / ratio) * hdx;
                    let pb = sl * ((self.max_pos as i64) / ratio) * hdx;
                    if let Some(k) = h.qsa_pool.as_ref() {
                        specs.push(((*k.device_ptr()) as u64, pn, pb));
                    }
                }
            }
        }
        let mut parts = Vec::with_capacity(specs.len());
        for (dst, n, soff) in specs {
            let mut tmp = self.dev.alloc_zeros::<u16>(n as usize)?;
            xqlaunch!(l, "xq_copy_u16", grid(n), (256, 1, 1), 0,
                      (&mut tmp, dst, n, 0i64, soff))?;
            parts.push((dst, soff, tmp));
        }
        let hist = self.ple_hist.lock().unwrap().clone();
        Ok(PackShadow { parts, hist })
    }

    /// Restore the shadow: the same copies replayed back (u16 words are byte-exact).
    pub(super) fn packstate_restore(&self, l: &Launcher, sh: &PackShadow) -> Result<()> {
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        for (dst, doff, tmp) in sh.parts.iter() {
            let n = tmp.len() as i64;
            xqlaunch!(l, "xq_copy_u16", grid(n), (256, 1, 1), 0,
                      (*dst, tmp, n, *doff, 0i64))?;
        }
        self.ple_hist.lock().unwrap().copy_from_slice(&sh.hist);
        Ok(())
    }

    /// PACK2 diagnostic: the region layout of packstate_snap's output (name,
    /// start word) for a given rows count — mirrors its push order exactly.
    fn packstate_regions(&self, rows: usize) -> Vec<(String, usize)> {
        let cfg = &self.tc;
        let s_e = (cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim) as usize;
        let c_e = (cfg.key_dim() * 2 + cfg.value_dim()) * cfg.conv_kernel;
        let p_e = 10240 * 9;
        let rb = kv_rowbytes(self.kv_fmt, cfg.head_dim);
        let hdx = cfg.indexer_head_dim as usize;
        let ratio = cfg.indexer_compress_ratio.max(1);
        let rows = rows.min(self.max_pos);
        let kv_w = rows * rb / 2;
        let he = cfg.hc_count.max(1) * cfg.hidden_size;
        let mut out: Vec<(String, usize)> = Vec::new();
        let mut at = 1usize;
        let mut push = |out: &mut Vec<(String, usize)>, at: &mut usize, name: String, n: usize| {
            out.push((name, *at)); *at += n;
        };
        for slot in 0usize..2 {
            for li in 0..self.s_state.len() { push(&mut out, &mut at, format!("s{slot}.{li}"), s_e); }
            for li in 0..self.s_state.len() { push(&mut out, &mut at, format!("c{slot}.{li}"), c_e); }
            push(&mut out, &mut at, format!("ple{slot}"), p_e);
            push(&mut out, &mut at, format!("taps{slot}"), (MTP_MAX_K + 1) * he); // the probes' default scratch (vrows = MTP_MAX_K + 1)
            for att in 0..self.k_cache.len() { push(&mut out, &mut at, format!("kv{slot}.{att}"), kv_w); }
            if hdx > 0 {
                for att in 0..self.k_cache.len() {
                    if att < self.qsa_keys.len() { push(&mut out, &mut at, format!("qk{slot}.{att}"), rows * hdx); }
                    if att < self.qsa_pool.len() { push(&mut out, &mut at, format!("qp{slot}.{att}"), (rows / ratio) * hdx); }
                }
            }
        }
        if self.mtp.is_some() {
            for slot in 0usize..2 {
                push(&mut out, &mut at, format!("hkv{slot}"), kv_w);
                if hdx > 0 {
                    push(&mut out, &mut at, format!("hqk{slot}"), rows * hdx);
                    push(&mut out, &mut at, format!("hqp{slot}"), (rows / ratio) * hdx);
                }
            }
        }
        push(&mut out, &mut at, "hist".to_string(), self.width * 2);
        out
    }

    /// PACK2 diagnostic: hash the commit INPUT rows (GDN ring, conv raw saves,
    /// PLE normed rows) after a verify so packed-vs-lone staging can be compared
    /// row by row. `packed` selects the buffer layout (16-row planes + absolute
    /// rows vs the served 8-row planes).
    fn packstate_dump_inputs(&self, l: &Launcher, sc: &Scratch, tag: &str, packed: bool,
                             layers: &[usize], nrows: usize) -> Result<()> {
        let cfg = &self.tc;
        let nh = cfg.lin_num_v_heads;
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let rs = GDN_RS; // 260
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        let mut tmp = self.dev.alloc_zeros::<u16>(nh * rs * 2.max(nrows))?;
        // r13 word-level diff: stash LONE B's layer-0 raw rows 0..4 on host, then in the
        // PACKED arm diff PACK plane rows 5..9 against the stash (same-time reads).
        if tag == "LONE" && Self::r13_stash_borrow().is_none() {
            let cd = (self.tc.key_dim() * 2 + self.tc.value_dim()) as usize;
            let n2 = (cd * 2) as i64;
            let mut t = self.dev.alloc_zeros::<u16>(cd * 2)?;
            let mut stash: Vec<Vec<u16>> = Vec::new();
            for r in 0..nrows.min(5) {
                let off = ((*sc.qkv_save.device_ptr()) as u64 + ((r * cd * 2) as u64));
                xqlaunch!(l, "xq_copy_u16", grid(n2), (256, 1, 1), 0, (&mut t, off, n2, 0i64, 0i64))?;
                stash.push(self.dtoh(&t)?);
            }
            *Self::r13_stash_borrow() = Some(stash);
            crate::dprintln!("QKVDIFF: stashed {} LONE B raw rows (layer 0)", nrows.min(5));
        }
        if tag == "PACK" {
            if let Some(stash) = Self::r13_stash_borrow().clone() {
                let cd = (self.tc.key_dim() * 2 + self.tc.value_dim()) as usize;
                let n2 = (cd * 2) as i64;
                let mut t = self.dev.alloc_zeros::<u16>(cd * 2)?;
                for (rp, rl, label) in [(7usize, 2usize, "B-r2"), (8usize, 3usize, "B-r3"), (9usize, 4usize, "B-r4"), (5usize, 0usize, "B-r0"), (6usize, 1usize, "B-r1")] {
                    if rl >= stash.len() { continue; }
                    let off = ((*sc.qkv_save.device_ptr()) as u64 + ((rp * cd * 2) as u64));
                    xqlaunch!(l, "xq_copy_u16", grid(n2), (256, 1, 1), 0, (&mut t, off, n2, 0i64, 0i64))?;
                    let vp = self.dtoh(&t)?;
                    let vl = &stash[rl];
                    let diffs: Vec<usize> = (0..cd * 2).filter(|&i| vp[i] != vl[i]).collect();
                    let mut buckets = [0usize; 10];
                    for &i in diffs.iter() { buckets[(i / 1024).min(9)] += 1; }
                    crate::dprintln!("QKVDIFF2 PACK r{rp} vs LONE r{rl} ({label}): {} differ; buckets {:?}; first {:?} (p,l) {:?}",
                             diffs.len(), buckets, &diffs[..diffs.len().min(6)],
                             diffs.iter().take(6).map(|&i| (vp[i], vl[i])).collect::<Vec<_>>());
                }
                *Self::r13_stash_borrow() = None;
            }
        }
        // resid rows at verify input (he f32 each) — rows 0..nrows (packed: absolute)
        {
            let he = (self.cfg.hc_count.max(1) * self.cfg.hidden_size) as i64;
            let mut rtmp = self.dev.alloc_zeros::<f32>(he as usize)?;
            for r in 0..nrows {
                let off = ((*sc.resid.device_ptr()) as u64 + ((r as u64) * (he as u64) * 4));
                xqlaunch!(l, "xq_copy_f32", grid(he), (256, 1, 1), 0, (&mut rtmp, off, he, 0i64, 0i64))?;
                let hb = self.dtoh(&rtmp)?;
                let mut bytes = Vec::with_capacity((he * 4) as usize);
                for &x in hb.iter() { bytes.extend_from_slice(&x.to_bits().to_le_bytes()); }
                let h = fnv64_u16(bytemuck::cast_slice::<u8, u16>(&bytes));
                crate::dprintln!("INP {tag} resid r{r} {h:016x}");
            }
        }
        for &li in layers {
            for r in 0..nrows {
                let (off, n) = if packed {
                    ((*sc.gdn_ring_p.device_ptr()) as u64
                        + ((li * 16 + r) * nh * rs * 4) as u64, (nh * rs * 2) as i64)
                } else {
                    ((*sc.gdn_ring.device_ptr()) as u64
                        + ((li * sc.vrows + r) * nh * rs * 4) as u64, (nh * rs * 2) as i64)
                };
                xqlaunch!(l, "xq_copy_u16", grid(n), (256, 1, 1), 0, (&mut tmp, off, n, 0i64, 0i64))?;
                let h = fnv64_u16(&self.dtoh(&tmp)?[..n as usize]);
                print!("INP {tag} ring l{li} r{r} {:016x}\n", h);
            }
            for r in 0..nrows {
                let off = ((*sc.qkv_save.device_ptr()) as u64
                    + ((li * SAVE_PLANE_ROWS + r) * conv_dim * 2) as u64);
                let n = (conv_dim * 2) as i64;
                xqlaunch!(l, "xq_copy_u16", grid(n), (256, 1, 1), 0, (&mut tmp, off, n, 0i64, 0i64))?;
                let h = fnv64_u16(&self.dtoh(&tmp)?[..n as usize]);
                print!("INP {tag} qkv l{li} r{r} {:016x}\n", h);
            }
            for r in 0..nrows {
                let off = ((*sc.ple_normed.device_ptr()) as u64 + ((r * 10240) * 2) as u64);
                let n = (10240 * 2) as i64;
                xqlaunch!(l, "xq_copy_u16", grid(n), (256, 1, 1), 0, (&mut tmp, off, n, 0i64, 0i64))?;
                let h = fnv64_u16(&self.dtoh(&tmp)?[..n as usize]);
                print!("INP {tag} ple l{li} r{r} {:016x}\n", h);
            }
        }
        let _ = grid;
        Ok(())
    }

    /// PACK2 r13 word-diff stash (probe-only): the LONE B raw rows on host.
    fn r13_stash_borrow() -> std::sync::MutexGuard<'static, Option<Vec<Vec<u16>>>> {
        static STASH: std::sync::Mutex<Option<Vec<Vec<u16>>>> = std::sync::Mutex::new(None);
        STASH.lock().unwrap()
    }

    /// PACK2 diagnostic: hash the COMMITTED state rows per layer for one slot
    /// (s_state + conv_state) - ground truth without the region map.
    fn packstate_dump_state(&self, l: &Launcher, slot: usize, tag: &str) -> Result<()> {
        let cfg = &self.tc;
        let s_e = (cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim) as i64;
        let c_e = ((cfg.key_dim() * 2 + cfg.value_dim()) * cfg.conv_kernel) as i64;
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        let mut f_tmp = self.dev.alloc_zeros::<u16>((s_e * 2) as usize)?;
        let mut c_tmp = self.dev.alloc_zeros::<u16>((c_e * 2) as usize)?;
        for li in 0..self.s_state.len() {
            let off = (*self.s_state[li].device_ptr()) as u64 + ((slot as u64) * (s_e as u64) * 4);
            xqlaunch!(l, "xq_copy_u16", grid(s_e * 2), (256, 1, 1), 0, (&mut f_tmp, off, s_e * 2, 0i64, 0i64))?;
            let h = fnv64_u16(&self.dtoh(&f_tmp)?[..]);
            crate::dprintln!("ST {tag} s{slot}.l{li} {h:016x}");
        }
        for li in 0..self.conv_state.len() {
            let off = (*self.conv_state[li].device_ptr()) as u64 + ((slot as u64) * (c_e as u64) * 4);
            xqlaunch!(l, "xq_copy_u16", grid(c_e * 2), (256, 1, 1), 0, (&mut c_tmp, off, c_e * 2, 0i64, 0i64))?;
            let h = fnv64_u16(&self.dtoh(&c_tmp)?[..]);
            crate::dprintln!("ST {tag} c{slot}.l{li} {h:016x}");
        }
        Ok(())
    }

    /// PACK2 diagnostic: the round-0 staging as the verify saw it — toks, pos,
    /// and the ple_emb row hashes (the host ring pairs are row inputs).
    fn packstate_dump_stage(&self, l: &Launcher, sc: &Scratch, tag: &str, nrows: usize) -> Result<()> {
        let toks = self.dtoh(&sc.toks)?;
        let pos = self.dtoh(&sc.pos)?;
        let emb = self.dtoh(&sc.ple_emb)?;
        crate::dprintln!("STAGE {tag} toks {:?} pos {:?}", &toks[..nrows], &pos[..nrows]);
        let grid = (1u32, 1, 1);
        let mut tmp = self.dev.alloc_zeros::<u16>(10240)?;
        for r in 0..nrows {
            xqlaunch!(l, "xq_copy_u16", grid, (256, 1, 1), 0,
                      (&mut tmp, ((*sc.ple_emb.device_ptr()) as u64) + ((r * 10240) * 2) as u64,
                       (10240 * 2) as i64, 0i64, 0i64))?;
            let h = fnv64_u16(&self.dtoh(&tmp)?);
            crate::dprintln!("STAGE {tag} emb r{r} {h:016x}");
        }
        Ok(())
    }

    /// PACK2 LEG 2: the packed-ROUND probe (--probe-exl3-packstate).
    /// Real MTP draft chains (the served draft_block per request), ONE packed
    /// verify per round (per-segment accept + device-keyed commit + per-request
    /// re-prime), proven bit-identical to the same rounds run serially (lone
    /// verify per request through the SERVED verify_block): emitted tokens,
    /// accepted counts, and a full state snapshot after N rounds.
    pub fn probe_packstate(&self) -> Result<()> {
        let cfg = &self.tc;
        // the seg commit is the RING commit (the served GDN commit on this geometry)
        anyhow::ensure!(gdn_commit_mode(cfg.lin_k_dim, cfg.lin_v_dim) == 1,
                        "packstate: needs the ring commit mode (gdn_commit_mode == 1)");
        let head = match self.mtp.as_ref() {
            Some(h) => h,
            None => anyhow::bail!("pack has no mtp.* draft head — packstate probe unavailable"),
        };
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let k = 4usize;
        let m = k + 1;
        let rounds = 20usize;
        let mut sc = Scratch::new(&self.dev, &self.cfg, 16, self.cfg.rotary_dim, self.max_pos)?;
        anyhow::ensure!(sc.d_dev.len() >= m + k, "packstate: d_dev narrower than 2 x k");
        crate::dprintln!("PACKSTATE: packed ROUND probe (2 requests, k={k}, {rounds} rounds x 3 pairs; \
                  serial = lone verify per request through the served verify_block; \
                  packed = one R_pad verify, per-segment accept/commit/re-prime)");
        let pairs = [(97usize, 89usize), (513usize, 523usize), (2201usize, 2201usize)];
        let mut unequal_rounds = 0usize;
        for (pi, &(la, lb)) in pairs.iter().enumerate() {
            let synth = |n: usize, salt: usize| -> Vec<u32> {
                (0..n).map(|i| ((i * 7919 + salt * 104729 + 12345) % (cfg.vocab_size - 2000) + 1000) as u32).collect()
            };
            let pa = synth(la, 1);
            let pb = synth(lb, 2);
            // ---- prime (chunked prefill + one forward_step; the PACK1 probe pattern) ----
            // After forward_step, sc.resid row 0 IS the stream after n-1 (the true
            // tap that predicted b) -> tap_snapshot per slot starts round 1 at row 0.
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
            // ---- SERIAL arm: each request runs the SERVED lone round per round ----
            self.reset_slot(0)?;
            self.reset_slot(1)?;
            let mut psc: Option<PrefillScratch> = None;
            let (b0, p0) = prime(0, &pa, &mut sc, &mut psc)?;
            let (b1, p1) = prime(1, &pb, &mut sc, &mut psc)?;
            let cap = p0.max(p1) + rounds * (k + 1) + 8; // every row any round can write
            let snap0 = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
            let shadow = self.packstate_save(&l, &sc, &[0, 1], cap)?;
            let mut st = [(b0, p0, 0usize), (b1, p1, 0usize)];
            let mut tok_s = [Vec::<i32>::new(), Vec::<i32>::new()];
            let mut acc_s = [Vec::<usize>::new(), Vec::<usize>::new()];
            let mut ms_s = Vec::new();
            let mut snaps_s: Vec<Vec<u64>> = Vec::new();
            for r in 0..rounds {
                let mut vms = 0f64;
                for j in 0..2usize {
                    let (b, p, tr) = st[j];
                    self.packstate_tap_restore(&l, &mut sc, j, tr)?;
                    let drafts = self.packstate_draft(&l, &mut sc, head, j, k, b, p)?;
                    if r >= 12 { crate::dprintln!("DRAFTS LONE r{r} j{j}: {drafts:?}"); }
                    let hist_pre = { let h = self.ple_hist.lock().unwrap(); (h[j * 2], h[j * 2 + 1]) };
                    let toks_v: Vec<i32> = std::iter::once(b).chain(drafts.iter().cloned()).collect();
                    let t0 = std::time::Instant::now();
                    let out = self.packstate_verify_lone(&l, &mut sc, j, k, b, p, &drafts)?;
                    if r == 0 {
                        let ring_h = self.dtoh(&sc.gdn_ring)?;
                        let nh = self.tc.lin_num_v_heads as usize;
                        let base = 2 * sc.vrows * nh * GDN_RS;
                        let seg = 5 * nh * GDN_RS;
                        let mut h: u64 = 0xcbf29ce484222325;
                        for &f in &ring_h[base..base + seg] { h = (h ^ (f.to_bits() as u64)).wrapping_mul(0x100000001b3); }
                        crate::deprintln!("RINGHASH L2 serial j{j}: {h:016x}");
                    }
                    vms += t0.elapsed().as_secs_f64() * 1e3;
                    let a = out[k + 2] as usize;
                    anyhow::ensure!(a <= k, "packstate: serial accept out of range");
                    self.packstate_hist_restore(j, hist_pre, &toks_v, a, b);
                    self.head_reprime(&mut sc, head, j, p, &drafts[..a])?;
                    tok_s[j].extend(out[..=a].to_vec());
                    acc_s[j].push(a);
                    st[j] = (out[a], p + a + 1, a);
                }

                ms_s.push(vms);
                if r <= 5 { snaps_s.push(self.packstate_snap(&l, &sc, &[0, 1], st[0].1.max(st[1].1) + 8)?); }
                if r == 0 || r + 1 == rounds {
                    crate::dprintln!("PACKSTATE SERIAL pair {pi} round {r}: a=({},{}) pos=({},{})",
                             acc_s[0][r], acc_s[1][r], st[0].1, st[1].1);
                }
            }
            let pmax_s = st[0].1.max(st[1].1);
            let snap_s = self.packstate_snap(&l, &sc, &[0, 1], pmax_s + 8)?;
            // ---- restore the saved snapshot (bitwise) and run the PACKED arm ----
            self.packstate_restore(&l, &shadow)?;
            let snap0r = self.packstate_snap(&l, &sc, &[0, 1], cap)?;
            let rdiff = snap0.iter().zip(snap0r.iter()).filter(|(x, y)| x != y).count();
            crate::dprintln!("PACKSTATE PAIR {pi}: save/restore diff = {rdiff} words (must be 0)");
            anyhow::ensure!(rdiff == 0, "PACKSTATE pair {pi}: device save/restore not bitwise ({rdiff} words)");
            let mut st = [(b0, p0, 0usize), (b1, p1, 0usize)];
            let mut tok_p = [Vec::<i32>::new(), Vec::<i32>::new()];
            let mut acc_p = [Vec::<usize>::new(), Vec::<usize>::new()];
            let mut ms_p = Vec::new();
            let mut uneq_pair = 0usize;
            let mut uneq_fb = 0usize; // unequal accepts in FALLBACK rounds (not packed evidence)
            for r in 0..rounds {
                crate::deprintln!("PR2 r{r} begin pos {}/{}", st[0].1, st[1].1);
                // One regime AND grid bucket per packed verify (design §3.5/R7: lanes
                // disagreeing on the QSA regime fall back to serial for that round).
                let qa = self.qsa_live(st[0].1 + k);
                let qb = self.qsa_live(st[1].1 + k);
                let (ga, ba) = self.qsa_grid_bucket_key(st[0].1 + k);
                let (gb, bb) = self.qsa_grid_bucket_key(st[1].1 + k);
                if qa != qb || ga != gb || ba != bb {
                    crate::dprintln!("[exl3-serve] lane-pack skip: mixed regime (pair {pi} round {r}, pos {}/{}, bucket {ga}/{gb}) — serial fallback",
                             st[0].1, st[1].1);
                    let mut vms = 0f64;
                    let mut a2 = [0usize; 2];
                    for j in 0..2usize {
                        let (b, p, tr) = st[j];
                        self.packstate_tap_restore(&l, &mut sc, j, tr)?;
                        let drafts = self.packstate_draft(&l, &mut sc, head, j, k, b, p)?;
                        let hist_pre = { let h = self.ple_hist.lock().unwrap(); (h[j * 2], h[j * 2 + 1]) };
                        let toks_v: Vec<i32> = std::iter::once(b).chain(drafts.iter().cloned()).collect();
                        let t1 = std::time::Instant::now();
                        let out = self.packstate_verify_lone(&l, &mut sc, j, k, b, p, &drafts)?;
                        vms += t1.elapsed().as_secs_f64() * 1e3;
                        let a = out[k + 2] as usize;
                        anyhow::ensure!(a <= k, "packstate: fallback accept out of range");
                        self.packstate_hist_restore(j, hist_pre, &toks_v, a, b);
                        self.head_reprime(&mut sc, head, j, p, &drafts[..a])?;
                        tok_p[j].extend(out[..=a].to_vec());
                        acc_p[j].push(a);
                        a2[j] = a;
                        st[j] = (out[a], p + a + 1, a);
                    }
                    if r < 6 && r < snaps_s.len() {
                        let sp = self.packstate_snap(&l, &sc, &[0, 1], st[0].1.max(st[1].1) + 8)?;
                        let d = sp.iter().zip(snaps_s[r].iter()).filter(|(x, y)| x != y).count();
                        anyhow::ensure!(d == 0 && sp.len() == snaps_s[r].len(),
                            "PACKSTATE pair {pi} round {r}: fallback round state diverged ({d} words)");
                    }
                    if a2[0] != a2[1] { uneq_fb += 1; }
                    ms_p.push(vms);
                    if r == 0 || r + 1 == rounds {
                        crate::dprintln!("PACKSTATE PACKED pair {pi} round {r}: a=({},{}) pos=({},{}) [serial-fb]",
                                 a2[0], a2[1], st[0].1, st[1].1);
                    }
                    continue;
                }
                // (i) draft per request — TODAY's single-request chain, sequential
                let mut dss = [Vec::<i32>::new(), Vec::<i32>::new()];
                for j in 0..2usize {
                    let (b, p, tr) = st[j];
                    self.packstate_tap_restore(&l, &mut sc, j, tr)?;
                    dss[j] = self.packstate_draft(&l, &mut sc, head, j, k, b, p)?;
                    if r >= 12 { crate::dprintln!("DRAFTS PACK r{r} j{j}: {:?}", dss[j]); }
                }
                crate::deprintln!("PR2 r{r} drafts ok");
                // (ii) ONE packed verify over both chains
                let mut rows: Vec<(usize, i32, usize)> =
                    (0..m).map(|i| (0usize, if i == 0 { st[0].0 } else { dss[0][i - 1] }, st[0].1 + i)).collect();
                rows.extend((0..m).map(|i| (1usize, if i == 0 { st[1].0 } else { dss[1][i - 1] }, st[1].1 + i)));
                let hp0 = { let h = self.ple_hist.lock().unwrap(); (h[0], h[1]) };
                let hp1 = { let h = self.ple_hist.lock().unwrap(); (h[2], h[3]) };
                let t0 = std::time::Instant::now();
                let (out, accf) = self.packstate_verify_packed(&l, &mut sc, &rows, &dss[0], &dss[1],
                                                               k, st[0].1.max(st[1].1))?;
                crate::deprintln!("PR2 r{r} verify ok {}ms", t0.elapsed().as_millis());
                if r == 0 {
                    let ring_h = self.dtoh(&sc.gdn_ring_p)?;
                    let nh = self.tc.lin_num_v_heads as usize;
                    let base = 2 * 16 * nh * GDN_RS;
                    let seg = 5 * nh * GDN_RS;
                    for half in 0..2usize {
                        let mut h: u64 = 0xcbf29ce484222325;
                        for &f in &ring_h[base + half * seg..base + (half + 1) * seg] { h = (h ^ (f.to_bits() as u64)).wrapping_mul(0x100000001b3); }
                        crate::deprintln!("RINGHASH L2 packed half{half}: {h:016x}");
                    }
                }
                let vms = t0.elapsed().as_secs_f64() * 1e3;
                // r0 dumps stripped: the PACK dump_inputs wedge (probe-only) blocked G2;
                // its evidence (QKVDIFF2 row aliasing) is already banked in run22.
                // (iii)-(v) per-segment accept -> per-segment commit (ran inside the
                // block, device-keyed) -> re-prime per request with ITS rows' hiddens
                let mut a2 = [0usize; 2];
                for j in 0..2usize {
                    let a = accf[j * 2 + 1] as usize;
                    crate::deprintln!("PR3 r{r} j{j} a={a} begin");
                    anyhow::ensure!(a <= k, "packstate: packed accept out of range (seg {j})");
                    let row0 = j * m;
                    let toks_v: Vec<i32> =
                        std::iter::once(st[j].0).chain(dss[j].iter().cloned()).collect();
                    crate::deprintln!("PR3 r{r} j{j} hist_restore begin");
                    self.packstate_hist_restore(j, if j == 0 { hp0 } else { hp1 }, &toks_v, a, st[j].0);
                    crate::deprintln!("PR3 r{r} j{j} hist ok");
                    crate::deprintln!("PR3 r{r} j{j} head_reprime begin n={}", &dss[j][..a].len());
                    self.head_reprime(&mut sc, head, j, st[j].1, &dss[j][..a])?;
                    crate::deprintln!("PR3 r{r} j{j} reprime ok");
                    let emitted = out[row0..=row0 + a].to_vec();
                    tok_p[j].extend(emitted.iter().cloned());
                    acc_p[j].push(a);
                    a2[j] = a;
                    st[j] = (emitted[a], st[j].1 + a + 1, a);
                }
                crate::deprintln!("PR2 r{r} reprime ok");
                if r <= 5 {
                    let sp = self.packstate_snap(&l, &sc, &[0, 1], st[0].1.max(st[1].1) + 8)?;
                    let n = sp.iter().zip(snaps_s[r].iter()).filter(|(x, y)| x != y).count();
                    let lmis = sp.len() != snaps_s[r].len();
                    let didx: Vec<usize> = sp.iter().zip(snaps_s[r].iter()).enumerate()
                        .filter(|(_, (x, y))| x != y).map(|(i, _)| i).take(12).collect();
                    crate::deprintln!("PR4 r{r} snapdiff {n} lenmatch {} first {didx:?}", !lmis);
                    if r == 0 {
                        for &i in didx.iter().take(4) {
                            let a = sp.get(i).copied().unwrap_or(0xDEAD);
                            let b = snaps_s[r].get(i).copied().unwrap_or(0xDEAD);
                            let (af, bf) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
                            crate::deprintln!("PR4V idx {i}: packed {a:#018x} ({af:.6e}) serial {b:#018x} ({bf:.6e})");
                        }
                    }
                }

                if a2[0] != a2[1] { unequal_rounds += 1; uneq_pair += 1; }
                ms_p.push(vms);
                if r == 0 || r + 1 == rounds || a2[0] != a2[1] {
                    crate::dprintln!("PACKSTATE PACKED pair {pi} round {r}: a=({},{}){} pos=({},{})",
                             a2[0], a2[1], if a2[0] != a2[1] { " UNEQUAL" } else { "" }, st[0].1, st[1].1);
                }
            }
            let pmax_p = st[0].1.max(st[1].1);
            let snap_p = self.packstate_snap(&l, &sc, &[0, 1], pmax_p + 8)?;
            // ---- compare ----
            let tok_ok = tok_p.to_vec() == tok_s;
            let acc_ok = acc_p.to_vec() == acc_s;
            let snap_diff = if snap_s.len() == snap_p.len() {
                let n = snap_s.iter().zip(snap_p.iter()).filter(|(x, y)| x != y).count();
                if n > 0 {
                    let didx: Vec<usize> = snap_s.iter().zip(snap_p.iter()).enumerate()
                        .filter(|(_, (x, y))| x != y).map(|(i, _)| i).take(12).collect();
                    crate::dprintln!("PACKSTATE DIAG pair {pi}: snap diff {n} first_idx {didx:?}");
                }
                n
            } else { usize::MAX };
            let ms_s_mean = ms_s.iter().sum::<f64>() / ms_s.len().max(1) as f64;
            let ms_p_mean = ms_p.iter().sum::<f64>() / ms_p.len().max(1) as f64;
            crate::dprintln!("PACKSTATE PAIR {pi} verdict: tokens {tok_ok} accepts {acc_ok} snap_diff {snap_diff} \
                      unequal_rounds {uneq_pair} (fb {uneq_fb}) | serial {ms_s_mean:.2} ms/round, packed {ms_p_mean:.2} ms/round \
                      (verify only, eager) | accS0 {:?} accP0 {:?} accS1 {:?} accP1 {:?}",
                     &acc_s[0][..], &acc_p[0][..], &acc_s[1][..], &acc_p[1][..]);
            anyhow::ensure!(tok_ok, "PACKSTATE pair {pi}: emitted tokens diverged");
            anyhow::ensure!(acc_ok, "PACKSTATE pair {pi}: accepted counts diverged");
            anyhow::ensure!(snap_diff == 0, "PACKSTATE pair {pi}: state snapshot differs in {snap_diff} words");
        }
        crate::dprintln!("PACKSTATE SUMMARY: {rounds} rounds x {} pairs OK; unequal-accept rounds observed: {unequal_rounds} \
                  (must be > 0 for the per-segment commit to be exercised)", pairs.len());
        anyhow::ensure!(unequal_rounds > 0, "PACKSTATE: no unequal-accept round observed — \
                        the per-segment commit was never exercised with a0 != a1");
        Ok(())
    }

}
