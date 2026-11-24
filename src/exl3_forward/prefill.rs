//! The chunked / segmented prefill driver: `prefill_scratch`, `ple_forward_segs`, `prefill_chunk`,
//! `prefill_segs` and the large `prefill_segs_inner`. The served path chunks at the configured width (AGENTS 5).
//! See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    pub fn prefill_scratch(&self, c: usize) -> Result<PrefillScratch> {
        let cfg = &self.cfg;
        let dev = &self.dev;
        let h = cfg.hidden_size;
        let hc = cfg.hc_count.max(1);
        let lr = cfg.hc_lowrank;
        let rw = h * hc;
        let v = cfg.vocab_size;
        // EXL27B: dense packs' widths (see Scratch::new); every other model: the FN constants
        let dense = cfg.num_experts == 1;
        let (kvw, mlpw) = (cfg.num_kv_heads * cfg.head_dim, cfg.moe_intermediate_size);
        let topk = cfg.num_experts_per_tok;
        let r = c * topk;
        let qsa_on = cfg.has_indexer();
        let nblk_stride = (self.max_pos / cfg.indexer_compress_ratio).max(1);
        let sel_max = cfg.indexer_budget + cfg.indexer_compress_ratio;
        let hf = if self.mtp.is_some() { c * rw } else { 1 };
        let ne = cfg.num_experts;
        let (gu_w, d_w) = self.layers.first().map(|ly| (ly.moe.gu_words, ly.moe.d_words)).unwrap_or((0, 0));
        // FP4L: the NVFP4 experts (trellis word counts 0) run ONLY on the device-routed tile lists (fp4l_fwd::moe_prefill)
        let pf_same = gu_w > 0 && self.layers.iter().all(|ly| ly.moe.gu_words == gu_w && ly.moe.d_words == d_w)
            || self.layers.iter().any(|ly| ly.moe.fx.is_some());
        // EXL27B: Qwen3.5-class MoE layers carry their own offset tables (mixed 3/4-bit layers) and
        // need the tile lists whatever pf_same says (FN: split_moe is false, pf_same decides as before)
        let split_moe = self.layers.iter().any(|ly| ly.moe.split.is_some());
        // #26 Fix 3: ANY non-dense MoE layer needs the tile lists -- a mixed-width pack
        // (47xK3+1xK4, 47xK4+1xK5) fails pf_same, and without tile lists the device-routed EP
        // arm is unreachable (the host-routed arm refuses under EP = the #26 wall).
        let any_moe = self.layers.iter().any(|ly| !ly.moe.dense);
        let max_tiles = r / 16 + ne;
        let wp18 = if qsa_on { Wp18Scratch::new(self, c)? } else { None };
        // LEG-B: build the fallback decode Scratch only when a plain MoE layer is fractional.
        let frac_moe = self.layers.iter().any(|ly| !ly.moe.dense && ly.moe.split.is_none() && ly.moe.k2 % 2 == 1);
        let frac_sc = if frac_moe { Some(Scratch::new(dev, cfg, crate::exl3_forward::FRAC_PF_BLK,
                                                          cfg.rotary_dim, self.max_pos)?) } else { None };
        Ok(PrefillScratch {
            frac_sc,
            c,
            pf_tiles: dev.alloc_zeros::<i32>(if pf_same || split_moe || any_moe { 2 * max_tiles } else { 1 })?,
            pf_ntiles: dev.alloc_zeros::<i32>(1)?,
            pf_tiles4: dev.alloc_zeros::<i32>(if pf_same || any_moe { 8 * max_tiles } else { 1 })?,
            pf_ntiles4: dev.alloc_zeros::<i32>(4)?,
            pf_ident: dev.htod_sync_copy(&(0..ne as i32).collect::<Vec<i32>>())?,
            pf_dident: dev.htod_sync_copy(&(0..r as i32).collect::<Vec<i32>>())?,
            hfill_s: dev.alloc_zeros::<f32>(hf)?,
            hfill_a: dev.alloc_zeros::<u16>(hf)?,
            hfill_b: dev.alloc_zeros::<u16>(hf)?,
            hfill_carry: dev.alloc_zeros::<f32>(if self.mtp.is_some() { rw } else { 1 })?,
            hfill_carry_slots: dev.alloc_zeros::<f32>(if self.mtp.is_some() { self.width * rw } else { 1 })?,
            sids: dev.htod_sync_copy(&vec![0i32, 0i32])?,
            qsa_qk: dev.alloc_zeros::<u16>(c * if qsa_on { (cfg.indexer_n_heads + 1) * cfg.indexer_head_dim } else { 0 })?,
            qsa_qkb: dev.alloc_zeros::<u16>(c * if qsa_on { (cfg.indexer_n_heads + 1) * cfg.indexer_head_dim } else { 0 })?,
            qsa_blocks: dev.alloc_zeros::<u16>(if qsa_on { nblk_stride * cfg.indexer_head_dim } else { 0 })?,
            // WP18: the [c][nblk_stride] score matrix (0.54 GB at C = 2048 / 262K) exists only on the
            // old path, or as the XCHECK reference.
            qsa_scores: dev.alloc_zeros::<f32>(c * if qsa_on && (wp18.is_none() || wp18_xcheck()) { nblk_stride } else { 0 })?,
            qsa_sel: dev.alloc_zeros::<i32>(c * if qsa_on { sel_max } else { 0 })?,
            qsa_psel: dev.alloc_zeros::<i32>(c)?,
            wp18,
            pos_rows: dev.alloc_zeros::<i32>(c)?,
            vis_img: None,
            vis_src: Vec::new(),
            vis_rows: dev.alloc_zeros::<i32>(c)?,
            slot_rows: dev.alloc_zeros::<i32>(c)?,
            slotpos_rows: dev.alloc_zeros::<i32>(c * 2)?,
            qstage: dev.alloc_zeros::<f32>(c * cfg.num_heads * cfg.head_dim)?,
            pf_qh: dev.alloc_zeros::<u16>(if cfg.has_indexer() && !pf_tc_qsa_on() { 1 } else { c * cfg.num_heads * cfg.head_dim })?,
            qsa_pm: dev.alloc_zeros::<f32>(c * cfg.num_heads * QSA_SPLITS)?,
            qsa_pl: dev.alloc_zeros::<f32>(c * cfg.num_heads * QSA_SPLITS)?,
            qsa_pacc: dev.alloc_zeros::<f32>(c * cfg.num_heads * QSA_SPLITS * cfg.head_dim)?,
            resid: dev.alloc_zeros::<f32>(c * rw)?,
            hn: dev.alloc_zeros::<u16>(c * rw)?,
            dd: dev.alloc_zeros::<u16>(c * lr)?,
            uu: dev.alloc_zeros::<u16>(c * rw)?,
            inj: if lr == 0 { dev.htod_sync_copy(&vec![1.0f32; c * hc])? } else { dev.alloc_zeros::<f32>(c * hc)? }, // EXL27B: plain residual
            x: dev.alloc_zeros::<u16>(c * h)?,
            x_last: dev.alloc_zeros::<u16>(h)?,
            // S-A3-f-b FIX: same as Scratch — must hold c * max_chain_K
            // (out_proj K=6144; c*2560 caused OOB reads from row c*2560/K).
            chain_xh: dev.alloc_zeros::<u16>(c * if dense { 12288usize.max(h).max(mlpw) } else { 12288 })?,
            // PFX1 (c): every m = c chain has N <= PF_CHAIN_NMAX (asserted in exl3_chain_rb /
            // pf_chain_check), the head fc chains are 4c x 2560, and the vocab-wide lm_head runs
            // at m = 1 — c*vocab (1.02 GB at C = 2048) was never needed; max(c*12288, vocab) is.
            chain_yraw: dev.alloc_zeros::<u16>(if pfx1_on(PFX1_YRAW) { (c * if dense { PF_CHAIN_NMAX.max(mlpw) } else { PF_CHAIN_NMAX }).max(v) } else { c * v })?,
            // S-A3-f-d step 0b FIX (serve smoke, both boxes): this ws is NOT dummy —
            // a final chunk with <= 16 rows runs the decode branch of exl3_chain_impl,
            // where ks>1 chains write split-K partials here ("prefill is m>16" was a
            // stale assumption; short-tail prompts crashed with "chain_ws too small").
            // Same 2M-float sizing as Scratch::new (8 MB).
            chain_ws: dev.alloc_zeros::<f32>(2_000_000)?,
            // S-A3-f-g: reconstruct staging + cuBLASLt workspace (uninitialized is
            // fine: staging is written whole by the decode kernel before any read;
            // the Lt workspace is written by cuBLASLt).
            recon_ws: unsafe { dev.alloc::<u16>(self.recon_staging_elems()) }?,
            lt_ws: unsafe { dev.alloc::<u8>(32 * 1024 * 1024) }?,
            qkv: dev.alloc_zeros::<u16>(c * 10240)?,
            qg: dev.alloc_zeros::<u16>(c * 12288)?,
            kp: dev.alloc_zeros::<u16>(c * if dense { kvw } else { 512 })?,
            vp: dev.alloc_zeros::<u16>(c * if dense { kvw } else { 512 })?,
            zbuf: dev.alloc_zeros::<u16>(c * 6144)?,
            a_out: dev.alloc_zeros::<u16>(c * 48)?,
            b_out: dev.alloc_zeros::<u16>(c * 48)?,
            core: dev.alloc_zeros::<u16>(c * 48 * 128)?,
            normed: dev.alloc_zeros::<u16>(c * 6144)?,
            mixer_out: dev.alloc_zeros::<u16>(c * h)?,
            logits_r: dev.alloc_zeros::<f32>(c * cfg.num_experts)?,
            ids: dev.alloc_zeros::<i32>(r)?,
            wts: dev.alloc_zeros::<f32>(r)?,
            idxmap: dev.alloc_zeros::<i32>(cfg.num_experts)?,
            moe_out: dev.alloc_zeros::<u16>(c * h)?,
            sh1: dev.alloc_zeros::<u16>(c * cfg.moe_intermediate_size)?,
            sh2: dev.alloc_zeros::<u16>(c * cfg.moe_intermediate_size)?,
            ysh: dev.alloc_zeros::<u16>(c * h)?,
            cos: dev.alloc_zeros::<f32>(c * 2 * cfg.rotary_dim)?,
            toks: dev.alloc_zeros::<i32>(c)?,
            xh_e: dev.alloc_zeros::<u16>(r * 2560)?,
            ygu_raw: dev.alloc_zeros::<u16>(r * 1280)?,
            ygu: dev.alloc_zeros::<u16>(r * 1280)?,
            din: dev.alloc_zeros::<u16>(r * 640)?,
            xhd: dev.alloc_zeros::<u16>(r * 640)?,
            yd_raw: dev.alloc_zeros::<u16>(r * 2560)?,
            yd: dev.alloc_zeros::<u16>(r * 2560)?,
            offs_gu: dev.alloc_zeros::<u64>(cfg.num_experts)?,
            offs_d: dev.alloc_zeros::<u64>(cfg.num_experts)?,
            ids_ep: dev.alloc_zeros::<i32>(r)?,
            row_tok: dev.alloc_zeros::<i32>(r)?,
            row_eidx: dev.alloc_zeros::<i32>(r)?,
            cand: dev.alloc_zeros::<i32>(r)?,
            offs_row: dev.alloc_zeros::<i32>(cfg.num_experts + 1)?,
            cnt: dev.alloc_zeros::<i32>(cfg.num_experts)?,
            moerows: dev.alloc_zeros::<i32>(1)?,
            ple_emb: dev.alloc_zeros::<u16>(c * 2560)?,
            ple_key16: dev.alloc_zeros::<u16>(c * 10240)?,
            ple_val: dev.alloc_zeros::<u16>(c * 2560)?,
            ple_kn: dev.alloc_zeros::<f32>(c * 4 * 2560)?,
            ple_qn: dev.alloc_zeros::<f32>(c * 4 * 2560)?,
            ple_gated: dev.alloc_zeros::<u16>(c * 4 * 2560)?,
            ple_normed: dev.alloc_zeros::<u16>(c * 4 * 2560)?,
            logits: dev.alloc_zeros::<u16>(v)?,
            argmax: dev.alloc_zeros::<i32>(1)?,
            wp05: match Wp05PfStage::new(dev, c, self.ple.is_some()) {
                Ok(s) => Some(s),
                Err(e) => { crate::dprintln!("WP05: pinned prefill staging unavailable ({e:#}) — pageable path"); None }
            },
            pf_hook: None,
            pf_hooked: false,
            row_indep: false,
            sp_rows: None,
            ppl: None,
            cpv: None,
            // FP4L: zeroed by an explicit upload (alloc_zeros does not zero, AGENTS §2.2); xf_sg_gate writes
            // columns 0..6 of each row and nothing ever writes the rest
            fxg: if self.fxc.as_ref().is_some_and(|f| f.plain) { Some(dev.htod_sync_copy(&vec![0u16; c * h])?) } else { None },
        })
    }

    /// PACKPF: the PLE block over a (possibly packed) pass — the n-gram history and the conv history
    /// per segment (each its own slot), the row build and the projections once over all rows.
    fn ple_forward_segs(&self, l: &Launcher, psc: &mut PrefillScratch, toks: &[i32], segs: &[PfSeg]) -> Result<()> {
        let slot = segs[0].slot;
        let multi = segs.len() > 1;
        let ple = match self.ple.as_ref() {
            Some(p) => p,
            None => return Ok(()),
        };
        let eps = self.cfg.rms_eps as f32;
        let scale = 1.0f32 / (2560f32).sqrt();
        let c = toks.len();
        let par = wp05_on("par") && psc.ple_emb.len() >= c * 2560
            && psc.wp05.as_ref().map_or(false, |s| s.c >= c && s.pin.len >= s.off_ple() + c * 2560 * 2);
        if par {
            // WP05: the n-gram history chain is the only order-dependent part — build it
            // sequentially (same ring updates as below), then decode the rows on up to 8 threads
            // into the pinned buffer and enqueue ONE async NULL-stream copy into the persistent
            // psc.ple_emb. No htod_sync_copy: the host never drains the device mid-chunk (layer 0
            // is still executing while the rows are built), and no per-chunk allocation.
            let h3s: Vec<[i32; 3]> = {
                let mut hist = self.ple_hist.lock().unwrap();
                let mut v = Vec::with_capacity(c);
                for sg in segs {
                    let slot = sg.slot;
                    for q in sg.r0..sg.r0 + sg.n {
                        v.push([hist[slot * 2], hist[slot * 2 + 1], toks[q]]);
                        hist[slot * 2] = hist[slot * 2 + 1];
                        hist[slot * 2 + 1] = toks[q];
                    }
                }
                v
            };
            let st = psc.wp05.as_mut().unwrap();
            if st.inflight_ple { self.dsync()?; st.inflight_ple = false; } // only after an error path
            let off = st.off_ple();
            let rows = st.pin.u16s(off, c * 2560);
            // Queue every SSD row of the chunk before the row build (same PERFFIX1 advise as the decode and
            // verify paths): the build's synchronous preads then find the pages cached instead of each
            // waiting for its own drive round trip while the GPU idles at the layer-1 boundary. Advisory
            // only, so the rows are unchanged; no-op for the RAM table.
            ple.prefetch_h3s_par(&h3s, 8);
            ple.rows_par(&h3s, rows, false)?;
            if ple_xcheck_on() {
                let mut refr = vec![0u16; c * 2560];
                ple.rows_par(&h3s, &mut refr, true)?;
                let (n, first) = wp05_diff_u16(rows, &refr);
                crate::dprintln!("PLE_XCHECK prefill slot {slot} rows {c}: {n} mismatching f16 of {} (first (idx, new, old) {first:?})",
                         c * 2560);
            }
            st.pin.htod(off, *psc.ple_emb.device_ptr() as u64, c * 2560 * 2)?;
            st.inflight_ple = true;
        } else {
            anyhow::ensure!(!multi, "PACKPF: the host PLE row build is single-segment");
            let mut hist = self.ple_hist.lock().unwrap();
            // S-A3-f-d night FIX: this buffer is sized c*2560 while ple_build_embed
            // writes c rows (c * 2560 u16s) — indexing it directly panicked as soon
            // as the PLE pack was driven through a 1-row, width-1 prefill scratch
            // (probe_scratch(1)) or a last chunk of 1 row (§2.2 class). The decode
            // path never hits this because it passes a width-sized buffer.
            let mut emb_buf = vec![0u16; c * 2560];
            for q in 0..c {
                let h3 = [hist[slot * 2], hist[slot * 2 + 1], toks[q]];
                let uids = ple.ngram_uids(&h3);
                ple.embed_row(&uids, &mut emb_buf[q * 2560..(q + 1) * 2560])?;
                hist[slot * 2] = hist[slot * 2 + 1];
                hist[slot * 2 + 1] = toks[q];
            }
            psc.ple_emb = self.dev.htod_sync_copy(&emb_buf)?;
        }
        // S-A3-q P2: wide-M PLE projections through the vendor GEMM (opA=T on the plain f16
        // weights) — the hc_pre_rows / a,b arm (prefill drift class); per-row kernel below rmin.
        let rmin = recon_min_rows();
        let use_lt = rmin > 0 && c >= rmin;
        let rb = self.recon_bufs(psc);
        let _ = eps;
        let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
        if ple.bf {
            // FP4L --fp4l-plain exact: the stored-bf16 projections at the chunk's rows
            fp4l_fwd::bf_wide(l, &ple.key_w, dp(psc.ple_emb.device_ptr()), dp(psc.ple_key16.device_ptr()), 0, c, 2560, 10240)?;
        } else if use_lt {
            exl3_lt_matmul(l, *ple.key_w.device_ptr() as u64, *psc.ple_emb.device_ptr() as u64,
                           *psc.ple_key16.device_ptr() as u64, 2560, 10240, c, true, rb.ltws, rb.ltws_len)?;
        } else {
            xq_gemm_f16_small!(l, &mut psc.ple_key16, &ple.key_w, &psc.ple_emb, c, 10240usize, 2560usize)?;
        }
        self.ple_norm(l, ple, 0, dp(psc.ple_key16.device_ptr()), dp(psc.ple_kn.device_ptr()), c)?;
        if ple.bf {
            fp4l_fwd::bf_wide(l, &ple.value_w, dp(psc.ple_emb.device_ptr()), dp(psc.ple_val.device_ptr()), 0, c, 2560, 2560)?;
        } else if use_lt {
            exl3_lt_matmul(l, *ple.value_w.device_ptr() as u64, *psc.ple_emb.device_ptr() as u64,
                           *psc.ple_val.device_ptr() as u64, 2560, 2560, c, true, rb.ltws, rb.ltws_len)?;
        } else {
            xq_gemm_f16_small!(l, &mut psc.ple_val, &ple.value_w, &psc.ple_emb, c, 2560usize, 2560usize)?;
        }
        self.ple_norm(l, ple, 1, dp(psc.resid.device_ptr()), dp(psc.ple_qn.device_ptr()), c)?;
        xqlaunch!(l, "xq_ple_gate", (c as u32 * 4, 1, 1), (256, 1, 1), 0,
                 (&psc.ple_qn, &psc.ple_kn, &psc.ple_val, &mut psc.ple_gated, scale))?;
        self.ple_norm(l, ple, 2, dp(psc.ple_gated.device_ptr()), dp(psc.ple_normed.device_ptr()), c)?;
        if let Some(cf) = ple.conv_f.as_ref() {
            anyhow::ensure!(!multi, "PACKPF: the FP4L PLE conv is single-segment");
            fp4l_fwd::ple_conv_chunk_f(l, dp(psc.resid.device_ptr()), dp(psc.ple_gated.device_ptr()), dp(psc.ple_normed.device_ptr()),
                                       dp(self.ple_state.device_ptr()), cf, slot, c)?;
        } else if !multi {
        xqlaunch!(l, "xq_ple_conv_chunk", ((10240u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (&mut psc.resid, &psc.ple_gated, &psc.ple_normed, &self.ple_state, &ple.conv_w, slot as i32, c as i32))?;
        } else {
            // PACKPF: each segment's rows against its own slot's conv history (row-offset pointers)
            let (rp, gp, np) = (dp(psc.resid.device_ptr()), dp(psc.ple_gated.device_ptr()), dp(psc.ple_normed.device_ptr()));
            for sg in segs {
                let (o32, o16) = ((sg.r0 * 10240 * 4) as u64, (sg.r0 * 10240 * 2) as u64);
                xqlaunch!(l, "xq_ple_conv_chunk", ((10240u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                         (rp + o32, gp + o16, np + o16, &self.ple_state, &ple.conv_w, sg.slot as i32, sg.n as i32))?;
            }
        }
        Ok(())
    }

    pub fn prefill_chunk(&self, psc: &mut PrefillScratch, toks: &[i32], pos0: usize,
                         slot: usize, want_logits: bool, ld: Option<&LadderCtx>) -> Result<()> {
        let segs = [PfSeg { r0: 0, n: toks.len(), pos0, slot }];
        self.prefill_segs_inner(psc, toks, &segs, want_logits, ld)
    }

    /// PACKPF (idea and first implementation: GitHub andersc, veloGB10 fork, PACK-PF): ONE prefill
    /// pass over several prompt segments packed back to back in the pass rows (`toks` = their
    /// concatenation, `segs` in row order, one segment per slot). Every row-independent launch
    /// (the chains, MoE, hc glue, norms, the indexer projection) runs once over all rows, so the
    /// per-pass weight streaming is shared; cuBLASLt GEMMs run once per segment at the segment's own
    /// row count (LT_SEGS); the sequence-dependent parts (PLE history / conv, GDN conv + scan, K/V
    /// writes, dense and sparse attention, QSA pooling, the head fill) loop over the segments with
    /// row-offset pointers. Each segment's state and KV are bit-identical to its unpacked chunk at
    /// (slot, pos0) — `--probe-exl3-packpf`. A segment may start anywhere (pos0 > 0: a later chunk
    /// of a long prompt, or the rest of a prompt after a prefix-cache resume); its head-fill carry
    /// comes from psc's per-slot carry store (pf_carry_to_slot / pf_carry_from_slot). One segment
    /// is exactly prefill_chunk.
    pub fn prefill_segs(&self, psc: &mut PrefillScratch, toks: &[i32], segs: &[PfSeg]) -> Result<()> {
        anyhow::ensure!(!segs.is_empty(), "prefill_segs: no segments");
        if segs.len() == 1 {
            let sg = segs[0];
            anyhow::ensure!(sg.r0 == 0 && sg.n == toks.len(), "prefill_segs: one segment must cover the pass");
            if sg.pos0 > 0 { self.pf_carry_from_slot(psc, sg.slot)?; }
            self.prefill_chunk(psc, toks, sg.pos0, sg.slot, false, None)?;
            return self.pf_carry_to_slot(psc, sg.slot);
        }
        LT_SEGS.with(|v| *v.borrow_mut() = segs.iter().map(|sg| sg.n).collect());
        let r = self.prefill_segs_inner(psc, toks, segs, false, None);
        LT_SEGS.with(|v| v.borrow_mut().clear());
        r
    }

    /// PACKPF: why this model's prefill cannot run packed passes (None = it can). Packing needs the
    /// WP05 prologue, the parallel GDN conv, the row-parallel K/V append, hd-256 flash attention
    /// (or the sweep) on every dense row, no FP4-Local weights, no tensor parallelism.
    pub fn pack_pf_unsupported(&self) -> Option<&'static str> {
        if self.tp.is_some() { return Some("tensor parallel"); }
        if self.fx_any() { return Some("FP4-Local weights"); }
        if !(wp05_on("pro") && self.cos_gather_ok) { return Some("the WP05 prefill prologue is off"); }
        if !wp05_on("par") { return Some("the WP05 parallel PLE row build is off"); }
        if !pq8_item(0) { return Some("the serial K/V append (--pq8-kvw 0)"); }
        if self.cfg.conv_kernel > 8 || tune::get_cur(&T_PF_CONV_PAR) == 0 { return Some("the serial GDN conv"); }
        if self.df2_prime.is_some() { return Some("DFlash2 prompt priming"); }
        for (att_i, ly) in self.layers.iter().filter_map(|ly| match &ly.mixer { Mixer::Attn(a) => Some(a), _ => None }).enumerate() {
            let cfg = self.acfg_i(att_i);
            if cfg.head_dim != 256 { return Some("an attention head dim other than 256"); }
            if self.pf_tc_ok(ly, att_i) { return Some("tensor-core dense prefill attention (PFTC)"); }
        }
        None
    }

    /// PACKPF: does any weight load as FP4-Local (its own prefill arms, not covered by the packed pass)?
    fn fx_any(&self) -> bool {
        self.lm_head.fx.is_some()
            || self.layers.iter().any(|ly| ly.moe.fx.is_some() || ly.moe.router_bf)
            || self.ple.as_ref().map_or(false, |p| p.bf || p.conv_f.is_some())
    }

    /// PACKPF: psc.hfill_carry (the carry a single-prompt chunk / restore / checkpoint left) into
    /// `slot`'s entry of the per-slot carry store.
    pub fn pf_carry_to_slot(&self, psc: &mut PrefillScratch, slot: usize) -> Result<()> {
        if self.mtp.is_none() { return Ok(()); }
        let rw = self.cfg.hidden_size * self.cfg.hc_count.max(1);
        anyhow::ensure!(psc.hfill_carry_slots.len() >= (slot + 1) * rw && psc.hfill_carry.len() >= rw, "PACKPF: carry slot {slot} out of range");
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        xqlaunch!(l, "xq_copy_f32", ((((rw as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut psc.hfill_carry_slots, &psc.hfill_carry, rw as i64, (slot * rw) as i64, 0i64))?;
        Ok(())
    }

    /// PACKPF: `slot`'s stored carry back into psc.hfill_carry (before that slot's seam prime,
    /// prefix snapshot or WP16 checkpoint, which read psc.hfill_carry).
    pub fn pf_carry_from_slot(&self, psc: &mut PrefillScratch, slot: usize) -> Result<()> {
        if self.mtp.is_none() { return Ok(()); }
        let rw = self.cfg.hidden_size * self.cfg.hc_count.max(1);
        anyhow::ensure!(psc.hfill_carry_slots.len() >= (slot + 1) * rw && psc.hfill_carry.len() >= rw, "PACKPF: carry slot {slot} out of range");
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        xqlaunch!(l, "xq_copy_f32", ((((rw as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut psc.hfill_carry, &psc.hfill_carry_slots, rw as i64, 0i64, (slot * rw) as i64))?;
        Ok(())
    }

    /// PACKPF: the most rows one pass can carry on this scratch.
    pub fn pf_rows_cap(&self, psc: &PrefillScratch) -> usize { psc.c.min(psc.toks.len()).min(psc.pos_rows.len()) }

    fn prefill_segs_inner(&self, psc: &mut PrefillScratch, toks: &[i32], segs: &[PfSeg],
                          want_logits: bool, ld: Option<&LadderCtx>) -> Result<()> {
        crate::thermal::pace(); // NIGHT1: the thermal governor (every rank, once per prefill pass)
        let c = toks.len();
        anyhow::ensure!(!segs.is_empty(), "prefill_segs: no segments");
        let multi = segs.len() > 1;
        let (pos0, slot) = (segs[0].pos0, segs[0].slot);
        if multi {
            let mut r = 0usize;
            for (i, sg) in segs.iter().enumerate() {
                anyhow::ensure!(sg.r0 == r && sg.n >= pf_min_seg_rows(), "prefill_segs: segments must tile the rows in order, each >= {} rows", pf_min_seg_rows());
                anyhow::ensure!(sg.pos0 + sg.n <= self.max_pos && sg.slot < self.width, "prefill_segs: segment past max_pos / slot range");
                anyhow::ensure!(segs[..i].iter().all(|o| o.slot != sg.slot), "prefill_segs: two segments of slot {}", sg.slot);
                r += sg.n;
            }
            anyhow::ensure!(r == c, "prefill_segs: rows {r} != {c}");
            anyhow::ensure!(ld.is_none() && !want_logits && psc.vis_img.is_none() && psc.ppl.is_none() && psc.cpv.is_none(),
                            "prefill_segs: a packed pass takes no trace / logits / image / perplexity / copy-span stash");
            if let Some(why) = self.pack_pf_unsupported() { anyhow::bail!("prefill_segs: packed passes unsupported here: {why}"); }
        }
        // KVPOOL: every segment writes rows [pos0, pos0+n) of ITS OWN lane — map them before any launch
        // below. Fork hunk H23 sat on prefill_chunk, which v0.7.10 turned into a one-segment wrapper that
        // delegates here; a packed pass enters prefill_segs_inner directly and would otherwise write rows
        // that were never mapped (PACKPF, --prefill-pack).
        for sg in segs { self.vmm_ensure(sg.slot, sg.pos0 + sg.n)?; }

        // TP-A rung 3: the chunked prefill MoE (moe_prefill) has no EP filter/reduce site yet —
        // refuse loudly instead of silently mixing a rank's partial into its residual.
        // TP-B: prefill is TP-wired (EP experts + head-sharded mixers + row-parallel reduces).
        let _tc = tune::enter(tune::Ctx::prefill(c)); // TUNE family scope
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        // S-A3-f-g: reconstruct+cuBLASLt handles (env-gated inside exl3_chain_rb).
        let rb = self.recon_bufs(psc);
        let rmin = recon_min_rows();
        let cfg = &self.tc; // TP-B: the trunk's per-rank geometry
        let h = cfg.hidden_size;
        let hcn = cfg.hc_count.max(1);
        let kd = cfg.lin_k_dim;
        let vd = cfg.lin_v_dim;
        let nh = cfg.lin_num_v_heads;
        let nk = cfg.lin_num_k_heads;
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let ck = cfg.conv_kernel;
        let rdim = cfg.rotary_dim;

        let pro = wp05_on("pro") && self.cos_gather_ok && segs.iter().all(|sg| sg.pos0 + sg.n <= self.max_pos)
            && psc.wp05.as_ref().map_or(false, |s| s.c >= c)
            && psc.toks.len() >= c && psc.pos_rows.len() >= c && psc.slot_rows.len() >= c
            && psc.slotpos_rows.len() >= 2 * c && psc.cos.len() >= c * 2 * rdim;
        if pro {
            // WP05 prologue: the chunk's tables go through the pinned stage into the PERSISTENT
            // psc buffers with async NULL-stream copies (ordered before the compute stream's
            // kernels by the legacy-stream rule — the same ordering htod_copy_into relied on),
            // and the RoPE rows are a device gather from the load-time cos_tab (bit-identical to
            // the host loop: verified at load, cos_gather_ok). No host drain, no per-chunk
            // allocation, no per-chunk powf/cos/sin loop.
            let has_idx = self.cfg.has_indexer();
            let st = psc.wp05.as_mut().unwrap();
            if st.inflight_tab { self.dsync()?; st.inflight_tab = false; } // only after an error path
            let cc = st.c;
            st.pin.i32s(0, c).copy_from_slice(toks);
            {
                let pr = st.pin.i32s(4 * cc, c);
                for sg in segs { for i in 0..sg.n { pr[sg.r0 + i] = (sg.pos0 + i) as i32; } }
            }
            {
                let sr = st.pin.i32s(8 * cc, c);
                for sg in segs { sr[sg.r0..sg.r0 + sg.n].fill(sg.slot as i32); }
            }
            {
                let sp = st.pin.i32s(12 * cc, 2 * c);
                for sg in segs {
                    for i in 0..sg.n { sp[2 * (sg.r0 + i)] = sg.slot as i32; sp[2 * (sg.r0 + i) + 1] = (sg.pos0 + i) as i32; }
                }
            }
            st.pin.i32s(20 * cc, 1)[0] = (c * cfg.num_experts_per_tok) as i32;
            st.pin.htod(0, *psc.toks.device_ptr() as u64, c * 4)?;
            st.pin.htod(20 * cc, *psc.moerows.device_ptr() as u64, 4)?;
            st.pin.htod(4 * cc, *psc.pos_rows.device_ptr() as u64, c * 4)?;
            if has_idx || multi {
                st.pin.htod(8 * cc, *psc.slot_rows.device_ptr() as u64, c * 4)?;
                st.pin.htod(12 * cc, *psc.slotpos_rows.device_ptr() as u64, c * 8)?;
            }
            st.inflight_tab = true;
            // VIS-2: the mrope-aware gather (text-only slots read cos_tab[pos], unchanged)
            // PACKPF: per-row slots from the [slot, pos] pairs (text slots: cos_tab[pos], as Const)
            let ssrc = if multi { SlotSrc::Pairs(*psc.slotpos_rows.device_ptr() as u64) } else { SlotSrc::Const(slot) };
            self.cos_gather(&l, &mut psc.cos, *psc.pos_rows.device_ptr() as u64, ssrc, c)?;
            if ple_xcheck_on() {
                self.dsync()?;
                let got = self.dtoh(&psc.cos)?;
                let want = prefill_cossin_host(&self.rope, rdim, pos0, c);
                let nd = got[..want.len()].iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                crate::dprintln!("PLE_XCHECK prefill-rope pos0 {pos0} rows {c}: {nd} mismatching f32 of {}", want.len());
            }
        } else {
        anyhow::ensure!(!multi, "prefill_segs: a packed pass needs the WP05 prologue");
        psc.toks = self.dev.htod_sync_copy(toks)?;
        // S-A3-u D: the MoE row count is a per-chunk constant (c * topk) — one upload per chunk.
        self.dev.htod_copy_into(vec![(c * cfg.num_experts_per_tok) as i32], &mut psc.moerows)?;
        // cossin rows for pos0..pos0+c (same layout contract as forward_step)
        let cossin = prefill_cossin_host(&self.rope, rdim, pos0, c);
        psc.cos = self.dev.htod_sync_copy(&cossin)?;
        // S-A3-i: per-row (pos, slot, slot|pos) tables for the bf16 selection kernels
        // (they read LOGICAL positions; the prefill slot is uniform per chunk).
        if self.cfg.has_indexer() {
            let posr: Vec<i32> = (pos0..pos0 + c).map(|p| p as i32).collect();
            psc.pos_rows = self.dev.htod_sync_copy(&posr)?;
            psc.slot_rows = self.dev.htod_sync_copy(&vec![slot as i32; c])?;
            let mut sp: Vec<i32> = Vec::with_capacity(c * 2);
            for p in pos0..pos0 + c { sp.push(slot as i32); sp.push(p as i32); }
            psc.slotpos_rows = self.dev.htod_sync_copy(&sp)?;
        }
        }

        self.embed_gather(&l, *psc.resid.device_ptr() as u64, *psc.toks.device_ptr() as u64, c)?;
        // VIS-2: an image request overwrites its image rows with the tower's merged embeddings
        if psc.vis_img.is_some() {
            anyhow::ensure!(pro, "VIS-2: an image prefill needs the WP05 prologue (the host RoPE fallback has no mrope)");
            anyhow::ensure!(pos0 + c <= psc.vis_src.len(), "VIS-2: chunk {pos0}..{} past the image map ({} rows)",
                            pos0 + c, psc.vis_src.len());
            let rows: Vec<i32> = psc.vis_src[pos0..pos0 + c].to_vec();
            if rows.iter().any(|&r| r >= 0) {
                let mut dst = psc.vis_rows.slice_mut(0..c);
                self.dev.htod_sync_copy_into(&rows, &mut dst)?;
                let img = psc.vis_img.as_ref().unwrap();
                xqlaunch!(l, "xq_splice_rows", (((h * hcn * c) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                         (&mut psc.resid, img, &psc.vis_rows, h as i32, hcn as i32, c as i32))?;
            }
        }
        self.tr_f32(ld, false, 0, &psc.resid, "embed", c)?;
        let mut gdn_i = 0usize;
        let mut att_i = 0usize;
        // A5-P2 prefill.hc_fuse: each hc_post inject is deferred into the next hc_pre's norm
        // (xq_pf_hc_inj_norm). Never with trace dumps (they read resid between the two).
        let pf_defer = self.pf_hc_fuse(&l).0 && ld.is_none();
        let mut pend: Option<u64> = None;
        // TP-I2 item 1 (--tp-prefill-overlap): the deferred inject + the next hc_pre's norm run per
        // landed row range INSIDE the preceding pipelined reduce; `normed` = that already happened, so
        // the next hc_pre_rows_n skips its xq_pf_hc_inj_norm. TP only, and only on the fused-norm path.
        let ov_rows = if pf_defer && self.tp.is_some() { xtp::pf_overlap_rows() } else { 0 };
        let ple_live = self.ple.is_some() && crate::opts::var(crate::opt!("exl3-no-ple")).is_err();
        let mut normed = false;
        // TP-SP1 (--tp-seq-parallel): this rank's row block when the chunk runs sequence-parallel (None =
        // the replicated prefill, every launch below as before). The layer-0 block stays replicated when
        // the PLE follows it (the PLE reads every row of resid at the layer-1 boundary).
        let sp = self.sp_rows_for(psc, c);
        let sp_from = if ple_live { 1 } else { 0 };
        let spl = |li: usize| if li >= sp_from { sp } else { None };
        for (li, layer) in self.layers.iter().enumerate() {
            // NIGHT1: the thermal governor inside the pass (a no-op on a cool box; a hot box drains and idles the GPU)
            if li > 0 && li % crate::thermal::PASS_EVERY_LAYERS == 0 {
                crate::thermal::pace_pass(self.stream_capturing(), || Ok(self.dsync()?))?;
            }
            // S-A3-f-b FIX: PLE injects at the LAYER-1 boundary (decode injects
            // inside the layer loop before block 1's hc_pre) - the pre-loop call
            // put the delta one layer early and poisoned layer 0 from token 0.
            if li == 1 && ple_live {
                // (layer 0's MoE reduce takes no hook when the PLE follows, so `normed` is false here)
                if let Some(y) = pend.take() { self.hc_post_rows(&l, psc, y, c)?; } // A5-P2: PLE reads resid
                self.ple_forward_segs(&l, psc, toks, segs)?;
            }
            match spl(li) {
                Some(s) => self.hc_pre_rows_sp(&l, psc, &layer.hc_attn, s, pend.take(), std::mem::take(&mut normed), true)?,
                None => self.hc_pre_rows_n(&l, psc, &layer.hc_attn, c, pend.take(), std::mem::take(&mut normed))?,
            }
            self.tr_u16(ld, false, li, &psc.x, &format!("xpre{li}"), c * h)?;
            match &layer.mixer {
                Mixer::Gdn(g) => {
                    exl3_chain_rb(&l, &g.in_qkv, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.qkv, c, &psc.chain_ws, &rb)?;
                    self.tr_u16(ld, true, li, &psc.qkv, &format!("g{li}_qkv"), c * conv_dim)?;
                    exl3_chain_rb(&l, &g.in_z, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.zbuf, c, &psc.chain_ws, &rb)?;
                    self.tr_u16(ld, true, li, &psc.zbuf, &format!("g{li}_z"), c * vd)?;
                    if let Some(ab) = g.ab_fx.as_deref() {
                        // FP4L: exact MXFP8 (m = c: the wide kernel; a 16-row-or-less tail: the small one)
                        fp4l_fwd::fx_ab_rows(&l, ab, &psc.x, &mut psc.a_out, &mut psc.b_out, c, &rb)?;
                    } else if rmin > 0 && c >= rmin {
                        // S-A3-f-g: the 48-wide a/b projections are the worst
                        // xq_gemm_f16 shape at wide M (one 64-thread block per row,
                        // K-serial loop — nsys: 46.7% of prefill GPU time): route
                        // them through the vendor GEMM (opA=T on the plain f16
                        // [48, K] weights — no reconstruct needed).
                        exl3_lt_matmul(&l, *g.a_w.device_ptr() as u64, *psc.x.device_ptr() as u64,
                                       *psc.a_out.device_ptr() as u64, h as usize, nh,
                                       c, true, rb.ltws, rb.ltws_len)?;
                        exl3_lt_matmul(&l, *g.b_w.device_ptr() as u64, *psc.x.device_ptr() as u64,
                                       *psc.b_out.device_ptr() as u64, h as usize, nh,
                                       c, true, rb.ltws, rb.ltws_len)?;
                    } else {
                        xq_gemm_f16_small!(l, &mut psc.a_out, &g.a_w, &psc.x, c, nh, h)?;
                        xq_gemm_f16_small!(l, &mut psc.b_out, &g.b_w, &psc.x, c, nh, h)?;
                    }
                    self.tr_u16(ld, true, li, &psc.a_out, &format!("g{li}_a"), c * nh)?;
                    self.tr_u16(ld, true, li, &psc.b_out, &format!("g{li}_b"), c * nh)?;
                    // PACKPF (from GitHub andersc's PACK-PF): the parallel conv's output stays in chain_xh and
                    // the scan reads it there — the copy back into qkv is skipped (chain_xh is dead until the
                    // out_proj chain; nothing else reads qkv). Same bytes into the scan. Kept on trace / the
                    // GDN xcheck (they read psc.qkv).
                    let mut qkv_p = *psc.qkv.device_ptr() as u64;
                    let conv_nocopy = ld.is_none() && crate::opts::var(crate::opt!("exl3-gdn-xcheck")).is_err();
                    if ck <= 8 && tune::get_cur(&T_PF_CONV_PAR) != 0 {
                        // A5 D17: parallel (token, channel) conv into chain_xh (dead between chains),
                        // ring advance from the intact inputs, then copy back — bit-identical to the
                        // serial per-channel loop (4.1 s of a 128K prefill). --exl3-conv-serial = old.
                        let st_p = (*self.conv_state[gdn_i].device_ptr()) as u64;
                        let n = (c * conv_dim) as i64;
                        let (xo, xi) = (*psc.chain_xh.device_ptr() as u64, *psc.qkv.device_ptr() as u64);
                        anyhow::ensure!(psc.chain_xh.len() >= c * conv_dim, "conv: chain_xh holds {} < {} halves", psc.chain_xh.len(), c * conv_dim);
                        for sg in segs {
                            let off = (sg.r0 * conv_dim * 2) as u64;
                            let ns = (sg.n * conv_dim) as i64;
                            xqlaunch!(l, "xq_conv1d_chunk_par", (((ns + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                                      (xo + off, xi + off, st_p, &g.conv_w,
                                       conv_dim as i32, ck as i32, sg.n as i32, sg.slot as i32, conv_dim as i32))?;
                            xqlaunch!(l, "xq_conv1d_state_upd", ((conv_dim as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                                      (st_p, xi + off, conv_dim as i32, ck as i32, sg.n as i32, sg.slot as i32, conv_dim as i32))?;
                        }
                        if conv_nocopy {
                            qkv_p = xo;
                        } else {
                        xqlaunch!(l, "xq_copy_u16", (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                                  (&mut psc.qkv, &psc.chain_xh, n, 0i64, 0i64))?;
                        }
                    } else {
                    anyhow::ensure!(!multi, "PACKPF: the serial GDN conv is single-segment");
                    xqlaunch!(l, "xq_conv1d_chunk", ((conv_dim as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                             (&mut psc.qkv, (*self.conv_state[gdn_i].device_ptr()) as u64, &g.conv_w,
                              conv_dim as i32, ck as i32, c as i32, slot as i32, conv_dim as i32))?;
                    }
                    self.tr_u16(ld, true, li, &psc.qkv, &format!("g{li}_conv"), c * conv_dim)?;
                    let tc_min = gdn_tc_min_rows();
                    // PACKPF: the scan kernel is chosen PER SEGMENT from its own rows (as its unpacked chunk does)
                    let tc_seg = |n: usize| tc_min > 0 && n >= tc_min;
                    let tc = if tc_min > 0 && segs.iter().any(|sg| tc_seg(sg.n)) && kd == 128 && vd == 128 {
                        // A5-L1 prefill.gdn_scan: the re-scheduled bitwise twin xq_gdn_chunk_tc2
                        let v2 = tune::get(&T_PF_GDN_SCAN, &l.cx()) != 0;
                        static SEEN: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
                        if SEEN.swap(1 + v2 as u8, std::sync::atomic::Ordering::Relaxed) != 1 + v2 as u8 {
                            crate::dprintln!("A5-L1 prefill gdn: scan={} ({})", v2 as u8,
                                     if v2 { "xq_gdn_chunk_tc2 wavefront solve + prefetch" } else { "xq_gdn_chunk_tc" });
                        }
                        gdn_chunk_tc_fn(v2)
                    } else { None };
                    // S-A3-v --exl3-gdn-xcheck: the sequential scan on a copy of this slot's state
                    // (twin core rows), diffed against the chunkwise scan below — per-layer error.
                    let gx = if !multi && tc.is_some() && crate::opts::var(crate::opt!("exl3-gdn-xcheck")).is_ok() {
                        let ns = nh * kd * vd;
                        let mut st2 = self.dev.alloc_zeros::<f32>(ns)?;
                        let mut core2 = self.dev.alloc_zeros::<u16>(c * nh * vd)?;
                        xqlaunch!(l, "xq_copy_f32", (((ns + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                                  (&mut st2, (*self.s_state[gdn_i].device_ptr()) as u64, ns as i64, 0i64, (slot * ns) as i64))?;
                        let smem = (kd * 33 + kd + 32 * 3 + kd * 2) * 4;
                        xqlaunch!(l, "xq_gdn_step_chunk", ((nh * (vd / 32)) as u32, 1, 1), (kd as u32, 1, 1), smem as u32,
                                 (&mut core2, &psc.qkv, (*st2.device_ptr()) as u64,
                                  &psc.b_out, &psc.a_out, (nh | (nk << 16)) as i32, kd as i32, vd as i32,
                                  &g.a_log, &g.dt_bias, (c as i32) << 12, 0u64))?;
                        Some((st2, core2))
                    } else { None };
                    if let Some(f) = tc {
                        // S-A3-v: chunkwise-parallel scan (tensor cores; f32 state master in registers).
                        // PACKPF: one launch per segment (its own slot's state, row-offset inputs)
                        for sg in segs.iter().filter(|sg| tc_seg(sg.n)) {
                        let mut a_core = *psc.core.device_ptr() as u64 + (sg.r0 * nh * vd * 2) as u64;
                        let mut a_qkv = qkv_p + (sg.r0 * conv_dim * 2) as u64;
                        let mut a_st = *self.s_state[gdn_i].device_ptr() as u64;
                        let mut a_b = *psc.b_out.device_ptr() as u64 + (sg.r0 * nh * 2) as u64;
                        let mut a_a = *psc.a_out.device_ptr() as u64 + (sg.r0 * nh * 2) as u64;
                        let mut a_alog = *g.a_log.device_ptr() as u64;
                        let mut a_dtb = *g.dt_bias.device_ptr() as u64;
                        let mut a_nhnk = (nh | (nk << 16)) as i32;
                        let mut a_slot = sg.slot as i32;
                        let mut a_n = sg.n as i32;
                        let mut args: [*mut std::ffi::c_void; 10] = [
                            &mut a_core as *mut u64 as *mut _, &mut a_qkv as *mut u64 as *mut _,
                            &mut a_st as *mut u64 as *mut _, &mut a_b as *mut u64 as *mut _,
                            &mut a_a as *mut u64 as *mut _, &mut a_alog as *mut u64 as *mut _,
                            &mut a_dtb as *mut u64 as *mut _, &mut a_nhnk as *mut i32 as *mut _,
                            &mut a_slot as *mut i32 as *mut _, &mut a_n as *mut i32 as *mut _,
                        ];
                        let r = unsafe {
                            cudarc::driver::sys::cuLaunchKernel(f, (nh * (vd / GTC_VC)) as u32, 1, 1, 256, 1, 1,
                                GTC_SMEM as u32, self.stream.stream, args.as_mut_ptr(), std::ptr::null_mut())
                        };
                        anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                                        "xq_gdn_chunk_tc launch failed ({r:?})");
                        }
                        if let Some((st2, core2)) = gx.as_ref() {
                            self.dsync()?;
                            let ns = nh * kd * vd;
                            let sa = self.dtoh(st2)?;
                            let sb = self.dtoh(&self.s_state[gdn_i])?;
                            let ca = self.dtoh(core2)?;
                            let cb = self.dtoh(&psc.core)?;
                            let stat = |x: &mut dyn Iterator<Item = (f32, f32)>| {
                                let (mut d2, mut r2, mut mx) = (0f64, 0f64, 0f32);
                                for (p, q) in x { d2 += ((p - q) as f64).powi(2); r2 += (p as f64).powi(2); mx = mx.max((p - q).abs()); }
                                ((d2 / r2.max(1e-30)).sqrt(), mx)
                            };
                            let (sr, sm) = stat(&mut sa.iter().zip(&sb[slot * ns..(slot + 1) * ns]).map(|(p, q)| (*p, *q)));
                            let (cr, cm) = stat(&mut ca.iter().zip(&cb[..c * nh * vd])
                                .map(|(p, q)| (half::f16::from_bits(*p).to_f32(), half::f16::from_bits(*q).to_f32())));
                            crate::dprintln!("GDN_XCHECK layer {li} pos0 {pos0} C {c}: core relL2 {cr:.3e} max {cm:.3e} | state relL2 {sr:.3e} max {sm:.3e}");
                        }
                    }
                    {
                    let nchunk = vd / 32;
                    let smem = (kd * 33 + kd + 32 * 3 + kd * 2) * 4;
                    // 12-arg cap: slot_C = slot | C<<12
                    // (the segments the chunkwise scan above did not take: all of them when tc is None)
                    for sg in segs.iter().filter(|sg| !(tc.is_some() && tc_seg(sg.n))) {
                    let slot_c: i32 = (sg.slot as i32) | ((sg.n as i32) << 12);
                    xqlaunch!(l, "xq_gdn_step_chunk", ((nh * nchunk) as u32, 1, 1), (kd as u32, 1, 1), smem as u32,
                             (*psc.core.device_ptr() as u64 + (sg.r0 * nh * vd * 2) as u64,
                              qkv_p + (sg.r0 * conv_dim * 2) as u64,
                              (*self.s_state[gdn_i].device_ptr()) as u64,
                              *psc.b_out.device_ptr() as u64 + (sg.r0 * nh * 2) as u64,
                              *psc.a_out.device_ptr() as u64 + (sg.r0 * nh * 2) as u64,
                              (nh | (nk << 16)) as i32, kd as i32, vd as i32,
                              &g.a_log, &g.dt_bias, slot_c, 0u64))?;
                    }
                    }
                    self.tr_u16(ld, true, li, &psc.core, &format!("g{li}_core"), c * vd)?;
                    if cfg.gdn_gate_sigmoid {
                    xqlaunch!(l, "xq_gdn_gate", ((c * nh) as u32, 1, 1), (vd as u32, 1, 1), (vd * 4) as u32,
                             (&mut psc.normed, &psc.core, &psc.zbuf, &g.norm, vd as i32, nh as i32, c as i32, cfg.rms_eps))?;
                    } else { // EXL27B: Qwen3.5-class gated norm, silu(z) (a kernel copy)
                    xqlaunch!(l, "xq_gdn_gate_silu", ((c * nh) as u32, 1, 1), (vd as u32, 1, 1), (vd * 4) as u32,
                             (&mut psc.normed, &psc.core, &psc.zbuf, &g.norm, vd as i32, nh as i32, c as i32, cfg.rms_eps))?;
                    }
                    self.tr_u16(ld, true, li, &psc.normed, &format!("g{li}_gate"), c * vd)?;
                    let hk = self.pf_hook(psc, &layer.hc_mlp, *psc.mixer_out.device_ptr() as u64, ov_rows);
                    normed = self.chain_o_rb(&l, &g.out_proj, &psc.normed, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.mixer_out, c, &psc.chain_ws, &rb, hk, spl(li))?; // TP-B: row-parallel -> all-reduce (TP-SP: reduce-scatter)

                    gdn_i += 1;
                }
                Mixer::Attn(a) => {
                    exl3_chain_rb(&l, &a.q_proj, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.qg, c, &psc.chain_ws, &rb)?;
                    exl3_chain_rb(&l, &a.k_proj, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.kp, c, &psc.chain_ws, &rb)?;
                    exl3_chain_rb(&l, &a.v_proj, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.vp, c, &psc.chain_ws, &rb)?;
                    let nh_a = cfg.num_heads;
                    let nkv = cfg.num_kv_heads;
                    let hd = cfg.head_dim;
                    // 12-arg cap packing (see kernel headers)
                    let slot_pos0: i32 = (slot as i32) | ((pos0 as i32) << 12);
                    if multi {
                        self.attn_prefill_segs(&l, psc, a, att_i, segs, c)?;
                        let hk = self.pf_hook(psc, &layer.hc_mlp, *psc.mixer_out.device_ptr() as u64, ov_rows);
                        normed = self.chain_o_rb(&l, &a.o_proj, &psc.normed, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.mixer_out, c, &psc.chain_ws, &rb, hk, spl(li))?;
                        att_i += 1;
                    } else {
                    // PQ8 item 1: row-parallel K/V append (bit-identical; --pq8-kvw=0 = serial)
                    self.kv_prefill_write(&l, psc, &self.k_cache[att_i], &a.qnkn, slot, pos0, c,
                                          &format!("layer {att_i}"))?;
                    // S-A3-i: QSA straddle. Rows whose visible position exceeds
                    // qsa_limit() ride the selection path (the model spec past the
                    // threshold — the seam contract needs prefill and decode to agree);
                    // rows at/below stay on the dense kernels, bit-identical to the
                    // pre-port engine. `t` = first sparse row.
                    let t = if a.idx.is_some() && self.qsa_live(pos0 + c - 1) {
                        let lim = self.qsa_limit();
                        if pos0 > lim { 0 } else { (lim + 1 - pos0).min(c) }
                    } else { c };
                    // S-A3-f-g: flash-style prefill attention (speed directive,
                    // owner 2026-09-21). One block per 16-row q-tile with SMEM
                    // K/V staging vs the sweep's one-block-per-head serial
                    // loop. --attn-flash=0 or c<16 keeps the sweep kernel.
                    // S-A3-u A: the hd=256 flash kernel tiles 8 q rows and masks rows >= C at the
                    // store, so it serves ANY t >= 1 (the t >= 16 gate was the 16-row hd=128
                    // kernel's); the per-head serial sweep cost 9.6 ms for 4 rows. The tile's
                    // unmasked tail rows read cache positions < pos0+8, so keep that in range.
                    // Only DEEP tails take it (pos0 >= 256): the sweep's cost is its serial walk over
                    // pos0+t positions, ~free near 0, and it is the decode-softmax class — a short
                    // prompt's chunk on flash drifts ~5x more (ids16 GDN maxd 2.86 -> 13.7) and moved
                    // its greedy text (served ids16 d5 3.90 -> 3.65 tok/round on different text).
                    let flash = (if hd == 256 { t >= 16 || (t >= 1 && pos0 >= 256 && pos0 + 8 <= self.max_pos) }
                                 else { t >= 16 })
                        && (hd == 128 || hd == 256)
                        && !matches!(crate::opts::var(crate::opt!("attn-flash")).as_deref(), Ok("0"));
                    if t > 0 {
                        if flash && hd == 256 {
                            // S-A3-f-d night: hd=256 instantiation (BT=8/BKV=16).
                            // The 128-shape kernel silently corrupted hd=256 packs
                            // (SMEM OOB + half-staged dims) — see the kernel note.
                            // PQ8 item 2: the bitwise flash256v twin by default; --pq8-flash=0 =
                            // flash256, =causal = flash256c (opt-in, output-changing).
                            if self.pf_tc_qsa_ok(a, att_i) {
                                // POSTURE1 --exl3-dense-pf-tc-qsa (opt-in): Flash-Next's dense rows on the same kernels
                                PF_TC_QSA_ROWS.fetch_add(t as u64, std::sync::atomic::Ordering::Relaxed);
                                self.pf_tc_launch(&l, psc, a, att_i, slot_pos0, t)?;
                            } else if self.pf_tc_ok(a, att_i) {
                                // PFTC (owner 2026-11-07): tensor-core dense prefill on the packs without an indexer
                                self.pf_tc_launch(&l, psc, a, att_i, slot_pos0, t)?;
                            } else {
                                self.flash256_launch(&l, psc, a, att_i, slot_pos0, t)?;
                            }
                        } else if flash && hd == 128 && pq8_item(1) && pq8_flash_causal() {
                            // w5/PFIX (opt-in, output-changing — the same --pq8-flash=causal switch
                            // as flash256c): the hd=128 causal-staging flash twin. Default: hd=128 keeps
                            // the sweep below (xq_attn_prefill_flash, the defective scalar hd=128 flash,
                            // has had no caller since 2463afc).
                            xqlaunch!(l, "xq_attn_prefill_flash128c", (((t as u32) + 15) / 16, nh_a as u32, 1), (512, 1, 1), 0,
                                 (&mut psc.normed, &psc.qg, (*self.k_cache[att_i].device_ptr()) as u64,
                                  &a.qnkn, &psc.cos, (nkv as i32) | ((nh_a as i32) << 16),
                                  (hd as i32) | ((rdim as i32) << 16), self.mpf(),
                                  slot_pos0, t as i32, cfg.rms_eps))?;
                        } else {
                            // S-A3-q FIX: this was `else if flash`, so a chunk with fewer than
                            // 16 dense rows (every tail chunk < 16 tokens, and the QSA-straddle
                            // chunk whose dense part t < 16) ran NO attention for those rows —
                            // psc.normed held stale scratch (prefillx: first-token MISMATCH past
                            // the QSA limit). The sweep kernel serves every non-flash case.
                            xqlaunch!(l, "xq_attn_prefill_q", (nh_a as u32, 1, 1), (hd as u32, 1, 1), 0,
                                 (&mut psc.normed, &psc.qg, (*self.k_cache[att_i].device_ptr()) as u64,
                                  &a.qnkn, &psc.cos, (nkv as i32) | ((nh_a as i32) << 16),
                                  (hd as i32) | ((rdim as i32) << 16), self.mpf(),
                                  slot_pos0, t as i32, cfg.rms_eps))?;
                        }
                    }
                    if t < c {
                        self.qsa_attn_prefill_rows(&l, psc, a, att_i, slot, pos0, t, c)?;
                    } else {
                        // S-A3-r FIX: fully-dense chunk — still append its raw indexer keys.
                        self.qsa_keys_append_dense_rows(&l, psc, a, att_i, slot, pos0, c)?;
                    }
                    let hk = self.pf_hook(psc, &layer.hc_mlp, *psc.mixer_out.device_ptr() as u64, ov_rows);
                    normed = self.chain_o_rb(&l, &a.o_proj, &psc.normed, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.mixer_out, c, &psc.chain_ws, &rb, hk, spl(li))?; // TP-B: row-parallel -> all-reduce (TP-SP: reduce-scatter)
                    att_i += 1;
                    }
                }
            }
            let yatt_p = *psc.mixer_out.device_ptr() as u64;
            if pf_defer { pend = Some(yatt_p); } else { self.hc_post_rows_sp(&l, psc, yatt_p, c, spl(li))?; }
            self.tr_f32(ld, false, li, &psc.resid, &format!("rmid{li}"), c)?;
            self.tr_u16(ld, false, li, &psc.mixer_out, &format!("yatt{li}"), c * h)?;
            match spl(li) {
                Some(s) => self.hc_pre_rows_sp(&l, psc, &layer.hc_mlp, s, pend.take(), std::mem::take(&mut normed), true)?,
                None => self.hc_pre_rows_n(&l, psc, &layer.hc_mlp, c, pend.take(), std::mem::take(&mut normed))?,
            }
            self.tr_u16(ld, false, li, &psc.x, &format!("xpre_mlp{li}"), c * h)?;
            self.tr_f32(ld, false, li, &psc.inj, &format!("inj{li}"), c)?;
            // TP-I2 item 1: the MoE reduce's consumer is the NEXT hc_pre (the next layer's hc_attn, or
            // the final mixer) — except before layer 1 when the PLE sits between (it reads resid).
            let next_hc = self.layers.get(li + 1).map_or(&self.hc_mixer, |nl| &nl.hc_attn);
            psc.pf_hook = if li == 0 && ple_live { None }
                          else { self.pf_hook(psc, next_hc, *psc.moe_out.device_ptr() as u64, ov_rows) };
            psc.pf_hooked = false;
            psc.sp_rows = spl(li); // TP-SP: the EP combine reduce-scatters into the own rows
            let r = self.moe_prefill(&l, psc, &layer.moe, c, &rb);
            psc.sp_rows = None;
            r?;
            normed = std::mem::take(&mut psc.pf_hooked);
            psc.pf_hook = None;
            self.tr_u16(ld, false, li, &psc.moe_out, &format!("ymlp{li}"), c * h)?;
            self.tr_u16(ld, false, li, &psc.ysh, &format!("ysh{li}"), c * h)?;
            self.tr_u16(ld, false, li, &psc.yd, &format!("yd{li}"), c * cfg.num_experts_per_tok * h)?;
            let ymlp_p = *psc.moe_out.device_ptr() as u64;
            if pf_defer { pend = Some(ymlp_p); } else { self.hc_post_rows_sp(&l, psc, ymlp_p, c, spl(li))?; }
            // EXL27C-DF2 (fwd): the DFlash2 PROMPT-PRIME tap - the same post-MLP-inject
            // residual into the PRIME sink's rows [0, c) (trunk-identical: gpu.rs's S5F
            // prefill capture writes rows [0, n); the serve agent then calls
            // df2_prime_window(round, pos_start, c) per chunk). Flushing a deferred inject
            // before the tap is bitwise the fused path.
            // P7-TP2: under TP sequence-parallel prefill psc.resid holds only this rank's
            // OWN rows (hc_post_rows_sp's lo*h*hcn offset), so the tap all-gathers the
            // peer's rows first - the same gather the pack head fill does at chunk end.
            // Both ranks then capture identical full rows [0, c). Gathered peer rows are
            // inert downstream: every SP op in the loop reads/writes own rows only.
            if self.df2_prime.is_some()
                && self.cfg.hidden_size == crate::dflash2::geom().hidden {
                if let Some(kk) = crate::dflash2::geom().tap_layers.iter().position(|&x| x == li) {
                    // FLASHNEXT-DF2-B (TP=2 boot, 2026-10-23): under --tp-prefill-overlap the
                    // pipelined MoE reduce's hook has ALREADY injected `pend` into resid and
                    // normed the next hc_pre's input (`normed` = true). Flushing it again here
                    // added the MoE output to the residual a second time and then tripped
                    // hc_pre_rows_n's "a hooked norm without the fused inj_norm path" (pend
                    // taken, normed still set) — the head aborted at the first prefill. When the
                    // hook ran, resid is complete: capture it and leave `pend` to the next
                    // hc_pre_rows_n (its contract consumes the token). Without the hook, flush.
                    if !normed {
                        if let Some(y) = pend.take() {
                            self.hc_post_rows_sp(&l, psc, y, c, spl(li))?;
                        }
                    }
                    if let (Some(s), Some(tp)) = (sp, self.tp.as_ref()) {
                        let rw = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
                        tp.sp_all_gather(&l, *psc.resid.device_ptr() as u64, rw * 4, &s)?;
                    }
                    crate::dflash2::capture::capture_prime_f32(
                        &self.dev, l.stream, self.df2_prime.as_ref().unwrap(),
                        *psc.resid.device_ptr() as u64,
                        self.cfg.hc_count.max(1) * self.cfg.hidden_size,
                        self.cfg.hidden_size, self.cfg.hc_count.max(1), c, kk)?;
                }
            }
            // TP-SP2 G-T1-b fault injection (--xtp-sp-fault, the xtp probe's sequence-parallel run on ONE
            // rank only): perturb one OWN row of the streams by one ulp — the redesigned gate must fire
            if let (Some(s), Some(spd)) = (spl(li), ld.and_then(|x| x.sp.as_ref())) {
                if spd.fault_layer == Some(li) { self.sp_fault_flip(psc, s.own.0, li)?; }
            }
            self.tr_f32(ld, false, li, &psc.resid, &format!("layer{li}"), c)?;
        }
        // final mixer over all rows (resid streams), then the LAST row through
        // lm_head (the only logits prefill needs).
        self.tr_f32(ld, false, 0, &psc.resid, "precollapse", c)?;
        match sp {
            // TP-SP: the final mix on the own rows; x is gathered only when the last row's logits are wanted
            // COPYPF1 TP: a stretch's judge reads EVERY row of x, so an attached stash that wants the rows gathers x too
            Some(s) => self.hc_pre_rows_sp(&l, psc, &self.hc_mixer, s, pend.take(), std::mem::take(&mut normed),
                                           want_logits || cpv::cpv_wants_rows(psc))?,
            None => self.hc_pre_rows_n(&l, psc, &self.hc_mixer, c, pend.take(), std::mem::take(&mut normed))?,
        }
        self.tr_u16(ld, false, 0, &psc.x, "finalmix", c * h)?;
        if want_logits {
            // copy row c-1 of psc.x into psc.x_last (compute-stream kernel), then
            // the 1-row chain + argmax. Interleaved chunks skip the vocab head.
            xqlaunch!(l, "xq_copy_row", (((h as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                     (&mut psc.x_last, &psc.x, ((c - 1) * h) as i64, h as i32))?;
            pf_chain_check(&self.lm_head, 1, &psc.chain_xh, &psc.chain_yraw)?; // PFX1 (c): m = 1, N = vocab
            exl3_chain(&l, &self.lm_head, &psc.x_last, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.logits, 1, &psc.chain_ws)?;
            let v = cfg.vocab_size;
            xqlaunch!(l, "xq_argmax", (1, 1, 1), (1024, 1, 1), (1024 * 12) as u32,
                     (&mut psc.argmax, 0i32, &psc.logits, 0i64, v as i32))?;
            self.tr_u16(ld, false, 0, &psc.logits, "logits", v)?;
        } else if psc.ppl.is_some() {
            // S-B9-REL-PPL (`--perplexity`): EVERY row of the chunk through the lm_head, scored on the
            // host. Same chunk, same kernels up to here; only the head differs from serving.
            let mut sink = psc.ppl.take().unwrap();
            let r = self.ppl_all_rows(&l, psc, &mut sink, c);
            psc.ppl = Some(sink);
            r?;
        }
        // COPYPF1 (--copy-span-prefill): stash this chunk's final-mixed rows and its last row's streams before
        // the head fill below overwrites psc.x / psc.resid. Only when a stash is attached (never in serving
        // without the flag: no launch, no branch taken).
        // TP-SP: psc.resid holds only this rank's own rows; the stash's tap (row c-1) needs the full streams, so the
        // gather the head fill below does once per chunk runs here first (and the head fill then skips it). Every rank
        // attaches the stash at the same program point (a lockstep-agreed stretch), so the collective pairs up.
        let mut resid_gathered = false;
        if psc.cpv.is_some() {
            if let (Some(s), Some(tp)) = (sp, self.tp.as_ref()) {
                tp.sp_all_gather(&l, *psc.resid.device_ptr() as u64, h * hcn * 4, &s)?;
                resid_gathered = true;
            }
            self.cpv_stash_rows(&l, psc, c)?;
        }
        // S-A3-t: fill the MTP draft head's KV for this chunk's prompt positions (AGENTS §2.6:
        // the head attends over the prompt; before this it saw ZEROS for every prompt position).
        if let Some(head) = self.mtp.as_ref() {
            if crate::opts::var(crate::opt!("exl3-head-fill")).as_deref() != Ok("0") {
                if let (Some(s), Some(tp)) = (sp, self.tp.as_ref()) {
                    // TP-SP: the head fill reads the final streams of EVERY row (shifted by one): gather
                    // the peer's rows of resid once per chunk (a bit copy; its guard below is the fill's)
                    let rw = h * hcn;
                    if psc.hfill_s.len() >= c * rw && !resid_gathered {
                        tp.sp_all_gather(&l, *psc.resid.device_ptr() as u64, rw * 4, &s)?;
                    }
                }
                // TP-SP2 G-T1-b: the final streams every row of the head fill reads (gathered under SP)
                // and the carry it leaves (WP16 checkpoints it; the seam prime reads it) — digested only
                // in the redesigned gate's runs (ld.sp), so the replicated G-T1-b is unchanged
                if ld.map_or(false, |x| x.sp.is_some()) { self.tr_f32(ld, false, 0, &psc.resid, "streams", c)?; }
                if multi { self.head_fill_segs(&l, psc, head, segs, c)?; } else {
                self.head_fill_rows(&l, psc, head, slot, pos0, c)?;
                }
                if ld.map_or(false, |x| x.sp.is_some()) { self.tr_f32(ld, false, 0, &psc.hfill_carry, "hcarry", 1)?; }
            }
        }
        self.dsync()?;
        // WP05: the NULL-stream sync retired every pinned-stage copy of this chunk
        if let Some(st) = psc.wp05.as_mut() { st.inflight_tab = false; st.inflight_ple = false; }
        Ok(())
    }

}
