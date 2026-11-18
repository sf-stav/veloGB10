//! Decode-side attention methods: QSA key / pool bookkeeping and `qsa_attn_step`, the dense v3 and split-K
//! steps, and the W4QSA / WP04 / WP14 cross-checks. See `mod.rs` for the file map and `attn_gates.rs` for the
//! predicates that choose between them.

use super::*;

impl FwdModel {
    pub fn qsa_limit(&self) -> usize { self.cfg.indexer_budget + self.cfg.indexer_compress_ratio - 1 }
    /// K1: the attention kernels' max_pos argument with the KV format in bits 28..31.
    pub(super) fn mpf(&self) -> i32 { (self.max_pos as i32) | ((self.kv_fmt as i32) << 28) }
    /// TP-B: the geometry of attention target `att_i` — the trunk's per-rank heads (`tc`), or the
    /// replicated MTP head's global ones (HEAD_ATT).
    pub(super) fn acfg_i(&self, att_i: usize) -> &crate::qwen::Config { if att_i == HEAD_ATT { &self.cfg } else { &self.tc } }
    /// TP-B: acfg_i keyed by the layer itself (the head's AttnLayer is the replicated one).
    fn acfg(&self, a: &AttnLayer) -> &crate::qwen::Config {
        if self.mtp.as_ref().map_or(false, |hh| std::ptr::eq(a, &hh.attn)) { &self.cfg } else { &self.tc }
    }
    /// WP25: xq_attn_sel_combine's `rows` with the KV format in bits 28..31 (q8 rotates the
    /// merged output back to the model basis there).
    pub(super) fn rows_kvf(&self, rows: usize) -> i32 { (rows as i32) | ((self.kv_fmt as i32) << 28) }
    /// Raw indexer key plane of attention target `att_i` (a trunk attention index, or
    /// HEAD_ATT = the MTP draft head); None = no plane.
    pub(super) fn qsa_keys_p(&self, att_i: usize) -> Option<u64> {
        if att_i == HEAD_ATT {
            if head_qsa_off() { return None; }
            self.mtp.as_ref().and_then(|h| h.qsa_keys.as_ref()).map(|k| *k.device_ptr() as u64)
        } else {
            self.qsa_keys.get(att_i).map(|k| *k.device_ptr() as u64)
        }
    }
    /// Pooled-key plane of attention target `att_i` (None = pooled path off for it).
    pub(super) fn qsa_pool_p(&self, att_i: usize) -> Option<u64> {
        if att_i == HEAD_ATT {
            if head_qsa_off() { return None; }
            self.mtp.as_ref().and_then(|h| h.qsa_pool.as_ref()).map(|k| *k.device_ptr() as u64)
        } else {
            self.qsa_pool.get(att_i).map(|k| *k.device_ptr() as u64)
        }
    }
    /// Combined K/V cache of attention target `att_i` (trunk layer or HEAD_ATT).
    fn kv_p(&self, att_i: usize) -> u64 {
        if att_i == HEAD_ATT {
            *self.mtp.as_ref().expect("HEAD_ATT without a head").kv.device_ptr() as u64
        } else {
            *self.k_cache[att_i].device_ptr() as u64
        }
    }
    /// Long-context decode fix (2026-09-26): the graphed QSA block scorer used a grid over the
    /// whole WINDOW (max_pos/ratio blocks per row); at a 262K window that is 8,192 CTAs per row per
    /// attention layer every step, almost all exiting at `j >= nb` — ~7 tok/s decode at 32K
    /// context. The grid is now the power-of-two bucket (>= 1024 blocks) covering the largest
    /// position of the step; returns (grid blocks, bucket id for the graph key). Rows mask by
    /// their own position exactly as before, so scores are identical.
    pub fn qsa_grid_bucket(&self, max_pos_in_step: usize) -> (usize, usize) {
        let ratio = self.cfg.indexer_compress_ratio.max(1);
        let stride = (self.max_pos / ratio).max(1);
        let need = (max_pos_in_step + 1) / ratio + 1;
        let b = need.next_power_of_two().max(1024).min(stride);
        (b, if b == stride { 63 } else { b.trailing_zeros() as usize })
    }
    /// WP04: true when EVERY scoring launch of a decode/verify/draft graph is the bucket-free
    /// `qsa_score_pooled_mr` (pooled planes on the trunk AND the head, kernel present, not
    /// escaped) — only then may the graph keys drop the grid bucket.
    pub(super) fn qsa_bucket_free(&self) -> bool {
        score_mr_on()
            && !self.qsa_pool.is_empty()
            && self.qsa_pool.len() == self.qsa_keys.len()
            && self.mtp.as_ref().map_or(true, |h| h.qsa_keys.is_none() || h.qsa_pool.is_some())
            && self.dev.get_func("gpu_batch", "qsa_score_pooled_mr").is_some()
    }
    /// qsa_grid_bucket with the KEY component: (grid blocks for the legacy scorer, bucket id for
    /// the graph key — 0 when the scorer is bucket-free, so one graph per regime serves every
    /// context and no request pays a capture at a power-of-two crossing).
    pub fn qsa_grid_bucket_key(&self, max_pos_in_step: usize) -> (usize, usize) {
        let (g, b) = self.qsa_grid_bucket(max_pos_in_step);
        (g, if self.qsa_bucket_free() { 0 } else { b })
    }
    /// S-B9-REL-QSAREGIME: the deepest speculative round at position `p` (depth <= `k`) whose
    /// verify rows p..p+depth are all on one side of the sparse switch. Past the switch, without
    /// an indexer, with --qsa-exl3 0 or with --qsa-row-regime 0 it is `k`.
    pub fn regime_depth(&self, p: usize, k: usize) -> usize {
        regime_depth_at(p, k, if self.qsa_live(usize::MAX) && row_regime_on() { Some(self.qsa_limit()) } else { None })
    }
    // (WP04's own `stream_capturing` was byte-identical to WP09's — p2 keeps WP09's copy.)
    /// QSA live at this context? (--qsa-exl3=0 = the owner's A/B escape hatch.)
    pub fn qsa_live(&self, max_visible: usize) -> bool {
        self.cfg.has_indexer() && max_visible > self.qsa_limit() && !qsa_disabled()
    }

    /// S-A3-r FIX: the raw indexer key plane is SPEC STATE maintained on EVERY forward (the
    /// reference: "side planes ... maintained on EVERY cached forward; the sparse attention itself
    /// only engages once some query position exceeds sparse_threshold()"). Before this, keys
    /// were written only once a row was past qsa_limit(), so the whole dense prefix (positions
    /// 0..=2051) stayed ZERO and every sparse selection ranked ~513 zero-key blocks at one
    /// constant score. Dense-regime rows now append their raw keys with the SAME launches the
    /// sparse path uses (indexer chain -> bf16 -> qsa_key_write_b; the key column is untouched
    /// by the q-only qnorm_rope, so it is not needed here). Decode/verify rows (m <= 16).
    /// W4/DENSE lever (d): an attention layer's same-input chains (all read sc.x) — q_proj
    /// (`with_q`), k|v and the indexer qk when the layer has one and the scratch holds its rows —
    /// as ONE exl3_dks launch (one ramp, one tail; the small k/v/indexer items fill the q items'
    /// last round). Every output is its own chain's, bitwise. Returns Ok(true) iff the indexer qk
    /// was produced here (sc.qsa_qk): the qsa key / step functions then skip their exl3_chain of
    /// it. Otherwise the old launches run (each may still take a DENSE launch of its own).
    pub(super) fn attn_in_chains(&self, l: &Launcher, sc: &mut Scratch, a: &AttnLayer, m: usize, with_q: bool) -> Result<bool> {
        if dense::multi_on() && m <= 16 {
            use dense::{dptr, DOut};
            let yr = (dptr(&sc.chain_yraw), sc.chain_yraw.len()); // FIX=0 tail: member by member
            let o = |y: &CudaSlice<u16>| DOut { y: dptr(y), y_len: y.len(), yraw: yr.0, yraw_len: yr.1, xh: 0, xh_len: 0 };
            let mut qs: Vec<&Quad> = Vec::with_capacity(4);
            let mut outs: Vec<DOut> = Vec::with_capacity(4);
            if with_q { qs.push(&a.q_proj); outs.push(o(&sc.qg)); }
            qs.push(&a.k_proj); outs.push(o(&sc.kp));
            qs.push(&a.v_proj); outs.push(o(&sc.vp));
            let idx = a.idx.as_ref().filter(|ix| sc.qsa_qk.len() >= m * ix.qk.n.max(0) as usize);
            let x = (dptr(&sc.x), sc.x.len());
            let ws = [(dptr(&sc.chain_ws), sc.chain_ws.len())];
            if let Some(ix) = idx {
                let (mut q2, mut o2) = (qs.clone(), outs.clone());
                q2.push(&ix.qk);
                o2.push(o(&sc.qsa_qk));
                if dense::dense_chains(l, &q2, &o2, x, (0, 0), m, &ws)? { return Ok(true); }
            }
            if dense::dense_chains(l, &qs, &outs, x, (0, 0), m, &ws)? { return Ok(false); }
        }
        if with_q {
            exl3_chain(l, &a.q_proj, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.qg, m, &sc.chain_ws)?;
        }
        exl3_chain_pair(l, &a.k_proj, &a.v_proj, &sc.x,
                        &mut sc.chain_xh, &mut sc.chain_yraw, &sc.chain_ws,
                        &mut sc.pair_xh, &mut sc.pair_yraw, &sc.pair_ws,
                        &mut sc.kp, &mut sc.vp, m)?;
        Ok(false)
    }

    /// W4/DENSE: the shared expert's down chain on silu(gate) * up. With the silu prologue (lever d:
    /// xq_silu_mul folded into the down's input transform) it is ONE launch reading sh1 / sh2 and
    /// sh1 keeps the gate output (nothing reads it after this); else the old xq_silu_mul (in place
    /// into sh1) + the down chain. Bitwise either way (xq_silu_mul's element expression verbatim).
    pub(super) fn shared_down(&self, l: &Launcher, sc: &mut Scratch, moe: &MoeLayer, m: usize) -> Result<()> {
        let mi = moe.sh_down.k as usize; // TP-B: the rank's shared-expert intermediate slice (640 at TP=1)
        if dense::silu_on() && m <= 16 && !(moe.dense && moe.sh_down.rowpar) {
            use dense::{dptr, DOut};
            let o = DOut { y: dptr(&sc.ysh), y_len: sc.ysh.len(), yraw: dptr(&sc.chain_yraw), yraw_len: sc.chain_yraw.len(),
                           xh: 0, xh_len: 0 };
            if sc.sh1.len() >= m * mi && sc.sh2.len() >= m * mi
                && dense::dense_chains(l, &[&moe.sh_down], &[o], (dptr(&sc.sh1), m * mi), (dptr(&sc.sh2), m * mi), m,
                                       &[(dptr(&sc.chain_ws), sc.chain_ws.len())])? {
                return Ok(());
            }
        }
        let sh1p = *sc.sh1.device_ptr() as u64;
        xqlaunch!(l, "xq_silu_mul", (((m * mi) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                 (&mut sc.sh1, sh1p, &sc.sh2, (m * mi) as i64))?;
        if moe.dense && moe.sh_down.rowpar {
            self.chain_o(l, &moe.sh_down, &sc.sh1, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.ysh, m, &sc.chain_ws)
        } else {
            exl3_chain(l, &moe.sh_down, &sc.sh1, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.ysh, m, &sc.chain_ws)
        }
    }

    /// `idx_pre`: sc.qsa_qk already holds this step's indexer projection (W4/DENSE attention group,
    /// attn_in_chains — the same exl3 chain, same bits), so the chain is not re-run here.
    pub(super) fn qsa_keys_append_dense(&self, l: &Launcher, sc: &mut Scratch, a: &AttnLayer, att_i: usize, m: usize,
                             idx_pre: bool) -> Result<()> {
        let Some(ix) = a.idx.as_ref() else { return Ok(()); };
        if !self.cfg.has_indexer() || qsa_disabled() { return Ok(()); }
        let Some(keys_p) = self.qsa_keys_p(att_i) else { return Ok(()); };
        let cfg = &self.cfg;
        let (hdx, heads) = (cfg.indexer_head_dim, cfg.indexer_n_heads);
        let qk_dim = (heads + 1) * hdx;
        if sc.qsa_qk.len() < m * qk_dim { return Ok(()); }
        if !idx_pre {
            exl3_chain(l, &ix.qk, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.qsa_qk, m, &sc.chain_ws)?;
        }
        let ncnv = (m * qk_dim) as i64;
        xqlaunch!(l, "xq_qsa_h2b", ((((ncnv + 255) / 256).max(1)) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.qsa_qkb, &sc.qsa_qk, ncnv))?;
        gblaunch!(l, "qsa_key_write_b", ((m * hdx) as u32, 1, 1), (256, 1, 1), 0,
                  (keys_p, &sc.qsa_qkb, &sc.pos, &sc.slot_ids, self.max_pos as i32, 0i32,
                   qk_dim as i32, (heads * hdx) as i32, hdx as i32, m as i32))?;
        if let (Some(pool_p), Some(params)) = (self.qsa_pool_p(att_i), ix.params.as_ref()) {
            let nblk_stride = (self.max_pos / cfg.indexer_compress_ratio).max(1);
            let geom: u64 = crate::gpu::pack_qsa_geom(self.max_pos, 0, nblk_stride);
            gblaunch!(l, "qsa_pool_update_b", ((m.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0,
                      (pool_p, keys_p, &sc.pos, &sc.slot_ids, (*params.device_ptr()) as u64, geom as i64, m as i32))?;
        }
        Ok(())
    }

    /// S-A3-r FIX, prefill twin: a fully-dense chunk (no sparse rows) appends its raw indexer
    /// keys too — the same launches qsa_attn_prefill_rows runs for all c rows of a straddle chunk.
    pub(super) fn qsa_keys_append_dense_rows(&self, l: &Launcher, psc: &mut PrefillScratch, a: &AttnLayer,
                                  att_i: usize, slot: usize, pos0: usize, c: usize) -> Result<()> {
        let Some(ix) = a.idx.as_ref() else { return Ok(()); };
        if !self.cfg.has_indexer() || qsa_disabled() { return Ok(()); }
        let Some(keys_p) = self.qsa_keys_p(att_i) else { return Ok(()); };
        let cfg = &self.cfg;
        let (hdx, heads) = (cfg.indexer_head_dim, cfg.indexer_n_heads);
        let qk_dim = (heads + 1) * hdx;
        // prefill scratch sizes the indexer buffers only when QSA can engage (qsa_on)
        if psc.qsa_qk.len() < c * qk_dim || psc.qsa_qkb.len() < c * qk_dim || psc.pos_rows.len() < c { return Ok(()); }
        pf_chain_check_n(&ix.qk, c, &psc.chain_xh, &psc.chain_yraw)?; // PFX1 (c)
        // TP2BIT5 #26 Fix 5: rb wired — Pack A's index_qk_proj is 6-bit; without the recon
        // buffers this chain refuses at the wide arm (F3 forced-reconstruct needs rb). The
        // indexer quad (2560x640) always fits the trunk-sized recon staging.
        let rb = self.recon_bufs(psc);
        exl3_chain_rb(l, &ix.qk, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.qsa_qk, c, &psc.chain_ws, &rb)?;
        let ncnv = (c * qk_dim) as i64;
        xqlaunch!(l, "xq_qsa_h2b", ((((ncnv + 255) / 256).max(1)) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut psc.qsa_qkb, &psc.qsa_qk, ncnv))?;
        gblaunch!(l, "qsa_key_write_b", ((c * hdx) as u32, 1, 1), (256, 1, 1), 0,
                  (keys_p, &psc.qsa_qkb, &psc.pos_rows, &psc.slot_rows, self.max_pos as i32, 0i32,
                   qk_dim as i32, (heads * hdx) as i32, hdx as i32, c as i32))?;
        self.qsa_pool_range(l, a, att_i, keys_p, slot, pos0, c)?;
        Ok(())
    }

    /// Refresh the pooled-key plane over the blocks covered by rows [pos0, pos0+c) of `slot`.
    pub(super) fn qsa_pool_range(&self, l: &Launcher, a: &AttnLayer, att_i: usize, keys_p: u64,
                      slot: usize, pos0: usize, c: usize) -> Result<()> {
        let (Some(pool_p), Some(ix)) = (self.qsa_pool_p(att_i), a.idx.as_ref()) else { return Ok(()); };
        let Some(params) = ix.params.as_ref() else { return Ok(()); };
        if c == 0 { return Ok(()); }
        let cfg = &self.cfg;
        let (hdx, ratio) = (cfg.indexer_head_dim, cfg.indexer_compress_ratio);
        let nblk_stride = (self.max_pos / ratio).max(1);
        let j0 = pos0 / ratio;
        let j1 = ((pos0 + c - 1) / ratio + 1).min(nblk_stride);
        if j1 <= j0 { return Ok(()); }
        gblaunch!(l, "qsa_pool_range_b", (((j1 - j0).div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0,
                  (pool_p + (slot * nblk_stride * hdx * 2) as u64, keys_p + (slot * self.max_pos * hdx * 2) as u64,
                   (*params.device_ptr()) as u64, j0 as i32, j1 as i32, slot as i32))?;
        Ok(())
    }

    /// S-A3-i: the QSA sparse attention step for `m` rows (decode m=1 or verify m=k+1):
    /// indexer projection -> raw-key append -> block scoring -> top-k (ascending) ->
    /// K/V cache write (bit-identical words, shared fn) + q staging -> gathered
    /// split-K flash-decode -> split merge. Replaces xq_attn_decode / _group above
    /// the dense threshold; see PLAN/S_A3_I_ATTENTION_DECODE.md for the contracts.
    /// `idx_pre`: sc.qsa_qk already holds this step's indexer projection (W4/DENSE attention group).
    /// `one_slot`: every row belongs to ONE slot at consecutive positions (a verify) — only then
    /// is the W4/QSA row-shared union gather used (decode steps carry one row per lane/slot).
    pub(super) fn qsa_attn_step(&self, l: &Launcher, sc: &mut Scratch, a: &AttnLayer, att_i: usize, m: usize,
                     idx_pre: bool, one_slot: bool) -> Result<()> {
        let cfg = self.acfg_i(att_i); // TP-B: per-rank heads (trunk) / global (the replicated head)
        let (ix, params) = a.idx.as_ref().and_then(|i| i.params.as_ref().map(|p| (i, p)))
            .expect("qsa_attn_step: indexer without params");
        let (hdx, heads, ratio) = (cfg.indexer_head_dim, cfg.indexer_n_heads, cfg.indexer_compress_ratio);
        let qk_dim = (heads + 1) * hdx;
        let nblk_stride = qsa_nblk_stride(self.max_pos, ratio);
        // The scorers write scores[row * nblk_stride + j]: a scratch pitched for a narrower window
        // (public PR #21) must fail here, not write into whatever the allocator put after it.
        anyhow::ensure!(sc.qsa_scores.len() >= m * nblk_stride,
                        "qsa_attn_step: score scratch holds {} f32 < {m} rows x {nblk_stride} blocks (max_pos {}): \
                         the scratch was not sized for this model's served window", sc.qsa_scores.len(), self.max_pos);
        let sel_max = self.qsa_limit();
        let keys_p = self.qsa_keys_p(att_i).context("qsa_attn_step: no key plane")?;
        let params_p = (*params.device_ptr()) as u64;
        // indexer projection (EXL3 3bpw chain) -> bf16 -> q heads normed+roped in place
        // (idx_pre: the W4/DENSE attention group already produced it, same chain, same bits)
        if !idx_pre {
            exl3_chain(l, &ix.qk, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.qsa_qk, m, &sc.chain_ws)?;
        }
        let ncnv = (m * qk_dim) as i64;
        xqlaunch!(l, "xq_qsa_h2b", ((((ncnv + 255) / 256).max(1)) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.qsa_qkb, &sc.qsa_qk, ncnv))?;
        xqlaunch!(l, "xq_qsa_qnorm_rope", ((m * heads) as u32, 1, 1), (hdx as u32, 1, 1), 0,
                  (&mut sc.qsa_qkb, &ix.qn, &sc.cos, heads as i32, hdx as i32,
                   cfg.rotary_dim as i32, m as i32, cfg.rms_eps))?;
        // raw-key append + block scoring (pooled keys recomputed from raws — the
        // HF-validated bf16 kernels, launched here against the exl3 slot/pos tables)
        gblaunch!(l, "qsa_key_write_b", ((m * hdx) as u32, 1, 1), (256, 1, 1), 0,
                  (keys_p, &sc.qsa_qkb, &sc.pos, &sc.slot_ids, self.max_pos as i32, 0i32,
                   qk_dim as i32, (heads * hdx) as i32, hdx as i32, m as i32))?;
        let geom: u64 = crate::gpu::pack_qsa_geom(self.max_pos, qk_dim, nblk_stride);
        let grid_nblk = if sc.qsa_grid_nblk > 0 { sc.qsa_grid_nblk.min(nblk_stride) } else { nblk_stride };
        if let Some(pool_p) = self.qsa_pool_p(att_i) {
            // refresh the blocks this step's rows touched, then score from the plane (exact)
            gblaunch!(l, "qsa_pool_update_b", ((m.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0,
                      (pool_p, keys_p, &sc.pos, &sc.slot_ids, params_p, geom as i64, m as i32))?;
            if self.qsa_bucket_free() {
                // WP04: bucket-free persistent scorer — fixed grid, per-row block bound on device,
                // bit-identical scores (--exl3-score-mr=0 = the bucketed kernel below).
                let smem = (m.min(QSA_MR_ROWS) * heads * hdx * 4) as u32;
                gblaunch!(l, "qsa_score_pooled_mr", (qsa_mr_ctas(), m.div_ceil(QSA_MR_ROWS) as u32, 1), (256, 1, 1), smem,
                          (&mut sc.qsa_scores, &sc.qsa_qkb, pool_p, &sc.pos, &sc.slot_ids, params_p,
                           (geom | score_mr_geom_flag()) as i64, m as i32))?;
                if wp14_xcheck(crate::opt!("exl3-score-xcheck")) && !self.stream_capturing() {
                    self.wp04_xcheck_scores(l, sc, att_i, m, pool_p, params_p, geom, nblk_stride)?;
                }
            } else {
            gblaunch!(l, "qsa_score_pooled_b", ((grid_nblk.div_ceil(8)) as u32, m as u32, 1), (256, 1, 1), 0,
                      (&mut sc.qsa_scores, &sc.qsa_qkb, pool_p, &sc.pos, &sc.slot_ids, params_p, geom as i64))?;
            }
        } else {
            gblaunch!(l, "qsa_score_b", ((grid_nblk.div_ceil(8)) as u32, m as u32, 1), (256, 1, 1), 0,
                      (&mut sc.qsa_scores, &sc.qsa_qkb, keys_p, &sc.pos, &sc.slot_ids,
                       0u64, 0u64, params_p, geom as i64))?;
        }
        // S-A3-i review fix: SELECT_ALL is unsafe past sel_max (list overflow) and
        // redundant below it (the dense branch IS the full selection) — constant 0.
        let sel_all = 0i32;
        // WP14 stage 1: the bit-identical top-k twin (--exl3-topk-v2=0 = the original).
        // A5-K3 (attn.qsa_select / --qsa-select=1): the latency-lean twin asc3 (same bits as asc2).
        let topk_k = if qsa_select_on() { "xq_qsa_topk_asc3" } else if topk_v2_on() { "xq_qsa_topk_asc2" }
                     else { "xq_qsa_topk_asc" };
        if qsa_select_on() {
            static K3_ONCE: std::sync::Once = std::sync::Once::new();
            K3_ONCE.call_once(|| crate::dprintln!("A5-K3: QSA top-k select asc3 ON (xq_qsa_topk_asc3, grid m x 1024; \
                                           --qsa-select=0 = xq_qsa_topk_asc2)"));
        }
        xqlaunch!(l, topk_k, (m as u32, 1, 1), (1024, 1, 1), 0,
                  (&mut sc.qsa_sel, &mut sc.qsa_psel, &sc.qsa_scores, &sc.pos,
                   params_p, nblk_stride as i32, sel_all))?;
        if (topk_v2_on() || qsa_select_on()) && wp14_xcheck(crate::opt!("exl3-topk-xcheck")) && !self.stream_capturing() {
            self.wp14_xcheck_topk(l, sc, att_i, m, params_p, nblk_stride, sel_max, topk_k)?;
        }
        if let Some((k, ds_, dp_, _, _)) = self.qsa_rowdbg.as_ref() {
            if *k == att_i {
                xqlaunch!(l, "xq_dbg_scatter_sel", (m as u32, 1, 1), (256, 1, 1), 0,
                          (*ds_.device_ptr() as u64, *dp_.device_ptr() as u64, &sc.qsa_sel, &sc.qsa_psel,
                           &sc.pos, sel_max as i32, QSA_ROWDUMP_CAP as i32))?;
            }
        }
        if let Some((k, ds, dc, _, _)) = self.qsa_dbg.as_ref() {
            if *k == att_i {
                xqlaunch!(l, "xq_copy_f32", (((sel_max as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                          (*ds.device_ptr() as u64, *sc.qsa_sel.device_ptr() as u64, sel_max as i64, 0i64, 0i64))?;
                xqlaunch!(l, "xq_copy_f32", (((nblk_stride as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                          (*dc.device_ptr() as u64, *sc.qsa_scores.device_ptr() as u64, nblk_stride as i64, 0i64, 0i64))?;
            }
        }
        // K/V write (bit-identical words) + q staging, then the gathered attention
        let nh_p = (cfg.num_heads as i32) | ((cfg.num_kv_heads as i32) << 16);
        let hd_p = (cfg.head_dim as i32) | ((cfg.rotary_dim as i32) << 16);
        xqlaunch!(l, "xq_attn_prep", ((m * cfg.num_kv_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                  (&mut sc.qstage, self.kv_p(att_i),
                   &sc.qg, &sc.kp, &sc.vp, &a.qnkn, &sc.cos, &sc.slots,
                   nh_p, hd_p, self.mpf(), cfg.rms_eps))?;
        let nsd = qsa_splits_dec();
        let split_len = sel_max.div_ceil(nsd);
        let g2: u64 = (nsd as u64) | ((sel_max as u64) << 20) | ((self.mpf() as u32 as u64) << 32);
        // W4/QSA (p5d: DEFAULT OFF, opt-in --w4qsa=1 — bitwise but slower in-round, see
        // w4qsa_on): the union gather stages each selected K/V row ONCE for all of a verify's
        // rows (one slot, consecutive positions). Bit-identical partials; the old dispatch serves
        // by default and always for decode steps (one row per lane/slot), m = 1 and the
        // --w4qsa-off=1 escape.
        let union_smem = if one_slot && m >= 2 && w4qsa_on() {
            qsa_union_fits(self.kv_fmt, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, m, split_len)
        } else { None };
        if let Some(smem) = union_smem {
            let xc = w4qsa_xcheck() && !self.stream_capturing();
            if xc {
                // poison the partials so a skipped write can never compare equal by accident
                self.w4qsa_poison(l, sc)?;
            }
            let ngrp = (cfg.num_heads / cfg.num_kv_heads) / QU_HPW;
            let split_hd = ((split_len as u32) | ((cfg.head_dim as u32) << 16)) as i32;
            xqlaunch!(l, "xq_attn_sel_union", ((cfg.num_kv_heads * nsd * ngrp) as u32, 1, 1), ((32 * m) as u32, 1, 1),
                      smem as u32,
                      (&mut sc.qsa_pm, &mut sc.qsa_pl, &mut sc.qsa_pacc, &sc.qstage,
                       self.kv_p(att_i), &sc.qsa_sel, &sc.qsa_psel, &sc.slots,
                       nh_p, g2 as i64, split_hd, m as i32))?;
            if xc {
                self.w4qsa_xcheck_gather(l, sc, att_i, m, nh_p, g2, split_len, nsd)?;
            }
        } else {
            let (pm, pl, pa) = (*sc.qsa_pm.device_ptr() as u64, *sc.qsa_pl.device_ptr() as u64,
                                *sc.qsa_pacc.device_ptr() as u64);
            let kname = self.qsa_gather_old(l, sc, att_i, m, nh_p, g2, split_len, nsd, pm, pl, pa)?;
            if kname.starts_with("xq_attn_sel_hg") && wp14_xcheck(crate::opt!("exl3-gather-xcheck")) && !self.stream_capturing() {
                self.wp14_xcheck_gather(l, sc, att_i, m, nh_p, g2, split_len, nsd, kname)?;
            }
        }
        xqlaunch!(l, "xq_attn_sel_combine", ((m * cfg.num_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                  (&mut sc.normed, &sc.qsa_pm, &sc.qsa_pl, &sc.qsa_pacc, &sc.qg,
                   nh_p, nsd as i32, self.rows_kvf(m)))?;
        Ok(())
    }

    /// The pre-W4/QSA decode/verify gather dispatch into the partial buffers at (pm, pl, pacc) —
    /// hg1 / hg3 (WP14) or splitk4 / splitk (all bit-identical partials). Returns the kernel name.
    #[allow(clippy::too_many_arguments)]
    fn qsa_gather_old(&self, l: &Launcher, sc: &Scratch, att_i: usize, m: usize, nh_p: i32, g2: u64,
                      split_len: usize, nsd: usize, pm: u64, pl: u64, pa: u64) -> Result<&'static str> {
        let cfg = self.acfg_i(att_i); // TP-B: per-rank heads (trunk) / global (the replicated head)
        // A5 D6: decode/verify take the float4 twin (bit-identical partials, S-A3-v SEL_XCHECK);
        // the scalar kernel was 10.7 ms of a 32K-context verify. --exl3-sel-vec=0 = scalar.
        let sel_kname = if tune::get_cur(&T_SEL_VEC) == 0 {
            "xq_attn_sel_splitk" } else { "xq_attn_sel_splitk4" };
        // WP14 rungs 1/2: head-grouped gather with a cp.async K/V ring — same grid, splits and
        // split_len, bit-identical partials (--exl3-sel-gather=legacy|hg1|hg3|auto, default auto).
        let hpw = sel_gather_hpw(m);
        let gqa = cfg.num_heads / cfg.num_kv_heads.max(1);
        let hg = hpw > 0 && cfg.head_dim == 256 && gqa >= 1 && gqa <= 12
            && gqa * cfg.num_kv_heads == cfg.num_heads
            && kv_rowbytes(self.kv_fmt, cfg.head_dim) <= 1024 && split_len < 0x10000;
        if hg {
            let kname = if hpw == 1 { "xq_attn_sel_hg1" } else { "xq_attn_sel_hg3" };
            let nw = gqa.div_ceil(hpw);
            let split_hd = ((split_len as u32) | ((cfg.head_dim as u32) << 16)) as i32;
            xqlaunch!(l, kname, ((m * cfg.num_kv_heads * nsd) as u32, 1, 1), ((nw * 32) as u32, 1, 1), 0,
                      (pm, pl, pa, &sc.qstage,
                       self.kv_p(att_i), &sc.qsa_sel, &sc.qsa_psel, &sc.slots,
                       nh_p, g2 as i64, split_hd))?;
            Ok(kname)
        } else {
            xqlaunch!(l, sel_kname, ((m * cfg.num_kv_heads * nsd) as u32, 1, 1), (256, 1, 1), 0,
                      (pm, pl, pa, &sc.qstage,
                       self.kv_p(att_i), &sc.qsa_sel, &sc.qsa_psel, &sc.slots,
                       nh_p, g2 as i64, split_len as i32))?;
            Ok(sel_kname)
        }
    }

    /// W4/QSA XCHECK helper: fill the verify partial buffers with a NaN pattern (0x7FC0DEAD) on the
    /// compute stream, so a union launch that skips a (row, head, split) cannot match the old
    /// kernel's value by reusing a stale one.
    fn w4qsa_poison(&self, l: &Launcher, sc: &Scratch) -> Result<()> {
        use cudarc::driver::sys;
        for (p, n) in [(*sc.qsa_pm.device_ptr(), sc.qsa_pm.len()), (*sc.qsa_pl.device_ptr(), sc.qsa_pl.len()),
                       (*sc.qsa_pacc.device_ptr(), sc.qsa_pacc.len())] {
            let r = unsafe { sys::cuMemsetD32Async(p, 0x7FC0_DEAD, n, l.stream.stream) };
            anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "W4QSA_XCHECK poison memset failed: {r:?}");
        }
        Ok(())
    }

    /// W4/QSA --w4qsa-xcheck=1 (eager only): rerun the OLD dispatch (the kernel --w4qsa-off=1
    /// would launch: hg3 by default) into poisoned scratch partials and diff pm / pl / pacc bitwise
    /// over every (row, head, split) against the union kernel's. Prints the first calls, every
    /// mismatch and powers of two, with a running count of mismatching calls; a mismatching call
    /// is repaired with the old kernel's partials so the run continues on the old path's output.
    #[allow(clippy::too_many_arguments)]
    fn w4qsa_xcheck_gather(&self, l: &Launcher, sc: &Scratch, att_i: usize, m: usize, nh_p: i32, g2: u64,
                           split_len: usize, nsd: usize) -> Result<()> {
        let (nh, hd) = (self.acfg_i(att_i).num_heads, self.cfg.head_dim);
        let np = m * nh * nsd;
        let poison = f32::from_bits(0x7FC0_DEAD);
        let (xm, xl, xa) = (self.dev.htod_sync_copy(&vec![poison; np])?, self.dev.htod_sync_copy(&vec![poison; np])?,
                            self.dev.htod_sync_copy(&vec![poison; np * hd])?);
        let kname = self.qsa_gather_old(l, sc, att_i, m, nh_p, g2, split_len, nsd,
                                        *xm.device_ptr() as u64, *xl.device_ptr() as u64, *xa.device_ptr() as u64)?;
        self.dsync()?;
        let (tm, tl, ta) = (self.dtoh(&sc.qsa_pm)?, self.dtoh(&sc.qsa_pl)?,
                            self.dtoh(&sc.qsa_pacc)?);
        let (rm, rl, ra) = (self.dtoh(&xm)?, self.dtoh(&xl)?, self.dtoh(&xa)?);
        let nd = |x: &[f32], y: &[f32]| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let (dm, dl, da) = (nd(&tm[..np], &rm[..np]), nd(&tl[..np], &rl[..np]), nd(&ta[..np * hd], &ra[..np * hd]));
        // success signal: the reference is real (no poison left, live splits present)
        let live = rm[..np].iter().filter(|v| **v > f32::NEG_INFINITY).count();
        let left = rm[..np].iter().chain(&rl[..np]).chain(&ra[..np * hd]).filter(|v| v.to_bits() == 0x7FC0_DEAD).count();
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static BAD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let c = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let bad = dm + dl + da + left > 0 || live == 0;
        if bad { BAD.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        if c < 4 || bad || (c + 1).is_power_of_two() {
            let lay = if att_i == HEAD_ATT { "head".to_string() } else { att_i.to_string() };
            let first = (0..np).find(|&i| tm[i].to_bits() != rm[i].to_bits() || tl[i].to_bits() != rl[i].to_bits())
                .map(|i| format!(" first partial {i}: m {:e}/{:e} l {:e}/{:e}", tm[i], rm[i], tl[i], rl[i]))
                .unwrap_or_default();
            crate::dprintln!("W4QSA_XCHECK #{c} layer {lay} m={m} kv_fmt={}: xq_attn_sel_union vs {kname} bit-differ pm {dm} pl {dl} \
                      pacc {da} of {np} / {} (live splits {live}, poison left {left}; mismatching calls so far {}){first}",
                     self.kv_fmt, np * hd, BAD.load(std::sync::atomic::Ordering::Relaxed));
        }
        if bad {
            // repair: continue the request on the OLD kernel's partials (the combine reads them
            // next), so an XCHECK run always emits the old path's tokens and keeps counting
            use cudarc::driver::sys;
            for (dst, src, n) in [(*sc.qsa_pm.device_ptr(), *xm.device_ptr(), np), (*sc.qsa_pl.device_ptr(), *xl.device_ptr(), np),
                                  (*sc.qsa_pacc.device_ptr(), *xa.device_ptr(), np * hd)] {
                let r = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, n * 4, l.stream.stream) };
                anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "W4QSA_XCHECK repair copy failed: {r:?}");
            }
            self.dsync()?;
        }
        Ok(())
    }

    /// WP04 --exl3-score-xcheck (eager only): rerun the bucketed `qsa_score_pooled_b` over the
    /// FULL window into a scratch row set and diff every score both kernels write (j < nb_b).
    #[allow(clippy::too_many_arguments)]
    fn wp04_xcheck_scores(&self, l: &Launcher, sc: &Scratch, att_i: usize, m: usize, pool_p: u64,
                          params_p: u64, geom: u64, nblk_stride: usize) -> Result<()> {
        let mut xs = self.dev.alloc_zeros::<f32>(m * nblk_stride)?;
        gblaunch!(l, "qsa_score_pooled_b", ((nblk_stride.div_ceil(8)) as u32, m as u32, 1), (256, 1, 1), 0,
                  (&mut xs, &sc.qsa_qkb, pool_p, &sc.pos, &sc.slot_ids, params_p, geom as i64))?;
        self.dsync()?;
        let (a, b) = (self.dtoh(&sc.qsa_scores)?, self.dtoh(&xs)?);
        let pos = self.dtoh(&sc.pos)?;
        if a.len() < m * nblk_stride {
            crate::dprintln!("SCORE_XCHECK skipped: score scratch {} < {m} x {nblk_stride}", a.len());
            return Ok(());
        }
        let ratio = self.cfg.indexer_compress_ratio.max(1);
        let (mut nd, mut tot, mut first) = (0usize, 0usize, String::new());
        for r in 0..m {
            let nb = (((pos[r] + 1).max(0) as usize) / ratio).min(nblk_stride);
            for j in 0..nb {
                tot += 1;
                let (x, y) = (a[r * nblk_stride + j], b[r * nblk_stride + j]);
                if x.to_bits() != y.to_bits() {
                    if nd == 0 { first = format!(" first (row {r}, block {j}): mr {x:e} vs {y:e}"); }
                    nd += 1;
                }
            }
        }
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let c = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if c < 4 || nd > 0 || c.is_power_of_two() {
            let lay = if att_i == HEAD_ATT { "head".to_string() } else { att_i.to_string() };
            crate::dprintln!("SCORE_XCHECK #{c} layer {lay} m={m} pos0={}: qsa_score_pooled_mr vs pooled_b bit-differ {nd}/{tot}{first}",
                     pos[0]);
        }
        Ok(())
    }

    /// WP14 --exl3-topk-xcheck (eager only): rerun `xq_qsa_topk_asc` into scratch lists and
    /// diff pos_sel and the emitted entries [0, pos_sel] of every row.
    #[allow(clippy::too_many_arguments)]
    fn wp14_xcheck_topk(&self, l: &Launcher, sc: &Scratch, att_i: usize, m: usize, params_p: u64,
                        nblk_stride: usize, sel_max: usize, kname: &str) -> Result<()> {
        let mut xsel = self.dev.alloc_zeros::<i32>(m * sel_max)?;
        let mut xps = self.dev.alloc_zeros::<i32>(m)?;
        xqlaunch!(l, "xq_qsa_topk_asc", (m as u32, 1, 1), (1024, 1, 1), 0,
                  (&mut xsel, &mut xps, &sc.qsa_scores, &sc.pos, params_p, nblk_stride as i32, 0i32))?;
        self.dsync()?;
        let (s1, p1) = (self.dtoh(&sc.qsa_sel)?, self.dtoh(&sc.qsa_psel)?);
        let (s0, p0) = (self.dtoh(&xsel)?, self.dtoh(&xps)?);
        let (mut nd, mut tot, mut first) = (0usize, 0usize, String::new());
        for r in 0..m {
            tot += 1;
            if p0[r] != p1[r] {
                if nd == 0 { first = format!(" first: row {r} pos_sel v2 {} vs {}", p1[r], p0[r]); }
                nd += 1;
                continue;
            }
            let n = ((p0[r] + 1).max(0) as usize).min(sel_max);
            for i in 0..n {
                tot += 1;
                if s0[r * sel_max + i] != s1[r * sel_max + i] {
                    if nd == 0 { first = format!(" first: row {r} entry {i}: v2 {} vs {}", s1[r * sel_max + i], s0[r * sel_max + i]); }
                    nd += 1;
                }
            }
        }
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let c = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if c < 4 || nd > 0 || c.is_power_of_two() {
            let lay = if att_i == HEAD_ATT { "head".to_string() } else { att_i.to_string() };
            crate::dprintln!("TOPK_XCHECK #{c} layer {lay} m={m} pos_sel0={}: {kname} vs asc differ {nd}/{tot}{first}", p1[0]);
        }
        Ok(())
    }

    /// WP14 --exl3-gather-xcheck (eager only): rerun the legacy `xq_attn_sel_splitk4` into
    /// scratch partials and diff pm / pl / pacc bitwise over every (row, head, split).
    #[allow(clippy::too_many_arguments)]
    fn wp14_xcheck_gather(&self, l: &Launcher, sc: &Scratch, att_i: usize, m: usize, nh_p: i32, g2: u64,
                          split_len: usize, nsd: usize, kname: &str) -> Result<()> {
        let (nh, hd) = (self.acfg_i(att_i).num_heads, self.cfg.head_dim);
        let np = m * nh * nsd;
        let (mut xm, mut xl, mut xa) = (self.dev.alloc_zeros::<f32>(np)?, self.dev.alloc_zeros::<f32>(np)?,
                                        self.dev.alloc_zeros::<f32>(np * hd)?);
        xqlaunch!(l, "xq_attn_sel_splitk4", ((m * self.acfg_i(att_i).num_kv_heads * nsd) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut xm, &mut xl, &mut xa, &sc.qstage, self.kv_p(att_i), &sc.qsa_sel, &sc.qsa_psel, &sc.slots,
                   nh_p, g2 as i64, split_len as i32))?;
        self.dsync()?;
        let (tm, tl, ta) = (self.dtoh(&sc.qsa_pm)?, self.dtoh(&sc.qsa_pl)?,
                            self.dtoh(&sc.qsa_pacc)?);
        let (rm, rl, ra) = (self.dtoh(&xm)?, self.dtoh(&xl)?, self.dtoh(&xa)?);
        let nd = |x: &[f32], y: &[f32]| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let (dm, dl, da) = (nd(&tm[..np], &rm[..np]), nd(&tl[..np], &rl[..np]), nd(&ta[..np * hd], &ra[..np * hd]));
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let c = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if c < 4 || dm + dl + da > 0 || c.is_power_of_two() {
            let lay = if att_i == HEAD_ATT { "head".to_string() } else { att_i.to_string() };
            let first = (0..np).find(|&i| tm[i].to_bits() != rm[i].to_bits() || tl[i].to_bits() != rl[i].to_bits())
                .map(|i| format!(" first partial {i}: m {:e}/{:e} l {:e}/{:e}", tm[i], rm[i], tl[i], rl[i]))
                .unwrap_or_default();
            crate::dprintln!("GATHER_XCHECK #{c} layer {lay} m={m}: {kname} vs xq_attn_sel_splitk4 bit-differ pm {dm} pl {dl} pacc {da} of {np} / {}{first}",
                     np * hd);
        }
        Ok(())
    }

    /// WP13: may this trunk attention layer run the v3 dense kernels? They are built for
    /// hd 256 / GQA 12 (TP-4G: or GQA 6, the W=4 shape) and hold at most DV3_TS positions per row, so the dense regime must be
    /// BOUNDED by the QSA switch: the layer has an indexer, QSA is not disabled, and the
    /// switch point (positions <= qsa_limit) fits the plane. Graph keys carry the regime, so a
    /// graph captured dense is never replayed past qsa_limit. Everything else keeps v2.
    pub(super) fn dense_v3_ok(&self, a: &AttnLayer) -> bool {
        let cfg = self.acfg(a); // TP-B
        dense_v3_on()
            && cfg.head_dim == 256
            && cfg.num_kv_heads > 0
            && dv3_group(cfg.num_heads, cfg.num_kv_heads).is_some()
            && a.idx.is_some()
            && cfg.has_indexer()
            && !qsa_disabled()
            && self.qsa_limit() < DV3_TS
    }

    /// WP13: dense attention v3 for `m` rows (decode step: rows are lanes with their own
    /// slot; verify: one slot, positions p..p+m-1). Output -> sc.normed, K/V words -> the
    /// cache — bit-identical to xq_attn_decode_group (verify) / xq_attn_decode (step).
    /// Launches only; every grid depends on m alone (graph-safe), positions from sc.slots.
    pub(super) fn dense_attn_v3(&self, l: &Launcher, sc: &mut Scratch, a: &AttnLayer, kv: u64, m: usize,
                     verify: bool, att_i: usize) -> Result<()> {
        let cfg = self.acfg_i(att_i); // TP-B: per-rank heads (trunk) / global (the replicated head)
        let nh_p = (cfg.num_heads as i32) | ((cfg.num_kv_heads as i32) << 16);
        let hd_p = (cfg.head_dim as i32) | ((cfg.rotary_dim as i32) << 16);
        let nkv = cfg.num_kv_heads;
        let g = dv3_group(cfg.num_heads, nkv)
            .ok_or_else(|| anyhow::anyhow!("dense_attn_v3: {} heads / {} kv heads is not a dense-v3 group", cfg.num_heads, nkv))?;
        let (dots_name, kold, sd, acc_block) = dv3_plan(g, m);
        xqlaunch!(l, "xq_attn_dense_prep", ((m * cfg.num_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                  (&mut sc.qstage, kv, &sc.qg, &sc.kp, &sc.vp, &a.qnkn, &sc.cos, &sc.slots,
                   nh_p, hd_p, self.mpf(), cfg.rms_eps))?;
        // ~96 CTAs (2/SM at 46 KB smem) whatever m is; each strides over the row's 64-position
        // chunks on device (36 = the whole plane at m = 1).
        let nchg = (96 / (nkv * m)).clamp(1, DV3_NCH);
        xqlaunch!(l, dots_name, (nchg as u32, nkv as u32, m as u32), (256, 1, 1), 0,
                  (&mut sc.dv3_s, &mut sc.dv3_pm, &sc.qstage, kv, &sc.slots, nh_p, hd_p, self.mpf()))?;
        // one row: 16-dim slices (4x the CTAs); verify widths: 64-dim slices (4 chains/thread)
        // W4S (dattn): the 5-stage-ring twin (bitwise; --w4s-off=dattn = kold)
        let kname = self.w4s_dattn_kname(kold);
        xqlaunch!(l, kname, ((cfg.head_dim / sd) as u32, nkv as u32, m as u32), (acc_block, 1, 1), 0,
                  (&mut sc.normed, &sc.dv3_s, &sc.dv3_pm, kv, &sc.qg, &sc.slots, nh_p, hd_p, self.mpf()))?;
        self.w4s_dattn_xcheck(l, sc, kv, m, kname, kold, sd, cfg, acc_block)?;
        if dense_xcheck_on() && sc.dv3_xc.len() >= m * cfg.num_heads * cfg.head_dim {
            // the old kernel into the side buffer (it rewrites the same K/V words), then diff
            if verify {
                xqlaunch!(l, "xq_attn_decode_group", ((cfg.num_heads as u32), m as u32, 1), (cfg.head_dim as u32, 1, 1), 0,
                          (&mut sc.dv3_xc, &sc.qg, &sc.kp, &sc.vp, kv, &a.qnkn, &sc.cos, &sc.slots,
                           nh_p, hd_p, self.mpf(), cfg.rms_eps))?;
            } else {
                xqlaunch!(l, "xq_attn_decode", ((m as u32) * (cfg.num_heads as u32), 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                          (&mut sc.dv3_xc, &sc.qg, &sc.kp, &sc.vp, kv, &a.qnkn, &sc.cos, &sc.slots,
                           nh_p, hd_p, self.mpf(), cfg.rms_eps))?;
            }
            let tag = (att_i.min(255) | (m.min(255) << 8) | ((verify as usize) << 16)) as i32;
            xqlaunch!(l, "xq_dv3_xcheck", (1, 1, 1), (256, 1, 1), 0,
                      (&sc.normed, &sc.dv3_xc, (m * cfg.num_heads * cfg.head_dim) as i32, &mut sc.dv3_xst, tag))?;
        }
        Ok(())
    }

    /// The dense attention of `m` decode-step rows (rows = lanes, own slots) into sc.normed rows
    /// 0..m: dense v3 where the layer admits it, else xq_attn_decode — the dense branch of the
    /// decode-step dispatch, shared by a dense step (all its rows) and a mixed step (its leading
    /// short rows, S-B9-REL-QSAREGIME). Launches only; every grid depends on m alone (graph-safe).
    pub(super) fn dense_attn_step(&self, l: &Launcher, sc: &mut Scratch, a: &AttnLayer, att_i: usize, m: usize,
                       nh_p: i32, hd_p: i32) -> Result<()> {
        let cfg = &self.tc; // TP-B: the trunk's per-rank geometry
        if self.dense_v3_ok(a) {
            // WP13: dense v3, bitwise to xq_attn_decode (--exl3-dense-v3=0 = it)
            self.dense_attn_v3(l, sc, a, (*self.k_cache[att_i].device_ptr()) as u64, m, false, att_i)?;
        } else if self.dense_splitk_ok(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) {
            // DSPLITK (owner 2026-11-07): long-context dense attention, split-K over the context
            self.dense_attn_splitk(l, sc, a, (*self.k_cache[att_i].device_ptr()) as u64, m, nh_p, hd_p, cfg.rms_eps, false)?;
        } else {
            xqlaunch!(l, "xq_attn_decode", ((m as u32) * (cfg.num_heads as u32), 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                      (&mut sc.normed, &sc.qg, &sc.kp, &sc.vp,
                       (*self.k_cache[att_i].device_ptr()) as u64,
                       &a.qnkn, &sc.cos, &sc.slots,
                       nh_p, hd_p, self.mpf(), cfg.rms_eps))?;
        }
        Ok(())
    }

    /// DSPLITK (owner 2026-11-07, the EXL3 35B long-context decode collapse): is the trunk's dense
    /// attention served split-K? `--exl3-dense-splitk on` and a geometry the kernels cover: head_dim 256
    /// (xq_attn_dense_splitk4: 32 lanes x 8 dims) and at most 16 query heads per KV head (8 warps x 2).
    pub(super) fn dense_splitk_ok(&self, nh: usize, nkv: usize, hd: usize) -> bool {
        // never Flash-Next (its dense regime is dense v3; a geometry v3 refuses keeps the old kernels, byte-for-byte)
        dense_splitk_on() && !self.cfg.has_indexer() && dense_splitk_geometry_ok(nh, nkv, hd)
    }

    /// DSPLITK: the draft head's split-K pipeline (A5 D5) on a trunk layer — xq_attn_prep (K/V cache write
    /// + q staging with xq_attn_decode's exact arithmetic) -> xq_attn_dense_splitk4 (grid rows x KV heads x
    /// DRAFT_ATTN_SPLITS; a block's warps are the KV head's query heads, so a K/V row is fetched once per
    /// block; each row's slices span ITS OWN keys 0..=pos) -> xq_attn_sel_combine (non-empty slices,
    /// ascending; the gate). Rows are decode lanes (own slots) or verify rows (one slot, ascending
    /// positions). Grids depend on m alone: graph-safe. Summation order differs from xq_attn_decode.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_attn_splitk(&self, l: &Launcher, sc: &mut Scratch, a: &AttnLayer, kv: u64, m: usize,
                         nh_p: i32, hd_p: i32, eps: f32, one_slot: bool) -> Result<()> {
        let nh = (nh_p & 0xFFFF) as usize;
        let nkv = ((nh_p >> 16) & 0xFFFF) as usize;
        let hd = (hd_p & 0xFFFF) as usize;
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::reprintln!("[exl3] DSPLITK: dense attention ENGAGED on the trunk's full-attention layers — \
            split-K over the context ({DRAFT_ATTN_SPLITS} slices x {nkv} KV heads per row; {} query heads share each K/V read); \
            /metrics velogb10_exl3_dense_splitk_calls_total counts the launches (graph replays not included)", nh / nkv.max(1)));
        crate::metrics::exl3_dense_splitk_call();
        let geom: u64 = (DRAFT_ATTN_SPLITS as u64) | ((self.mpf() as u32 as u64) << 32);
        // round 2b: the W4S draft-head machinery on the trunk — xq_attn_dense_prep (grid m*nh) + the cp.async smem-ring
        // split-K (_w4, and its shared-read verify twin _vr_w4) + combine_w4: bitwise == xq_attn_prep + splitk4 +
        // sel_combine (the W4S contract; the draft head serves it). Needs the cache row to fit the 1-KB ring slot.
        let rbytes = kv_rowbytes(self.kv_fmt, hd);
        if dense_splitk_w4_on() && rbytes <= 1024 && rbytes % 16 == 0 && DRAFT_ATTN_SPLITS <= 256 {
            static W4_ONCE: std::sync::Once = std::sync::Once::new();
            W4_ONCE.call_once(|| crate::reprintln!("[exl3] DSPLITK: K/V staged through the cp.async smem ring (split-K _w4 kernels)"));
            xqlaunch!(l, "xq_attn_dense_prep", ((m * nh) as u32, 1, 1), (hd as u32, 1, 1), 0,
                      (&mut sc.qstage, kv, &sc.qg, &sc.kp, &sc.vp, &a.qnkn, &sc.cos, &sc.slots,
                       nh_p, hd_p, self.mpf(), eps))?;
            // round 3: q8 / fp8 rows are decoded ONCE per block (the _t4 twin; _vr_w4 tiles them itself) instead of
            // once per query-head warp; f16 / f32 rows keep the per-warp convert. Bitwise either way.
            let tiled = kv_split_tiled(self.kv_fmt) && rbytes == XQ_T4_RB;
            if tiled {
                static T4_ONCE: std::sync::Once = std::sync::Once::new();
                T4_ONCE.call_once(|| crate::reprintln!("[exl3] DSPLITK: {} K/V rows decoded once per block (round 3: split-K _t4 + tiled shared-read verify)",
                                                       kv_fmt_name(self.kv_fmt)));
            }
            let vr_min = if tiled { XQ_VR_MIN_ROWS_TILED } else { XQ_VR_MIN_ROWS_W4 };
            if one_slot && m >= vr_min && dense_splitk_vr_on() {
                static VRW_ONCE: std::sync::Once = std::sync::Once::new();
                VRW_ONCE.call_once(|| crate::reprintln!("[exl3] DSPLITK: shared-read verify ENGAGED — every verify of {vr_min}+ rows: \
                    up to {XQ_VR_ROWS} rows per block share each K/V read"));
                xqlaunch!(l, "xq_attn_dense_splitk4_vr_w4", ((m.div_ceil(XQ_VR_ROWS) * nkv * DRAFT_ATTN_SPLITS) as u32, 1, 1), (256, 1, 1), 0,
                          (&mut sc.dpm, &mut sc.dpl, &mut sc.dpacc, &sc.qstage, kv, &sc.slots, nh_p, geom as i64, m as i32))?;
            } else if tiled {
                // the deep-ring, row-fastest twin (the rows of one verify share each K/V line through L2)
                xqlaunch!(l, "xq_attn_dense_splitk4_t4", ((m * nkv * DRAFT_ATTN_SPLITS) as u32, 1, 1), (256, 1, 1), 0,
                          (&mut sc.dpm, &mut sc.dpl, &mut sc.dpacc, &sc.qstage, kv, &sc.slots, nh_p, geom as i64, m as i32))?;
            } else {
                xqlaunch!(l, "xq_attn_dense_splitk4_w4", ((m * nkv * DRAFT_ATTN_SPLITS) as u32, 1, 1), (256, 1, 1), 0,
                          (&mut sc.dpm, &mut sc.dpl, &mut sc.dpacc, &sc.qstage, kv, &sc.slots, nh_p, geom as i64))?;
            }
            xqlaunch!(l, "xq_attn_sel_combine_w4", ((m * nh) as u32, 1, 1), (hd as u32, 1, 1), 0,
                      (&mut sc.normed, &sc.dpm, &sc.dpl, &sc.dpacc, &sc.qg, nh_p, DRAFT_ATTN_SPLITS as i32, self.rows_kvf(m)))?;
            return Ok(());
        }
        xqlaunch!(l, "xq_attn_prep", ((m * nkv) as u32, 1, 1), (hd as u32, 1, 1), 0,
                  (&mut sc.qstage, kv, &sc.qg, &sc.kp, &sc.vp, &a.qnkn, &sc.cos, &sc.slots,
                   nh_p, hd_p, self.mpf(), eps))?;
        // 4+ rows: measured on the 35B EXL3 at 48K (TP1, ms/round per depth, per-row -> shared): d2 (3 rows) 36.7 -> 40.1
        // (a third of the blocks, each serial over its rows, under-fills the GPU), d3 47.1 -> 44.7, d4 57.1 -> 53.3,
        // d6 70.5 -> 62.2, d10 103.6 -> 84.4. So the shared-read twin takes 4+ rows; 2-3 rows keep the per-row kernel.
        if one_slot && m >= XQ_VR_ROWS && dense_splitk_vr_on() {
            // round 2: the rows of ONE verify share every K/V read (XQ_VR_ROWS rows per block; bitwise per row)
            static VR_ONCE: std::sync::Once = std::sync::Once::new();
            VR_ONCE.call_once(|| crate::reprintln!("[exl3] DSPLITK: shared-read verify ENGAGED — {XQ_VR_ROWS} verify rows per block share each K/V read"));
            xqlaunch!(l, "xq_attn_dense_splitk4_vr", ((m.div_ceil(XQ_VR_ROWS) * nkv * DRAFT_ATTN_SPLITS) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut sc.dpm, &mut sc.dpl, &mut sc.dpacc, &sc.qstage, kv, &sc.slots, nh_p, geom as i64, m as i32))?;
        } else {
            xqlaunch!(l, "xq_attn_dense_splitk4", ((m * nkv * DRAFT_ATTN_SPLITS) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut sc.dpm, &mut sc.dpl, &mut sc.dpacc, &sc.qstage, kv, &sc.slots, nh_p, geom as i64))?;
        }
        xqlaunch!(l, "xq_attn_sel_combine", ((m * nh) as u32, 1, 1), (hd as u32, 1, 1), 0,
                  (&mut sc.normed, &sc.dpm, &sc.dpl, &sc.dpacc, &sc.qg, nh_p, DRAFT_ATTN_SPLITS as i32, self.rows_kvf(m)))?;
        Ok(())
    }

}
