//! Prefill-side attention methods: QSA prefill rows and index / sparse rows, KV prefill writes, the PFTC and
//! flash256 launches, per-segment attention and head fill. See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// S-A3-i: QSA selection + gathered attention for the prefill rows [t0, c) of a
    /// chunk (eager, pointer-offset over the PrefillScratch row buffers). K/V for the
    /// whole chunk are already written by xq_attn_kv_prefill; the indexer projection
    /// runs over all c rows (a 640-wide GEMV — noise next to the chunk's GEMMs) and
    /// only the sparse row range consumes it. Splits = 1 (one block per row x kv head).
    pub(super) fn qsa_attn_prefill_rows(&self, l: &Launcher, psc: &mut PrefillScratch, a: &AttnLayer, att_i: usize,
                             slot: usize, pos0: usize, t0: usize, c: usize) -> Result<()> {
        self.qsa_pf_index_rows(l, psc, a, att_i, c)?;
        self.qsa_pf_sparse_rows(l, psc, a, att_i, slot, pos0, t0, c, 0)
    }

    /// QSA prefill step 1 over rows [0, c) of the pass: indexer projection -> bf16 -> q heads
    /// normed + roped in place -> raw-key append (per-row slot / pos tables). Row-independent, so a
    /// packed pass runs it once over every segment (PACKPF).
    fn qsa_pf_index_rows(&self, l: &Launcher, psc: &mut PrefillScratch, a: &AttnLayer, att_i: usize,
                         c: usize) -> Result<()> {
        let cfg = self.acfg_i(att_i);
        let ix = a.idx.as_ref().expect("qsa_pf_index_rows: no indexer");
        let (hdx, heads) = (cfg.indexer_head_dim, cfg.indexer_n_heads);
        let qk_dim = (heads + 1) * hdx;
        let keys_p = *self.qsa_keys[att_i].device_ptr() as u64;
        // 1. indexer projection (all c rows) -> bf16 -> q heads normed+roped in place
        pf_chain_check_n(&ix.qk, c, &psc.chain_xh, &psc.chain_yraw)?; // PFX1 (c)
        // TP2BIT5 #26 Fix 5 (site 2): rb wired — same as qsa_keys_append_dense_rows; Pack A's
        // 6-bit index_qk_proj must ride the F3 forced-reconstruct arm on the trunk chunk path.
        let rb = self.recon_bufs(psc);
        exl3_chain_rb(l, &ix.qk, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw,
                   &mut psc.qsa_qk, c, &psc.chain_ws, &rb)?;
        let ncnv = (c * qk_dim) as i64;
        xqlaunch!(l, "xq_qsa_h2b", ((((ncnv + 255) / 256).max(1)) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut psc.qsa_qkb, &psc.qsa_qk, ncnv))?;
        xqlaunch!(l, "xq_qsa_qnorm_rope", ((c * heads) as u32, 1, 1), (hdx as u32, 1, 1), 0,
                  (&mut psc.qsa_qkb, &ix.qn, &psc.cos, heads as i32, hdx as i32,
                   cfg.rotary_dim as i32, c as i32, cfg.rms_eps))?;
        // 2. raw-key append (all c rows), pooled block keys, tiled scoring
        gblaunch!(l, "qsa_key_write_b", ((c * hdx) as u32, 1, 1), (256, 1, 1), 0,
                  (keys_p, &psc.qsa_qkb, &psc.pos_rows, &psc.slot_rows, self.max_pos as i32, 0i32,
                   qk_dim as i32, (heads * hdx) as i32, hdx as i32, c as i32))?;
        Ok(())
    }

    /// QSA prefill step 2 for ONE prompt segment — rows [r0, r0 + c) of the pass (positions
    /// pos0.. of `slot`), sparse rows [t0, c) of it: pooled-plane refresh, block scoring, top-k,
    /// gathered attention into psc.normed. r0 = 0 is the unpacked chunk.
    #[allow(clippy::too_many_arguments)]
    fn qsa_pf_sparse_rows(&self, l: &Launcher, psc: &mut PrefillScratch, a: &AttnLayer, att_i: usize,
                          slot: usize, pos0: usize, t0: usize, c: usize, r0: usize) -> Result<()> {
        let cfg = self.acfg_i(att_i); // TP-B: per-rank heads (trunk) / global (the replicated head)
        let (_ix, params) = a.idx.as_ref().and_then(|i| i.params.as_ref().map(|p| (i, p)))
            .expect("qsa_attn_prefill_rows: indexer without params");
        let (hdx, heads, ratio) = (cfg.indexer_head_dim, cfg.indexer_n_heads, cfg.indexer_compress_ratio);
        const QSA_PF_QT: usize = 8;   // mirrors kernels/gpu_batch.cu QSA_PF_QT (queries per block in qsa_score_prefill_b)
        let qk_dim = (heads + 1) * hdx;
        let nblk_stride = (self.max_pos / ratio).max(1);
        let sel_max = self.qsa_limit();
        let rows = c - t0;
        let nblk = ((pos0 + c) / ratio).min(nblk_stride);
        let params_p = (*params.device_ptr()) as u64;
        let keys_p = *self.qsa_keys[att_i].device_ptr() as u64;
        // PACKPF: the segment's rows of the pass-wide buffers (r0 = 0: the old pointers)
        let t0g = r0 + t0;
        let qkb_seg = *psc.qsa_qkb.device_ptr() as u64 + (r0 * qk_dim * 2) as u64;
        let scores_seg = *psc.qsa_scores.device_ptr() as u64 + (r0 * nblk_stride * 4) as u64;
        // pooled plane: refresh only this chunk's blocks, then score straight from the slot's
        // plane (exact) — replaces re-pooling ALL nblk blocks every chunk (O(n^2) over a prompt).
        let pool_slot = if self.qsa_pool_p(att_i).is_some() {
            self.qsa_pool_range(l, a, att_i, keys_p, slot, pos0, c)?;
            Some(self.qsa_pool_p(att_i).unwrap() + (slot * nblk_stride * hdx * 2) as u64)
        } else {
            gblaunch!(l, "qsa_block_keys_b", ((nblk.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0,
                      ((*psc.qsa_blocks.device_ptr()) as u64, keys_p + (slot * self.max_pos * hdx * 2) as u64,
                       params_p, nblk as i32, slot as i32))?;
            None
        };
        let blocks_p = pool_slot.unwrap_or(*psc.qsa_blocks.device_ptr() as u64);
        // sq[] staging is QSA_PF_QT*heads*hd f32 of DYNAMIC shared (kernel contract,
        // gpu_batch.cu:12540): the launch must size it or every sq[i] write is out of
        // bounds — the 2026-09-25 memcheck fault (8,963 invalid __shared__ writes on
        // the first sparse rows). gpu.rs's call site passes the same value.
        let smem = (QSA_PF_QT * heads * hdx * 4) as u32;
        // WP18: the bounded tiled walk (bit-identical lists) replaces the full matrix + top-k;
        // --wp18-xcheck runs the old pair below into reference lists first and diffs.
        let wp18_xc = psc.wp18.as_ref().map(|w| w.x.is_some() && !self.stream_capturing());
        if wp18_xc != Some(false) {
            // Grid over the blocks VISIBLE at the end of this chunk (nblk), not the window's
            // nblk_stride: every j >= nblk returned before writing anyway (j >= (pos0+tlast+1)/ratio),
            // but only AFTER each CTA staged 8 queries x all heads into smem — at a 262K window that
            // is ~65K block slots x 256 query tiles per chunk per layer of pure waste (32K prefill:
            // 704 tok/s at max_pos 262K vs 950 at 40K, 2026-09-26). Same written scores, bit-exact.
            gblaunch!(l, "qsa_score_prefill_b", ((nblk.max(1).div_ceil(256)) as u32, (c.div_ceil(QSA_PF_QT)) as u32, 1), (256, 1, 1), smem,
                      (scores_seg, qkb_seg, blocks_p, params_p,
                       pos0 as i32, c as i32, nblk_stride as i32, qk_dim as i32))?;
            // 3. top-k for the sparse rows [t0, c) — ascending emission (XCHECK: into the reference lists)
            // S-A3-i review fix: SELECT_ALL is unsafe past sel_max (list overflow) and
            // redundant below it (the dense branch IS the full selection) — constant 0.
            let sel_all = 0i32;
            let (sel_b, psel_b) = match psc.wp18.as_ref().and_then(|w| w.x.as_ref()) {
                Some((xs, xp, _)) => (*xs.device_ptr() as u64, *xp.device_ptr() as u64),
                None => (*psc.qsa_sel.device_ptr() as u64, *psc.qsa_psel.device_ptr() as u64),
            };
            xqlaunch!(l, "xq_qsa_topk_asc", (rows as u32, 1, 1), (1024, 1, 1), 0,
                      (sel_b + (t0g * sel_max * 4) as u64,
                       psel_b + (t0g * 4) as u64,
                       scores_seg + (t0 * nblk_stride * 4) as u64,
                       (*psc.pos_rows.device_ptr()) as u64 + (t0g * 4) as u64,
                       params_p, nblk_stride as i32, sel_all))?;
        }
        if let Some(xc) = wp18_xc {
            self.wp18_select(l, psc, blocks_p, params_p, pos0, t0, c, xc && r0 == 0, r0)?;
            if xc && r0 == 0 { self.wp18_xcheck_report(psc, att_i, pos0, t0, c)?; }
        }
        if let Some((k, _, _, ps_, pp_)) = self.qsa_rowdbg.as_ref() {
            if *k == att_i {
                xqlaunch!(l, "xq_dbg_scatter_sel", (rows as u32, 1, 1), (256, 1, 1), 0,
                          (*ps_.device_ptr() as u64, *pp_.device_ptr() as u64,
                           (*psc.qsa_sel.device_ptr()) as u64 + (t0g * sel_max * 4) as u64,
                           (*psc.qsa_psel.device_ptr()) as u64 + (t0g * 4) as u64,
                           (*psc.pos_rows.device_ptr()) as u64 + (t0g * 4) as u64,
                           sel_max as i32, QSA_ROWDUMP_CAP as i32))?;
            }
        }
        if let Some((k, _, _, ps, pc)) = self.qsa_dbg.as_ref() {
            if *k == att_i && c <= 2048 && r0 == 0 {
                let ns = (c * sel_max) as i64; let nc = (c * nblk_stride) as i64;
                xqlaunch!(l, "xq_copy_f32", ((((ns + 255) / 256) as u32), 1, 1), (256, 1, 1), 0,
                          (*ps.device_ptr() as u64, *psc.qsa_sel.device_ptr() as u64, ns, 0i64, 0i64))?;
                xqlaunch!(l, "xq_copy_f32", ((((nc + 255) / 256) as u32), 1, 1), (256, 1, 1), 0,
                          (*pc.device_ptr() as u64, *psc.qsa_scores.device_ptr() as u64, nc, 0i64, 0i64))?;
            }
        }
        // 4. q staging + gathered attention over the sparse rows (splits = 1)
        let nh_p = (cfg.num_heads as i32) | ((cfg.num_kv_heads as i32) << 16);
        let hd_p = (cfg.head_dim as i32) | ((cfg.rotary_dim as i32) << 16);
        // S-A3-s FIX: a qg row is [q|gate] per head = num_heads * 2*head_dim f16 values, i.e.
        // num_heads*head_dim*4 BYTES (12,288 halves = 24,576 B here). This was
        // num_heads*head_dim*2 (the half-count of ONE of q/gate, read as bytes), so whenever the
        // sparse rows started at t0 > 0 — every chunk that straddles the QSA limit, i.e. mixes
        // dense and sparse rows — xq_attn_prep and xq_attn_sel_combine read the q and gate rows
        // of position p - t0/2 for position p. Pure-sparse chunks (t0 = 0) and decode never
        // offset, which is why only straddle chunks diverged.
        let qrow = cfg.num_heads * cfg.head_dim * 2 * 2;  // f16 bytes/row ([q|gate] per head)
        let krow = cfg.num_kv_heads * cfg.head_dim * 2;
        xqlaunch!(l, "xq_attn_prep", ((rows * cfg.num_kv_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                  ((*psc.qstage.device_ptr()) as u64 + (t0g * cfg.num_heads * cfg.head_dim * 4) as u64,
                   (*self.k_cache[att_i].device_ptr()) as u64,
                   (*psc.qg.device_ptr()) as u64 + (t0g * qrow) as u64,
                   (*psc.kp.device_ptr()) as u64 + (t0g * krow) as u64,
                   (*psc.vp.device_ptr()) as u64 + (t0g * krow) as u64,
                   &a.qnkn,
                   (*psc.cos.device_ptr()) as u64 + (t0g * 2 * cfg.rotary_dim * 4) as u64,
                   (*psc.slotpos_rows.device_ptr()) as u64 + (t0g * 2 * 4) as u64,
                   nh_p, hd_p, self.mpf(), cfg.rms_eps))?;
        // S-A3-u E: the SAME split geometry as the decode path (qsa_attn_step): QSA_SPLITS splits of
        // split_len = ceil(sel_max / QSA_SPLITS) per (row, kv head) — was splits = 1, one block
        // serially walking ~2,051 selected positions per row. Each row's reduction now matches
        // decode's exactly (cross-class), and the rows get QSA_SPLITS-way parallelism.
        let split_len = sel_max.div_ceil(QSA_SPLITS);
        let g2: u64 = (QSA_SPLITS as u64) | ((sel_max as u64) << 20) | ((self.mpf() as u32 as u64) << 32);
        let pmb = cfg.num_heads * QSA_SPLITS * 4;
        let pacb = cfg.num_heads * QSA_SPLITS * cfg.head_dim * 4;
        // S-A3-v: the float4 twin of splitk (same grid, values and expression order — bit-identical
        // partials; the prefill gather was LSU-bound on scalar loads). --exl3-sel-vec=0 keeps splitk;
        // --exl3-sel-xcheck=1 runs both and diffs.
        let vec4 = tune::get_cur(&T_SEL_VEC) != 0;
        let xcheck = vec4 && crate::opts::var(crate::opt!("exl3-sel-xcheck")).is_ok();
        // PQ8 item 3: the 4/8-bit-cache prefill gather (xq_attn_sel_pq8 — bit-identical partials to
        // splitk4: same grid/splits/order; each selected row unpacked ONCE per CTA into an f32 smem
        // tile, 4 warps x 3 heads, next tile's bytes prefetched). q8 and the probe-only q4x keep the
        // q8-layout raw path; q4k runs the same kernel on its raw twin (xq_q4k_raw/xq_q4k_deq8: 4
        // code bytes + the group's f16 scale at hd/2), == xq_kv_ld8's q4k branch. Its shape
        // contract: hd 256 (lane owns 8 dims), gqa exactly 12, split_len < 2^16. --pq8-gather=0 =
        // splitk4.
        let gqa = cfg.num_heads / cfg.num_kv_heads.max(1);
        let pq8 = vec4 && !xcheck && pq8_item(2) && self.kv_fmt >= KV_Q8 && cfg.head_dim == 256
            && gqa == 12 && gqa * cfg.num_kv_heads == cfg.num_heads && split_len < 0x10000;
        // POSTURE1 --exl3-sel-tc (opt-in): the same gather (grid, splits, entries, partial layout) on tensor cores
        let seltc = pq8 && sel_tc_on() && !pq8_xcheck();
        // POSTURE1 --sel-union-stats (diagnostic): how many distinct K/V rows groups of R neighbouring query rows select
        // (the bytes a union-staged gather would read), once per boot on the first chunk with >= 1024 sparse rows.
        if crate::opts::var(crate::opt!("sel-union-stats")).is_ok() && rows >= 1024 && !self.stream_capturing() {
            static DONE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if DONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
                self.dsync()?;
                let all = self.dtoh(&psc.qsa_sel)?;
                let ps = self.dtoh(&psc.qsa_psel)?;
                let mut line = format!("SEL_UNION layer {att_i} pos0 {pos0} rows {rows}:");
                let mut per_row = 0usize;
                for r in 0..rows { per_row += (ps[t0g + r] + 1).max(0) as usize; }
                line += &format!(" per-row entries mean {:.0};", per_row as f64 / rows as f64);
                for &g in &[2usize, 4, 8, 16, 32] {
                    let (mut uni, mut groups) = (0usize, 0usize);
                    let mut r = 0;
                    while r + g <= rows {
                        let mut set = std::collections::HashSet::new();
                        for q in r..r + g {
                            let n = (ps[t0g + q] + 1).max(0) as usize;
                            for &t in &all[(t0g + q) * sel_max..(t0g + q) * sel_max + n] { if t >= 0 { set.insert(t); } }
                        }
                        uni += set.len(); groups += 1; r += g;
                    }
                    line += &format!(" R={g}: union {:.0} per group = {:.2}x one row, bytes {:.2}x of per-row staging;",
                                     uni as f64 / groups as f64, uni as f64 / groups as f64 / (per_row as f64 / rows as f64),
                                     uni as f64 / groups as f64 / (g as f64 * per_row as f64 / rows as f64));
                }
                crate::rprintln!("{line}");
            }
        }
        // POSTURE1 union staging: 4 neighbouring rows share one block and read the union of their selections once
        // (indexer blocks of 4 tokens; another ratio keeps the per-row kernel)
        let union0 = seltc && cfg.indexer_compress_ratio == 4;
        // seltc6: 6 query heads per KV head (Flash-Next at TP=4: each rank holds 6 of a replicated KV head's 12) take
        // the union kernel with 8 tokens per block (xq_attn_sel_tcu8: 8 x 6 = the same 48 MMA rows); the default gather
        // there is splitk4 (pq8 takes 12 only), so the predicate is pq8's contract with gqa 6.
        let seltc80 = sel_tc_on() && !pq8_xcheck() && vec4 && !xcheck && pq8_item(2) && self.kv_fmt >= KV_Q8
            && cfg.head_dim == 256 && gqa == 6 && gqa * cfg.num_kv_heads == cfg.num_heads && split_len < 0x10000
            && cfg.indexer_compress_ratio == 4;
        // --exl3-sel-tc-xcheck (diagnostic, eager): run the tensor-core gather, then today's gather on the same rows (it
        // serves), one line per (layer, chunk) with the rel-L2 / max |diff| / non-finite count of the two attentions
        let xc_sel = (union0 || seltc80 || seltc) && sel_tc_xcheck_on() && !self.stream_capturing();
        let mut tc_out: Option<Vec<u16>> = None;
        for pass in 0..(if xc_sel { 2 } else { 1 }) {
        let (union, seltc8, seltc) = if pass == 0 { (union0, seltc80, seltc) } else { (false, false, false) };
            if seltc8 {
                // seltc6: 8 neighbouring tokens x 6 query heads = the same 48 MMA rows per block (Flash-Next at TP=4)
                sel_tc_announce("xq_attn_sel_tcu8 (8-token union staging, 6 query heads per KV head)");
                SEL_TC_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                SEL_TC8_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                SEL_TC_ROWS.fetch_add(rows as u64, std::sync::atomic::Ordering::Relaxed);
                let rows_ratio = (rows as i32) | (4i32 << 24);
                xqlaunch!(l, "xq_attn_sel_tcu8", (rows.div_ceil(8) as u32, cfg.num_kv_heads as u32, 1), (192, 1, 1), 0,
                          ((*psc.qsa_pm.device_ptr()) as u64 + (t0g * pmb) as u64,
                           (*psc.qsa_pl.device_ptr()) as u64 + (t0g * pmb) as u64,
                           (*psc.qsa_pacc.device_ptr()) as u64 + (t0g * pacb) as u64,
                           (*psc.qstage.device_ptr()) as u64 + (t0g * cfg.num_heads * cfg.head_dim * 4) as u64,
                           (*self.k_cache[att_i].device_ptr()) as u64,
                           (*psc.qsa_sel.device_ptr()) as u64 + (t0g * sel_max * 4) as u64,
                           (*psc.qsa_psel.device_ptr()) as u64 + (t0g * 4) as u64,
                           (*psc.slotpos_rows.device_ptr()) as u64 + (t0g * 2 * 4) as u64,
                           nh_p, g2 as i64, rows_ratio))?;
            } else if union {
                sel_tc_announce("xq_attn_sel_tcu (4-row union staging)");
                SEL_TC_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                SEL_TC_ROWS.fetch_add(rows as u64, std::sync::atomic::Ordering::Relaxed);
                let rows_ratio = (rows as i32) | (4i32 << 24);
                xqlaunch!(l, "xq_attn_sel_tcu", (rows.div_ceil(4) as u32, cfg.num_kv_heads as u32, 1), (192, 1, 1), 0,
                          ((*psc.qsa_pm.device_ptr()) as u64 + (t0g * pmb) as u64,
                           (*psc.qsa_pl.device_ptr()) as u64 + (t0g * pmb) as u64,
                           (*psc.qsa_pacc.device_ptr()) as u64 + (t0g * pacb) as u64,
                           (*psc.qstage.device_ptr()) as u64 + (t0g * cfg.num_heads * cfg.head_dim * 4) as u64,
                           (*self.k_cache[att_i].device_ptr()) as u64,
                           (*psc.qsa_sel.device_ptr()) as u64 + (t0g * sel_max * 4) as u64,
                           (*psc.qsa_psel.device_ptr()) as u64 + (t0g * 4) as u64,
                           (*psc.slotpos_rows.device_ptr()) as u64 + (t0g * 2 * 4) as u64,
                           nh_p, g2 as i64, rows_ratio))?;
            } else if seltc {
                sel_tc_announce("xq_attn_sel_tc");
                SEL_TC_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                SEL_TC_ROWS.fetch_add(rows as u64, std::sync::atomic::Ordering::Relaxed);
                let split_hd = (split_len as i32) | ((cfg.head_dim as i32) << 16);
                xqlaunch!(l, "xq_attn_sel_tc", ((rows * cfg.num_kv_heads * QSA_SPLITS) as u32, 1, 1), (128, 1, 1), 0,
                          ((*psc.qsa_pm.device_ptr()) as u64 + (t0g * pmb) as u64,
                           (*psc.qsa_pl.device_ptr()) as u64 + (t0g * pmb) as u64,
                           (*psc.qsa_pacc.device_ptr()) as u64 + (t0g * pacb) as u64,
                           (*psc.qstage.device_ptr()) as u64 + (t0g * cfg.num_heads * cfg.head_dim * 4) as u64,
                           (*self.k_cache[att_i].device_ptr()) as u64,
                           (*psc.qsa_sel.device_ptr()) as u64 + (t0g * sel_max * 4) as u64,
                           (*psc.qsa_psel.device_ptr()) as u64 + (t0g * 4) as u64,
                           (*psc.slotpos_rows.device_ptr()) as u64 + (t0g * 2 * 4) as u64,
                           nh_p, g2 as i64, split_hd))?;
            } else if pq8 {
                if sel_tc_on() && pass == 0 { SEL_TC_FALLBACK_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                sel_tc_announce("xq_attn_sel_pq8");
                let pq8x = pq8_xcheck() && !self.stream_capturing();
                let (a0, a1) = (t0g * cfg.num_heads * QSA_SPLITS, (t0g + rows) * cfg.num_heads * QSA_SPLITS);
                let split_hd = (split_len as i32) | ((cfg.head_dim as i32) << 16);
                let mut want: Option<(Vec<f32>, Vec<f32>, Vec<f32>)> = None;
                let names: &[&str] = if pq8x { &["xq_attn_sel_splitk4", "xq_attn_sel_pq8"] } else { &["xq_attn_sel_pq8"] };
                for &kname in names {
                    if kname == "xq_attn_sel_pq8" {
                        if pq8x {
                            self.dsync()?;
                            want = Some((self.dtoh(&psc.qsa_pm)?, self.dtoh(&psc.qsa_pl)?,
                                         self.dtoh(&psc.qsa_pacc)?));
                            // poison the rows' partials with f32 NaN (0xFFFFFFFF): an unwritten partial must differ
                            for (buf, n) in [(*psc.qsa_pm.device_ptr() as u64 + (a0 * 4) as u64, (a1 - a0) * 2),
                                             (*psc.qsa_pl.device_ptr() as u64 + (a0 * 4) as u64, (a1 - a0) * 2),
                                             (*psc.qsa_pacc.device_ptr() as u64 + (a0 * cfg.head_dim * 4) as u64,
                                              (a1 - a0) * cfg.head_dim * 2)] {
                                xqlaunch!(l, "xq_memset_u16", (((n as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                                          (buf, n as i64, 0xFFFFi32))?;
                            }
                        }
                        xqlaunch!(l, "xq_attn_sel_pq8", ((rows * cfg.num_kv_heads * QSA_SPLITS) as u32, 1, 1), (128, 1, 1), 0,
                                  ((*psc.qsa_pm.device_ptr()) as u64 + (t0g * pmb) as u64,
                                   (*psc.qsa_pl.device_ptr()) as u64 + (t0g * pmb) as u64,
                                   (*psc.qsa_pacc.device_ptr()) as u64 + (t0g * pacb) as u64,
                                   (*psc.qstage.device_ptr()) as u64 + (t0g * cfg.num_heads * cfg.head_dim * 4) as u64,
                                   (*self.k_cache[att_i].device_ptr()) as u64,
                                   (*psc.qsa_sel.device_ptr()) as u64 + (t0g * sel_max * 4) as u64,
                                   (*psc.qsa_psel.device_ptr()) as u64 + (t0g * 4) as u64,
                                   (*psc.slotpos_rows.device_ptr()) as u64 + (t0g * 2 * 4) as u64,
                                   nh_p, g2 as i64, split_hd))?;
                    } else {
                        xqlaunch!(l, kname, ((rows * cfg.num_kv_heads * QSA_SPLITS) as u32, 1, 1), (256, 1, 1), 0,
                                  ((*psc.qsa_pm.device_ptr()) as u64 + (t0g * pmb) as u64,
                                   (*psc.qsa_pl.device_ptr()) as u64 + (t0g * pmb) as u64,
                                   (*psc.qsa_pacc.device_ptr()) as u64 + (t0g * pacb) as u64,
                                   (*psc.qstage.device_ptr()) as u64 + (t0g * cfg.num_heads * cfg.head_dim * 4) as u64,
                                   (*self.k_cache[att_i].device_ptr()) as u64,
                                   (*psc.qsa_sel.device_ptr()) as u64 + (t0g * sel_max * 4) as u64,
                                   (*psc.qsa_psel.device_ptr()) as u64 + (t0g * 4) as u64,
                                   (*psc.slotpos_rows.device_ptr()) as u64 + (t0g * 2 * 4) as u64,
                                   nh_p, g2 as i64, split_len as i32))?;
                    }
                }
                if let Some((rpm, rpl, racc)) = want {
                    self.dsync()?;
                    let (tpm, tpl, tacc) = (self.dtoh(&psc.qsa_pm)?, self.dtoh(&psc.qsa_pl)?,
                                            self.dtoh(&psc.qsa_pacc)?);
                    let nd = |x: &[f32], y: &[f32]| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                    let hd = cfg.head_dim;
                    let (dm, dl, da) = (nd(&rpm[a0..a1], &tpm[a0..a1]), nd(&rpl[a0..a1], &tpl[a0..a1]),
                                        nd(&racc[a0 * hd..a1 * hd], &tacc[a0 * hd..a1 * hd]));
                    crate::dprintln!("PQ8_XCHECK gather layer {att_i} pos0 {pos0} rows {rows} (t0 {t0}) fmt {}: pq8 vs splitk4 bit-differ pm {dm} pl {dl} pacc {da} of {} / {}{}",
                             kv_fmt_name(self.kv_fmt), a1 - a0, (a1 - a0) * hd,
                             if dm + dl + da == 0 { "" } else { "  <-- MISMATCH" });
                }
            }
            let mut xref: Option<(Vec<f32>, Vec<f32>, Vec<f32>)> = None;
            for kname in ["xq_attn_sel_splitk", "xq_attn_sel_splitk4"] {
                if pq8 || seltc8 { break; }
                if sel_tc_on() && pass == 0 && kname == "xq_attn_sel_splitk" {
                    // POSTURE1: --exl3-sel-tc asked on a cache / shape it does not serve: today's gather (counted)
                    SEL_TC_FALLBACK_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    sel_tc_announce(if vec4 { "xq_attn_sel_splitk4" } else { "xq_attn_sel_splitk" });
                }
                let four = kname == "xq_attn_sel_splitk4";
                if four != vec4 && !(xcheck && !four) { continue; }
                xqlaunch!(l, kname, ((rows * cfg.num_kv_heads * QSA_SPLITS) as u32, 1, 1), (256, 1, 1), 0,
                          ((*psc.qsa_pm.device_ptr()) as u64 + (t0g * pmb) as u64,
                           (*psc.qsa_pl.device_ptr()) as u64 + (t0g * pmb) as u64,
                           (*psc.qsa_pacc.device_ptr()) as u64 + (t0g * pacb) as u64,
                           (*psc.qstage.device_ptr()) as u64 + (t0g * cfg.num_heads * cfg.head_dim * 4) as u64,
                           (*self.k_cache[att_i].device_ptr()) as u64,
                           (*psc.qsa_sel.device_ptr()) as u64 + (t0g * sel_max * 4) as u64,
                           (*psc.qsa_psel.device_ptr()) as u64 + (t0g * 4) as u64,
                           (*psc.slotpos_rows.device_ptr()) as u64 + (t0g * 2 * 4) as u64,
                           nh_p, g2 as i64, split_len as i32))?;
                if xcheck && !four {
                    self.dsync()?;
                    xref = Some((self.dtoh(&psc.qsa_pm)?, self.dtoh(&psc.qsa_pl)?,
                                 self.dtoh(&psc.qsa_pacc)?));
                }
            }
            if let Some((rpm, rpl, racc)) = xref {
                self.dsync()?;
                let (tpm, tpl, tacc) = (self.dtoh(&psc.qsa_pm)?, self.dtoh(&psc.qsa_pl)?,
                                        self.dtoh(&psc.qsa_pacc)?);
                let (a, b) = (t0g * cfg.num_heads * QSA_SPLITS, (t0g + rows) * cfg.num_heads * QSA_SPLITS);
                let nd = |x: &[f32], y: &[f32]| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                let (dm, dl) = (nd(&rpm[a..b], &tpm[a..b]), nd(&rpl[a..b], &tpl[a..b]));
                let da = nd(&racc[a * cfg.head_dim..b * cfg.head_dim], &tacc[a * cfg.head_dim..b * cfg.head_dim]);
                crate::dprintln!("SEL_XCHECK layer {att_i} rows {rows} (t0 {t0}): splitk4 vs splitk bit-differ pm {dm} pl {dl} pacc {da} of {} / {}",
                         b - a, (b - a) * cfg.head_dim);
            }
            xqlaunch!(l, "xq_attn_sel_combine", ((rows * cfg.num_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                      ((*psc.normed.device_ptr()) as u64 + (t0g * cfg.num_heads * cfg.head_dim * 2) as u64,
                       (*psc.qsa_pm.device_ptr()) as u64 + (t0g * pmb) as u64,
                       (*psc.qsa_pl.device_ptr()) as u64 + (t0g * pmb) as u64,
                       (*psc.qsa_pacc.device_ptr()) as u64 + (t0g * pacb) as u64,
                       (*psc.qg.device_ptr()) as u64 + (t0g * qrow) as u64,
                       nh_p, QSA_SPLITS as i32, self.rows_kvf(rows)))?;
            if xc_sel {
                // --exl3-sel-tc-xcheck: pass 0 = the tensor-core gather, pass 1 = today's f32 gather on the same rows (it serves)
                let (o0, n) = (t0g * cfg.num_heads * cfg.head_dim, rows * cfg.num_heads * cfg.head_dim);
                self.dsync()?;
                let got: Vec<u16> = self.dtoh(&psc.normed)?[o0..o0 + n].to_vec();
                match tc_out.take() {
                    None => tc_out = Some(got),
                    Some(tcv) => {
                        let (mut num, mut den, mut mx, mut nonfinite) = (0f64, 0f64, 0f32, 0usize);
                        for (x, y) in tcv.iter().zip(got.iter()) {
                            let (a1, b1) = (half::f16::from_bits(*x).to_f32(), half::f16::from_bits(*y).to_f32());
                            if !a1.is_finite() || !b1.is_finite() { nonfinite += 1; continue; }
                            num += ((a1 - b1) as f64).powi(2); den += (b1 as f64).powi(2); mx = mx.max((a1 - b1).abs());
                        }
                        crate::reprintln!("[seltc-xcheck] layer {att_i} pos0 {pos0} rows {rows} (t0 {t0}) {} query heads per KV head: \
    rel-L2 {:.3e} max|d| {mx:.3e} non-finite {nonfinite} (tensor-core gather vs {})", gqa, (num / den.max(1e-30)).sqrt(),
                                          if pq8 { "xq_attn_sel_pq8" } else { "xq_attn_sel_splitk4" });
                    }
                }
            }
        }
        Ok(())
    }

    /// PQ8 item 1 (SURPASS_PLAN WP06 item 1): the chunk's K/V append — rows pos0..pos0+c of `slot`
    /// into `cache` (a trunk k_cache, or the MTP head's kv), from psc.kp/psc.vp/psc.cos; the
    /// contract of xq_attn_kv_prefill. Default: the row-parallel twin xq_attn_kv_prefill_rows (grid
    /// c*nkv — every cache byte equal to the serial kernel's); --pq8-kvw=0 / --pq8-off=1 =
    /// the serial kernel (grid nkv). --pq8-xcheck=1 (eager): serial kernel first (reference),
    /// then for poison 0x00 and 0xFF: poison the rows, run the twin, diff every byte.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn kv_prefill_write(&self, l: &Launcher, psc: &mut PrefillScratch, cache: &CudaSlice<u8>,
                        qnkn: &CudaSlice<u16>, slot: usize, pos0: usize, c: usize, tag: &str) -> Result<()> {
        // TP-B: the trunk caches hold this rank's KV heads; the replicated head's holds all of them.
        let cfg = if self.mtp.as_ref().map_or(false, |hh| std::ptr::eq(cache, &hh.kv)) { &self.cfg } else { &self.tc };
        let (nkv, hd, rdim) = (cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);
        let cache_p = *cache.device_ptr() as u64;
        let nkv_hd = (nkv as i32) | ((hd as i32) << 16);
        let rdim_mp = (rdim as i32) | ((self.max_pos as i32) << 8) | ((self.kv_fmt as i32) << 28);
        let slot_pos0: i32 = (slot as i32) | ((pos0 as i32) << 12);
        let rows = pq8_item(0);
        let xcheck = rows && pq8_xcheck() && !self.stream_capturing();
        if !rows || xcheck {
            xqlaunch!(l, "xq_attn_kv_prefill", (nkv as u32, 1, 1), (hd as u32, 1, 1), 0,
                      (&mut psc.kp, &mut psc.vp, cache_p, qnkn, &psc.cos, nkv_hd, rdim_mp, slot_pos0,
                       c as i32, cfg.rms_eps))?;
            if !rows { return Ok(()); }
        }
        if !xcheck {
            xqlaunch!(l, "xq_attn_kv_prefill_rows", ((c * nkv) as u32, 1, 1), (hd as u32, 1, 1), 0,
                      (&mut psc.kp, &mut psc.vp, cache_p, qnkn, &psc.cos, nkv_hd, rdim_mp, slot_pos0,
                       c as i32, cfg.rms_eps))?;
            return Ok(());
        }
        // ---- XCHECK: the 2*nkv row segments (K|V x kv head) of this chunk, as xq_kv_make lays them out
        let rb = kv_rowbytes(self.kv_fmt, hd);
        let seg = |g: usize| (slot * 2 * nkv + g) * self.max_pos * rb + pos0 * rb;
        let snap = |dst: &mut Vec<u8>| -> Result<()> {
            self.dsync()?;
            dst.resize(2 * nkv * c * rb, 0);
            for g in 0..2 * nkv {
                self.dtoh_into(&cache.slice(seg(g)..seg(g) + c * rb),
                                             &mut dst[g * c * rb..(g + 1) * c * rb])?;
            }
            Ok(())
        };
        let mut want = Vec::new();
        snap(&mut want)?;
        let mut nd = [0usize; 2];
        for (pi, poison) in [0x0000i32, 0xFFFF].into_iter().enumerate() {
            let nh16 = (c * rb / 2) as i64;
            for g in 0..2 * nkv {
                xqlaunch!(l, "xq_memset_u16", (((nh16 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                          (cache_p + seg(g) as u64, nh16, poison))?;
            }
            xqlaunch!(l, "xq_attn_kv_prefill_rows", ((c * nkv) as u32, 1, 1), (hd as u32, 1, 1), 0,
                      (&mut psc.kp, &mut psc.vp, cache_p, qnkn, &psc.cos, nkv_hd, rdim_mp, slot_pos0,
                       c as i32, cfg.rms_eps))?;
            let mut got = Vec::new();
            snap(&mut got)?;
            nd[pi] = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        }
        crate::dprintln!("PQ8_XCHECK kvw {tag} pos0 {pos0} c {c} fmt {}: rows-twin vs serial bytes differ {} (poison 0x00) / {} (poison 0xFF) of {}{}",
                 kv_fmt_name(self.kv_fmt), nd[0], nd[1], want.len(),
                 if nd == [0, 0] { "" } else { "  <-- MISMATCH" });
        Ok(())
    }

    /// PQ8 item 2: the dense-row flash launch (rows [0, t) of the chunk; hd = 256). p5d default:
    /// the bitwise twin xq_attn_prefill_flash256v (PQ8-v2 stages flash256's exact smem image —
    /// .14, 22022ab, binary d9eb3f4f: PQ8_XCHECK flash differ 0 on every layer, f32 and q8; all 8
    /// f32/q8 greedy hashes exact; q8 prefill 2K 995 -> 1,155 tok/s); --pq8-flash=0 =
    /// xq_attn_prefill_flash256. (p3 had it default OFF: v1 was NOT bitwise — p4c @ 6300674 on
    /// .14, 6,039,782 of 12,582,912 halves differ.) --pq8-flash=causal (opt-in, a behaviour
    /// change): xq_attn_prefill_flash256c, the twin with causal K/V staging (flash256 zeroes dims
    /// of the diagonal tile's rows; PQ8-v2 note).
    /// --pq8-xcheck=1 (eager): old kernel first (reference), poison psc.normed rows [0, t)
    /// with f16 NaN (0xFFFF), the twin, diff every f16 of those rows.
    #[allow(clippy::too_many_arguments)]
    /// PFTC: does this layer's dense prefill take the tensor-core kernels? `--exl3-dense-pf-tc on` (default), a layer
    /// WITHOUT an indexer (never Flash-Next: its dense rows keep flash256, byte-for-byte), head_dim 256 and a GQA group
    /// that divides the 96-row block (35B 16/2 = 8, 27B 24/4 = 6, 9B 16/4 = 4; per rank under TP).
    /// POSTURE1 `--exl3-dense-pf-tc-qsa on` (opt-in, output-changing): the dense prefill rows (positions up to the sparse
    /// switch) of a layer WITH an indexer (Flash-Next) also take PFTC; the rows past the switch keep their gather.
    pub(super) fn pf_tc_qsa_ok(&self, a: &AttnLayer, att_i: usize) -> bool {
        let cfg = self.acfg_i(att_i);
        pf_tc_qsa_on() && a.idx.is_some() && cfg.has_indexer() && pf_tc_geometry_ok(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim)
    }

    pub(super) fn pf_tc_ok(&self, a: &AttnLayer, att_i: usize) -> bool {
        let cfg = self.acfg_i(att_i);
        dense_pf_tc_on() && a.idx.is_none() && !cfg.has_indexer() && pf_tc_geometry_ok(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim)
    }

    /// PFTC: q staging (norm + rope [+ q8 rotation] -> f16) -> tensor-core attention (un-gated f32 o/l into
    /// psc.qstage, unused by dense prefill on these packs) -> rotate back + gate into psc.normed rows [0, t).
    pub(super) fn pf_tc_launch(&self, l: &Launcher, psc: &mut PrefillScratch, a: &AttnLayer, att_i: usize,
                    slot_pos0: i32, t: usize) -> Result<()> {
        let cfg = self.acfg_i(att_i);
        let (nh_a, nkv, hd, rdim) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);
        let nh_nkv = (nkv as i32) | ((nh_a as i32) << 16);
        let hd_rdim = (hd as i32) | ((rdim as i32) << 16);
        let kc_p = *self.k_cache[att_i].device_ptr() as u64;
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::reprintln!("[exl3] PFTC: dense prefill attention ENGAGED on the tensor cores (96-row blocks, \
            {} query heads per KV head share each K/V tile); /metrics velogb10_exl3_pf_tc_rows_total counts the prefill rows", nh_a / nkv.max(1)));
        crate::metrics::exl3_pf_tc_rows(t as u64);
        xqlaunch!(l, "xq_attn_pf_qstage", ((t * nh_a) as u32, 1, 1), (hd as u32, 1, 1), 0,
                  (&mut psc.pf_qh, &psc.qg, &a.qnkn, &psc.cos, nh_nkv, hd_rdim, self.mpf(), cfg.rms_eps))?;
        let br_t = 96 / (nh_a / nkv);
        // round 3: the cp.async-pipelined twin (the next K/V tile's raw bytes fly while this one computes; bitwise ==
        // xq_attn_pf_tc256) whenever the cache row fits its one-tile raw buffer (q8 / fp8 / f16; f32 keeps the old body).
        let rbytes = kv_rowbytes(self.kv_fmt, hd);
        if rbytes <= 512 && rbytes % 16 == 0 {
            static P_ONCE: std::sync::Once = std::sync::Once::new();
            P_ONCE.call_once(|| crate::reprintln!("[exl3] PFTC: K/V tiles prefetched with cp.async while the previous tile computes (round 3, {} cache)",
                                                  kv_fmt_name(self.kv_fmt)));
            let smem = (2 * 32 * hd * 2 + 32 * 2 * rbytes) as u32;   // Ks + Vs f16 tiles + one raw tile
            xqlaunch_raw!(l, "xq_attn_pf_tc256p", (t.div_ceil(br_t) as u32, nkv as u32, 1), (192, 1, 1), smem,
                          (&mut psc.qstage, &psc.pf_qh, kc_p, nh_nkv, self.mpf(), slot_pos0, t as i32))?;
        } else {
            xqlaunch!(l, "xq_attn_pf_tc256", (t.div_ceil(br_t) as u32, nkv as u32, 1), (192, 1, 1), 0,
                      (&mut psc.qstage, &psc.pf_qh, kc_p, nh_nkv, self.mpf(), slot_pos0, t as i32))?;
        }
        xqlaunch!(l, "xq_attn_pf_tc_epi", ((t * nh_a) as u32, 1, 1), (hd as u32, 1, 1), 0,
                  (&mut psc.normed, &psc.qstage, &psc.qg, nh_nkv, hd as i32, self.mpf()))?;
        if pf_tc_xcheck_on() && !self.stream_capturing() {
            // --exl3-pf-tc-xcheck (diagnostic, eager): the scalar flash256 on the same rows into psc.normed; one line per
            // (layer, chunk): rel-L2 and max |diff| of PFTC vs flash256 over rows [0, t). The flash256 output stays
            // (the run serves the old kernel's attention while checking).
            let n = t * nh_a * hd;
            self.dsync()?;
            let tc: Vec<u16> = self.dtoh(&psc.normed)?[..n].to_vec();
            self.flash256_launch(l, psc, a, att_i, slot_pos0, t)?;
            self.dsync()?;
            let rf: Vec<u16> = self.dtoh(&psc.normed)?[..n].to_vec();
            let (mut num, mut den, mut mx, mut nonfinite) = (0f64, 0f64, 0f32, 0usize);
            for (x, y) in tc.iter().zip(rf.iter()) {
                let (a1, b1) = (half::f16::from_bits(*x).to_f32(), half::f16::from_bits(*y).to_f32());
                if !a1.is_finite() || !b1.is_finite() { nonfinite += 1; continue; }
                num += ((a1 - b1) as f64).powi(2); den += (b1 as f64).powi(2); mx = mx.max((a1 - b1).abs());
            }
            let pos0 = (slot_pos0 as u32 >> 12) as usize;
            crate::reprintln!("[pftc-xcheck] layer {att_i} pos0 {pos0} rows {t}: rel-L2 {:.3e} max|d| {mx:.3e} non-finite {nonfinite}",
                              (num / den.max(1e-30)).sqrt());
        }
        Ok(())
    }

    pub(super) fn flash256_launch(&self, l: &Launcher, psc: &mut PrefillScratch, a: &AttnLayer, att_i: usize,
                       slot_pos0: i32, t: usize) -> Result<()> {
        let cfg = self.acfg_i(att_i); // TP-B: per-rank heads (trunk) / global (the replicated head)
        let (nh_a, nkv, hd, rdim) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);
        let kc_p = *self.k_cache[att_i].device_ptr() as u64;
        let v = pq8_item(1);
        let causal = v && pq8_flash_causal();
        let xcheck = v && pq8_xcheck() && !self.stream_capturing();
        let twin = if causal { "xq_attn_prefill_flash256c" } else { "xq_attn_prefill_flash256v" };
        let pair = ["xq_attn_prefill_flash256", twin];
        let names: &[&str] = if xcheck { &pair[..] } else if v { &pair[1..] } else { &pair[..1] };
        let n16 = t * nh_a * hd;
        let mut want: Vec<u16> = Vec::new();
        for (ki, &kname) in names.iter().enumerate() {
            if xcheck && ki == 1 {
                self.dsync()?;
                want = self.dtoh(&psc.normed)?;
                want.truncate(n16);
                xqlaunch!(l, "xq_memset_u16", ((((n16 as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                          (*psc.normed.device_ptr() as u64, n16 as i64, 0xFFFFi32))?;
            }
            xqlaunch!(l, kname, (((t as u32) + 7) / 8, nh_a as u32, 1), (256, 1, 1), 0,
                      (&mut psc.normed, &psc.qg, kc_p, &a.qnkn, &psc.cos, (nkv as i32) | ((nh_a as i32) << 16),
                       (hd as i32) | ((rdim as i32) << 16), self.mpf(), slot_pos0, t as i32, cfg.rms_eps))?;
        }
        if xcheck {
            self.dsync()?;
            let got = self.dtoh(&psc.normed)?;
            let nd = want.iter().zip(&got[..n16]).filter(|(x, y)| x != y).count();
            crate::dprintln!("PQ8_XCHECK flash layer {att_i} pos0 {} rows {t} fmt {}: {} vs flash256 f16 differ {nd} of {n16}{}",
                     ((slot_pos0 as u32) >> 12), kv_fmt_name(self.kv_fmt), twin.trim_start_matches("xq_attn_prefill_"),
                     if causal { "  (causal staging: differs by design)" } else if nd == 0 { "" } else { "  <-- MISMATCH" });
        }
        Ok(())
    }

    /// PACKPF: kv_prefill_write's row-parallel append for segment `sg` (rows sg.r0.. of kp / vp / cos).
    fn kv_prefill_write_seg(&self, l: &Launcher, psc: &mut PrefillScratch, cache: &CudaSlice<u8>,
                            qnkn: &CudaSlice<u16>, sg: PfSeg) -> Result<()> {
        let cfg = if self.mtp.as_ref().map_or(false, |hh| std::ptr::eq(cache, &hh.kv)) { &self.cfg } else { &self.tc };
        let (nkv, hd, rdim) = (cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);
        anyhow::ensure!(pq8_item(0), "PACKPF: needs the row-parallel K/V append (--pq8-kvw)");
        let cache_p = *cache.device_ptr() as u64;
        let nkv_hd = (nkv as i32) | ((hd as i32) << 16);
        let rdim_mp = (rdim as i32) | ((self.max_pos as i32) << 8) | ((self.kv_fmt as i32) << 28);
        let slot_pos0: i32 = (sg.slot as i32) | ((sg.pos0 as i32) << 12);
        let ok = (sg.r0 * nkv * hd * 2) as u64;
        let oc = (sg.r0 * 2 * rdim * 4) as u64;
        xqlaunch!(l, "xq_attn_kv_prefill_rows", ((sg.n * nkv) as u32, 1, 1), (hd as u32, 1, 1), 0,
                  (*psc.kp.device_ptr() as u64 + ok, *psc.vp.device_ptr() as u64 + ok, cache_p, qnkn,
                   *psc.cos.device_ptr() as u64 + oc, nkv_hd, rdim_mp, slot_pos0, sg.n as i32, cfg.rms_eps))?;
        Ok(())
    }

    /// PACKPF: the dense attention kernels of the unpacked chunk (same selection rule: flash256c /
    /// flash256v / flash256 for t >= 16 or deep tails, else the sweep) for `t` rows from pass row `r0`.
    #[allow(clippy::too_many_arguments)]
    fn dense_attn_rows_off(&self, l: &Launcher, psc: &mut PrefillScratch, a: &AttnLayer, att_i: usize,
                           slot: usize, pos0: usize, t: usize, r0: usize) -> Result<()> {
        let cfg = self.acfg_i(att_i);
        let (nh_a, nkv, hd, rdim) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);
        anyhow::ensure!(hd == 256 && !self.pf_tc_ok(a, att_i), "PACKPF: dense packed attention is the hd-256 flash / sweep class");
        let slot_pos0: i32 = (slot as i32) | ((pos0 as i32) << 12);
        let flash = (t >= 16 || (t >= 1 && pos0 >= 256 && pos0 + 8 <= self.max_pos))
            && !matches!(crate::opts::var(crate::opt!("attn-flash")).as_deref(), Ok("0"));
        let po = *psc.normed.device_ptr() as u64 + (r0 * nh_a * hd * 2) as u64;
        let pq = *psc.qg.device_ptr() as u64 + (r0 * nh_a * hd * 2 * 2) as u64;
        let pc = *psc.cos.device_ptr() as u64 + (r0 * 2 * rdim * 4) as u64;
        let kc_p = *self.k_cache[att_i].device_ptr() as u64;
        let nh_nkv = (nkv as i32) | ((nh_a as i32) << 16);
        let hd_rdim = (hd as i32) | ((rdim as i32) << 16);
        if flash {
            // flash256_launch's kernel choice (no xcheck in a packed pass)
            let v = pq8_item(1);
            let kname = if !v { "xq_attn_prefill_flash256" } else if pq8_flash_causal() { "xq_attn_prefill_flash256c" } else { "xq_attn_prefill_flash256v" };
            xqlaunch!(l, kname, (((t as u32) + 7) / 8, nh_a as u32, 1), (256, 1, 1), 0,
                      (po, pq, kc_p, &a.qnkn, pc, nh_nkv, hd_rdim, self.mpf(), slot_pos0, t as i32, cfg.rms_eps))?;
        } else {
            xqlaunch!(l, "xq_attn_prefill_q", (nh_a as u32, 1, 1), (hd as u32, 1, 1), 0,
                      (po, pq, kc_p, &a.qnkn, pc, nh_nkv, hd_rdim, self.mpf(), slot_pos0, t as i32, cfg.rms_eps))?;
        }
        Ok(())
    }

    /// PACKPF: one attention layer of a packed pass — K/V append per segment, the indexer projection
    /// and raw-key append once over all rows, then per segment: its dense rows (flash / sweep), its
    /// sparse rows past the QSA switch (pooled refresh, scoring, top-k, gathered attention) or, for
    /// a fully dense segment, the pooled-plane refresh. Each segment's bytes equal its unpacked
    /// chunk's: the same kernels on the same rows, row-offset pointers.
    pub(super) fn attn_prefill_segs(&self, l: &Launcher, psc: &mut PrefillScratch, a: &AttnLayer, att_i: usize,
                         segs: &[PfSeg], c: usize) -> Result<()> {
        for sg in segs { self.kv_prefill_write_seg(l, psc, &self.k_cache[att_i], &a.qnkn, *sg)?; }
        let cfg = &self.cfg;
        // the indexer rows (raw keys are spec state on every forward; the sparse rows also read the
        // roped q heads), once over the pass
        let idx_live = a.idx.is_some() && cfg.has_indexer() && !qsa_disabled() && self.qsa_keys_p(att_i).is_some()
            && psc.qsa_qk.len() >= c * (cfg.indexer_n_heads + 1) * cfg.indexer_head_dim
            && psc.qsa_qkb.len() >= c * (cfg.indexer_n_heads + 1) * cfg.indexer_head_dim
            && psc.pos_rows.len() >= c;
        let any_sparse = segs.iter().any(|sg| a.idx.is_some() && self.qsa_live(sg.pos0 + sg.n - 1));
        anyhow::ensure!(!any_sparse || idx_live, "PACKPF: sparse rows without the indexer buffers");
        if idx_live { self.qsa_pf_index_rows(l, psc, a, att_i, c)?; }
        for sg in segs {
            let (pos0, n) = (sg.pos0, sg.n);
            let t = if a.idx.is_some() && self.qsa_live(pos0 + n - 1) {
                let lim = self.qsa_limit();
                if pos0 > lim { 0 } else { (lim + 1 - pos0).min(n) }
            } else { n };
            if t > 0 { self.dense_attn_rows_off(l, psc, a, att_i, sg.slot, pos0, t, sg.r0)?; }
            if t < n {
                self.qsa_pf_sparse_rows(l, psc, a, att_i, sg.slot, pos0, t, n, sg.r0)?;
            } else if idx_live {
                let keys_p = self.qsa_keys_p(att_i).unwrap();
                self.qsa_pool_range(l, a, att_i, keys_p, sg.slot, pos0, n)?;
            }
        }
        Ok(())
    }

    /// PACKPF: the indexer-key part of qsa_keys_append_dense_rows over a packed pass (the head's
    /// attention: no sparse rows): chain + key write once, the pooled refresh per segment.
    fn qsa_keys_append_dense_segs(&self, l: &Launcher, psc: &mut PrefillScratch, a: &AttnLayer,
                                  att_i: usize, segs: &[PfSeg], c: usize) -> Result<()> {
        let Some(ix) = a.idx.as_ref() else { return Ok(()); };
        if !self.cfg.has_indexer() || qsa_disabled() { return Ok(()); }
        let Some(keys_p) = self.qsa_keys_p(att_i) else { return Ok(()); };
        let cfg = &self.cfg;
        let (hdx, heads) = (cfg.indexer_head_dim, cfg.indexer_n_heads);
        let qk_dim = (heads + 1) * hdx;
        if psc.qsa_qk.len() < c * qk_dim || psc.qsa_qkb.len() < c * qk_dim || psc.pos_rows.len() < c { return Ok(()); }
        pf_chain_check_n(&ix.qk, c, &psc.chain_xh, &psc.chain_yraw)?;
        let rb = self.recon_bufs(psc);
        exl3_chain_rb(l, &ix.qk, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.qsa_qk, c, &psc.chain_ws, &rb)?;
        let ncnv = (c * qk_dim) as i64;
        xqlaunch!(l, "xq_qsa_h2b", ((((ncnv + 255) / 256).max(1)) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut psc.qsa_qkb, &psc.qsa_qk, ncnv))?;
        gblaunch!(l, "qsa_key_write_b", ((c * hdx) as u32, 1, 1), (256, 1, 1), 0,
                  (keys_p, &psc.qsa_qkb, &psc.pos_rows, &psc.slot_rows, self.max_pos as i32, 0i32,
                   qk_dim as i32, (heads * hdx) as i32, hdx as i32, c as i32))?;
        for sg in segs { self.qsa_pool_range(l, a, att_i, keys_p, sg.slot, sg.pos0, sg.n)?; }
        Ok(())
    }

    /// PACKPF: head_fill_rows over a packed pass. Row r0 of each segment pairs with that segment's
    /// carry (zero at pos0 = 0, else its slot's entry of the per-slot carry store), rows r0+1.. with
    /// the streams after the previous row of the same segment; each segment's new carry (its last
    /// row's streams) goes back to its slot's entry. Then the head's entry, hc mix, keys and K/V
    /// once over all rows (the fc chains at m = c * hc split per segment by LT_SEGS).
    pub(super) fn head_fill_segs(&self, l: &Launcher, psc: &mut PrefillScratch, head: &DraftHead,
                      segs: &[PfSeg], c: usize) -> Result<()> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let hcn = cfg.hc_count.max(1);
        let rw = h * hcn;
        anyhow::ensure!(psc.hfill_s.len() >= c * rw, "PACKPF: head-fill rows {} < {}", psc.hfill_s.len() / rw.max(1), c);
        anyhow::ensure!(psc.hfill_carry_slots.len() >= self.width * rw, "PACKPF: carry store too small");
        let rb = self.recon_bufs(psc);
        let g = |n: usize| (((n as u32) + 255) / 256, 1u32, 1u32);
        if c > 1 {
            xqlaunch!(l, "xq_copy_f32", g((c - 1) * rw), (256, 1, 1), 0,
                      (&mut psc.hfill_s, &psc.resid, ((c - 1) * rw) as i64, rw as i64, 0i64))?;
        }
        for sg in segs {
            if sg.pos0 == 0 {
                xqlaunch!(l, "xq_memset_f32", g(rw), (256, 1, 1), 0, (&psc.hfill_s, (sg.r0 * rw) as i64, rw as i64))?;
            } else {
                xqlaunch!(l, "xq_copy_f32", g(rw), (256, 1, 1), 0,
                          (&mut psc.hfill_s, &psc.hfill_carry_slots, rw as i64, (sg.r0 * rw) as i64, (sg.slot * rw) as i64))?;
            }
            xqlaunch!(l, "xq_copy_f32", g(rw), (256, 1, 1), 0,
                      (&mut psc.hfill_carry_slots, &psc.resid, rw as i64, (sg.slot * rw) as i64, ((sg.r0 + sg.n - 1) * rw) as i64))?;
        }
        self.mtp_pre_norm(l, head, 0, *psc.hn.device_ptr() as u64, *psc.hfill_s.device_ptr() as u64, c)?;
        exl3_chain_rb(l, &head.fc_hidden, &psc.hn, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.hfill_a,
                      c * hcn, &psc.chain_ws, &rb)?;
        self.embed_gather(l, *psc.hfill_s.device_ptr() as u64, *psc.toks.device_ptr() as u64, c)?;
        self.mtp_pre_norm(l, head, 1, *psc.hn.device_ptr() as u64, *psc.hfill_s.device_ptr() as u64, c)?;
        exl3_chain_rb(l, &head.fc_embed, &psc.hn, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.hfill_b,
                      c * hcn, &psc.chain_ws, &rb)?;
        xqlaunch!(l, "xq_add_f16_f32", (((rw * c) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                  (&mut psc.resid, &psc.hfill_a, &psc.hfill_b, (rw * c) as i64))?;
        self.hc_pre_rows(l, psc, &head.hc_attn, c)?;
        let a = &head.attn;
        self.qsa_keys_append_dense_segs(l, psc, a, HEAD_ATT, segs, c)?;
        exl3_chain_rb(l, &a.k_proj, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.kp, c, &psc.chain_ws, &rb)?;
        exl3_chain_rb(l, &a.v_proj, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.vp, c, &psc.chain_ws, &rb)?;
        for sg in segs { self.kv_prefill_write_seg(l, psc, &head.kv, &a.qnkn, *sg)?; }
        Ok(())
    }

    /// S-A3-t: CHUNKED fill of the MTP draft head's KV cache over prompt rows [pos0, pos0+c) —
    /// the reference implementation's MTP-layer prefill (`draft_model.prefill` over every prompt token with the
    /// shifted target hidden). Position p gets the head's K/V computed from (trunk streams at
    /// p-1, token p) — the same pairing the draft passes use (tap that predicted b, b) — with
    /// the previous chunk's last trunk row carried across chunks and zeros at p = 0 (the reference implementation's
    /// carry). Only the head's entry + hc_attn pre-mix + k/v projections run: its KV depends on
    /// nothing downstream (no attention, MoE or logits). All wide-M launches, no per-token loop.
    /// Runs after the trunk's final mixer: it overwrites psc.resid/hn/x (nothing reads them after
    /// a chunk returns). The serve's single last-position prime (head_prime) is unchanged.
    pub(super) fn head_fill_rows(&self, l: &Launcher, psc: &mut PrefillScratch, head: &DraftHead,
                      slot: usize, pos0: usize, c: usize) -> Result<()> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let hcn = cfg.hc_count.max(1);
        let rw = h * hcn;
        if psc.hfill_s.len() < c * rw { return Ok(()); }
        let rb = self.recon_bufs(psc);
        let g = |n: usize| (((n as u32) + 255) / 256, 1u32, 1u32);
        // 1. shifted trunk streams: row 0 = carry (zero at the prompt start), rows 1.. = resid rows 0..c-2
        if pos0 == 0 {
            xqlaunch!(l, "xq_memset_f32", g(rw), (256, 1, 1), 0, (&psc.hfill_carry, 0i64, rw as i64))?;
        }
        xqlaunch!(l, "xq_copy_f32", g(rw), (256, 1, 1), 0,
                  (&mut psc.hfill_s, &psc.hfill_carry, rw as i64, 0i64, 0i64))?;
        if c > 1 {
            xqlaunch!(l, "xq_copy_f32", g((c - 1) * rw), (256, 1, 1), 0,
                      (&mut psc.hfill_s, &psc.resid, ((c - 1) * rw) as i64, rw as i64, 0i64))?;
        }
        xqlaunch!(l, "xq_copy_f32", g(rw), (256, 1, 1), 0,
                  (&mut psc.hfill_carry, &psc.resid, rw as i64, 0i64, ((c - 1) * rw) as i64))?;
        // 2. entry (head_launches' exact sequence at m = c rows): norm(hidden) -> fc_hidden;
        //    embed -> norm -> fc_embed; streams = sum (f32)
        self.mtp_pre_norm(l, head, 0, *psc.hn.device_ptr() as u64, *psc.hfill_s.device_ptr() as u64, c)?;
        exl3_chain_rb(l, &head.fc_hidden, &psc.hn, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.hfill_a,
                      c * hcn, &psc.chain_ws, &rb)?;
        self.embed_gather(l, *psc.hfill_s.device_ptr() as u64, *psc.toks.device_ptr() as u64, c)?;
        self.mtp_pre_norm(l, head, 1, *psc.hn.device_ptr() as u64, *psc.hfill_s.device_ptr() as u64, c)?;
        exl3_chain_rb(l, &head.fc_embed, &psc.hn, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.hfill_b,
                      c * hcn, &psc.chain_ws, &rb)?;
        xqlaunch!(l, "xq_add_f16_f32", (((rw * c) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                  (&mut psc.resid, &psc.hfill_a, &psc.hfill_b, (rw * c) as i64))?;
        // 3. the head block's pre-attention mix, then k/v and the cache write (slot, main positions)
        self.hc_pre_rows(l, psc, &head.hc_attn, c)?;
        let a = &head.attn;
        // head QSA: the head's raw indexer keys over the prompt rows (spec state on every forward)
        self.qsa_keys_append_dense_rows(l, psc, a, HEAD_ATT, slot, pos0, c)?;
        exl3_chain_rb(l, &a.k_proj, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.kp, c, &psc.chain_ws, &rb)?;
        exl3_chain_rb(l, &a.v_proj, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.vp, c, &psc.chain_ws, &rb)?;
        // PQ8 item 1: row-parallel K/V append into the head's cache (--pq8-kvw=0 = serial)
        self.kv_prefill_write(l, psc, &head.kv, &a.qnkn, slot, pos0, c, "head")?;
        Ok(())
    }

}
