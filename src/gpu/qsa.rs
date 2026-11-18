//! qwen4_exp QSA sparse-attention indexer, host side: `QsaIn` and the `qsa_*` methods. Gated by `qsa_enabled(kv_stride)`;
//! off, the engine is the pre-indexer engine byte for byte. Map: `mod.rs`.

use super::*;

// ===================== qwen4_exp QSA sparse-attention indexer (host side) =====================
// Kernels + reference notes: kernels/gpu_batch.cu, "qwen4_exp QSA". Everything here is gated by
// `qsa_enabled(kv_stride)` — off, the engine is the pre-indexer engine byte for byte.

/// Per-call indexer inputs for `full_attn_batch`: the layer's indexer, its raw-key cache (slot-0 base
/// for decode/verify — the kernels add slot_ids[b]*stride —, the SLOT base for a prefill chain), the
/// LOGICAL positions and the verify's rank→column tables (0 = identity).
pub struct QsaIn<'a> {
    pub idx: &'a GpuIndexer,
    pub keys_ptr: u64,
    pub logical_ptr: u64,
    pub path_ptr: u64,
    pub cps_ptr: u64,
}

impl GpuModel {
    /// Visible-token count up to which the indexer selects EVERY block (dense attention is exact):
    /// indexer_budget + compress_ratio - 1. Also the row pitch of the selection lists.
    pub fn qsa_limit(&self) -> usize { self.cfg.indexer_budget + self.cfg.indexer_compress_ratio - 1 }

    /// Whether the indexer is live at this context length. A query can only lose blocks once it sees
    /// more than `qsa_limit()` tokens, so a shorter `kv_stride` is served dense (bit-identical to the
    /// pre-indexer engine, CUDA graphs included). --q4-dense-attn=1 forces dense at any length —
    /// an A/B escape, NOT the reference model past the limit. The sparse kernels read a bf16 KV cache.
    pub fn qsa_enabled(&self, kv_stride: usize) -> bool {
        if !self.cfg.has_indexer() || kv_stride <= self.qsa_limit() { return false; }
        if crate::opts::var(crate::opt!("q4-dense-attn")).is_ok() { return false; }
        // k8v4 is served by gqa_attn_sel_splitk_k8v4 (decode/verify) and by the bf16 dequant
        // mirror/scratch (prefill); q4 and tq have no selection-list reader yet.
        // k8v8 (dev's dense int8 mode) has no selection-list reader either — a QSA column under
        // k8v8 would reduce sparse positions against the dense cache (TRAP 2). Fail loud.
        assert!(!(self.kv_quant || self.kv_tq || self.kv_k8v8),
                "qwen4_exp QSA sparse attention reads a bf16 or k8v4 KV cache: use --kv-cache bf16|k8v4 (or --max-seq-len <= {})", self.qsa_limit());
        true
    }

    /// Elements of the MTP head's K buffer: its KV heads, plus (QSA live) the head's raw-key cache
    /// `[kv_stride][hd_idx]` appended at the end — see `qsa_in_mtp`.
    pub fn mtp_kc_elems(&self, kv_stride: usize) -> usize {
        let base = self.mtp_kv_heads() * kv_stride * self.cfg.head_dim;
        let head_has_indexer = self.mtp.as_ref().map_or(false, |m| m.fa.indexer.is_some());
        if head_has_indexer && self.qsa_enabled(kv_stride) { base + kv_stride * self.cfg.indexer_head_dim } else { base }
    }

    /// The device `QsaParams` block (layout mirrors the kernel struct: 4 pointers, eps, 7 ints).
    pub(super) fn qsa_params(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config, cos_t: u64, sin_t: u64, kw: u64, qw: u64)
                  -> cudarc::driver::CudaSlice<u8> {
        let (hd, heads, ratio) = (cfg.indexer_head_dim, cfg.indexer_n_heads, cfg.indexer_compress_ratio);
        assert!(hd % 32 == 0 && hd <= 128, "qwen4_exp QSA: indexer_head_dim {hd} must be a multiple of 32 <= 128");
        assert!((1..=8).contains(&heads), "qwen4_exp QSA: indexer_n_heads {heads} must be 1..=8");
        assert!((1..=8).contains(&ratio), "qwen4_exp QSA: indexer_compress_ratio {ratio} must be 1..=8");
        assert!(cfg.rotary_dim <= hd && cfg.rotary_dim % 2 == 0, "qwen4_exp QSA: rotary_dim {} vs indexer_head_dim {hd}", cfg.rotary_dim);
        let mut b = Vec::with_capacity(64);
        for p in [cos_t, sin_t, kw, qw] { b.extend_from_slice(&p.to_le_bytes()); }
        b.extend_from_slice(&(cfg.rms_eps as f32).to_le_bytes());
        for v in [hd, heads, cfg.rotary_dim, ratio, cfg.indexer_budget / ratio, cfg.indexer_budget + ratio - 1, 0] {
            b.extend_from_slice(&(v as i32).to_le_bytes());
        }
        b.extend_from_slice(&0u64.to_le_bytes()); // VIS-3 QsaParams.mr: no per-slot mrope on this engine
        dev.htod_sync_copy(&b).expect("QsaParams upload")
    }

    /// Indexer inputs of main layer `li` (None when the layer has no indexer or QSA is off).
    pub(crate) fn qsa_in<'a>(&'a self, state: &BatchGpuState, fa: &'a GpuFullAttn, li: usize, kv_stride: usize,
                             logical_ptr: u64, path_ptr: u64, cps_ptr: u64) -> Option<QsaIn<'a>> {
        if !self.qsa_enabled(kv_stride) { return None; }
        let idx = fa.indexer.as_ref()?;
        let keys = state.qsa_keys.get(li).and_then(|k| k.as_ref()).expect("qwen4_exp QSA: raw-key cache missing for a full-attention layer");
        Some(QsaIn { idx, keys_ptr: d(keys), logical_ptr, path_ptr, cps_ptr })
    }

    /// Indexer inputs of the MTP head: its raw-key cache is the tail of its K buffer (`mtp_kc_elems`).
    pub(crate) fn qsa_in_mtp<'a>(&self, fa: &'a GpuFullAttn, mtp_kc_ptr: u64, kv_stride: usize, logical_ptr: u64) -> Option<QsaIn<'a>> {
        if !self.qsa_enabled(kv_stride) { return None; }
        let idx = fa.indexer.as_ref()?;
        let keys_ptr = mtp_kc_ptr + (self.mtp_kv_heads() * kv_stride * self.cfg.head_dim * 2) as u64;
        Some(QsaIn { idx, keys_ptr, logical_ptr, path_ptr: 0, cps_ptr: 0 })
    }

    /// index_qk_proj on `batch` columns → `[qk_dim, batch]` bf16 (heads*hd q | raw key), and the q
    /// heads normed (q_layernorm) + roped in place at the columns' positions (`cos`/`sin` [batch][rdim]).
    fn qsa_qk(&self, pool: &mut Pool, idx: &GpuIndexer, hidden: &B, cos: &S, sin: &S, batch: usize) -> (B, usize) {
        let cfg = &self.cfg;
        let (hdx, heads) = (cfg.indexer_head_dim, cfg.indexer_n_heads);
        let qk_dim = (heads + 1) * hdx;
        let mut qk = pool.get_bf16(qk_dim * batch);
        self.gemm_act(&idx.qk_proj, hidden, &mut qk, cfg.hidden_size, qk_dim, batch);
        blaunch!(self, "rmsnorm_rope_b", ((batch*heads) as u32,1,1), (hdx as u32,1,1), (hdx*4) as u32,
            (d(&qk), d(&idx.q_norm), d(cos), d(sin), heads as i32, hdx as i32, cfg.rotary_dim as i32, batch as i32,
             fbits(cfg.rms_eps), 0i32, qk_dim as i32));
        (qk, qk_dim)
    }

    /// Decode / verify selection: writes the columns' raw keys at (slot_ids[b], pos[b]), scores every
    /// complete block of each column's visible ranks, keeps the top-k blocks + tail. Returns
    /// (sel `[batch][sel_max]` i32 cache columns, pos_sel `[batch]` = nsel-1), pool-owned.
    pub(crate) fn qsa_select_cols(&self, pool: &mut Pool, idx: &GpuIndexer, hidden: &B, keys_ptr: u64, stride: usize,
                                  pos_ptr: u64, logical_ptr: u64, slot_ids_ptr: u64, path_ptr: u64, cps_ptr: u64,
                                  cos: &S, sin: &S, batch: usize, max_pc: usize) -> (S, S) {
        let cfg = &self.cfg;
        let (hdx, heads, ratio) = (cfg.indexer_head_dim, cfg.indexer_n_heads, cfg.indexer_compress_ratio);
        let (qk, qk_dim) = self.qsa_qk(pool, idx, hidden, cos, sin, batch);
        blaunch!(self, "qsa_key_write_b", grid(batch*hdx), (256,1,1), 0,
            (keys_ptr, d(&qk), pos_ptr, slot_ids_ptr, stride as i32, 0i32, qk_dim as i32, (heads*hdx) as i32, hdx as i32, batch as i32));
        let nblk_stride = (max_pc / ratio).max(1);
        let scores = pool.get(batch * nblk_stride);
        let geom: u64 = pack_qsa_geom(stride, qk_dim, nblk_stride);
        blaunch!(self, "qsa_score_b", (nblk_stride.div_ceil(8) as u32, batch as u32, 1), (256,1,1), 0,
            (d(&scores), d(&qk), keys_ptr, logical_ptr, slot_ids_ptr, path_ptr, cps_ptr, d(&idx.params), geom as i64));
        let sel_max = self.qsa_limit();
        let sel = pool.get(batch * sel_max);
        let pos_sel = pool.get(batch);
        blaunch!(self, "qsa_topk_b", (batch as u32,1,1), (1024,1,1), 0,
            (d(&sel), d(&pos_sel), d(&scores), logical_ptr, path_ptr, cps_ptr, d(&idx.params), nblk_stride as i32, 0i32));
        if crate::opts::var(crate::opt!("qsa-dump")).is_ok() { for b in 0..batch { self.qsa_dump_scores("cols", &scores, nblk_stride, b); } }
        pool.release(scores, batch * nblk_stride);
        pool.release_bf16(qk, qk_dim * batch);
        if crate::opts::var(crate::opt!("qsa-dump")).is_ok() { self.qsa_dump("cols", keys_ptr, &sel, &pos_sel, batch, None); }
        (sel, pos_sel)
    }

    fn qsa_dump_scores(&self, tag: &str, scores: &S, row_stride: usize, row: usize) {
        self.sync_stream();
        let mut h = vec![0f32; row_stride];
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut h, *scores.device_ptr() + (row * row_stride * 4) as u64).unwrap(); }
        eprintln!("[qsa-scores] {tag} col={row} scores={:?}", h);
    }

    /// --qsa-dump=1: print the selection lists (validation against the HF reference; syncs).
    fn qsa_dump(&self, tag: &str, keys_ptr: u64, sel: &S, pos_sel: &S, n: usize, only: Option<usize>) {
        self.sync_stream();
        let sel_max = self.qsa_limit();
        let mut hs = vec![0i32; n * sel_max]; let mut hp = vec![0i32; n];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(&mut hs, *sel.device_ptr()).unwrap();
            cudarc::driver::result::memcpy_dtoh_sync(&mut hp, *pos_sel.device_ptr()).unwrap();
        }
        for b in 0..n {
            if only.map_or(false, |o| o != b) { continue; }
            let ns = (hp[b] + 1) as usize;
            let row = &hs[b * sel_max..b * sel_max + ns];
            eprintln!("[qsa-dump] {tag} keys={keys_ptr:#x} col={b} nsel={ns} sel={:?}", row);
        }
    }

    /// Prefill selection for a chain of `n` tokens at positions pos_start.. into ONE slot (`keys_ptr`
    /// = that slot's raw-key base): block keys are built once from the committed raw keys, then every
    /// query is scored against the blocks it can see. Same return contract as `qsa_select_cols`.
    pub(crate) fn qsa_select_prefill(&self, pool: &mut Pool, idx: &GpuIndexer, hidden: &B, keys_ptr: u64, stride: usize,
                                     pos_start: usize, n: usize, cos: &S, sin: &S) -> Option<(S, S)> {
        let cfg = &self.cfg;
        let (hdx, heads, ratio) = (cfg.indexer_head_dim, cfg.indexer_n_heads, cfg.indexer_compress_ratio);
        let timing = crate::opts::var(crate::opt!("qsa-time")).is_ok();
        let t0 = std::time::Instant::now();
        let (qk, qk_dim) = self.qsa_qk(pool, idx, hidden, cos, sin, n);
        blaunch!(self, "qsa_key_write_b", grid(n*hdx), (256,1,1), 0,
            (keys_ptr, d(&qk), 0u64, 0u64, stride as i32, pos_start as i32, qk_dim as i32, (heads*hdx) as i32, hdx as i32, n as i32));
        if pos_start + n <= self.qsa_limit() {
            // Every query of this window sees at most `limit` tokens: the indexer would select every
            // block. Raw keys are cached (later windows need them); the attention stays on the dense
            // tensor-core path.
            pool.release_bf16(qk, qk_dim * n);
            return None;
        }
        let sel_max = self.qsa_limit();
        let sel = pool.get(n * sel_max);
        let pos_sel = pool.get(n);
        let nblk = (pos_start + n) / ratio;
        if nblk == 0 {
            // Every query sees fewer than `ratio` tokens: tail ranks only, no scores read.
            blaunch!(self, "qsa_topk_b", (n as u32,1,1), (1024,1,1), 0,
                (d(&sel), d(&pos_sel), d(&pos_sel), 0u64, 0u64, 0u64, d(&idx.params), 1i32, pos_start as i32));
        } else {
            let blocks = pool.get_bf16(nblk * hdx);
            blaunch!(self, "qsa_block_keys_b", (nblk.div_ceil(8) as u32,1,1), (256,1,1), 0,
                (d(&blocks), keys_ptr, d(&idx.params), nblk as i32, -1i32));
            if timing { self.sync_stream(); eprintln!("[qsa-time] prefill n={n} pos_start={pos_start}: qk+keys+blocks {:.2} ms", t0.elapsed().as_secs_f32()*1e3); }
            // Scores: one tensor-core GEMM batched over the heads (A = the block keys, shared across
            // the batch via strideA = 0; B = head h of the roped q at offset h*hd, ld = qk_dim;
            // 1/√hd folded into alpha — relu commutes with a positive scale), then a relu-sum +
            // causal-mask combine. --qsa-scalar=1 keeps the scalar kernel (A/B; ~+30 % prefill
            // time on a 7K prompt).
            let scalar = crate::opts::var(crate::opt!("qsa-scalar")).is_ok();
            const QCH: usize = 256;              // query chunk: bounds the [heads][QCH][nblk] f32 GEMM scratch
            const QT: usize = 8;                 // QSA_PF_QT
            let qch = n.min(QCH);
            let scores = pool.get(qch * nblk);
            let gbuf = if scalar { None } else { Some(pool.get(heads * qch * nblk)) };
            let smem = (QT * heads * hdx * 4) as u32;
            for t0 in (0..n).step_by(QCH) {
                let nch = (n - t0).min(QCH);
                if let Some(g) = &gbuf {
                    use cudarc::cublas::sys::{cudaDataType, cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t as OP};
                    let alpha: f32 = 1.0 / (hdx as f32).sqrt();
                    let zero: f32 = 0.0;
                    unsafe {
                        cudarc::cublas::result::gemm_strided_batched_ex(
                            *self.blas.handle(),
                            OP::CUBLAS_OP_T, OP::CUBLAS_OP_N,
                            nblk as i32, nch as i32, hdx as i32,
                            &alpha as *const f32 as *const _,
                            *blocks.device_ptr() as *const _, cudaDataType::CUDA_R_16BF, hdx as i32, 0i64,
                            (d(&qk) + (t0 * qk_dim * 2) as u64) as *const _, cudaDataType::CUDA_R_16BF, qk_dim as i32, hdx as i64,
                            &zero as *const f32 as *const _,
                            *g.device_ptr() as *mut _, cudaDataType::CUDA_R_32F, nblk as i32, (nch * nblk) as i64,
                            heads as i32, cublasComputeType_t::CUBLAS_COMPUTE_32F, cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                        ).expect("qsa score GEMM");
                    }
                    blaunch!(self, "qsa_score_combine_b", grid(nch * nblk), (256,1,1), 0,
                        (d(&scores), d(g), d(&idx.params), (pos_start + t0) as i32, nch as i32, nblk as i32));
                } else {
                blaunch!(self, "qsa_score_prefill_b", (nblk.div_ceil(256) as u32, nch.div_ceil(QT) as u32, 1), (256,1,1), smem,
                    (d(&scores), d(&qk) + (t0 * qk_dim * 2) as u64, d(&blocks), d(&idx.params),
                     (pos_start + t0) as i32, nch as i32, nblk as i32, qk_dim as i32));
                }
                blaunch!(self, "qsa_topk_b", (nch as u32,1,1), (1024,1,1), 0,
                    (d(&sel) + (t0 * sel_max * 4) as u64, d(&pos_sel) + (t0 * 4) as u64, d(&scores), 0u64, 0u64, 0u64,
                     d(&idx.params), nblk as i32, (pos_start + t0) as i32));
                if crate::opts::var(crate::opt!("qsa-dump")).is_ok() && t0 + nch == n { self.qsa_dump_scores("prefill", &scores, nblk, nch - 1); }
            }
            pool.release(scores, qch * nblk);
            if let Some(g) = gbuf { pool.release(g, heads * qch * nblk); }
            pool.release_bf16(blocks, nblk * hdx);
        }
        pool.release_bf16(qk, qk_dim * n);
        if timing { self.sync_stream(); eprintln!("[qsa-time] prefill n={n}: scores+topk done at {:.2} ms", t0.elapsed().as_secs_f32()*1e3); }
        if crate::opts::var(crate::opt!("qsa-dump")).is_ok() { self.qsa_dump("prefill", keys_ptr, &sel, &pos_sel, n, Some(n - 1)); }
        Some((sel, pos_sel))
    }

    pub(crate) fn qsa_release(&self, pool: &mut Pool, sel: S, pos_sel: S, n: usize) {
        pool.release(sel, n * self.qsa_limit());
        pool.release(pos_sel, n);
    }

    /// Causal prefill attention over per-query selection lists (`gqa_attn_sel_prefill`); `kc/vc`
    /// are the slot's bases, `q` packed `[n][nh*hd]`, out `attn` `[n][nh*hd]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qsa_attn_prefill_ptrs(&self, attn: &mut B, q: &B, kc_ptr: u64, vc_ptr: u64, stride: usize,
                                        nh: usize, nkv: usize, hd: usize, scale: f32, n: usize, sel_ptr: u64, pos_sel_ptr: u64) {
        let nh_packed = pack_nh(nh, hd, nkv);
        if hd <= 256 && crate::opts::var(crate::opt!("qsa-sel-v1")).is_err() {
            // v2: K/V rows fetched once per 4-head group, 4 keys in flight (bit-identical to v1).
            let ngroups = (nh / nkv).div_ceil(4);
            blaunch!(self, "gqa_attn_sel_prefill2", ((n.div_ceil(8) * nkv * ngroups) as u32,1,1), (256,1,1), 0,
                (d(attn), d(q), kc_ptr, vc_ptr, stride as i32, nh_packed, fbits(scale), n as i32, sel_ptr, pos_sel_ptr, self.qsa_limit() as i32));
        } else {
        blaunch!(self, "gqa_attn_sel_prefill", ((n.div_ceil(8) * nh) as u32,1,1), (256,1,1), 0,
            (d(attn), d(q), kc_ptr, vc_ptr, stride as i32, nh_packed, fbits(scale), n as i32, sel_ptr, pos_sel_ptr, self.qsa_limit() as i32));
        }
    }
}
