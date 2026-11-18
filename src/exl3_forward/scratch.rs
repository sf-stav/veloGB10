//! `impl Scratch` (the serving scratch constructor), the QSA score-plane sizing helpers and the packed-position
//! guard, plus the FwdModel host-wait helpers (dsync, dtoh, dtoh_into). The `Scratch` struct itself is in `mod.rs`.

use super::*;

/// The widest `max_pos` (cache positions per slot) the launch-argument bit-packings can carry: the
/// QSA geometry word (`crate::gpu::pack_qsa_geom`, unpacked `geom & 0xFFFFF` in gpu_batch.cu) and
/// `rdim_mp` (`rdim | max_pos << 8 | kv_fmt << 28`, exl3_bench.cu) both give it 20 bits. One past
/// that lands in the neighbouring field, so `FwdModel::load_tp` refuses it (public PR #21).
pub const MAX_POS_PACKED: usize = 1 << 20;

/// `FwdModel::load_tp`'s window guard (pure: unit-tested without a device).
pub(crate) fn check_max_pos_packable(max_pos: usize) -> Result<()> {
    anyhow::ensure!(max_pos < MAX_POS_PACKED,
        "max_pos {max_pos} does not fit the launch-argument bit-packings (max_pos < {MAX_POS_PACKED} = 2^20: \
         the QSA geometry word and the attention rdim/max_pos word carry it in 20 bits, so a wider window \
         would overflow into a neighbouring field). Lower --max-seq-len.");
    Ok(())
}

/// Per-row pitch of every QSA score plane: one f32 per `ratio`-token block of the SERVED window.
/// The scorers write `scores[row * pitch + j]` for `j < (pos + 1) / ratio` (gpu_batch.cu
/// `qsa_score_pooled_mr` / `qsa_score_pooled_b` / `qsa_score_b`), so the scratch that receives them
/// must be sized with this same function of the same `max_pos` (public PR #21: it was sized from the
/// trained window capped at 262144, and a wider served window made rows >= 1 write past its end).
pub(crate) fn qsa_nblk_stride(max_pos: usize, ratio: usize) -> usize {
    (max_pos / ratio.max(1)).max(1)
}

/// Tail slack of the score plane, in f32. The top-k kernels load whole quads: `xq_tk3_ld4`
/// (exl3_bench.cu) reads blocks 4q..4q+3 whenever 4q < nb, so with a pitch that is not a multiple
/// of 4 the last row's last quad reaches up to 3 f32 past the row. Inside the plane that is the
/// next row (masked, never used); past the last row it must still be the scratch's own memory.
/// (The old trained-window size hid this for every window below 262144; review of PR #21's port.)
pub(crate) const QSA_SCORES_PAD: usize = 8;

/// Element count of a pooled QSA key plane (trunk `qsa_pool`, the MTP head's `qsa_pool`): `[width][nblk][hd]` bf16 with
/// `nblk = qsa_nblk_stride(max_pos, ratio)`, the SAME per-lane block stride the kernels use (`qsa_pool_update_b` /
/// `qsa_pool_range_b` / `qsa_score_pooled_b` address `slot * nblk_stride + j` and never touch `j >= nblk_stride`).
/// Issue sf-stav/veloGB10#41 (reported by @AbsolomBane): the plane was allocated with `ceil(max_pos / ratio)` blocks per
/// lane while the kernels step lanes by `floor`, and the spill / prefix-pool copies derive the lane stride from the
/// plane size: with `max_pos % ratio != 0` (every `--max-seq-len` that is a multiple of 4, since max_pos = window + 1)
/// a state restored into another lane landed its pooled keys `source - target` blocks off.
pub(crate) fn qsa_pool_plane_elems(width: usize, max_pos: usize, ratio: usize, hd: usize) -> usize {
    width * qsa_nblk_stride(max_pos, ratio) * hd
}

/// f32 count of `Scratch::qsa_scores` for `w` rows of a model serving `max_pos` positions per slot
/// (0 when the pack has no indexer): `w` rows of `qsa_nblk_stride` plus `QSA_SCORES_PAD`.
pub(crate) fn qsa_scores_len(has_indexer: bool, ratio: usize, w: usize, max_pos: usize) -> usize {
    if has_indexer { w * qsa_nblk_stride(max_pos, ratio) + QSA_SCORES_PAD } else { 0 }
}


impl Scratch {
    #[allow(clippy::too_many_arguments)]
    /// `max_pos` is the SERVED window of the model this scratch works for (`FwdModel::max_pos()`):
    /// the QSA score plane is pitched by it, never by the config's trained window.
    pub fn new(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config, w: usize, rdim: usize, max_pos: usize) -> Result<Self> {
        Self::new_vrows(dev, cfg, w, rdim, max_pos, MTP_MAX_K + 1)
    }

    /// WIDEVERIFY1 refinement 2: `new` with the lone verify's row capacity `vrows` (MTP_MAX_K + 1 =
    /// `new`; COPY_VROWS = the wide copy verify, which also needs `w >= COPY_VROWS`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_vrows(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config, w: usize, rdim: usize, max_pos: usize,
                     vrows: usize) -> Result<Self> {
        anyhow::ensure!(vrows == MTP_MAX_K + 1 || (vrows == COPY_VROWS && w >= COPY_VROWS),
                        "scratch: verify rows {vrows} (width {w}) — {} or {COPY_VROWS} on a width >= {COPY_VROWS} scratch", MTP_MAX_K + 1);
        let h = cfg.hidden_size;
        let hc = cfg.hc_count.max(1);
        let rw = h * hc;
        let lr = cfg.hc_lowrank;
        let v = cfg.vocab_size;
        let emax = w * cfg.num_experts_per_tok;
        // EXL27B: a dense pack (MoeLayer::dense) — its widths for the buffers the FN constants below
        // cannot hold (27B: h 5120, 4 KV heads x 256, MLP 17408); every other model: the FN constants.
        let dense = cfg.num_experts == 1;
        let (kvw, hh, mlpw) = (cfg.num_kv_heads * cfg.head_dim, h, cfg.moe_intermediate_size);
        Ok(Self {
            resid: dev.alloc_zeros::<f32>(w * rw)?,
            hn: dev.alloc_zeros::<u16>(w * rw)?,
            dd: dev.alloc_zeros::<u16>(w * lr)?,
            uu: dev.alloc_zeros::<u16>(w * rw)?,
            // EXL27B: plain residual — the inject gate must be 1.0 (the inject multiplies the layer output by it)
            inj: if lr == 0 { dev.htod_sync_copy(&vec![1.0f32; w * hc])? } else { dev.alloc_zeros::<f32>(w * hc)? },
            x: dev.alloc_zeros::<u16>(w * h)?,
            // S-A3-f-b FIX: must hold w * max_chain_K (out_proj K=6144; the old
            // w*2560 made every K>2560 chain read OOB for rows >= w*2560/K —
            // the "corrupt band [213,426)" — adjacent-buffer garbage).
            chain_xh: dev.alloc_zeros::<u16>(w * if dense { 12288usize.max(hh).max(mlpw) } else { 12288 })?,
            chain_yraw: dev.alloc_zeros::<u16>(w * v)?,
            qkv: dev.alloc_zeros::<u16>(w * 10240)?,
            qg: dev.alloc_zeros::<u16>(w * 12288)?,
            kp: dev.alloc_zeros::<u16>(w * if dense { kvw } else { 512 })?,
            vp: dev.alloc_zeros::<u16>(w * if dense { kvw } else { 512 })?,
            zbuf: dev.alloc_zeros::<u16>(w * 6144)?,
            a_out: dev.alloc_zeros::<u16>(w * 48)?,
            b_out: dev.alloc_zeros::<u16>(w * 48)?,
            core: dev.alloc_zeros::<u16>(w * 48 * 128)?,
            normed: dev.alloc_zeros::<u16>(w * 6144)?,
            mixer_out: dev.alloc_zeros::<u16>(w * h)?,
            logits_r: dev.alloc_zeros::<f32>(w * cfg.num_experts)?,
            router_ws: dev.alloc_zeros::<f32>(128 * w * cfg.num_experts)?,
            route_cnt: dev.htod_sync_copy(&[0i32])?,   // explicit zero (alloc_zeros does not zero, §2.2)
            fold_cnt: dev.htod_sync_copy(&[0i32])?,    // WP10: explicit zero (§2.2)
            ids: dev.alloc_zeros::<i32>(w * cfg.num_experts_per_tok)?,
            wts: dev.alloc_zeros::<f32>(w * cfg.num_experts_per_tok)?,
            slotmap: dev.alloc_zeros::<i32>(cfg.num_experts)?,
            offs_gu: dev.alloc_zeros::<u64>(emax)?,
            offs_d: dev.alloc_zeros::<u64>(emax)?,
            idxmap: dev.alloc_zeros::<i32>(emax)?,
            esel: dev.alloc_zeros::<i32>(1)?,
            xh_e: dev.alloc_zeros::<u16>(emax * w * 2560)?,
            ygu_raw: dev.alloc_zeros::<u16>(emax * w * 1280)?,
            ygu: dev.alloc_zeros::<u16>(emax * w * 1280)?,
            din: dev.alloc_zeros::<u16>(emax * w * 640)?,
            xhd: dev.alloc_zeros::<u16>(emax * w * 640)?,
            yd_raw: dev.alloc_zeros::<u16>(emax * w * 2560)?,
            yd: dev.alloc_zeros::<u16>(emax * w * 2560)?,
            sh1: dev.alloc_zeros::<u16>(w * if dense { mlpw } else { 640 })?,
            sh2: dev.alloc_zeros::<u16>(w * if dense { mlpw } else { 640 })?,
            ysh_raw: dev.alloc_zeros::<u16>(w * if dense { hh } else { 2560 })?,
            ysh: dev.alloc_zeros::<u16>(w * if dense { hh } else { 2560 })?,
            moe_out: dev.alloc_zeros::<u16>(w * h)?,
            cos: dev.alloc_zeros::<f32>(w * 2 * rdim)?,
            sin: dev.alloc_zeros::<f32>(w * rdim)?,
            toks: dev.alloc_zeros::<i32>(w)?,
            pos: dev.alloc_zeros::<i32>(w)?,
            slots: dev.alloc_zeros::<i32>(w * 2)?, // slotpos: [slot,pos] per lane
            // S-A3-i QSA pipes (0-sized when the pack has no indexer)
            qsa_qk: dev.alloc_zeros::<u16>(w * if cfg.has_indexer() { (cfg.indexer_n_heads + 1) * cfg.indexer_head_dim } else { 0 })?,
            qsa_qkb: dev.alloc_zeros::<u16>(w * if cfg.has_indexer() { (cfg.indexer_n_heads + 1) * cfg.indexer_head_dim } else { 0 })?,
            qsa_scores: dev.alloc_zeros::<f32>(qsa_scores_len(cfg.has_indexer(), cfg.indexer_compress_ratio, w, max_pos))?,
            qsa_sel: dev.alloc_zeros::<i32>(w * if cfg.has_indexer() { cfg.indexer_budget + cfg.indexer_compress_ratio } else { 0 })?,
            qsa_psel: dev.alloc_zeros::<i32>(w)?,
            slot_ids: dev.alloc_zeros::<i32>(w)?,
            qstage: dev.alloc_zeros::<f32>(w * cfg.num_heads * cfg.head_dim)?,
            qsa_pm: dev.alloc_zeros::<f32>(w * cfg.num_heads * QSA_SPLITS_DEC)?,
            qsa_pl: dev.alloc_zeros::<f32>(w * cfg.num_heads * QSA_SPLITS_DEC)?,
            qsa_pacc: dev.alloc_zeros::<f32>(w * cfg.num_heads * QSA_SPLITS_DEC * cfg.head_dim)?,
            logits: dev.alloc_zeros::<u16>(w * v)?,
            argmax: dev.alloc_zeros::<i32>(w)?,
            // WP09: keys are fully written before read; the counters need a REAL zero (§2.2).
            am_part: dev.alloc_zeros::<u64>(w.max(1) * XQ_AM_NB_MAX)?,
            am_cnt: dev.htod_sync_copy(&vec![0u32; w.max(1)])?,
            vp_keys: dev.htod_sync_copy(&vec![0u64; 16])?, // TP-H #4: explicit zero (padding rows are never written)
            vp_rows: dev.alloc_zeros::<u16>((w * xtp::vp_rows_cols()).max(1))?, // TP-H2: fully written before read
            ple_emb: dev.alloc_zeros::<u16>(w * 2560)?,
            ple_key16: dev.alloc_zeros::<u16>(w * 10240)?,
            ple_val: dev.alloc_zeros::<u16>(w * 2560)?,
            ple_kn: dev.alloc_zeros::<f32>(w * 10240)?,
            ple_qn: dev.alloc_zeros::<f32>(w * 10240)?,
            ple_gated: dev.alloc_zeros::<u16>(w * 10240)?,
            ple_normed: dev.alloc_zeros::<u16>(w * 10240)?,
            // K4 (REL v0.7.3 review): the PACKED staging (stage_packed / xq_accept_seg) uses
            // row indices to R_pad - 1 (up to 15); 10 ints truncated the staged drafts above
            // R_pad 10 and made the pack3-family identity compare partly vacuous there.
            d_dev: dev.alloc_zeros::<i32>(accept_out_words_for(vrows))?,
            // S-A3-x: xq_accept writes out[0..=k+2] — size from MTP_MAX_K (was a fixed 8: an
            // OOB device write at k >= 6 that no K<=5 path could reach). K=5 kept 8; WP23 K=7 -> 10;
            // the packed tail (no header slot) runs to R_pad = 16 rows.
            accept_out: dev.alloc_zeros::<i32>(accept_out_words_for(vrows))?,
            acc2: dev.alloc_zeros::<i32>(2)?,
            seg: dev.alloc_zeros::<i32>(PACK_SEG_MAX * 4)?,
            acc_seg: dev.alloc_zeros::<i32>(PACK_SEG_MAX * 2)?,
            gdn_ring_p: dev.alloc_zeros::<f32>(
                cfg.layer_types.iter().filter(|t| matches!(t, crate::qwen::LayerType::LinearAttention)).count()
                    * 16 * cfg.lin_num_v_heads * GDN_RS,
            )?,
            // WP03: per-slot windows [w][MTP_MAX_K+1][hc*h] (was ONE w*hc*h window shared by every
            // lane: with >= 2 decoding lanes each round drafted from another lane's hidden).
            // (--a-taps=shared keeps the old size: the old seam read this buffer's uninitialized
            // contents for round 1, so the A/B arm keeps the old allocation sequence.)
            taps_keep: dev.alloc_zeros::<f32>(
                if a_flags().taps_shared { w * hc * h } else { w * vrows * hc * h })?,
            // Verify shadows: one slot's GDN/conv state per layer + the PLE ring
            // (~112 MB at this model's 36 GDN layers; the live states stay
            // untouched until the device-keyed commit replays rows 0..a).
            s_state_sh: (0..cfg.num_layers).map(|_| {                dev.alloc_zeros::<f32>(cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim)
            }).collect::<Result<_, _>>()?,
            conv_sh: (0..cfg.num_layers).map(|_| {
                dev.alloc_zeros::<f32>((cfg.key_dim() * 2 + cfg.value_dim()) * cfg.conv_kernel)
            }).collect::<Result<_, _>>()?,
            // PACK1 LEG 1: x PACK_SEG_MAX planes — segment j's PLE shadow lives at plane j.
            ple_ring_sh: dev.alloc_zeros::<u16>(10240 * 9 * PACK_SEG_MAX)?,
            // Raw pre-conv qkv rows per GDN layer for the commit replay:
            // n_gdn * SAVE_PLANE_ROWS * conv_dim u16 (PACK2/D1: the save planes are 16 rows).
            qkv_save: dev.alloc_zeros::<u16>(
                cfg.layer_types.iter().filter(|t| matches!(t, crate::qwen::LayerType::LinearAttention)).count()
                    * SAVE_PLANE_ROWS
                    * (cfg.key_dim() * 2 + cfg.value_dim()),
            )?,
            qkv_post: dev.alloc_zeros::<u16>(
                cfg.layer_types.iter().filter(|t| matches!(t, crate::qwen::LayerType::LinearAttention)).count()
                    * SAVE_PLANE_ROWS
                    * (cfg.key_dim() * 2 + cfg.value_dim()),
            )?,
            a_save: dev.alloc_zeros::<u16>(
                cfg.layer_types.iter().filter(|t| matches!(t, crate::qwen::LayerType::LinearAttention)).count()
                    * SAVE_PLANE_ROWS * 48,
            )?,
            b_save: dev.alloc_zeros::<u16>(
                cfg.layer_types.iter().filter(|t| matches!(t, crate::qwen::LayerType::LinearAttention)).count()
                    * SAVE_PLANE_ROWS * 48,
            )?,
            gdn_ring: dev.alloc_zeros::<f32>(
                cfg.layer_types.iter().filter(|t| matches!(t, crate::qwen::LayerType::LinearAttention)).count()
                    * vrows * cfg.lin_num_v_heads * GDN_RS,
            )?,
            draft_meta: dev.htod_sync_copy(&[0i32; 4])?,
            rp: match RpRt::new(dev, rp_accept_words(vrows)) {
                Ok(r) => Some(r),
                Err(e) => { crate::dprintln!("REPRIME: device re-prime runtime unavailable ({e:#}) — eager re-prime"); None }
            },
            // a REAL zero (alloc_zeros does not zero, AGENTS §2.2): zero rows = greedy.
            samp: dev.htod_sync_copy(&vec![0u32; w.max(1) * 8])?,
            samp_live: false,
            // real zeros (AGENTS §2.2): zero rows = unpenalized
            pen: dev.htod_sync_copy(&vec![0u32; w.max(1) * 8])?,
            pen_live: false,
            pen_mono: true,
            vmask: dev.htod_sync_copy(&vec![0u32; w.max(1) * vmask_words(cfg.vocab_size)])?,
            vmask_flags: dev.htod_sync_copy(&vec![0i32; w.max(1)])?,
            vmask_live: false,
            lp_live: false,
            pen_hist: dev.htod_sync_copy(&vec![0i32; w.max(1) * PEN_RING])?,
            pen_draft_i: None,
            dconf: dev.htod_sync_copy(&vec![0f32; vrows])?, // WIDEVERIFY1: >= the verify width (wp05_stage_fits)
            qsa_grid_nblk: 0,
            dpm: dev.alloc_zeros::<f32>(w * cfg.num_heads * DRAFT_ATTN_SPLITS)?,
            dpl: dev.alloc_zeros::<f32>(w * cfg.num_heads * DRAFT_ATTN_SPLITS)?,
            dpacc: dev.alloc_zeros::<f32>(w * cfg.num_heads * DRAFT_ATTN_SPLITS * cfg.head_dim)?,
            gdn_chk: dev.alloc_zeros::<f32>(if crate::opts::var(crate::opt!("exl3-gdn-ring-check")).is_ok() {
                w * cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim } else { 1 })?,
            hcx: dev.alloc_zeros::<u16>(w * hc * h)?,
            hcy: dev.alloc_zeros::<u16>(w * hc * h)?,
            hf: dev.alloc_zeros::<u16>(w * hc * h)?,
            he32: dev.alloc_zeros::<f32>(w * hc * h)?,
            // alloc_zeros does NOT zero (AGENTS §2.2) — the barrier counter gets a REAL
            // zero via an htod copy. Scratch::new runs outside graph capture (the htod
            // sync is illegal inside one); replays reuse the buffer, which the kernel
            // self-resets to 0 every launch.
            // [count, generation]: the trunk's barriers use word 0; FP4L's xf_hc_fuse_b32 also uses word 1
            hc_bar: dev.htod_sync_copy(&[0i32; 2])?,
            // 2M floats (8 MB): worst chain needs ks*m*n <= ~400K (asserted at launch).
            chain_ws: dev.alloc_zeros::<f32>(2_000_000)?,
            pair_xh: dev.alloc_zeros::<u16>(w * if dense { hh } else { 2560 })?,
            pair_yraw: dev.alloc_zeros::<u16>(w * if dense { kvw.max(mlpw).max(2560) } else { 2560 })?,
            pair_ws: dev.alloc_zeros::<f32>(2_000_000)?,
            wp05: match Wp05Stage::new(dev, w) {
                Ok(s) => Some(s),
                Err(e) => { crate::dprintln!("WP05: pinned decode staging unavailable ({e:#}) — pageable path"); None }
            },
            // a REAL zero (alloc_zeros does not zero, AGENTS §2.2); outside any capture.
            moe_cnt: dev.htod_sync_copy(&vec![0u32; emax * (cfg.moe_intermediate_size / 128).max(1) + h / 128 + 1])?,
            moe_sgv: dev.alloc_zeros::<f32>(w.max(1))?,
            // WP13: real zeros (htod). Every USED read is of a slot written earlier in the same
            // launch sequence; the zeros only keep the acc kernel's last-tile prefetch past
            // p (loaded, never used) initcheck-clean.
            dv3_s: dev.htod_sync_copy(&vec![0f32; w * cfg.num_heads * DV3_TS])?,
            dv3_pm: dev.htod_sync_copy(&vec![0f32; w * cfg.num_heads * DV3_NCH])?,
            dv3_xc: dev.alloc_zeros::<u16>(if dense_xcheck_on() { w * cfg.num_heads * cfg.head_dim } else { 1 })?,
            dv3_xst: dev.htod_sync_copy(&[0i32; 4])?,
            // WP27: fully written before read (had_suh -> xh, split-K partials -> ws, combine -> yraw).
            z_xh: dev.alloc_zeros::<u16>(w * h)?,
            z_yraw: dev.alloc_zeros::<u16>(w * cfg.value_dim())?,
            z_ws: dev.alloc_zeros::<f32>(
                (chain_ks(h as i32, cfg.value_dim() as i32).max(xtp::tp_ks_cap()) as usize * w * cfg.value_dim()).max(1))?,
            wp27_cls: 0,
            gid: SCRATCH_IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            vrows,
            // WP24: real -1 ids / a real zero counter (AGENTS §2.2 — alloc_zeros does not zero);
            // rq_part is fully written by stage 1 before the last block reads it.
            // PACK4S: one list row per verify ROW a real-q sampler can read — a packed verify
            // reads list row row0_j + i (up to row 14 of 16), so the list follows the scratch
            // width (lane_pack::rq_list_rows: MTP_MAX_K + 1 rows, as before, up to 8 rows).
            rq_list: dev.htod_sync_copy(&vec![u32::MAX; crate::lane_pack::rq_list_rows(w) * crate::wp24::RQ_ROW])?,
            rq_part: dev.alloc_zeros::<u64>(crate::wp24::RQ_NB_MAX * crate::wp24::RQ_K)?,
            rq_cnt: dev.htod_sync_copy(&[0u32])?,
            rq_dsamp: dev.htod_sync_copy(&[0u32; 8])?,
            rq_dsamp_host: [0u32; 8],
            rq_mode: 0,
            rq_live: false,
            rq_dctr: 0,
            rq_draft_i: None,
            fxs: if fp4l_fwd::fx_scratch_on() { Some(fp4l_fwd::FxScratch::new(dev, cfg, w)?) } else { None },
        })
    }

    /// WP27: the in_z chain `q` at width m fits the duplicated z scratch (else it keeps chain_*).
    pub(super) fn wp27_z_fits(&self, q: &Quad, m: usize) -> bool {
        let (k, n) = (q.k.max(0) as usize, q.n.max(0) as usize);
        m <= 16 && m * k <= self.z_xh.len() && m * n <= self.z_yraw.len()
            && (q.ks <= 1 || q.ks as usize * m * n <= self.z_ws.len())
    }
}

impl FwdModel {
    /// NIGHT1 (forum report by jetspark 2026-11-14: `--gpu-wait` did nothing on EXL3 / FP4-Local): every host wait
    /// of this model first waits for the COMPUTE stream through `host_wait` (`--gpu-wait`: hybrid sleeps until just
    /// before the call site's predicted end and finishes by polling; spin / block as before), THEN keeps the original
    /// NULL-stream synchronize — a strict superset of the old wait (the legacy NULL stream already waits for this
    /// blocking stream), so ordering and values are unchanged; the driver call that follows only spins for whatever
    /// NULL-stream work (a copy) is still queued. `#[track_caller]`: each call site keeps its own prediction.
    #[track_caller]
    pub(crate) fn dsync(&self) -> std::result::Result<(), cudarc::driver::DriverError> {
        crate::host_wait::stream_wait(self.stream.stream)?;
        self.dev.synchronize()
    }
    /// NIGHT1: `dev.dtoh_sync_copy` preceded by the compute-stream wait of [`Self::dsync`].
    #[track_caller]
    pub(crate) fn dtoh<T: cudarc::driver::DeviceRepr>(&self, src: &CudaSlice<T>)
        -> std::result::Result<Vec<T>, cudarc::driver::DriverError> {
        crate::host_wait::stream_wait(self.stream.stream)?;
        self.dev.dtoh_sync_copy(src)
    }
    /// NIGHT1: `dev.dtoh_sync_copy_into` preceded by the compute-stream wait of [`Self::dsync`].
    #[track_caller]
    pub(crate) fn dtoh_into<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(&self, src: &Src, dst: &mut [T])
        -> std::result::Result<(), cudarc::driver::DriverError> {
        crate::host_wait::stream_wait(self.stream.stream)?;
        self.dev.dtoh_sync_copy_into(src, dst)
    }
}
