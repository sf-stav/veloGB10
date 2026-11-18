//! MTP head re-prime: the RpMode / RpRt runtime types and the `impl FwdModel` head_reprime*, reprime_dev_*
//! and rp_* methods (host and device-driven re-prime, its graph capture and cross-check).
//! The MTP head's KV is SINGLE-SLOT (AGENTS 2.5): its write slot is always 0. See `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------
// REPRIME (w2): SURPASS WP22 rung 1 + the fixed-width, graphed head re-prime.
// ---------------------------------------------------------------------------

/// REPRIME escapes (diagnostics / A-B / bisect only — AGENTS §1b; the device path is the default):
/// - `--exl3-reprime=0 | full`: unchanged (no re-prime | the eager full-head reference pass);
/// - `--exl3-reprime-eager=1`: the eager a-row re-prime after the accept readback (host
///   uploads + ~45 launches), WP22 rung 1 still on — the XCHECK reference path;
/// - `--wp22r1-off=1`: rung 1 off AND the eager path = byte-for-byte the pre-REPRIME behaviour
///   (a*hc > 16 fc rows back on the non-split wide kernel). The device path REQUIRES rung 1: its
///   fixed MTP_MAX_K*hc fc rows would otherwise take the wide kernel while the eager a <= 4 rows
///   took split-K — different bits for the same row;
/// - `--exl3-reprime-graph=0`: the device path launched eagerly every round (no capture).
#[derive(Clone, Copy, PartialEq, Debug)]
enum RpMode { Off, Full, Eager, Dev }

fn rp_mode() -> RpMode {
    static M: std::sync::OnceLock<RpMode> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let m = match crate::opts::var(crate::opt!("exl3-reprime")).as_deref() {
            Ok("0") => RpMode::Off,
            Ok("full") => RpMode::Full,
            _ if crate::opts::var(crate::opt!("exl3-reprime-eager")).as_deref() == Ok("1") || wp22r1_off() => RpMode::Eager,
            _ => RpMode::Dev,
        };
        if m != RpMode::Dev {
            crate::dprintln!("REPRIME: mode {m:?} (--exl3-reprime / --exl3-reprime-eager / --wp22r1-off — diagnostic escape)");
        }
        m
    })
}

/// WP22 rung 1 escape: --wp22r1-off=1 = the head fc chains through exl3_chain at a*hc rows.
fn wp22r1_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| crate::opts::var(crate::opt!("wp22r1-off")).as_deref() == Ok("1"))
}

/// WP22 rung 1: the (first row, rows) split-K slabs of an m-row chain — consecutive, <= 16 rows
/// each (the m16 tile of exl3_gemm_body_ks), covering [0, m) exactly.
pub(super) fn wp22r1_slabs(m: usize) -> Vec<(usize, usize)> {
    (0..m).step_by(16).map(|r0| (r0, (m - r0).min(16))).collect()
}

/// --reprime-xcheck=1: eager cross-check of every device re-prime (syncs twice per round):
/// head K/V rows, indexer-key rows and pooled blocks over positions [p, p+R] are read before the
/// device re-prime (S0), after it (S1), and after the eager a-row re-prime re-runs on top (S2).
/// Accepted rows p+1..p+a must satisfy S1 == S2 (bitwise); every other row S1 == S0 (masked
/// rows write nothing); a pooled block holding an accepted row S1 == S2, every other S1 == S0.
fn rp_xcheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("reprime-xcheck")).map_or(false, |v| !v.is_empty() && v != "0"))
}

/// Capture the device re-prime into a graph? (the draft chain's conditions + the eager-only
/// diagnostics whose host syncs cannot be recorded)
fn rp_graph_want() -> bool {
    crate::opts::var(crate::opt!("exl3-no-graph")).is_err()
        && tune::get_cur(&T_REPRIME_GRAPH) != 0
        && crate::opts::var(crate::opt!("mtp-dump")).map_or(true, |s| s.is_empty())
        && crate::opts::var(crate::opt!("hc-int8-check")).is_err()
        && crate::opts::var(crate::opt!("exl3-chain-pair-check")).is_err()
        && !wp12_xcheck()
}

/// CUevent handle (only used from the scheduler thread; same contract as CudaGraph).
struct RpEvent(cudarc::driver::sys::CUevent);
unsafe impl Send for RpEvent {}
unsafe impl Sync for RpEvent {}

/// This round's device re-prime, enqueued by mtp_verify, consumed by head_reprime.
#[derive(Clone, Copy, Debug)]
pub(super) struct RpPending {
    slot: usize,
    p: usize,
    rows: usize,
    key: usize,
    /// launched eagerly this round with graphs wanted: capture after the readback
    capture: bool,
}

/// --reprime-xcheck snapshot of the head state over positions [lo, hi] of one slot.
struct RpSnap {
    lo: usize,
    hi: usize,
    j0: usize,
    kv: Vec<u8>,    // [2*nkv][hi-lo+1][rowbytes]
    keys: Vec<u16>, // [hi-lo+1][hdx] (empty without a head key plane)
    pool: Vec<u16>, // [j1-j0][hdx] (empty without a pooled plane)
}

/// REPRIME: per-Scratch runtime of the device re-prime.
pub(crate) struct RpRt {
    /// {slot, pos0 = p+1, n = a, 0}, written on device by xq_reprime_stage
    meta: CudaSlice<i32>,
    /// accept_out words, copied (async, compute stream) right behind the verify
    pin: PinnedHost,
    /// recorded right after that copy: the host waits for it, not for the re-prime
    ev: RpEvent,
    pub(super) pending: Option<RpPending>,
    snap0: Option<RpSnap>,
}

impl RpRt {
    pub(super) fn new(dev: &Arc<CudaDevice>, words: usize) -> Result<Self> {
        use cudarc::driver::sys;
        let pin = PinnedHost::new(dev, words * 4)?; // binds the context
        let mut ev: sys::CUevent = std::ptr::null_mut();
        // CU_EVENT_DISABLE_TIMING (2): a pure completion marker
        let r = unsafe { sys::cuEventCreate(&mut ev, 2) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS && !ev.is_null(), "cuEventCreate ({r:?})");
        Ok(Self { meta: dev.htod_sync_copy(&[0i32; 4])?, pin, ev: RpEvent(ev), pending: None, snap0: None })
    }
}

impl Drop for RpRt {
    fn drop(&mut self) {
        let _ = self.pin.dev.bind_to_thread();
        unsafe { cudarc::driver::sys::cuEventDestroy_v2(self.ev.0); }
    }
}




impl FwdModel {
    /// One MTP round for one lane. `b` = the lane's bonus token (last emitted,
    /// committed-but-unconsumed, entering at position `p`); `tap_row` selects the
    /// saved verify tap that PREDICTED b (the "tap-before-token" pairing — the
    /// tap after consuming b only exists after this round's verify row 0).
    ///
    /// Host syncs per round (iteration 1): ONE dtoh of the drafts (feeds the
    /// host-side PLE n-gram hash + the verify toks upload) and ONE dtoh of
    /// {emitted, a}. The reference implementation's single-readback structure needs the PLE hash
    /// on device — queued follow-up (recorded deviation).
    /// Head re-prime (the reference implementation's generator.py:1343-1377): the head's KV / indexer-key entries
    /// for the ACCEPTED draft positions p+1..p+a were written by the draft chain from the head's
    /// OWN chained hidden. Rewrite them from the TARGET's verify taps (row i = the trunk stream
    /// stack after the token at p+i, paired with the accepted draft d_i at p+1+i), so the head's
    /// context is built on real target states — exactly what the prompt prime does. Position p
    /// was already drafted from a target tap; p+a+1 (the bonus) is the next round's pass 0.
    /// Draft-only state: the head never feeds the trunk, so emitted tokens are unchanged
    /// (greedy stays bitwise); only acceptance moves. --exl3-reprime=0 = off (A/B).
    pub(super) fn head_reprime(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, p: usize,
                    accepted: &[i32]) -> Result<()> {
        // REPRIME (w2): mtp_verify already enqueued this round's fixed-width device re-prime right
        // behind the verify; finish it (first-round graph capture, --reprime-xcheck).
        if let Some(pend) = sc.rp.as_mut().and_then(|r| r.pending.take()) {
            return self.reprime_dev_finish(sc, head, slot, p, accepted, pend);
        }
        let a = accepted.len();
        if a == 0 || crate::opts::var(crate::opt!("exl3-reprime")).as_deref() == Ok("0") {
            return Ok(());
        }
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let n = (he * a) as i64;
        let tb = self.tap_base(sc, slot)?; // WP03: this lane's window
        xqlaunch!(l, "xq_copy_f32", ((((n + 255) / 256).max(1)) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.resid, &sc.taps_keep, n, 0i64, tb as i64))?;
        if crate::opts::var(crate::opt!("exl3-reprime")).as_deref() == Ok("full") {
            // reference arm: the whole head pass (attention, MoE, mixer) — same cache writes
            let pos: Vec<usize> = (p + 1..=p + a).collect();
            self.head_forward(sc, head, accepted, &pos, &vec![slot; a], false, false, false)?;
            return Ok(());
        }
        self.head_kv_rows(sc, head, slot, p + 1, accepted)
    }

    /// KV-only head write for rows toks[i] at contiguous positions pos0+i of one slot, from the
    /// taps already in sc.resid rows 0..m: stage (toks, positions, slot tables), then
    /// `head_reprime_launches`. Shared by the re-prime, the WP03 seam prime and the control round.
    pub(super) fn head_kv_rows(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, pos0: usize,
                    toks: &[i32]) -> Result<()> {
        let a = toks.len();
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let w = sc.toks.len();
        anyhow::ensure!(a >= 1 && a <= w, "head_kv_rows: {a} rows on a width-{w} scratch");
        let mut toks_v = toks.to_vec();
        toks_v.resize(w, 0);
        self.dev.htod_copy_into(toks_v, &mut sc.toks)?;
        let mut pos_v: Vec<i32> = (pos0..pos0 + a).map(|x| x as i32).collect();
        pos_v.resize(w, 0);
        self.dev.htod_copy_into(pos_v, &mut sc.pos)?;
        let mut slotpos: Vec<i32> = vec![0; w * 2];
        for i in 0..a {
            slotpos[i * 2] = slot as i32;
            slotpos[i * 2 + 1] = (pos0 + i) as i32;
        }
        self.dev.htod_copy_into(slotpos, &mut sc.slots)?;
        self.dev.htod_copy_into(vec![slot as i32; sc.slot_ids.len()], &mut sc.slot_ids)?;
        self.head_reprime_launches(&l, sc, head, a, slot, pos0)
    }

    /// The KV-only half of the re-prime: the head's cache entry for a row depends only on its
    /// input stage + the attention hyper-connection + K/V (and indexer-key) projections, so the
    /// attention itself, o_proj, the MoE, the mixer and the logits are skipped. Rows are the
    /// contiguous positions pos0..pos0+m of one slot (xq_attn_kv_prefill's contract).
    fn head_reprime_launches(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, m: usize,
                             slot: usize, pos0: usize) -> Result<()> {
        self.head_reprime_launches_inner(l, sc, head, m, slot, pos0, None)
    }

    /// S-C4PRIMEA: `pairs = Some(ptr)` switches the two slot-addressed points to per-row
    /// addressing (cos_gather via SlotSrc::Pairs, final KV write via xq_attn_kv_prefill_pairs).
    /// `pairs = None` is EXACTLY the legacy sequence (same kernels, args, order) — the control
    /// site and every existing caller pass None.
    pub(super) fn head_reprime_launches_inner(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, m: usize,
                                   slot: usize, pos0: usize, pairs: Option<u64>) -> Result<()> {
        let _tc = tune::enter(tune::Ctx::fam(tune::Fam::Reprime, m)); // TUNE family scope
        let cfg = &self.cfg;
        // WP02: rows pos0..pos0+m must lie inside the head KV / key-plane / cos_tab windows.
        // (Per-row callers check their own p < max_pos; the contiguous window check is a
        // legacy-path statement about pos0..pos0+m of ONE slot.)
        if pairs.is_none() {
            anyhow::ensure!(pos0 + m <= self.max_pos,
                            "head re-prime rows {pos0}..{} past the KV window (max_pos {})", pos0 + m, self.max_pos);
        }
        let h = cfg.hidden_size;
        let hcn = cfg.hc_count.max(1);
        let rdim = cfg.rotary_dim;
        let cs_stride = (2 * rdim) as i32;
        { let _ = cs_stride; // VIS-2: the mrope-aware gather (text-only slots: cos_tab[pos], unchanged)
        let slots_src = match pairs { None => SlotSrc::Const(slot), Some(p) => SlotSrc::Pairs(p) };
        self.cos_gather(&l, &mut sc.cos, *sc.pos.device_ptr() as u64, slots_src, m)?; }
        // entry — head_launches' exact sequence
        self.mtp_pre_norm(l, head, 0, *sc.hn.device_ptr() as u64, *sc.resid.device_ptr() as u64, m)?;
        Self::wp22r1_chain(l, &head.fc_hidden, &sc.hn, &mut sc.hcx, &mut sc.hcy, &mut sc.hf, m * hcn, &sc.chain_ws)?; // WP22 rung 1
        self.embed_gather(l, *sc.he32.device_ptr() as u64, *sc.toks.device_ptr() as u64, m)?;
        self.mtp_pre_norm(l, head, 1, *sc.hn.device_ptr() as u64, *sc.he32.device_ptr() as u64, m)?;
        Self::wp22r1_chain(l, &head.fc_embed, &sc.hn, &mut sc.hcx, &mut sc.hcy, &mut sc.yd, m * hcn, &sc.chain_ws)?; // WP22 rung 1
        xqlaunch!(l, "xq_add_f16_f32", (((h * hcn * m) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                 (&mut sc.resid, &sc.hf, &sc.yd, ((h * hcn * m) as i64)))?;
        self.hc_pre(l, sc, &head.hc_attn, m)?;
        let a = &head.attn;
        // k|v (+ the indexer qk) — W4/DENSE: one launch; else the old pair
        let idx_pre = self.attn_in_chains(l, sc, a, m, false)?;
        // indexer keys + pooled-plane refresh (spec state on every forward, S-A3-r rule)
        self.qsa_keys_append_dense(l, sc, a, HEAD_ATT, m, idx_pre)?;
        match pairs {
            None => {
                let slot_pos0: i32 = (slot as i32) | ((pos0 as i32) << 12);
                xqlaunch!(l, "xq_attn_kv_prefill", (cfg.num_kv_heads as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                         (&mut sc.kp, &mut sc.vp, (*head.kv.device_ptr()) as u64,
                          &a.qnkn, &sc.cos, (cfg.num_kv_heads as i32) | ((cfg.head_dim as i32) << 16),
                          (rdim as i32) | ((self.max_pos as i32) << 8) | ((self.kv_fmt as i32) << 28),
                          slot_pos0, m as i32, cfg.rms_eps))?;
            }
            Some(pp) => {
                xqlaunch!(l, "xq_attn_kv_prefill_pairs", ((m * cfg.num_kv_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                         (&mut sc.kp, &mut sc.vp, (*head.kv.device_ptr()) as u64,
                          &a.qnkn, &sc.cos, pp,
                          (cfg.num_kv_heads as i32) | ((cfg.head_dim as i32) << 16),
                          (rdim as i32) | ((self.max_pos as i32) << 8) | ((self.kv_fmt as i32) << 28),
                          m as i32, cfg.rms_eps))?;
            }
        }
        Ok(())
    }

    /// WP22 rung 1 (SURPASS_PLAN WP22): the re-prime's fc_hidden / fc_embed run at a*hc rows. Up
    /// to 16 rows they take the split-K chain (ks slabs, ascending combine); above 16 (a = 5 at
    /// hc = 4) exl3_chain fell to the non-split wide kernel (20 CTAs, ks = 1 reduction order), so
    /// the head-KV bits of a row depended on a — and the wide kernel was the slow one. Here > 16
    /// rows run as 16-row slabs of the SAME split-K body and combine: row r of a slab is computed
    /// from its own A row over the same [kb0, kb1) K slices, and its partials are summed in the
    /// same ascending order, so every row is bit-identical to the <= 16-row chain whatever the
    /// slab or row count (the batch-invariance contract of the decode chains; the Hadamards are
    /// per 128-element row block). --wp22r1-off=1 = exl3_chain (the old dispatch).
    #[allow(clippy::too_many_arguments)]
    fn wp22r1_chain(l: &Launcher, q: &Quad, x: &CudaSlice<u16>, xh: &mut CudaSlice<u16>,
                    yraw: &mut CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize,
                    ws: &CudaSlice<f32>) -> Result<()> {
        if q.fx.is_some() || m <= 16 || q.ks <= 1 || wp22r1_off() {
            return exl3_chain(l, q, x, xh, yraw, y, m, ws); // FP4L: 16-row blocks of the small kernel
        }
        let (k, n) = (q.k as usize, q.n as usize);
        anyhow::ensure!(((q.k as u32) / 16) % q.ks == 0, "wp22r1: ks {} does not divide kb {}", q.ks, q.k / 16);
        anyhow::ensure!(q.ks as usize * 16 * n <= ws.len(), "wp22r1: chain_ws {} < ks*16*n {}", ws.len(), q.ks as usize * 16 * n);
        anyhow::ensure!(m * k <= x.len() && m * k <= xh.len() && m * n <= yraw.len() && m * n <= y.len(),
                        "wp22r1: {m} rows exceed the chain buffers");
        if dense::dense_on() {
            // W4/DENSE: each <= 16-row slab is one DENSE launch over ITS rows of x / y (every
            // Hadamard is per 128-element row block and every row's partials use the same K slices
            // and ascending combine => row-local, bitwise the slab loop below). The slabs share one
            // eligibility (same shape, widths <= 16), so the first slab decides for all of them.
            use dense::{dptr, DOut};
            let (x_p, xh_p, yr_p, y_p) = (dptr(x), dptr(xh), dptr(yraw), dptr(y));
            let mut taken = 0usize;
            for (r0, ms) in wp22r1_slabs(m) {
                let o = DOut { y: y_p + (r0 * n * 2) as u64, y_len: ms * n, yraw: yr_p + (r0 * n * 2) as u64,
                               yraw_len: ms * n, xh: xh_p + (r0 * k * 2) as u64, xh_len: ms * k };
                if dense::dense_chains(l, &[q], &[o], (x_p + (r0 * k * 2) as u64, ms * k), (0, 0), ms,
                                       &[(dptr(ws), ws.len())])? {
                    taken += 1;
                } else {
                    anyhow::ensure!(taken == 0, "wp22r1: W4/DENSE took {taken} slab(s) then refused slab rows {r0}+{ms}");
                    break;
                }
            }
            if taken > 0 { return Ok(()); }
        }
        xqlaunch!(l, "exl3_had_suh", ((m * (k / 128)) as u32, 1, 1), (32, 1, 1), 0, (x, &q.suh, &mut *xh, q.k))?;
        let (xh_p, yr_p) = (*xh.device_ptr() as u64, *yraw.device_ptr() as u64);
        // p3c (w3-PSK x REPRIME): each <= 16-row slab takes the persistent split-K grid exactly as
        // exl3_chain_impl does (bitwise == exl3_hmma_gemm_ks for every G and every M 1..16; G is
        // m-independent); p3 default OFF (--psk=1 opts in) / --psk-off=1 / no capacity =
        // the old (N/128, ks) grid.
        let g = psk_grid(l.dev, q.k, q.n, q.bits, q.ks, false);
        if g == 0 && wp21_on(q.ks) && wp21_cnt_ok(q) && wp21_fx_width_ok(q) {
            // WP21 x REPRIME: one fused launch per slab — the slab's ks GEMM + the last-arriving
            // CTA's ascending combine + H128 svh straight into ITS rows of y (every Hadamard is
            // per (row, 128-block), so the per-slab svh equals the old single svh over all m rows;
            // bitwise the loop below). The slabs reuse ws and q.cnt: stream-ordered, and every
            // launch leaves the counters at 0. PSK (opt-in) keeps the old tail below.
            let y_p = *y.device_ptr() as u64;
            wp21_occupancy_once(l);
            for (r0, ms) in wp22r1_slabs(m) {
                // MTPAGREE1 (public v0.7.6 TP=2 report): the fused kernel's switch is keyed on k2 (W3, 2026-10-18),
                // and this site — the MTP head's fc_hidden / fc_embed prime + re-prime, the one wp22r1_chain
                // caller — kept passing `bits`. For the 5-bit head quads (bits 5, k2 10) no case matched and the
                // kernel returned without computing: sc.hf / sc.yd kept whatever the buffer held, per rank, so the
                // replicated head's hidden and its re-primed KV rows drifted apart across TP ranks (confidences by
                // 1-2 fp16 ulps per round, sampled drafts diverging) while every trunk digest stayed identical.
                xqlaunch!(l, "exl3_hmma_gemm_ks_fx", ((q.n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
                         (&q.tr, xh_p + (r0 * k * 2) as u64, ws, &q.svh, y_p + (r0 * n * 2) as u64, &q.cnt,
                          ms as i32, q.k, q.n, q.k2))?;
            }
            if wp21_xcheck() { wp21_check_slabs(l, q, xh, yraw, y, m, ws)?; }
            return Ok(());
        }
        for (r0, ms) in wp22r1_slabs(m) {
            let xs = xh_p + (r0 * k * 2) as u64;
            if g > 0 {
                xqlaunch!(l, "exl3_hmma_gemm_ks_p", (g, 1, 1), (256, 1, 1), 0,
                         (&q.tr, xs, ws, ms as i32, q.k, q.n, psk_cfg(q.bits, q.ks), &q.tr))?;
                if psk_xcheck() { psk_xcheck_single(l, q, xs, ws, ms, g)?; }
            } else {
            xqlaunch!(l, "exl3_hmma_gemm_ks", ((q.n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
                     (&q.tr, xs, ws, ms as i32, q.k, q.n, q.k2))?;
            }
            let mn = (ms * n) as u32;
            xqlaunch!(l, "exl3_ks_combine", ((mn + 255) / 256, 1, 1), (256, 1, 1), 0,
                     (ws, yr_p + (r0 * n * 2) as u64, ms as i32, q.n, q.ks as i32))?;
        }
        xqlaunch!(l, "exl3_had_svh", ((m * (n / 128)) as u32, 1, 1), (32, 1, 1), 0, (&mut *yraw, &q.svh, &mut *y, q.n))?;
        Ok(())
    }

    /// REPRIME: rows of the fixed-width device re-prime = MTP_MAX_K (every accepted count fits:
    /// a <= k <= MTP_MAX_K; derived, so a K raise carries it), bounded by the scratch's verify
    /// width (k + 1 <= w).
    fn rp_rows(sc: &Scratch) -> usize {
        MTP_MAX_K.min(sc.toks.len().saturating_sub(1))
    }

    /// WIDEVERIFY1 refinement 2: the fixed re-prime width for a round of `k` drafts. Every round of
    /// k <= MTP_MAX_K (every neural round, every copy round of the default width) keeps `rp_rows`
    /// (the same graph key as before); a wide COPY round (k > MTP_MAX_K, only on a COPY_VROWS
    /// scratch) re-primes COPY_MAX_K rows so all of its accepted drafts reach the head (its own
    /// graph key: `rp_key` carries the rows).
    fn rp_rows_for(sc: &Scratch, k: usize) -> usize {
        rp_rows_of(sc.toks.len(), sc.vrows, k)
    }

    /// REPRIME: is the device re-prime live for this scratch?
    pub(super) fn reprime_dev_on(&self, sc: &Scratch) -> bool {
        rp_mode() == RpMode::Dev && self.mtp.is_some() && sc.rp.is_some() && Self::rp_rows(sc) >= 1
    }

    /// REPRIME graph key: one graph per (slot, rows) — every other input is device-resident.
    fn rp_key(slot: usize, rows: usize) -> usize {
        (6usize << 30) | (slot << 8) | rows
    }

    /// REPRIME: the head re-prime as launches only (graph-capturable), at a FIXED row count m with
    /// the accepted count a read on device. Mirrors head_reprime_launches row for row:
    /// xq_reprime_stage writes rows 0..a exactly as head_reprime's host staging (tap row r ->
    /// resid row r, token d_r, position p+1+r) and masked duplicates for rows a..m; every
    /// transform after it is row-local and batch-invariant (hc norms, the rung-1 fc chains, embed
    /// gather, float add, hc_pre, the K/V pair and indexer chains at <= 16 rows), so row r < a is
    /// bit-identical to row r of the eager a-row call. Persistent writes: K/V rows and key rows
    /// only for rows < a (device count); the pooled refresh of a masked row targets block(p+a),
    /// which row a-1 refreshes from the same raws (a >= 1) or which the chain's pass 0 left
    /// consistent with its raws this round (a = 0: identical bits). `tap_off` = the taps_keep
    /// element offset of the slot's verify taps = tap_base(sc, slot) (WP03's per-slot windows;
    /// 0 only under --a-taps=shared). A per-slot constant, so the graph key keeps the slot.
    fn head_reprime_fixed_launches(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, m: usize,
                                   tap_off: usize) -> Result<()> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let hcn = cfg.hc_count.max(1);
        let he = h * hcn;
        let rdim = cfg.rotary_dim;
        let cs_stride = (2 * rdim) as i32;
        let meta_p = *sc.rp.as_ref().context("REPRIME: runtime missing")?.meta.device_ptr() as u64;
        anyhow::ensure!(m >= 1 && m < sc.toks.len() && m <= sc.pos.len() && tap_off + m * he <= sc.taps_keep.len(),
                        "REPRIME: {m} rows do not fit the scratch (width {})", sc.toks.len());
        xqlaunch!(l, "xq_reprime_stage", ((((m * he) as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (&mut sc.resid, &mut sc.toks, &mut sc.pos, meta_p, &sc.taps_keep, &sc.draft_meta,
                  &sc.acc2, &sc.accept_out, m as i32, he as i64, tap_off as i64))?;
        { let _ = cs_stride; // VIS-2: the mrope-aware gather (text-only slots: cos_tab[pos], unchanged)
        let slots_src = SlotSrc::Ptr(meta_p);
        self.cos_gather(&l, &mut sc.cos, *sc.pos.device_ptr() as u64, slots_src, m)?; }
        self.mtp_pre_norm(l, head, 0, *sc.hn.device_ptr() as u64, *sc.resid.device_ptr() as u64, m)?;
        Self::wp22r1_chain(l, &head.fc_hidden, &sc.hn, &mut sc.hcx, &mut sc.hcy, &mut sc.hf, m * hcn, &sc.chain_ws)?;
        self.embed_gather(l, *sc.he32.device_ptr() as u64, *sc.toks.device_ptr() as u64, m)?;
        self.mtp_pre_norm(l, head, 1, *sc.hn.device_ptr() as u64, *sc.he32.device_ptr() as u64, m)?;
        Self::wp22r1_chain(l, &head.fc_embed, &sc.hn, &mut sc.hcx, &mut sc.hcy, &mut sc.yd, m * hcn, &sc.chain_ws)?;
        xqlaunch!(l, "xq_add_f16_f32", (((h * hcn * m) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                 (&mut sc.resid, &sc.hf, &sc.yd, ((h * hcn * m) as i64)))?;
        self.hc_pre(l, sc, &head.hc_attn, m)?;
        let a = &head.attn;
        // k|v (+ the indexer qk) — W4/DENSE: one launch; else the old pair
        let idx_pre = self.attn_in_chains(l, sc, a, m, false)?;
        self.qsa_keys_append_dense_masked(l, sc, a, HEAD_ATT, m, meta_p, idx_pre)?;
        xqlaunch!(l, "xq_attn_kv_rows_dn", (cfg.num_kv_heads as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                 (&mut sc.kp, &mut sc.vp, (*head.kv.device_ptr()) as u64,
                  &a.qnkn, &sc.cos, (cfg.num_kv_heads as i32) | ((cfg.head_dim as i32) << 16),
                  (rdim as i32) | ((self.max_pos as i32) << 8) | ((self.kv_fmt as i32) << 28),
                  meta_p, m as i32, cfg.rms_eps))?;
        Ok(())
    }

    /// REPRIME: qsa_keys_append_dense with the key write masked to rows < meta[2] (the device
    /// accepted count). The pooled refresh keeps qsa_pool_update_b over all m rows: a masked row's
    /// position is p+a (xq_reprime_stage), whose block is refreshed from the same raws by row a-1
    /// (a >= 1) or was left consistent with its raws by this round's draft pass 0 at p (a = 0) —
    /// either way the stored bits do not change.
    fn qsa_keys_append_dense_masked(&self, l: &Launcher, sc: &mut Scratch, a: &AttnLayer, att_i: usize,
                                    m: usize, meta_p: u64, idx_pre: bool) -> Result<()> {
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
        xqlaunch!(l, "xq_qsa_key_write_dn", ((((m * hdx) as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                  (keys_p, &sc.qsa_qkb, &sc.pos, &sc.slot_ids, self.max_pos as i32,
                   qk_dim as i32, (heads * hdx) as i32, hdx as i32, m as i32, meta_p))?;
        if let (Some(pool_p), Some(params)) = (self.qsa_pool_p(att_i), ix.params.as_ref()) {
            let nblk_stride = (self.max_pos / cfg.indexer_compress_ratio).max(1);
            let geom: u64 = crate::gpu::pack_qsa_geom(self.max_pos, 0, nblk_stride);
            gblaunch!(l, "qsa_pool_update_b", ((m.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0,
                      (pool_p, keys_p, &sc.pos, &sc.slot_ids, (*params.device_ptr()) as u64, geom as i64, m as i32))?;
        }
        Ok(())
    }

    /// REPRIME: mtp_verify's readback + the device re-prime. Compute-stream order: verify ->
    /// async copy of accept_out into pinned memory -> event -> the fixed-width re-prime (its graph;
    /// its launches on a key's first round). The host waits for the EVENT only, so the re-prime
    /// (was: ~45 eager launches + 4 pageable uploads issued after the readback) runs on the GPU
    /// while the host finishes this round and issues the next; the next round's first draft pass
    /// is stream-ordered behind it. Returns the accept_out words (the old dtoh_sync_copy).
    pub(super) fn reprime_dev_enqueue(&self, l: &Launcher, sc: &mut Scratch, slot: usize, p: usize, k: usize) -> Result<Vec<i32>> {
        use cudarc::driver::sys;
        let head = self.mtp.as_ref().context("REPRIME: no draft head")?;
        let rows = Self::rp_rows_for(sc, k);
        // NOT sc.accept_out.len(): that is 16+ words wide (K4) and the pin is RP_ACCEPT_WORDS (13) on a
        // default scratch; WIDEVERIFY1: rp_accept_words(vrows) = the pin RpRt::new was given (18 wide)
        let words = rp_accept_words(sc.vrows);
        anyhow::ensure!(k + 3 <= words, "REPRIME: a {k}-draft verify writes {} accept words > the {words}-word readback", k + 3);
        anyhow::ensure!(words <= sc.accept_out.len(), "REPRIME: accept_out narrower than the readback");
        {
            let rp = sc.rp.as_mut().context("REPRIME: runtime missing")?;
            anyhow::ensure!(words * 4 <= rp.pin.len, "REPRIME: {words} accept words > pinned {} B", rp.pin.len);
            rp.pending = None;
            let r = unsafe {
                sys::cuMemcpyDtoHAsync_v2(rp.pin.ptr as *mut std::ffi::c_void, *sc.accept_out.device_ptr(),
                                          words * 4, self.stream.stream)
            };
            anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "REPRIME: accept readback ({r:?})");
            let r = unsafe { sys::cuEventRecord(rp.ev.0, self.stream.stream) };
            anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "REPRIME: event record ({r:?})");
        }
        if rp_xcheck() {
            self.dsync()?; // verify done, re-prime not yet issued
            let s0 = self.rp_snapshot(head, slot, p, rows)?;
            if let Some(rp) = sc.rp.as_mut() { rp.snap0 = Some(s0); }
        }
        let key = Self::rp_key(slot, rows) | sk(sc); // TP-C: + the scratch id
        let want = rp_graph_want();
        let graphed = want && self.graphs.lock().unwrap().contains_key(&key);
        if graphed {
            self.graphs.lock().unwrap().get(&key).unwrap().launch();
        } else {
            // p2 merge (REPRIME x WP03): the verify wrote this lane's taps into ITS window
            let tb = self.tap_base(sc, slot)?;
            self.head_reprime_fixed_launches(l, sc, head, rows, tb)?;
        }
        let rp = sc.rp.as_mut().context("REPRIME: runtime missing")?;
        rp.pending = Some(RpPending { slot, p, rows, key, capture: want && !graphed });
        // NIGHT1: under --gpu-wait (hybrid sleeps then polls; this wait spun the decode thread at 100%)
        crate::host_wait::event_wait(rp.ev.0).map_err(|e| anyhow::anyhow!("REPRIME: event wait ({e:?})"))?;
        Ok((0..words).map(|i| rp.pin.read_u32(i * 4) as i32).collect())
    }

    /// Record head_reprime_fixed_launches into a graph (after an eager round of it, so the
    /// capture only RECORDS). Failure is loud and leaves the device re-prime eager (never wrong).
    fn capture_reprime_graph(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, slot: usize, rows: usize,
                             key: usize) {
        use cudarc::driver::sys;
        let raw = self.stream.stream;
        // p2 merge (REPRIME x WP03): the slot's taps window base, baked into this slot's graph
        let tb = match self.tap_base(sc, slot) {
            Ok(t) => t,
            Err(e) => {
                crate::drprintln!("EXL3_GRAPH: re-prime capture skipped ({e:#}) at rows={rows} — staying eager");
                return;
            }
        };
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(raw, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            crate::metrics::exl3_frac_capture_reset(); // K35: this capture's fractional-expert round tally starts at 0
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::rprintln!("EXL3_GRAPH: re-prime BeginCapture failed ({r:?}) at rows={rows} — staying eager");
                crate::metrics::graph_capture_lost("reprime", &format!("rows={rows}, slot={slot}"), &format!("{r:?}"));
                return;
            }
        }
        let cap = self.head_reprime_fixed_launches(l, sc, head, rows, tb);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r2 = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
        if r2 != sys::CUresult::CUDA_SUCCESS || cap.is_err() {
            crate::rprintln!("EXL3_GRAPH: re-prime capture failed (end={r2:?}, body={:?}) at rows={rows} — staying eager", cap.err());
            crate::metrics::graph_capture_lost("reprime", &format!("rows={rows}, slot={slot}"), &capture_lost_status(r2));
            return;
        }
        if !self.tp_spine_audit(graph, "reprime", rows) { unsafe { sys::cuGraphDestroy(graph); } return; } // TP-C
        unsafe {
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            let r3 = sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0);
            sys::cuGraphDestroy(graph);
            if r3 == sys::CUresult::CUDA_SUCCESS {
                crate::metrics::set_graph_entries({ let mut g = self.graphs.lock().unwrap(); g.insert(key, crate::gpu::CudaGraph::from_exec(exec, raw).with_frac_rounds(crate::metrics::exl3_frac_capture_take())); g.len() as u64 });
                crate::rprintln!("EXL3_GRAPH: captured head re-prime at rows={rows} (key {key:#x})");
            } else {
                crate::rprintln!("EXL3_GRAPH: re-prime instantiate failed ({r3:?}) at rows={rows} — staying eager");
                crate::metrics::graph_capture_lost("reprime", &format!("rows={rows}, slot={slot}"), &format!("instantiate {r3:?}"));
            }
        }
    }

    /// REPRIME: head_reprime's half of a device re-prime enqueued by mtp_verify — the first-round
    /// capture and the optional cross-check. Nothing else: the re-prime already ran on the device.
    fn reprime_dev_finish(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, p: usize,
                          accepted: &[i32], pend: RpPending) -> Result<()> {
        anyhow::ensure!(pend.slot == slot && pend.p == p && accepted.len() <= pend.rows,
                        "REPRIME: pending re-prime {pend:?} does not match the round (slot {slot}, p {p}, a {})",
                        accepted.len());
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        if pend.capture {
            self.dsync()?; // this round's eager launches are done; the capture only records
            self.capture_reprime_graph(&l, sc, head, pend.slot, pend.rows, pend.key);
        }
        if rp_xcheck() {
            self.reprime_xcheck(sc, head, slot, p, accepted, pend.rows)?;
        }
        Ok(())
    }

    /// --reprime-xcheck: raw head state over positions [p, p+rows] of `slot` (clamped to the
    /// window): K/V rows (every kv head, K and V), indexer-key rows, the pooled blocks covering them.
    fn rp_snapshot(&self, head: &DraftHead, slot: usize, p: usize, rows: usize) -> Result<RpSnap> {
        let cfg = &self.cfg;
        let (nkv, hd) = (cfg.num_kv_heads, cfg.head_dim);
        let rb = kv_rowbytes(self.kv_fmt, hd);
        let lo = p.min(self.max_pos - 1);
        let hi = (p + rows).min(self.max_pos - 1);
        let nrow = hi - lo + 1;
        let mut kv = vec![0u8; 2 * nkv * nrow * rb];
        for g in 0..2 * nkv {
            // xq_kv_make: rows of (slot, K|V, kvh) start at ((slot*2 + kv)*nkv + kvh) * max_pos
            let base = (slot * 2 * nkv + g) * self.max_pos * rb;
            self.dtoh_into(&head.kv.slice(base + lo * rb..base + (hi + 1) * rb),
                                         &mut kv[g * nrow * rb..(g + 1) * nrow * rb])?;
        }
        let hdx = cfg.indexer_head_dim;
        let mut keys = Vec::new();
        if let Some(k) = head.qsa_keys.as_ref() {
            let b = (slot * self.max_pos + lo) * hdx;
            keys = vec![0u16; nrow * hdx];
            self.dtoh_into(&k.slice(b..b + nrow * hdx), &mut keys)?;
        }
        let ratio = cfg.indexer_compress_ratio.max(1);
        let nbs = (self.max_pos / ratio).max(1);
        let (j0, j1) = (lo / ratio, (hi / ratio + 1).min(nbs));
        let mut pool = Vec::new();
        if let (Some(pl), true) = (head.qsa_pool.as_ref(), j1 > j0) {
            let b = (slot * nbs + j0) * hdx;
            pool = vec![0u16; (j1 - j0) * hdx];
            self.dtoh_into(&pl.slice(b..b + (j1 - j0) * hdx), &mut pool)?;
        }
        Ok(RpSnap { lo, hi, j0, kv, keys, pool })
    }

    /// --reprime-xcheck body (see rp_xcheck): S0 = before the device re-prime (mtp_verify),
    /// S1 = after it, S2 = after the eager a-row re-prime re-ran on top (the reference).
    fn reprime_xcheck(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, p: usize,
                      accepted: &[i32], rows: usize) -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        static BAD: AtomicUsize = AtomicUsize::new(0);
        let Some(s0) = sc.rp.as_mut().and_then(|r| r.snap0.take()) else {
            crate::dprintln!("REPRIME_XCHECK: no pre-snapshot for slot {slot} p {p} — skipped");
            return Ok(());
        };
        self.dsync()?;
        let s1 = self.rp_snapshot(head, slot, p, rows)?;
        // the reference: the eager a-row re-prime (this call's pending hand-off is already taken)
        self.head_reprime(sc, head, slot, p, accepted)?;
        self.dsync()?;
        let s2 = self.rp_snapshot(head, slot, p, rows)?;
        let a = accepted.len();
        let cfg = &self.cfg;
        let nkv = cfg.num_kv_heads;
        let rb = kv_rowbytes(self.kv_fmt, cfg.head_dim);
        let hdx = cfg.indexer_head_dim;
        let ratio = cfg.indexer_compress_ratio.max(1);
        let nrow = s0.hi - s0.lo + 1;
        let valid = |q: usize| q > p && q <= p + a;
        // [accepted rows: new != eager, other rows: new touched them]
        let (mut kv_bad, mut key_bad, mut pool_bad) = ([0usize; 2], [0usize; 2], [0usize; 2]);
        for g in 0..2 * nkv {
            for t in 0..nrow {
                let r = g * nrow * rb + t * rb..g * nrow * rb + (t + 1) * rb;
                if valid(s0.lo + t) {
                    kv_bad[0] += (s1.kv[r.clone()] != s2.kv[r]) as usize;
                } else {
                    kv_bad[1] += (s1.kv[r.clone()] != s0.kv[r]) as usize;
                }
            }
        }
        if !s0.keys.is_empty() {
            for t in 0..nrow {
                let r = t * hdx..(t + 1) * hdx;
                if valid(s0.lo + t) {
                    key_bad[0] += (s1.keys[r.clone()] != s2.keys[r]) as usize;
                } else {
                    key_bad[1] += (s1.keys[r.clone()] != s0.keys[r]) as usize;
                }
            }
        }
        for jb in 0..s0.pool.len() / hdx.max(1) {
            let j = s0.j0 + jb;
            let r = jb * hdx..(jb + 1) * hdx;
            if (p + 1..=p + a).any(|q| q / ratio == j) {
                pool_bad[0] += (s1.pool[r.clone()] != s2.pool[r]) as usize;
            } else {
                pool_bad[1] += (s1.pool[r.clone()] != s0.pool[r]) as usize;
            }
        }
        let bad = kv_bad.iter().chain(key_bad.iter()).chain(pool_bad.iter()).sum::<usize>();
        let n = N.fetch_add(1, Ordering::SeqCst) + 1;
        if bad > 0 { BAD.fetch_add(1, Ordering::SeqCst); }
        if bad > 0 || n == 1 || n % 256 == 0 {
            crate::dprintln!("REPRIME_XCHECK: rounds {n} bad {} | this: slot {slot} p {p} a {a} rows {rows} \
                      kv rows (accepted new!=eager {}, others touched {}) key rows ({}, {}) pooled blocks \
                      (refreshed new!=eager {}, others changed {}) over positions {}..={}",
                     BAD.load(Ordering::SeqCst), kv_bad[0], kv_bad[1], key_bad[0], key_bad[1],
                     pool_bad[0], pool_bad[1], s0.lo, s0.hi);
        }
        Ok(())
    }

}
