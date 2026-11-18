//! DFlash2 (DF2) hooks on `FwdModel`: capture setters, draft sampling, `df2_round`, `df2_prime_window` /
//! `df2_reprime`, boot pointer tables, `recon_bufs`, `scratch` and `scratch_vrows`. See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// S-A3-f-g: build the reconstruct-path buffer handles for this prefill scratch.
    // ================= EXL27C-DF2 (fwd): the DFlash2 interface ========================
    // The serve-side agent (src/exl3_serve.rs) drives these; the signatures are the agreed
    // interface contract - DO NOT change them unilaterally.

    /// EXL27C-DF2: arm/disarm the DFlash2 tap + prime capture. The serve agent calls this at
    /// boot with the sinks it created for the Df2Round (mirroring main.rs's trunk wiring,
    /// `load_df2_round_dir` -> `gpu.set_df2_capture`). `None` disarms - every tap hook is
    /// then a dead `Option` check (zero-cost, the R1 rule).
    pub fn set_df2_capture(&mut self,
                           sink: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
                           prime: Option<std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>>) {
        self.df2_sink = sink;
        self.df2_prime = prime;
    }

    /// EXL27C-DF2: one DFlash2 draft round - the EXL3 mirror of the trunk's `df2_lane_step`
    /// core (batch.rs ~5760): (1) `draft_round` with anchor = the committed token `b`,
    /// window = k; (2) `mtp_verify` - UNTOUCHED - on the k drafts; (3) `sync_staging_from_sink`
    /// BEFORE `inject_dev` (batch.rs:4974: the round must re-base its staging on THIS round's
    /// taps before the ring advances, or later drafts condition on stale features); (4)
    /// `inject_dev(out.a + 1)` advances the drafter's ring over the accepted span.
    /// FLASHNEXT-DF2-117: the sampled selector walk alone (no verify) — what the HEAD runs and ships
    /// under TP, and what a single box runs before its own `df2_round`. Refreshes the block positions
    /// first (the ring attention reads `pos_blk`; see EXL27C-Fix4 in `df2_round`).
    pub fn df2_draft_sample(&self, round: &mut crate::dflash2::round::Df2Round, b: u32, seeds: &[u32], tau: f32)
                            -> Result<crate::dflash2::round::Df2SampleOut> {
        round.refresh_block_pos()?;
        round.draft_round_dev_sample(b, seeds, tau)
    }

    /// FLASHNEXT-DF2-117 (greedy twin): the greedy draft alone — the captured round graph's replay (or the
    /// eager zeroed-ints path) — what the HEAD runs and ships under TP. The greedy top-1 also flips
    /// across ranks on a near-tie (TP=2: one round in ~1,500; TP=4: within ~700), so greedy drafts are
    /// head-authoritative too.
    pub fn df2_draft_greedy(&self, round: &mut crate::dflash2::round::Df2Round, b: u32) -> Result<Vec<u32>> {
        round.refresh_block_pos()?;
        round.draft_round_graph(b)
    }

    /// `sample`: FLASHNEXT-DF2-117 — a draft drawn earlier (`df2_draft_sample` / `df2_draft_greedy`, this rank's
    /// or the head's shipped one); with `rq` Some it REPLACES the walk here, so every rank verifies
    /// one draft with one proposal law. None with `rq` Some = draw here (the pre-117 single-box path).
    pub fn df2_round(&self, sc: &mut Scratch, round: &mut crate::dflash2::round::Df2Round,
                     slot: usize, k: usize, b: i32, p: usize,
                     rq: Option<(&[u32], f32)>, sample: Option<crate::dflash2::round::Df2SampleOut>,
                     t: [std::time::Instant; 3]) -> Result<MtpRoundOut> {
        use std::sync::atomic::Ordering::Relaxed;
        let tr = DF2_TRACE.load(Relaxed);
        if tr { let _ = self.dsync(); eprintln!("[df2trace] draft begin k={k} b={b} p={p}"); }
        let mut ttr = std::time::Instant::now();
        // EXL27C-Fix4: the trunk refreshes pos_blk before EVERY draft (batch.rs x5;
        // round.rs:1450 'Callers must refresh_block_pos() first'). Without it pos_blk holds
        // stale/garbage absolute positions and the ring attention (gqa_attn_band_ring_b)
        // maps wild rows: sanitizer OOB __shared__ scores writes + CUDA_ERROR_ILLEGAL_ADDRESS
        // (2026-10-16, .14, block 8 and 16 alike).
        // PERFFIX1: skipped when this rank's draft (df2_draft_sample / df2_draft_greedy) already refreshed this frontier.
        round.refresh_block_pos_if_stale()?;
        // EXL27D-RQ (item 1, 2026-10-21): a SAMPLED lane (sc.rq_mode 2 — --spec-sampling
        // ratio, the default) drafts through the S5F2 sampled selector walk (the vendor
        // CandidateSelector.sample_path multinomial at tau) and stages the walk's candidate
        // table into sc.rq_list below, so the verify's ratio rows (xq_sample_rows_rq) judge
        // every draft under the real-q criterion u·q' < p with the exact relu(p−q) residual.
        // The greedy chain never writes list rows: the kernel's q' lookup missed, rewrote
        // the proposal to the delta law q'=1 and accepted iff u < p(draft) — E[p(argmax)]
        // ≈ 12% at temp 1.0 on thinking/code content. That delta law was the −37.3%
        // signature (.14, 2026-10-21: 414 rounds, 0.85 accepted/round vs the vendor's
        // 4.1–5.5 acceptance length at the same sampling parameters).
        let (drafts, samp_out) = match rq {
            Some((seeds, tau)) => {
                let out = match sample { Some(o) => o, None => round.draft_round_dev_sample(b as u32, seeds, tau)? };
                let drafts: Vec<i32> = out.tokens.iter().take(k).map(|&tk| tk as i32).collect();
                (drafts, Some(out))
            }
            None => {
                // FLASHNEXT-DF2-B (ISSUES work[112], 2026-10-23): the greedy draft takes the
                // trunk's proven entry (batch.rs): the captured round graph when present,
                // else the eager kernels with the graph's device ints ZEROED. The old call
                // `draft_round(b, k, ..)` passed the verify width k as the attention WINDOW
                // (a 7-key band instead of 2048) and ran the eager kernels with `ntot_dev`
                // still pointing at the boot-capture's device int (max_c + block) whenever a
                // round graph existed — the band kernel then walked [lo, 262152) against an
                // eager-sized smem: compute-sanitizer's `gqa_attn_band_ring_b` __shared__
                // OOB, then CUDA_ERROR_ILLEGAL_ADDRESS (engine DEAD). The sampled branch
                // above never had either defect (draft_round_dev_sample zeroes the ints and
                // uses the sliding window).
                let toks = match sample { Some(o) => o.tokens, None => round.draft_round_graph(b as u32)? };
                if DF2_TRACE.load(std::sync::atomic::Ordering::Relaxed) {
                    static D2N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                    if D2N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
                        let t: Vec<u32> = toks.iter().take(6).copied().collect();
                        eprintln!("[df2trace] drafts={t:?} nprev={}", round.nprev());
                    }
                }
                if tr { let _ = self.dsync(); eprintln!("[df2trace] draft done ntok={}", toks.len()); }
                let drafts: Vec<i32> = toks.iter().take(k).map(|&tk| tk as i32).collect();
                (drafts, None)
            }
        };
        // EXL27C-Fix5: xq_accept compares the verify argmax rows against sc.d_dev — the
        // adaptive MTP path fills it device-side from its draft kernels; the DF2 chain
        // drafts in its OWN buffers, so the drafts must be uploaded here or the accept
        // compares against stale bytes and rejects every round (324 rounds, 0 accepted,
        // drafts[0] matched the target's argmax every round — 2026-10-16, .14).
        // Layout: d_vec[0..k-1] plain draft ids (kernel loop bounds a < k; pad is never read).
        {
            let mut d_w = drafts.clone();
            d_w.resize(sc.d_dev.len(), 0);
            self.dev.htod_copy_into(d_w, &mut sc.d_dev)?;
        }
        // EXL27D-RQ: the candidate table rows the ratio rows read (RQ_ROW=96 layout, the
        // one xq_rq_draft writes for the pack-MTP path): [0..32) candidate ids (u32::MAX
        // pads trail), [32..64) their logits (ln q; the ratio path reads only ids+q), [64..96)
        // the proposal q' — the SAME law the walk sampled from (cand_q sums to 1 per
        // position), so the accept and the residual are exact for this proposal.
        if let Some(out) = &samp_out {
            let l_ = crate::dflash2::levels();
            let mut rows = vec![u32::MAX; sc.rq_list.len()];
            for j in 0..k.min(l_) {
                for t in 0..16usize {
                    let q = out.cand_q[j * 16 + t];
                    rows[j * crate::wp24::RQ_ROW + t] = out.cand_tok[j * 16 + t];
                    rows[j * crate::wp24::RQ_ROW + 32 + t] = if q > 0.0 { q.ln().to_bits() } else { 0 };
                    rows[j * crate::wp24::RQ_ROW + 64 + t] = q.to_bits();
                }
            }
            self.dev.htod_copy_into(rows, &mut sc.rq_list)?;
        }
        if tr { let _ = self.dsync(); eprintln!("[df2trace] verify begin (d_dev uploaded)"); }
        let out = self.mtp_verify(sc, slot, k, b, p, drafts, t)?;
        if tr { let _ = self.dsync(); eprintln!("[df2trace] verify done a={} ms={:.1}", out.a, ttr.elapsed().as_secs_f32()*1e3); ttr = std::time::Instant::now(); }
        if DF2_TRACE.load(std::sync::atomic::Ordering::Relaxed) {
            static V2N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if V2N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
                eprintln!("[df2trace] verify: a={} emitted0..2={:?}", out.a,
                          out.emitted.iter().take(3).collect::<Vec<_>>());
            }
        }
        round.sync_staging_from_sink()?;
        if tr { let _ = self.dsync(); eprintln!("[df2trace] staging synced ms={:.1}", ttr.elapsed().as_secs_f32()*1e3); ttr = std::time::Instant::now(); }
        round.inject_dev(out.a + 1, None)?;
        if tr { let _ = self.dsync(); eprintln!("[df2trace] injected {} ms={:.1}", out.a + 1, ttr.elapsed().as_secs_f32()*1e3); }
        Ok(out)
    }

    /// EXL27C-DF2: prime the drafter's ring over one prefill window's taps. The serve agent
    /// calls this per prefill chunk, after `prefill_chunk`'s tap hook captured the chunk's
    /// rows into the prime sink; `pos_start` = the chunk's first absolute position.
    pub fn df2_prime_window(&self, round: &mut crate::dflash2::round::Df2Round,
                            pos_start: usize, n: usize) -> Result<()> {
        let prime = self.df2_prime.as_ref().ok_or_else(|| {
            anyhow::anyhow!("df2_prime_window: no prime sink armed (set_df2_capture first)")
        })?;
        round.prime_window(&prime.taps, n, pos_start)
    }

    /// EXL27C-DF2-Fix2: re-prime the drafter's ring from the taps the LAST target forward
    /// captured (the sink is attached for the whole boot; every solo forward - plain step,
    /// control round, MTP fallback round - writes its columns into the staging). Call after
    /// any NON-DF2 solo round with n = that round's committed span, so the next solo round
    /// is DF2 again instead of falling back forever (`df2_round` advances the ring itself).
    pub fn df2_reprime(&self, round: &mut crate::dflash2::round::Df2Round, n: usize) -> Result<()> {
        round.sync_staging_from_sink()?;
        round.inject_dev(n, None)?;
        Ok(())
    }

    /// EXL27C-DF2: the (embed, head) device pointers the Df2Round borrows - the EXL3 analog
    /// of the trunk's `gpu.df2_borrow()`. BOOT-TIME ONLY (one-shot, ~5 GB of transients +
    /// ~5 GB persistent on the 27B; logged via drprintln). Idempotent: a second call
    /// returns the same pointers.
    ///
    /// * embed: the pack stages embed_tokens as **f16** (`get_plain_f16`; the xq stack is
    ///   f16), while the DF2 round's `embed_gather_b` reads **bf16** - so we materialize a
    ///   bf16 twin ONCE (`xq_qsa_h2b`). [vocab, hidden] row-major, dense.
    /// * head: the 6-bit EXL3 lm_head quad dequantized ONCE into [vocab, hidden] bf16
    ///   row-major with the pack's block-Hadamard algebra FOLDED into the matrix. The serve
    ///   logits are y = B_N(W_dec^T(A_K x)) with A_K = H128*diag(suh)/sqrt(128)
    ///   (exl3_had_suh: pre-scale, butterfly) and B_N = diag(svh)*H128/sqrt(128)
    ///   (exl3_had_svh: butterfly, post-scale), so the equivalent plain matrix is
    ///   W_eq = B_N W_dec^T A_K, built as:
    ///   (a) exl3_reconstruct -> W_dec [k, n] f16 (the oracle-validated dequant);
    ///   (b) fold B_N over W_dec's ROWS = exl3_had_svh in place (butterfly-then-svh);
    ///   (c) transpose [k, n] -> [n, k] (`xq_df2_transpose_f16`, f16 bits);
    ///   (d) fold A_K^T over the rows = exl3_had_svh with svh:=suh in place
    ///       (A_K^T = diag(suh)*H128/sqrt(128) - butterfly FIRST then suh, that kernel);
    ///   (e) f16 -> bf16 in place (`xq_qsa_h2b`).
    /// All steps reuse serving-path kernels; raw device pointers are passed for the
    /// in-place steps (no &mut aliasing). Numerics note: the composition carries 2-3 f16
    /// roundings vs the serve path's interleaved roundings (rel ~1e-3) - acceptable for
    /// the DRAFTER's scoring; acceptance is judged by the target's own logits in
    /// mtp_verify.
    /// The NVFP4-mma pack of the dequantized EXL3 lm_head (one-time; `--no-df2-q2stage` opts out
    /// at the borrow site in exl3_serve). Same recipe as gpu.rs `build_df2_q_head`: RTW quantize
    /// the full [n,k] bf16 head, MMA-repack, gs = 1/global per 16-row tile (kernel convention).
    pub fn df2_q_head_ptrs(&self) -> Option<crate::dflash2::round::Nvfp4Ptrs> {
        let g = self.df2_q_head.get_or_init(|| {
            let guard = self.df2_head_bf16.lock().unwrap();
            let b = guard.as_ref()?;
            let q = &self.lm_head;
            let (kk, nn) = (q.k as usize, q.n as usize);
            if nn % 16 != 0 || kk % 16 != 0 { return None; }
            eprintln!("[df2] quantizing EXL3 borrowed head bf16->NVFP4-mma in memory ({nn} x {kk}) — one-time");
            let t0 = std::time::Instant::now();
            let host: Vec<half::bf16> = self.dtoh(b).ok()?;
            crate::quant::set_repack_threading(true);
            let t = crate::quant::quantize_nvfp4(&host, nn, kk);
            drop(host);
            let (wt, st) = crate::quant::repack_nvfp4_mma(&t.qweight, &t.scales, nn, kk);
            let inv_gs = 1.0 / t.global_scale;
            let gsv = vec![inv_gs; nn / 16];
            let qweight = self.dev.htod_sync_copy(&wt).ok()?;
            let scales = self.dev.htod_sync_copy(&st).ok()?;
            let gs = self.dev.htod_sync_copy(&gsv).ok()?;
            eprintln!("[df2] EXL3 head quantized in {:.1}s ({} MB packed, was {} MB bf16) — df2 head now reads ~1/3.7x",
                      t0.elapsed().as_secs_f32(), wt.len() / 1_000_000, nn * kk * 2 / 1_000_000);
            Some((qweight, scales, gs))
        });
        let (q, sc, gs) = g.as_ref()?;
        Some(crate::dflash2::round::Nvfp4Ptrs {
            qweight: *q.device_ptr() as u64, scales: *sc.device_ptr() as u64, gs: *gs.device_ptr() as u64,
        })
    }

    pub fn df2_boot_ptrs(&self) -> Result<(u64, u64)> {
        let t0 = std::time::Instant::now();
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let h = self.cfg.hidden_size;

        // ---- embed twin (f16 staging -> bf16) ------------------------------------
        let embed_ptr = {
            let mut g = self.df2_embed_bf16.lock().unwrap();
            if g.is_none() {
                let n = self.embed.len();
                anyhow::ensure!(n % h == 0, "df2 embed twin: len {n} not a multiple of h {h}");
                let twin = self.dev.alloc_zeros::<half::bf16>(n)
                    .map_err(|e| anyhow::anyhow!("df2 embed twin alloc: {e:?}"))?;
                let total = n as i64;
                let blocks = (((total + 255) / 256).min(65535)) as u32;
                if self.embed_bf {
                    // FP4L --fp4l-plain exact: the table already holds the stored bf16 bits — a bit copy on the
                    // compute stream (AGENTS §2.3), no conversion
                    fp4l_fwd::copy_u16(&l, *twin.device_ptr() as u64, *self.embed.device_ptr() as u64, n)?;
                    *g = Some(twin);
                } else {
                xqlaunch!(l, "xq_qsa_h2b", (blocks, 1, 1), (256, 1, 1), 0,
                          (*twin.device_ptr() as u64, *self.embed.device_ptr() as u64, total))?;
                *g = Some(twin);
                }
            }
            *g.as_ref().unwrap().device_ptr() as u64
        };

        // ---- head: dequant + fold the Hadamard algebra -----------------------------
        let head_ptr = {
            let mut g = self.df2_head_bf16.lock().unwrap();
            if g.is_none() {
                if let Some(fx) = self.lm_head.fx.as_deref() {
                    // FP4L: the head is already [vocab, hidden]; the DFlash2 head is its bf16 view: the stored bits
                    // (bf16), one exact cast (f16 shift), or the exact bf16 dequant of the MXFP8 copy (mxfp8)
                    let n_el = self.lm_head.n as usize * self.lm_head.k as usize;
                    let head = self.dev.alloc_zeros::<half::bf16>(n_el).map_err(|e| anyhow::anyhow!("df2 head alloc: {e:?}"))?;
                    fp4l_fwd::head_to_bf16(&l, *head.device_ptr() as u64, fx, self.lm_head.n as usize, self.lm_head.k as usize)?;
                    crate::drprintln!("df2_boot_ptrs: FP4L {} head [{} x {h}] -> bf16 (exact), {:.2} s", fx.kind(), n_el / h,
                                     t0.elapsed().as_secs_f64());
                    *g = Some(head);
                }
            }
            if g.is_none() {
                let q = &self.lm_head;
                let (kk, nn) = (q.k as usize, q.n as usize);
                anyhow::ensure!(kk == h, "df2 head: quad k {kk} != hidden {h}");
                anyhow::ensure!(kk % 128 == 0 && nn % 128 == 0,
                                "df2 head: quad {kk}x{nn} not 128-aligned (had_svh blocks)");
                // (a) the dequant: W_dec [K, N] f16 row-major into ws.
                let ws = self.dev.alloc_zeros::<u16>(kk * nn)
                    .map_err(|e| anyhow::anyhow!("df2 head ws alloc: {e:?}"))?;
                let ws_ptr = *ws.device_ptr() as u64;
                exl3_reconstruct(&l, *q.tr.device_ptr() as u64,
                                 *self.rc_lut.device_ptr() as u64, ws_ptr, q.k, q.n, q.k2)?;
                // (b) fold B_N over W_dec's rows (in place; each thread owns its 4 elements).
                xqlaunch!(l, "exl3_had_svh", ((kk * (nn / 128)) as u32, 1, 1), (32, 1, 1), 0,
                          (ws_ptr, &q.svh, ws_ptr, q.n))?;
                // (c)+(d)+(e): transpose to [N, K], fold A_K^T (butterfly-then-suh) over the
                // rows, and swap f16 -> bf16 - the head buffer is allocated with its final
                // bf16 type and its raw pointer is bit-reinterpreted for the f16 passes.
                let mut head = self.dev.alloc_zeros::<half::bf16>(nn * kk)
                    .map_err(|e| anyhow::anyhow!("df2 head alloc: {e:?}"))?;
                let head_ptr = *head.device_ptr() as u64;
                xqlaunch!(l, "xq_df2_transpose_f16",
                          (((kk + 31) / 32) as u32, ((nn + 31) / 32) as u32, 1), (32, 32, 1), 0,
                          (head_ptr, ws_ptr, q.k, q.n))?;
                xqlaunch!(l, "exl3_had_svh", ((nn * (kk / 128)) as u32, 1, 1), (32, 1, 1), 0,
                          (head_ptr, &q.suh, head_ptr, q.k))?;
                let total = (nn * kk) as i64;
                let blocks = (((total + 255) / 256).min(65535)) as u32;
                xqlaunch!(l, "xq_qsa_h2b", (blocks, 1, 1), (256, 1, 1), 0,
                          (head_ptr, head_ptr, total))?;
                drop(ws); // free the transient before parking the persistent buffer
                crate::drprintln!(
                    "df2_boot_ptrs: head [{nn}x{kk}] dequant+fold+transpose, embed twin, {:.2} s (one-time)",
                    t0.elapsed().as_secs_f64());
                *g = Some(head);
            }
            *g.as_ref().unwrap().device_ptr() as u64
        };

        Ok((embed_ptr, head_ptr))
    }

    pub(super) fn recon_bufs(&self, psc: &PrefillScratch) -> ReconBufs {
        ReconBufs {
            ws: *psc.recon_ws.device_ptr() as u64,
            ws_len: psc.recon_ws.len(),
            rc: *self.rc_lut.device_ptr() as u64,
            ltws: *psc.lt_ws.device_ptr() as u64,
            ltws_len: psc.lt_ws.len(),
        }
    }

    /// Scratch constructor for the serve backend (fields stay private to this module).
    pub fn scratch(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config, w: usize, max_pos: usize) -> Result<Scratch> {
        Scratch::new(dev, cfg, w, cfg.rotary_dim, max_pos)
    }

    /// WIDEVERIFY1 refinement 2: the served scratch with `vrows` lone-verify rows (`Scratch::new_vrows`).
    pub fn scratch_vrows(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config, w: usize, max_pos: usize,
                         vrows: usize) -> Result<Scratch> {
        Scratch::new_vrows(dev, cfg, w, cfg.rotary_dim, max_pos, vrows)
    }
}
