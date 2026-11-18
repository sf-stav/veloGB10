//! Prefill-side hyper-connection methods over many rows (hc_pre_rows*, hc_post_rows*, the sp variants and
//! `sp_fault_flip`) and the small tr_* transposes. See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// hc_pre against the PrefillScratch buffers (same launches as hc_pre at m=C).
    pub(super) fn hc_pre_rows(&self, l: &Launcher, psc: &mut PrefillScratch, hc: &HcDev, c: usize) -> Result<()> {
        self.hc_pre_rows_p(l, psc, hc, c, None)
    }

    /// A5-P2: which of the fused prefill hc kernels apply to this model (knob AND shape).
    /// (norm, mix): xq_pf_hc_inj_norm needs h % 512 == 0 && h <= 4096; xq_pf_hc_mix_2560x4 the
    /// served h = 2560, hc = 4. Anything else keeps the old launches.
    pub(super) fn pf_hc_fuse(&self, l: &Launcher) -> (bool, bool) {
        let on = tune::get(&T_PF_HC_FUSE, &l.cx()) != 0;
        let (h, hcn) = (self.cfg.hidden_size, self.cfg.hc_count.max(1));
        let norm = on && h % 512 == 0 && h <= 4096;
        let mix = on && h == 2560 && hcn == 4;
        static SEEN: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
        let key = 1u8 | (norm as u8) << 1 | (mix as u8) << 2;
        if SEEN.swap(key, std::sync::atomic::Ordering::Relaxed) != key {
            crate::dprintln!("A5-P2 prefill hc: fuse={} (inj_norm {} mix {}; h {h} hc {hcn})", on as u8,
                     if norm { "xq_pf_hc_inj_norm (deferred inject)" } else { "xq_hc_inject+xq_hc_norm" },
                     if mix { "xq_pf_hc_mix_2560x4" } else { "xq_hc_mix4" });
        }
        (norm, mix)
    }

    /// hc_pre_rows with an optional PENDING inject (`pend` = the sublayer output a deferred
    /// hc_post_rows did not launch). With prefill.hc_fuse the inject rides the norm launch
    /// (xq_pf_hc_inj_norm); otherwise the pending inject is launched first (xq_hc_inject).
    fn hc_pre_rows_p(&self, l: &Launcher, psc: &mut PrefillScratch, hc: &HcDev, c: usize,
                     pend: Option<u64>) -> Result<()> {
        self.hc_pre_rows_n(l, psc, hc, c, pend, false)
    }

    /// TP-I2 item 1: the rows-landed hook of a pipelined prefill reduce (xtp::PfHook) for the
    /// hc_pre `hc` whose pending inject is `y`; None when the overlap is off (`rows` == 0).
    pub(super) fn pf_hook(&self, psc: &PrefillScratch, hc: &HcDev, y: u64, rows: usize) -> Option<xtp::PfHook> {
        if rows == 0 { return None; }
        Some(xtp::PfHook {
            hn: *psc.hn.device_ptr() as u64, resid: *psc.resid.device_ptr() as u64, y,
            inj: *psc.inj.device_ptr() as u64,
            // FP4L --fp4l-norms f32: the hook's per-row consumer is the fp32-weight twin
            norm: hc.norm_f.as_ref().map_or(*hc.norm.device_ptr() as u64, |f| *f.device_ptr() as u64),
            norm_f32: hc.norm_f.is_some(),
            h: self.cfg.hidden_size, hcn: self.cfg.hc_count.max(1), eps: self.cfg.rms_eps, rows,
            dual: xtp::pf_overlap_dual(),
            fold: xtp::pf_overlap_fold(),
        })
    }

    /// hc_pre_rows_p; `normed` (TP-I2 item 1) = the pending inject + this hc_pre's norm already ran
    /// per row range inside the preceding reduce (the fused xq_pf_hc_inj_norm), so it is skipped
    /// here and `pend` (that inject's sublayer output) is consumed.
    pub(super) fn hc_pre_rows_n(&self, l: &Launcher, psc: &mut PrefillScratch, hc: &HcDev, c: usize,
                     pend: Option<u64>, normed: bool) -> Result<()> {
        let h = self.cfg.hidden_size;
        let hcn = self.cfg.hc_count.max(1);
        let lr = self.cfg.hc_lowrank;
        let rw = h * hcn;
        let (fnorm, fmix) = self.pf_hc_fuse(l);
        let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
        if normed {
            anyhow::ensure!(fnorm && pend.is_some(), "prefill overlap: a hooked norm without the fused inj_norm path");
        } else if fnorm {
            if let Some(nf) = hc.norm_f.as_ref() {
                // FP4L --fp4l-norms f32: xq_pf_hc_inj_norm's twin on the stored fp32 norm
                fp4l_fwd::pf_hc_inj_norm_f(l, dp(psc.hn.device_ptr()), dp(psc.resid.device_ptr()), pend.unwrap_or(0u64),
                                           dp(psc.inj.device_ptr()), dp(nf.device_ptr()), h, hcn, c, self.cfg.rms_eps)?;
            } else {
            xqlaunch!(l, "xq_pf_hc_inj_norm", ((hcn * c) as u32, 1, 1), (128, 1, 1), 0,
                     (&mut psc.hn, &mut psc.resid, pend.unwrap_or(0u64), &psc.inj, &hc.norm,
                      h as i32, hcn as i32, c as i32, self.cfg.rms_eps))?;
            }
        } else {
            if let Some(y) = pend { self.hc_post_rows(l, psc, y, c)?; }
            if let Some(nf) = hc.norm_f.as_ref() {
                fp4l_fwd::hc_norm_f(l, dp(psc.hn.device_ptr()), dp(psc.resid.device_ptr()), nf, h, hcn, c, self.cfg.rms_eps)?;
            } else {
            xqlaunch!(l, "xq_hc_norm", ((hcn * c) as u32, 1, 1), (1024, 1, 1), 4096,
                     (&mut psc.hn, &psc.resid, &hc.norm, h as i32, hcn as i32, c as i32, self.cfg.rms_eps))?;
            }
        }
        if lr == 0 {
            // EXL27B: plain residual — the normed stream IS the mixer input (hn -> x)
            let n = (rw * c) as i64;
            xqlaunch!(l, "xq_copy_u16", (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut psc.x, &psc.hn, n, 0i64, 0i64))?;
            return Ok(());
        }
        // S-A3-f-g: the hc down/up projections are the bulk of the prefill
        // xq_gemm_f16 cost (nsys: ~216 launches/chunk, same per-row K-serial
        // shape as a_w/b_w). Wide-M rows go through the vendor GEMM (opA=T);
        // the custom kernel stays for narrow workloads (rmin=0 → unchanged).
        let rmin = recon_min_rows();
        if hc.bf {
            // FP4L --fp4l-plain exact: the stored-bf16 down / up at the chunk's rows (two bf16 MMAs per f16 one)
            let (hn, dd, uu) = (dp(psc.hn.device_ptr()), dp(psc.dd.device_ptr()), dp(psc.uu.device_ptr()));
            fp4l_fwd::bf_wide(l, &hc.down, hn, dd, 0, c, rw, lr)?;
            xqlaunch!(l, "xq_silu_div", (((lr * c) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                     (dd, hcn as f32, (lr * c) as i32))?;
            fp4l_fwd::bf_wide(l, &hc.up, dd, uu, 0, c, lr, rw)?;
        } else if psc.row_indep {
            // LANEWIDE-EXACT: the cuBLASLt algos pinned per shape (lwv::wide_pin), whatever the pass's row count
            let rb = self.recon_bufs(psc);
            exl3_lt_matmul_pinned(l, *hc.down.device_ptr() as u64, *psc.hn.device_ptr() as u64, *psc.dd.device_ptr() as u64,
                                  rw, lr as usize, c, lwv::wide_pin(rw, lr, rw, lr), true, rb.ltws, rb.ltws_len)?;
            xqlaunch!(l, "xq_silu_div", (((lr * c) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                     (&mut psc.dd, hcn as f32, (lr * c) as i32))?;
            exl3_lt_matmul_pinned(l, *hc.up.device_ptr() as u64, *psc.dd.device_ptr() as u64, *psc.uu.device_ptr() as u64,
                                  lr as usize, rw, c, lwv::wide_pin(lr, rw, rw, lr), true, rb.ltws, rb.ltws_len)?;
        } else if rmin > 0 && c >= rmin {
            let rb = self.recon_bufs(psc);
            exl3_lt_matmul(l, *hc.down.device_ptr() as u64, *psc.hn.device_ptr() as u64,
                           *psc.dd.device_ptr() as u64, rw, lr as usize, c, true,
                           rb.ltws, rb.ltws_len)?;
            // activation between the two projections is elementwise: keep the
            // original kernel in both arms.
            xqlaunch!(l, "xq_silu_div", (((lr * c) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                     (&mut psc.dd, hcn as f32, (lr * c) as i32))?;
            exl3_lt_matmul(l, *hc.up.device_ptr() as u64, *psc.dd.device_ptr() as u64,
                           *psc.uu.device_ptr() as u64, lr as usize, rw, c, true,
                           rb.ltws, rb.ltws_len)?;
        } else {
            xq_gemm_f16_small!(l, &mut psc.dd, &hc.down, &psc.hn, c, lr, rw)?;
            xqlaunch!(l, "xq_silu_div", (((lr * c) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                     (&mut psc.dd, hcn as f32, (lr * c) as i32))?;
            xq_gemm_f16_small!(l, &mut psc.uu, &hc.up, &psc.dd, c, rw, lr)?;
        }
        if hc.bf {
            // FP4L --fp4l-plain exact: the mix twins on the stored bf16 inject weights
            let winj = hc.winj.as_ref().map_or(0u64, |w| *w.device_ptr() as u64);
            let (xo, inj, hn, uu) = (dp(psc.x.device_ptr()), dp(psc.inj.device_ptr()), dp(psc.hn.device_ptr()), dp(psc.uu.device_ptr()));
            return if fmix { fp4l_fwd::pf_hc_mix_bf(l, xo, inj, hn, uu, winj, c) }
                   else { fp4l_fwd::hc_mix_bf(l, xo, inj, hn, uu, winj, h, hcn, c) };
        }
        if fmix {
            let winj = hc.winj.as_ref().map_or(0u64, |w| *w.device_ptr() as u64);
            xqlaunch!(l, "xq_pf_hc_mix_2560x4", (c as u32, 1, 1), (128, 1, 1), 0,
                     (&mut psc.x, &mut psc.inj, &psc.hn, &psc.uu, winj, c as i32))?;
            return Ok(());
        }
        let (mixk, mixs) = if hcn <= 4 && hc_mix4_on() { ("xq_hc_mix4", 4 * 1024 * 4) } else { ("xq_hc_mix", 4096) };
        match &hc.winj {
            Some(w) => xqlaunch!(l, mixk, (c as u32, 1, 1), (1024, 1, 1), mixs as u32,
                                (&mut psc.x, &mut psc.inj, &psc.hn, &psc.uu, w, h as i32, hcn as i32, c as i32))?,
            None => xqlaunch!(l, mixk, (c as u32, 1, 1), (1024, 1, 1), mixs as u32,
                             (&mut psc.x, &mut psc.inj, &psc.hn, &psc.uu, 0u64, h as i32, hcn as i32, c as i32))?,
        }
        Ok(())
    }

    /// TP-SP1: this rank's row block when a c-row chunk runs sequence-parallel, else None (the
    /// replicated prefill). SPMD: every input is identical on all ranks — the flag (TpConfig), c, the
    /// transport, and the pinned-plan readiness of EVERY rank's block (lt_pin_ready checks all blocks,
    /// at world 4 the quarter blocks — TP-4S).
    pub(super) fn sp_rows_for(&self, psc: &PrefillScratch, c: usize) -> Option<xtp::SpRows> {
        let tp = self.tp.as_ref()?;
        if !xtp::seq_parallel_on() || !(xtp::SP_MIN_ROWS..=xtp::SP_MAX_ROWS).contains(&c) || !tp.sp_ready() {
            return None;
        }
        static WHY: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        let once = |w: String| {
            let mut v = WHY.lock().unwrap();
            if !v.contains(&w) { crate::dprintln!("[exl3-tp] TP-SP: {w}"); v.push(w); }
        };
        if xtp::pf_overlap_fold() {
            once("sequence-parallel prefill is not composed with --tp-prefill-overlap fold: chunks run replicated".into());
            return None;
        }
        let (h, hcn, lr) = (self.cfg.hidden_size, self.cfg.hc_count.max(1), self.cfg.hc_lowrank);
        let (world, rank) = (tp.world as usize, tp.rank as usize);
        let rmin = recon_min_rows();
        if rmin > 0 && c >= rmin {
            let rb = self.recon_bufs(psc);
            for (k, n) in sp_split_shapes(h, hcn, lr) {
                if let Err(e) = lt_pin_ready(k, n, c, world, true, rb.ltws_len) {
                    once(format!("chunk of {c} rows runs replicated: the pinned hc plan (k {k} n {n}) is not runnable ({e:#})"));
                    return None;
                }
            }
        }
        let sp = xtp::SpRows::new(c, rank, world);
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::dprintln!("[exl3-tp] TP-SP sequence-parallel prefill ACTIVE at world {world} (first chunk c {c}: rank {rank} owns rows {:?}, \
                                    row blocks {:?}; chunks of {}..={} rows split, hc GEMM plans pinned to the full chunk's)",
                                   sp.own, &sp.blocks[..world.min(4)], xtp::SP_MIN_ROWS, xtp::SP_MAX_ROWS));
        Some(sp)
    }

    /// TP-SP1: hc_pre_rows_n on this rank's OWN rows of the chunk (every launch of hc_pre_rows_n at the
    /// same per-row arithmetic, on row-offset pointers over sp.own; the hc down/up GEMMs through the
    /// full chunk's cuBLASLt plan PINNED on the own-row layouts), then the all-gather of x (the peer's
    /// rows land in this rank's x) when `gather_x`. inj / hn / resid / dd / uu are written for the own
    /// rows only.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_pre_rows_sp(&self, l: &Launcher, psc: &mut PrefillScratch, hc: &HcDev, sp: xtp::SpRows,
                      pend: Option<u64>, normed: bool, gather_x: bool) -> Result<()> {
        let tp = self.tp.as_ref().context("TP-SP hc_pre without a TP attachment")?;
        let h = self.cfg.hidden_size;
        let hcn = self.cfg.hc_count.max(1);
        let lr = self.cfg.hc_lowrank;
        let rw = h * hcn;
        let c = sp.c;
        let (lo, hi) = sp.own;
        let b = hi - lo;
        anyhow::ensure!(hi > lo && hi <= c, "TP-SP: empty own block {:?} of {c}", sp.own);
        let (fnorm, fmix) = self.pf_hc_fuse(l);
        let hn = *psc.hn.device_ptr() as u64 + (lo * rw * 2) as u64;
        let resid = *psc.resid.device_ptr() as u64 + (lo * rw * 4) as u64;
        let inj = *psc.inj.device_ptr() as u64 + (lo * hcn * 4) as u64;
        let xo = *psc.x.device_ptr() as u64 + (lo * h * 2) as u64;
        let dd = *psc.dd.device_ptr() as u64 + (lo * lr * 2) as u64;
        let uu = *psc.uu.device_ptr() as u64 + (lo * rw * 2) as u64;
        let pend_o = pend.map(|y| y + (lo * h * 2) as u64);
        if normed {
            anyhow::ensure!(fnorm && pend.is_some(), "prefill overlap: a hooked norm without the fused inj_norm path");
        } else if fnorm {
            if let Some(nf) = hc.norm_f.as_ref() {
                fp4l_fwd::pf_hc_inj_norm_f(l, hn, resid, pend_o.unwrap_or(0u64), inj, *nf.device_ptr() as u64, h, hcn, b, self.cfg.rms_eps)?;
            } else {
            xqlaunch!(l, "xq_pf_hc_inj_norm", ((hcn * b) as u32, 1, 1), (128, 1, 1), 0,
                     (hn, resid, pend_o.unwrap_or(0u64), inj, &hc.norm, h as i32, hcn as i32, b as i32, self.cfg.rms_eps))?;
            }
        } else {
            if let Some(y) = pend_o {
                xqlaunch!(l, "xq_hc_inject", (((h * hcn * b) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                         (resid, y, inj, h as i32, hcn as i32, b as i32))?;
            }
            if let Some(nf) = hc.norm_f.as_ref() {
                fp4l_fwd::hc_norm_f(l, hn, resid, nf, h, hcn, b, self.cfg.rms_eps)?;
            } else {
            xqlaunch!(l, "xq_hc_norm", ((hcn * b) as u32, 1, 1), (1024, 1, 1), 4096,
                     (hn, resid, &hc.norm, h as i32, hcn as i32, b as i32, self.cfg.rms_eps))?;
            }
        }
        let rmin = recon_min_rows();
        if hc.bf {
            // FP4L --fp4l-plain exact: the stored-bf16 down / up on the own rows (row-independent: the full chunk's bits)
            fp4l_fwd::bf_wide(l, &hc.down, hn, dd, 0, b, rw, lr)?;
            xqlaunch!(l, "xq_silu_div", (((lr * b) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0, (dd, hcn as f32, (lr * b) as i32))?;
            fp4l_fwd::bf_wide(l, &hc.up, dd, uu, 0, b, lr, rw)?;
        } else if rmin > 0 && c >= rmin {
            // the replicated chunk takes the vendor GEMM at m = c: run ITS plan's algo on the own rows
            let rb = self.recon_bufs(psc);
            exl3_lt_matmul_pinned(l, *hc.down.device_ptr() as u64, hn, dd, rw, lr, b, c, true, rb.ltws, rb.ltws_len)?;
            xqlaunch!(l, "xq_silu_div", (((lr * b) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0, (dd, hcn as f32, (lr * b) as i32))?;
            exl3_lt_matmul_pinned(l, *hc.up.device_ptr() as u64, dd, uu, lr, rw, b, c, true, rb.ltws, rb.ltws_len)?;
        } else {
            // one CTA row per chunk row: row-offset pointers are the full launch's bits on those rows
            xqlaunch!(l, "xq_gemm_f16", (((lr as u32) + 127) / 128, b as u32, 1), (128, 1, 1), 0,
                     (dd, &hc.down, hn, b as i32, lr as i32, rw as i32))?;
            xqlaunch!(l, "xq_silu_div", (((lr * b) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0, (dd, hcn as f32, (lr * b) as i32))?;
            xqlaunch!(l, "xq_gemm_f16", (((rw as u32) + 127) / 128, b as u32, 1), (128, 1, 1), 0,
                     (uu, &hc.up, dd, b as i32, rw as i32, lr as i32))?;
        }
        let winj = hc.winj.as_ref().map_or(0u64, |w| *w.device_ptr() as u64);
        if hc.bf {
            if fmix { fp4l_fwd::pf_hc_mix_bf(l, xo, inj, hn, uu, winj, b)?; }
            else { fp4l_fwd::hc_mix_bf(l, xo, inj, hn, uu, winj, h, hcn, b)?; }
        } else if fmix {
            xqlaunch!(l, "xq_pf_hc_mix_2560x4", (b as u32, 1, 1), (128, 1, 1), 0, (xo, inj, hn, uu, winj, b as i32))?;
        } else {
            let (mixk, mixs) = if hcn <= 4 && hc_mix4_on() { ("xq_hc_mix4", 4 * 1024 * 4) } else { ("xq_hc_mix", 4096) };
            xqlaunch!(l, mixk, (b as u32, 1, 1), (1024, 1, 1), mixs as u32,
                     (xo, inj, hn, uu, winj, h as i32, hcn as i32, b as i32))?;
        }
        if gather_x {
            tp.sp_all_gather(l, *psc.x.device_ptr() as u64, h * 2, &sp)?;
        }
        Ok(())
    }

    /// TP-SP2 (--xtp-sp-fault, probe only): flip the lowest mantissa bit of the first stream element of
    /// row `row` of the chunk's streams (resid) — a one-rank, one-owned-row divergence.
    pub(super) fn sp_fault_flip(&self, psc: &mut PrefillScratch, row: usize, li: usize) -> Result<()> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let i = row * self.cfg.hidden_size * self.cfg.hc_count.max(1);
        self.dsync()?;
        let mut v = vec![0f32; 1];
        self.dtoh_into(&psc.resid.slice(i..i + 1), &mut v)?;
        let f = f32::from_bits(v[0].to_bits() ^ 1);
        let tmp = self.dev.htod_sync_copy(&[f])?;
        xqlaunch!(l, "xq_copy_f32", (1, 1, 1), (256, 1, 1), 0, (&mut psc.resid, &tmp, 1i64, i as i64, 0i64))?;
        self.dsync()?;
        let rank = self.tp.as_ref().map_or(0, |t| t.rank);
        crate::dprintln!("[exl3-tp] rank {rank}: TP-SP2 FAULT INJECTION (--xtp-sp-fault {li}): layer {li} streams row {row} element 0 \
                  {:e} -> {:e} (one ulp, this rank's own row only)", v[0], f);
        Ok(())
    }

    /// TP-SP1: hc_post_rows over this rank's own rows (the non-deferred path: trace dumps / unfused hc);
    /// None = hc_post_rows over every row.
    pub(super) fn hc_post_rows_sp(&self, l: &Launcher, psc: &mut PrefillScratch, y_ptr: u64, c: usize,
                       sp: Option<xtp::SpRows>) -> Result<()> {
        let Some(sp) = sp else { return self.hc_post_rows(l, psc, y_ptr, c) };
        let h = self.cfg.hidden_size;
        let hcn = self.cfg.hc_count.max(1);
        let (lo, hi) = sp.own;
        let b = hi - lo;
        xqlaunch!(l, "xq_hc_inject", (((h * hcn * b) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                 (*psc.resid.device_ptr() as u64 + (lo * h * hcn * 4) as u64, y_ptr + (lo * h * 2) as u64,
                  *psc.inj.device_ptr() as u64 + (lo * hcn * 4) as u64, h as i32, hcn as i32, b as i32))
    }

    /// hc_post against the PrefillScratch residual.
    pub(super) fn hc_post_rows(&self, l: &Launcher, psc: &mut PrefillScratch, y_ptr: u64, c: usize) -> Result<()> {
        let h = self.cfg.hidden_size;
        let hcn = self.cfg.hc_count.max(1);
        xqlaunch!(l, "xq_hc_inject", (((h * hcn * c) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                 (&mut psc.resid, y_ptr, &psc.inj, h as i32, hcn as i32, c as i32))?;
        Ok(())
    }

    // S-A3-f-d night: optional per-stage trace dumps inside prefill_chunk
    // (--probe-exl3-pfxtrace). Same file names / row-0 semantics as the
    // forward_step LadderCtx dumps, so the two arms diff file-for-file.
    pub(super) fn tr_u16(&self, ld: Option<&LadderCtx>, steps: bool, li: usize,
              buf: &CudaSlice<u16>, name: &str, n: usize) -> Result<()> {
        if let Some(l) = ld {
            let go = if steps { l.want_steps(li) } else { l.want(li) };
            if go {
                self.dsync()?;
                l.dump_u16(&self.dev, buf, name, n)?;
            }
        }
        Ok(())
    }
    pub(super) fn tr_f32(&self, ld: Option<&LadderCtx>, steps: bool, li: usize,
              buf: &CudaSlice<f32>, name: &str, m: usize) -> Result<()> {
        if let Some(l) = ld {
            let go = if steps { l.want_steps(li) } else { l.want(li) };
            if go {
                self.dsync()?;
                l.dump(&self.dev, buf, name, m)?;
            }
        }
        Ok(())
    }
}
