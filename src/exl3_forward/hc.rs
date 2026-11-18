//! Hyper-connection (hc) side of the trunk: the round profiler, the int8 hc quantization helpers and the
//! `impl FwdModel` hc methods (hc_pre, hc_mixer, fused / unfused mixers, hc_post, the W4HC cross-check).
//! See `mod.rs` for the file map; the tunables that gate these paths live in `tune_hc.rs`.

use super::*;

/// S-A3-p diagnostic --exl3-round-prof: host wall per mtp_round phase — [draft issue,
/// draft drain (sync), drafts dtoh + PLE host build + verify uploads, verify issue, verify
/// drain + accept dtoh]; medians printed every 100 rounds.
pub(super) fn round_prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-round-prof")).is_ok())
}
pub(super) fn round_prof_add(v: [f64; 5]) {
    static ST: std::sync::Mutex<Vec<[f64; 5]>> = std::sync::Mutex::new(Vec::new());
    let mut g = ST.lock().unwrap();
    g.push(v);
    if g.len() % 100 == 0 {
        let med = |j: usize| { let mut c: Vec<f64> = g.iter().map(|r| r[j]).collect(); c.sort_by(|a, b| a.partial_cmp(b).unwrap()); c[c.len() / 2] };
        crate::drprintln!("ROUND_PROF n={} median ms: draft_issue {:.2} draft_drain {:.2} host_ple_upload {:.2} verify_issue {:.2} verify_drain {:.2} (sum {:.2})",
                 g.len(), med(0), med(1), med(2), med(3), med(4), (0..5).map(med).sum::<f64>());
    }
}

/// --qsa-exl3=0: the owner's A/B escape (dense attention everywhere; key planes unused).
pub(super) fn qsa_disabled() -> bool {
    crate::opts::var(crate::opt!("qsa-exl3")).as_deref() == Ok("0")
}

/// S-A3-q item 0: int8 hc mixers are the DEFAULT (owner ruling, roadmap log rev 280: flip ON,
/// escape flag stays — the reference implementation's served config). `--hc-fp16` (CLI; --hc-fp16 transport, or
/// --hc-int8=0) restores the bit-exact fp16 mixer. `--hc-int8` is still accepted (no-op).
pub fn hc_int8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("hc-fp16")).is_err()
        && crate::opts::var(crate::opt!("hc-int8")).as_deref() != Ok("0"))
}

/// Quantize a row-major [rows][cols] f16 matrix: per-row s = max|w| / 127 (floor 1e-8),
/// q = round-half-even(w / s) clamped to [-128, 127] (torch.round semantics).
fn quant_rows_i8(bits: &[u16], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut sc = vec![0f32; rows];
    for r in 0..rows {
        let row = &bits[r * cols..(r + 1) * cols];
        let amax = row.iter().map(|&b| half::f16::from_bits(b).to_f32().abs()).fold(0f32, f32::max);
        let s = amax.max(1e-8) / 127.0;
        sc[r] = s;
        for (c, &b) in row.iter().enumerate() {
            let v = (half::f16::from_bits(b).to_f32() / s).round_ties_even();
            q[r * cols + c] = v.clamp(-128.0, 127.0) as i8;
        }
    }
    (q, sc)
}

/// S-A3-p --hc-int8-check: rel-L2 of the int8 mixer output vs the fp16 twin + NaN/Inf
/// count, accumulated; printed every 200 calls and on any non-finite value.
fn hc_int8_check(r: &[u16], t: &[u16]) {
    use std::sync::Mutex;
    static ST: Mutex<(usize, f64, f64, usize)> = Mutex::new((0, 0.0, 0.0, 0));
    let (mut num, mut den, mut bad) = (0f64, 0f64, 0usize);
    for (&a, &b) in r.iter().zip(t.iter()) {
        let (x, y) = (half::f16::from_bits(a).to_f32() as f64, half::f16::from_bits(b).to_f32() as f64);
        if !y.is_finite() { bad += 1; continue; }
        num += (x - y) * (x - y);
        den += x * x;
    }
    let rel = (num / den.max(1e-30)).sqrt();
    let mut g = ST.lock().unwrap();
    g.0 += 1; g.1 = g.1.max(rel); g.2 += rel; g.3 += bad;
    if bad > 0 || g.0 % 200 == 0 {
        crate::dprintln!("HC_INT8_CHECK: calls {} rel-L2 mean {:.3e} max {:.3e} nonfinite {} (this {:.3e})",
                 g.0, g.2 / g.0 as f64, g.1, g.3, rel);
    }
}

pub(super) fn hc_quantize_i8(dev: &Arc<CudaDevice>, hc: &mut HcDev, lr: usize, rw: usize, kmajor: bool) -> Result<()> {
    let mut d = dev.dtoh_sync_copy(&hc.down)?;
    let mut u = dev.dtoh_sync_copy(&hc.up)?;
    if hc.bf {
        // the device copies are B32 (fp4l_fwd::bf16_frag): back to row-major first
        d = fp4l_fwd::bf16_unfrag(&d, lr, rw);
        u = fp4l_fwd::bf16_unfrag(&u, rw, lr);
        // FP4L --fp4l-plain exact + --hc-int8: the device copies hold stored bf16 bits; quant_rows_i8 reads f16 bits
        // (RN to f16 first: it only moves values far below a row's int8 step, which quantize to 0 either way)
        let cvt = |v: &mut Vec<u16>| for b in v.iter_mut() { *b = half::f16::from_f32(f32::from_bits((*b as u32) << 16)).to_bits(); };
        cvt(&mut d);
        cvt(&mut u);
    }
    anyhow::ensure!(d.len() == lr * rw && u.len() == rw * lr, "hc int8: mixer shape {} / {} != {lr}x{rw}", d.len(), u.len());
    let (dq, ds) = quant_rows_i8(&d, lr, rw);
    let (uq, us) = quant_rows_i8(&u, rw, lr);
    // WP11 R3: k-major relayout of up — byte (n, k) moves to ((k/16)*rw + n)*16 + k%16, so
    // phase-B step s of output n is one 16-B load and a warp's 32 outputs are contiguous.
    let kmajor = kmajor && lr % 16 == 0;
    let (up_q, up_q_rm) = if kmajor {
        let mut uk = vec![0i8; rw * lr];
        for n in 0..rw {
            for k in 0..lr {
                uk[((k / 16) * rw + n) * 16 + (k % 16)] = uq[n * lr + k];
            }
        }
        let rm = if hc_step_check_on() { Some(dev.htod_sync_copy(&uq)?) } else { None };
        (dev.htod_sync_copy(&uk)?, rm)
    } else {
        (dev.htod_sync_copy(&uq)?, None)
    };
    hc.q8 = Some(HcQ8 {
        down_q: dev.htod_sync_copy(&dq)?,
        down_s: dev.htod_sync_copy(&ds)?,
        up_q,
        up_s: dev.htod_sync_copy(&us)?,
        up_kmajor: kmajor,
        up_q_rm,
    });
    Ok(())
}

impl FwdModel {
    pub(super) fn hc_pre(&self, l: &Launcher, sc: &mut Scratch, hc: &HcDev, m: usize) -> Result<()> {
        self.hc_pre_p(l, sc, hc, m, None)
    }

    /// HC (w4): hc_pre with an optional PENDING inject (`pend` = the y pointer a deferred hc_post
    /// did not launch): the inject then rides the norm launch (xq_hc_inj_norm, bitwise ==
    /// xq_hc_inject + xq_hc_norm).
    pub(super) fn hc_pre_p(&self, l: &Launcher, sc: &mut Scratch, hc: &HcDev, m: usize, pend: Option<u64>) -> Result<()> {
        // HC --w4hc-xcheck: eager p4c-vs-new bit-diff (never inside a graph capture).
        if w4hc_xcheck() && !stream_capturing(l) {
            return self.w4hc_xcheck_run(l, sc, hc, m, pend);
        }
        self.hc_norm_p(l, sc, hc, m, pend)?;
        if self.cfg.hc_lowrank == 0 { return self.plain_publish(l, sc, m); } // EXL27B: plain residual
        // WP11 --hc-step-check: eager old-vs-new bit-diff (never inside a graph capture).
        if hc_step_check_on() && !wp11_off() && !stream_capturing(l) {
            return self.hc_step_check(l, sc, hc, m);
        }
        self.hc_mixer(l, sc, hc, m, wp11_off(), true, false)
    }

    /// The norm half of hc_pre: xq_hc_norm, or (pending inject) the fused xq_hc_inj_norm.
    fn hc_norm_p(&self, l: &Launcher, sc: &mut Scratch, hc: &HcDev, m: usize, pend: Option<u64>) -> Result<()> {
        let h = self.cfg.hidden_size;
        let hcn = self.cfg.hc_count.max(1);
        if let Some(nf) = hc.norm_f.as_ref() {
            // FP4L --fp4l-norms f32: the same two kernels' twins on the stored fp32 norm weights
            let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
            return match pend {
                Some(y_ptr) => fp4l_fwd::hc_inj_norm_f(l, dp(sc.hn.device_ptr()), dp(sc.resid.device_ptr()), y_ptr,
                                                       dp(sc.inj.device_ptr()), nf, h, hcn, m, self.cfg.rms_eps),
                None => fp4l_fwd::hc_norm_f(l, dp(sc.hn.device_ptr()), dp(sc.resid.device_ptr()), nf, h, hcn, m, self.cfg.rms_eps),
            };
        }
        match pend {
            Some(y_ptr) => xqlaunch!(l, "xq_hc_inj_norm", ((hcn * m) as u32, 1, 1), (1024, 1, 1), 0,
                                     (&mut sc.hn, &mut sc.resid, y_ptr, &sc.inj, &hc.norm,
                                      h as i32, hcn as i32, m as i32, self.cfg.rms_eps))?,
            None => xqlaunch!(l, "xq_hc_norm", ((hcn * m) as u32, 1, 1), (1024, 1, 1), 4096,
                              (&mut sc.hn, &sc.resid, &hc.norm, h as i32, hcn as i32, m as i32, self.cfg.rms_eps))?,
        }
        Ok(())
    }

    /// HC (w4): may this forward defer its hc_post injects into the next hc_pre's norm? Callers
    /// AND this with "no diagnostic reads/writes of sc.resid between the two" (dumps, poison).
    pub(super) fn hc_defer(&self) -> bool {
        w4hc_inj_on() && self.cfg.hidden_size <= 4096
    }

    /// HC (w4): hc_post, deferred when `defer` (the next hc_pre_p takes `pend`). A still-pending
    /// inject is flushed first (never two in flight).
    pub(super) fn hc_post_d(&self, l: &Launcher, sc: &mut Scratch, y_ptr: u64, m: usize, defer: bool,
                 pend: &mut Option<u64>) -> Result<()> {
        self.hc_flush(l, sc, m, pend)?;
        if defer {
            *pend = Some(y_ptr);
            Ok(())
        } else {
            self.hc_post(l, sc, y_ptr, m)
        }
    }

    /// HC (w4): launch a pending inject as the plain xq_hc_inject (before anything else touches
    /// sc.resid: the PLE boundary).
    pub(super) fn hc_flush(&self, l: &Launcher, sc: &mut Scratch, m: usize, pend: &mut Option<u64>) -> Result<()> {
        match pend.take() {
            Some(y_ptr) => self.hc_post(l, sc, y_ptr, m),
            None => Ok(()),
        }
    }

    /// HC --w4hc-xcheck: the p4c hc_pre (pending inject as xq_hc_inject, xq_hc_norm, WP11 fuse,
    /// separate mix) then the new one (xq_hc_inj_norm, xq_hc_w4 with uu written) on the same
    /// inputs — sc.resid and sc.inj (the inject's gate input, overwritten by the mix) restored in
    /// between — and bit-compare resid/hn/dd/uu/x/inj over the m live rows. The NEW results stay.
    fn w4hc_xcheck_run(&self, l: &Launcher, sc: &mut Scratch, hc: &HcDev, m: usize, pend: Option<u64>) -> Result<()> {
        use std::sync::Mutex;
        static ST: Mutex<(u64, u64)> = Mutex::new((0, 0)); // (calls, calls with a mismatch)
        let (h, hcn, lr) = (self.cfg.hidden_size, self.cfg.hc_count.max(1), self.cfg.hc_lowrank);
        let rw = h * hcn;
        self.dsync()?;
        let (r_in, j_in) = (self.dtoh(&sc.resid)?, self.dtoh(&sc.inj)?);
        // ---- p4c path
        if let Some(y) = pend { self.hc_post(l, sc, y, m)?; }
        self.hc_norm_p(l, sc, hc, m, None)?;
        self.hc_mixer(l, sc, hc, m, wp11_off(), false, true)?;
        self.dsync()?;
        let o = (self.dtoh(&sc.resid)?, self.dtoh(&sc.hn)?, self.dtoh(&sc.dd)?,
                 self.dtoh(&sc.uu)?, self.dtoh(&sc.x)?, self.dtoh(&sc.inj)?);
        // ---- restore the inputs, new path
        self.dev.htod_copy_into(r_in, &mut sc.resid)?;
        self.dev.htod_copy_into(j_in, &mut sc.inj)?;
        self.dsync()?;
        let fused_inj = pend.is_some() && self.hc_defer();
        if fused_inj {
            self.hc_norm_p(l, sc, hc, m, pend)?;
        } else {
            if let Some(y) = pend { self.hc_post(l, sc, y, m)?; }
            self.hc_norm_p(l, sc, hc, m, None)?;
        }
        let w4 = self.w4hc_takes(hc, m);
        self.hc_mixer(l, sc, hc, m, wp11_off(), true, true)?;
        self.dsync()?;
        let n = (self.dtoh(&sc.resid)?, self.dtoh(&sc.hn)?, self.dtoh(&sc.dd)?,
                 self.dtoh(&sc.uu)?, self.dtoh(&sc.x)?, self.dtoh(&sc.inj)?);
        let f32d = |a: &[f32], b: &[f32]| -> (usize, Option<usize>) {
            let mut k = (0usize, None);
            for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
                if p.to_bits() != q.to_bits() { k.0 += 1; k.1.get_or_insert(i); }
            }
            k
        };
        let u16d = |a: &[u16], b: &[u16]| -> (usize, Option<usize>) {
            let mut k = (0usize, None);
            for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
                if p != q { k.0 += 1; k.1.get_or_insert(i); }
            }
            k
        };
        let dr = f32d(&o.0[..m * rw], &n.0[..m * rw]);
        let dh = u16d(&o.1[..m * rw], &n.1[..m * rw]);
        let dd = u16d(&o.2[..m * lr], &n.2[..m * lr]);
        let du = u16d(&o.3[..m * rw], &n.3[..m * rw]);
        let dx = u16d(&o.4[..m * h], &n.4[..m * h]);
        let di = f32d(&o.5[..m * hcn], &n.5[..m * hcn]);
        let bad = dr.0 + dh.0 + dd.0 + du.0 + dx.0 + di.0;
        let nz = o.4[..m * h].iter().filter(|&&b| b & 0x7fff != 0).count();
        let mut g = ST.lock().unwrap();
        g.0 += 1;
        if bad > 0 { g.1 += 1; }
        if (bad > 0 && g.1 <= 20) || g.0 % 500 == 1 {
            crate::dprintln!("W4HC_XCHECK: call {} m={m} inj-fused {} w4 {} gates {} | mismatches resid {}/{} hn {}/{} dd {}/{} uu {}/{} \
                      x {}/{} inj {}/{} (first {:?} {:?} {:?} {:?} {:?} {:?}) | ref x nonzero {nz} | calls-with-mismatch {}/{}",
                     g.0, fused_inj as u8, w4 as u8, hc.winj.is_some() as u8, dr.0, m * rw, dh.0, m * rw, dd.0, m * lr,
                     du.0, m * rw, dx.0, m * h, di.0, m * hcn, dr.1, dh.1, dd.1, du.1, dx.1, di.1, g.1, g.0);
        }
        Ok(())
    }

    /// HC (w4) + TUNE T0c: the xq_hc_w4 grid in the current tune context — w4hc.g raised to the
    /// phase-B minimum and clamped by the load-time co-residency exactly as at load (so == w4hc_grid at
    /// the default). Never 0 while w4hc_grid > 0: the minimum and the margin do not depend on the target.
    fn w4hc_grid_now(&self) -> u32 {
        if self.w4hc_grid == 0 { return 0; }
        w4hc_default_grid(self.w4hc_cap, self.cfg.hidden_size)
    }

    /// HC (w4): does hc_mixer take xq_hc_w4 for this mixer at width m (non-legacy, w4 allowed)?
    fn w4hc_takes(&self, hc: &HcDev, m: usize) -> bool {
        // S-B9-CF-R16B FIX: m <= 8 ONLY. The R16 leg added an r16 arm here
        // (`|| (exl3_r16_on() && m <= 16)`) but never extended xq_hc_w4 to 9..16 rows
        // (CF_R16_REPORT deviations: "the rbk twin only (xq_hc_w4 NOT extended)"). The
        // arm was dead code on master (exl3_r16_on() was false at default flags — the
        // `.is_ok()` gate); R16B fixing that gate woke it: at m = 9..16 the W4 branch
        // swallowed the round BEFORE the int8 rbk_r16 twin and faulted (compute-
        // sanitizer: Invalid __shared__ write in xq_hc_w4, host frame w4hc_launch).
        // m > 8 belongs to the int8 twins below, exactly as CF_R16 intended.
        self.w4hc_grid > 0 && w4hc_fuse_on() && m >= 1 && m <= 8
            && (m >= 2 || w4hc_m1_on())
            && hc.q8.as_ref().is_some_and(|q| q.up_kmajor)
            && !matches!(crate::opts::var(crate::opt!("hc-fuse")).as_deref(), Ok("0"))
            && crate::opts::var(crate::opt!("hc-int8-check")).is_err()
    }

    /// WP11 --hc-step-check: run the pre-WP11 fuse+mix, then the WP11 one, on the same hn;
    /// bit-compare dd/uu/x/inj. The WP11 results stay in the buffers for the rest of the step.
    fn hc_step_check(&self, l: &Launcher, sc: &mut Scratch, hc: &HcDev, m: usize) -> Result<()> {
        use std::sync::Mutex;
        static ST: Mutex<(u64, u64)> = Mutex::new((0, 0)); // (calls, calls with a mismatch)
        let (h, hcn, lr) = (self.cfg.hidden_size, self.cfg.hc_count.max(1), self.cfg.hc_lowrank);
        let rw = h * hcn;
        self.hc_mixer(l, sc, hc, m, true, false, true)?;
        self.dsync()?;
        let (d0, u0, x0, i0) = (self.dtoh(&sc.dd)?, self.dtoh(&sc.uu)?,
                                self.dtoh(&sc.x)?, self.dtoh(&sc.inj)?);
        self.hc_mixer(l, sc, hc, m, false, true, true)?;
        self.dsync()?;
        let (d1, u1, x1, i1) = (self.dtoh(&sc.dd)?, self.dtoh(&sc.uu)?,
                                self.dtoh(&sc.x)?, self.dtoh(&sc.inj)?);
        // (mismatching elements, first mismatch index, max |old - new|)
        let d16 = |a: &[u16], b: &[u16]| {
            let (mut n, mut first, mut mx) = (0usize, None, 0f32);
            for (i, (&p, &q)) in a.iter().zip(b.iter()).enumerate() {
                if p != q {
                    n += 1;
                    first.get_or_insert(i);
                    mx = mx.max((f16::from_bits(p).to_f32() - f16::from_bits(q).to_f32()).abs());
                }
            }
            (n, first, mx)
        };
        let (dd, uu, xx) = (d16(&d0[..m * lr], &d1[..m * lr]), d16(&u0[..m * rw], &u1[..m * rw]),
                            d16(&x0[..m * h], &x1[..m * h]));
        let (mut ni, mut fi, mut mi) = (0usize, None, 0f32);
        for (i, (&p, &q)) in i0[..m * hcn].iter().zip(i1[..m * hcn].iter()).enumerate() {
            if p.to_bits() != q.to_bits() { ni += 1; fi.get_or_insert(i); mi = mi.max((p - q).abs()); }
        }
        let bad = dd.0 + uu.0 + xx.0 + ni;
        let mut g = ST.lock().unwrap();
        g.0 += 1;
        if bad > 0 { g.1 += 1; }
        if (bad > 0 && g.1 <= 20) || g.0 % 500 == 1 {
            crate::dprintln!("HC_STEP_CHECK: call {} m={} mismatches dd {}/{} uu {}/{} x {}/{} inj {}/{} \
                      (first dd {:?} uu {:?} x {:?} inj {:?}; max|d| {:.3e} {:.3e} {:.3e} {:.3e}) \
                      calls-with-mismatch {}/{}",
                     g.0, m, dd.0, m * lr, uu.0, m * rw, xx.0, m * h, ni, m * hcn,
                     dd.1, uu.1, xx.1, fi, dd.2, uu.2, xx.2, mi, g.1, g.0);
        }
        Ok(())
    }

    /// TP-4Z2 (`--tp-k2-prefetch`): tell the decode K2 about to be launched which weights the kernel behind it streams
    /// (the next hc mixer's int8 down/up matrices, 6.5 MB), so the K2's idle threads pull them into L2 while it waits for the
    /// peer. Decode-sized steps on the W4 mixer path only (the case `hc_mixer` takes the xq_hc_w4 launch); anything else arms
    /// nothing. A prefetch hint: no data path, no output byte.
    pub(super) fn tp_pf_hint(&self, hc: &HcDev, m: usize) {
        let Some(tp) = self.tp.as_ref() else { return };
        if m > 16 { return; }
        let Some(q) = hc.q8.as_ref() else { return };
        if !self.w4hc_takes(hc, m) { return; }
        let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
        tp.set_pf_hint([(dp(q.down_q.device_ptr()), q.down_q.len() as u32), (dp(q.up_q.device_ptr()), q.up_q.len() as u32)]);
    }

    /// The fuse + mix half of hc_pre. `legacy` = the exact pre-WP11 launches (--wp11-off=1,
    /// and the reference run of --hc-step-check). `w4` = xq_hc_w4 allowed (false = the p4c
    /// path: the reference run of --w4hc-xcheck); `want_uu` = xq_hc_w4 must also write uu
    /// (the checks compare it; the mix-fused launch otherwise keeps it in smem only).
    fn hc_mixer(&self, l: &Launcher, sc: &mut Scratch, hc: &HcDev, m: usize, legacy: bool, w4: bool,
                want_uu: bool) -> Result<()> {
        let h = self.cfg.hidden_size;
        let hcn = self.cfg.hc_count.max(1);
        let lr = self.cfg.hc_lowrank;
        let rw = h * hcn;
        // HC (w4): the regridded mixer (>= 2 CTAs/SM, row groups, gates before the barrier) with
        // the mix fused behind it — bitwise == the WP11 twin + xq_hc_mix_rg below.
        if !legacy && w4 && self.w4hc_takes(hc, m) {
            let q = hc.q8.as_ref().context("hc w4: int8 mixer missing")?;
            // FP4L (--fp4l-plain exact + --hc-int8): winj holds STORED bf16 bits, which the fused mix would read as
            // f16 — the mix then runs as the bf16 twin in hc_mix_only
            let mix = w4hc_mix_on() && !hc.bf;
            let flags = w4hc_geom_flags() | (mix as i32) | ((want_uu as i32) << 1);
            let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
            let winj = hc.winj.as_ref().map_or(0u64, |w| dp(w.device_ptr()));
            let ptrs = [dp(sc.dd.device_ptr()), dp(sc.uu.device_ptr()), dp(sc.x.device_ptr()), dp(sc.inj.device_ptr()),
                        dp(sc.hn.device_ptr()), dp(q.down_q.device_ptr()), dp(q.down_s.device_ptr()),
                        dp(q.up_q.device_ptr()), dp(q.up_s.device_ptr()), winj, dp(sc.hc_bar.device_ptr())];
            w4hc_launch(l.stream.stream, self.w4hc_grid_now(), ptrs, hcn as f32, lr, m, h, hcn, flags)?;
            if mix { return Ok(()); }
            return self.hc_mix_only(l, sc, hc, m, legacy);
        }
        // FP4L --fp4l-plain exact: the stored-bf16 mixer (no int8 copies: with --hc-int8 the int8 kernels
        // below never read down/up, except the hc-int8-check reference, which is branched there)
        if hc.bf && hc.q8.is_none() {
            return self.fx_hc_mixer(l, sc, hc, m, legacy);
        }
        // S-A3-f-f ITEM 1: the fused hc mixer — down+silu+up in ONE launch with a
        // self-resetting grid barrier between the K=10240 down GEMV and the up GEMV.
        // The old 3-launch path here was the worst kernel family in the engine
        // (97x grid-3 @ 9.3 GB/s = 68.3 ms/round; f-e budget table §1).
        // Guards: measured co-residency (occupancy API; a barrier without it can
        // deadlock), explicit escape, float4 K-step and smem-staging shape limits.
        // grid.x = clamp(cap*5/6 / m, 4, 40): the WHOLE grid (bx*m blocks) stays
        // co-resident for every m <= 16, so decode and every verify width ride ONE
        // path (batch invariance — per-output reduction order is m-independent).
        let fuse = self.hc_fuse_cap > 0
            && !matches!(crate::opts::var(crate::opt!("hc-fuse")).as_deref(), Ok("0"))
            && rw % 256 == 0 && lr > 0 && lr <= 512 && lr % 8 == 0;
        if fuse {
            let cap = (self.hc_fuse_cap * 5) / 6; // 1/6 safety margin on residency
            let bx = (cap / m as u32).clamp(4, 40);
            if bx * m as u32 <= cap {
                match hc.q8.as_ref() {
                    Some(q) if lr % 16 == 0 => {
                        // WP11 R1/R3/R4: the k-major twins run iff up_q holds the k-major
                        // relayout (load-time, gated on THEIR measured residency); `legacy`
                        // takes the old kernels on the row-major copy. Per-output arithmetic
                        // is identical in all of them (bitwise; bx never changes a bit).
                        let upk = q.up_kmajor && !legacy;
                        let up_rm = if q.up_kmajor { q.up_q_rm.as_ref() } else { Some(&q.up_q) };
                        let pf = wp11_pf_mode();
                        let capk = (self.hc_fuse_cap.min(self.wp11_cap.0) * 5) / 6;
                        let bxk = (capk / m as u32).clamp(1, 40);
                        let capr = (self.hc_fuse_cap.min(self.wp11_cap.1) * 5) / 6;
                        // S-B9-CF-R16: the 9..16-row rbk twin's own co-residency (64 KB smem),
                        // measured once. < 4 CTAs (or the flag off) and the width keeps today's
                        // m-grid path below — never a silently skipped mixer.
                        // S-B9-CF-R16B: NO pdl_on() gate any more — the twin launches through the
                        // RAW-module handle (xqlaunch_raw!) at EVERY flags value, so its > 48 KB
                        // carve-out is opted in on the raw copy even with --exl3-pdl=0. Only
                        // --exl3-r16=0 keeps m > 8 on today's m-grid path (cap16 = 0).
                        let cap16 = if m > 8 && exl3_r16_on() {
                            static C16: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
                            // capacity at the WORST carve-out (R4 on, m = 16): conservative
                            // for smaller widths, and the barrier is never oversubscribed.
                            let need = m * 4096;
                            let c = if need > 65536 { 0 } else {
                                *C16.get_or_init(|| {
                                    if !r16_hc_smem_optin() { return 0; }
                                    wp11_capacity_raw_dsmem(&self.dev, "xq_hc_fuse_i8_rbk_r16", 65536)
                                })
                            };
                            if exl3_r16_on() && c < 4 {
                                static R16NC: std::sync::Once = std::sync::Once::new();
                                R16NC.call_once(|| crate::dprintln!("EXL3-R16: xq_hc_fuse_i8_rbk_r16 not co-resident \
                                                       ({c} CTAs); m > 8 keeps the m-grid path"));
                            }
                            c
                        } else { 0 };
                        // S-B9-CF-R16B: the precise reason when the r16 twin does NOT take this
                        // width (the receipt names the entry that served instead).
                        let r16_skip = if !exl3_r16_on() { "--exl3-r16=0" }
                            else if !hc_rb_on() { "--hc-rb=0" }
                            else if !upk { "up_q not k-major (legacy load)" }
                            else if cap16 < 4 { "xq_hc_fuse_i8_rbk_r16 co-residency < 4 CTAs" }
                            else { "rb budget" };
                        if crate::opts::var(crate::opt!("hc-int8-check")).is_ok() {
                            // diagnostic (eager: --exl3-no-graph=1): fp16 twin first, then int8
                            // (FP4L stored-bf16 mixers: the reference is the exact mixer, unfused)
                            if hc.bf {
                                self.fx_hc_mixer_unfused(l, sc, hc, m)?;
                            } else {
                            xqlaunch!(l, "xq_hc_fuse", (bx, m as u32, 1), (256, 1, 1), 0,
                                     (&mut sc.dd, &mut sc.uu, &sc.hn, &hc.down, &hc.up, &sc.hc_bar,
                                      hcn as f32, lr as i32, rw as i32))?;
                            }
                            self.dsync()?;
                            let r = self.dtoh(&sc.uu)?;
                            if q.up_kmajor {
                                xqlaunch!(l, "xq_hc_fuse_i8k", (bxk, m as u32, 1), (256, 1, 1), 0,
                                         (&mut sc.dd, &mut sc.uu, &sc.hn, &q.down_q, &q.down_s, &q.up_q, &q.up_s,
                                          &sc.hc_bar, hcn as f32, lr as i32, rw as i32, pf))?;
                            } else {
                                xqlaunch!(l, "xq_hc_fuse_i8", (bx, m as u32, 1), (256, 1, 1), 0,
                                         (&mut sc.dd, &mut sc.uu, &sc.hn, &q.down_q, &q.down_s, &q.up_q, &q.up_s,
                                          &sc.hc_bar, hcn as f32, lr as i32, rw as i32))?;
                            }
                            self.dsync()?;
                            let t = self.dtoh(&sc.uu)?;
                            hc_int8_check(&r[..m * rw], &t[..m * rw]);
                        } else if upk && m >= 2 && hc_rb_on()
                                   && ((m <= 8 && capr >= 4)
                                       || (m > 8 && exl3_r16_on() && m <= 16 && cap16 >= 4)) {
                            // WP11 row-batched twin (R1 prefetch, R3 k-major, R4 hn staging).
                            // S-B9-CF-R16: at 9..16 rows the r16 twin serves the width,
                            // bit-identical (64 KB smem, M = 9..16 cases); its grid comes from
                            // ITS measured co-residency, never more CTAs than fit.
                            let r4 = if wp11_r4_on() && rw % 1024 == 0 { 16 } else { 0 };
                            // the r16 twin's dynamic carve-out: R4 staging m*4096 B,
                            // else phase B m*lr*4 B (both <= 65536 for this family).
                            let (kern, capb, dsmem) = if m > 8 {
                                ("xq_hc_fuse_i8_rbk_r16", cap16,
                                 (m * if r4 != 0 { 4096 } else { lr * 4 }) as u32)
                            } else { ("xq_hc_fuse_i8_rbk", capr, 0u32) };
                            // S-B9-REL-FIXB K7: the r16 twin's grid keeps the same 1/6
                            // residency margin as every other grid-barrier twin (cap /
                            // capk / capr all derive *5/6 before the min); a tuner pick
                            // of 48 can never reach the bare co-residency. m <= 8 (capr)
                            // is already margined at its own derivation.
                            let bxr = hc_bxr().min(if m > 8 { (capb * 5) / 6 } else { capb });
                            if m > 8 {
                                // S-B9-CF-R16B: the r16 twin's > 48 KB dynamic-smem opt-in lives
                                // on the RAW-module copy (exl3_bench.ptx is loaded TWICE; cudarc's
                                // `get_func` handle cannot be opted in from outside the crate), so
                                // the launch goes through xqlaunch_raw! — a plain cuLaunchKernel on
                                // the raw function when pdl_on() is 0 (same stream, graph-capture-
                                // safe), the identical PDL launch when --exl3-pdl=1. Same kernel,
                                // same args, bitwise-same outputs either way. m <= 8 (the rbk twin)
                                // keeps the cudarc launch, untouched.
                                r16_liveness(kern, None);
                                xqlaunch_raw!(l, kern, (bxr, 1, 1), (256, 1, 1), dsmem,
                                         (&mut sc.dd, &mut sc.uu, &sc.hn, &q.down_q, &q.down_s, &q.up_q, &q.up_s,
                                          &sc.hc_bar, hcn as f32, (lr as i32) | ((m as i32) << 16), rw as i32, pf | r4))?;
                            } else {
                                xqlaunch!(l, kern, (bxr, 1, 1), (256, 1, 1), dsmem,
                                     (&mut sc.dd, &mut sc.uu, &sc.hn, &q.down_q, &q.down_s, &q.up_q, &q.up_s,
                                      &sc.hc_bar, hcn as f32, (lr as i32) | ((m as i32) << 16), rw as i32, pf | r4))?;
                            }
                        } else if upk && bxk * m as u32 <= capk {
                            // WP11 m-grid twin (m = 1 decode/draft/seam; any width rb does not take).
                            let cvt = if wp11_cvt_i2f() { 4 } else { 0 };
                            if m > 8 { r16_liveness("xq_hc_fuse_i8k", Some(r16_skip)); }
                            xqlaunch!(l, "xq_hc_fuse_i8k", (bxk, m as u32, 1), (256, 1, 1), 0,
                                     (&mut sc.dd, &mut sc.uu, &sc.hn, &q.down_q, &q.down_s, &q.up_q, &q.up_s,
                                      &sc.hc_bar, hcn as f32, lr as i32, rw as i32, pf | cvt))?;
                        } else if m >= 2 && m <= 8 && lr <= 512 && hc_rb_on() {
                            // row-batched twin: weights streamed once for all m rows, per-row
                            // arithmetic identical (bitwise). 40 blocks, one grid row.
                            // S-B9-REL-FIXB K1 (HC audit): m <= 8 ONLY — the R16 leg widened
                            // this gate to m <= 16 under --exl3-r16, but xq_hc_fuse_i8_rb's
                            // switch covers m 1..8 (default: break = a silent no-op), so a
                            // 9..16-row legacy-load forward (or cap16 < 4) wrote NOTHING to
                            // uu/dd. The 9..16-row width falls to the m-grid twins below,
                            // exactly as pre-R16 (the same class w4hc_takes already fixed).
                            let up_rm = up_rm.context("hc int8: row-major q_up missing for the old kernel")?;
                            let bxr = hc_bxr().min(cap.max(4));
                            if m > 8 { r16_liveness("xq_hc_fuse_i8_rb", Some(r16_skip)); }
                            xqlaunch!(l, "xq_hc_fuse_i8_rb", (bxr, 1, 1), (256, 1, 1), 0,
                                     (&mut sc.dd, &mut sc.uu, &sc.hn, &q.down_q, &q.down_s, up_rm, &q.up_s,
                                      &sc.hc_bar, hcn as f32, (lr as i32) | ((m as i32) << 16), rw as i32))?;
                        } else {
                            let up_rm = up_rm.context("hc int8: row-major q_up missing for the old kernel")?;
                            if m > 8 { r16_liveness("xq_hc_fuse_i8", Some(r16_skip)); }
                            xqlaunch!(l, "xq_hc_fuse_i8", (bx, m as u32, 1), (256, 1, 1), 0,
                                     (&mut sc.dd, &mut sc.uu, &sc.hn, &q.down_q, &q.down_s, up_rm, &q.up_s,
                                      &sc.hc_bar, hcn as f32, lr as i32, rw as i32))?;
                        }
                    }
                    _ if hc.bf => self.fx_hc_mixer_unfused(l, sc, hc, m)?,
                    _ => xqlaunch!(l, "xq_hc_fuse", (bx, m as u32, 1), (256, 1, 1), 0,
                                  (&mut sc.dd, &mut sc.uu, &sc.hn, &hc.down, &hc.up, &sc.hc_bar,
                                   hcn as f32, lr as i32, rw as i32))?,
                }
            } else if hc.bf {
                // FP4L stored-bf16 mixers (with --hc-int8 copies): the f16 fallbacks must not read them
                self.fx_hc_mixer_unfused(l, sc, hc, m)?;
            } else {
                // Width beyond the co-resident budget (bench-only widths > 16):
                // fall back to the vectorized 2-kernel path for THIS model-load,
                // consistently for every m within the load.
                xqlaunch!(l, "xq_gemm_f16_v", (((lr as u32) + 7) / 8, m as u32, 1), (256, 1, 1), 0,
                         (&mut sc.dd, &hc.down, &sc.hn, m as i32, lr as i32, rw as i32))?;
                xqlaunch!(l, "xq_silu_div", (((lr * m) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                         (&mut sc.dd, hcn as f32, (lr * m) as i32))?;
                xqlaunch!(l, "xq_gemm_f16_v", (((rw as u32) + 7) / 8, m as u32, 1), (256, 1, 1), 0,
                         (&mut sc.uu, &hc.up, &sc.dd, m as i32, rw as i32, lr as i32))?;
            }
        } else if hc.bf {
            // FP4L stored-bf16 mixers (with --hc-int8 copies): the --hc-fuse=0 escape on the stored bits
            self.fx_hc_mixer_unfused(l, sc, hc, m)?;
        } else {
            // Original path (also the --hc-fuse=0 escape — consistent for all m).
            xqlaunch!(l, "xq_gemm_f16", (((lr as u32) + 127) / 128, m as u32, 1), (128, 1, 1), 0,
                     (&mut sc.dd, &hc.down, &sc.hn, m as i32, lr as i32, rw as i32))?;
            xqlaunch!(l, "xq_silu_div", (((lr * m) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                     (&mut sc.dd, hcn as f32, (lr * m) as i32))?;
            xqlaunch!(l, "xq_gemm_f16", (((rw as u32) + 127) / 128, m as u32, 1), (128, 1, 1), 0,
                     (&mut sc.uu, &hc.up, &sc.dd, m as i32, rw as i32, lr as i32))?;
        }
        self.hc_mix_only(l, sc, hc, m, legacy)
    }

    /// FP4L --fp4l-plain exact: the hc mixer on the STORED bf16 down / up (B32 layout; decode / verify widths): the
    /// fused tensor-core kernel xf_hc_fuse_b32 (ONE launch of a co-resident grid; every output a fixed sequence of
    /// its own row's values, so one path for decode and verify = batch invariance), else (kernel not co-resident,
    /// widths above 16 = bench only, or --hc-fuse=0) the unfused stored-bf16 GEMMs; then the mix twin.
    fn fx_hc_mixer(&self, l: &Launcher, sc: &mut Scratch, hc: &HcDev, m: usize, legacy: bool) -> Result<()> {
        let (h, hcn, lr) = (self.cfg.hidden_size, self.cfg.hc_count.max(1), self.cfg.hc_lowrank);
        let rw = h * hcn;
        let g = fp4l_fwd::hc_fuse_grid(self.fxc.as_ref().map_or(0, |f| f.hc_cap) * 5 / 6);
        let fuse = g > 0 && m <= 16 && !matches!(crate::opts::var(crate::opt!("hc-fuse")).as_deref(), Ok("0"))
            && rw % 1024 == 0 && lr > 0 && lr <= 480 && lr % 32 == 0;
        let ws_ok = sc.fxs.as_ref().is_some_and(|f| f.ws_h.len() >= fp4l_fwd::HCF_WS_PER_LR * lr);
        if fuse && ws_ok {
            let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
            let fxs = sc.fxs.as_ref().unwrap();
            fp4l_fwd::hc_fuse_b32(l, dp(sc.dd.device_ptr()), dp(sc.uu.device_ptr()), dp(sc.hn.device_ptr()), &hc.down, &hc.up,
                                  &fxs.ws_h, dp(sc.hc_bar.device_ptr()), hcn as f32, lr, rw, g, m)?;
        } else {
            self.fx_hc_mixer_unfused(l, sc, hc, m)?;
        }
        self.hc_mix_only(l, sc, hc, m, legacy)
    }

    /// FP4L: the unfused stored-bf16 mixer: dd = f16(down . hn), dd = silu(dd / hcn) (the trunk's
    /// xq_silu_div), uu = f16(up . dd) — the stored-bf16 GEMMs on the Scratch's hc slabs / counters.
    fn fx_hc_mixer_unfused(&self, l: &Launcher, sc: &mut Scratch, hc: &HcDev, m: usize) -> Result<()> {
        let (h, hcn, lr) = (self.cfg.hidden_size, self.cfg.hc_count.max(1), self.cfg.hc_lowrank);
        let rw = h * hcn;
        let fxs = sc.fxs.as_ref().context("FP4L stored-bf16 mixer: the Scratch has no FP4L buffers")?;
        let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
        let (hn, dd, uu) = (dp(sc.hn.device_ptr()), dp(sc.dd.device_ptr()), dp(sc.uu.device_ptr()));
        fp4l_fwd::bf_gemm(l, &hc.down, hn, dd, 0, m, rw, lr, &fxs.ws_h, &fxs.cnt_h)?;
        xqlaunch!(l, "xq_silu_div", (((lr * m) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0, (dd, hcn as f32, (lr * m) as i32))?;
        fp4l_fwd::bf_gemm(l, &hc.up, dd, uu, 0, m, lr, rw, &fxs.ws_h, &fxs.cnt_h)
    }

    /// The mix half of hc_mixer (x + the inject gates from hn and uu): WP11 R2 xq_hc_mix_rg, or
    /// xq_hc_mix4 / xq_hc_mix (legacy / R2 off).
    fn hc_mix_only(&self, l: &Launcher, sc: &mut Scratch, hc: &HcDev, m: usize, legacy: bool) -> Result<()> {
        let h = self.cfg.hidden_size;
        let hcn = self.cfg.hc_count.max(1);
        if hc.bf {
            // FP4L --fp4l-plain exact: xq_hc_mix_rg's twin on the stored bf16 inject weights (every width)
            let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
            let winj = hc.winj.as_ref().map_or(0u64, |w| dp(w.device_ptr()));
            return fp4l_fwd::hc_mix_bf(l, dp(sc.x.device_ptr()), dp(sc.inj.device_ptr()), dp(sc.hn.device_ptr()),
                                       dp(sc.uu.device_ptr()), winj, h, hcn, m);
        }
        // WP11 R2: regridded mix — each inject tree alone in its block (loads hoisted, same
        // leaves + pairing), x spread over h/1024 blocks; grid (ninj + nxb, m), block 1024. Bitwise.
        if !legacy && wp11_r2_on() && m <= 64 {
            let ninj = if hc.winj.is_some() { hcn } else { 0 };
            let gx = (ninj + (h + 1023) / 1024) as u32;
            match &hc.winj {
                Some(w) => xqlaunch!(l, "xq_hc_mix_rg", (gx, m as u32, 1), (1024, 1, 1), 0,
                                    (&mut sc.x, &mut sc.inj, &sc.hn, &sc.uu, w, h as i32, hcn as i32, m as i32))?,
                None => xqlaunch!(l, "xq_hc_mix_rg", (gx, m as u32, 1), (1024, 1, 1), 0,
                                 (&mut sc.x, &mut sc.inj, &sc.hn, &sc.uu, 0u64, h as i32, hcn as i32, m as i32))?,
            }
            return Ok(());
        }
        // xq_hc_mix4: the hc inject reductions share one tree pass (bitwise; --hc-mix4=0 = old)
        let (mixk, mixs) = if hcn <= 4 && hc_mix4_on() { ("xq_hc_mix4", 4 * 1024 * 4) } else { ("xq_hc_mix", 4096) };
        match &hc.winj {
            Some(w) => xqlaunch!(l, mixk, (m as u32, 1, 1), (1024, 1, 1), mixs as u32,
                                (&mut sc.x, &mut sc.inj, &sc.hn, &sc.uu, w, h as i32, hcn as i32, m as i32))?,
            None => xqlaunch!(l, mixk, (m as u32, 1, 1), (1024, 1, 1), mixs as u32,
                             (&mut sc.x, &mut sc.inj, &sc.hn, &sc.uu, 0u64, h as i32, hcn as i32, m as i32))?,
        }
        Ok(())
    }

    /// The token-embedding gather into an f32 stream stack: resid[b][s*h+i] = f32(embed[tok[b]][i]) — the stored
    /// bf16 table under --fp4l-plain exact (xf_embed_resid_bf), else xq_embed_resid on the f16 table. Every
    /// embedding gather goes through here (embed_bf decides which kernel may read `embed`).
    pub(super) fn embed_gather(&self, l: &Launcher, resid: u64, toks: u64, b: usize) -> Result<()> {
        let (h, hcn) = (self.cfg.hidden_size, self.cfg.hc_count.max(1));
        if self.embed_bf {
            return fp4l_fwd::embed_resid_bf(l, resid, &self.embed, toks, h, hcn, b);
        }
        xqlaunch!(l, "xq_embed_resid", (((h * hcn * b) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                 (resid, &self.embed, toks, h as i32, hcn as i32, b as i32))?;
        Ok(())
    }

    /// The MTP head's entry norms (`which` 0 = pre_fc_norm_hidden, 1 = pre_fc_norm_embedding, tiled hc x) over b
    /// rows of an f32 stream stack: xq_hc_norm, or its fp32-weight twin under --fp4l-norms f32.
    pub(super) fn mtp_pre_norm(&self, l: &Launcher, head: &DraftHead, which: u8, out: u64, x: u64, b: usize) -> Result<()> {
        let (h, hcn) = (self.cfg.hidden_size, self.cfg.hc_count.max(1));
        if let Some((fh, fe)) = head.pre_f.as_ref() {
            return fp4l_fwd::hc_norm_f(l, out, x, if which == 0 { fh } else { fe }, h, hcn, b, self.cfg.rms_eps);
        }
        let w = if which == 0 { &head.pre_fc_norm_hidden } else { &head.pre_fc_norm_embedding };
        xqlaunch!(l, "xq_hc_norm", ((hcn * b) as u32, 1, 1), (1024, 1, 1), 4096,
                 (out, x, w, h as i32, hcn as i32, b as i32, self.cfg.rms_eps))?;
        Ok(())
    }

    fn hc_post(&self, l: &Launcher, sc: &mut Scratch, y_ptr: u64, m: usize) -> Result<()> {
        let h = self.cfg.hidden_size;
        let hcn = self.cfg.hc_count.max(1);
        xqlaunch!(l, "xq_hc_inject", (((h * hcn * m) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                 (&mut sc.resid, y_ptr, &sc.inj, h as i32, hcn as i32, m as i32))?;
        Ok(())
    }

}
