//! MTP feed side: tap snapshots, `mtp_seam_prime`, `mtp_plain_prime*`, head KV row bytes, tap cross-check
//! record / verify and the WP12 cross-check helpers. MTP semantics: AGENTS 2.6. See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// WP03: element offset of `slot`'s taps window in sc.taps_keep ([w][vrows][hc*h]; vrows =
    /// MTP_MAX_K+1 unless the scratch holds the wide copy verify).
    /// Verify writes rows 0..m of the lane's OWN window; restores / re-primes read it back, so
    /// lanes whose rounds run back to back never draft from each other's hidden.
    pub(super) fn tap_base(&self, sc: &Scratch, slot: usize) -> Result<usize> {
        if a_flags().taps_shared {
            return Ok(0);
        }
        let win = sc.vrows * self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        anyhow::ensure!((slot + 1) * win <= sc.taps_keep.len(),
                        "taps_keep holds {} slot windows; slot {slot} out of range", sc.taps_keep.len() / win);
        Ok(slot * win)
    }

    /// Snapshot the CURRENT stream tap (sc.resid row 0) into row 0 of `slot`'s taps window —
    /// the serve's post-prefill prime (and every control round) uses it as the next round's
    /// draft-extend tap. WP03: copies the WHOLE hc*h row (the old launch was grid 1 x block 1,
    /// i.e. ONE float: round 1 drafted from the previous request's stale taps).
    pub fn tap_snapshot(&self, sc: &mut Scratch, slot: usize) -> Result<()> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let base = self.tap_base(sc, slot)?;
        xqlaunch!(l, "xq_copy_f32", ((((he as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.taps_keep, &sc.resid, he as i64, base as i64, 0i64))?;
        Ok(())
    }

    /// CF-P1e load-adaptive batching: `tap_snapshot` for row `row` of a batched plain step's resid (m rows, one per lane).
    pub fn tap_snapshot_row(&self, sc: &mut Scratch, slot: usize, row: usize) -> Result<()> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let base = self.tap_base(sc, slot)?;
        xqlaunch!(l, "xq_copy_f32", ((((he as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.taps_keep, &sc.resid, he as i64, base as i64, (row * he) as i64))?;
        Ok(())
    }

    /// The pre-WP03 snapshot launch, byte for byte (--a-seam=old A/B arm only).
    fn tap_snapshot_legacy(&self, sc: &mut Scratch) -> Result<()> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        xqlaunch!(l, "xq_copy_f32", (1, 1, 1), (1, 1, 1), 0,
                  (&mut sc.taps_keep, &sc.resid, he as i64, 0i64, 0i64))?;
        Ok(())
    }

    /// WP03 seam prime (replaces snapshot + `head_prime`). Snapshot the seam step's tap (the
    /// stream AFTER plen-1, which predicted the first generated token) into row 0 of the slot's
    /// window for round 1, then write the head's row plen-1 KV-only from the stream AFTER plen-2
    /// — `psc.hfill_carry`, the pairing head_fill_rows and the reference implementation use for every other prompt
    /// row (row q <- streams after q-1, token q). The old prime paired row plen-1 with the tap
    /// after plen-1 (mispaired for the whole request) and ran a full 67-kernel head pass.
    /// Zero carry when plen == 1 (the reference implementation's p = 0 carry) or --exl3-head-fill=0 (no fill ran,
    /// the carry is stale). No psc (bench token-by-token posture: no head fill at all) keeps the
    /// old head_prime. Draft-only state: greedy output is unchanged.
    pub fn mtp_seam_prime(&self, sc: &mut Scratch, psc: Option<&PrefillScratch>, head: &DraftHead,
                          last_prompt_tok: i32, plen: usize, slot: usize) -> Result<()> {
        let f = a_flags();
        let pos = plen.saturating_sub(1);
        if f.seam == 2 {
            self.tap_snapshot_legacy(sc)?;
            return self.head_prime(sc, head, last_prompt_tok, pos, slot);
        }
        self.tap_snapshot(sc, slot)?;
        let psc = match psc {
            Some(p) if f.seam == 0 => p,
            _ => return self.head_prime(sc, head, last_prompt_tok, pos, slot),
        };
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let rw = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let g = ((((rw as i64) + 255) / 256) as u32, 1, 1);
        let fill_on = crate::opts::var(crate::opt!("exl3-head-fill")).as_deref() != Ok("0");
        if plen >= 2 && fill_on && psc.hfill_carry.len() >= rw {
            xqlaunch!(l, "xq_copy_f32", g, (256, 1, 1), 0,
                      (&mut sc.resid, &psc.hfill_carry, rw as i64, 0i64, 0i64))?;
        } else {
            xqlaunch!(l, "xq_memset_f32", g, (256, 1, 1), 0, (&sc.resid, 0i64, rw as i64))?;
        }
        self.head_kv_rows(sc, head, slot, pos, &[last_prompt_tok])
    }

    /// WP03 control-round hole: a plain (control) step consumes b at p but runs no draft pass,
    /// so the head's row p (a normal round's pass 0 writes it from (tap that predicted b, b))
    /// kept a rejected draft's entry. Write it KV-only from the lane's saved tap; the caller
    /// then runs forward_step and snapshots its tap (`tap_snapshot`) as the next round's row 0.
    pub fn mtp_plain_prime(&self, sc: &mut Scratch, head: &DraftHead, slot: usize, b: i32, p: usize,
                           tap_row: usize) -> Result<()> {
        if tap_row != usize::MAX {
            anyhow::ensure!(tap_row < sc.vrows, "tap_row {tap_row} out of range");
            let l = Launcher { dev: &self.dev, stream: &self.stream };
            let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
            let base = self.tap_base(sc, slot)?;
            xqlaunch!(l, "xq_copy_f32", ((((he as i64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut sc.resid, &sc.taps_keep, he as i64, 0i64, (base + tap_row * he) as i64))?;
        }
        self.head_kv_rows(sc, head, slot, p, &[b])
    }

    /// S-C4PRIMEA: the BATCHED shared-step prime. All lanes' tap-row copies
    /// (ONE xq_gather_tap_rows launch) and the whole re-prime sequence (ONE
    /// pass at m = lanes.len() with per-row (slot,pos,tok) addressing, final
    /// write xq_attn_kv_prefill_pairs) replace the per-lane mtp_plain_prime
    /// calls of a shared plain step. Per-row bytes are bit-identical to the
    /// per-lane path (same kernels, per-row math unchanged; the one-time GPU
    /// identity xcheck at the call site proves it on the real path). The
    /// CONTROL site (single-lane rounds) keeps mtp_plain_prime untouched.
    /// lanes[i] = (slot, token b, pos p, tap_row).
    pub fn mtp_plain_prime_batched(&self, sc: &mut Scratch, head: &DraftHead,
                                   lanes: &[(usize, i32, usize, usize)]) -> Result<()> {
        let m = lanes.len();
        anyhow::ensure!(m >= 2 && m <= sc.toks.len(),
                        "mtp_plain_prime_batched: {m} lanes on a width-{} scratch", sc.toks.len());
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let hcn = self.cfg.hc_count.max(1);
        let he = hcn * self.cfg.hidden_size;
        let shared = a_flags().taps_shared;
        let win = sc.vrows * he;
        let mut meta: Vec<i32> = Vec::with_capacity(m);
        for &(slot, _b, p, tr) in lanes {
            anyhow::ensure!(tr != usize::MAX && tr < sc.vrows, "batched prime: tap_row {tr} out of range");
            anyhow::ensure!(p < self.max_pos, "batched prime: pos {p} past max_pos {}", self.max_pos);
            if !shared {
                let base = self.tap_base(sc, slot)?;
                anyhow::ensure!(base + (tr + 1) * he <= sc.taps_keep.len(),
                                "batched prime: tap window OOB (slot {slot})");
            }
            // taps_shared windows alias slot 0 (tap_base's contract); the pack keeps the row.
            meta.push(if shared { (tr as i32) << 12 } else { (slot as i32) | ((tr as i32) << 12) });
        }
        // ONE gather of all lanes' tap rows into the lane-major resid rows. meta rides
        // sc.toks stream-ordered (the gather consumes it before the token overwrite below).
        // cudarc htod_copy_into asserts src.len() == dst.len(): pad to the scratch width.
        let w_pad = sc.toks.len();
        meta.resize(w_pad, 0);
        self.dev.htod_copy_into(meta, &mut sc.toks)?;
        xqlaunch!(l, "xq_gather_tap_rows", (m as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.resid, &sc.taps_keep, &sc.toks, win as i64, he as i32, m as i32))?;
        // per-row payloads: the same device fills head_kv_rows does, but per-row slot/pos.
        let w = sc.toks.len();
        let mut toks_v: Vec<i32> = lanes.iter().map(|&(_, b, _, _)| b).collect();
        toks_v.resize(w, 0);
        self.dev.htod_copy_into(toks_v, &mut sc.toks)?;
        let mut pos_v: Vec<i32> = lanes.iter().map(|&(_, _, p, _)| p as i32).collect();
        pos_v.resize(w, 0);
        self.dev.htod_copy_into(pos_v, &mut sc.pos)?;
        let mut slotpos: Vec<i32> = vec![0; w * 2];
        let mut sids: Vec<i32> = vec![0; sc.slot_ids.len()];
        for (i, &(sl, _, p, _)) in lanes.iter().enumerate() {
            slotpos[i * 2] = sl as i32;
            slotpos[i * 2 + 1] = p as i32;
            if i < sids.len() { sids[i] = sl as i32; }
        }
        self.dev.htod_copy_into(slotpos, &mut sc.slots)?;
        self.dev.htod_copy_into(sids, &mut sc.slot_ids)?;
        self.head_reprime_launches_inner(&l, sc, head, m, lanes[0].0, lanes[0].2,
                                         Some(*sc.slots.device_ptr() as u64))
    }

    /// S-C4PRIMEA identity gate: compact grab of the raw head-KV bytes of the given
    /// (slot,pos) rows (all 2*nkv planes, rb bytes each) via xq_copy_u16 launches on
    /// the compute stream, then one host readback. Used by the one-time-per-boot
    /// legacy-vs-batched xcheck at the call site.
    pub fn head_kv_row_bytes(&self, sc: &mut Scratch, head: &DraftHead,
                             lanes: &[(usize, i32, usize, usize)]) -> Result<Vec<u8>> {
        let _ = sc;
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let nkv = self.cfg.num_kv_heads;
        let seg = 2 * nkv;
        let rb = kv_rowbytes(self.kv_fmt, self.cfg.head_dim);
        anyhow::ensure!(rb % 2 == 0, "kv row grab: odd row bytes ({rb})");
        let total = lanes.len() * seg * rb;
        let mut d = unsafe { self.dev.alloc::<u8>(total)? };
        let mut off = 0usize;
        for &(sl, _b, p, _tr) in lanes {
            anyhow::ensure!(p < self.max_pos, "kv row grab: pos {p} past max_pos");
            for r in 0..seg {
                let so = (sl * seg * self.max_pos + r * self.max_pos + p) * rb;
                anyhow::ensure!(so % 2 == 0, "kv row grab: odd source offset");
                xqlaunch!(l, "xq_copy_u16", (((rb / 2 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                          (&mut d, &head.kv, (rb / 2) as i64, (off / 2) as i64, (so / 2) as i64))?;
                off += rb;
            }
        }
        self.dsync()?;
        Ok(self.dtoh(&d)?)
    }

    /// --a-xcheck: host copy of `slot`'s taps row `tap_row` (bits), for tap_xcheck_verify.
    fn tap_row_bits(&self, sc: &Scratch, slot: usize, tap_row: usize) -> Result<Vec<u32>> {
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let off = self.tap_base(sc, slot)? + tap_row * he;
        self.dsync()?;
        let all = self.dtoh(&sc.taps_keep)?;
        anyhow::ensure!(off + he <= all.len(), "tap xcheck: row out of range");
        Ok(all[off..off + he].iter().map(|x| x.to_bits()).collect())
    }

    /// --a-xcheck: remember the tap `slot`'s next round restores (usize::MAX = none: the old
    /// control-round marker, the tap then lives in sc.resid).
    pub fn tap_xcheck_record(&self, sc: &Scratch, slot: usize, tap_row: usize) -> Result<()> {
        if tap_row == usize::MAX {
            TAP_XCHECK.lock().unwrap().remove(&slot);
            return Ok(());
        }
        let bits = self.tap_row_bits(sc, slot, tap_row)?;
        TAP_XCHECK.lock().unwrap().insert(slot, (tap_row, bits));
        Ok(())
    }

    /// --a-xcheck: at a round's start, bit-compare the tap it is about to restore with the one
    /// recorded when the lane's previous round (or prime / control step) ended.
    pub fn tap_xcheck_verify(&self, sc: &Scratch, slot: usize, tap_row: usize) -> Result<()> {
        let rec = TAP_XCHECK.lock().unwrap().get(&slot).cloned();
        let Some((row, want)) = rec else { return Ok(()) };
        let n = TAP_XCHECK_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let bad = if row != tap_row || tap_row == usize::MAX {
            crate::deprintln!("[a-xcheck] slot {slot}: tap row {row} recorded, round restores row {tap_row}");
            true
        } else {
            let got = self.tap_row_bits(sc, slot, tap_row)?;
            let diff: Vec<usize> = (0..want.len()).filter(|&i| want[i] != got[i]).collect();
            if let Some(&i) = diff.first() {
                crate::deprintln!("[a-xcheck] slot {slot}: tap row {row} CLOBBERED since its round — {} of {} floats \
                           differ, first at {i}: {} (recorded) vs {} (now)", diff.len(), want.len(),
                          f32::from_bits(want[i]), f32::from_bits(got[i]));
            }
            !diff.is_empty()
        };
        let nb = TAP_XCHECK_BAD.fetch_add(bad as usize, std::sync::atomic::Ordering::Relaxed) + bad as usize;
        if n % 256 == 0 {
            crate::deprintln!("[a-xcheck] taps: {n} round starts checked, {nb} clobbered");
        }
        Ok(())
    }

    /// S-A3-f-d BUG-5 FIX (tap half): the serve's chunked prefill runs through the
    /// PrefillScratch, so `sc.resid` never sees the prompt — a plain `tap_snapshot`
    /// after `prefill_chunk` snapshots UNINITIALIZED scratch (alloc_zeros does not
    /// zero, §2.2) and round 1's draft-extend tap is garbage. The bench arm is
    /// lossless precisely because its token-by-token prefill writes `sc`. The
    /// prefill's PRE-COLLAPSE residual (`psc.resid`, the "precollapse" stage,
    /// row-aligned ≤1e-3 vs forward_step per pfxtrace) IS the tap that predicted
    /// the first bonus — copy its last row into taps_keep[0] directly.
    pub fn tap_snapshot_from_prefill(&self, psc: &PrefillScratch, sc: &mut Scratch,
                                     last_row: usize) -> Result<()> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        xqlaunch!(l, "xq_copy_f32", (1, 1, 1), (1, 1, 1), 0,
                  (&mut sc.taps_keep, &psc.resid, he as i64, 0i64,
                   (last_row * he) as i64))?;
        Ok(())
    }

    /// Feed the head one committed token with the current stream tap (no logits)
    /// — the serve's post-prefill M1 prime.
    pub fn head_prime(&self, sc: &mut Scratch, head: &DraftHead, tok: i32,
                      pos: usize, slot: usize) -> Result<()> {
        self.head_forward(sc, head, &[tok], &[pos], &[slot], false, false, false)
            .map(|_| ())
    }

    /// WP12 XCHECK: is the compute stream recording a graph right now? (the checks sync
    /// and read back, so they must never run inside a capture).
    pub(super) fn wp12_capturing(&self) -> bool {
        use cudarc::driver::sys;
        let mut st = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let r = unsafe { sys::cuStreamIsCapturing(self.stream.stream, &mut st) };
        r != sys::CUresult::CUDA_SUCCESS || st != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    }

    /// WP12 XCHECK (a|b): the old two xq_gemm_f16_v launches into twins, bit-diffed
    /// against the one-launch outputs already in sc.a_out / sc.b_out.
    pub(super) fn wp12_xcheck_ab(&self, l: &Launcher, sc: &Scratch, g: &GdnLayer, m: usize, site: &str) -> Result<()> {
        let h = self.tc.hidden_size;
        let mut a2 = self.dev.alloc_zeros::<u16>(sc.a_out.len())?;
        let mut b2 = self.dev.alloc_zeros::<u16>(sc.b_out.len())?;
        self.dsync()?;
        xqlaunch!(l, "xq_gemm_f16_v", (6u32, m as u32, 1), (256, 1, 1), 0,
                 (&mut a2, &g.a_w, &sc.x, m as i32, g.a_w.len() as i32 / h as i32, h as i32))?;
        xqlaunch!(l, "xq_gemm_f16_v", (6u32, m as u32, 1), (256, 1, 1), 0,
                 (&mut b2, &g.b_w, &sc.x, m as i32, g.a_w.len() as i32 / h as i32, h as i32))?;
        self.dsync()?;
        let n = m * (g.a_w.len() / h as usize);
        let (an, ao) = (self.dtoh(&sc.a_out)?, self.dtoh(&a2)?);
        let (bn, bo) = (self.dtoh(&sc.b_out)?, self.dtoh(&b2)?);
        wp12_report_f16(&format!("{site} a"), &an[..n], &ao[..n]);
        wp12_report_f16(&format!("{site} b"), &bn[..n], &bo[..n]);
        Ok(())
    }

    /// WP12 XCHECK (verify conv): the old path on twins — live ring row -> shadow copy,
    /// the saved raw rows -> twin qkv, in-place xq_conv1d_chunk — vs the nostore output.
    pub(super) fn wp12_xcheck_conv(&self, l: &Launcher, sc: &Scratch, g: &GdnLayer, gdn_i: usize,
                        slot: usize, m: usize) -> Result<()> {
        let cfg = &self.tc;
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let ck = cfg.conv_kernel;
        let c_e = conv_dim * ck;
        let n = m * conv_dim;
        let mut q2 = self.dev.alloc_zeros::<u16>(n)?;
        let mut sh2 = self.dev.alloc_zeros::<f32>(c_e)?;
        self.dsync()?;
        xqlaunch!(l, "xq_copy_f32", (((c_e + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sh2, &self.conv_state[gdn_i], c_e as i64, 0i64, (slot * c_e) as i64))?;
        xqlaunch!(l, "xq_copy_u16", (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut q2, &sc.qkv_save, n as i64, 0i64, (gdn_i * SAVE_PLANE_ROWS * conv_dim) as i64))?;
        xqlaunch!(l, "xq_conv1d_chunk", ((conv_dim as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                  (&mut q2, &mut sh2, &g.conv_w, conv_dim as i32, ck as i32, m as i32, 0i32, conv_dim as i32))?;
        self.dsync()?;
        let new = self.dtoh(&sc.qkv)?;
        let old = self.dtoh(&q2)?;
        wp12_report_f16("verify conv", &new[..n], &old[..n]);
        Ok(())
    }

    /// WP12 XCHECK (verify GDN step): the old xq_gdn_step_chunk (nostore: the live state is
    /// only read) into twin core rows + a twin ring; diff core and every WRITTEN ring float
    /// (delta[vd] | krow[kd] | g — the 3 pad floats of each XQ_GDN_RS record are never written).
    pub(super) fn wp12_xcheck_verify_step(&self, l: &Launcher, sc: &Scratch, g: &GdnLayer, gdn_i: usize,
                               slot_c: i32, m: usize) -> Result<()> {
        let cfg = &self.tc;
        let (kd, vd, nh, nk) = (cfg.lin_k_dim, cfg.lin_v_dim, cfg.lin_num_v_heads, cfg.lin_num_k_heads);
        let nr = m * nh * GDN_RS;
        let mut core2 = self.dev.alloc_zeros::<u16>(sc.core.len())?;
        let ring2 = self.dev.alloc_zeros::<f32>(nr)?;
        self.dsync()?;
        let smem = (kd * 33 + kd + 32 * 3 + kd * 2) * 4;
        xqlaunch!(l, "xq_gdn_step_chunk", ((nh * (vd / 32)) as u32, 1, 1), (kd as u32, 1, 1), smem as u32,
                 (&mut core2, &sc.qkv, (*self.s_state[gdn_i].device_ptr()) as u64,
                  &sc.b_out, &sc.a_out, (nh | (nk << 16)) as i32, kd as i32, vd as i32,
                  &g.a_log, &g.dt_bias, slot_c, (*ring2.device_ptr()) as u64))?;
        self.dsync()?;
        let nc = m * nh * vd;
        let cn = self.dtoh(&sc.core)?;
        let co = self.dtoh(&core2)?;
        wp12_report_f16("verify gdn core", &cn[..nc], &co[..nc]);
        let off = gdn_i * sc.vrows * nh * GDN_RS;
        let mut rn = vec![0f32; nr];
        self.dtoh_into(&sc.gdn_ring.slice(off..off + nr), &mut rn)?;
        let ro = self.dtoh(&ring2)?;
        let live = vd + kd + 1;
        let pick = |v: &[f32]| v.iter().enumerate().filter(|(j, _)| j % GDN_RS < live)
            .map(|(_, x)| *x).collect::<Vec<f32>>();
        wp12_report("verify gdn ring", &pick(&rn), &pick(&ro[..nr]));
        Ok(())
    }

    /// WP12 XCHECK (decode step): a copy of the whole live state plane, taken BEFORE the
    /// new kernel runs (the old kernel replays the step on it afterwards).
    pub(super) fn wp12_state_twin(&self, l: &Launcher, gdn_i: usize) -> Result<CudaSlice<f32>> {
        let n = self.s_state[gdn_i].len();
        let mut st2 = self.dev.alloc_zeros::<f32>(n)?;
        self.dsync()?;
        xqlaunch!(l, "xq_copy_f32", (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut st2, &self.s_state[gdn_i], n as i64, 0i64, 0i64))?;
        Ok(st2)
    }

    /// WP12 XCHECK (decode step): the old xq_gdn_step on the pre-step twin plane; diff the
    /// core rows and the whole post-step plane against the live one.
    pub(super) fn wp12_xcheck_decode_step(&self, l: &Launcher, sc: &Scratch, g: &GdnLayer, gdn_i: usize,
                               m: usize, st2: CudaSlice<f32>) -> Result<()> {
        let cfg = &self.tc;
        let (kd, vd, nh, nk) = (cfg.lin_k_dim, cfg.lin_v_dim, cfg.lin_num_v_heads, cfg.lin_num_k_heads);
        let mut core2 = self.dev.alloc_zeros::<u16>(sc.core.len())?;
        self.dsync()?;
        let smem = (kd * 33 + kd + 32 * 3 + kd * 2) * 4;
        xqlaunch!(l, "xq_gdn_step", ((m * nh * (vd / 32)) as u32, 1, 1), (kd as u32, 1, 1), smem as u32,
                 (&mut core2, &sc.qkv, (*st2.device_ptr()) as u64,
                  &sc.b_out, &sc.a_out, nh as i32, nk as i32, kd as i32, vd as i32,
                  &g.a_log, &g.dt_bias, &sc.slots))?;
        self.dsync()?;
        let nc = m * nh * vd;
        let cn = self.dtoh(&sc.core)?;
        let co = self.dtoh(&core2)?;
        wp12_report_f16("step gdn core", &cn[..nc], &co[..nc]);
        let sn = self.dtoh(&self.s_state[gdn_i])?;
        let so = self.dtoh(&st2)?;
        wp12_report("step gdn state", &sn, &so);
        Ok(())
    }

}
