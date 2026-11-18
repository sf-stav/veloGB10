//! The verify round: `verify_shadow`, the large `verify_kernels`, `verify_commit*` and the verify_block variants.
//! Batch-invariant: column 0 of an N-wide verify must equal a decode (AGENTS 2.4); GDN decode and verify share one
//! device function and reduction order (AGENTS 2.8). See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// Shadow-state staging: live -> shadow for every layer's GDN/conv state and
    /// the PLE ring (compute-stream copy kernels). Shadows hold ONE slot
    /// (index 0); the verify launches chain kernels with slot=0.
    pub(super) fn verify_shadow(&self, l: &Launcher, sc: &mut Scratch, slot: usize) -> Result<()> {
        let s_e = (self.tc.lin_num_v_heads * self.tc.lin_k_dim * self.tc.lin_v_dim) as i64;
        let c_e = ((self.tc.key_dim() * 2 + self.tc.value_dim()) * self.tc.conv_kernel) as i64;
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        // S-A3-m: the verify GDN chunk reads the LIVE S row directly (nostore), so
        // the S shadow is only materialized for the MTP_DIAG dump below.
        let dump_s = crate::opts::var(crate::opt!("mtp-dump")).map_or(false, |d| !d.is_empty());
        for li in 0..self.s_state.len() {
            let s = (slot as i64) * s_e;
            if dump_s && li == 0 {
                xqlaunch!(l, "xq_copy_f32", grid(s_e), (256, 1, 1), 0,
                          (&mut sc.s_state_sh[li], &self.s_state[li], s_e, 0i64, s))?;
            }
            let c = (slot as i64) * c_e;
            // WP12 conv: the verify conv reads the LIVE ring row (xq_conv1d_chunk_ns) — the
            // shadow is only materialized for the MTP_DIAG dump (or with --wp12-off=conv).
            if !wp12_on(WP12_CONV) || dump_s {
            xqlaunch!(l, "xq_copy_f32", grid(c_e), (256, 1, 1), 0,
                      (&mut sc.conv_sh[li], &self.conv_state[li], c_e, 0i64, c))?;
            }
            // MTP_DIAG: the live GDN/conv state row this round starts from
            // (shadow row 0 == live row `slot` post-copy) — layer 0 only.
            if li == 0 {
                if let Ok(d) = crate::opts::var(crate::opt!("mtp-dump")) {
                    if !d.is_empty() {
                        self.dsync()?;
                        let host = self.dtoh(&sc.s_state_sh[0])?;
                        let mut f = std::fs::File::create(format!("{}/g0sin_s000.f32", d))?;
                        for &v in &host { f.write_all(&v.to_le_bytes())?; }
                        let hostc = self.dtoh(&sc.conv_sh[0])?;
                        let mut f2 = std::fs::File::create(format!("{}/g0cin_s000.f32", d))?;
                        for &v in &hostc { f2.write_all(&v.to_le_bytes())?; }
                    }
                }
            }
        }
        let pr = (slot as i64) * (10240 * 9) as i64;
        xqlaunch!(l, "xq_copy_u16", grid((10240 * 9) as i64), (256, 1, 1), 0,
                  (&mut sc.ple_ring_sh, &self.ple_state, (10240 * 9) as i64, 0i64, pr))?;
        Ok(())
    }

    /// The chained trunk forward at m = k+1 rows of ONE slot: embed -> PLE (conv
    /// chunk on the shadow ring) -> layers (GDN conv/state chunks + attn group)
    /// -> final mixer -> FULL lm_head -> argmax per row. Launches ONLY (uploads
    /// and shadows happen in the caller — graph-capturable shape).
    /// Off-dequant contract: every chain goes through the m<=16 decode branch of
    /// exl3_chain (never exl3_chain_rb) — the reconstruct/Lt prefill path is
    /// unreachable here BY CONSTRUCTION; m is asserted <= MTP_MAX_K+1 <= 16.
    pub(super) fn verify_kernels(&self, l: &Launcher, sc: &mut Scratch, m: usize, qsa: bool, slot: usize,
                      seg_k: usize) -> Result<()> {
        anyhow::ensure!(m >= 2 && m <= 16 && (m <= sc.vrows || seg_k > 0),
                        "verify width {m} out of range (seg_k {seg_k}, scratch verify rows {})", sc.vrows);
        if seg_k > 0 { anyhow::ensure!(seg_k <= PACK_SEG_MAX && m <= 16, "packed: k_seg {seg_k} m {m}"); }
        let cfg = &self.tc; // TP-B: the trunk's per-rank geometry
        let h = cfg.hidden_size;
        let hcn = cfg.hc_count.max(1);
        let kd = cfg.lin_k_dim;
        let vd = cfg.lin_v_dim;
        let nh = cfg.lin_num_v_heads;
        let nk = cfg.lin_num_k_heads;
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let ck = cfg.conv_kernel;
        let eps = cfg.rms_eps;
        // WP27: graph class of this body + the early PLE prefix (see ple_prefix).
        sc.wp27_cls = crate::exl3_wp27::CLS_VERIFY;
        let ple_early = self.wp27_ple_early();
        let mut wp27_ple = crate::exl3_wp27::Cap::none();
        let mut wp27_ple_tail = crate::exl3_wp27::Tail::default();

        self.embed_gather(l, *sc.resid.device_ptr() as u64, *sc.toks.device_ptr() as u64, m)?;

        // MTP_DIAG: verify-vs-plain col-0 diff (--mtp-dump=<dir>; row 0 for the
        // [m,hc] buffers is the same-buffer-echo convention — buffered rows carry
        // true verification data; unaffected otherwise).
        let bmtd = crate::opts::var(crate::opt!("mtp-dump")).ok().map(|d| std::path::PathBuf::from(d));
        let dump_f = |d: &std::path::PathBuf, buf: &CudaSlice<f32>, name: &str, r: usize| -> Result<()> {
            let host = self.dtoh(buf)?;
            let path = format!("{}/{}_s{:03}.f32", d.display(), name, 0);
            let mut f = std::fs::File::create(&path)?;
            let uid = buf.len() / m;
            for &v in &host[(r * uid)..((r + 1) * uid).min(host.len())] {
                f.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        };
        let dump_u = |d: &std::path::PathBuf, buf: &CudaSlice<u16>, name: &str, r: usize| -> Result<()> {
            let host = self.dtoh(buf)?;
            let path = format!("{}/{}_s{:03}.f32", d.display(), name, 0);
            let mut f = std::fs::File::create(&path)?;
            let uid = buf.len() / m;
            for &b in &host[(r * uid)..((r + 1) * uid).min(host.len())] {
                f.write_all(&f16::from_bits(b).to_f32().to_le_bytes())?;
            }
            Ok(())
        };
        let dump_i32 = |d: &std::path::PathBuf, buf: &CudaSlice<i32>, name: &str, r: usize| -> Result<()> {
            let host = self.dtoh(buf)?;
            let path = format!("{}/{}_s{:03}.f32", d.display(), name, 0);
            let mut f = std::fs::File::create(&path)?;
            let uid = buf.len() / m;
            for &b in &host[(r * uid)..((r + 1) * uid).min(host.len())] {
                f.write_all(&b.to_le_bytes())?;
            }
            Ok(())
        };

        // ---- PLE (S-A3-f-d Item 1b FIX): the WHOLE device chain now runs at the
        // layer-1 boundary inside the loop below — the same point as the decode
        // path (ple_kernels at li==1) and prefill (forward_rows li==1). It
        // previously ran pre-loop (before layer 0), so verify col-0 saw the PLE
        // delta one layer early and its logits diverged from the plain step
        // (argmax 1156 vs 1144) — the trunk-perturbation defect. ple_emb itself
        // is rebuilt per round by mtp_round (host n-gram hash + upload above).

        let rdim = cfg.rotary_dim;
        let cs_stride = (2 * rdim) as i32;
        { let _ = cs_stride; // VIS-2: the mrope-aware gather (text-only slots: cos_tab[pos], unchanged)
        let slots_src = SlotSrc::Pairs(*sc.slots.device_ptr() as u64);
        self.cos_gather(&l, &mut sc.cos, *sc.pos.device_ptr() as u64, slots_src, m)?; }

        // MTP_DIAG: col-0 diagnostic dumps (verify-vs-plain diff; same-buffer col 0,
        // row-0 for the [m,hc] buffers — the bench-diag harness prearranges the
        // same-buffer-echo so buffered rows hold true verification data).
        if let Some(d) = bmtd.as_ref() {
            dump_f(&d, &sc.resid, "stagingCTX", 0)?;
            dump_f(&d, &sc.cos, "stagingCOS", 0)?;
            dump_i32(&d, &sc.pos, "stagingPOS", 0)?;
            dump_i32(&d, &sc.slots, "stagingSLOTS", 0)?;
            dump_i32(&d, &sc.toks, "stagingTOKS", 0)?;
        }

        let nchunk = vd / 32;
        let smem = (kd * 33 + kd + 32 * 3 + kd * 2) * 4;
        // WP12 XCHECK: eager only (never inside a capture — it syncs and reads back)
        let wp12_xchk = wp12_xcheck() && !self.wp12_capturing();
        let mut gdn_i = 0usize;
        let mut att_i = 0usize;
        // HC (w4): each hc_post's inject rides the next hc_pre's norm (xq_hc_inj_norm) unless a
        // diagnostic dump reads sc.resid in between; flushed before the PLE chain (it reads and
        // writes sc.resid at the layer-1 boundary).
        let hc_def = self.hc_defer() && bmtd.is_none();
        let mut hc_pend: Option<u64> = None;
        for (lk, layer) in self.layers.iter().enumerate() {
            if lk == 1 && self.ple.is_some() { self.hc_flush(l, sc, m, &mut hc_pend)?; }
            // S-A3-f-d Item 1b FIX: the full PLE device chain at the layer-1
            // boundary — same point as decode (ple_kernels at li==1) and prefill;
            // shadow ring, live ring commits later. ple_qn depends on sc.resid, so
            // the whole chain must see the layer-1-entry residual.
            if lk == 1 && self.ple.is_some() && crate::opts::var(crate::opt!("exl3-no-ple")).is_err() {
                let ple = self.ple.as_ref().unwrap();
                let scale = 1.0f32 / (2560f32).sqrt();
                // S-A3-m: row-batched bit-exact twin (weights read once for all m rows)
                // key/value projections + key norm (ple_prefix: the same launches in the same
                // order) — here, or already launched early beside layer 0's MoE window (WP27).
                if ple_early {
                    wp27_ple.join(&wp27_ple_tail)?; // WP27: the early PLE prefix branch joins here
                } else {
                    self.ple_prefix(l, sc, m)?;
                }
                if let Some(d) = bmtd.as_ref() { dump_u(&d, &sc.ple_emb, "stagingPLE", 0)?; }
                let _ = eps;
                let dpp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
                self.ple_norm(l, ple, 1, dpp(sc.resid.device_ptr()), dpp(sc.ple_qn.device_ptr()), m)?;
                xqlaunch!(l, "xq_ple_gate", (m as u32 * 4, 1, 1), (256, 1, 1), 0,
                         (&sc.ple_qn, &sc.ple_kn, &sc.ple_val, &mut sc.ple_gated, scale))?;
                self.ple_norm(l, ple, 2, dpp(sc.ple_gated.device_ptr()), dpp(sc.ple_normed.device_ptr()), m)?;
                if let Some(cf) = ple.conv_f.as_ref() {
                    // FP4L --fp4l-plain exact: the conv twins on the fp32-widened stored taps
                    let (r, g, n, st) = (dpp(sc.resid.device_ptr()), dpp(sc.ple_gated.device_ptr()), dpp(sc.ple_normed.device_ptr()),
                                         dpp(sc.ple_ring_sh.device_ptr()));
                    if seg_k > 0 { fp4l_fwd::ple_conv_chunk_seg_f(l, r, g, n, st, cf, dpp(sc.seg.device_ptr()), seg_k)?; }
                    else { fp4l_fwd::ple_conv_chunk_f(l, r, g, n, st, cf, 0, m)?; }
                } else if seg_k > 0 {
                    // PACK1 LEG 1: segmented PLE conv on the per-segment shadow planes.
                    xqlaunch!(l, "xq_ple_conv_chunk_seg", ((10240u32 + 255) / 256, seg_k as u32, 1), (256, 1, 1), 0,
                             (&mut sc.resid, &sc.ple_gated, &sc.ple_normed, &mut sc.ple_ring_sh,
                              &ple.conv_w, &sc.seg))?;
                } else {
                xqlaunch!(l, "xq_ple_conv_chunk", ((10240u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                         (&mut sc.resid, &sc.ple_gated, &sc.ple_normed, &mut sc.ple_ring_sh,
                          &ple.conv_w, 0i32, m as i32))?;
                }
            }
            // MTP_DIAG: per-layer col-0 dump of the pre-mixer hidden state.
            if let Some(d) = bmtd.as_ref() { dump_u(&d, &sc.x, &format!("xpre{lk}"), 0)?; }
            self.hc_pre_p(l, sc, &layer.hc_attn, m, hc_pend.take())?;
            self.tp_pf_hint(&layer.hc_mlp, m);   // TP-4Z2: the mixer-out reduce's K2 prefetches the mlp-site hc weights (verify graphs)
            match &layer.mixer {
                Mixer::Gdn(g) => {
                    // WP27 GDN fork: the in_z chain (its own scratch) runs beside saves -> conv
                    // -> scan (it starts after in_qkv, off its bandwidth window) and joins before
                    // the gate (zbuf); a|b + its saves run beside in_qkv and join before the scan
                    // (b_out/a_out).
                    let zdup = crate::exl3_wp27::fork_on(crate::exl3_wp27::FORK_GDN) && sc.wp27_z_fits(&g.in_z, m);
                    let mut wp27 = if zdup {
                        crate::exl3_wp27::Cap::open(l.stream.stream, sc.wp27_cls, crate::exl3_wp27::FORK_GDN)
                    } else { crate::exl3_wp27::Cap::none() };
                    exl3_chain(l, &g.in_qkv, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.qkv, m, &sc.chain_ws)?;
                    let wp27_q = wp27.mark()?; // in_qkv tail: the z chain and the saves/conv hang off it
                    wp27.side_begin()?;
                    if zdup {
                        exl3_chain(l, &g.in_z, &sc.x, &mut sc.z_xh, &mut sc.z_yraw, &mut sc.zbuf, m, &sc.z_ws)?;
                    } else {
                    exl3_chain(l, &g.in_z, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.zbuf, m, &sc.chain_ws)?;
                    }
                    wp27.side_end()?;
                    let wp27_z = wp27.cut()?; // a|b restarts from the fork base
                    // a/b projections (row-batched GEMV — same as the decode path)
                    // WP12 ab: both in ONE launch (bitwise per output; --wp12-off=ab = two).
                    if let Some(ab) = g.ab_fx.as_deref() {
                        fp4l_fwd::fx_ab(l, ab, &sc.x, &mut sc.a_out, &mut sc.b_out, m)?; // FP4L: exact MXFP8
                    } else if wp12_on(WP12_AB) {
                        xqlaunch!(l, "xq_gemm_f16_v_ab", (12u32, m as u32, 1), (256, 1, 1), 0,
                                 (&mut sc.a_out, &mut sc.b_out, &g.a_w, &g.b_w, &sc.x, m as i32, nh as i32, h as i32))?;
                        if wp12_xchk { self.wp12_xcheck_ab(l, sc, g, m, "verify a|b")?; }
                    } else {
                    xqlaunch!(l, "xq_gemm_f16_v", (6u32, m as u32, 1), (256, 1, 1), 0,
                             (&mut sc.a_out, &g.a_w, &sc.x, m as i32, nh as i32, h as i32))?;
                    xqlaunch!(l, "xq_gemm_f16_v", (6u32, m as u32, 1), (256, 1, 1), 0,
                             (&mut sc.b_out, &g.b_w, &sc.x, m as i32, nh as i32, h as i32))?;
                    }
                    // WP12 diet: a_save/b_save/qkv_post are read ONLY by the xq_gdn_commit
                    // replay, which the ring commit (mode 1) never runs — skip them then.
                    let replay_saves = !(wp12_on(WP12_DIET) && gdn_commit_mode(kd, vd) == 1);
                    if replay_saves {
                    // S-A3-f-d Item 1b FIX-2: save this layer's gate rows — the
                    // shared a_out/b_out hold the LAST layer's at commit time.
                    xqlaunch!(l, "xq_copy_u16", (((m * nh) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                             (&mut sc.a_save, &sc.a_out, (m * nh) as i64,
                              (gdn_i * SAVE_PLANE_ROWS * nh) as i64, 0i64))?;
                    xqlaunch!(l, "xq_copy_u16", (((m * nh) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                             (&mut sc.b_save, &sc.b_out, (m * nh) as i64,
                              (gdn_i * SAVE_PLANE_ROWS * nh) as i64, 0i64))?;
                    }
                    let wp27_ab = wp27.cut_to(&wp27_q)?; // WP27: resume from the in_qkv tail
                    // conv over the group: rows sequential in-kernel (shadow state)
                    // S-A3-f-d Item 1b FIX: save the raw (pre-conv) rows first —
                    // the chunk overwrites qkv in place, and the conv commit must
                    // replay the state window with the RAW projections.
                    xqlaunch!(l, "xq_copy_u16", (((m * conv_dim) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                             (&mut sc.qkv_save, &sc.qkv, (m * conv_dim) as i64,
                              (gdn_i * SAVE_PLANE_ROWS * conv_dim) as i64, 0i64))?;
                    if lk == 0 { if let Some(d) = bmtd.as_ref() { dump_u(&d, &sc.qkv_save, "l0qkvsave", 0)?; } }
                    if wp12_on(WP12_CONV) {
                        // WP12 conv: nostore chunk on the LIVE ring row (read-only here; the
                        // conv commit replays the accepted raw rows) — no shadow copy.
                        if seg_k > 0 {
                        // PACK1 LEG 1: segmented WP12 conv — per-segment live-state reads (nostore).
                        xqlaunch!(l, "xq_conv1d_chunk_seg_ns", ((conv_dim as u32 + 255) / 256, seg_k as u32, 1), (256, 1, 1), 0,
                                 (&mut sc.qkv, (*self.conv_state[gdn_i].device_ptr()) as u64, &g.conv_w,
                                  conv_dim as i32, ck as i32, &sc.seg, conv_dim as i32))?;
                        } else {
                        xqlaunch!(l, "xq_conv1d_chunk_ns", ((conv_dim as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                                 (&mut sc.qkv, (*self.conv_state[gdn_i].device_ptr()) as u64, &g.conv_w,
                                  conv_dim as i32, ck as i32, m as i32, slot as i32, conv_dim as i32))?;
                        }
                        if wp12_xchk { self.wp12_xcheck_conv(l, sc, g, gdn_i, slot, m)?; }
                    } else {
                    // K6 (REL v0.7.3 review): the lone conv kernel is per-slot — a packed
                    // verify (seg_k > 0) must not silently fall to it when WP12_CONV is off.
                    anyhow::ensure!(seg_k == 0,
                        "packed verify (seg_k {seg_k}) needs WP12_CONV; the lone xq_conv1d_chunk is per-slot");
                    xqlaunch!(l, "xq_conv1d_chunk", ((conv_dim as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                             (&mut sc.qkv, &mut sc.conv_sh[gdn_i], &g.conv_w,
                              conv_dim as i32, ck as i32, m as i32, 0i32, conv_dim as i32))?;
                    }
                    if replay_saves {
                    // S-A3-f-d Item 1b FIX-2: save the post-conv rows — the gdn
                    // commit replays rows 0..a with THIS layer's q/k/v, but the
                    // shared sc.qkv holds only the last GDN layer's at commit time.
                    xqlaunch!(l, "xq_copy_u16", (((m * conv_dim) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                             (&mut sc.qkv_post, &sc.qkv, (m * conv_dim) as i64,
                              (gdn_i * SAVE_PLANE_ROWS * conv_dim) as i64, 0i64))?;
                    }
                    // MTP_DIAG layer-0 bisect: post-conv plane (shadow state).
                    if lk == 0 { if let Some(d) = bmtd.as_ref() { dump_u(&d, &sc.qkv, "l0conv", 0)?; } }
                    wp27.join(&wp27_ab)?; // WP27: the scan reads b_out/a_out
                    // GDN recurrent over the group. S-A3-m: reads the LIVE state row
                    // (slot) with nostore — never written back (commit replays from
                    // live), so no shadow copy. slot_C = slot | m<<12 | 1<<30.
                    let slot_c = (slot | (m << 12) | (1 << 30)) as i32;
                    if seg_k > 0 && kd == 128 && vd == 128 {
                        // PACK1 LEG 1: segmented register-body scan, grid (nh, k_seg); the ring
                        // is the packed 16-row-deep buffer (plane stride 16 rows); the kernel
                        // offsets the ring base by row0 so ring rows land at absolute rows.
                        xqlaunch!(l, "xq_gdn_step_chunk_seg", (nh as u32, seg_k as u32, 1), (128, 1, 1), 0,
                                 (&mut sc.core, &sc.qkv, (*self.s_state[gdn_i].device_ptr()) as u64,
                                  &sc.b_out, &sc.a_out, (nh | (nk << 16)) as i32, kd as i32, vd as i32,
                                  &g.a_log, &g.dt_bias, &sc.seg,
                                  (*sc.gdn_ring_p.device_ptr()) as u64
                                      + (gdn_i * 16 * nh * GDN_RS * 4) as u64))?;
                    } else if wp12_on(WP12_STEP) && kd == 128 && vd == 128 && m <= 8 {
                        // WP12 step: register-resident scan (same device body as the decode
                        // step), grid nh x 128 threads; bitwise == xq_gdn_step_chunk.
                        xqlaunch!(l, "xq_gdn_step_chunk_r", (nh as u32, 1, 1), (128, 1, 1), 0,
                                 (&mut sc.core, &sc.qkv, (*self.s_state[gdn_i].device_ptr()) as u64,
                                  &sc.b_out, &sc.a_out, (nh | (nk << 16)) as i32, kd as i32, vd as i32,
                                  &g.a_log, &g.dt_bias, slot_c,
                                  (*sc.gdn_ring.device_ptr()) as u64
                                      + (gdn_i * sc.vrows * nh * GDN_RS * 4) as u64))?;
                        if wp12_xchk { self.wp12_xcheck_verify_step(l, sc, g, gdn_i, slot_c, m)?; }
                    } else {
                    // K6: the lone chunk scan is per-slot (slot_c) — kd/vd != 128 has no
                    // segmented twin, so a packed verify must refuse instead of falling to it.
                    anyhow::ensure!(seg_k == 0,
                        "packed verify (seg_k {seg_k}) needs kd = vd = 128; the lone xq_gdn_step_chunk is per-slot");
                    xqlaunch!(l, "xq_gdn_step_chunk", ((nh * nchunk) as u32, 1, 1), (kd as u32, 1, 1), smem as u32,
                             (&mut sc.core, &sc.qkv, (*self.s_state[gdn_i].device_ptr()) as u64,
                              &sc.b_out, &sc.a_out, (nh | (nk << 16)) as i32, kd as i32, vd as i32,
                              &g.a_log, &g.dt_bias, slot_c,
                              (*sc.gdn_ring.device_ptr()) as u64
                                  + (gdn_i * sc.vrows * nh * GDN_RS * 4) as u64))?;
                    }
                    if lk == 0 { if let Some(d) = bmtd.as_ref() { dump_u(&d, &sc.core, "l0core", 0)?; } }
                    wp27.join(&wp27_z)?; // WP27: the gate reads zbuf
                    if cfg.gdn_gate_sigmoid {
                    xqlaunch!(l, "xq_gdn_gate", ((m * nh) as u32, 1, 1), (vd as u32, 1, 1), (vd * 4) as u32,
                             (&mut sc.normed, &sc.core, &sc.zbuf, &g.norm, vd as i32, nh as i32, m as i32, eps))?;
                    } else { // EXL27B: Qwen3.5-class gated norm, silu(z) (a kernel copy)
                    xqlaunch!(l, "xq_gdn_gate_silu", ((m * nh) as u32, 1, 1), (vd as u32, 1, 1), (vd * 4) as u32,
                             (&mut sc.normed, &sc.core, &sc.zbuf, &g.norm, vd as i32, nh as i32, m as i32, eps))?;
                    }
                    if lk == 0 { if let Some(d) = bmtd.as_ref() { dump_u(&d, &sc.normed, "l0gate", 0)?; } }
                    self.chain_o(l, &g.out_proj, &sc.normed, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.mixer_out, m, &sc.chain_ws)?; // TP-B: row-parallel -> all-reduce
                    if lk == 0 { if let Some(d) = bmtd.as_ref() { dump_u(&d, &sc.mixer_out, "l0yatt", 0)?; } }
                    gdn_i += 1;
                }
                Mixer::Attn(a) => {
                    // q, k|v (+ the indexer qk) — W4/DENSE: one launch; else the old chains
                    let idx_pre = self.attn_in_chains(l, sc, a, m, true)?;
                    let nh_p = (cfg.num_heads as i32) | ((cfg.num_kv_heads as i32) << 16);
                    let hd_p = (cfg.head_dim as i32) | ((cfg.rotary_dim as i32) << 16);
                    // chained attention: grid (nh, m) — rows loop in-block, each row
                    // writes its K/V before any later row's softmax reads it.
                    // S-A3-i: above the dense threshold the SAME qsa_attn_step serves
                    // the verify rows (per-row selections; prep writes all K/V first,
                    // so row r still only ever attends positions <= its own).
                    if qsa && a.idx.is_some() {
                        // K6 (corrected by GATES2): the union gather hard-wires ONE slot, so a packed
                        // (multi-slot) verify must never take it. Pass one_slot = (seg_k == 0): packed
                        // rows get the slot-aware gather (qsa_gather_old) — exactly what the packed
                        // probes ran before K6 (the union path is opt-in --w4qsa=1 and only for
                        // m >= 2 single-slot verifies). K6's first form, ensure!(seg_k == 0), refused
                        // EVERY packed verify at a qsa-live context and broke --probe-exl3-pack2 /
                        // --probe-exl3-packstate ("packed verify (seg_k 2): ... single-slot").
                        self.qsa_attn_step(l, sc, a, att_i, m, idx_pre, seg_k == 0)?;
                    } else {
                    self.qsa_keys_append_dense(l, sc, a, att_i, m, idx_pre)?;
                    if self.dense_v3_ok(a) {
                        // WP13: dense v3, bitwise to v2 below (--exl3-dense-v3=0 = v2)
                        self.dense_attn_v3(l, sc, a, (*self.k_cache[att_i].device_ptr()) as u64, m, true, att_i)?;
                    } else if self.dense_splitk_ok(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) {
                        // DSPLITK: the verify rows take the decode step's split-K pipeline — each row's
                        // slices come from its OWN position, so column 0 == a decode at that position
                        // (prep writes every row's K/V first; a row attends positions <= its own only)
                        self.dense_attn_splitk(l, sc, a, (*self.k_cache[att_i].device_ptr()) as u64, m, nh_p, hd_p, eps, seg_k == 0)?;
                    } else {
                    // S-A3-m: attention v2 — one block per (head, row), static smem.
                    xqlaunch!(l, "xq_attn_decode_group", ((cfg.num_heads as u32), m as u32, 1), (cfg.head_dim as u32, 1, 1), 0,
                              (&mut sc.normed, &sc.qg, &sc.kp, &sc.vp,
                               (*self.k_cache[att_i].device_ptr()) as u64,
                               &a.qnkn, &sc.cos, &sc.slots,
                               nh_p, hd_p, self.mpf(), eps))?;
                    }
                    }
                    self.chain_o(l, &a.o_proj, &sc.normed, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.mixer_out, m, &sc.chain_ws)?; // TP-B: row-parallel -> all-reduce
                    att_i += 1;
                }
            }
            let yatt_p = *sc.mixer_out.device_ptr() as u64;
            self.hc_post_d(l, sc, yatt_p, m, hc_def, &mut hc_pend)?;
            if lk == 0 { if let Some(d) = bmtd.as_ref() { dump_f(&d, &sc.resid, "l0rattn", 0)?; } }
            self.hc_pre_p(l, sc, &layer.hc_mlp, m, hc_pend.take())?;
            if lk == 0 && ple_early {
                // WP27: the PLE prefix (reads only sc.ple_emb, staged before the graph) as a
                // sibling branch of layer 0's MoE window; joined at the layer-1 PLE boundary
                // (the window holds no hc grid-barrier kernel).
                wp27_ple = crate::exl3_wp27::Cap::open(l.stream.stream, sc.wp27_cls, crate::exl3_wp27::FORK_PLE);
                wp27_ple.side_begin()?;
                self.ple_prefix(l, sc, m)?;
                wp27_ple.side_end()?;
                wp27_ple_tail = wp27_ple.cut()?;
            }
            self.tp_pf_hint(self.layers.get(lk + 1).map_or(&self.hc_mixer, |nl| &nl.hc_attn), m);   // TP-4Z2: the MoE reduce's K2 prefetches the next hc mixer's weights (verify graphs)
            self.moe(l, sc, &layer.moe, m)?;
            if lk == 0 { if let Some(d) = bmtd.as_ref() { dump_u(&d, &sc.moe_out, "l0ymlp", 0)?; } }
            let ymlp_p = *sc.moe_out.device_ptr() as u64;
            self.hc_post_d(l, sc, ymlp_p, m, hc_def, &mut hc_pend)?;
            // EXL27C-DF2 (fwd): the DFlash2 tap on the VERIFY pass (this shared kernel
            // sequence serves verify_block and every packed variant - one hook covers them
            // all; m = the k+1-wide round, cols [0, m) land in the sink exactly as the
            // trunk's graphed verify tap does). Same guards as the decode hook in
            // forward_step_kernels; flush-before-tap so the residual is materialized.
            if self.df2_sink.is_some() && m <= crate::dflash2::block()
                && self.cfg.hidden_size == crate::dflash2::geom().hidden {
                if let Some(kk) = crate::dflash2::geom().tap_layers.iter().position(|&x| x == lk) {
                    if hc_pend.is_some() { self.hc_flush(l, sc, m, &mut hc_pend)?; }
                    crate::dflash2::capture::capture_cols_f32(
                        &self.dev, l.stream, self.df2_sink.as_ref().unwrap(),
                        *sc.resid.device_ptr() as u64,
                        self.cfg.hc_count.max(1) * self.cfg.hidden_size,
                        self.cfg.hidden_size, self.cfg.hc_count.max(1), m, kk)?;
                }
            }
            if lk == 0 { if let Some(d) = bmtd.as_ref() { dump_f(&d, &sc.resid, "l0rmlp", 0)?; } }
        }
        // final mixer -> hidden; lm_head (FULL head — never the pruned slice);
        // argmax per row -> sc.argmax[0..m]
        self.hc_pre_p(l, sc, &self.hc_mixer, m, hc_pend.take())?;
        // MTP_DIAG: final-mixer pre-RMS input (col 0) — the head out_proj plane.
        if let Some(d) = bmtd.as_ref() { dump_f(&d, &sc.resid, "finalmix", 0)?; }
        let v = cfg.vocab_size;
        if self.vp_greedy(sc, m) {
            // TP-H #4: vocab-parallel greedy tail (shard GEMM + argmax keys + one key exchange) —
            // sc.argmax[0..m] = the replicated argmax bit for bit; no sampler (samp_live is false,
            // so xq_sample_rows would return at once on every row).
            self.vp_greedy_tail(l, sc, m)?;
        } else {
        if self.vp_gather(sc, m) {
            // TP-H2 (--tp-vp-sampled): shard GEMM + row all-gather -> sc.logits rows 0..m = the replicated
            // head's rows bit for bit on both ranks; the penalty / argmax / samplers below are unchanged.
            self.vp_gather_tail(l, sc, m)?;
        } else {
        exl3_chain(l, &self.lm_head, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.logits, m, &sc.chain_ws)?;
        }
        // WP15: row i is penalized over the committed history + b, d_0..d_{i-1} (exactly the
        // plain decode's past at that position), before the argmax and the sampler.
        self.pen_rows(l, sc, m)?;
        // WP09: one m x nb-block launch replaces the m single-block launches (bitwise ids).
        self.argmax_rows(l, sc, m, v)?;
        if sc.rq_live {
            // WP24 real-q (--spec-sampling ratio): rows 0..m-2 judge d_i under min(1, p/q'),
            // residual on reject; the bonus row m-1 samples p. xq_accept below is unchanged.
            xqlaunch!(l, "xq_sample_rows_rq", (m as u32, 1, 1), (1024, 1, 1), 0,
                     (&mut sc.argmax, &sc.logits, v as i32, &sc.samp, &sc.d_dev, &sc.rq_list,
                      (m - 1) as i32))?;
        } else {
        // API-parity G2: sampled lanes sample every verify row (accept while sample == draft).
        xqlaunch!(l, "xq_sample_rows", (m as u32, 1, 1), (1024, 1, 1), 0,
                 (&mut sc.argmax, &sc.logits, v as i32, &sc.samp))?;
        }
        }
        // MTP_DIAG: per-column logits + argmax — the direct investigate-your-defect read.
        if let Some(d) = bmtd.as_ref() {
            dump_u(&d, &sc.logits, "press0logits", 0)?;
            let amh = self.dtoh(&sc.argmax)?;
            crate::dprintln!("MTP_DIAG argmax[:{}]= {:?}  v={}", m, &amh[..m], v);
        }
        Ok(())
    }

    /// Commit: replay rows 0..a of every stateful op from the LIVE state
    /// (a read on device from sc.acc2 = {slot, a}). Bit-identical to the
    /// sequential decode of the same prefix; rows a+1..k never commit.
    pub(super) fn verify_commit(&self, l: &Launcher, sc: &mut Scratch) -> Result<()> {
        let cfg = &self.tc; // TP-B: the trunk's per-rank geometry
        let kd = cfg.lin_k_dim;
        let vd = cfg.lin_v_dim;
        let nh = cfg.lin_num_v_heads;
        let nk = cfg.lin_num_k_heads;
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let ck = cfg.conv_kernel;
        let nchunk = vd / 32;
        let smem = (kd * 33 + kd + 32 * 3 + kd * 2) * 4;
        // xq_gdn_commit_ring is specialised to kd = 128 (4 warps x 32 rows) and C <= 8 (WP23).
        let ring_mode = gdn_commit_mode(kd, vd); // WP12: the ONE predicate verify_kernels also uses
        // W4S (gdn): every GDN layer's conv + ring commit and the PLE ring commit in ONE launch
        // (bitwise; --w4s-off=gdn = the per-layer launches below). Not with --mtp-dump
        // (its post-commit dump sits between layer 0's conv and ring commits).
        let mtp_dump = crate::opts::var(crate::opt!("mtp-dump")).map_or(false, |d| !d.is_empty());
        if !mtp_dump && self.w4s_commit_all(l, sc, ring_mode)? { return Ok(()); }
        let mut gdn_i = 0usize;
        for layer in self.layers.iter() {
            if let Mixer::Gdn(g) = &layer.mixer {
                // live states passed as raw device pointers (the k_cache pattern):
                // the kernels write them, but only under the device-side commit key
                xqlaunch!(l, "xq_conv_commit", ((conv_dim as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                         ((*self.conv_state[gdn_i].device_ptr()) as u64, &g.conv_w,
                          conv_dim as i32, ck as i32, &sc.qkv_save,
                          (gdn_i * SAVE_PLANE_ROWS * conv_dim) as i64, &sc.acc2))?;
                // MTP_DIAG: the live conv/GDN state right after the commit — layer 0.
                if gdn_i == 0 {
                    if let Ok(d) = crate::opts::var(crate::opt!("mtp-dump")) {
                        if !d.is_empty() {
                            self.dsync()?;
                            let hostc = self.dtoh(&self.conv_state[0])?;
                            let mut f2 = std::fs::File::create(format!("{}/g0cpost_s000.f32", d))?;
                            for &v in &hostc { f2.write_all(&v.to_le_bytes())?; }
                            let hosts = self.dtoh(&self.s_state[0])?;
                            let mut f3 = std::fs::File::create(format!("{}/g0spost_s000.f32", d))?;
                            for &v in &hosts { f3.write_all(&v.to_le_bytes())?; }
                        }
                    }
                }
                // S-A3-f-d Item 1b FIX-2: per-layer plane pointers (byte offsets
                // into the save buffers — xq_gdn_commit indexes rows internally).
                let ring_p = (*sc.gdn_ring.device_ptr()) as u64
                    + (gdn_i * sc.vrows * nh * GDN_RS * 4) as u64;
                if ring_mode == 1 {
                    // S-A3-o G1: rank-1 ring commit (bitwise == the replay below).
                    xqlaunch!(l, "xq_gdn_commit_ring", ((nh * nchunk) as u32, 1, 1), (kd as u32, 1, 1), 0,
                             ((*self.s_state[gdn_i].device_ptr()) as u64, ring_p,
                              nh as i32, kd as i32, vd as i32, &sc.acc2))?;
                    gdn_i += 1;
                    continue;
                }
                if ring_mode == 2 {
                    // check: ring-commit a copy of the whole state plane, then diff vs the replay
                    let n = self.s_state[gdn_i].len();
                    anyhow::ensure!(n <= sc.gdn_chk.len(), "gdn ring check: plane {n} > chk {}", sc.gdn_chk.len());
                    xqlaunch!(l, "xq_copy_f32", (((n as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                             (&mut sc.gdn_chk, &self.s_state[gdn_i], n as i64, 0i64, 0i64))?;
                    xqlaunch!(l, "xq_gdn_commit_ring", ((nh * nchunk) as u32, 1, 1), (kd as u32, 1, 1), 0,
                             ((*sc.gdn_chk.device_ptr()) as u64, ring_p,
                              nh as i32, kd as i32, vd as i32, &sc.acc2))?;
                }
                xqlaunch!(l, "xq_gdn_commit", ((nh * nchunk) as u32, 1, 1), (kd as u32, 1, 1), smem as u32,
                         (&mut sc.core,
                          ((*sc.qkv_post.device_ptr()) as u64
                           + (gdn_i * SAVE_PLANE_ROWS * conv_dim * 2) as u64) as u64,
                          (*self.s_state[gdn_i].device_ptr()) as u64,
                          ((*sc.b_save.device_ptr()) as u64
                           + (gdn_i * SAVE_PLANE_ROWS * nh * 2) as u64) as u64,
                          ((*sc.a_save.device_ptr()) as u64
                           + (gdn_i * SAVE_PLANE_ROWS * nh * 2) as u64) as u64,
                          nh as i32, nk as i32, kd as i32, vd as i32,
                          &g.a_log, &g.dt_bias, &sc.acc2))?;
                if ring_mode == 2 {
                    self.dsync()?;
                    let live = self.dtoh(&self.s_state[gdn_i])?;
                    let chk = self.dtoh(&sc.gdn_chk)?;
                    let n = live.len();
                    let mism = live.iter().zip(chk[..n].iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    let acc = self.dtoh(&sc.acc2)?;
                    use std::sync::atomic::{AtomicUsize, Ordering};
                    static CHK_N: AtomicUsize = AtomicUsize::new(0);
                    static CHK_BAD: AtomicUsize = AtomicUsize::new(0);
                    let k = CHK_N.fetch_add(1, Ordering::SeqCst) + 1;
                    if mism > 0 { CHK_BAD.fetch_add(1, Ordering::SeqCst); }
                    if mism > 0 || k % 360 == 0 {
                        crate::dprintln!("GDN_RING_CHECK: layer-commits {} bad {} (this: gdn {} a {} mism {})",
                                 k, CHK_BAD.load(Ordering::SeqCst), gdn_i, acc.get(1).copied().unwrap_or(-1), mism);
                    }
                }
                gdn_i += 1;
            }
        }
        if self.ple.is_some() && crate::opts::var(crate::opt!("exl3-no-ple")).is_err() {
            xqlaunch!(l, "xq_ple_ring_commit", ((10240u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                     (&sc.ple_normed, (*self.ple_state.device_ptr()) as u64, &sc.acc2))?;
        }
        Ok(())
    }

    /// S-A3-m: the verify block of an MTP round — launches only (uploads happen in
    /// mtp_round before it; the accept key lives in sc.acc2), graph-capturable.
    pub(super) fn verify_block(&self, l: &Launcher, sc: &mut Scratch, slot: usize, m: usize, k: usize,
                    qsa: bool) -> Result<()> {
        let _tc = tune::enter(tune::Ctx::fam(tune::Fam::Verify, m).regime(qsa)); // TUNE family scope
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        self.verify_shadow(l, sc, slot)?;
        self.verify_kernels(l, sc, m, qsa, slot, 0)?;
        // taps snapshot (rows 0..m-1 for re-prime / next draft-extend) into THIS slot's window
        // (WP03; the offset is a per-slot constant — verify graphs are keyed per slot)
        let tb = self.tap_base(sc, slot)?;
        xqlaunch!(l, "xq_copy_f32", grid((he * m) as i64), (256, 1, 1), 0,
                  (&mut sc.taps_keep, &sc.resid, (he * m) as i64, tb as i64, 0i64))?;
        // accept (device) + commit (device-keyed)
        xqlaunch!(l, "xq_accept", (1, 1, 1), (32, 1, 1), 0,
                  (&sc.d_dev, &sc.argmax, &mut sc.accept_out, &mut sc.acc2,
                   k as i32, slot as i32))?;
        self.verify_commit(l, sc)
    }

    // ==================== PACK1 LEG 1 (S-B9-CF-PACK1) ====================
    // EAGER 2-request packed speculative verify: probe-only helpers. No graphs, no
    // scheduler integration, no flag; reachable only from --probe-exl3-pack2. The
    // served verify_block is untouched (shadow -> kernels -> taps -> accept -> commit);
    // the packed block is shadow_seg -> kernels(seg) -> accept_seg, NO taps, NO commit
    // (nostore: the packed verify never writes live state — gate G3).
    pub(super) fn verify_shadow_seg(&self, l: &Launcher, sc: &mut Scratch, segs: &[[i32; 4]]) -> Result<()> {
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        // WP12: GDN/conv shadows are skipped (the seg kernels read the LIVE rows with
        // nostore, exactly like the served lone verify). Only the PLE ring shadow is
        // unconditional — one copy per segment into its own plane of ple_ring_sh.
        for (j, sg) in segs.iter().enumerate() {
            let pr = (sg[0] as i64) * (10240 * 9) as i64;
            xqlaunch!(l, "xq_copy_u16", grid((10240 * 9) as i64), (256, 1, 1), 0,
                      (&mut sc.ple_ring_sh, &self.ple_state, (10240 * 9) as i64,
                       (j as i64) * (10240 * 9) as i64, pr))?;
        }
        Ok(())
    }

    pub(super) fn verify_block_packed(&self, l: &Launcher, sc: &mut Scratch, segs: &[[i32; 4]],
                           r: usize, k_seg: usize, qsa: bool) -> Result<()> {
        let _tc = tune::enter(tune::Ctx::fam(tune::Fam::Verify, r).regime(qsa));
        self.verify_shadow_seg(l, sc, segs)?;
        self.verify_kernels(l, sc, r, qsa, segs[0][0] as usize, k_seg)?;
        xqlaunch!(l, "xq_accept_seg", (k_seg as u32, 1, 1), (32, 1, 1), 0,
                  (&sc.d_dev, &sc.argmax, &mut sc.accept_out, &mut sc.acc_seg, &sc.seg))?;
        Ok(())
    }

    pub(super) fn verify_block_nostore(&self, l: &Launcher, sc: &mut Scratch, slot: usize, m: usize,
                            k: usize, qsa: bool) -> Result<()> {
        let _tc = tune::enter(tune::Ctx::fam(tune::Fam::Verify, m).regime(qsa));
        self.verify_shadow(l, sc, slot)?;
        self.verify_kernels(l, sc, m, qsa, slot, 0)?;
        xqlaunch!(l, "xq_accept", (1, 1, 1), (32, 1, 1), 0,
                  (&sc.d_dev, &sc.argmax, &mut sc.accept_out, &mut sc.acc2,
                   k as i32, slot as i32))?;
        Ok(())
    }


    // ==================== PACK2 LEG 2 (S-B9-CF-PACK2) ====================
    // Per-segment device-keyed commit + the packed ROUND probe (--probe-exl3-packstate).
    // The served path is untouched: seg_k=0 launches exactly today's kernels; the
    // packed block below is reachable only from the probe.

    /// The segmented commit: the served per-layer commit bodies (xq_conv_commit /
    /// xq_gdn_commit_ring / xq_ple_ring_commit) with the commit key (slot, a) read
    /// ON DEVICE from acc_seg[j*2..+2] (xq_accept_seg's output) and rows from the
    /// seg table. Ring-mode only (the ring commit IS the served GDN commit here;
    /// gdn_commit_mode == 1 is asserted by the probe).
    pub(super) fn verify_commit_seg(&self, l: &Launcher, sc: &mut Scratch, k_seg: usize) -> Result<()> {
        let cfg = &self.tc; // TP-B: the trunk's per-rank geometry
        let kd = cfg.lin_k_dim;
        let vd = cfg.lin_v_dim;
        let nh = cfg.lin_num_v_heads;
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let ck = cfg.conv_kernel;
        let nchunk = vd / 32;
        let mut gdn_i = 0usize;
        for layer in self.layers.iter() {
            if let Mixer::Gdn(g) = &layer.mixer {
                // raw conv rows: this layer's qkv_save plane (rows are ABSOLUTE —
                // the seg kernel adds seg.row0); state writes keyed by acc_seg.
                xqlaunch!(l, "xq_conv_commit_seg", ((conv_dim as u32 + 255) / 256, k_seg as u32, 1), (256, 1, 1), 0,
                          ((*self.conv_state[gdn_i].device_ptr()) as u64, &g.conv_w,
                           conv_dim as i32, ck as i32,
                           ((*sc.qkv_save.device_ptr()) as u64
                            + (gdn_i * SAVE_PLANE_ROWS * conv_dim * 2) as u64) as u64, // ELEMENTS->BYTES
                           &sc.seg, &sc.acc_seg))?;
                // packed ring planes: 16 rows per GDN layer (xq_gdn_step_chunk_seg's
                // layout); ring rows are absolute (row0 + t).
                let ring_p = (*sc.gdn_ring_p.device_ptr()) as u64
                    + (gdn_i * 16 * nh * GDN_RS * 4) as u64;
                xqlaunch!(l, "xq_gdn_commit_ring_seg", ((nh * nchunk) as u32, k_seg as u32, 1), (kd as u32, 1, 1), 0,
                          ((*self.s_state[gdn_i].device_ptr()) as u64, ring_p,
                           nh as i32, kd as i32, vd as i32, &sc.seg, &sc.acc_seg))?;
                gdn_i += 1;
            }
        }
        if self.ple.is_some() && crate::opts::var(crate::opt!("exl3-no-ple")).is_err() {
            xqlaunch!(l, "xq_ple_ring_commit_seg", ((10240u32 + 255) / 256, k_seg as u32, 1), (256, 1, 1), 0,
                      (&sc.ple_normed, (*self.ple_state.device_ptr()) as u64, &sc.seg, &sc.acc_seg))?;
        }
        Ok(())
    }

    /// The packed verify BLOCK (LEG 2): shadow_seg -> seg kernels -> PER-SEGMENT
    /// taps snapshot (each request's OWN rows into ITS window) -> accept_seg ->
    /// the segmented commit. The re-prime then reads each slot's taps exactly
    /// like the served lone path.
    pub(super) fn verify_block_packed_full(&self, l: &Launcher, sc: &mut Scratch, segs: &[[i32; 4]],
                                r: usize, k_seg: usize, qsa: bool) -> Result<()> {
        let _tc = tune::enter(tune::Ctx::fam(tune::Fam::Verify, r).regime(qsa));
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        self.verify_shadow_seg(l, sc, segs)?;
        self.verify_kernels(l, sc, r, qsa, segs[0][0] as usize, k_seg)?;
        // per-segment taps snapshot: rows row0..row0+len -> slot's window rows 0..len
        for sg in segs.iter() {
            let (slot, row0, len) = (sg[0] as usize, sg[1] as usize, sg[2] as usize);
            let tb = self.tap_base(sc, slot)?;
            xqlaunch!(l, "xq_copy_f32", grid((he * len) as i64), (256, 1, 1), 0,
                      (&mut sc.taps_keep, &sc.resid, (he * len) as i64, tb as i64,
                       (row0 * he) as i64))?;
        }
        xqlaunch!(l, "xq_accept_seg", (k_seg as u32, 1, 1), (32, 1, 1), 0,
                  (&sc.d_dev, &sc.argmax, &mut sc.accept_out, &mut sc.acc_seg, &sc.seg))?;
        self.verify_commit_seg(l, sc, k_seg)
    }

}
