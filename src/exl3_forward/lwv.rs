//! LANEWIDE step 1 (2026-11-12): `--probe-lane-wide-cost`, the cost shapes that decide whether a
//! many-lane verify through the prefill-class kernels can pay (GPU, one box, timing only).
//!
//! Every slot 0..L-1 reads the prompt first (served chunked prefill), so the timed passes run at a
//! realistic position with real KV behind them. Then, each timed with a device sync around it and
//! reported as the median of `reps` after two discarded warm runs:
//!  A. `prefill_chunk` of R distinct real tokens on slot 0 (the prefill-class pass: row-wise GEMMs and
//!     MoE over R rows, one sequence's mixers), its running state restored after every run;
//!  B. the verify tail over R rows in groups of <= 16 (lm_head, argmax: the judge a wide verify needs);
//!  C. `forward_step` over m DISTINCT slots, one row each (the decode-class shared plain step);
//!  D. one lane's fixed-depth draft chain of k drafts (the MTP head, decode class).
//! Output is timing only: the tokens are never judged and the state written by C/D is discarded.

use super::*;

fn med(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// `--probe-lane-wide-cost --model-dir <pack> [--prompt-ids <file>] [--probe-lane-wide-rows 16,32,...]
/// [--probe-lane-wide-reps N]`.
pub fn probe_lane_wide_cost(a: &FwdArgs, rows: &[usize], reps: usize) -> Result<()> {
    let lanes = 16usize;
    dres::init_from_opts(); // --exl3-dense-resident applies to the timed passes as in serving
    dres::mid_init_from_opts(); // so does --exl3-mid-gemm
    let model = FwdModel::load(&a.dir, lanes, a.max_pos)?;
    anyhow::ensure!(model.tp.is_none(), "--probe-lane-wide-cost runs on one box");
    let mut sc = Scratch::new(&model.dev, &model.cfg, lanes, model.cfg.rotary_dim, model.max_pos)?;
    let ids = gate_prompt_ids(a)?;
    let rmax = rows.iter().copied().max().unwrap_or(16);
    anyhow::ensure!(ids.len() >= rmax + 64, "--probe-lane-wide-cost: the prompt needs at least {} tokens (got {})", rmax + 64, ids.len());
    // the prompt read by every slot (the last rmax + 32 ids are kept as the timed rows' tokens)
    let plen = ids.len() - rmax - 32;
    let cw = a.prefill_chunk.unwrap_or(2048).max(rmax + 1);
    let mut psc = model.prefill_scratch(cw)?;
    let mut bufs = CpvBufs::new();
    model.set_sampling(&mut sc, &vec![None; lanes])?;
    model.set_penalties(&mut sc, &vec![None; lanes])?;
    for s in 0..lanes {
        model.reset_slot(s)?;
        let mut pos = 0usize;
        while pos < plen {
            let end = (pos + cw).min(plen);
            let toks: Vec<i32> = ids[pos..end].iter().map(|&t| t as i32).collect();
            model.cpv_chunk(&mut psc, &mut bufs, &toks, pos, s, false)?;
            pos = end;
        }
    }
    model.dev.synchronize()?;
    crate::dprintln!("LANE_WIDE_COST: {lanes} slots read a {plen}-token prompt; timed rows start at position {plen}; median of {reps} after 2 warm runs");
    let reps = reps.max(1);
    let mut ck = model.ckpt_alloc()?;
    model.ckpt_save(0, &psc, &mut ck)?;
    let tail: Vec<i32> = ids[plen..].iter().map(|&t| t as i32).collect();
    // A + B
    for &r in rows {
        let toks = &tail[..r];
        let (mut ta, mut tb) = (Vec::new(), Vec::new());
        for it in 0..reps + 2 {
            model.ckpt_restore(0, &mut psc, &ck)?;
            model.dev.synchronize()?;
            // an idle gap between runs, so a kernel trace splits them cleanly (outside the timed span)
            std::thread::sleep(std::time::Duration::from_millis(10));
            let t0 = std::time::Instant::now();
            model.cpv_chunk(&mut psc, &mut bufs, toks, plen, 0, true)?;
            model.dev.synchronize()?;
            std::thread::sleep(std::time::Duration::from_millis(5));
            let t1 = std::time::Instant::now();
            let mut r0 = 0usize;
            let g = model.cpv_group_rows(&sc);
            while r0 < r {
                let m = g.min(r - r0);
                let xs = &bufs.stash.as_ref().unwrap().xs;
                model.cpv_head_group(&mut sc, xs, r0, m, &vec![None; m])?;
                r0 += m;
            }
            model.dev.synchronize()?;
            if it >= 2 {
                ta.push((t1 - t0).as_secs_f64() * 1e3);
                tb.push(t1.elapsed().as_secs_f64() * 1e3);
            }
        }
        let (ma, mb) = (med(ta), med(tb));
        crate::dprintln!("LANE_WIDE_COST prefill R={r:4}: chunk {ma:7.2} ms ({:.3} ms/row), judge {mb:6.2} ms, sum {:7.2} ms",
                         ma / r as f64, ma + mb);
    }
    model.ckpt_restore(0, &mut psc, &ck)?;
    // C: the decode-class plain step over m distinct slots (positions advance; the rows written are discarded)
    for m in [1usize, 2, 4, 8, 12, 16] {
        let mut t = Vec::new();
        for it in 0..reps + 2 {
            let toks: Vec<i32> = (0..m).map(|i| tail[(i + it) % tail.len()]).collect();
            let pos: Vec<usize> = vec![plen + (it % 8); m];
            let slots: Vec<usize> = (0..m).collect();
            model.dev.synchronize()?;
            let t0 = std::time::Instant::now();
            model.forward_step(&mut sc, &toks, &pos, &slots, None)?;
            model.dev.synchronize()?;
            if it >= 2 { t.push(t0.elapsed().as_secs_f64() * 1e3); }
        }
        let mm = med(t);
        crate::dprintln!("LANE_WIDE_COST decode m={m:2} distinct slots: {mm:7.2} ms ({:.2} ms/row)", mm / m as f64);
    }
    // D: one lane's fixed-depth draft chain (slot 1; its tap = the stash of the last chunk it read is not kept, so
    // the chain drafts from whatever row 0 of its window holds: timing only)
    if let Some(head) = model.mtp.as_ref() {
        for k in [1usize, 2, 3, 4, 5, 6] {
            let mut t = Vec::new();
            for it in 0..reps + 2 {
                model.dev.synchronize()?;
                let t0 = std::time::Instant::now();
                model.mtp_draft_fixed(&mut sc, head, 1, k, tail[it % tail.len()], plen, 0)?;
                model.dev.synchronize()?;
                if it >= 2 { t.push(t0.elapsed().as_secs_f64() * 1e3); }
            }
            crate::dprintln!("LANE_WIDE_COST draft chain k={k}: {:7.2} ms", med(t));
        }
    }
    if dres::on() {
        let (n, b) = dres::kept();
        crate::dprintln!("LANE_WIDE_COST dense-resident ON: {n} kept matrices, {:.2} GiB", b as f64 / (1u64 << 30) as f64);
    }
    crate::dprintln!("RESULT: LANE_WIDE_COST_DONE");
    Ok(())
}

// =============================================================================================
// LANEWIDE step 2 (2026-11-12): the many-lane wide verify (TP=1).
//
// One prefill-class pass verifies the speculative rows of up to `WIDE_KMAX` lanes at once: each lane brings its
// bonus token b at position p and its drafts d_1..d_k (rows b, d_1..d_k at p..p+k of ITS slot). The row-wise parts
// (dense chains through the reconstruct / kept-matrix arm, the a/b projections, the MoE, the hc mixes, the PLE
// projections, the lm_head) run once over all R rows on a PrefillScratch; the per-sequence parts run per segment
// with the packed verify's segmented kernels: the conv (`xq_conv1d_chunk_seg_ns`, live ring read, never written),
// the GDN scan (`xq_gdn_step_chunk_seg`, nostore, every row's rank-1 factors into the wide ring), the PLE conv on
// per-segment shadow planes (`xq_ple_conv_chunk_seg`); KV rows are written per row (`xq_attn_kv_prefill_pairs`) and
// attention runs per segment over the lane's own cache. Each lane then accepts its agreeing prefix (the verify's
// sample-then-compare rule; real-q rows under `--spec-sampling ratio`) and the per-segment commit replays exactly the
// accepted rows into the live state (`xq_conv_commit_seg`, `xq_gdn_commit_ring_seg`, `xq_ple_ring_commit_seg`):
// the replay rollback. The prefill-class kernels are not bit-identical to the decode-class verify, so output can
// differ on near-ties from the same request served alone.

/// Rows a wide verify can hold.
pub const WIDE_RMAX: usize = 160;
/// Lanes (segments) a wide verify can hold.
pub const WIDE_KMAX: usize = 16;

/// The wide verify's own buffers (allocated once per scheduler on first use).
pub struct WideBufs {
    pub psc: PrefillScratch,
    seg: CudaSlice<i32>,
    acc_seg: CudaSlice<i32>,
    ring: CudaSlice<f32>,
    qkv_save: CudaSlice<u16>,
    ple_sh: CudaSlice<u16>,
    pub(super) logits: CudaSlice<u16>,
    argmax: CudaSlice<i32>,
    am_part: CudaSlice<u64>,
    am_cnt: CudaSlice<u32>,
    samp: CudaSlice<u32>,
    d_dev: CudaSlice<i32>,
    rq_list: CudaSlice<u32>,
    yraw16: CudaSlice<u16>,
    rmax: usize,
    /// Probe-only (--probe-lane-wide-xclass): when Some, the trunk copies the rows [row0, row0 + n) of every
    /// intermediate buffer it writes to the host after each kernel (synchronizing). None on every served path.
    pub dbg: Option<WideDbg>,
}

/// Probe-only per-kernel taps of a row window of the wide trunk, in execution order: (name, rows x width as f32).
pub struct WideDbg { pub row0: usize, pub n: usize, pub taps: Vec<(String, usize, Vec<f32>)> }

fn wtap16(dev: &Arc<CudaDevice>, dbg: &mut Option<WideDbg>, buf: &CudaSlice<u16>, name: String, w: usize) -> Result<()> {
    let Some(d) = dbg.as_mut() else { return Ok(()); };
    dev.synchronize()?;
    let h = dev.dtoh_sync_copy(buf)?;
    d.taps.push((name, w, h[d.row0 * w..(d.row0 + d.n) * w].iter().map(|&b| f16::from_bits(b).to_f32()).collect()));
    Ok(())
}
fn wtap32(dev: &Arc<CudaDevice>, dbg: &mut Option<WideDbg>, buf: &CudaSlice<f32>, name: String, w: usize) -> Result<()> {
    let Some(d) = dbg.as_mut() else { return Ok(()); };
    dev.synchronize()?;
    let h = dev.dtoh_sync_copy(buf)?;
    d.taps.push((name, w, h[d.row0 * w..(d.row0 + d.n) * w].to_vec()));
    Ok(())
}
fn wtapi(dev: &Arc<CudaDevice>, dbg: &mut Option<WideDbg>, buf: &CudaSlice<i32>, name: String, w: usize) -> Result<()> {
    let Some(d) = dbg.as_mut() else { return Ok(()); };
    dev.synchronize()?;
    let h = dev.dtoh_sync_copy(buf)?;
    d.taps.push((name, w, h[d.row0 * w..(d.row0 + d.n) * w].iter().map(|&x| x as f32).collect()));
    Ok(())
}

/// One wide round's host-side timings (ms).
#[derive(Clone, Copy, Debug, Default)]
pub struct WideMs { pub stage: f64, pub trunk: f64, pub judge: f64, pub commit: f64 }

impl FwdModel {
    /// Why this boot cannot run wide verifies (None = it can).
    pub fn wide_unsupported(&self) -> Option<&'static str> {
        let cfg = &self.tc;
        // LANEWIDE-TP: TP=2 runs it (every rank the same rounds; the head's policy inputs); TP=4 is not wired yet
        if self.tp.as_ref().map_or(false, |t| t.world > 2) { return Some("tp"); }
        if self.mtp.is_none() || mtp_disabled_by_opt() { return Some("no_mtp"); }
        if cfg.lin_k_dim != 128 || cfg.lin_v_dim != 128 { return Some("unsupported_pack"); }
        if gdn_commit_mode(cfg.lin_k_dim, cfg.lin_v_dim) != 1 { return Some("unsupported_pack"); }
        if a_flags().taps_shared { return Some("unsupported_pack"); }
        if let Some(p) = self.ple.as_ref() { if p.bf || p.conv_f.is_some() || p.norm_f.is_some() { return Some("unsupported_pack"); } }
        if self.cfg.num_experts == 1 { return Some("unsupported_pack"); }
        if self.layers.iter().any(|l| l.moe.fx.is_some() || l.moe.router_bf || l.moe.split.is_some()) { return Some("unsupported_pack"); }
        if recon_min_rows() == 0 { return Some("unsupported_pack"); }
        None
    }

    /// LANEWIDE-TP: before a host lockstep exchange at world 2 that follows device collectives (the batched draft's
    /// head passes, the wide verify's reduces): device sync (our K2 consumed the peer's epochs) + the proxy retired
    /// every epoch we published, so no doorbell payload can alias the exchange's ring slots. No-op at TP=1 / world > 2
    /// (the hub exchange there rides dedicated control slots).
    pub fn tp_drain_before_exchange(&self) -> Result<()> {
        if self.tp.as_ref().map_or(false, |t| t.world == 2) {
            self.dsync()?;
            crate::net::drain_sends(std::time::Duration::from_millis(200))?;
        }
        Ok(())
    }

    pub fn wide_bufs(&self, rmax: usize) -> Result<WideBufs> {
        let cfg = &self.tc;
        let rmax = rmax.clamp(16, WIDE_RMAX);
        let n_gdn = self.s_state.len();
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let nh = cfg.lin_num_v_heads;
        let v = self.cfg.vocab_size;
        let dev = &self.dev;
        Ok(WideBufs {
            psc: self.prefill_scratch(rmax)?,
            seg: dev.alloc_zeros::<i32>(WIDE_KMAX * 4)?,
            acc_seg: dev.alloc_zeros::<i32>(WIDE_KMAX * 2)?,
            ring: dev.alloc_zeros::<f32>(n_gdn.max(1) * rmax * nh * GDN_RS)?,
            qkv_save: dev.alloc_zeros::<u16>(n_gdn.max(1) * rmax * conv_dim)?,
            ple_sh: dev.alloc_zeros::<u16>(10240 * 9 * WIDE_KMAX)?,
            logits: dev.alloc_zeros::<u16>(rmax * v)?,
            argmax: dev.alloc_zeros::<i32>(rmax)?,
            am_part: dev.alloc_zeros::<u64>(rmax * XQ_AM_NB_MAX)?,
            am_cnt: dev.htod_sync_copy(&vec![0u32; rmax])?, // a REAL zero (AGENTS 2.2)
            samp: dev.htod_sync_copy(&vec![0u32; rmax * 8])?,
            d_dev: dev.alloc_zeros::<i32>(rmax)?,
            rq_list: dev.htod_sync_copy(&vec![u32::MAX; rmax * crate::wp24::RQ_ROW])?,
            yraw16: dev.alloc_zeros::<u16>(16 * v)?,
            rmax,
            dbg: None,
        })
    }

    /// A drafted real-q lane's lists (served list rows 0..w, one per draft pass) -> rows row0.. of the wide list.
    /// Issued on the compute stream BEFORE the next lane drafts (every chain writes the served rows 0..w).
    pub fn wide_list_handoff(&self, sc: &Scratch, wb: &mut WideBufs, row0: usize, d: &PackDraft) -> Result<()> {
        if !d.samp.ratio { return Ok(()); }
        let rq = crate::wp24::RQ_ROW;
        let w = d.drafts.len();
        anyhow::ensure!((row0 + w) * rq <= wb.rq_list.len() && w * rq <= sc.rq_list.len(), "wide verify: list rows out of range");
        use cudarc::driver::sys;
        self.dev.bind_to_thread()?;
        let (dst, src) = (*wb.rq_list.device_ptr() + (row0 * rq * 4) as u64, *sc.rq_list.device_ptr());
        let r = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, w * rq * 4, self.stream.stream) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "wide verify: list copy failed ({r:?})");
        Ok(())
    }

    /// The trunk of a wide verify over the staged rows (module note). `segs[j] = [slot, row0, len, p]`.
    fn wide_trunk(&self, l: &Launcher, wb: &mut WideBufs, segs: &[[i32; 4]], r: usize) -> Result<()> {
        let cfg = &self.tc;
        let h = cfg.hidden_size;
        let kd = cfg.lin_k_dim;
        let vd = cfg.lin_v_dim;
        let nh = cfg.lin_num_v_heads;
        let nk = cfg.lin_num_k_heads;
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let ck = cfg.conv_kernel;
        let ns = segs.len();
        let rmax = wb.rmax;
        let psc = &mut wb.psc;
        let dbg = &mut wb.dbg;
        let dev = &self.dev;
        let he = cfg.hc_count.max(1) * h;
        let vdim = cfg.value_dim();
        let (ne, topk) = (self.cfg.num_experts, self.cfg.num_experts_per_tok);
        macro_rules! t16 { ($b:expr, $n:expr, $w:expr) => { if dbg.is_some() { wtap16(dev, dbg, &$b, $n, $w)?; } } }
        macro_rules! t32 { ($b:expr, $n:expr, $w:expr) => { if dbg.is_some() { wtap32(dev, dbg, &$b, $n, $w)?; } } }
        macro_rules! ti { ($b:expr, $n:expr, $w:expr) => { if dbg.is_some() { wtapi(dev, dbg, &$b, $n, $w)?; } } }
        let exact = dres::exact_on();
        psc.row_indep = exact;
        if exact { dres::exact_pass(); }
        let rb = self.recon_bufs(psc);
        let rmin = recon_min_rows();
        psc.pf_hook = None;
        psc.sp_rows = None;
        self.embed_gather(l, *psc.resid.device_ptr() as u64, *psc.toks.device_ptr() as u64, r)?;
        t32!(psc.resid, "embed".into(), he);
        let ple_live = self.ple.is_some() && crate::opts::var(crate::opt!("exl3-no-ple")).is_err();
        let pf_defer = self.pf_hc_fuse(l).0;
        let mut pend: Option<u64> = None;
        let (mut gdn_i, mut att_i) = (0usize, 0usize);
        for (li, layer) in self.layers.iter().enumerate() {
            if li == 1 && ple_live {
                if let Some(y) = pend.take() { self.hc_post_rows(l, psc, y, r)?; }
                self.wide_ple(l, psc, &wb.ple_sh, &wb.seg, ns, r)?;
                t16!(psc.ple_key16, "w_ple_key".into(), 10240);
                t16!(psc.ple_val, "w_ple_val".into(), 2560);
                t32!(psc.resid, "w_ple_resid".into(), he);
            }
            self.hc_pre_rows_n(l, psc, &layer.hc_attn, r, pend.take(), false)?;
            t16!(psc.x, format!("xpre{li}"), h);
            match &layer.mixer {
                Mixer::Gdn(g) => {
                    exl3_chain_rb(l, &g.in_qkv, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.qkv, r, &psc.chain_ws, &rb)?;
                    t16!(psc.qkv, format!("w_qkv{li}"), conv_dim);
                    exl3_chain_rb(l, &g.in_z, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.zbuf, r, &psc.chain_ws, &rb)?;
                    t16!(psc.zbuf, format!("w_z{li}"), vdim);
                    if exact {
                        let pin = wide_pin(h, nh, 0, 0);
                        exl3_lt_matmul_pinned(l, *g.a_w.device_ptr() as u64, *psc.x.device_ptr() as u64, *psc.a_out.device_ptr() as u64,
                                              h, nh, r, pin, true, rb.ltws, rb.ltws_len)?;
                        exl3_lt_matmul_pinned(l, *g.b_w.device_ptr() as u64, *psc.x.device_ptr() as u64, *psc.b_out.device_ptr() as u64,
                                              h, nh, r, pin, true, rb.ltws, rb.ltws_len)?;
                    } else if rmin > 0 && r >= rmin {
                        exl3_lt_matmul(l, *g.a_w.device_ptr() as u64, *psc.x.device_ptr() as u64, *psc.a_out.device_ptr() as u64,
                                       h as usize, nh, r, true, rb.ltws, rb.ltws_len)?;
                        exl3_lt_matmul(l, *g.b_w.device_ptr() as u64, *psc.x.device_ptr() as u64, *psc.b_out.device_ptr() as u64,
                                       h as usize, nh, r, true, rb.ltws, rb.ltws_len)?;
                    } else {
                        xq_gemm_f16_small!(l, &mut psc.a_out, &g.a_w, &psc.x, r, nh, h)?;
                        xq_gemm_f16_small!(l, &mut psc.b_out, &g.b_w, &psc.x, r, nh, h)?;
                    }
                    t16!(psc.a_out, format!("w_a{li}"), nh);
                    t16!(psc.b_out, format!("w_b{li}"), nh);
                    // the raw (pre-conv) rows for the conv commit's replay
                    let n = (r * conv_dim) as i64;
                    xqlaunch!(l, "xq_copy_u16", ((((n + 255) / 256).max(1)) as u32, 1, 1), (256, 1, 1), 0,
                              (&mut wb.qkv_save, &psc.qkv, n, (gdn_i * rmax * conv_dim) as i64, 0i64))?;
                    xqlaunch!(l, "xq_conv1d_chunk_seg_ns", ((conv_dim as u32 + 255) / 256, ns as u32, 1), (256, 1, 1), 0,
                              (&mut psc.qkv, (*self.conv_state[gdn_i].device_ptr()) as u64, &g.conv_w,
                               conv_dim as i32, ck as i32, &wb.seg, conv_dim as i32))?;
                    t16!(psc.qkv, format!("w_conv{li}"), conv_dim);
                    xqlaunch!(l, "xq_gdn_step_chunk_seg", (nh as u32, ns as u32, 1), (128, 1, 1), 0,
                              (&mut psc.core, &psc.qkv, (*self.s_state[gdn_i].device_ptr()) as u64,
                               &psc.b_out, &psc.a_out, (nh | (nk << 16)) as i32, kd as i32, vd as i32,
                               &g.a_log, &g.dt_bias, &wb.seg,
                               (*wb.ring.device_ptr()) as u64 + (gdn_i * rmax * nh * GDN_RS * 4) as u64))?;
                    t16!(psc.core, format!("w_core{li}"), vdim);
                    if cfg.gdn_gate_sigmoid {
                        xqlaunch!(l, "xq_gdn_gate", ((r * nh) as u32, 1, 1), (vd as u32, 1, 1), (vd * 4) as u32,
                                  (&mut psc.normed, &psc.core, &psc.zbuf, &g.norm, vd as i32, nh as i32, r as i32, cfg.rms_eps))?;
                    } else {
                        xqlaunch!(l, "xq_gdn_gate_silu", ((r * nh) as u32, 1, 1), (vd as u32, 1, 1), (vd * 4) as u32,
                                  (&mut psc.normed, &psc.core, &psc.zbuf, &g.norm, vd as i32, nh as i32, r as i32, cfg.rms_eps))?;
                    }
                    t16!(psc.normed, format!("w_gate{li}"), vdim);
                    self.chain_o_rb(l, &g.out_proj, &psc.normed, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.mixer_out, r, &psc.chain_ws, &rb, None, None)?;
                    gdn_i += 1;
                }
                Mixer::Attn(a) => {
                    exl3_chain_rb(l, &a.q_proj, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.qg, r, &psc.chain_ws, &rb)?;
                    exl3_chain_rb(l, &a.k_proj, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.kp, r, &psc.chain_ws, &rb)?;
                    exl3_chain_rb(l, &a.v_proj, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.vp, r, &psc.chain_ws, &rb)?;
                    let (nh_a, nkv, hd, rdim) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);
                    t16!(psc.qg, format!("w_q{li}"), nh_a * hd * 2);
                    t16!(psc.kp, format!("w_k{li}"), nkv * hd);
                    t16!(psc.vp, format!("w_v{li}"), nkv * hd);
                    let nkv_hd = (nkv as i32) | ((hd as i32) << 16);
                    let rdim_mp = (rdim as i32) | ((self.max_pos as i32) << 8) | ((self.kv_fmt as i32) << 28);
                    let kc_p = *self.k_cache[att_i].device_ptr() as u64;
                    xqlaunch!(l, "xq_attn_kv_prefill_pairs", ((r * nkv) as u32, 1, 1), (hd as u32, 1, 1), 0,
                              (&mut psc.kp, &mut psc.vp, kc_p, &a.qnkn, &psc.cos, &psc.slotpos_rows,
                               nkv_hd, rdim_mp, r as i32, cfg.rms_eps))?;
                    t16!(psc.kp, format!("w_krope{li}"), nkv * hd);
                    let nh_nkv = (nkv as i32) | ((nh_a as i32) << 16);
                    let hd_rdim = (hd as i32) | ((rdim as i32) << 16);
                    let (qg0, out0, cs0) = (*psc.qg.device_ptr() as u64, *psc.normed.device_ptr() as u64, *psc.cos.device_ptr() as u64);
                    for sg in segs {
                        let (slot, row0, len, p) = (sg[0] as usize, sg[1] as usize, sg[2] as usize, sg[3] as usize);
                        let slot_pos0: i32 = (slot as i32) | ((p as i32) << 12);
                        let qg_p = qg0 + (row0 * nh_a * hd * 2 * 2) as u64;
                        let out_p = out0 + (row0 * nh_a * hd * 2) as u64;
                        let cs_p = cs0 + (row0 * 2 * rdim * 4) as u64;
                        // the flash body serves any t >= 1 (8-row tiles masked at the store; the tile's tail reads stay below
                        // p + 8); prompt prefill keeps the sweep below position 256 for byte stability, a wide verify does not
                        // need to (its rows are judged, and the sweep's serial walk cost ~1.5 ms per call per segment)
                        let flash = hd == 256 && p + 8 <= self.max_pos
                            && !matches!(crate::opts::var(crate::opt!("attn-flash")).as_deref(), Ok("0"));
                        if flash {
                            // CAUSAL staging (flash256c): every key / value a row may read is staged in full. The default
                            // prompt kernel (flash256 / its bitwise twin flash256v) zeroes the dims of earlier warps in the
                            // keys a later row reads on its diagonal 8-row tile (kernel note above a3fg_stage_v); a whole
                            // wide segment sits in that tile, so with it every draft row saw truncated keys of the rows
                            // before it (--probe-lane-wide-xclass: logits rel-L2 up to 0.15 at segment row 3, argmax flips
                            // at margins of 1.2)
                            xqlaunch!(l, "xq_attn_prefill_flash256c", (((len as u32) + 7) / 8, nh_a as u32, 1), (256, 1, 1), 0,
                                      (out_p, qg_p, kc_p, &a.qnkn, cs_p, nh_nkv, hd_rdim, self.mpf(), slot_pos0, len as i32, cfg.rms_eps))?;
                        } else {
                            xqlaunch!(l, "xq_attn_prefill_q", (nh_a as u32, 1, 1), (hd as u32, 1, 1), 0,
                                      (out_p, qg_p, kc_p, &a.qnkn, cs_p, nh_nkv, hd_rdim, self.mpf(), slot_pos0, len as i32, cfg.rms_eps))?;
                        }
                    }
                    t16!(psc.normed, format!("w_attn{li}"), nh_a * hd);
                    self.wide_qsa_keys(l, psc, a, att_i, segs, r)?;
                    self.chain_o_rb(l, &a.o_proj, &psc.normed, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.mixer_out, r, &psc.chain_ws, &rb, None, None)?;
                    att_i += 1;
                }
            }
            t16!(psc.mixer_out, format!("yatt{li}"), h);
            let yatt = *psc.mixer_out.device_ptr() as u64;
            if pf_defer { pend = Some(yatt); } else { self.hc_post_rows(l, psc, yatt, r)?; }
            self.hc_pre_rows_n(l, psc, &layer.hc_mlp, r, pend.take(), false)?;
            t16!(psc.x, format!("xpre_mlp{li}"), h);
            self.moe_prefill(l, psc, &layer.moe, r, &rb)?;
            t32!(psc.logits_r, format!("rlog{li}"), ne);
            ti!(psc.ids, format!("w_ids{li}"), topk);
            t32!(psc.wts, format!("w_wts{li}"), topk);
            t16!(psc.moe_out, format!("ymlp{li}"), h);
            let ymlp = *psc.moe_out.device_ptr() as u64;
            if pf_defer { pend = Some(ymlp); } else { self.hc_post_rows(l, psc, ymlp, r)?; }
        }
        // the streams of every row stay in resid (the taps); the final mix -> x (the lm_head input)
        if let Some(y) = pend.take() {
            // flush into resid first so the taps read the complete streams, then mix without a pending inject
            self.hc_post_rows(l, psc, y, r)?;
        }
        self.hc_pre_rows_n(l, psc, &self.hc_mixer, r, None, false)?;
        t16!(psc.x, "finalmix".into(), h);
        Ok(())
    }

    /// The indexer keys of every row (per-row slot / position tables) and the pooled plane per segment.
    fn wide_qsa_keys(&self, l: &Launcher, psc: &mut PrefillScratch, a: &AttnLayer, att_i: usize, segs: &[[i32; 4]], r: usize) -> Result<()> {
        let Some(ix) = a.idx.as_ref() else { return Ok(()); };
        if !self.cfg.has_indexer() || qsa_disabled() { return Ok(()); }
        let Some(keys_p) = self.qsa_keys_p(att_i) else { return Ok(()); };
        let cfg = &self.cfg;
        let (hdx, heads) = (cfg.indexer_head_dim, cfg.indexer_n_heads);
        let qk_dim = (heads + 1) * hdx;
        anyhow::ensure!(psc.qsa_qk.len() >= r * qk_dim && psc.qsa_qkb.len() >= r * qk_dim, "wide verify: indexer buffers narrower than {r} rows");
        let rb = self.recon_bufs(psc);
        exl3_chain_rb(l, &ix.qk, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.qsa_qk, r, &psc.chain_ws, &rb)?;
        let ncnv = (r * qk_dim) as i64;
        xqlaunch!(l, "xq_qsa_h2b", ((((ncnv + 255) / 256).max(1)) as u32, 1, 1), (256, 1, 1), 0, (&mut psc.qsa_qkb, &psc.qsa_qk, ncnv))?;
        gblaunch!(l, "qsa_key_write_b", ((r * hdx) as u32, 1, 1), (256, 1, 1), 0,
                  (keys_p, &psc.qsa_qkb, &psc.pos_rows, &psc.slot_rows, self.max_pos as i32, 0i32,
                   qk_dim as i32, (heads * hdx) as i32, hdx as i32, r as i32))?;
        for sg in segs {
            self.qsa_pool_range(l, a, att_i, keys_p, sg[0] as usize, sg[3] as usize, sg[2] as usize)?;
        }
        Ok(())
    }

    /// The PLE at the layer-1 boundary: the n-gram rows were staged into psc.ple_emb by the caller; projections over
    /// all rows, then the segmented conv on per-segment shadow planes (copied from the live rings by the caller).
    fn wide_ple(&self, l: &Launcher, psc: &mut PrefillScratch, ple_sh: &CudaSlice<u16>, seg: &CudaSlice<i32>, ns: usize, r: usize) -> Result<()> {
        let Some(ple) = self.ple.as_ref() else { return Ok(()); };
        let scale = 1.0f32 / (2560f32).sqrt();
        let rmin = recon_min_rows();
        let rb = self.recon_bufs(psc);
        let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
        let exact = psc.row_indep;
        if exact {
            exl3_lt_matmul_pinned(l, *ple.key_w.device_ptr() as u64, *psc.ple_emb.device_ptr() as u64, *psc.ple_key16.device_ptr() as u64,
                                  2560, 10240, r, wide_pin(2560, 10240, 0, 0), true, rb.ltws, rb.ltws_len)?;
        } else if rmin > 0 && r >= rmin {
            exl3_lt_matmul(l, *ple.key_w.device_ptr() as u64, *psc.ple_emb.device_ptr() as u64, *psc.ple_key16.device_ptr() as u64,
                           2560, 10240, r, true, rb.ltws, rb.ltws_len)?;
        } else {
            xq_gemm_f16_small!(l, &mut psc.ple_key16, &ple.key_w, &psc.ple_emb, r, 10240usize, 2560usize)?;
        }
        self.ple_norm(l, ple, 0, dp(psc.ple_key16.device_ptr()), dp(psc.ple_kn.device_ptr()), r)?;
        if exact {
            exl3_lt_matmul_pinned(l, *ple.value_w.device_ptr() as u64, *psc.ple_emb.device_ptr() as u64, *psc.ple_val.device_ptr() as u64,
                                  2560, 2560, r, wide_pin(2560, 2560, 0, 0), true, rb.ltws, rb.ltws_len)?;
        } else if rmin > 0 && r >= rmin {
            exl3_lt_matmul(l, *ple.value_w.device_ptr() as u64, *psc.ple_emb.device_ptr() as u64, *psc.ple_val.device_ptr() as u64,
                           2560, 2560, r, true, rb.ltws, rb.ltws_len)?;
        } else {
            xq_gemm_f16_small!(l, &mut psc.ple_val, &ple.value_w, &psc.ple_emb, r, 2560usize, 2560usize)?;
        }
        self.ple_norm(l, ple, 1, dp(psc.resid.device_ptr()), dp(psc.ple_qn.device_ptr()), r)?;
        xqlaunch!(l, "xq_ple_gate", (r as u32 * 4, 1, 1), (256, 1, 1), 0, (&psc.ple_qn, &psc.ple_kn, &psc.ple_val, &mut psc.ple_gated, scale))?;
        self.ple_norm(l, ple, 2, dp(psc.ple_gated.device_ptr()), dp(psc.ple_normed.device_ptr()), r)?;
        xqlaunch!(l, "xq_ple_conv_chunk_seg", ((10240u32 + 255) / 256, ns as u32, 1), (256, 1, 1), 0,
                  (&mut psc.resid, &psc.ple_gated, &psc.ple_normed, *ple_sh.device_ptr() as u64, &ple.conv_w, seg))?;
        Ok(())
    }

    /// ONE wide verify for the drafted lanes `ds` (each drafted on the served scratch `sc`, a real-q lane's lists
    /// already handed to its row offset with `wide_list_handoff`): stage, trunk, judge, per-lane accept, the
    /// per-segment commit (replay of the accepted rows), each lane's taps into its window of `sc`, the host n-gram
    /// rings restored to the accepted prefix. Returns each lane's round as `mtp_verify` returns it.
    pub fn wide_verify(&self, sc: &mut Scratch, wb: &mut WideBufs, ds: &[PackDraft]) -> Result<(Vec<MtpRoundOut>, WideMs)> {
        let t0 = std::time::Instant::now();
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let ns = ds.len();
        anyhow::ensure!(ns >= 1 && ns <= WIDE_KMAX, "wide verify: {ns} lanes (1..={WIDE_KMAX})");
        let (h, v) = (self.cfg.hidden_size, self.cfg.vocab_size);
        let he = self.cfg.hc_count.max(1) * h;
        let topk = self.tc.num_experts_per_tok;
        // rows, segments, drafts per row
        let mut rows: Vec<(usize, i32, usize)> = Vec::new();
        let mut segs: Vec<[i32; 4]> = Vec::with_capacity(ns);
        let mut d_dev = vec![0i32; wb.rmax];
        let mut slots_seen = std::collections::HashSet::new();
        for d in ds {
            anyhow::ensure!(!d.drafts.is_empty() && d.drafts.len() + 1 <= 11, "wide verify: slot {} brings {} drafts", d.slot, d.drafts.len());
            anyhow::ensure!(slots_seen.insert(d.slot), "wide verify: slot {} twice", d.slot);
            anyhow::ensure!(d.p + d.drafts.len() + 1 < self.max_pos, "wide verify: slot {} rows past the KV window", d.slot);
            anyhow::ensure!(!self.qsa_live(d.p + d.drafts.len()), "wide verify: slot {} is past the sparse switch", d.slot);
            let row0 = rows.len();
            for i in 0..=d.drafts.len() {
                rows.push((d.slot, if i == 0 { d.b } else { d.drafts[i - 1] }, d.p + i));
            }
            anyhow::ensure!(rows.len() <= wb.rmax, "wide verify: {} rows > {}", rows.len(), wb.rmax);
            d_dev[row0..row0 + d.drafts.len()].copy_from_slice(&d.drafts);
            segs.push([d.slot as i32, row0 as i32, (d.drafts.len() + 1) as i32, d.p as i32]);
        }
        let r = rows.len();
        let vclamp = |t: i32| if t < 0 || t >= v as i32 { 0 } else { t };
        // the host n-gram ring pair of each slot BEFORE the staging advances it
        let hp: Vec<(i32, i32)> = { let hst = self.ple_hist.lock().unwrap(); ds.iter().map(|d| (hst[d.slot * 2], hst[d.slot * 2 + 1])).collect() };
        {
            let psc = &mut wb.psc;
            let toks: Vec<i32> = rows.iter().map(|x| vclamp(x.1)).collect();
            let pos: Vec<i32> = rows.iter().map(|x| x.2 as i32).collect();
            let slots: Vec<i32> = rows.iter().map(|x| x.0 as i32).collect();
            let sp: Vec<i32> = rows.iter().flat_map(|x| [x.0 as i32, x.2 as i32]).collect();
            self.dev.htod_sync_copy_into(&toks, &mut psc.toks.slice_mut(0..r))?;
            self.dev.htod_sync_copy_into(&pos, &mut psc.pos_rows.slice_mut(0..r))?;
            self.dev.htod_sync_copy_into(&slots, &mut psc.slot_rows.slice_mut(0..r))?;
            self.dev.htod_sync_copy_into(&sp, &mut psc.slotpos_rows.slice_mut(0..2 * r))?;
            self.dev.htod_sync_copy_into(&[(r * topk) as i32], &mut psc.moerows.slice_mut(0..1))?;
            self.cos_gather(&l, &mut psc.cos, *psc.pos_rows.device_ptr() as u64, SlotSrc::Pairs(*psc.slotpos_rows.device_ptr() as u64), r)?;
            if self.ple.is_some() {
                let mut emb = vec![0u16; r * 2560];
                let sl: Vec<usize> = rows.iter().map(|x| x.0).collect();
                self.ple_build_embed(r, &toks, &sl, &mut emb)?;
                self.dev.htod_sync_copy_into(&emb, &mut psc.ple_emb.slice_mut(0..r * 2560))?;
            }
        }
        let mut seg_flat: Vec<i32> = segs.iter().flat_map(|s| [s[0], s[1], s[2], 1]).collect();
        seg_flat.resize(WIDE_KMAX * 4, 0);
        self.dev.htod_sync_copy_into(&seg_flat, &mut wb.seg)?;
        self.dev.htod_sync_copy_into(&d_dev, &mut wb.d_dev)?;
        // sampler rows: lane j's row i = its params with counter ctr + i (flag 2 = a real-q draft row, 1 = sampled)
        let mut words = vec![0u32; wb.rmax * 8];
        let (mut samp_live, mut rq_live) = (false, false);
        for (d, sg) in ds.iter().zip(&segs) {
            let Some(p) = d.samp.samp else { continue };
            samp_live = true;
            rq_live |= d.samp.ratio;
            let (row0, len) = (sg[1] as usize, sg[2] as usize);
            for i in 0..len {
                let flag = if d.samp.ratio && i + 1 < len { 2 } else { 1 };
                let row = SampParams { counter: p.counter.wrapping_add(i as u32), ..p };
                words[(row0 + i) * 8..(row0 + i) * 8 + 8].copy_from_slice(&row.pack_flag(flag));
            }
        }
        if samp_live { self.dev.htod_sync_copy_into(&words, &mut wb.samp)?; }
        // PLE shadows: segment j's live ring row -> plane j
        if self.ple.is_some() {
            let pe = (10240 * 9) as i64;
            for (j, sg) in segs.iter().enumerate() {
                xqlaunch!(l, "xq_copy_u16", ((((pe + 255) / 256) as u32), 1, 1), (256, 1, 1), 0,
                          (&mut wb.ple_sh, &self.ple_state, pe, (j as i64) * pe, (sg[0] as i64) * pe))?;
            }
        }
        let t1 = std::time::Instant::now();
        let _tc = tune::enter(tune::Ctx::prefill(r));
        self.wide_trunk(&l, wb, &segs, r)?;
        let t2 = std::time::Instant::now();
        // judge: lm_head over the rows in groups of <= 16 (the decode-class lm_head body), argmax, sampler
        let q = &self.lm_head;
        anyhow::ensure!(q.lmh && !lmh_off() && q.k as usize == h, "wide verify: the lm_head is not on the LMH path");
        let x0 = *wb.psc.x.device_ptr() as u64;
        let xh = *wb.psc.chain_xh.device_ptr() as u64;
        let lg0 = *wb.logits.device_ptr() as u64;
        let had = lmh_had();
        let mut r0 = 0usize;
        while r0 < r {
            let m = 16.min(r - r0);
            xqlaunch!(l, "exl3_had_suh", (m as u32 * (q.k as u32 / 128), 1, 1), (32, 1, 1), 0, (x0 + (r0 * h * 2) as u64, &q.suh, xh, q.k))?;
            let out = if had { lg0 + (r0 * v * 2) as u64 } else { *wb.yraw16.device_ptr() as u64 };
            anyhow::ensure!(lmh_gemm_launch(l.dev, l.stream.stream, *q.tr.device_ptr(), xh, out, *q.svh.device_ptr(), m,
                                            q.k as usize, q.n as usize, q.bits as usize, lmh_nst(), had)?,
                            "wide verify: the lm_head GEMM refused m = {m}");
            if !had {
                xqlaunch!(l, "exl3_had_svh", (m as u32 * (q.n as u32 / 128), 1, 1), (32, 1, 1), 0,
                          (&wb.yraw16, &q.svh, lg0 + (r0 * v * 2) as u64, q.n))?;
            }
            r0 += m;
        }
        let nb = wp09_am_nb(r);
        xqlaunch!(l, "xq_argmax_rows", (nb as u32, r as u32, 1), (256, 1, 1), 0,
                  (&mut wb.argmax, 0i32, &wb.logits, 0i64, v as i64, v as i32, &mut wb.am_part, &mut wb.am_cnt))?;
        if rq_live {
            xqlaunch!(l, "xq_sample_rows_rq", (r as u32, 1, 1), (1024, 1, 1), 0,
                      (&mut wb.argmax, &wb.logits, v as i32, &wb.samp, &wb.d_dev, &wb.rq_list, (r - 1) as i32))?;
        } else if samp_live {
            xqlaunch!(l, "xq_sample_rows", (r as u32, 1, 1), (1024, 1, 1), 0, (&mut wb.argmax, &wb.logits, v as i32, &wb.samp))?;
        }
        if samp_live {
            // NIGHT1 follow-up (PR #51): this blocking NULL-stream upload would wait for the whole packed verify above inside the
            // driver (a spin under --gpu-wait hybrid); wait for the compute stream through host_wait first, then upload.
            self.dsync()?;
            // greedy staging for the next round (a later all-greedy round uploads nothing)
            self.dev.htod_sync_copy_into(&vec![0u32; r * 8], &mut wb.samp.slice_mut(0..r * 8))?;
        }
        let choices = self.dtoh(&wb.argmax)?;
        let t3 = std::time::Instant::now();
        // per lane: the agreeing prefix, the emitted tokens
        let mut outs: Vec<MtpRoundOut> = Vec::with_capacity(ns);
        let mut acc = vec![0i32; WIDE_KMAX * 2];
        for (j, (d, sg)) in ds.iter().zip(&segs).enumerate() {
            let row0 = sg[1] as usize;
            let ch = &choices[row0..row0 + d.drafts.len() + 1];
            let a = crate::copy_prefill::accept_prefix(ch, &d.drafts);
            let emitted = crate::copy_prefill::emitted(&d.drafts, ch, a).context("wide verify: the judged row is missing")?;
            anyhow::ensure!(emitted.iter().all(|&t| t >= 0 && (t as usize) < v), "wide verify: slot {} emitted an id outside the vocabulary", d.slot);
            acc[j * 2] = d.slot as i32;
            acc[j * 2 + 1] = a as i32;
            outs.push(MtpRoundOut { a, emitted: emitted.clone(), pos_next: d.p + a + 1, last_tok: emitted[a], tap_row: a,
                                    drafts: d.drafts.clone(), diag_ms: 0.0 });
        }
        self.dev.htod_sync_copy_into(&acc, &mut wb.acc_seg)?;
        // the per-segment commit: replay exactly the accepted rows (a + 1 per lane) into the live state
        {
            let cfg = &self.tc;
            let (kd, vd, nh) = (cfg.lin_k_dim, cfg.lin_v_dim, cfg.lin_num_v_heads);
            let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
            let ck = cfg.conv_kernel;
            let nchunk = vd / 32;
            let rmax = wb.rmax;
            let mut gdn_i = 0usize;
            for layer in self.layers.iter() {
                if let Mixer::Gdn(g) = &layer.mixer {
                    xqlaunch!(l, "xq_conv_commit_seg", ((conv_dim as u32 + 255) / 256, ns as u32, 1), (256, 1, 1), 0,
                              ((*self.conv_state[gdn_i].device_ptr()) as u64, &g.conv_w, conv_dim as i32, ck as i32,
                               (*wb.qkv_save.device_ptr()) as u64 + (gdn_i * rmax * conv_dim * 2) as u64, &wb.seg, &wb.acc_seg))?;
                    xqlaunch!(l, "xq_gdn_commit_ring_seg", ((nh * nchunk) as u32, ns as u32, 1), (kd as u32, 1, 1), 0,
                              ((*self.s_state[gdn_i].device_ptr()) as u64,
                               (*wb.ring.device_ptr()) as u64 + (gdn_i * rmax * nh * GDN_RS * 4) as u64,
                               nh as i32, kd as i32, vd as i32, &wb.seg, &wb.acc_seg))?;
                    gdn_i += 1;
                }
            }
            if self.ple.is_some() && crate::opts::var(crate::opt!("exl3-no-ple")).is_err() {
                xqlaunch!(l, "xq_ple_ring_commit_seg", ((10240u32 + 255) / 256, ns as u32, 1), (256, 1, 1), 0,
                          (&wb.psc.ple_normed, (*self.ple_state.device_ptr()) as u64, &wb.seg, &wb.acc_seg))?;
            }
        }
        // taps: each lane's rows of the final streams -> rows 0..len of its window in the served scratch
        let grid = |n: i64| (((n + 255) / 256).max(1) as u32, 1, 1);
        for sg in &segs {
            let (slot, row0, len) = (sg[0] as usize, sg[1] as usize, sg[2] as usize);
            let tb = self.tap_base(sc, slot)?;
            xqlaunch!(l, "xq_copy_f32", grid((he * len) as i64), (256, 1, 1), 0,
                      (&mut sc.taps_keep, &wb.psc.resid, (he * len) as i64, tb as i64, (row0 * he) as i64))?;
        }
        self.dsync()?;
        // the host n-gram rings: the staging advanced each by its rows; a + 1 tokens were committed
        {
            let mut hist = self.ple_hist.lock().unwrap();
            for (j, (d, o)) in ds.iter().zip(&outs).enumerate() {
                let a = o.a;
                if a == 0 {
                    hist[d.slot * 2] = hp[j].1;
                    hist[d.slot * 2 + 1] = d.b;
                } else {
                    let tok = |i: usize| if i == 0 { d.b } else { d.drafts[i - 1] };
                    hist[d.slot * 2] = tok(a - 1);
                    hist[d.slot * 2 + 1] = tok(a);
                }
            }
        }
        if let Some(st) = sc.wp05.as_mut() { st.inflight = false; }
        let t4 = std::time::Instant::now();
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        Ok((outs, WideMs { stage: ms(t0, t1), trunk: ms(t1, t2), judge: ms(t2, t3), commit: ms(t3, t4) }))
    }
}

/// One lane of a batched draft (`wide_draft_batch`).
#[derive(Clone, Copy, Debug)]
pub struct WideLane {
    pub slot: usize,
    pub b: i32,
    pub p: usize,
    pub tap_row: usize,
    /// drafts this lane brings to the wide verify (>= 1)
    pub k: usize,
    pub samp: crate::lane_pack::SegSamp,
    pub rq: Option<RqStage>,
}

impl FwdModel {
    /// LANEWIDE batched drafting: the drafts of every lane as ONE head pass per depth (m = lanes rows, each at its own
    /// slot / position), the pruned draft head's logits for all rows, then per row its draft: the argmax (greedy and
    /// match-rule lanes) or the real-q draw from q' with the lane's own draft sampler and counter (`xq_rq_draft`,
    /// list rows written straight into the wide verify's list at the lane's row offset). Fixed depth per lane (no
    /// confidence stop); a lane with fewer drafts than the deepest keeps its first k. Rows are laid out as
    /// `wide_verify` lays them (lane order, k + 1 rows each). One sync at the end.
    pub fn wide_draft_batch(&self, sc: &mut Scratch, wb: &mut WideBufs, lanes: &[WideLane]) -> Result<Vec<PackDraft>> {
        let head = self.mtp.as_ref().context("wide draft: no MTP head")?;
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let n = lanes.len();
        anyhow::ensure!(n >= 1 && n <= sc.toks.len() && n <= 16, "wide draft: {n} lanes (scratch width {})", sc.toks.len());
        let kmax = lanes.iter().map(|x| x.k).max().unwrap_or(1);
        anyhow::ensure!(kmax >= 1 && kmax <= MTP_MAX_K, "wide draft: depth {kmax}");
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let grid = |m: i64| (((m + 255) / 256).max(1) as u32, 1, 1);
        let q = self.lm_head_draft.as_ref().unwrap_or(&self.lm_head);
        let vd = q.n as usize;
        let any_rq = lanes.iter().any(|x| x.samp.ratio);
        if any_rq { anyhow::ensure!(crate::wp24::draft_fits(vd), "wide draft: draft vocab {vd} outside the real-q envelope"); }
        // per-lane draft sampler blocks and metas (the counter base = the lane's dctr), staged once
        let mut ds_words = vec![0u32; 16 * 8];
        let mut metas = vec![0i32; 16 * 4];
        let mut row0 = Vec::with_capacity(n);
        let mut at = 0usize;
        for (j, x) in lanes.iter().enumerate() {
            row0.push(at);
            at += x.k + 1;
            if let (true, Some(rq)) = (x.samp.ratio, x.rq) { ds_words[j * 8..j * 8 + 8].copy_from_slice(&rq.pack()); }
            metas[j * 4..j * 4 + 4].copy_from_slice(&[x.b, x.p as i32, x.slot as i32, x.rq.map_or(0, |r| r.dctr as i32)]);
        }
        anyhow::ensure!(at <= wb.rmax, "wide draft: {at} rows > {}", wb.rmax);
        let (dsamp, meta) = if any_rq {
            (Some(self.dev.htod_sync_copy(&ds_words)?), Some(self.dev.htod_sync_copy(&metas)?))
        } else { (None, None) };
        // pass 0's input streams: each lane's saved tap (row tap_row of its window) -> resid row j
        for (j, x) in lanes.iter().enumerate() {
            let tb = self.tap_base(sc, x.slot)?;
            xqlaunch!(l, "xq_copy_f32", grid(he as i64), (256, 1, 1), 0,
                      (&mut sc.resid, &sc.taps_keep, he as i64, (j * he) as i64, (tb + x.tap_row * he) as i64))?;
        }
        let mut sids = vec![0i32; sc.slot_ids.len()];
        for (j, x) in lanes.iter().enumerate() { sids[j] = x.slot as i32; }
        self.dev.htod_copy_into(sids, &mut sc.slot_ids)?;
        let mut d_all = self.dev.alloc_zeros::<i32>(kmax * n)?;
        let toks0: Vec<i32> = lanes.iter().map(|x| x.b).collect();
        for i in 0..kmax {
            // stage {tok, pos, slot|pos} of every row (tokens of pass i > 0 = the previous pass's drafts, on device)
            if i == 0 {
                let mut t = toks0.clone();
                t.resize(sc.toks.len(), 0);
                self.dev.htod_copy_into(t, &mut sc.toks)?;
            }
            let mut pos = vec![0i32; sc.pos.len()];
            let mut sp = vec![0i32; sc.slots.len()];
            for (j, x) in lanes.iter().enumerate() {
                pos[j] = (x.p + i) as i32;
                sp[j * 2] = x.slot as i32;
                sp[j * 2 + 1] = (x.p + i) as i32;
            }
            self.dev.htod_copy_into(pos, &mut sc.pos)?;
            self.dev.htod_copy_into(sp, &mut sc.slots)?;
            let qsa = self.qsa_live(lanes.iter().map(|x| x.p + i).max().unwrap_or(0));
            self.head_launches(&l, sc, head, n, qsa, false)?;
            exl3_chain(&l, q, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.logits, n, &sc.chain_ws)?;
            self.argmax_rows(&l, sc, n, vd)?;
            if let (Some(ds), Some(mt)) = (dsamp.as_ref(), meta.as_ref()) {
                let nb = crate::wp24::draft_nb(vd);
                let rq = crate::wp24::RQ_ROW;
                for (j, x) in lanes.iter().enumerate() {
                    if !x.samp.ratio || i >= x.k { continue; }
                    xqlaunch!(l, "xq_rq_draft", (nb as u32, 1, 1), (256, 1, 1), 0,
                              (*sc.argmax.device_ptr() as u64 + (j * 4) as u64, *sc.logits.device_ptr() as u64 + (j * vd * 2) as u64,
                               vd as i32, &mut sc.rq_part, &mut sc.rq_cnt,
                               *wb.rq_list.device_ptr() as u64 + (row0[j] * rq * 4) as u64,
                               *ds.device_ptr() as u64 + (j * 8 * 4) as u64, *mt.device_ptr() as u64 + (j * 4 * 4) as u64,
                               i as i32, 2i32))?;
                }
            }
            // this pass's drafts: kept for the readback and fed to the next pass
            use cudarc::driver::sys;
            self.dev.bind_to_thread()?;
            let r1 = unsafe { sys::cuMemcpyDtoDAsync_v2(*d_all.device_ptr() + (i * n * 4) as u64, *sc.argmax.device_ptr(), n * 4, self.stream.stream) };
            let r2 = unsafe { sys::cuMemcpyDtoDAsync_v2(*sc.toks.device_ptr(), *sc.argmax.device_ptr(), n * 4, self.stream.stream) };
            anyhow::ensure!(r1 == sys::CUresult::CUDA_SUCCESS && r2 == sys::CUresult::CUDA_SUCCESS, "wide draft: draft copy failed");
        }
        self.dsync()?;
        let all = self.dtoh(&d_all)?;
        let v = self.cfg.vocab_size as i32;
        Ok(lanes.iter().enumerate().map(|(j, x)| {
            let drafts: Vec<i32> = (0..x.k).map(|i| { let d = all[i * n + j]; if d < 0 || d >= v { 0 } else { d } }).collect();
            PackDraft { slot: x.slot, b: x.b, p: x.p, drafts, confs: Vec::new(), spec: false, samp: x.samp, rq: x.rq,
                        launched: kmax, wasted: false }
        }).collect())
    }
}

// ---------------------------------------------------------------------------------------------
// --probe-lane-wide-xclass: the wide verify against the decode-class ground truth
// ---------------------------------------------------------------------------------------------

fn rel_l2_f32(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { let d = *x as f64 - *y as f64; num += d * d; den += (*y as f64) * (*y as f64); }
    (num / den.max(1e-30)).sqrt()
}

/// --probe-lane-wide-xclass: (1) the first kernel taps where the lane's rows as ONE segment differ from the same rows
/// inside the batch pass (bits), in execution order; (2) per layer, the rel-L2 of the mixed GEMM input (xpre), the
/// mixer output (yatt), the router logits (rlog) and the MoE output (ymlp) of each row against the token-by-token
/// decode of the same token (decode step i = segment row i).
fn xclass_taps_report(step: usize, st: &WideDbg, bt: &WideDbg, dec: &[std::collections::HashMap<String, Vec<f32>>]) {
    let mut diffs: Vec<String> = Vec::new();
    let mut ndiff = 0usize;
    for ((sn, w, sv), (bn, _, bv)) in st.taps.iter().zip(bt.taps.iter()) {
        if sn != bn { diffs.push(format!("ORDER {sn}/{bn}")); ndiff += 1; continue; }
        if sv.iter().zip(bv).all(|(a, b)| a.to_bits() == b.to_bits()) { continue; }
        ndiff += 1;
        if diffs.len() < 12 {
            let rows: Vec<String> = (0..st.n).map(|r| {
                let (a, b) = (&sv[r * w..(r + 1) * w], &bv[r * w..(r + 1) * w]);
                let nd = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                format!("r{r}:{nd}/{:.1e}", rel_l2_f32(a, b))
            }).collect();
            diffs.push(format!("{sn} [{}]", rows.join(" ")));
        }
    }
    crate::dprintln!("XCLASS_TAPS step {step}: {ndiff} of {} taps differ single vs batch; first: {}", st.taps.len(), diffs.join(" ; "));
    for (lab, t) in [("single", st), ("batch", bt)] {
        for kind in ["embed", "xpre", "yatt", "rlog", "ymlp", "finalmix"] {
            let mut cells = Vec::new();
            for (name, w, v) in t.taps.iter() {
                let li = if kind == "embed" || kind == "finalmix" { if name != kind { continue } 0 }
                         else { let Some(li) = name.strip_prefix(kind).and_then(|x| x.parse::<usize>().ok()) else { continue }; li };
                let rr: Vec<String> = (0..t.n.min(dec.len())).map(|r| match dec[r].get(name) {
                    Some(d) if d.len() >= *w => format!("{:.1e}", rel_l2_f32(&v[r * w..(r + 1) * w], &d[..*w])),
                    _ => "-".into(),
                }).collect();
                cells.push(format!("L{li}:{}", rr.join("/")));
            }
            crate::dprintln!("XCLASS_HID step {step} {lab} {kind} relL2 vs decode rows0/1/2/3: {}", cells.join(" "));
        }
    }
}

fn rel_l2_f16(a: &[u16], b: &[u16]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (half::f16::from_bits(*x).to_f32() as f64, half::f16::from_bits(*y).to_f32() as f64);
        if !x.is_finite() || !y.is_finite() { continue; }
        num += (x - y) * (x - y);
        den += y * y;
    }
    (num / den.max(1e-30)).sqrt()
}

fn top2(row: &[u16]) -> (usize, f32) {
    let (mut i1, mut v1, mut v2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &b) in row.iter().enumerate() {
        let x = half::f16::from_bits(b).to_f32();
        if !x.is_finite() { continue; }
        if x > v1 { v2 = v1; v1 = x; i1 = i; } else if x > v2 { v2 = x; }
    }
    (i1, v1 - v2)
}

/// `--probe-lane-wide-xclass --model-dir <pack> --prompt-ids <file: JSON list of prompt id lists> [--probe-lane-wide-rows S]`:
/// lane 0 = prompt 0 (the request under test), lanes 1.. = the other prompts (batch fillers). Every lane reads its prompt
/// (served chunked prefill) and takes its first token from a decode-class step. Then for S greedy steps of lane 0:
///  (gt) the decode-class trajectory: 4 token-by-token decode steps from lane 0's state (logits per row);
///  (single) the wide verify with ONE segment [b, d1, d2, d3] (d = the decode trajectory) from the same state;
///  (batch) the wide verify with every lane as a segment, lane 0 in the middle;
/// each with --exl3-mid-gemm off and on; per row: logits rel-L2 vs gt, argmax agreement, gt top-2 margin; single vs
/// batch logits equality (row-independence of the whole pass). Every lane's state is restored after each pass, and
/// lane 0 then advances by one decode step.
pub fn probe_lane_wide_xclass(a: &FwdArgs, steps: usize) -> Result<bool> {
    let lanes_max = 16usize;
    let model = FwdModel::load(&a.dir, lanes_max, a.max_pos)?;
    anyhow::ensure!(model.tp.is_none(), "--probe-lane-wide-xclass runs on one box");
    dres::init_from_opts();
    let prompts: Vec<Vec<u32>> = serde_json::from_str(&std::fs::read_to_string(a.ids_file.as_ref().context("--prompt-ids required")?)?)?;
    let nl = prompts.len().min(lanes_max);
    anyhow::ensure!(nl >= 2, "--probe-lane-wide-xclass: at least two prompts");
    let mut sc = Scratch::new(&model.dev, &model.cfg, lanes_max, model.cfg.rotary_dim, model.max_pos)?;
    let mut psc = model.prefill_scratch(2048)?;
    let mut bufs = CpvBufs::new();
    let mut wb = model.wide_bufs(128)?;
    let v = model.cfg.vocab_size;
    model.set_sampling(&mut sc, &vec![None; lanes_max])?;
    model.set_penalties(&mut sc, &vec![None; lanes_max])?;
    let mut st: Vec<(i32, usize)> = Vec::new(); // (bonus, position) per lane
    for (s, pr) in prompts.iter().take(nl).enumerate() {
        model.reset_slot(s)?;
        let n = pr.len();
        let mut pos = 0;
        while pos + 1 < n {
            let end = (pos + 2048).min(n - 1);
            let toks: Vec<i32> = pr[pos..end].iter().map(|&t| t as i32).collect();
            model.cpv_chunk(&mut psc, &mut bufs, &toks, pos, s, false)?;
            pos = end;
        }
        let b = model.forward_step(&mut sc, &[pr[n - 1] as i32], &[n - 1], &[s], None)?[0];
        st.push((b, n));
    }
    let mut cks: Vec<RecurCkpt> = (0..nl).map(|_| model.ckpt_alloc()).collect::<Result<_>>()?;
    let save = |model: &FwdModel, psc: &PrefillScratch, cks: &mut Vec<RecurCkpt>| -> Result<Vec<i32>> {
        for s in 0..nl { model.ckpt_save(s, psc, &mut cks[s])?; }
        Ok(model.ple_hist.lock().unwrap().clone())
    };
    let restore = |model: &FwdModel, psc: &mut PrefillScratch, cks: &Vec<RecurCkpt>, hist: &Vec<i32>| -> Result<()> {
        for s in 0..nl { model.ckpt_restore(s, psc, &cks[s])?; }
        *model.ple_hist.lock().unwrap() = hist.clone();
        model.dev.synchronize()?;
        Ok(())
    };
    let tgt = nl / 2; // lane 0's segment index in the batch pass
    let mut all_ok = true;
    crate::dprintln!("XCLASS: {nl} lanes (lane 0 = the request under test, at segment {tgt} of the batch pass); prompts {:?} tokens",
                     prompts.iter().take(nl).map(|p| p.len()).collect::<Vec<_>>());
    for step in 0..steps {
        let (b0, p0) = st[0];
        let hist = save(&model, &psc, &mut cks)?;
        // gt: four decode-class steps
        let mut gt_rows: Vec<Vec<u16>> = Vec::new();
        let mut traj = vec![b0];
        for i in 0..4 {
            let t = model.forward_step(&mut sc, &[traj[i]], &[p0 + i], &[0], None)?[0];
            gt_rows.push(model.logits_host(&sc)?[..v].to_vec());
            traj.push(t);
        }
        restore(&model, &mut psc, &cks, &hist)?;
        // the same four decode steps again, eager, with the per-layer ladder in memory (row 0 of each step)
        let mut dec: Vec<std::collections::HashMap<String, Vec<f32>>> = Vec::new();
        {
            let mut ld = LadderCtx { dir: String::new(), step: 0, layers: (0..model.layers.len()).collect(), only_step: None,
                                     mem: Some(std::sync::Mutex::new(Vec::new())), digest: false, sp: None };
            for i in 0..4 {
                model.forward_step(&mut sc, &[traj[i]], &[p0 + i], &[0], Some(&mut ld))?;
                let taken = std::mem::take(&mut *ld.mem.as_ref().unwrap().lock().unwrap());
                dec.push(taken.into_iter().collect());
            }
        }
        restore(&model, &mut psc, &cks, &hist)?;
        let mk = |slot: usize, b: i32, p: usize, d: Vec<i32>| PackDraft {
            slot, b, p, drafts: d, confs: Vec::new(), spec: false, samp: crate::lane_pack::SegSamp::GREEDY, rq: None, launched: 3, wasted: false };
        let mut line = format!("XCLASS step {step} pos {p0}:");
        let mut single_rows: Vec<Vec<Vec<u16>>> = Vec::new();
        // variants: (mid, exact) = today's wide pass, --exl3-mid-gemm on, and mid + --lane-wide-exact (the per-kernel
        // taps run on the last one)
        for (vi, (mid, exact)) in [(false, false), (true, false), (true, true)].into_iter().enumerate() {
            dres::set_mid(mid);
            dres::set_exact(exact);
            let last = vi == 2;
            // single segment
            let ds = vec![mk(0, b0, p0, traj[1..4].to_vec())];
            if last { wb.dbg = Some(WideDbg { row0: 0, n: 4, taps: Vec::new() }); }
            model.wide_verify(&mut sc, &mut wb, &ds)?;
            let stap = wb.dbg.take();
            let lg = model.dev.dtoh_sync_copy(&wb.logits)?;
            restore(&model, &mut psc, &cks, &hist)?;
            let srows: Vec<Vec<u16>> = (0..4).map(|i| lg[i * v..(i + 1) * v].to_vec()).collect();
            // batch: every lane, lane 0 at segment tgt
            let mut ds = Vec::new();
            let mut order: Vec<usize> = (1..nl).collect();
            order.insert(tgt.min(order.len()), 0);
            let mut row0 = 0;
            let mut tgt_row0 = 0;
            for &s in &order {
                if s == 0 { tgt_row0 = row0; ds.push(mk(0, b0, p0, traj[1..4].to_vec())); }
                else { let (b, p) = st[s]; ds.push(mk(s, b, p, vec![b, b, b])); }
                row0 += 4;
            }
            if last { wb.dbg = Some(WideDbg { row0: tgt_row0, n: 4, taps: Vec::new() }); }
            model.wide_verify(&mut sc, &mut wb, &ds)?;
            let btap = wb.dbg.take();
            let lg = model.dev.dtoh_sync_copy(&wb.logits)?;
            restore(&model, &mut psc, &cks, &hist)?;
            if let (Some(st_), Some(bt_)) = (stap.as_ref(), btap.as_ref()) {
                xclass_taps_report(step, st_, bt_, &dec);
            }
            if step == 0 {
                // the batch pass's time (untapped, median of 7 after 2 warm runs; state restored after each)
                let mut tv = Vec::new();
                for it in 0..9 {
                    model.dev.synchronize()?;
                    let t0 = std::time::Instant::now();
                    model.wide_verify(&mut sc, &mut wb, &ds)?;
                    model.dev.synchronize()?;
                    if it >= 2 { tv.push(t0.elapsed().as_secs_f64() * 1e3); }
                    restore(&model, &mut psc, &cks, &hist)?;
                }
                crate::dprintln!("XCLASS_TIME mid {} exact {}: wide verify of {} lanes x 4 rows = {} rows: {:.2} ms (median of 7)",
                                 mid as u8, exact as u8, ds.len(), ds.len() * 4, med(tv));
            }
            let brows: Vec<Vec<u16>> = (0..4).map(|i| lg[(tgt_row0 + i) * v..(tgt_row0 + i + 1) * v].to_vec()).collect();
            for i in 0..4 {
                let (ga, gm) = top2(&gt_rows[i]);
                let (sa, _) = top2(&srows[i]);
                let (ba, _) = top2(&brows[i]);
                let sb_eq = srows[i] == brows[i];
                if last { all_ok &= sb_eq; }
                line.push_str(&format!(" | mid {} exact {} row {i}: gt margin {gm:.3}; single relL2 {:.2e} argmax {}; batch relL2 {:.2e} argmax {}; single==batch {}",
                                       mid as u8, exact as u8, rel_l2_f16(&srows[i], &gt_rows[i]), if sa == ga { "=" } else { "DIFF" },
                                       rel_l2_f16(&brows[i], &gt_rows[i]), if ba == ga { "=" } else { "DIFF" }, sb_eq));
            }
            single_rows.push(srows);
        }
        crate::dprintln!("{line}");
        // segment-start dependence (mid + exact on): the rows of positions p0+2, p0+3 verified inside one segment that
        // starts at p0 (rows 2, 3 of [b0, d1, d2, d3]) vs at the head of the next segment after a wide round that
        // committed b0 and d1 (its own wide verify + replay commit): [d2, d3] at p0+2
        {
            let srows = single_rows.last().unwrap();
            let a0 = top2(&srows[0]).0 as i32;
            let mut msg = String::new();
            if a0 == traj[1] {
                let (outs, _) = model.wide_verify(&mut sc, &mut wb, &[mk(0, b0, p0, vec![traj[1]])])?;
                let o = &outs[0];
                if o.a == 1 && o.pos_next == p0 + 2 {
                    let (_o2, _) = model.wide_verify(&mut sc, &mut wb, &[mk(0, traj[2], p0 + 2, vec![traj[3], traj[3]])])?;
                    let lg = model.dev.dtoh_sync_copy(&wb.logits)?;
                    for j in 0..2 {
                        let r2 = &lg[j * v..(j + 1) * v];
                        msg.push_str(&format!(" pos {}: row {} of the p0 segment vs row {j} of the p0+2 segment: {} (relL2 {:.2e});",
                                              p0 + 2 + j, j + 2, if r2 == &srows[j + 2][..] { "IDENTICAL" } else { "DIFFER" },
                                              rel_l2_f16(r2, &srows[j + 2])));
                    }
                } else { msg = format!(" first round accepted {} (pos_next {}), skipped", o.a, o.pos_next); }
                restore(&model, &mut psc, &cks, &hist)?;
            } else { msg = " row 0's argmax is not d1, skipped".into(); }
            crate::dprintln!("XCLASS_SEGSTART step {step}:{msg}");
        }
        // advance lane 0 by one decode step (the ground-truth trajectory)
        let t = model.forward_step(&mut sc, &[b0], &[p0], &[0], None)?[0];
        st[0] = (t, p0 + 1);
    }
    dres::set_mid(false);
    dres::set_exact(false);
    crate::dprintln!("RESULT: {}", if all_ok { "XCLASS_DONE batch-independent with --exl3-mid-gemm on --lane-wide-exact on" }
                                   else { "XCLASS_DONE batch-DEPENDENT with --exl3-mid-gemm on --lane-wide-exact on" });
    Ok(all_ok)
}

/// LANEWIDE-EXACT: the cuBLASLt algo a wide pass under --lane-wide-exact pins for the (k, n) projection: the algo of
/// the plan of `pin` rows, run on the pass's m-row layouts (exl3_lt_matmul_pinned). Chosen by --probe-lane-wide-pin
/// (2026-11-13, every m in 1..152 at row offsets 0..16 bit-identical to the pinned 160-row call): the hc down
/// (rw -> lr) only under the 160-row algo (the 32 / 64 / 96-row algos differ at row offsets 3 and 7), everything else
/// (hc up, GDN a / b, PLE key / value) under the 96-row algo (the 32-row algo differs for GDN a / PLE value; the
/// 160-row one for GDN a).
pub(super) fn wide_pin(k: usize, n: usize, rw: usize, lr: usize) -> usize { if k == rw && n == lr { WIDE_RMAX } else { 96 } }

/// `--probe-lane-wide-pin --model-dir <pack> [--probe-lane-wide-reps N]` (GPU, one box): the small fp16 projections of a
/// wide pass (hc down / up, GDN a / b, PLE key / value; the pack's own weights, synthetic rows) through cuBLASLt with
/// the algo PINNED to the plan of WIDE_RMAX rows at every row count m: (1) row-independence: rows [off, off + m) of an
/// m-row call (off 0 and 7) against the same rows of the WIDE_RMAX-row call, bit for bit; (2) the time at 16..128 rows:
/// the fixed-sequence tail GEMM vs cuBLASLt's own plan vs the pinned plan.
pub fn probe_lane_wide_pin(dir: &str, reps: usize) -> Result<bool> {
    let model = FwdModel::load(dir, 1, 4096)?;
    anyhow::ensure!(model.tp.is_none(), "--probe-lane-wide-pin runs on one box");
    let l = Launcher { dev: &model.dev, stream: &model.stream };
    let pm = WIDE_RMAX;
    let psc = model.prefill_scratch(pm + 8)?;
    let rb = model.recon_bufs(&psc);
    let dev = &model.dev;
    let l0 = &model.layers[0];
    let ga = model.layers.iter().find_map(|ly| if let Mixer::Gdn(g) = &ly.mixer { Some(g) } else { None }).context("no GDN layer")?;
    let (h, hcn, lr) = (model.cfg.hidden_size, model.cfg.hc_count.max(1), model.cfg.hc_lowrank);
    let nh = model.tc.lin_num_v_heads;
    let mut shapes: Vec<(&str, &CudaSlice<u16>, usize, usize)> = vec![
        ("hc down", &l0.hc_attn.down, h * hcn, lr), ("hc up", &l0.hc_attn.up, lr, h * hcn), ("gdn a", &ga.a_w, h, nh)];
    if let Some(p) = model.ple.as_ref() { shapes.push(("ple key", &p.key_w, 2560, 10240)); shapes.push(("ple value", &p.value_w, 2560, 2560)); }
    let kmax = shapes.iter().map(|s| s.2).max().unwrap();
    let nmax = shapes.iter().map(|s| s.3).max().unwrap();
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0 };
    let host: Vec<u16> = (0..(pm + 8) * kmax).map(|_| half::f16::from_f32(rnd()).to_bits()).collect();
    let x = dev.htod_sync_copy(&host)?;
    let mut yref = dev.alloc_zeros::<u16>(pm * nmax)?;
    let mut y = dev.alloc_zeros::<u16>((pm + 8) * nmax)?;
    let p = |s: &CudaSlice<u16>| *s.device_ptr() as u64;
    let mut ok = true;
    let reps = reps.max(3);
    for &(name, w, k, n) in &shapes {
        for pin in [wide_pin(k, n, h * hcn, lr)] {
            if exl3_lt_matmul_pinned(&l, p(w), p(&x), p(&yref), k, n, pm, pin, true, rb.ltws, rb.ltws_len).is_err() {
                crate::dprintln!("WIDE_PIN {name} (k {k} n {n}) algo of m {pin}: refused at {pm} rows");
                continue;
            }
            dev.synchronize()?;
            let yr = dev.dtoh_sync_copy(&yref)?;
            let mut cells = Vec::new();
            for m in [1usize, 2, 3, 4, 5, 7, 8, 12, 16, 17, 20, 24, 31, 32, 40, 48, 64, 80, 96, 97, 128, 143, 144] {
                for off in 0usize..=16 {
                    if off + m > pm { continue; }
                    if let Err(e) = exl3_lt_matmul_pinned(&l, p(w), p(&x) + (off * k * 2) as u64, p(&y), k, n, m, pin, true, rb.ltws, rb.ltws_len) {
                        cells.push(format!("m{m}+{off}:REFUSED({e:#})")); continue;
                    }
                    dev.synchronize()?;
                    let yv = dev.dtoh_sync_copy(&y)?;
                    let nd = (0..m * n).filter(|&i| yv[i] != yr[off * n + i]).count();
                    if nd != 0 { cells.push(format!("m{m}+{off}:{nd}diff")); }
                }
            }
            let mut tl = Vec::new();
            for m in [4usize, 16, 32, 48, 64, 96, 128] {
                let mut tv = Vec::new();
                for it in 0..reps + 2 {
                    dev.synchronize()?;
                    let t0 = std::time::Instant::now();
                    for _ in 0..10 { exl3_lt_matmul_pinned(&l, p(w), p(&x), p(&y), k, n, m, pin, true, rb.ltws, rb.ltws_len)?; }
                    dev.synchronize()?;
                    if it >= 2 { tv.push(t0.elapsed().as_secs_f64() * 1e3 / 10.0 * 1e3); }
                }
                tl.push(format!("m{m} {:.0}", med(tv)));
            }
            if !cells.is_empty() { ok = false; }
            crate::dprintln!("WIDE_PIN {name} (k {k} n {n}) algo of m {pin}: {} | us per call {}",
                             if cells.is_empty() { "EXACT at every m and offset".to_string() } else { cells.join(" ") }, tl.join(", "));
        }
        let mut tl = Vec::new();
        for m in [4usize, 16, 32, 48, 64, 96, 128] {
            let mut t = [Vec::new(), Vec::new()];
            for it in 0..reps + 2 {
                for (a, tv) in t.iter_mut().enumerate() {
                    dev.synchronize()?;
                    let t0 = std::time::Instant::now();
                    for _ in 0..10 {
                        if a == 0 { xq_gemm_f16_tail!(&l, &mut y, w, &x, m, n, k)?; }
                        else { exl3_lt_matmul(&l, p(w), p(&x), p(&y), k, n, m, true, rb.ltws, rb.ltws_len)?; }
                    }
                    dev.synchronize()?;
                    if it >= 2 { tv.push(t0.elapsed().as_secs_f64() * 1e3 / 10.0 * 1e3); }
                }
            }
            let [a, b] = t;
            tl.push(format!("m{m}: tail {:.0} / lt {:.0}", med(a), med(b)));
        }
        crate::dprintln!("WIDE_PIN {name} time (us per call, median of {reps}): {}", tl.join("; "));
    }
    crate::dprintln!("RESULT: {}", if ok { "WIDE_PIN_EXACT" } else { "WIDE_PIN_MISMATCH" });
    Ok(ok)
}
