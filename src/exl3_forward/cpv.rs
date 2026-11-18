//! COPYPF1 (2026-11-12): the `--copy-span-prefill` device step (prefill-width copy verify) and the
//! `--probe-copy-prefill` probe. The host logic (schedule, accept prefix, decision, counters) is
//! `crate::copy_prefill`; the serving arm is `Exl3Scheduler::step_cpv`.
//!
//! One step, for one lane at TP=1 (`b` unconsumed at position `p`, its draft tap in row `tap_row` of
//! the lane's taps window, `offered` = c_1..c_N copied from the context):
//!  1. the head-fill carry := the lane's tap (the streams after p-1, which predicted b), so the chunk's
//!     head fill writes the draft head's KV row p from (tap, b) — the pairing every round uses;
//!  2. checkpoint the lane's running state (WP16 `ckpt_save`: GDN S, conv, PLE ring, head-fill carry,
//!     the host PLE n-gram pair) before the chunk;
//!  3. `prefill_chunk([b, c_1..c_N], p)` — the SERVED prompt-reading path — with the stash attached:
//!     its final-mixed rows and its last row's streams are copied out before the head fill reuses
//!     the buffers;
//!  4. judge the rows in groups of <= 16 through the verify's own tail on the decode scratch: lm_head
//!     (`exl3_chain`, the m <= 16 batch-invariant body: a row's logits do not depend on its group),
//!     `argmax_rows`, `xq_sample_rows` with the lane's (seed, ctr + row) — the sample-then-compare rule
//!     of the verify. Stops after the first group holding a disagreement;
//!  5. a = the agreeing prefix; on a disagreement (a < N) restore the checkpoint and prefill the a+1
//!     rows [b, c_1..c_a] again (positional planes — trunk / head KV, indexer keys — are rewritten
//!     for those rows; rows past them are rewritten before any read, as after a rejected verify);
//!  6. the streams of the last committed row (the tap that predicted the new bonus) go to row 0 of the
//!     lane's taps window. Emitted: c_1..c_a and the model's choice at row a.
//!
//! TP (COPYPF1-TP, 2026-11-12): every rank runs this step at the same program point with the same inputs (the
//! serving arm agrees them first). The chunk is the served TP prefill (sequence-parallel rows gathered for the
//! stash: x when the rows are judged, the streams for the tap); the judge is the verify's own TP tail (the
//! vocab-parallel greedy key merge, or the shard row all-gather for sampled rows, else the replicated head), so
//! every rank holds the same choices; the serving arm agrees the accept count before the re-run.
//!
//! Gated on hardware at TP=1 (one Spark: probe exact, file returns exact) and at TP=2 / TP=4 (greedy thinking-off
//! identity with the no-speculation reference, file returns, stops and limits, zero lockstep failures). The probe
//! below loads one box: it does not run under TP.

use super::*;

/// The prefill's final mix gathers x on every rank (TP-SP) when an attached stash wants the rows.
pub(super) fn cpv_wants_rows(psc: &PrefillScratch) -> bool {
    psc.cpv.as_ref().map_or(false, |s| s.want_rows)
}

/// What `prefill_chunk` copies out while it is attached (`PrefillScratch::cpv`).
pub struct CpvStash {
    /// [cap][hidden] f16 bits: the chunk's final-mixed rows (the lm_head input), when `want_rows`
    pub(super) xs: CudaSlice<u16>,
    /// [hc * hidden] f32: the streams of the chunk's last row (the tap that predicted the next token)
    tap: CudaSlice<f32>,
    cap: usize,
    want_rows: bool,
}

/// The step's persistent buffers (one set per scheduler; allocated on the first step).
pub struct CpvBufs {
    pub(super) stash: Option<CpvStash>,
    ck: Option<RecurCkpt>,
}

impl CpvBufs {
    pub fn new() -> CpvBufs { CpvBufs { stash: None, ck: None } }
}

impl Default for CpvBufs {
    fn default() -> Self { Self::new() }
}

/// One step's outcome.
#[derive(Clone, Debug)]
pub struct CpvOut {
    /// copied tokens the model's choices agreed with
    pub a: usize,
    /// c_1..c_a, then the model's choice at row a (a + 1 tokens)
    pub emitted: Vec<i32>,
    /// the per-row choices that were computed (rows 0.. up to the end of the group holding row a)
    pub choices: Vec<i32>,
    /// rows prefilled again after a disagreement (0 = all agreed)
    pub rerun_rows: usize,
    /// rows that went through the lm_head
    pub judged: usize,
}

impl FwdModel {
    fn cpv_stash_new(&self, cap: usize) -> Result<CpvStash> {
        let h = self.cfg.hidden_size;
        let rw = self.cfg.hc_count.max(1) * h;
        Ok(CpvStash { xs: self.dev.alloc_zeros::<u16>(cap.max(1) * h)?, tap: self.dev.alloc_zeros::<f32>(rw)?, cap, want_rows: false })
    }

    /// The hook `prefill_chunk` runs when a stash is attached: rows 0..c of psc.x (final-mixed, f16) and
    /// row c-1 of psc.resid (the streams), on the compute stream before the head fill.
    pub(super) fn cpv_stash_rows(&self, l: &Launcher, psc: &mut PrefillScratch, c: usize) -> Result<()> {
        let h = self.cfg.hidden_size;
        let rw = self.cfg.hc_count.max(1) * h;
        let st = psc.cpv.as_mut().context("cpv_stash_rows without a stash")?;
        if st.want_rows {
            anyhow::ensure!(c <= st.cap, "copy-span-prefill: chunk of {c} rows > stash of {}", st.cap);
            let n = c * h;
            xqlaunch!(l, "xq_copy_row", ((((n as u64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut st.xs, &psc.x, 0i64, n as i32))?;
        }
        xqlaunch!(l, "xq_copy_f32", ((((rw as u64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut st.tap, &psc.resid, rw as i64, 0i64, ((c - 1) * rw) as i64))?;
        Ok(())
    }

    /// Rows judged per lm_head group: <= 16 (the batch-invariant m <= 16 body) and what the decode
    /// scratch's verify tail holds (its logits / argmax / sampler / chain planes).
    pub(super) fn cpv_group_rows(&self, sc: &Scratch) -> usize {
        let (h, v) = (self.cfg.hidden_size, self.cfg.vocab_size);
        let g = 16usize.min(sc.argmax.len()).min(sc.samp.len() / 8).min(sc.logits.len() / v.max(1))
            .min(sc.x.len() / h.max(1)).min(sc.chain_yraw.len() / v.max(1)).min(sc.am_cnt.len()).min(sc.toks.len());
        // TP: the vocab-parallel tails' planes (the shard rows, the merged keys)
        match self.lm_head_vp.as_ref() {
            Some(vp) => g.min(sc.vp_rows.len() / (vp.n_sh as usize).max(1)).min(sc.vp_keys.len()),
            None => g,
        }
    }

    /// Judge rows r0..r0+m of the stash: the verify's tail (lm_head, argmax, sampler) at m rows on the
    /// decode scratch, the sampler rows staged from `samp` (row i = the lane's counter + r0 + i).
    pub(super) fn cpv_head_group(&self, sc: &mut Scratch, xs: &CudaSlice<u16>, r0: usize, m: usize,
                      samp: &[Option<SampParams>]) -> Result<Vec<i32>> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let (h, v) = (self.cfg.hidden_size, self.cfg.vocab_size);
        anyhow::ensure!(samp.len() == m, "copy-span-prefill: {} sampler rows for a group of {m}", samp.len());
        // the match rule (no real-q staging: a copied token has no draft law), as the copy round stages it
        self.set_spec_sampling(sc, samp, None)?;
        let n = m * h;
        xqlaunch!(l, "xq_copy_row", ((((n as u64) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.x, xs, (r0 * h) as i64, n as i32))?;
        // the verify's own lm_head tail (verify_kernels): at TP the vocab-parallel greedy key merge or the shard row
        // all-gather give every rank the same ids / full rows; at TP=1 (no shard) the replicated head, as before
        if self.vp_greedy(sc, m) {
            self.vp_greedy_tail(&l, sc, m)?;
        } else {
            if self.vp_gather(sc, m) {
                self.vp_gather_tail(&l, sc, m)?;
            } else {
                exl3_chain(&l, &self.lm_head, &sc.x, &mut sc.chain_xh, &mut sc.chain_yraw, &mut sc.logits, m, &sc.chain_ws)?;
            }
            self.argmax_rows(&l, sc, m, v)?;
            // sampled rows overwrite their id (zero-param rows exit at once): the verify's sample-then-compare
            xqlaunch!(l, "xq_sample_rows", (m as u32, 1, 1), (1024, 1, 1), 0,
                      (&mut sc.argmax, &sc.logits, v as i32, &sc.samp))?;
        }
        self.dsync()?;
        let ids = self.dtoh(&sc.argmax)?;
        Ok(ids[..m].to_vec())
    }

    /// Prefill `toks` at `pos0` of `slot` with the stash attached (`want_rows`: also the final-mixed rows).
    pub(super) fn cpv_chunk(&self, psc: &mut PrefillScratch, bufs: &mut CpvBufs, toks: &[i32], pos0: usize, slot: usize,
                 want_rows: bool) -> Result<()> {
        if bufs.stash.as_ref().map_or(true, |s| s.cap < psc.c) {
            bufs.stash = Some(self.cpv_stash_new(psc.c)?);
        }
        let mut st = bufs.stash.take().unwrap();
        st.want_rows = want_rows;
        psc.cpv = Some(st);
        let r = self.prefill_chunk(psc, toks, pos0, slot, false, None);
        bufs.stash = psc.cpv.take();
        r
    }

    /// The `--copy-span-prefill` step (module doc). `samp` = the sampler rows for chunk rows 0..=N (the
    /// lane's counter + row; None = greedy). The caller has staged no penalty and no mask (penalized / schema /
    /// logprob lanes never reach it). `agree(a, choices)`: the caller's lockstep point between the judge and the
    /// re-run (TP: every rank's accept count and choices compared, the link aborted on a difference, before any
    /// rank runs a re-run chunk whose width depends on them; TP=1: nothing).
    #[allow(clippy::too_many_arguments)]
    pub fn copy_prefill_verify(&self, sc: &mut Scratch, psc: &mut PrefillScratch, bufs: &mut CpvBufs, slot: usize,
                               b: i32, p: usize, tap_row: usize, offered: &[i32], samp: &[Option<SampParams>],
                               agree: &mut dyn FnMut(usize, &[i32]) -> Result<()>)
                               -> Result<CpvOut> {
        let n = offered.len();
        let c = n + 1;
        anyhow::ensure!(n >= 1 && c <= psc.c && samp.len() >= c && p + c + 1 < self.max_pos,
                        "copy-span-prefill: {n} copied tokens at p={p} (prefill rows {}, sampler rows {}, KV window {})",
                        psc.c, samp.len(), self.max_pos);
        anyhow::ensure!(tap_row < sc.vrows, "copy-span-prefill: tap row {tap_row} out of range");
        // the judge applies no penalty and no schema mask (penalized lanes never reach this step; schema lanes never
        // speculate); a staged penalty here would be a caller bug, so refuse it rather than judge without it
        anyhow::ensure!(!sc.pen_live, "copy-span-prefill: penalties are staged (never for this step)");
        let g = self.cpv_group_rows(sc);
        anyhow::ensure!(g >= 1, "copy-span-prefill: the decode scratch holds no verify row");
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let he = self.cfg.hc_count.max(1) * self.cfg.hidden_size;
        let grid = ((((he as u64) + 255) / 256) as u32, 1, 1);
        let base = self.tap_base(sc, slot)?;
        // the draft chain's clamp (an id outside the vocabulary becomes 0; a context id never is)
        let v = self.cfg.vocab_size as i32;
        let mut toks: Vec<i32> = Vec::with_capacity(c);
        toks.push(b);
        toks.extend(offered.iter().map(|&d| if d < 0 || d >= v { 0 } else { d }));
        // 1. the carry the head fill pairs with row p: the lane's tap that predicted b
        if psc.hfill_carry.len() >= he {
            xqlaunch!(l, "xq_copy_f32", grid, (256, 1, 1), 0,
                      (&mut psc.hfill_carry, &sc.taps_keep, he as i64, 0i64, (base + tap_row * he) as i64))?;
        }
        // 2. the running state before the chunk
        if bufs.ck.is_none() { bufs.ck = Some(self.ckpt_alloc()?); }
        self.ckpt_save(slot, psc, bufs.ck.as_mut().unwrap())?;
        // 3. the chunk, through the served prefill
        self.cpv_chunk(psc, bufs, &toks, p, slot, true)?;
        // 4. judge, group by group; stop after the group that holds the first disagreement
        let mut choices: Vec<i32> = Vec::with_capacity(c);
        let mut r0 = 0usize;
        while r0 < c {
            let m = g.min(c - r0);
            let xs = &bufs.stash.as_ref().unwrap().xs;
            let ids = self.cpv_head_group(sc, xs, r0, m, &samp[r0..r0 + m])?;
            choices.extend_from_slice(&ids);
            r0 += m;
            if crate::copy_prefill::accept_prefix(&choices, &toks[1..]) < choices.len().min(n) { break; }
        }
        let a = crate::copy_prefill::accept_prefix(&choices, &toks[1..]);
        let emitted = crate::copy_prefill::emitted(&toks[1..], &choices, a)
            .with_context(|| format!("copy-span-prefill: row {a} was not judged ({} rows)", choices.len()))?;
        agree(a, &choices)?;
        // 5. a disagreement: back to the state before the chunk, then exactly the agreed rows again
        let rerun = crate::copy_prefill::rerun_rows(a, n);
        if rerun > 0 {
            self.ckpt_restore(slot, psc, bufs.ck.as_ref().unwrap())?;
            self.cpv_chunk(psc, bufs, &toks[..rerun], p, slot, false)?;
        }
        // 6. the tap that predicted the new bonus -> row 0 of the lane's window
        let tap = &bufs.stash.as_ref().unwrap().tap;
        xqlaunch!(l, "xq_copy_f32", grid, (256, 1, 1), 0,
                  (&mut sc.taps_keep, tap, he as i64, base as i64, 0i64))?;
        self.dsync()?;
        let judged = choices.len();
        Ok(CpvOut { a, emitted, choices, rerun_rows: rerun, judged })
    }
}

// ---------------------------------------------------------------------------------------------
// --probe-copy-prefill
// ---------------------------------------------------------------------------------------------

/// Largest minus second-largest logit of one f16 row (the near-tie margin), and the argmax.
fn top2_margin(row: &[u16]) -> (usize, f32) {
    let (mut i1, mut v1, mut v2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &b) in row.iter().enumerate() {
        let x = half::f16::from_bits(b).to_f32();
        if !x.is_finite() { continue; }
        if x > v1 { v2 = v1; v1 = x; i1 = i; } else if x > v2 { v2 = x; }
    }
    (i1, v1 - v2)
}

/// Reset slot 0 and prefill the prompt prefix `ids[..n-1]` in chunks of `cw` (the seam token is
/// ids[n-1]), the stash attached to every chunk: afterwards its tap = the streams after n-2.
fn cpv_prefix(model: &FwdModel, psc: &mut PrefillScratch, bufs: &mut CpvBufs, ids: &[u32], cw: usize) -> Result<()> {
    model.reset_slot(0)?;
    let end_all = ids.len() - 1;
    let mut pos = 0usize;
    while pos < end_all {
        let end = (pos + cw).min(end_all);
        let toks: Vec<i32> = ids[pos..end].iter().map(|&t| t as i32).collect();
        model.cpv_chunk(psc, bufs, &toks, pos, 0, false)?;
        pos = end;
    }
    model.dev.synchronize()?;
    Ok(())
}

/// The stash's tap -> the head-fill carry and row 0 of slot 0's taps window (what the serving lane holds
/// at a round boundary).
fn cpv_tap_into_lane(model: &FwdModel, sc: &mut Scratch, psc: &mut PrefillScratch, bufs: &CpvBufs) -> Result<()> {
    let l = Launcher { dev: &model.dev, stream: &model.stream };
    let he = model.cfg.hc_count.max(1) * model.cfg.hidden_size;
    let grid = ((((he as u64) + 255) / 256) as u32, 1, 1);
    let tap = &bufs.stash.as_ref().context("no stash")?.tap;
    let base = model.tap_base(sc, 0)?;
    xqlaunch!(l, "xq_copy_f32", grid, (256, 1, 1), 0, (&mut sc.taps_keep, tap, he as i64, base as i64, 0i64))?;
    if psc.hfill_carry.len() >= he {
        xqlaunch!(l, "xq_copy_f32", grid, (256, 1, 1), 0, (&mut psc.hfill_carry, tap, he as i64, 0i64, 0i64))?;
    }
    model.dev.synchronize()?;
    Ok(())
}

/// `--probe-copy-prefill --model-dir <pack> [--prompt-ids <file>] [--probe-copy-prefill-n N] [--prefill-chunk C]`
/// (GPU, one box). The copy = the prompt's own greedy DECODE continuation d_0..d_{N-1} (so the decoding
/// path agrees with every copied token by construction). Arms, each from a fresh slot 0 with the prompt
/// prefix ids[..n-1] prefilled at width C and b = ids[n-1] at p = n-1:
///  (c) decode: forward_step over b, d_0, .., d_{N-1} (greedy; the choices d_0..d_N and their margins);
///  (a) the step: `copy_prefill_verify` with offered = d_0..d_{N-1} (greedy) -> a, choices, state A;
///  (b1) reference rows: [b, d_0..d_{N-1}] as ONE prompt chunk, every row judged -> choices must equal the
///       step's (EXACT: same function, same width) and are compared with (c) (the near-tie divergence);
///  (b2) reference state: [b, d_0..d_{a-1}] as one prompt chunk -> GDN / conv / PLE / carry checkpoint,
///       trunk KV rows 0..p+a and the tap must equal state A bit for bit.
/// Returns Ok(true) iff (b1) and (b2) are exact.
pub fn probe_copy_prefill(a: &FwdArgs, n_req: usize, brk: usize) -> Result<bool> {
    let model = FwdModel::load(&a.dir, 1, a.max_pos)?;
    anyhow::ensure!(model.tp.is_none(), "--probe-copy-prefill runs on one box");
    let mut sc = Scratch::new(&model.dev, &model.cfg, 16, model.cfg.rotary_dim, model.max_pos)?;
    let ids = gate_prompt_ids(a)?;
    let n = ids.len();
    anyhow::ensure!(n >= 2, "--probe-copy-prefill: the prompt needs at least 2 tokens (got {n})");
    let nst = n_req.max(1);
    let cw = a.prefill_chunk.unwrap_or(2048).max(nst + 1);
    let (b, p) = (ids[n - 1] as i32, n - 1);
    anyhow::ensure!(p + nst + 2 < model.max_pos, "--probe-copy-prefill: prompt {n} + {nst} copied tokens do not fit \
                    --max-seq-len {}", model.max_pos);
    let v = model.cfg.vocab_size;
    let he = model.cfg.hc_count.max(1) * model.cfg.hidden_size;
    let mut psc = model.prefill_scratch(cw)?;
    let mut bufs = CpvBufs::new();
    model.set_sampling(&mut sc, &[None])?;
    model.set_penalties(&mut sc, &[None])?;
    crate::dprintln!("COPY_PREFILL probe: prompt {n} tokens (seam token {b} at p={p}), copy = the greedy decode continuation of \
{nst} tokens, prefill chunk {cw}, judge groups of {}", model.cpv_group_rows(&sc));

    // (c) the decoding path
    cpv_prefix(&model, &mut psc, &mut bufs, &ids, cw)?;
    let t_dec = std::time::Instant::now();
    let mut dec: Vec<i32> = Vec::with_capacity(nst + 1);
    let mut dmar: Vec<f32> = Vec::with_capacity(nst + 1);
    let mut t = b;
    // rows whose chosen id is not the host's first-index argmax of the logits row read back (should be 0;
    // informative: it says the margins below describe the rows that made the choices)
    let mut am_mis = 0usize;
    for i in 0..=nst {
        let x = model.forward_step(&mut sc, &[t], &[p + i], &[0], None)?[0];
        let lg = model.logits_host(&sc)?;
        let (am, mg) = top2_margin(&lg[..v]);
        am_mis += (am as i32 != x) as usize;
        dec.push(x);
        dmar.push(mg);
        t = x;
    }
    let dec_ms = t_dec.elapsed().as_secs_f64() * 1e3;
    let mut offered: Vec<i32> = dec[..nst].to_vec();
    // --probe-copy-prefill-break K (1-based): copied token K is replaced by another id, so the step must stop at
    // row K-1 at the latest and run its roll-back + re-run (rows 0..K-1 still see the decode arm's exact inputs)
    let cmp_rows = if brk >= 1 && brk <= nst {
        offered[brk - 1] = (offered[brk - 1] + 1) % (v as i32);
        crate::dprintln!("COPY_PREFILL probe: copied token {brk} replaced ({} -> {}): the stretch must disagree by row {}",
                         dec[brk - 1], offered[brk - 1], brk - 1);
        brk
    } else { nst + 1 };

    // (a) the step
    cpv_prefix(&model, &mut psc, &mut bufs, &ids, cw)?;
    cpv_tap_into_lane(&model, &mut sc, &mut psc, &bufs)?;
    let samp = vec![None; nst + 1];
    let t_step = std::time::Instant::now();
    let out = model.copy_prefill_verify(&mut sc, &mut psc, &mut bufs, 0, b, p, 0, &offered, &samp, &mut |_, _| Ok(()))?;
    let step_ms = t_step.elapsed().as_secs_f64() * 1e3;
    let mut ck_a = model.ckpt_alloc()?;
    model.ckpt_save(0, &psc, &mut ck_a)?;
    let kv_a = snap_slot0_kv(&model, p + out.a + 1)?;
    let base = model.tap_base(&sc, 0)?;
    let tap_a: Vec<u32> = model.dev.dtoh_sync_copy(&sc.taps_keep)?[base..base + he].iter().map(|x| x.to_bits()).collect();
    crate::dprintln!("COPY_PREFILL step: kept {} of {nst} copied tokens, emitted {} tokens, judged {} rows, re-ran {} rows, \
{step_ms:.1} ms (one run, not a benchmark; the decode arm took {dec_ms:.1} ms for {} steps)",
                     out.a, out.emitted.len(), out.judged, out.rerun_rows, nst + 1);

    // (b1) the same rows as one prompt chunk, every row judged
    cpv_prefix(&model, &mut psc, &mut bufs, &ids, cw)?;
    cpv_tap_into_lane(&model, &mut sc, &mut psc, &bufs)?;
    let mut toks = vec![b];
    toks.extend_from_slice(&offered);
    model.cpv_chunk(&mut psc, &mut bufs, &toks, p, 0, true)?;
    let g = model.cpv_group_rows(&sc);
    let mut refc: Vec<i32> = Vec::with_capacity(nst + 1);
    let mut pmar: Vec<f32> = Vec::with_capacity(nst + 1);
    let mut r0 = 0usize;
    while r0 < nst + 1 {
        let m = g.min(nst + 1 - r0);
        let xs = &bufs.stash.as_ref().unwrap().xs;
        let ids_g = model.cpv_head_group(&mut sc, xs, r0, m, &vec![None; m])?;
        let lg = model.logits_rows(&sc, 0, m)?;
        for i in 0..m {
            let (am, mg) = top2_margin(&lg[i * v..(i + 1) * v]);
            am_mis += (am as i32 != ids_g[i]) as usize;
            pmar.push(mg);
        }
        refc.extend_from_slice(&ids_g);
        r0 += m;
    }
    let choices_ok = out.choices.len() <= refc.len() && out.choices[..] == refc[..out.choices.len()];
    let mut want_emit = offered[..out.a].to_vec();
    want_emit.push(refc[out.a]);
    let emit_ok = out.emitted == want_emit;
    // (b2) the committed rows as one prompt chunk (the same chunk as (b1) when every row agreed)
    if out.a < nst {
        cpv_prefix(&model, &mut psc, &mut bufs, &ids, cw)?;
        cpv_tap_into_lane(&model, &mut sc, &mut psc, &bufs)?;
        model.cpv_chunk(&mut psc, &mut bufs, &toks[..out.a + 1], p, 0, false)?;
    }
    model.dev.synchronize()?;
    let mut ck_b = model.ckpt_alloc()?;
    model.ckpt_save(0, &psc, &mut ck_b)?;
    let kv_b = snap_slot0_kv(&model, p + out.a + 1)?;
    let tap_b: Vec<u32> = model.dev.dtoh_sync_copy(&bufs.stash.as_ref().unwrap().tap)?.iter().map(|x| x.to_bits()).collect();
    let d = model.ckpt_diff(&ck_a, &ck_b)?;
    let kv_ok = kv_a.len() == kv_b.len()
        && kv_a.iter().zip(&kv_b).all(|(x, y)| x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
    let tap_ok = tap_a == tap_b;
    crate::dprintln!("COPY_PREFILL exactness vs the same tokens read as prompt: choices {} ({} rows), emitted {}, state: {}, \
trunk KV rows 0..{} {}, tap {}",
                     if choices_ok { "EQUAL" } else { "DIFFER" }, out.choices.len(), if emit_ok { "EQUAL" } else { "DIFFER" },
                     d.line(), p + out.a + 1, if kv_ok { "EQUAL" } else { "DIFFER" }, if tap_ok { "EQUAL" } else { "DIFFER" });

    // (c) vs (b1): where the prompt-reading path chooses differently from the decoding path (same inputs)
    let diffs: Vec<usize> = (0..cmp_rows).filter(|&i| refc[i] != dec[i]).collect();
    let min_d = dmar[..cmp_rows].iter().copied().fold(f32::INFINITY, f32::min);
    crate::dprintln!("COPY_PREFILL prompt-reading vs decoding path: {} of {} positions choose differently (first at row {:?}); \
smallest decode margin over those rows {min_d:.4} logit (rows whose id is not the host argmax of the row read back: {am_mis}, \
expected 0)", diffs.len(), cmp_rows, diffs.first());
    // the differing rows, and every near-tie row (decode margin < 1 logit) whether or not it differs: the margin at a
    // divergence seen in serving is read from here (gate session 2026-11-12)
    let shown: Vec<usize> = (0..cmp_rows).filter(|&i| refc[i] != dec[i] || dmar[i] < 1.0).take(16).collect();
    for &i in shown.iter() {
        crate::dprintln!("  row {i} (position {}): decode chose {} (margin {:.4}), prefill chose {} (margin {:.4}){}",
                         p + i + 1, dec[i], dmar[i], refc[i], pmar[i], if refc[i] != dec[i] { "  DIFFER" } else { "" });
    }
    let exact = choices_ok && emit_ok && d.exact() && kv_ok && tap_ok;
    crate::dprintln!("RESULT: {}", if exact { "COPY_PREFILL_EXACT" } else { "COPY_PREFILL_MISMATCH" });
    Ok(exact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top2_margin_finds_the_gap_and_skips_non_finite() {
        let row: Vec<u16> = [1.0f32, 3.5, 3.0, f32::NAN, -2.0].iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let (i, m) = top2_margin(&row);
        assert_eq!(i, 1);
        assert!((m - 0.5).abs() < 1e-6);
        let tie: Vec<u16> = [2.0f32, 2.0].iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        assert_eq!(top2_margin(&tie), (0, 0.0), "first index wins a tie, margin 0");
    }
}
