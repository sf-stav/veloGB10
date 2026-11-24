//! Leg PK5: the DSpark draft step at TP slice shapes on ONE box (the per-kernel budget).
//!
//! Rank 0 of a world-N split: layers 40-42, the head and the Markov head as the TP forward cuts them
//! (`RankSrc::pack` = the shipped placement, `Ranges` / `TpLayer` = the call-time views), and every kernel leg FW's
//! `TpSpec::draft` / `TpSpec::main_rows` launch, in their order, with their shapes and bytes. The collectives (the
//! two reduces per stage, the 5 vocab exchanges) are left out: their cost is the transport's (T0 §5.2 prices them
//! separately). Inputs are synthetic and the partial sums are not reduced, so the values are not the model's; the
//! router still picks from real gate weights, and the distinct-expert count per stage is printed next to the MoE
//! time (the MoE's bytes depend on it).
//!
//! `--probe-dsv41-dspark --model-dir <pack> --dsv41-kernel-timing budget[,tp1,tp2,tp4,tp8][,1,2,4]`:
//! per (world, lanes): each op's GPU time (CUDA events around every op of an eager step that the host enqueues
//! behind a 20 ms busy kernel, so launch gaps are the GPU's own), median of 7; the eager step and the same step
//! captured as one CUDA graph and replayed.

use super::vocab::{Round, Shard};
use super::xchg::Local;
use super::{KvArgs, Lane, RepArgs, SetupArgs, BLOCK};
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::fwd::dev::{bf16, f32a, Buf, Ev, Gpu};
use crate::dsv41::fwd::glue::{self, Scr};
use crate::dsv41::fwd::layer::{fmt_of, Bump, Fwd, HC_DIM};
use crate::dsv41::gpu::{attn, mixers as km};
use crate::dsv41::tp::frame::MIX_STRIDE;
use crate::dsv41::tp::fwd::{MixK, Ranges, TpLayer};
use crate::dsv41::tp::wload::RankSrc;
use crate::dsv41::weights::{Accum, BackendKind, DenseWeight, DevPtr, Route, Scratch};
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use std::path::Path;

const SLOT_Q: usize = 512;
const SLOT_S: usize = 16;
const RING: usize = 134;
/// main rows per lane (one verify's taps: k + 1)
const RPL: usize = BLOCK + 1;

/// CUDA events between the ops of one step.
struct Marks { ev: Vec<Ev>, names: Vec<&'static str>, n: usize, on: bool }
impl Marks {
    fn new(cap: usize) -> Result<Marks> { Ok(Marks { ev: (0..cap).map(|_| Ev::new()).collect::<Result<_>>()?, names: vec![""; cap], n: 0, on: false }) }
    fn start(&mut self, g: &Gpu) -> Result<()> { self.n = 0; self.on = true; self.mark(g, "") }
    fn mark(&mut self, g: &Gpu, name: &'static str) -> Result<()> {
        if !self.on { return Ok(()); }
        ensure!(self.n < self.ev.len(), "too many marks");
        self.ev[self.n].rec(g)?;
        self.names[self.n] = name;
        self.n += 1;
        Ok(())
    }
    fn deltas(&mut self) -> Result<Vec<(&'static str, f32)>> {
        self.on = false;
        (1..self.n).map(|i| Ok((self.names[i], self.ev[i].ms_since(&self.ev[i - 1])?))).collect()
    }
}

/// The rank's drafter: stages, head, Markov pieces, state for `cap` lanes.
struct Rank {
    cfg: Dsv41Config,
    world: usize,
    rg: Ranges,
    stages: Vec<TpLayer>,
    head: (DenseWeight, BackendKind),
    embed: (DenseWeight, BackendKind),
    me: (DenseWeight, BackendKind),
    mh: (DenseWeight, BackendKind),
    cw: (DenseWeight, BackendKind),
    main_proj: (DenseWeight, BackendKind),
    sh: Shard,
    cap: usize,
    round: Round,
    x: Local,
    ring_q: Buf, ring_s: Buf, blk_q: Buf, blk_s: Buf,
    eids: Buf, bpos: Buf, bwend: Buf, blane: Buf, pre: Buf, e_rows: Buf, x0: Buf,
    dlog: Buf, cin: Buf, xn: Buf, conf: Buf, freq: Buf,
    fa: Buf, fb: Buf,
    taps: Buf, mrow: Buf, mpos: Buf, mlane: Buf,
    busy: Buf,
    /// run leg PK5's kernels (false: the sequences they replace, the "before" column)
    opt: bool,
}

impl Rank {
    fn load(g: &Gpu, md: &Path, world: usize, cap: usize) -> Result<Rank> {
        let src = RankSrc::pack(md, world, 0, "inner")?;
        let cfg = src.cfg.clone();
        let rg = Ranges::of(&cfg, world, 0);
        let mut stages = Vec::new();
        for s in 0..3 {
            let w = src.load_layer(g, cfg.layers + s).with_context(|| format!("DSpark stage {s}"))?;
            let tl = TpLayer::new(w, &cfg, &rg, world)?;
            if let Some((m, BackendKind::Exl3)) = tl.w.moe.as_ref() {
                crate::dsv41::weights::exl3::moe::prepare_moe(m)?;
                crate::dsv41::weights::exl3::moe::prepare_moe(&tl.shared)?;
            }
            stages.push(tl);
        }
        let glob = src.load_global(g)?;
        let head = glob.head.clone();
        let embed = glob.embed.clone().context("rank 0 holds no embedding")?;
        std::mem::forget(glob); // keep the arena resident for the process (a probe)
        let w2 = &stages[2].w;
        let (me, mh, cw) = (w2.lin(C::DsMarkovEmbed)?.clone(), w2.lin(C::DsMarkovHead)?.clone(), w2.lin(C::DsConfidence)?.clone());
        let main_proj = stages[0].w.lin(C::DsMainProj)?.clone();
        let vr = crate::dsv41::load::splits::vocab_range(cfg.vocab, world, 0);
        let sh = Shard { v0: vr.start, vlocal: vr.len() };
        let (d, r) = (cfg.hidden, cfg.dspark_markov_rank);
        let n = cap * BLOCK;
        let m = cap * RPL;
        let freq = g.buf(128)?;
        let tab = crate::dsv41::rope::table_for(&cfg, crate::dsv41::rope::Class::Plain)?;
        g.h2d(freq.ptr(), super::vocab::as_bytes(&tab))?;
        let fr = n.max(m) * (d + MIX_STRIDE) * 4;
        Ok(Rank {
            round: Round::new(g, cap, RPL, sh.nblk(), 1)?, x: Local { max_bytes: 64 * 1024 },
            ring_q: g.buf(3 * cap * RING * SLOT_Q)?, ring_s: g.buf(3 * cap * RING * SLOT_S)?, blk_q: g.buf(n * SLOT_Q)?, blk_s: g.buf(n * SLOT_S)?,
            eids: g.buf(n * 4)?, bpos: g.buf(n * 4)?, bwend: g.buf(n * 4)?, blane: g.buf(n * 4)?, pre: g.buf(n * 16)?,
            e_rows: g.buf(n * d * 2)?, x0: g.buf(n * HC_DIM * 2)?,
            dlog: g.buf(n * sh.vlocal * 4)?, cin: g.buf(n * (d + r) * 2)?, xn: g.buf(n * d * 2)?, conf: g.buf(n * 4)?, freq,
            fa: g.buf(fr)?, fb: g.buf(fr)?,
            taps: g.buf(m * cfg.dspark_targets.len() * d * 2)?, mrow: g.buf(m * d * 4)?, mpos: g.buf(m * 4)?, mlane: g.buf(m * 4)?,
            busy: g.buf(2 << 30)?, opt: true,
            cfg, world, rg, stages, head, embed, me, mh, cw, main_proj, sh, cap,
        })
    }

    fn ring(&self, s: usize, l: usize) -> (u64, u64) {
        (self.ring_q.ptr() + ((s * self.cap + l) * RING * SLOT_Q) as u64, self.ring_s.ptr() + ((s * self.cap + l) * RING * SLOT_S) as u64)
    }

    fn mix_partial(&self, f: &mut Fwd, mk: MixK, x: u64, rows: usize, rider: u64) -> Result<()> {
        let chunks = km::partial_chunks(mk.klen);
        let rgp = km::pick_rg(rows);
        let ngroups = rows.div_ceil(rgp);
        let part = f.b.take(chunks * rows * 32 * 4)?;
        let cnt = f.b.take(ngroups * 4)?;
        f.g.memset(cnt, ngroups * 4, 0)?;
        let k = self.cfg.hc * self.cfg.hidden;
        let a = km::MixPartialArgs { x, x_ld: k as i64, fn_: mk.fn_ptr, fn_ld: mk.fn_ld as i64, part, chunk_stride: (rows * 32) as i64, row_stride: 32,
                                     rank_part: rider, rp_row_stride: MIX_STRIDE as i64, counters: cnt, k0: mk.k0 as i32, klen: mk.klen as i32, rows: rows as i32, rg: rgp as i32 };
        f.g.launch1("d41_mix_partial", (chunks as u32, ngroups as u32, 1), (128, 1, 1), 0, &a)
    }

    fn mix_finish(&self, f: &mut Fwd, s: usize, attn_side: bool, rider: u64, rows: usize) -> Result<u64> {
        let (sc, bc) = if attn_side { (C::HcAttnScale, C::HcAttnBase) } else { (C::HcFfnScale, C::HcFfnBase) };
        let w = &self.stages[s].w;
        let out = f.b.take(rows * 24 * 4)?;
        let k = self.cfg.hc * self.cfg.hidden;
        let fa = km::MixFinishArgs { part: rider, slot_stride: (rows * MIX_STRIDE) as i64, row_stride: MIX_STRIDE as i64, scale: w.param(sc)?.ptr.0, base: w.param(bc)?.ptr.0,
                                     out, out_ld: 24, nslots: 1, rows: rows as i32, inv_k: 1.0 / k as f32, eps_norm: self.cfg.rms_eps as f32, eps_hc: self.cfg.hc_eps as f32,
                                     sinkhorn_iters: self.cfg.sinkhorn_iters as i32 };
        f.g.launch1("d41_mix_finish", ((rows as u32).div_ceil(8), 1, 1), (128, 1, 1), 0, &fa)?;
        Ok(out)
    }

    /// One stage on the block rows (FW's `TpModel::stage` without its two reduces).
    fn stage(&self, f: &mut Fwd, mk: &mut Marks, s: usize, xs: u64, pre: u64, ps: usize, rows: usize, nl: usize, ids_out: &mut Vec<u64>) -> Result<(u64, u64)> {
        let cfg = &self.cfg;
        let (d, hd, nh, ng) = (cfg.hidden, cfg.head_dim, self.rg.nh, self.rg.ng);
        let tl = &self.stages[s];
        let (fa, fb) = (self.fa.ptr(), self.fb.ptr());
        let g = f.g;
        self.mix_partial(f, tl.mix_a, xs, rows, fa + (rows * d * 4) as u64)?;
        mk.mark(g, "attn mix_partial")?;
        let ax = f.hc_pre_norm(xs, pre, ps, tl.w.param(C::AttnNorm)?, rows)?;
        mk.mark(g, "attn hc_pre_norm")?;
        let (ql, olr) = (cfg.q_lora_rank, cfg.o_lora_rank);
        let qa = f.b.take(rows * ql * 2)?;
        f.lin_rows("pk5.wq_a", &tl.w, C::AttnWqA, None, ax, d, d, rows, qa, ql, false)?;
        mk.mark(g, "wq_a (replicated)")?;
        let qn = tl.w.param(C::AttnQNorm)?;
        let qr = f.b.take(rows * ql * 2)?;
        let a = km::RmsNormArgs { x: qa, x_ld: ql as i64, w: qn.ptr.0, y: qr, y_ld: ql as i64, w_fmt: fmt_of(qn)?, dim: ql as i32, rows: rows as i32, eps: cfg.rms_eps as f32, inv_dim: 1.0 / ql as f32 };
        g.launch1("d41_rmsnorm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        mk.mark(g, "q_norm")?;
        let q = f.b.take(rows * nh * hd * 2)?;
        glue::linear(g, "pk5.wq_b", &tl.wq_b, &bf16(qr, rows, ql, ql), &{ let mut a = bf16(q, rows, nh * hd, nh * hd); a.cols = nh * hd; a }, Accum::Overwrite, &f.scr)?;
        mk.mark(g, "wq_b (heads)")?;
        glue::rope_rows(g, q, nh * hd, self.bpos.ptr(), rows, nh, hd, false, &f.rope.q.plain)?;
        mk.mark(g, "rope q")?;
        let raw = f.b.take(rows * SLOT_Q * 2)?;
        f.lin_rows("pk5.blk_wkv", &tl.w, C::AttnWkv, None, ax, d, d, rows, raw, SLOT_Q, false)?;
        mk.mark(g, "wkv block (replicated)")?;
        let norm = tl.w.param(C::AttnKvNorm)?;
        g.launch1("d41_ds_kv_rows", (rows as u32, 1, 1), (128, 1, 1), 0,
                  &KvArgs { kv_raw: raw, norm_w: norm.ptr.0, inv_freq: self.freq.ptr(), pos: self.bpos.ptr(), chunk_q: self.blk_q.ptr(), chunk_s: self.blk_s.ptr(),
                            rows: rows as i32, norm_fmt: fmt_of(norm)?, eps: cfg.rms_eps as f32, ..Default::default() })?;
        mk.mark(g, "kv rows (block keys)")?;
        let lanes: Vec<attn::LaneSrc> = (0..nl).map(|ln| {
            let (rq, rs) = self.ring(s, ln);
            attn::LaneSrc { swa_q: DevPtr(rq), swa_s: DevPtr(rs), slots: RING, lin: None, kv: None,
                            ext: Some((DevPtr(self.blk_q.ptr() + (ln * BLOCK * SLOT_Q) as u64), DevPtr(self.blk_s.ptr() + (ln * BLOCK * SLOT_S) as u64), BLOCK)) }
        }).collect();
        let o = f.b.take(rows * nh * hd * 2)?;
        let err = f.b.take(4)?;
        g.memset(err, 4, 0)?;
        let call = attn::AttnCall { rows, heads: nh, q: DevPtr(q), q_stride: nh * hd, sink: DevPtr(tl.sink), pos: DevPtr(self.bpos.ptr()),
                                    win_end: DevPtr(self.bwend.ptr()), lane: DevPtr(self.blane.ptr()), sel: DevPtr(0), sel_stride: 0, sel_cap: 0, n_sel: DevPtr(0),
                                    lanes, out: DevPtr(o), out_stride: nh * hd, out_q: None, out_s: None, epi: attn::EPI_ROPE | attn::EPI_AQ,
                                    rope_freq: f.rope.ka.plain, err: DevPtr(err) };
        super::dsattn::launch(g, &call)?;
        mk.mark(g, "attention (ring window + block keys)")?;
        let gin = nh * hd / ng;
        let mid = f.b.take(rows * ng * olr * 2)?;
        for i in 0..ng {
            f.lin_rows("pk5.wo_a", &tl.w, C::AttnWoA, Some((self.rg.g0 + i) as u16), o + (i * gin * 2) as u64, nh * hd, gin, rows, mid + (i * olr * 2) as u64, ng * olr, false)?;
        }
        mk.mark(g, "wo_a (groups)")?;
        let mut ya = f32a(fa, rows, d, d);
        ya.cols = d;
        glue::linear(g, "pk5.wo_b", &tl.wo_b, &bf16(mid, rows, ng * olr, ng * olr), &ya, Accum::Overwrite, &f.scr)?;
        mk.mark(g, "wo_b (fp32 partial)")?;
        let rec_a = self.mix_finish(f, s, true, fa + (rows * d * 4) as u64, rows)?;
        mk.mark(g, "attn mix_finish")?;
        let ao = f.b.take(rows * d * 2)?;
        glue::add_round(g, fa, 0, ao, rows, d)?;
        mk.mark(g, "attn add_round")?;
        let hm = f.hc_post(ao, xs, rec_a, rows)?;
        mk.mark(g, "attn hc_post")?;
        self.mix_partial(f, tl.mix_f, hm, rows, fb + (rows * d * 4) as u64)?;
        mk.mark(g, "ffn mix_partial")?;
        let fin = f.hc_pre_norm(hm, rec_a, 24, tl.w.param(C::FfnNorm)?, rows)?;
        mk.mark(g, "ffn hc_pre_norm")?;
        let (rid, rw) = f.router(&tl.w, fin, rows)?;
        mk.mark(g, "router")?;
        ids_out.push(rid);
        g.memset(fb, rows * d * 4, 0)?;
        let (mw, mb) = tl.w.moe.as_ref().context("no routed experts")?;
        let route = Route { ids: DevPtr(rid), weights: DevPtr(rw), top_k: mw.top_k };
        let scr = Scratch { ptr: DevPtr(f.scr.ptr), bytes: f.scr.bytes };
        match mb {
            BackendKind::Exl3 => {
                crate::dsv41::weights::exl3::moe::moe_ex(&g.l(), mw, Some(&tl.shared), &bf16(fin, rows, d, d), &route, &f32a(fb, rows, d, d), scr, crate::dsv41::weights::exl3::moe::MoePath::Auto)?;
            }
            b => anyhow::bail!("the budget harness runs the TP forward's EXL3 MoE call (moe_ex, routed + shared); stage {s} has {b:?} experts"),
        }
        mk.mark(g, "MoE routed + shared")?;
        let rec_f = self.mix_finish(f, s, false, fb + (rows * d * 4) as u64, rows)?;
        mk.mark(g, "ffn mix_finish")?;
        let fo = f.b.take(rows * d * 2)?;
        glue::add_round(g, fb, 0, fo, rows, d)?;
        mk.mark(g, "ffn add_round")?;
        let xo = f.hc_post(fo, hm, rec_f, rows)?;
        mk.mark(g, "ffn hc_post")?;
        Ok((xo, rec_f))
    }

    /// The draft step of `nl` lanes with `k` drafts (FW's `draft_body`, collectives omitted). Returns each stage's router ids.
    fn draft(&self, f: &mut Fwd, mk: &mut Marks, nl: usize, k: usize) -> Result<Vec<u64>> {
        let g = f.g;
        let cfg = &self.cfg;
        let (d, r, vl) = (cfg.hidden, cfg.dspark_markov_rank, self.sh.vlocal);
        let rows = nl * BLOCK;
        f.b.reset();
        g.launch1("d41_ds_setup", ((rows as u32).div_ceil(32), 1, 1), (32, 1, 1), 0,
                  &SetupArgs { lanes: self.round.lanes.ptr(), eids: self.eids.ptr(), bpos: self.bpos.ptr(), bwend: self.bwend.ptr(), blane: self.blane.ptr(),
                               ids: self.round.ids.ptr(), pre: self.pre.ptr(), nlanes: nl as i32, noise: cfg.dspark_noise_token as i32 })?;
        mk.mark(g, "setup")?;
        glue::backend(self.embed.1).gather(&g.l(), &self.embed.0, DevPtr(self.eids.ptr()), rows, &bf16(self.e_rows.ptr(), rows, d, d))?;
        g.launch1("d41_ds_rep", ((d as u32).div_ceil(256), rows as u32, 1), (256, 1, 1), 0, &RepArgs { e: self.e_rows.ptr(), x: self.x0.ptr(), rows: rows as i32, dim: d as i32 })?;
        mk.mark(g, "embed + hc copies")?;
        let (mut xs, mut pre, mut ps) = (self.x0.ptr(), self.pre.ptr(), 4usize);
        let mut ids = Vec::new();
        for s in 0..3 {
            let (xo, rec) = self.stage(f, mk, s, xs, pre, ps, rows, nl, &mut ids)?;
            xs = xo;
            pre = rec;
            ps = 24;
        }
        let ld = d + r;
        let nw = self.stages[2].w.param(C::DsNorm)?;
        let a = km::HcPreNormArgs { x: xs, x_ld: HC_DIM as i64, pre, pre_stride: 24, w: nw.ptr.0, y_norm: self.xn.ptr(), yn_ld: d as i64,
                                    y_raw: self.cin.ptr(), yr_ld: ld as i64, w_fmt: fmt_of(nw)?, dim: d as i32, rows: rows as i32, eps: cfg.rms_eps as f32, inv_dim: 1.0 / d as f32 };
        g.launch1("d41_hc_pre_norm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        mk.mark(g, "head hc_pre_norm")?;
        let mut ya = f32a(self.dlog.ptr(), rows, vl, vl);
        ya.cols = vl;
        if !(self.opt && super::chain::plain_rows_gemv(g, &self.head, &bf16(self.xn.ptr(), rows, d, d), &ya, Accum::Overwrite)?) {
            glue::linear(g, "pk5.head", &self.head, &bf16(self.xn.ptr(), rows, d, d), &ya, Accum::Overwrite, &f.scr)?;
        }
        mk.mark(g, "tied head (vocab rows)")?;
        let ch = super::chain::Chain { me: &self.me, mh: &self.mh, cin: self.cin.ptr(), ld, d, r, logits: self.dlog.ptr(), vl, sh: self.sh, table: self.round.ids.ptr() };
        if self.opt { super::chain::run(g, &f.scr, &ch, &self.round, &self.x, nl, k, &mut |n| mk.mark(g, n))?; }
        else { super::chain::run_reference(g, &f.scr, &ch, &self.round, &self.x, nl, k, &mut |n| mk.mark(g, n))?; }
        if self.opt { super::chain::confidence(g, &f.scr, &self.cw, self.cin.ptr(), rows, ld, self.conf.ptr())?; }
        else { glue::linear(g, "pk5.confidence", &self.cw, &bf16(self.cin.ptr(), rows, ld, ld), &f32a(self.conf.ptr(), rows, 1, 1), Accum::Overwrite, &f.scr)?; }
        mk.mark(g, "confidence head")?;
        Ok(ids)
    }

    /// The main rows of a round (6 per lane): main_proj's K-slice partial (FW: it rides layer 39's reduce), main_norm,
    /// each stage's wkv + kv rows into the rings.
    fn main_rows(&self, f: &mut Fwd, mk: &mut Marks, rows: usize) -> Result<()> {
        let g = f.g;
        let cfg = &self.cfg;
        let (d, eps) = (cfg.hidden, cfg.rms_eps as f32);
        let kin = cfg.dspark_targets.len() * d;
        f.b.reset();
        let mut ya = f32a(self.mrow.ptr(), rows, d, d);
        ya.cols = d;
        let mut xa = bf16(self.taps.ptr(), rows, kin, kin);
        if self.world > 1 { if let crate::dsv41::weights::Shard::In(rg) = &self.main_proj.0.shard { xa = bf16(self.taps.ptr() + rg.start * 2, rows, (rg.end - rg.start) as usize, kin); } }
        glue::linear(g, "pk5.main_proj", &self.main_proj, &xa, &ya, Accum::Overwrite, &f.scr)?;
        mk.mark(g, "main_proj (K slice, fp32 partial)")?;
        let mxb = f.b.take(rows * d * 2)?;
        glue::add_round(g, self.mrow.ptr(), 0, mxb, rows, d)?;
        let nw = self.stages[0].w.param(C::DsMainNorm)?;
        let mx = f.b.take(rows * d * 2)?;
        let a = km::RmsNormArgs { x: mxb, x_ld: d as i64, w: nw.ptr.0, y: mx, y_ld: d as i64, w_fmt: fmt_of(nw)?, dim: d as i32, rows: rows as i32, eps, inv_dim: 1.0 / d as f32 };
        g.launch1("d41_rmsnorm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        mk.mark(g, "main add_round + main_norm")?;
        for s in 0..3 {
            let w = &self.stages[s].w;
            let raw = f.b.take(rows * SLOT_Q * 2)?;
            f.lin_rows("pk5.main_wkv", w, C::AttnWkv, None, mx, d, d, rows, raw, SLOT_Q, false)?;
            let norm = w.param(C::AttnKvNorm)?;
            let (rq, rs) = self.ring(s, 0);
            g.launch1("d41_ds_kv_rows", (rows as u32, 1, 1), (128, 1, 1), 0,
                      &KvArgs { kv_raw: raw, norm_w: norm.ptr.0, inv_freq: self.freq.ptr(), pos: self.mpos.ptr(), lane: self.mlane.ptr(), ring_q: rq, ring_s: rs,
                                ring_lq: (RING * SLOT_Q) as i64, ring_ls: (RING * SLOT_S) as i64, rows: rows as i32, norm_fmt: fmt_of(norm)?, eps, ..Default::default() })?;
        }
        mk.mark(g, "3 x (wkv + kv rows -> rings)")?;
        Ok(())
    }

    /// A ~20 ms GPU-busy op ahead of a measured step (the host enqueues the step while it runs).
    fn busy(&self, g: &Gpu) -> Result<()> { g.memset(self.busy.ptr(), self.busy.bytes, 1) }
}

fn med(mut v: Vec<f32>) -> f32 { v.sort_by(|a, b| a.total_cmp(b)); if v.is_empty() { 0.0 } else { v[v.len() / 2] } }

/// Bytes a linear reads per call: its resident tensors, scaled to the call's view when the tensors are whole and
/// the shard names a slice (the TP4 attention views).
fn lin_bytes(w: &DenseWeight) -> f64 {
    let tot: f64 = w.tensors.iter().map(|t| t.bytes as f64).sum();
    let tr = w.tensors.iter().find(|t| matches!(t.role, crate::dsv41::census::classify::Role::Trellis | crate::dsv41::census::classify::Role::Weight));
    let Some(t) = tr else { return tot };
    let exl3 = t.role == crate::dsv41::census::classify::Role::Trellis;
    let (held_out, held_in) = if exl3 { (t.shape[1] * 16, t.shape[0] * 16) } else { (t.shape[0], t.shape.get(1).copied().unwrap_or(1)) };
    match &w.shard {
        crate::dsv41::weights::Shard::Out(r) if held_out == w.out => tot * (r.end - r.start) as f64 / w.out as f64,
        crate::dsv41::weights::Shard::In(r) if held_in == w.inn => tot * (r.end - r.start) as f64 / w.inn as f64,
        _ => tot,
    }
}

/// One measurement: per-op medians (name, ms, count), the eager step, the graph (eager, replay, nodes), distinct experts
/// per stage, the MoE bytes of the three stages.
struct Meas { ops: Vec<(&'static str, f32, usize)>, eager: f32, ge: f32, gr: f32, nodes: usize, distinct: Vec<usize>, moe_bytes: f64 }

fn measure(g: &Gpu, rk: &Rank, f: &mut Fwd, mk: &mut Marks, nl: usize) -> Result<Meas> {
    let lanes_v: Vec<Lane> = (0..nl).map(|l| Lane { seed: 7 + l as u64, inv_t: 1.0, vbase: 501, k: 5, wend: 505, tok: 1000 + 17 * l as i32, active: 1, ..Default::default() }).collect();
    let mut acc: Vec<(&'static str, Vec<f32>)> = Vec::new();
    let mut tot_e = Vec::new();
    let mut distinct = vec![0usize; 3];
    let mut moe_bytes = 0f64;
    for it in 0..8 {
        rk.round.set_lanes(g, &lanes_v)?;
        g.sync()?;
        rk.busy(g)?;
        mk.start(g)?;
        rk.main_rows(f, mk, nl * RPL)?;
        let ids = rk.draft(f, mk, nl, BLOCK)?;
        let dl = mk.deltas()?;
        if it == 0 {
            for (s, &p) in ids.iter().enumerate() {
                let mut u: Vec<i32> = g.down::<i32>(p, nl * BLOCK * 3)?;
                u.sort_unstable();
                u.dedup();
                distinct[s] = u.len();
                let tl = &rk.stages[s];
                let mw = &tl.w.moe.as_ref().context("experts")?.0;
                moe_bytes += u.iter().map(|&e| mw.experts[e as usize].iter().map(lin_bytes).sum::<f64>()).sum::<f64>();
                moe_bytes += tl.shared.experts[0].iter().map(lin_bytes).sum::<f64>();
            }
            continue;
        }
        tot_e.push(dl.iter().map(|x| x.1).sum::<f32>());
        for (i, (n, t)) in dl.into_iter().enumerate() {
            if acc.len() <= i { acc.push((n, Vec::new())); }
            acc[i].1.push(t);
        }
    }
    let mut ops: Vec<(&'static str, f32, usize)> = Vec::new();
    for (n, v) in &acc {
        let m_ = med(v.clone());
        if let Some(e) = ops.iter_mut().find(|e| e.0 == *n) { e.1 += m_; e.2 += 1; } else { ops.push((n, m_, 1)); }
    }
    let (ge, gr, nodes) = graph_time(g, rk, f, mk, nl, &lanes_v)?;
    Ok(Meas { ops, eager: med(tot_e), ge, gr, nodes, distinct, moe_bytes })
}

/// The budget tables. `worlds` (1, 2, 4, 8), `lanes` (1..=4).
pub fn run(g: &Gpu, md: &Path, worlds: &[usize], lanes: &[usize], say: &mut dyn FnMut(String)) -> Result<()> {
    let cap = lanes.iter().copied().max().unwrap_or(1).max(1);
    for &w in worlds {
        let t0 = std::time::Instant::now();
        let mut rk = Rank::load(g, md, w, cap)?;
        let cfg = rk.cfg.clone();
        let scr = g.buf(256 << 20)?;
        let mut f = Fwd { wide_min: 0, g, cfg: &cfg, scr: Scr { ptr: scr.ptr(), bytes: 256 << 20 }, b: Bump::new(g, 512 << 20)?, rope: crate::dsv41::fwd::rope_set(&cfg)?, checks: false };
        let mut mk = Marks::new(256)?;
        say(format!("PK5 budget: TP{w} rank 0 (heads {}, o_groups {}, hc K {}, vocab rows {}..{}), head {:?}, loaded in {:.1} s",
                    rk.rg.nh, rk.rg.ng, rk.rg.klen, rk.sh.v0, rk.sh.v0 + rk.sh.vlocal, rk.head.1, t0.elapsed().as_secs_f64()));
        // synthetic taps; the rings start zeroed (valid FP8 keys: the attention's cost does not depend on the values)
        let d3 = 3 * cfg.hidden;
        let m = cap * RPL;
        let tap: Vec<u16> = (0..m * d3).map(|i| crate::dsv41::fwd::cmp::f32_to_bf16_bits(((i * 7919 % 1000) as f32 - 500.0) * 2e-3)).collect();
        g.h2d(rk.taps.ptr(), super::vocab::as_bytes(&tap))?;
        let pos: Vec<i32> = (0..m).map(|i| 500 + (i % RPL) as i32).collect();
        let ln: Vec<i32> = (0..m).map(|i| (i / RPL) as i32).collect();
        g.h2d(rk.mpos.ptr(), super::vocab::as_bytes(&pos))?;
        g.h2d(rk.mlane.ptr(), super::vocab::as_bytes(&ln))?;
        // bytes per call of the byte-bound ops (rank 0's held / viewed bytes)
        let st = &rk.stages[0];
        let wsum = |c: C| -> f64 { st.w.lin(c).map(|l| lin_bytes(&l.0)).unwrap_or(0.0) };
        let woa: f64 = (0..rk.rg.ng).map(|i| st.w.lin_sub(C::AttnWoA, (rk.rg.g0 + i) as u16).map(|l| lin_bytes(&l.0)).unwrap_or(0.0)).sum();
        let bytes_of = |n: &str| -> f64 {
            match n {
                "wq_a (replicated)" => 3.0 * wsum(C::AttnWqA),
                "wq_b (heads)" => 3.0 * lin_bytes(&st.wq_b.0),
                "wkv block (replicated)" => 3.0 * wsum(C::AttnWkv),
                "wo_a (groups)" => 3.0 * woa,
                "wo_b (fp32 partial)" => 3.0 * lin_bytes(&st.wo_b.0),
                "router" => 3.0 * st.w.weight_tensor(C::GateWeight).map(|t| t.bytes as f64).unwrap_or(0.0),
                "tied head (vocab rows)" => lin_bytes(&rk.head.0),
                "main_proj (K slice, fp32 partial)" => lin_bytes(&rk.main_proj.0),
                "3 x (wkv + kv rows -> rings)" => 3.0 * wsum(C::AttnWkv),
                "markov head GEMV (+= logits)" | "markov chain step (mstep + vscan + vfin)" => BLOCK as f64 * lin_bytes(&rk.mh.0),
                _ => 0.0,
            }
        };
        for &nl in lanes {
            if nl > cap { continue; }
            rk.opt = false;
            let b0 = measure(g, &rk, &mut f, &mut mk, nl)?;
            rk.opt = true;
            let b1 = measure(g, &rk, &mut f, &mut mk, nl)?;
            say(format!("  TP{w} lanes {nl} (block rows {}, main rows {}): distinct experts per stage {:?} of {} picks (MoE bytes {:.1} MB over 3 stages); per op, GPU ms \
                         (median of 7, summed over stages / chain steps) BEFORE (DS's sequences) | AFTER (PK5) | MB read | GB/s after",
                        nl * BLOCK, nl * RPL, b1.distinct, nl * BLOCK * 3, b1.moe_bytes / 1e6));
            let mut names: Vec<&'static str> = b0.ops.iter().map(|e| e.0).collect();
            for e in &b1.ops { if !names.contains(&e.0) { names.push(e.0); } }
            for n in names {
                let t0 = b0.ops.iter().find(|e| e.0 == n);
                let t1 = b1.ops.iter().find(|e| e.0 == n);
                let mb = if n == "MoE routed + shared" { b1.moe_bytes } else { bytes_of(n) };
                let tt = t1.or(t0).map(|e| e.1).unwrap_or(0.0);
                let gbs = if mb > 0.0 && tt > 0.0 { format!("{:>7.1}", mb / (tt as f64 * 1e-3) / 1e9) } else { "      -".into() };
                say(format!("    {:<42} {:>9} | {:>9} | {:>8} | {gbs}", n, t0.map(|e| format!("{:.4} x{}", e.1, e.2)).unwrap_or("-".into()),
                            t1.map(|e| format!("{:.4} x{}", e.1, e.2)).unwrap_or("-".into()), if mb > 0.0 { format!("{:.1}", mb / 1e6) } else { "-".into() }));
            }
            let (s0, s1): (f32, f32) = (b0.ops.iter().map(|e| e.1).sum(), b1.ops.iter().map(|e| e.1).sum());
            say(format!("    {:<42} {:>9.4} | {:>9.4}   (eager step {:.4} | {:.4} ms)", "SUM", s0, s1, b0.eager, b1.eager));
            say(format!("    one CUDA graph (main rows + draft step): BEFORE {} nodes, replay {:.4} ms (eager {:.4}) | AFTER {} nodes, replay {:.4} ms (eager {:.4})",
                        b0.nodes, b0.gr, b0.ge, b1.nodes, b1.gr, b1.ge));
        }
    }
    Ok(())
}

fn graph_time(g: &Gpu, rk: &Rank, f: &mut Fwd, mk: &mut Marks, nl: usize, lanes: &[Lane]) -> Result<(f32, f32, usize)> {
    let st = g.stream as sys::CUstream;
    let (e0, e1) = (Ev::new()?, Ev::new()?);
    let mut te = Vec::new();
    for _ in 0..7 {
        rk.round.set_lanes(g, lanes)?;
        g.sync()?;
        rk.busy(g)?;
        e0.rec(g)?;
        rk.main_rows(f, mk, nl * RPL)?;
        rk.draft(f, mk, nl, BLOCK)?;
        e1.rec(g)?;
        te.push(e1.ms_since(&e0)?);
    }
    rk.round.set_lanes(g, lanes)?;
    g.sync()?;
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    ensure!(unsafe { sys::cuStreamBeginCapture_v2(st, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) } == sys::CUresult::CUDA_SUCCESS, "begin capture");
    let r = rk.main_rows(f, mk, nl * RPL).and_then(|_| rk.draft(f, mk, nl, BLOCK).map(|_| ()));
    let e = unsafe { sys::cuStreamEndCapture(st, &mut graph) };
    r.context("capturing the draft step")?;
    ensure!(e == sys::CUresult::CUDA_SUCCESS, "end capture: {e:?}");
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    ensure!(unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) } == sys::CUresult::CUDA_SUCCESS, "instantiate");
    let mut nn = 0usize;
    unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut nn) };
    let mut tg = Vec::new();
    for _ in 0..8 {
        rk.round.set_lanes(g, lanes)?;
        g.sync()?;
        e0.rec(g)?;
        ensure!(unsafe { sys::cuGraphLaunch(exec, st) } == sys::CUresult::CUDA_SUCCESS, "graph launch");
        e1.rec(g)?;
        tg.push(e1.ms_since(&e0)?);
    }
    unsafe { sys::cuGraphExecDestroy(exec); sys::cuGraphDestroy(graph); }
    tg.remove(0);
    Ok((med(te), med(tg), nn))
}
