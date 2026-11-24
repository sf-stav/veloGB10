//! The DSpark drafter on the GPU (inference/model.py DSparkBlock 1103-1159, Transformer.forward_spec 1278-1286;
//! semantics pinned by leg OD's oracle, `PLAN/DSV41_OD_REPORT.md` §10).
//!
//! Two calls per round, both graph-capturable (every per-round value is read from device memory):
//! * [`Drafter::main_rows`] — the drafter's KEYS. For every main row the round produced (the verify rows of the
//!   lane, or the prompt's last rows after a prefill): `main_x = main_norm(main_proj(tap))`, tap = the bf16 mean
//!   over the 4 hc copies of the stream at the input of layers 37 | 38 | 39 (concatenated, 15,360 wide); then for
//!   each stage `act_quant(rope(kv_norm(wkv(main_x)), p))` into the lane's ring at slot p % 134. Rejected verify
//!   rows land in slots the next round overwrites before any window reads them (134 = 128 + 6, X-DI §4.1).
//! * [`Drafter::block`] — the DRAFTS. The block `[tok, noise x 4]` at positions wend + 1 .. wend + 5 runs through
//!   the 3 stages (SL's per-layer pieces: mHC mixes with the pre-mix one sublayer late, q path with plain RoPE, the
//!   block's own keys as KA's extra keys over the lane's 128-row window, wo, the 128-expert top-3 MoE + shared),
//!   then `hc_pre` with the last stage's FFN pre-mix (the un-normed output feeds the confidence head), the stage-2
//!   norm, the tied head in f32 on all 5 rows, and the sequential Markov chain: for i in 0..k,
//!   `logits[i] += markov_head(markov_embed[ids[i]])`, `ids[i + 1]` = a draw from the processed distribution of
//!   `logits[i]` (greedy: argmax) through the vocab exchange. Confidence = `proj . [hc_pre out ; markov embed]`
//!   (a raw f32 logit; the policy applies the sigmoid).
//!
//! TP: the stage pieces need the trunk's reduces (the TP forward's, not here); the head, the Markov head and the
//! draws are vocab-parallel through [`super::xchg::VocabExchange`] (this rank's shard of the logits only).

use super::vocab::{Round, Shard};
use super::xchg::VocabExchange;
use super::{KvArgs, RepArgs, SetupArgs, BLOCK};
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::fwd::dev::{bf16, f32a, Buf, Gpu};
use crate::dsv41::fwd::layer::{fmt_of, Fwd, HC_DIM};
use crate::dsv41::fwd::receipts;
use crate::dsv41::fwd::weights::{load_layer, LayerW, Pack};
use crate::dsv41::gpu::{attn, mixers as km};
use crate::dsv41::weights::{Accum, BackendKind, DenseWeight, DevPtr, DevTensor, Scratch, Shard as WShard};
use anyhow::{ensure, Context, Result};
use std::os::unix::fs::FileExt;

/// A global linear (the tied head, the tied embedding) on the device.
pub struct GlobalW { pub w: (DenseWeight, BackendKind), _buf: Buf }

/// Load the global linear of class `class` (head / embed) with this rank's vocab rows as its shard.
pub fn load_global(g: &Gpu, p: &Pack, class: C, sh: Shard) -> Result<GlobalW> {
    let c = &p.c;
    let m = c.modules.iter().find(|m| m.class == class && m.layer.is_none()).with_context(|| format!("{}: the pack has no {}", c.dir.display(), class.name()))?;
    let pl = p.planned(&m.key).with_context(|| format!("{}: not in the weight plan", m.key))?;
    let align = |x: u64| (x + 255) / 256 * 256;
    let total: u64 = m.tensors.iter().map(|&ti| align(c.entries[ti as usize].bytes)).sum();
    let buf = g.buf(total as usize)?;
    let mut off = 0u64;
    let mut tensors = Vec::new();
    for &ti in &m.tensors {
        let e = &c.entries[ti as usize];
        let path = c.dir.join(&c.files[e.file as usize]);
        let f = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut v = vec![0u8; e.bytes as usize];
        f.read_exact_at(&mut v, e.offset).with_context(|| format!("read {}", e.name))?;
        g.h2d(buf.ptr() + off, &v)?;
        tensors.push(DevTensor { role: e.role, ptr: DevPtr(buf.ptr() + off), dtype: e.dtype.clone(), shape: e.shape.clone(), bytes: e.bytes });
        off += align(e.bytes);
    }
    let shard = if sh.v0 == 0 && sh.vlocal as u64 == pl.out { WShard::Whole } else { WShard::Out(sh.v0 as u64..(sh.v0 + sh.vlocal) as u64) };
    let w = DenseWeight { key: m.key.clone(), class: m.class, layer: None, sub: None, format: m.format.clone(), out: pl.out, inn: pl.inn, shard, tensors };
    Ok(GlobalW { w: (w, pl.backend), _buf: buf })
}

/// The drafter's resident weights and per-lane state.
pub struct Drafter {
    pub cfg: Dsv41Config,
    pub stages: Vec<LayerW>,
    pub head: GlobalW,
    pub embed: GlobalW,
    pub shard: Shard,
    pub cap: usize,
    pub max_main: usize,
    ring_q: Buf,
    ring_s: Buf,
    blk_q: Buf,
    blk_s: Buf,
    eids: Buf,
    pub bpos: Buf,
    bwend: Buf,
    blane: Buf,
    pre: Buf,
    e_rows: Buf,
    x0: Buf,
    /// [cap][5][vlocal] f32: the head's logits, then + the Markov bias row by row (the drafts' distributions)
    pub logits: Buf,
    /// [cap][5][dim + markov rank] bf16: the confidence head's input
    pub cin: Buf,
    xn: Buf,
    /// [cap][5] f32: raw confidence logits
    pub conf: Buf,
    freq: Buf,
    /// [cap * 5] bf16 rows of the block's final hc_pre output (= cin's first dim columns; kept for the per-op check)
    pub dim: usize,
    pub rank: usize,
}

const SLOT_Q: usize = 512;
const SLOT_S: usize = 16;

impl Drafter {
    /// Load the 3 stages, the tied head and embedding (this rank's vocab shard) and size the state for `cap` lanes
    /// and `max_main` main rows per call.
    pub fn load(g: &Gpu, p: &Pack, cap: usize, max_main: usize, sh: Shard) -> Result<Drafter> {
        let cfg = p.c.cfg.clone();
        ensure!(cfg.dspark_block == BLOCK, "DSpark block {} (this build drafts blocks of {BLOCK})", cfg.dspark_block);
        ensure!(cfg.head_dim == SLOT_Q && cfg.swa_slots() == 134, "head_dim {} / ring slots {}", cfg.head_dim, cfg.swa_slots());
        ensure!(cap >= 1 && cap <= attn::MAX_LANES, "drafter lanes {cap} (1..={})", attn::MAX_LANES);
        let mut stages = Vec::new();
        for s in 0..3 { stages.push(load_layer(g, p, cfg.layers + s).with_context(|| format!("DSpark stage {s}"))?); }
        let head = load_global(g, p, C::Head, sh)?;
        let embed = load_global(g, p, C::Embed, Shard { v0: 0, vlocal: cfg.vocab })?;
        let (d, r) = (cfg.hidden, cfg.dspark_markov_rank);
        let n = cap * BLOCK;
        let freq = g.buf(128)?;
        let tab = crate::dsv41::rope::table_for(&cfg, crate::dsv41::rope::Class::Plain)?;
        g.h2d(freq.ptr(), super::vocab::as_bytes(&tab))?;
        Ok(Drafter {
            ring_q: g.buf(3 * cap * 134 * SLOT_Q)?, ring_s: g.buf(3 * cap * 134 * SLOT_S)?,
            blk_q: g.buf(n * SLOT_Q)?, blk_s: g.buf(n * SLOT_S)?,
            eids: g.buf(n * 4)?, bpos: g.buf(n * 4)?, bwend: g.buf(n * 4)?, blane: g.buf(n * 4)?, pre: g.buf(n * 16)?,
            e_rows: g.buf(n * d * 2)?, x0: g.buf(n * HC_DIM * 2)?,
            logits: g.buf(n * sh.vlocal * 4)?, cin: g.buf(n * (d + r) * 2)?, xn: g.buf(n * d * 2)?, conf: g.buf(n * 4)?,
            freq, dim: d, rank: r, cfg, stages, head, embed, shard: sh, cap, max_main,
        })
    }

    /// Zero every lane's rings (a new request in a lane must not see the previous one's keys).
    pub fn reset_rings(&self, g: &Gpu) -> Result<()> {
        g.memset(self.ring_q.ptr(), self.ring_q.bytes, 0)?;
        g.memset(self.ring_s.ptr(), self.ring_s.bytes, 0)
    }
    pub fn ring(&self, stage: usize, lane: usize) -> (u64, u64) {
        (self.ring_q.ptr() + ((stage * self.cap + lane) * 134 * SLOT_Q) as u64, self.ring_s.ptr() + ((stage * self.cap + lane) * 134 * SLOT_S) as u64)
    }

    /// The keys of `rows` main rows: taps bf16 [rows][3 * dim], positions `mpos` i32 [rows], lanes `mlane` i32 [rows].
    pub fn main_rows(&self, f: &mut Fwd, taps: u64, rows: usize, mpos: u64, mlane: u64) -> Result<()> {
        ensure!(rows >= 1 && rows <= self.max_main, "drafter main rows {rows} (1..={})", self.max_main);
        let (d, eps) = (self.dim, self.cfg.rms_eps as f32);
        let w0 = &self.stages[0];
        let kin = d * self.cfg.dspark_targets.len();
        let y = f.b.take(rows * d * 2)?;
        f.lin_rows("ds.main_proj", w0, C::DsMainProj, None, taps, kin, kin, rows, y, d, false)?;
        let nw = w0.param(C::DsMainNorm)?;
        let mx = f.b.take(rows * d * 2)?;
        let a = km::RmsNormArgs { x: y, x_ld: d as i64, w: nw.ptr.0, y: mx, y_ld: d as i64, w_fmt: fmt_of(nw)?, dim: d as i32, rows: rows as i32, eps, inv_dim: 1.0 / d as f32 };
        f.g.launch1("d41_rmsnorm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("ds.main_norm", "d41_rmsnorm", rows);
        for (s, w) in self.stages.iter().enumerate() {
            let raw = f.b.take(rows * SLOT_Q * 2)?;
            f.lin_rows("ds.main_wkv", w, C::AttnWkv, None, mx, d, d, rows, raw, SLOT_Q, false)?;
            let norm = w.param(C::AttnKvNorm)?;
            let (rq, rs) = self.ring(s, 0);
            let k = KvArgs { kv_raw: raw, norm_w: norm.ptr.0, inv_freq: self.freq.ptr(), pos: mpos, lane: mlane, ring_q: rq, ring_s: rs,
                             ring_lq: (134 * SLOT_Q) as i64, ring_ls: (134 * SLOT_S) as i64, rows: rows as i32, norm_fmt: fmt_of(norm)?, eps, ..Default::default() };
            f.g.launch1("d41_ds_kv_rows", (rows as u32, 1, 1), (128, 1, 1), 0, &k)?;
        }
        receipts::kernel("ds.ring_write", "d41_ds_kv_rows", rows);
        Ok(())
    }

    /// The block of `nlanes` lanes (lane state in `rd`: wend, tok, sampling) and `k` (<= 5) chain steps.
    /// `forced`: the Markov chain reads its inputs from `rd.fids` (the gate's forced chain) instead of its own picks.
    /// Returns each stage's (output stream bf16 [rows][4][dim], router ids i32 [rows][top-k]) in the bump arena,
    /// valid until its next reset (the gate's per-stage diagnosis reads them).
    pub fn block(&self, f: &mut Fwd, rd: &Round, x: &dyn VocabExchange, nlanes: usize, k: usize, forced: bool) -> Result<Vec<(u64, u64)>> {
        ensure!(nlanes >= 1 && nlanes <= self.cap && k <= BLOCK, "drafter block: {nlanes} lanes, {k} drafts");
        let g = f.g;
        let (d, r) = (self.dim, self.rank);
        let rows = nlanes * BLOCK;
        let cfg = &self.cfg;
        g.launch1("d41_ds_setup", ((rows as u32).div_ceil(32), 1, 1), (32, 1, 1), 0,
                  &SetupArgs { lanes: rd.lanes.ptr(), eids: self.eids.ptr(), bpos: self.bpos.ptr(), bwend: self.bwend.ptr(), blane: self.blane.ptr(),
                               ids: rd.ids.ptr(), pre: self.pre.ptr(), nlanes: nlanes as i32, noise: cfg.dspark_noise_token as i32 })?;
        // forward_embed: the tied embedding of [tok, noise x 4], copied to the 4 hc streams
        let eb = crate::dsv41::fwd::glue::backend(self.embed.w.1);
        eb.gather(&g.l(), &self.embed.w.0, DevPtr(self.eids.ptr()), rows, &bf16(self.e_rows.ptr(), rows, d, d))?;
        g.launch1("d41_ds_rep", ((d as u32).div_ceil(256), rows as u32, 1), (256, 1, 1), 0, &RepArgs { e: self.e_rows.ptr(), x: self.x0.ptr(), rows: rows as i32, dim: d as i32 })?;
        receipts::kernel("ds.embed", "d41x_gather+d41_ds_rep", rows);
        let mut xs = self.x0.ptr();
        let (mut pre, mut pre_stride) = (self.pre.ptr(), 4usize);
        let (h, hd) = (cfg.heads, cfg.head_dim);
        let mut rec_last = 0u64;
        let mut outs = Vec::with_capacity(3);
        for (s, w) in self.stages.iter().enumerate() {
            let rec_a = f.mix(w, true, xs, rows)?;
            let ax = f.hc_pre_norm(xs, pre, pre_stride, w.param(C::AttnNorm)?, rows)?;
            let (_qr, q) = f.q_path(w, ax, self.bpos.ptr(), rows, false)?;
            let raw = f.b.take(rows * SLOT_Q * 2)?;
            f.lin_rows("ds.blk_wkv", w, C::AttnWkv, None, ax, d, d, rows, raw, SLOT_Q, false)?;
            let norm = w.param(C::AttnKvNorm)?;
            g.launch1("d41_ds_kv_rows", (rows as u32, 1, 1), (128, 1, 1), 0,
                      &KvArgs { kv_raw: raw, norm_w: norm.ptr.0, inv_freq: self.freq.ptr(), pos: self.bpos.ptr(), chunk_q: self.blk_q.ptr(), chunk_s: self.blk_s.ptr(),
                                rows: rows as i32, norm_fmt: fmt_of(norm)?, eps: cfg.rms_eps as f32, ..Default::default() })?;
            receipts::kernel("ds.block_keys", "d41_ds_kv_rows", rows);
            let lanes: Vec<attn::LaneSrc> = (0..nlanes).map(|l| {
                let (rq, rs) = self.ring(s, l);
                attn::LaneSrc { swa_q: DevPtr(rq), swa_s: DevPtr(rs), slots: 134, lin: None, kv: None,
                                ext: Some((DevPtr(self.blk_q.ptr() + (l * BLOCK * SLOT_Q) as u64), DevPtr(self.blk_s.ptr() + (l * BLOCK * SLOT_S) as u64), BLOCK)) }
            }).collect();
            let o = f.b.take(rows * h * hd * 2)?;
            let err = f.b.take(4)?;
            g.memset(err, 4, 0)?;
            let call = attn::AttnCall { rows, heads: h, q: DevPtr(q), q_stride: h * hd, sink: w.param(C::AttnSink)?.ptr, pos: DevPtr(self.bpos.ptr()),
                                        win_end: DevPtr(self.bwend.ptr()), lane: DevPtr(self.blane.ptr()), sel: DevPtr(0), sel_stride: 0, sel_cap: 0, n_sel: DevPtr(0),
                                        lanes, out: DevPtr(o), out_stride: h * hd, out_q: None, out_s: None, epi: attn::EPI_ROPE | attn::EPI_AQ,
                                        rope_freq: f.rope.ka.plain, err: DevPtr(err) };
            super::dsattn::launch(g, &call)?;
            receipts::kernel("ds.attn (ring window + block keys)", super::dsattn::ENTRY, rows);
            if f.checks { ensure!(g.down::<i32>(err, 1)?[0] == 0, "DSpark stage {s}: attention error word"); }
            let ao = f.wo(w, o, rows)?;
            let hm = f.hc_post(ao, xs, rec_a, rows)?;
            let rec_f = f.mix(w, false, hm, rows)?;
            let fxv = f.hc_pre_norm(hm, rec_a, 24, w.param(C::FfnNorm)?, rows)?;
            let (ids, wts) = f.router(w, fxv, rows)?;
            let fo = f.ffn(w, fxv, rows, ids, wts)?;
            xs = f.hc_post(fo, hm, rec_f, rows)?;
            outs.push((xs, ids));
            pre = rec_f;
            pre_stride = 24;
            rec_last = rec_f;
        }
        // forward_head: hc_pre with the last stage's FFN pre-mix (un-normed -> the confidence input), the stage-2 norm
        let w2 = &self.stages[2];
        let nw = w2.param(C::DsNorm)?;
        let ld = d + r;
        let a = km::HcPreNormArgs { x: xs, x_ld: HC_DIM as i64, pre: rec_last, pre_stride: 24, w: nw.ptr.0, y_norm: self.xn.ptr(), yn_ld: d as i64,
                                    y_raw: self.cin.ptr(), yr_ld: ld as i64, w_fmt: fmt_of(nw)?, dim: d as i32, rows: rows as i32, eps: cfg.rms_eps as f32, inv_dim: 1.0 / d as f32 };
        g.launch1("d41_hc_pre_norm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("ds.head_pre_norm", "d41_hc_pre_norm", rows);
        self.head_and_chain(f, rd, x, nlanes, k, forced)?;
        Ok(outs)
    }

    /// The tied head on the block rows (input: `xn`, filled by `block` or by the per-op check), the Markov chain
    /// and the confidence head. The first `dim` columns of `cin` must hold the un-normed hc_pre output.
    pub fn head_and_chain(&self, f: &mut Fwd, rd: &Round, x: &dyn VocabExchange, nlanes: usize, k: usize, forced: bool) -> Result<()> {
        let g = f.g;
        let (d, r, v) = (self.dim, self.rank, self.shard.vlocal);
        let rows = nlanes * BLOCK;
        let ld = d + r;
        let hb = crate::dsv41::fwd::glue::backend(self.head.w.1);
        let need = hb.scratch_bytes(&self.head.w.0, rows);
        ensure!(need <= f.scr.bytes, "head scratch {need} > {}", f.scr.bytes);
        // a plain (BF16) head over > 8 rows: one pass (leg PK5's d41_ds_plain_r*), the same bits as the backend's GEMV
        if super::chain::plain_rows_gemv(g, &self.head.w, &bf16(self.xn.ptr(), rows, d, d), &f32a(self.logits.ptr(), rows, v, v), Accum::Overwrite)? {
            receipts::kernel("ds.head (plain, one pass)", "d41_ds_plain_r*", rows);
        } else {
            let cls = hb.logits(&g.l(), &self.head.w.0, &bf16(self.xn.ptr(), rows, d, d), &f32a(self.logits.ptr(), rows, v, v), Scratch { ptr: DevPtr(f.scr.ptr), bytes: f.scr.bytes })?;
            receipts::count("ds.head", self.head.w.1, if self.head.w.1 == BackendKind::Plain { "d41x_plain_gemv" } else { "d41x_prep+d41x_gemm+d41x_post (f32)" }, rows, cls);
        }
        let w2 = &self.stages[2];
        let me = w2.lin(C::DsMarkovEmbed)?;
        let mh = self_shard(w2.lin(C::DsMarkovHead)?, self.shard);
        let table = if forced { rd.fids.ptr() } else { rd.ids.ptr() };
        // markov_embed of the chain's input i -> cin[l][i][d .. d + r]; logits[l][i] += markov_head . embed (f32
        // accumulation, then one f32 add: the reference's logits.add_); the draw of row i -> ids[i + 1]
        let ch = super::chain::Chain { me, mh: &mh, cin: self.cin.ptr(), ld, d, r, logits: self.logits.ptr(), vl: v, sh: self.shard, table };
        super::chain::run(g, &f.scr, &ch, rd, x, nlanes, k, &mut |_| Ok(()))?;
        receipts::kernel("ds.markov_chain", super::chain::ENTRY, nlanes);
        // confidence = proj . [hc_pre out ; markov embed] (f32)
        let cw = w2.lin(C::DsConfidence)?;
        let entry = super::chain::confidence(g, &f.scr, cw, self.cin.ptr(), rows, ld, self.conf.ptr())?;
        receipts::kernel("ds.confidence", entry, rows);
        Ok(())
    }

    /// Set the head input from host rows (the per-op check feeds the trace's `Lhead.x`): `xh` bf16 values
    /// [rows][dim] un-normed; fills cin's first dim columns and xn = norm(xh).
    pub fn set_head_input(&self, f: &mut Fwd, xh_bits: &[u16], rows: usize) -> Result<()> {
        let (d, r) = (self.dim, self.rank);
        ensure!(xh_bits.len() == rows * d && rows <= self.cap * BLOCK, "head input {} values for {rows} rows", xh_bits.len());
        let ld = d + r;
        let mut cin = vec![0u16; rows * ld];
        for i in 0..rows { cin[i * ld..i * ld + d].copy_from_slice(&xh_bits[i * d..(i + 1) * d]); }
        f.g.h2d(self.cin.ptr(), super::vocab::as_bytes(&cin))?;
        let nw = self.stages[2].param(C::DsNorm)?;
        let a = km::RmsNormArgs { x: self.cin.ptr(), x_ld: ld as i64, w: nw.ptr.0, y: self.xn.ptr(), y_ld: d as i64, w_fmt: fmt_of(nw)?, dim: d as i32, rows: rows as i32,
                                  eps: self.cfg.rms_eps as f32, inv_dim: 1.0 / d as f32 };
        f.g.launch1("d41_rmsnorm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)
    }
}

/// The Markov head with this rank's vocab rows (it is split like the head: `Split::Out`).
fn self_shard(w: &(DenseWeight, BackendKind), sh: Shard) -> (DenseWeight, BackendKind) {
    let mut d = w.0.clone();
    if !(sh.v0 == 0 && sh.vlocal as u64 == d.out) { d.shard = WShard::Out(sh.v0 as u64..(sh.v0 + sh.vlocal) as u64); }
    (d, w.1)
}
