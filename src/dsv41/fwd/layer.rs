//! SL: the per-layer forward on one GPU, assembled from the merged kernel bands, every piece on ONE blocking
//! stream. What one trunk layer runs (inference/model.py Block.forward 970-996, REF_MAP §3-4):
//!
//! | piece | kernels (band) | here |
//! |---|---|---|
//! | Engram (layers 1, 14) | EIO rows (80) -> `engram.wkv` (backend) -> `d41_engram_gate` (70) | [`Fwd::engram`] |
//! | mHC mixes, hc_pre + norm, hc_post | `d41_mix_partial` / `d41_mix_finish` / `d41_hc_pre_norm` / `d41_hc_post` (70) | [`Fwd::mix`], [`Fwd::hc_pre_norm`], [`Fwd::hc_post`] |
//! | q: wq_a, q_norm, wq_b, RoPE | backend, `d41_rmsnorm` (70), `d41f_rope_rows` (05) | [`Fwd::q_path`] |
//! | window K/V: wkv, kv_norm, RoPE, FP8 | backend, `d41_kc_swa_write` (40) | [`Fwd::window`] |
//! | compressor + index K (KV sources) | backend (wkv / wgate / wk), `d41_kc_compress` / `d41_kc_stash` / `d41_kc_index_k` (40) | [`Fwd::compressor`] |
//! | indexer top-512 (+ layer-20 candidates) | backend (wq_b, weights_proj), `d41_idx_*` (50) | [`Fwd::indexer`] |
//! | sparse attention + inverse RoPE + act_quant | `d41_attn_prefill` / `d41_attn_decode` (60) | [`Fwd::attention`] |
//! | wo_a per o_group, wo_b | backend | [`Fwd::wo`] |
//! | router, routed experts, shared expert | `d41_router_*` (70), backend `moe`, backend dense + `d41f_swiglu` / `d41f_add_round` (05) | [`Fwd::router`], [`Fwd::ffn`] |
//!
//! Rows are one lane's rows (any positions: a prefill chunk, a verify block, the gate's dumped row subset).
//! Buffers come from a bump arena reset by the caller; caches that outlive a call (`SrcState`, `Window`) are
//! owned buffers.

use super::dev::{bf16, f32a, Buf, Gpu};
use super::glue::{self, Scr};
use super::receipts;
use super::weights::LayerW;
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::gpu::{attn, indexer as ki, kv, mixers as km};
use crate::dsv41::weights::{Accum, DevPtr, DevTensor, Route};
use anyhow::{bail, ensure, Context, Result};

/// The four RoPE tables a layer uses (per consumer, so the legacy builders can be compared with the unified one).
#[derive(Clone, Copy, Debug)]
pub struct RopeSet { pub q: Tabs, pub kc: Tabs, pub ka: Tabs, pub ki: [f32; 32] }
/// (compressed class: compress_rope_theta + YaRN, plain class: rope_theta)
#[derive(Clone, Copy, Debug)]
pub struct Tabs { pub compressed: [f32; 32], pub plain: [f32; 32] }
impl Tabs { pub fn of(&self, compressed: bool) -> [f32; 32] { if compressed { self.compressed } else { self.plain } } }

/// A bump arena over one device buffer (`peak`: the largest use seen, leg MEM1's ledger).
pub struct Bump { pub base: Buf, pub off: usize, pub peak: usize }
impl Bump {
    pub fn new(g: &Gpu, bytes: usize) -> Result<Bump> { Ok(Bump { base: g.buf(bytes)?, off: 0, peak: 0 }) }
    pub fn take(&mut self, bytes: usize) -> Result<u64> {
        let o = (self.off + 255) / 256 * 256;
        ensure!(o + bytes <= self.base.bytes, "bump arena exhausted ({} + {bytes} > {})", o, self.base.bytes);
        self.off = o + bytes;
        self.peak = self.peak.max(self.off);
        Ok(self.base.ptr() + o as u64)
    }
    pub fn reset(&mut self) { self.off = 0; }
}

pub struct Fwd<'a> {
    pub g: &'a Gpu, pub cfg: &'a Dsv41Config, pub scr: Scr, pub b: Bump, pub rope: RopeSet, /** read the kernels' error words (a host sync) */ pub checks: bool,
    /// Prefill calls (leg CE): every dense linear piece runs at least this many rows (a shorter piece is padded by
    /// repeating its last row), so a prompt row's bits never depend on how many rows its call had: the backends'
    /// row families differ (<= 32 rows vs the wide tensor-core family), and CED (a 1-row layer 39) and chunked
    /// prefill (any tail) must equal the single-shot exact prefill bit for bit. 0 = off (decode / verify, the gate).
    pub wide_min: usize,
}

/// The SWA window of one lane for one layer: the ring (134 slots) and the call's linear rows (chunk planes).
pub struct Window { pub ring_q: Buf, pub ring_s: Buf, pub lin_q: Buf, pub lin_s: Buf, pub lin_base: i64, pub rows: usize }
/// A KV source's caches: main KV (288-B records), index-K planes, the frontier ring (ratio >= 2).
pub struct SrcState { pub layer: usize, pub ratio: u32, pub n_groups: usize, pub kv_main: Buf, pub iq: Buf, pub is: Buf, pub front: Option<(Buf, Buf)>, pub latent: Buf }
/// Layer 20's candidate pool for a row-block layout (mode 1 of layers 24-36 reads it).
pub struct Union { pub uni: Buf, pub ucnt: Buf, pub rbs: Vec<(i32, i32)> }

pub fn fmt_of(t: &DevTensor) -> Result<u32> {
    Ok(match t.dtype.as_str() { "BF16" => 1, "F16" => 2, "F32" => 3, d => bail!("parameter dtype {d}: BF16 / F16 / F32 expected") })
}
pub const HC_DIM: usize = 4 * 5120;

impl<'a> Fwd<'a> {
    fn dim(&self) -> usize { self.cfg.hidden }
    fn eps(&self) -> f32 { self.cfg.rms_eps as f32 }

    // ------------------------------------------------------------------ mHC
    /// The sublayer's mix record [rows][24] f32 (pre 0..3, post 4..7, comb 8..23) of stream x [rows][4 * dim].
    pub fn mix(&mut self, w: &LayerW, attn_side: bool, x: u64, rows: usize) -> Result<u64> {
        let (fc, sc, bc) = if attn_side { (C::HcAttnFn, C::HcAttnScale, C::HcAttnBase) } else { (C::HcFfnFn, C::HcFfnScale, C::HcFfnBase) };
        let (f, s, b) = (w.param(fc)?, w.param(sc)?, w.param(bc)?);
        let k = 4 * self.dim();
        let chunks = km::partial_chunks(k);
        let rg = km::pick_rg(rows);
        let ngroups = rows.div_ceil(rg);
        let part = self.b.take(chunks * rows * 32 * 4)?;
        let rpart = self.b.take(rows * 32 * 4)?;
        let cnt = self.b.take(ngroups * 4)?;
        self.g.memset(cnt, ngroups * 4, 0)?;
        let out = self.b.take(rows * 24 * 4)?;
        let a = km::MixPartialArgs { x, x_ld: k as i64, fn_: f.ptr.0, fn_ld: k as i64, part, chunk_stride: (rows * 32) as i64, row_stride: 32,
                                     rank_part: rpart, rp_row_stride: 32, counters: cnt, k0: 0, klen: k as i32, rows: rows as i32, rg: rg as i32 };
        self.g.launch1("d41_mix_partial", (chunks as u32, ngroups as u32, 1), (128, 1, 1), 0, &a)?;
        let c = km::MixConsts::v41();
        let fa = km::MixFinishArgs { part: rpart, slot_stride: (rows * 32) as i64, row_stride: 32, scale: s.ptr.0, base: b.ptr.0, out, out_ld: 24, nslots: 1,
                                     rows: rows as i32, inv_k: 1.0 / k as f32, eps_norm: self.eps(), eps_hc: self.cfg.hc_eps as f32, sinkhorn_iters: self.cfg.sinkhorn_iters as i32 };
        let _ = c;
        self.g.launch1("d41_mix_finish", ((rows as u32).div_ceil(8), 1, 1), (128, 1, 1), 0, &fa)?;
        receipts::kernel(if attn_side { "mhc.attn_mix" } else { "mhc.ffn_mix" }, "d41_mix_partial+d41_mix_finish", rows);
        Ok(out)
    }

    /// RMSNorm(hc_pre(x, pre)) -> bf16 [rows][dim]; `pre` = a record (stride 24) or the dumped pre-mix (stride 4).
    pub fn hc_pre_norm(&mut self, x: u64, pre: u64, pre_stride: usize, norm: &DevTensor, rows: usize) -> Result<u64> {
        let d = self.dim();
        let y = self.b.take(rows * d * 2)?;
        let a = km::HcPreNormArgs { x, x_ld: (4 * d) as i64, pre, pre_stride: pre_stride as i64, w: norm.ptr.0, y_norm: y, yn_ld: d as i64, y_raw: 0, yr_ld: 0,
                                    w_fmt: fmt_of(norm)?, dim: d as i32, rows: rows as i32, eps: self.eps(), inv_dim: 1.0 / d as f32 };
        self.g.launch1("d41_hc_pre_norm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("mhc.pre_norm", "d41_hc_pre_norm", rows);
        Ok(y)
    }

    /// hc_post(xs, resid, post / comb of `rec`) -> stream [rows][4 * dim] bf16.
    pub fn hc_post(&mut self, xs: u64, resid: u64, rec: u64, rows: usize) -> Result<u64> {
        let d = self.dim();
        let out = self.b.take(rows * 4 * d * 2)?;
        let a = km::HcPostArgs { xs, xs_ld: d as i64, resid, r_ld: (4 * d) as i64, post: rec + 16, post_stride: 24, comb: rec + 32, comb_stride: 24,
                                 out, o_ld: (4 * d) as i64, dim: d as i32, rows: rows as i32 };
        self.g.launch1("d41_hc_post", ((d as u32).div_ceil(1024), rows as u32, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("mhc.post", "d41_hc_post", rows);
        Ok(out)
    }

    fn rmsnorm(&mut self, x: u64, w: &DevTensor, dim: usize, rows: usize) -> Result<u64> {
        let y = self.b.take(rows * dim * 2)?;
        let a = km::RmsNormArgs { x, x_ld: dim as i64, w: w.ptr.0, y, y_ld: dim as i64, w_fmt: fmt_of(w)?, dim: dim as i32, rows: rows as i32, eps: self.eps(), inv_dim: 1.0 / dim as f32 };
        self.g.launch1("d41_rmsnorm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        Ok(y)
    }

    /// A linear over many rows in row chunks (bounded scratch); y gets `y_dtype`.
    pub fn lin_rows(&mut self, what: &'static str, w: &LayerW, c: C, sub: Option<u16>, x: u64, x_ld: usize, x_cols: usize, rows: usize, y: u64, y_ld: usize, y_f32: bool) -> Result<()> {
        let lw = match sub { Some(s) => w.lin_sub(c, s).with_context(|| format!("layer {}: no {} slice {s}", w.layer, c.name()))?, None => w.lin(c)? };
        let out = lw.0.out as usize;
        let ycols = out.min(y_ld);
        const CH: usize = 4096;
        let es = if y_f32 { 4 } else { 2 };
        let mut r0 = 0;
        while r0 < rows {
            let n = (rows - r0).min(CH);
            if n < self.wide_min {
                self.lin_padded(what, lw, x + (r0 * x_ld * 2) as u64, x_ld, x_cols, n, y + (r0 * y_ld * es) as u64, y_ld, es, out, y_f32)?;
                r0 += n;
                continue;
            }
            let xa = bf16(x + (r0 * x_ld * 2) as u64, n, x_cols, x_ld);
            let mut ya = if y_f32 { f32a(y + (r0 * y_ld * es) as u64, n, ycols, y_ld) } else { bf16(y + (r0 * y_ld * es) as u64, n, ycols, y_ld) };
            ya.cols = out;
            glue::linear(self.g, what, lw, &xa, &ya, Accum::Overwrite, &self.scr)?;
            r0 += n;
        }
        Ok(())
    }

    /// One piece of `n < wide_min` rows run as `wide_min` rows: x rows copied to a scratch (the tail repeats the
    /// last row), the linear on the padded block, the first `n` output rows copied back.
    #[allow(clippy::too_many_arguments)]
    fn lin_padded(&mut self, what: &'static str, lw: &(crate::dsv41::weights::DenseWeight, crate::dsv41::weights::BackendKind), x: u64, x_ld: usize, x_cols: usize, n: usize,
                  y: u64, y_ld: usize, es: usize, out: usize, y_f32: bool) -> Result<()> {
        let m = self.wide_min;
        let (xb, yb) = (self.b.take(m * x_cols * 2)?, self.b.take(m * out * es)?);
        for r in 0..m {
            let src = x + (r.min(n - 1) * x_ld * 2) as u64;
            self.g.d2d(xb + (r * x_cols * 2) as u64, src, x_cols * 2)?;
        }
        let xa = bf16(xb, m, x_cols, x_cols);
        let ya = if y_f32 { f32a(yb, m, out, out) } else { bf16(yb, m, out, out) };
        glue::linear(self.g, what, lw, &xa, &ya, Accum::Overwrite, &self.scr)?;
        for r in 0..n { self.g.d2d(y + (r * y_ld * es) as u64, yb + (r * out * es) as u64, out * es)?; }
        Ok(())
    }

    // ------------------------------------------------------------------ attention
    /// qr = q_norm(wq_a(x)) [rows][q_lora] and q = rope(wq_b(qr)) [rows][heads * head_dim] bf16.
    pub fn q_path(&mut self, w: &LayerW, x: u64, pos: u64, rows: usize, compressed: bool) -> Result<(u64, u64)> {
        let (d, ql, h, hd) = (self.dim(), self.cfg.q_lora_rank, self.cfg.heads, self.cfg.head_dim);
        let qa = self.b.take(rows * ql * 2)?;
        self.lin_rows("attn.wq_a", w, C::AttnWqA, None, x, d, d, rows, qa, ql, false)?;
        let qr = self.rmsnorm(qa, w.param(C::AttnQNorm)?, ql, rows)?;
        let q = self.b.take(rows * h * hd * 2)?;
        self.lin_rows("attn.wq_b", w, C::AttnWqB, None, qr, ql, ql, rows, q, h * hd, false)?;
        glue::rope_rows(self.g, q, h * hd, pos, rows, h, hd, false, &self.rope.q.of(compressed))?;
        Ok((qr, q))
    }

    /// A new lane window sized for `rows` linear rows (ring of 134 slots).
    pub fn new_window(&self, rows: usize) -> Result<Window> {
        let (hd, slots) = (self.cfg.head_dim, self.cfg.swa_slots());
        Ok(Window { ring_q: self.g.buf(slots * hd)?, ring_s: self.g.buf(slots * hd / 32)?, lin_q: self.g.buf(rows * hd)?, lin_s: self.g.buf(rows * hd / 32)?, lin_base: 0, rows })
    }

    /// Window K/V of positions [p0, p0 + rows) from x [rows][dim]: ring (last 134) + linear rows.
    pub fn window(&mut self, w: &LayerW, x: u64, p0: i64, rows: usize, compressed: bool) -> Result<Window> {
        let mut win = self.new_window(rows)?;
        self.window_into(w, x, p0, rows, compressed, &mut win)?;
        Ok(win)
    }

    /// The same into an existing window (its ring keeps the older positions; the linear rows become this call's).
    pub fn window_into(&mut self, w: &LayerW, x: u64, p0: i64, rows: usize, compressed: bool, win: &mut Window) -> Result<()> {
        let d = self.dim();
        let hd = self.cfg.head_dim;
        ensure!(rows <= win.rows, "window: {rows} rows > its {} linear rows", win.rows);
        let raw = self.b.take(rows * hd * 2)?;
        self.lin_rows("attn.wkv", w, C::AttnWkv, None, x, d, d, rows, raw, hd, false)?;
        let norm = w.param(C::AttnKvNorm)?;
        let freq = self.b.take(128)?;
        self.g.h2d_async(freq, &self.rope.kc.of(compressed))?;
        let a = kv::SwaArgs { kv_raw: raw, norm_w: norm.ptr.0, inv_freq: freq, ring_q: win.ring_q.ptr(), ring_s: win.ring_s.ptr(), chunk_q: win.lin_q.ptr(),
                              chunk_s: win.lin_s.ptr(), p0, rows: rows as i32, norm_fmt: fmt_of(norm)?, eps: self.eps() };
        self.g.launch1("d41_kc_swa_write", (rows as u32, 1, 1), (128, 1, 1), 0, &a)?;
        receipts::kernel("kv.swa_write", "d41_kc_swa_write", rows);
        win.lin_base = p0;
        Ok(())
    }

    /// A KV source's compressor over positions [0, rows) of x (one prefill call), plus the index-K planes.
    pub fn compressor(&mut self, w: &LayerW, x: u64, rows: usize, cap: usize) -> Result<SrcState> {
        let (d, hd, l) = (self.dim(), self.cfg.head_dim, w.layer);
        let ratio = self.cfg.ratio(l);
        ensure!(ratio >= 1, "layer {l} has no compressor");
        let ng = rows / ratio as usize;
        let rr = self.cfg.frontier_rows(ratio);
        let st = SrcState { layer: l, ratio, n_groups: ng, kv_main: self.g.buf(cap * 288)?, iq: self.g.buf(cap * 64)?, is: self.g.buf(cap * 4)?,
                            front: if ratio >= 2 { Some((self.g.buf(rr * hd * 4)?, self.g.buf(rr * hd * 4)?)) } else { None }, latent: self.g.buf(ng.max(1) * hd * 2)? };
        let norm = w.param(C::CompNorm)?;
        let freq = self.b.take(128)?;
        self.g.h2d_async(freq, &self.rope.kc.compressed)?;
        let mut a = kv::CompressArgs { norm_w: norm.ptr.0, inv_freq: freq, latent_out: st.latent.ptr(), kv_main: st.kv_main.ptr(), p0: 0, rows: rows as i32,
                                       ratio: ratio as i32, ring_rows: rr as i32, g_first: 0, n_groups: ng as i32, norm_fmt: fmt_of(norm)?, eps: self.eps(), ..Default::default() };
        let (kvp, scp);
        if ratio >= 2 {
            kvp = self.b.take(rows * hd * 4)?;
            scp = self.b.take(rows * hd * 4)?;
            self.lin_rows("comp.wkv", w, C::CompWkv, None, x, d, d, rows, kvp, hd, true)?;
            self.lin_rows("comp.wgate", w, C::CompWgate, None, x, d, d, rows, scp, hd, true)?;
            let (fk, fs) = st.front.as_ref().map(|(a, b)| (a.ptr(), b.ptr())).unwrap_or((0, 0));
            a.kv = kvp; a.score = scp; a.ring_kv = fk; a.ring_score = fs;
        } else {
            kvp = self.b.take(rows * hd * 2)?;
            scp = 0;
            self.lin_rows("comp.wkv", w, C::CompWkv, None, x, d, d, rows, kvp, hd, false)?;
            a.kv16 = kvp;
        }
        if ng > 0 { self.g.launch1("d41_kc_compress", (ng as u32, 1, 1), (128, 1, 1), 0, &a)?; }
        if let Some((fk, fs)) = &st.front {
            let s = kv::StashArgs { kv: kvp, score: scp, ring_kv: fk.ptr(), ring_score: fs.ptr(), p0: 0, rows: rows as i32, ring_rows: rr as i32 };
            self.g.launch1("d41_kc_stash", (rows.min(rr) as u32, 1, 1), (128, 1, 1), 0, &s)?;
        }
        receipts::kernel("kv.compress", "d41_kc_compress+d41_kc_stash", rows);
        // index keys: k = k_norm(wk(latent)) -> RoPE -> FP4
        if ng > 0 {
            let id = self.cfg.index_head_dim;
            let kraw = self.b.take(ng * id * 2)?;
            self.lin_rows("idx.wk", w, C::IdxWk, None, st.latent.ptr(), hd, hd, ng, kraw, id, false)?;
            let kn = w.param(C::IdxKNorm)?;
            let ia = kv::IndexArgs { k_raw: kraw, norm_w: kn.ptr.0, inv_freq: freq, q_plane: st.iq.ptr(), s_plane: st.is.ptr(), ratio: ratio as i32, g_first: 0,
                                     n_groups: ng as i32, norm_fmt: fmt_of(kn)?, eps: self.eps() };
            self.g.launch1("d41_kc_index_k", ((ng as u32).div_ceil(4), 1, 1), (32, 4, 1), 0, &ia)?;
            receipts::kernel("kv.index_k", "d41_kc_index_k", ng);
        }
        Ok(st)
    }

    /// The indexer's selection for `rows` rows of one lane: sel [rows][512] (ascending, -1 padded) + counts.
    /// `cand_out` = layer 20 (writes the union); `cand_in` = layers 24-36 (mode 1 over layer 20's union).
    pub fn indexer(&mut self, w: &LayerW, qr: u64, x: u64, pos: u64, rows: usize, src: &SrcState, cand_out: bool, cand_in: Option<&Union>) -> Result<(u64, u64, Option<Union>)> {
        let (d, ql) = (self.dim(), self.cfg.q_lora_rank);
        let iq = self.b.take(rows * ki::HEADS * ki::HEAD_DIM * 2)?;
        self.lin_rows("idx.wq_b", w, C::IdxWqB, None, qr, ql, ql, rows, iq, ki::HEADS * ki::HEAD_DIM, false)?;
        let wp = self.b.take(rows * ki::HEADS * 2)?;
        self.lin_rows("idx.weights_proj", w, C::IdxWeightsProj, None, x, d, d, rows, wp, ki::HEADS, false)?;
        let r = 8usize;
        let rbs: Vec<(i32, i32)> = (0..rows).step_by(r).map(|i| (i as i32, (rows - i).min(r) as i32)).collect();
        if let Some(u) = cand_in { ensure!(u.rbs == rbs, "mode 1 needs layer 20's row blocks"); }
        let n_rb = rbs.len();
        let cap = src.n_groups;
        let ucap = r * ki::CAND_BLOCKS;
        let sz = ki::sizes(n_rb, r, 1, rows, cap.max(1), ucap);
        let rbv: Vec<u64> = rbs.iter().flat_map(|&(r0, n)| [src.iq.ptr(), src.is.ptr(), (r0 as u32 as u64) | ((n as u32 as u64) << 32), 0]).collect();
        let rbd = self.b.take(rbv.len() * 8)?;
        self.g.h2d_async(rbd, &rbv)?;
        let sel = self.b.take(sz.sel_i32 * 4)?;
        let cnt = self.b.take(rows * 4)?;
        let part = self.b.take(sz.part_u64.max(1) * 8)?;
        let pcnt = self.b.take(sz.pcnt_i32.max(1) * 4)?;
        let bmax = if cand_out { self.b.take(sz.bmax_u16.max(1) * 2)? } else { 0 };
        let cand = self.b.take(sz.cand_i32 * 4)?;
        let ccnt = self.b.take(rows * 4)?;
        let bm = self.b.take(sz.bm_u32.max(1) * 4)?;
        let err = self.b.take(4)?;
        self.g.memset(err, 4, 0)?;
        let un = if cand_out { Some(Union { uni: self.g.buf(sz.uni_u32.max(1) * 4)?, ucnt: self.g.buf(n_rb * 4)?, rbs: rbs.clone() }) } else { None };
        let (uni, ucnt) = match (&un, cand_in) { (Some(u), _) | (None, Some(u)) => (u.uni.ptr(), u.ucnt.ptr()), _ => (self.b.take(4)?, self.b.take(n_rb.max(1) * 4)?) };
        let mut a = ki::IdxArgs::default();
        a.iq = iq; a.w = wp; a.pos = pos; a.rbs = rbd; a.sel = sel; a.sel_cnt = cnt; a.part = part; a.pcnt = pcnt; a.bmax = bmax; a.cand = cand; a.cand_cnt = ccnt; a.bm = bm;
        a.uni = uni; a.ucnt = ucnt; a.err = err; a.tick = 0;
        a.n_rb = n_rb as i32; a.rows_total = rows as i32; a.iq_ld = (ki::HEADS * ki::HEAD_DIM) as i32; a.w_ld = ki::HEADS as i32; a.in_f32 = 0;
        a.ratio = src.ratio as i32; a.mode = cand_in.is_some() as i32; a.write_bmax = cand_out as i32;
        a.n_keys_cap = cap.max(1) as i32; a.nb_cap = cap.max(1).div_ceil(ki::CAND_BLOCK) as i32; a.nbw_cap = (a.nb_cap as usize).div_ceil(32) as i32;
        a.g = 1; a.min_tiles = 8; a.dbg_cap = 0; a.ucap = ucap as i32; a.r = r as i32; a.inv_freq = self.rope.ki;
        ki::select(&self.g.l(), &a, cand_out)?;
        let (entry, _) = ki::score_entry(r);
        receipts::kernel(if cand_out { "idx.select+cand" } else if cand_in.is_some() { "idx.select(mode1)" } else { "idx.select" }, entry, rows);
        if self.checks {
            let e = self.g.down::<i32>(err, 1)?[0];
            ensure!(e == 0, "indexer layer {}: kernel error word {e}", w.layer);
        }
        Ok((sel, cnt, un))
    }

    /// Sparse attention of `rows` query rows (q after RoPE) -> o [rows][heads * 512] (inverse RoPE + act_quant).
    /// `sel` = (ptr, stride, cap) of the compressed selections (None for SWA-only layers).
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&mut self, w: &LayerW, q: u64, pos: u64, rows: usize, win: &Window, src: Option<&SrcState>, sel: Option<(u64, usize, usize)>, decode: Option<&attn::DecodeScratch>) -> Result<u64> {
        let (h, hd) = (self.cfg.heads, self.cfg.head_dim);
        let compressed = self.cfg.ratio(w.layer) > 0;
        let out = self.b.take(rows * h * hd * 2)?;
        let err = self.b.take(4)?;
        self.g.memset(err, 4, 0)?;
        let lane = attn::LaneSrc { swa_q: win.ring_q.p(), swa_s: win.ring_s.p(), slots: self.cfg.swa_slots(),
                                   lin: Some((win.lin_q.p(), win.lin_s.p(), win.lin_base)), kv: src.map(|s| (s.kv_main.p(), s.n_groups)), ext: None };
        let (sp, sstride, scap) = sel.unwrap_or((0, 0, 0));
        let call = attn::AttnCall { rows, heads: h, q: DevPtr(q), q_stride: h * hd, sink: w.param(C::AttnSink)?.ptr, pos: DevPtr(pos), win_end: DevPtr(0), lane: DevPtr(0),
                                    sel: DevPtr(sp), sel_stride: sstride, sel_cap: scap, n_sel: DevPtr(0), lanes: vec![lane], out: DevPtr(out), out_stride: h * hd,
                                    out_q: None, out_s: None, epi: attn::EPI_ROPE | attn::EPI_AQ, rope_freq: self.rope.ka.of(compressed), err: DevPtr(err) };
        match decode {
            Some(s) => { attn::launch_decode(self.g.l(), &call, s)?; receipts::kernel("attn.sparse", "d41_attn_decode", rows); }
            None => { attn::launch_prefill(self.g.l(), &call)?; receipts::kernel("attn.sparse", "d41_attn_prefill", rows); }
        }
        if self.checks {
            let e = self.g.down::<i32>(err, 1)?[0];
            ensure!(e == 0, "attention layer {}: kernel error word {e}", w.layer);
        }
        Ok(out)
    }

    /// wo_a per o_group (block-diagonal) then wo_b -> attn_out [rows][dim] bf16.
    pub fn wo(&mut self, w: &LayerW, o: u64, rows: usize) -> Result<u64> {
        let (h, hd, gq, olr, d) = (self.cfg.heads, self.cfg.head_dim, self.cfg.o_groups, self.cfg.o_lora_rank, self.dim());
        let gin = h * hd / gq;
        let mid = self.b.take(rows * gq * olr * 2)?;
        for gi in 0..gq {
            self.lin_rows("attn.wo_a", w, C::AttnWoA, Some(gi as u16), o + (gi * gin * 2) as u64, h * hd, gin, rows, mid + (gi * olr * 2) as u64, gq * olr, false)?;
        }
        let out = self.b.take(rows * d * 2)?;
        self.lin_rows("attn.wo_b", w, C::AttnWoB, None, mid, gq * olr, gq * olr, rows, out, d, false)?;
        Ok(out)
    }

    // ------------------------------------------------------------------ FFN
    /// Router: ids i32 [rows][top_k], weights f32 [rows][top_k] from ffn_in x [rows][dim].
    pub fn router(&mut self, w: &LayerW, x: u64, rows: usize) -> Result<(u64, u64)> {
        self.router_vl(w, x, rows, 0)
    }

    /// Router with optional `image_mask` (`u8 [rows]`, `1` for image-span tokens -> selects `GateBiasVl`, `0` -> `GateBias`).
    /// When `image_mask == 0`, executes the exact text-only path.
    pub fn router_vl(&mut self, w: &LayerW, x: u64, rows: usize, image_mask: u64) -> Result<(u64, u64)> {
        let (e, k) = self.cfg.moe_of(w.layer);
        let d = self.dim();
        let gw = w.weight_tensor(C::GateWeight)?;
        let bias = w.param(C::GateBias)?;
        let bias_fmt = fmt_of(bias)?;
        let bias_vl = if image_mask != 0 {
            let bvl = w.param(C::GateBiasVl)?;
            ensure!(fmt_of(bvl)? == bias_fmt, "layer {}: GateBiasVl dtype {} != GateBias dtype {}", w.layer, bvl.dtype, bias.dtype);
            bvl.ptr.0
        } else {
            0
        };
        let sc = self.b.take(rows * e * 4)?;
        let mut r0 = 0usize;
        // leg PP1: a prefill call (> 32 rows) on the fp32 router: ONE launch of the wide entry (same score bits as the 32-row launches)
        if rows > km::ROUTER_ROWS_MAX && !km::router_tf32_on() && crate::dsv41::gpu::pf_on("dsv41-pf-router-wide") {
            let a = km::RouterScoresArgs { x, x_ld: d as i64, w: gw.ptr.0, w_ld: d as i64, scores: sc, s_ld: e as i64, w_fmt: fmt_of(gw)?, dim: d as i32, e: e as i32, rows: rows as i32 };
            let (name, (gx, gy), block) = km::router_wide_launch(&a, rows, km::WIDE);
            self.g.launch1(name, (gx, gy, 1), (block, 1, 1), 0, &a)?;
            receipts::kernel("ffn.router_scores (PP1 wide)", name, rows);
            crate::dsv41::gpu::PF_ROUTER_WIDE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            r0 = rows;
        }
        while r0 < rows {
            let n = (rows - r0).min(km::ROUTER_ROWS_MAX);
            let a = km::RouterScoresArgs { x: x + (r0 * d * 2) as u64, x_ld: d as i64, w: gw.ptr.0, w_ld: d as i64, scores: sc + (r0 * e * 4) as u64, s_ld: e as i64,
                                           w_fmt: fmt_of(gw)?, dim: d as i32, e: e as i32, rows: n as i32 };
            // fp32 r* entries by default; `--dsv41-router-tf32 on` takes the tf32 tensor-core entries (km::router_launch, leg PK1)
            let (name, grid, block) = km::router_launch(&a, n);
            self.g.launch1(name, (grid, 1, 1), (block, 1, 1), 0, &a)?;
            receipts::kernel("ffn.router_scores", name, n);
            r0 += n;
        }
        let (ow, oi) = (self.b.take(rows * k * 4)?, self.b.take(rows * k * 4)?);
        let a = km::RouterSelectArgs { scores: sc, s_ld: e as i64, bias: bias.ptr.0, bias_vl, image_mask, out_w: ow, out_i: oi, o_ld: k as i64, bias_fmt,
                                       e: e as i32, topk: k as i32, rows: rows as i32, route_scale: self.cfg.routed_scale as f32, temp: 1.0, norm_topk: self.cfg.norm_topk_prob as i32 };
        self.g.launch1("d41_router_select", (rows as u32, 1, 1), (128, 1, 1), 0, &a)?;
        Ok((oi, ow))
    }

    /// The MoE FFN of ffn_in x [rows][dim] with routing (ids, weights) -> ffn_out [rows][dim] bf16:
    /// routed experts (f32 sum) + the shared expert (bf16), one bf16 rounding (model.py MoE.forward 891-906).
    pub fn ffn(&mut self, w: &LayerW, x: u64, rows: usize, ids: u64, wts: u64) -> Result<u64> {
        let (d, inter) = (self.dim(), self.cfg.moe_inter);
        let mw = w.moe.as_ref().context("layer has no routed experts")?;
        let y = self.b.take(rows * d * 4)?;
        self.g.memset(y, rows * d * 4, 0)?;
        let route = Route { ids: DevPtr(ids), weights: DevPtr(wts), top_k: mw.0.top_k };
        glue::moe(self.g, mw, &bf16(x, rows, d, d), &route, &f32a(y, rows, d, d), &self.scr)?;
        // shared expert: w1 / w3 (bf16), SwiGLU, w2 (bf16)
        let (g1, g3, act, s) = (self.b.take(rows * inter * 2)?, self.b.take(rows * inter * 2)?, self.b.take(rows * inter * 2)?, self.b.take(rows * d * 2)?);
        self.lin_rows("ffn.shared_w1", w, C::SharedW1, None, x, d, d, rows, g1, inter, false)?;
        self.lin_rows("ffn.shared_w3", w, C::SharedW3, None, x, d, d, rows, g3, inter, false)?;
        glue::swiglu(self.g, g1, g3, act, inter, rows, inter, self.cfg.swiglu_limit as f32)?;
        self.lin_rows("ffn.shared_w2", w, C::SharedW2, None, act, inter, inter, rows, s, d, false)?;
        let out = self.b.take(rows * d * 2)?;
        glue::add_round(self.g, y, s, out, rows, d)?;
        Ok(out)
    }

    // ------------------------------------------------------------------ Engram
    /// The Engram module of layer `w.layer` on stream h [rows][4 * dim] given the rows' decoded n-gram
    /// embeddings e [rows][cols * 256] bf16 -> new stream [rows][4 * dim].
    pub fn engram(&mut self, w: &LayerW, h: u64, e: u64, rows: usize) -> Result<u64> {
        let d = self.dim();
        let kvw = w.lin(C::EngramWkv)?;
        let k_in = kvw.0.inn as usize;
        let kv_out = (self.cfg.hc + 1) * d;
        let kvb = self.b.take(rows * kv_out * 2)?;
        self.lin_rows("engram.wkv", w, C::EngramWkv, None, e, k_in, k_in, rows, kvb, kv_out, false)?;
        let (wq, wk) = (w.param(C::EngramQ)?, w.param(C::EngramK)?);
        let out = self.b.take(rows * 4 * d * 2)?;
        let a = km::EngramGateArgs { h, h_ld: (4 * d) as i64, kv: kvb, kv_ld: kv_out as i64, wq: wq.ptr.0, wk: wk.ptr.0, gate_on: 0, out, o_ld: (4 * d) as i64,
                                     wq_fmt: fmt_of(wq)?, wk_fmt: fmt_of(wk)?, dim: d as i32, rows: rows as i32, eps: self.eps(), inv_dim: 1.0 / d as f32,
                                     inv_sqrt_dim: (d as f64).powf(-0.5) as f32, clamp_min: 1e-6 };
        self.g.launch1("d41_engram_gate", (4, rows as u32, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("engram.gate", "d41_engram_gate", rows);
        Ok(out)
    }
}

pub fn bytemuck_f32(v: &[f32]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) } }
pub fn bytemuck_u64(v: &[u64]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) } }
pub fn bytemuck_i32(v: &[i32]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) } }
pub fn bytemuck_u16(v: &[u16]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) } }
