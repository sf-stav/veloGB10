//! CE: one layer of one lane over one prefill chunk, on the rows the schedule gives it (`ChunkRows`).
//!
//! Attention side ([`attn_chunk`]), in this order (every kernel on the one blocking stream):
//! 1. the KV-source pass on the chunk's KV rows (`compressor_chunk`: ratio-r pooling with the frontier ring
//!    carried across chunks, main-KV records and index-K planes at absolute group ids; KC's `group_range`);
//! 2. window K/V of the chunk's window rows into the LINEAR planes only (`window_lin`);
//! 3. q path, indexer (index sources; layer 20 keeps its rows' candidate bitmaps for 24-36 in [`Cands`]),
//!    sparse attention (history from the ring, this chunk from the planes), wo_a / wo_b;
//! 4. the ring update ([`super::lane::LayerLane::commit_ring`]).
//!
//! Row-local side ([`ffn_rows`]): x_mid = hc_post(attn_out, x, attn mix), the FFN mix, ffn_in (with the
//! attention side's pre-mix: one sublayer late), router, routed + shared experts, block_out = hc_post.
//!
//! Every dense linear of a prefill call runs in the WIDE row family (`Fwd::wide_min`): a call tail with
//! fewer rows is padded, so a row's bits never depend on how many rows its call had (REFb's condition for
//! CED == exact and chunked == single-shot bit for bit).

use super::lane::{Lane, LayerLane};
use super::schedule::{n_of, ChunkRows};
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::fwd::dev::Buf;
use crate::dsv41::fwd::layer::{fmt_of, Fwd, SrcState, HC_DIM};
use crate::dsv41::fwd::receipts;
use crate::dsv41::fwd::weights::LayerW;
use crate::dsv41::gpu::{indexer as ki, kv};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// Layer 20's per-row candidate bitmaps of one chunk (rows `[row0, row0 + n)`), which layers 24-36 turn into
/// their own row-block unions (each row keeps its own pool, so any row-block layout gives the same selection).
pub struct Cand { pub row0: usize, pub n: usize, pub bm: Buf, pub pos: Buf, pub nbw_cap: usize, pub nb_cap: usize, pub n_keys_cap: usize, pub ratio: u32 }
/// An index source's selection of one chunk's query rows `[row0, row0 + n)` ([n][512] i32), which the layers that
/// reuse it (3-7 read 2's, 21-23 read 20's, ...) read for their own query rows (a subset: q0 grows with the layer).
pub struct Sel { pub row0: usize, pub n: usize, pub sel: Buf }
/// What the layers of one chunk hand each other: candidates by chunk start, selections by (index source, chunk start).
#[derive(Default)]
pub struct Cands { pub cand: BTreeMap<usize, Cand>, pub sel: BTreeMap<(usize, usize), Sel> }
impl Cands {
    /// Drop what chunk `c0` stored (after its last layer ran).
    pub fn drop_chunk(&mut self, c0: usize) { self.cand.remove(&c0); self.sel.retain(|k, _| k.1 != c0); }
}

/// What the attention side of a layer produced for the chunk's query rows.
pub struct AttnChunk { pub q: (usize, usize), pub out: u64, pub sel: Option<u64>, pub sel_cnt: Option<u64> }

fn up_pos(f: &mut Fwd, r: (usize, usize)) -> Result<u64> {
    let v: Vec<i32> = (r.0..r.1).map(|p| p as i32).collect();
    let d = f.b.take(v.len().max(1) * 4)?;
    f.g.h2d_async(d, &v)?;
    Ok(d)
}

/// Window K/V of positions [p0, p0 + rows) from x [rows][dim] into the layer's LINEAR planes only.
pub fn window_lin(f: &mut Fwd, w: &LayerW, x: u64, p0: usize, rows: usize, ll: &mut LayerLane) -> Result<()> {
    let (d, hd) = (f.cfg.hidden, f.cfg.head_dim);
    ensure!(rows <= ll.win.rows, "layer {}: {rows} window rows > the lane's {} linear rows", w.layer, ll.win.rows);
    let compressed = f.cfg.ratio(w.layer) > 0;
    let raw = f.b.take(rows * hd * 2)?;
    f.lin_rows("attn.wkv", w, C::AttnWkv, None, x, d, d, rows, raw, hd, false)?;
    let norm = w.param(C::AttnKvNorm)?;
    let freq = f.b.take(128)?;
    f.g.h2d_async(freq, &f.rope.kc.of(compressed))?;
    let a = kv::SwaArgs { kv_raw: raw, norm_w: norm.ptr.0, inv_freq: freq, ring_q: 0, ring_s: 0, chunk_q: ll.win.lin_q.ptr(), chunk_s: ll.win.lin_s.ptr(),
                          p0: p0 as i64, rows: rows as i32, norm_fmt: fmt_of(norm)?, eps: f.cfg.rms_eps as f32 };
    f.g.launch1("d41_kc_swa_write", (rows as u32, 1, 1), (128, 1, 1), 0, &a)?;
    receipts::kernel("ced.swa_lin", "d41_kc_swa_write(planes)", rows);
    ll.win.lin_base = p0 as i64;
    Ok(())
}

/// The compressor over positions [p0, p0 + rows) of x into an existing source state (any p0: chunked prefill,
/// decode, verify are one code path, KC's `group_range`), plus the index keys of the groups it completes.
pub fn compressor_chunk(f: &mut Fwd, w: &LayerW, st: &mut SrcState, x: u64, p0: usize, rows: usize) -> Result<()> {
    let (d, hd, l) = (f.cfg.hidden, f.cfg.head_dim, w.layer);
    let ratio = st.ratio;
    let (g0, ng) = kv::group_range(p0 as u64, rows as u64, ratio);
    let (g0, ng) = (g0 as usize, ng as usize);
    ensure!(g0 == st.n_groups, "layer {l}: compressor call at {p0} starts group {g0}, the state holds {} (calls must be consecutive)", st.n_groups);
    ensure!((g0 + ng) * 288 <= st.kv_main.bytes, "layer {l}: {} groups > the cache", g0 + ng);
    let rr = f.cfg.frontier_rows(ratio);
    let norm = w.param(C::CompNorm)?;
    let freq = f.b.take(128)?;
    f.g.h2d_async(freq, &f.rope.kc.compressed)?;
    let latent = f.b.take(ng.max(1) * hd * 2)?;
    let mut a = kv::CompressArgs { norm_w: norm.ptr.0, inv_freq: freq, latent_out: latent, kv_main: st.kv_main.ptr(), p0: p0 as i64, rows: rows as i32,
                                   ratio: ratio as i32, ring_rows: rr as i32, g_first: g0 as i32, n_groups: ng as i32, norm_fmt: fmt_of(norm)?,
                                   eps: f.cfg.rms_eps as f32, ..Default::default() };
    let err = f.b.take(4)?;
    f.g.memset(err, 4, 0)?;
    a.err = err;
    let (kvp, scp);
    if ratio >= 2 {
        kvp = f.b.take(rows * hd * 4)?;
        scp = f.b.take(rows * hd * 4)?;
        f.lin_rows("comp.wkv", w, C::CompWkv, None, x, d, d, rows, kvp, hd, true)?;
        f.lin_rows("comp.wgate", w, C::CompWgate, None, x, d, d, rows, scp, hd, true)?;
        let (fk, fs) = st.front.as_ref().map(|(a, b)| (a.ptr(), b.ptr())).context("ratio >= 2 without a frontier ring")?;
        a.kv = kvp; a.score = scp; a.ring_kv = fk; a.ring_score = fs;
    } else {
        kvp = f.b.take(rows * hd * 2)?;
        scp = 0;
        f.lin_rows("comp.wkv", w, C::CompWkv, None, x, d, d, rows, kvp, hd, false)?;
        a.kv16 = kvp;
    }
    if ng > 0 { f.g.launch1("d41_kc_compress", (ng as u32, 1, 1), (128, 1, 1), 0, &a)?; }
    if let Some((fk, fs)) = &st.front {
        let s = kv::StashArgs { kv: kvp, score: scp, ring_kv: fk.ptr(), ring_score: fs.ptr(), p0: p0 as i64, rows: rows as i32, ring_rows: rr as i32 };
        f.g.launch1("d41_kc_stash", (rows.min(rr) as u32, 1, 1), (128, 1, 1), 0, &s)?;
    }
    receipts::kernel("ced.compress", "d41_kc_compress+d41_kc_stash (p0 > 0)", rows);
    if ng > 0 {
        let id = f.cfg.index_head_dim;
        let kraw = f.b.take(ng * id * 2)?;
        f.lin_rows("idx.wk", w, C::IdxWk, None, latent, hd, hd, ng, kraw, id, false)?;
        let kn = w.param(C::IdxKNorm)?;
        let ia = kv::IndexArgs { k_raw: kraw, norm_w: kn.ptr.0, inv_freq: freq, q_plane: st.iq.ptr(), s_plane: st.is.ptr(), ratio: ratio as i32, g_first: g0 as i32,
                                 n_groups: ng as i32, norm_fmt: fmt_of(kn)?, eps: f.cfg.rms_eps as f32 };
        f.g.launch1("d41_kc_index_k", ((ng as u32).div_ceil(4), 1, 1), (32, 4, 1), 0, &ia)?;
    }
    if f.checks {
        let e = f.g.down::<i32>(err, 1)?[0];
        ensure!(e == 0, "compressor layer {l} at {p0}: kernel error {e}");
    }
    st.n_groups = g0 + ng;
    Ok(())
}

/// The indexer's selection for the chunk's query rows q (positions `qr0..`) of an index source.
/// Layer 20 (candidate source) stores its rows' candidate bitmaps in `cands[c0]`; a candidate user (24-36)
/// builds its own row-block union from the stored bitmaps of its rows.
#[allow(clippy::too_many_arguments)]
fn indexer_chunk(f: &mut Fwd, w: &LayerW, qr: u64, x: u64, pos: u64, q: (usize, usize), src: &SrcState, keys_at_end: usize, c0: usize, cands: &mut Cands)
                 -> Result<(u64, u64)> {
    let (d, ql, l) = (f.cfg.hidden, f.cfg.q_lora_rank, w.layer);
    let rows = n_of(q);
    let cand_out = f.cfg.candidate_source == Some(l);
    let cand_in = f.cfg.uses_candidates(l);
    let iq = f.b.take(rows * ki::HEADS * ki::HEAD_DIM * 2)?;
    f.lin_rows("idx.wq_b", w, C::IdxWqB, None, qr, ql, ql, rows, iq, ki::HEADS * ki::HEAD_DIM, false)?;
    let wp = f.b.take(rows * ki::HEADS * 2)?;
    f.lin_rows("idx.weights_proj", w, C::IdxWeightsProj, None, x, d, d, rows, wp, ki::HEADS, false)?;
    let r = 8usize;
    let rbs: Vec<(i32, i32)> = (0..rows).step_by(r).map(|i| (i as i32, (rows - i).min(r) as i32)).collect();
    let n_rb = rbs.len();
    let cap = keys_at_end.max(1);
    let ucap = r * ki::CAND_BLOCKS;
    let sz = ki::sizes(n_rb, r, 1, rows, cap, ucap);
    let rbv: Vec<u64> = rbs.iter().flat_map(|&(r0, n)| [src.iq.ptr(), src.is.ptr(), (r0 as u32 as u64) | ((n as u32 as u64) << 32), 0]).collect();
    let rbd = f.b.take(rbv.len() * 8)?;
    f.g.h2d_async(rbd, &rbv)?;
    let sel = f.b.take(sz.sel_i32 * 4)?;
    let cnt = f.b.take(rows * 4)?;
    let part = f.b.take(sz.part_u64.max(1) * 8)?;
    let pcnt = f.b.take(sz.pcnt_i32.max(1) * 4)?;
    let bmax = if cand_out { f.b.take(sz.bmax_u16.max(1) * 2)? } else { 0 };
    let cand = f.b.take(sz.cand_i32.max(1) * 4)?;
    let ccnt = f.b.take(rows * 4)?;
    let err = f.b.take(4)?;
    f.g.memset(err, 4, 0)?;
    let mut a = ki::IdxArgs::default();
    a.iq = iq; a.w = wp; a.pos = pos; a.rbs = rbd; a.sel = sel; a.sel_cnt = cnt; a.part = part; a.pcnt = pcnt; a.bmax = bmax; a.cand = cand; a.cand_cnt = ccnt;
    a.err = err; a.tick = 0;
    a.n_rb = n_rb as i32; a.rows_total = rows as i32; a.iq_ld = (ki::HEADS * ki::HEAD_DIM) as i32; a.w_ld = ki::HEADS as i32; a.in_f32 = 0;
    a.ratio = src.ratio as i32; a.write_bmax = cand_out as i32;
    a.n_keys_cap = cap as i32; a.nb_cap = cap.div_ceil(ki::CAND_BLOCK) as i32; a.nbw_cap = (a.nb_cap as usize).div_ceil(32) as i32;
    a.g = 1; a.min_tiles = 8; a.dbg_cap = 0; a.ucap = ucap as i32; a.r = r as i32; a.inv_freq = f.rope.ki;
    let entry;
    if cand_in {
        // the union of THIS layer's row blocks from layer 20's per-row bitmaps of the same chunk
        let st = cands.cand.get(&c0).with_context(|| format!("layer {l}: no layer-20 candidates for the chunk at {c0}"))?;
        ensure!(q.0 >= st.row0 && q.1 <= st.row0 + st.n, "layer {l}: rows {q:?} outside layer 20's candidate rows {}..{}", st.row0, st.row0 + st.n);
        let off = q.0 - st.row0;
        let uni = f.b.take(n_rb * ucap * 4)?;
        let ucnt = f.b.take(n_rb * 4)?;
        let mut u = a;
        u.pos = st.pos.ptr() + (off * 4) as u64;
        u.bm = st.bm.ptr() + (off * st.nbw_cap * 4) as u64;
        u.n_keys_cap = st.n_keys_cap as i32; u.nb_cap = st.nb_cap as i32; u.nbw_cap = st.nbw_cap as i32; u.ratio = st.ratio as i32;
        u.uni = uni; u.ucnt = ucnt;
        ki::union(&f.g.l(), &u)?;
        a.uni = uni; a.ucnt = ucnt; a.mode = 1;
        a.n_keys_cap = st.n_keys_cap as i32; a.nb_cap = st.nb_cap as i32; a.nbw_cap = st.nbw_cap as i32;
        ki::score(&f.g.l(), &a)?;
        entry = "d41_idx_union(own rows)+d41_idx_score_r8(mode 1)";
    } else if cand_out {
        let bm = f.g.buf(sz.bm_u32.max(1) * 4)?;
        let pb = f.g.buf(rows.max(1) * 4)?;
        f.g.d2d(pb.ptr(), pos, rows * 4)?;
        a.bm = bm.ptr();
        a.uni = f.b.take(4)?; a.ucnt = f.b.take(n_rb.max(1) * 4)?;
        ki::score(&f.g.l(), &a)?;
        ki::cand(&f.g.l(), &a)?;
        cands.cand.insert(c0, Cand { row0: q.0, n: rows, bm, pos: pb, nbw_cap: a.nbw_cap as usize, nb_cap: a.nb_cap as usize, n_keys_cap: cap, ratio: src.ratio });
        entry = "d41_idx_score_r8(bmax)+d41_idx_cand";
    } else {
        a.bm = f.b.take(4)?; a.uni = f.b.take(4)?; a.ucnt = f.b.take(n_rb.max(1) * 4)?;
        ki::score(&f.g.l(), &a)?;
        entry = "d41_idx_score_r8";
    }
    receipts::kernel("ced.idx.select", entry, rows);
    if f.checks {
        let e = f.g.down::<i32>(err, 1)?[0];
        ensure!(e == 0, "indexer layer {l}: kernel error word {e}");
    }
    Ok((sel, cnt))
}

/// The attention side of layer `l` over one chunk. `ain` = attention-norm input rows `[cr.a.0, cr.a.1)` (bf16
/// [rows][dim]). Returns attn_out for the query rows (bf16 [rows][dim]).
pub fn attn_chunk(f: &mut Fwd, w: &LayerW, lane: &mut Lane, cr: &ChunkRows, ain: u64, c0: usize, cands: &mut Cands) -> Result<AttnChunk> {
    let cfg: &Dsv41Config = f.cfg;
    let (l, d) = (w.layer, cfg.hidden);
    let row = |r: usize| ain + ((r - cr.a.0) * d * 2) as u64;
    // 1. KV-source pass
    if let Some(kv) = cr.kv {
        if n_of(kv) > 0 {
            let st = lane.get_mut(l)?.src.as_mut().with_context(|| format!("layer {l}: KV source without caches"))?;
            compressor_chunk(f, w, st, row(kv.0), kv.0, n_of(kv))?;
        }
    }
    // 2. window rows -> linear planes
    let nw = n_of(cr.w);
    if nw > 0 { window_lin(f, w, row(cr.w.0), cr.w.0, nw, lane.get_mut(l)?)?; }
    let nq = n_of(cr.q);
    if nq == 0 {
        if nw > 0 { lane.get_mut(l)?.commit_ring(f.g, nw)?; }
        return Ok(AttnChunk { q: cr.q, out: 0, sel: None, sel_cnt: None });
    }
    // 3. queries
    let compressed = cfg.ratio(l) > 0;
    let pos = up_pos(f, cr.q)?;
    let qx = row(cr.q.0);
    let (qr, q) = f.q_path(w, qx, pos, nq, compressed)?;
    let (mut sel, mut sel_cnt) = (None, None);
    let o = if !compressed {
        let win = &lane.get(l)?.win;
        attend(f, w, q, pos, nq, win, None, None, cr.win_lo)?
    } else {
        let src_l = cfg.kv_source_of(l).context("compressed layer without a KV source")?;
        let isrc = cfg.index_source_of(l).context("compressed layer without an index source")?;
        let s = if isrc == l {
            let src = lane.src(src_l)?;
            let keys_at_end = cr.q.1 / src.ratio as usize;
            ensure!(keys_at_end <= src.n_groups, "layer {l}: index keys up to {keys_at_end} needed, the source holds {}", src.n_groups);
            let (s, c) = indexer_chunk(f, w, qr, qx, pos, cr.q, src, keys_at_end, c0, cands)?;
            sel_cnt = Some(c);
            // kept for the layers that reuse this selection in the same chunk
            let keep = f.g.buf(nq * 512 * 4)?;
            f.g.d2d(keep.ptr(), s, nq * 512 * 4)?;
            cands.sel.insert((l, c0), Sel { row0: cr.q.0, n: nq, sel: keep });
            s
        } else {
            let st = cands.sel.get(&(isrc, c0)).with_context(|| format!("layer {l}: index source {isrc} has no selection for the chunk at {c0}"))?;
            ensure!(cr.q.0 >= st.row0 && cr.q.1 <= st.row0 + st.n, "layer {l}: rows {:?} outside {isrc}'s selected rows {}..{}", cr.q, st.row0, st.row0 + st.n);
            receipts::kernel("ced.idx.reuse", "(index source's selection)", nq);
            st.sel.ptr() + ((cr.q.0 - st.row0) * 512 * 4) as u64
        };
        sel = Some(s);
        let src = lane.src(src_l)?;
        let win = &lane.get(l)?.win;
        // a constant cap: -1 entries are masked keys and a fully masked block is an exact identity (KA)
        attend(f, w, q, pos, nq, win, Some(src), Some((s, 512, 512)), cr.win_lo)?
    };
    let out = f.wo(w, o, nq)?;
    // 4. ring
    if nw > 0 { lane.get_mut(l)?.commit_ring(f.g, nw)?; }
    Ok(AttnChunk { q: cr.q, out, sel, sel_cnt })
}

/// Sparse attention of `rows` query rows (SL's `Fwd::attention`, prefill entry) with a window floor: keys below
/// `win_lo` masked (0 = none; the opt-in truncated replay).
#[allow(clippy::too_many_arguments)]
fn attend(f: &mut Fwd, w: &LayerW, q: u64, pos: u64, rows: usize, win: &crate::dsv41::fwd::layer::Window, src: Option<&SrcState>, sel: Option<(u64, usize, usize)>,
          win_lo: usize) -> Result<u64> {
    use crate::dsv41::gpu::attn;
    use crate::dsv41::weights::DevPtr;
    let (h, hd) = (f.cfg.heads, f.cfg.head_dim);
    let compressed = f.cfg.ratio(w.layer) > 0;
    let out = f.b.take(rows * h * hd * 2)?;
    let err = f.b.take(4)?;
    f.g.memset(err, 4, 0)?;
    let lane = attn::LaneSrc { swa_q: win.ring_q.p(), swa_s: win.ring_s.p(), slots: f.cfg.swa_slots(),
                               lin: if win.lin_base == i64::MAX { None } else { Some((win.lin_q.p(), win.lin_s.p(), win.lin_base)) },
                               kv: src.map(|s| (s.kv_main.p(), s.n_groups)), ext: None };
    let (sp, sstride, scap) = sel.unwrap_or((0, 0, 0));
    let call = attn::AttnCall { rows, heads: h, q: DevPtr(q), q_stride: h * hd, sink: w.param(C::AttnSink)?.ptr, pos: DevPtr(pos), win_end: DevPtr(0), lane: DevPtr(0),
                                sel: DevPtr(sp), sel_stride: sstride, sel_cap: scap, n_sel: DevPtr(0), lanes: vec![lane], out: DevPtr(out), out_stride: h * hd,
                                out_q: None, out_s: None, epi: attn::EPI_ROPE | attn::EPI_AQ, rope_freq: f.rope.ka.of(compressed), err: DevPtr(err) };
    attn::launch_prefill_floor(f.g.l(), &call, win_lo)?;
    receipts::kernel("ced.attn", if win_lo > 0 { "d41_attn_prefill(window floor)" } else { "d41_attn_prefill" }, rows);
    if f.checks {
        let e = f.g.down::<i32>(err, 1)?[0];
        ensure!(e == 0, "attention layer {}: kernel error word {e}", w.layer);
    }
    Ok(out)
}

/// The row-local rest of a layer for `rows` query rows: x [rows][4 dim] (the layer's input stream), attn_out
/// [rows][dim], rec_a = the attention mix records of these rows. Returns (block_out [rows][4 dim], ffn mix
/// records [rows][24]: their pre-mix (offset 0, stride 24) is the next layer's pre_mix_in), and the pieces between.
pub fn ffn_rows(f: &mut Fwd, w: &LayerW, x: u64, attn_out: u64, rec_a: u64, rows: usize, own_route: bool, route: Option<(u64, u64)>) -> Result<FfnOut> {
    let xm = f.hc_post(attn_out, x, rec_a, rows)?;
    let rec_f = f.mix(w, false, xm, rows)?;
    let fin = f.hc_pre_norm(xm, rec_a, 24, w.param(C::FfnNorm)?, rows)?;
    let (ids, wts) = match route { Some(r) if !own_route => r, _ => f.router(w, fin, rows)? };
    let fo = f.ffn(w, fin, rows, ids, wts)?;
    let bo = f.hc_post(fo, xm, rec_f, rows)?;
    Ok(FfnOut { block_out: bo, rec_f, ffn_in: fin, ffn_out: fo, x_mid: xm, ids })
}

/// The row-local results (device, `rows` rows each).
pub struct FfnOut { pub block_out: u64, pub rec_f: u64, pub ffn_in: u64, pub ffn_out: u64, pub x_mid: u64, pub ids: u64 }

/// attn_in of rows: RMSNorm(hc_pre(x, pre)) with the previous layer's FFN pre-mix (records, stride 24) or the
/// identity pre-mix at layer 0 / a dumped pre_mix_in (stride 4).
pub fn attn_in_rows(f: &mut Fwd, w: &LayerW, x: u64, pre: u64, pre_stride: usize, rows: usize) -> Result<u64> {
    f.hc_pre_norm(x, pre, pre_stride, w.param(C::AttnNorm)?, rows)
}

/// The identity pre-mix [rows][4] (1, 0, 0, 0) of layer 0 (model.py make_identity_pre_mix).
pub fn identity_pre(f: &mut Fwd, rows: usize) -> Result<u64> {
    let mut v = vec![0f32; rows * 4];
    for r in 0..rows { v[r * 4] = 1.0; }
    let d = f.b.take(rows.max(1) * 16)?;
    f.g.h2d_async(d, &v)?;
    Ok(d)
}

pub const STREAM: usize = HC_DIM;
