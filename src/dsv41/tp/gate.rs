//! DSV4.1 TP (leg TP): the per-layer gate AT TP (plan Revision 5): every trunk layer of the full model, fed its
//! DUMPED input from REFb (`/mnt/models/dsv41/ref/`), never a chain. Every rank runs the same sequence (every
//! reduce is a collective); rank 0 prints. The points are SL's (`fwd::gate`), with SL's EXL3 tolerances
//! (`fwd::cmp::tol`), now through the TP split:
//!
//! | point | what TP changes |
//! |---|---|
//! | `attn_mix` / `ffn_mix` | the rank's K slice of `hc_*_fn`, the 25 partials reduced, then Sinkhorn |
//! | `attn_out` (dumped top-k) / `attn_out_own` (our top-k) | the rank's heads, `wo_a` groups and `wo_b` input range, FP32 partials reduced |
//! | `ffn_out` (dumped routing) / `ffn_out_own` (ours) | every expert and the shared expert at the rank's inner width, FP32 partials reduced |
//! | `engram` (1, 14) | the rank's Engram heads' rows and `wkv` columns, the FP32 partial reduced, then the gate |
//! | `attn_in`, `x_mid`, `ffn_in`, `router`, `block_out`, `topk` | replicated pieces (SL's), consuming the reduced mixes |
//!
//! The window / compressor / index caches are built from the dumped `attn_in` of every prompt row (prefill shape,
//! one call), as SL's gate does; their own cache checks are SL's (replicated code, not re-checked here).

use super::comm::Dsv41Comm;
use super::fwd::{Lane, TpModel};
use super::frame::MIX_STRIDE;
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::dump::{RefTensor, RefbDump};
use crate::dsv41::fwd::cmp::{self, Cmp, Pass};
use crate::dsv41::fwd::dev::{Buf, Gpu};
use crate::dsv41::fwd::layer::{SrcState, Union, Window};
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Row { pub fixture: String, pub layer: usize, pub point: &'static str, pub c: Cmp, pub pass: Option<bool>, pub extra: String }

fn rows_at(t: &RefTensor, pos: &[i64]) -> Result<Vec<f32>> {
    let s = t.stride();
    let mut v = Vec::with_capacity(pos.len() * s);
    for &p in pos { let i = t.index_of(p).with_context(|| format!("{}: no row {p}", t.name))?; v.extend_from_slice(t.row_f(i)); }
    Ok(v)
}
fn rows_at_i(t: &RefTensor, pos: &[i64]) -> Result<Vec<i64>> {
    let s = t.stride();
    let mut v = Vec::with_capacity(pos.len() * s);
    for &p in pos { let i = t.index_of(p).with_context(|| format!("{}: no row {p}", t.name))?; v.extend_from_slice(t.row_i(i)); }
    Ok(v)
}
fn bf16_bits(v: &[f32]) -> Vec<u16> { v.iter().map(|&x| cmp::f32_to_bf16_bits(x)).collect() }

/// Inputs uploaded for one layer (dropped at the next).
struct Keep(Vec<cudarc::driver::CudaSlice<u8>>);
impl Keep {
    fn bytes(&mut self, g: &Gpu, b: &[u8]) -> Result<u64> { let s = g.up(b)?; let p = crate::dsv41::fwd::dev::dp(&s); self.0.push(s); Ok(p) }
    fn bf16(&mut self, g: &Gpu, v: &[f32]) -> Result<u64> { let b = bf16_bits(v); self.bytes(g, unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 2) }) }
    fn f32(&mut self, g: &Gpu, v: &[f32]) -> Result<u64> { self.bytes(g, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }) }
    fn i32(&mut self, g: &Gpu, v: &[i32]) -> Result<u64> { self.bytes(g, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }) }
}

pub struct TpGate { pub rows: Vec<Row>, pub pass: Pass }

impl TpGate {
    fn push(&mut self, fx: &str, layer: usize, point: &'static str, c: Cmp, extra: String) {
        let t = cmp::tol(self.pass, point, layer);
        let pass = if t.rel.is_infinite() { None } else { Some(c.rel <= t.rel && c.same >= t.same) };
        self.rows.push(Row { fixture: fx.into(), layer, point, c, pass, extra });
    }

    pub fn fmt(r: &Row) -> String {
        let v = match r.pass { Some(true) => "PASS", Some(false) => "FAIL", None => "info" };
        format!("{:<11} L{:02} {:<13} rel-L2 {:.3e}  same {:7.3}%  n {:>9}  {v} {}", r.fixture, r.layer, r.point, r.c.rel, 100.0 * r.c.same, r.c.n, r.extra)
    }

    /// Every trunk layer of `m` on fixture `d` (named `name`). `say` prints (rank 0).
    pub fn run_fixture(&mut self, m: &mut TpModel, d: &RefbDump, name: &str, say: &mut dyn FnMut(String)) -> Result<()> {
        let cfg = m.cfg;
        let g = m.f.g;
        let p = d.prompt_len()?;
        let (dm, hc) = (cfg.hidden, cfg.hc);
        let hd = cfg.head_dim;
        // a gate lane: windows over every prompt row, KV-source caches for P positions
        let mut gl = Lane { pos: 0, win: Vec::new(), src: HashMap::new(), hash: crate::dsv41::engram::hash::NgramHashState::new(), cap_rows: p, max_seq: p,
                           hist: Vec::new(), temp: 0.0, seed: 0, kind: crate::dsv41::ced::schedule::StateKind::Full,
                           vl_types: Vec::new(), vl_row_ops: Vec::new(), vl_aligner: None };
        for _ in 0..cfg.layers {
            let slots = cfg.swa_slots();
            gl.win.push(Window { ring_q: g.buf(slots * hd)?, ring_s: g.buf(slots * hd / 32)?, lin_q: g.buf((p + 128) * hd)?, lin_s: g.buf((p + 128) * hd / 32)?, lin_base: 0, rows: p + 128 });
        }
        for &s in &cfg.kv_sources {
            let ratio = cfg.ratio(s);
            let cap = p / ratio.max(1) as usize + 2;
            let rr = cfg.frontier_rows(ratio);
            gl.src.insert(s, SrcState { layer: s, ratio, n_groups: 0, kv_main: g.buf(cap * 288)?, iq: g.buf(cap * 64)?, is: g.buf(cap * 4)?,
                                        front: if ratio >= 2 { Some((g.buf(rr * hd * 4)?, g.buf(rr * hd * 4)?)) } else { None }, latent: g.buf((p + 1) * hd * 2)? });
        }
        std::mem::swap(&mut m.lanes[0], &mut gl);
        let saved = std::mem::replace(&mut m.segs, vec![super::fwd::Seg { lane: 0, row0: 0, n: p, p0: 0 }]);
        let r = self.layers(m, d, name, p, say);
        m.segs = saved;
        std::mem::swap(&mut m.lanes[0], &mut gl);
        drop(gl);
        g.sync()?;
        let _ = (dm, hc);
        r
    }

    fn layers(&mut self, m: &mut TpModel, d: &RefbDump, name: &str, p: usize, say: &mut dyn FnMut(String)) -> Result<()> {
        let cfg = m.cfg;
        let g = m.f.g;
        let (dm, hc) = (cfg.hidden, cfg.hc);
        let hcd = hc * dm;
        let mut uni: Option<Union> = None;
        let mut sel_keep: HashMap<usize, Buf> = HashMap::new();
        for l in 0..cfg.layers {
            if m.layers[l].is_none() { continue; }
            let n0 = self.rows.len();
            g.sync()?;
            // the ranks read this layer's dumps from different disks: meet before its first reduce
            m.barrier(0x6000 | l as u32)?;
            m.f.b.reset();
            let mut keep = Keep(Vec::new());
            let x_in = d.layer(l, "x_in")?;
            let pos = x_in.positions();
            let rows = pos.len();
            let dpos = keep.i32(g, &pos.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
            let dx = keep.bf16(g, &x_in.f)?;
            // one frame for this layer's reduces: [rows][dim] hidden, [rows][32] mix, [rows][(hc + 1) dim] extra
            let frame = g.buf(rows * (dm + MIX_STRIDE + (hc + 1) * dm) * 4)?;
            let (fh, fmx, fex) = (frame.ptr(), frame.ptr() + (rows * dm * 4) as u64, frame.ptr() + (rows * (dm + MIX_STRIDE) * 4) as u64);
            // ---- Engram: dumped x_pre_engram + the dumped hash ids -> x_in
            if let Some(li) = cfg.engram_layers.iter().position(|&e| e == l).filter(|_| m.eio.is_some()) {
                let h = d.layer(l, "x_pre_engram")?;
                let dh = keep.bf16(g, &rows_at(&h, &pos)?)?;
                let ids_all = d.global("engram_hash_ids")?;
                let ids: Vec<u32> = rows_at_i(&ids_all, &pos)?.into_iter().map(|x| x as u32).collect();
                let per = ids_all.stride();
                let kv_out = (hc + 1) * dm;
                let chunk = m.max_rows;
                let mut r0 = 0;
                while r0 < rows {
                    let n = (rows - r0).min(chunk);
                    let (e, stride) = m.engram_rows_ids(&ids[r0 * per..(r0 + n) * per], n)?;
                    m.engram_partial(l, e + (li * stride) as u64, n, fex + (r0 * kv_out * 4) as u64)?;
                    g.sync()?;
                    m.engram_check_pub()?;
                    r0 += n;
                }
                m.comm.all_reduce_raw(m.cs, fex, rows * kv_out)?;
                let kvb = m.f.b.take(rows * kv_out * 2)?;
                crate::dsv41::fwd::glue::add_round(g, fex, 0, kvb, rows, kv_out)?;
                let out = m.engram_gate(l, dh, kvb, rows)?;
                let got = g.down_bf16(out, rows * hcd)?;
                self.push(name, l, "engram", cmp::cmp(&got, &x_in.f), String::new());
            }
            // ---- attention-side mix (K slice, reduced)
            m.mix_partial(m.layers[l].as_ref().unwrap().mix_a, dx, rows, fmx)?;
            m.comm.all_reduce_raw(m.cs, fmx, rows * MIX_STRIDE)?;
            let rec_a = m.mix_finish(l, true, fmx, rows)?;
            let got = g.down::<f32>(rec_a, rows * 24)?;
            self.push(name, l, "attn_mix", cmp::cmp(&got, &mix_ref(d, l, "attn", &pos)?), String::new());
            let pre_in = d.layer(l, "pre_mix_in")?;
            let dpre = keep.f32(g, &rows_at(&pre_in, &pos)?)?;
            let an = m.layers[l].as_ref().unwrap().w.param(C::AttnNorm)?.clone();
            let ain = m.f.hc_pre_norm(dx, dpre, 4, &an, rows)?;
            let ain_ref = d.layer(l, "attn_in")?;
            ensure!(ain_ref.n_rows() == p, "layer {l}: attn_in holds {} rows, the prompt {p}", ain_ref.n_rows());
            let got = g.down_bf16(ain, rows * dm)?;
            self.push(name, l, "attn_in", cmp::cmp(&got, &rows_at(&ain_ref, &pos)?), String::new());
            // ---- window + compressor over every prompt row (dumped attn_in), the gated rows' q path
            let ratio = cfg.ratio(l);
            let compressed = ratio > 0;
            let dall = keep.bf16(g, &ain_ref.f)?;
            m.window(l, dall, p, compressed)?;
            if compressed && cfg.kv_source_of(l) == Some(l) {
                m.lanes[0].src.get_mut(&l).unwrap().n_groups = 0;
                m.compressor(l, dall, p)?;
            }
            let qx = keep.bf16(g, &rows_at(&ain_ref, &pos)?)?;
            let qr = m.q_lora(l, qx, rows)?;
            let mut outs: Vec<(&'static str, Option<(u64, usize, usize)>)> = Vec::new();
            let src = if compressed { Some(cfg.kv_source_of(l).context("no KV source")?) } else { None };
            if !compressed { outs.push(("attn_out", None)); } else {
                let s = src.unwrap();
                let ngr = m.lanes[0].src[&s].n_groups;
                if cfg.index_source_of(l) == Some(l) {
                    let cand_out = cfg.candidate_source == Some(l);
                    let cin = if cfg.uses_candidates(l) { Some(uni.as_ref().context("layer 20's candidates not computed")?) } else { None };
                    let (sel, _cnt, un) = m.f.indexer(&m.layers[l].as_ref().unwrap().w, qr, qx, dpos, rows, &m.lanes[0].src[&s], cand_out, cin)?;
                    if let Some(u) = un { uni = Some(u); }
                    let tk = d.layer(l, "topk_compressed_pos")?;
                    let refk = rows_at_i(&tk, &pos)?;
                    let ours = g.down::<i32>(sel, rows * 512)?;
                    let (same, ov, worst) = cmp::topk_agree(&ours, 512, &refk, tk.stride(), rows);
                    let ok = ov >= cmp::topk_overlap_min(self.pass);
                    self.push(name, l, "topk", Cmp { rel: 0.0, same: same as f64 / rows as f64, max_abs: 0.0, n: rows }, format!("mean overlap {ov:.4}, worst {worst:.4}"));
                    if let Some(r) = self.rows.last_mut() { r.pass = Some(r.pass.unwrap_or(true) && ok); }
                    let kb = g.buf(rows * 512 * 4)?;
                    g.d2d(kb.ptr(), sel, rows * 512 * 4)?;
                    sel_keep.insert(l, kb);
                }
                let isrc = cfg.index_source_of(l).context("no index source")?;
                let tk = d.layer(isrc, "topk_compressed_pos")?;
                let ks = tk.stride();
                let refk: Vec<i32> = rows_at_i(&tk, &pos)?.into_iter().map(|x| x as i32).collect();
                let mut padded = vec![-1i32; rows * 512];
                for r in 0..rows { padded[r * 512..r * 512 + ks.min(512)].copy_from_slice(&refk[r * ks..r * ks + ks.min(512)]); }
                let dsel = keep.i32(g, &padded)?;
                outs.push(("attn_out", Some((dsel, 512, ks.min(512)))));
                if let Some(kb) = sel_keep.get(&isrc) { if cfg.index_source_of(l) == Some(l) { outs.push(("attn_out_own", Some((kb.ptr(), 512, 512.min(ngr.max(1)))))); } }
            }
            let aref = rows_at(&d.layer(l, "attn_out")?, &pos)?;
            for (pt, sel) in outs {
                m.attn_core(l, qr, dpos, rows, src, sel, fh)?;
                m.comm.all_reduce_raw(m.cs, fh, rows * dm)?;
                let ao = m.f.b.take(rows * dm * 2)?;
                crate::dsv41::fwd::glue::add_round(g, fh, 0, ao, rows, dm)?;
                let got = g.down_bf16(ao, rows * dm)?;
                self.push(name, l, pt, cmp::cmp(&got, &aref), if pt == "attn_out_own" { "own top-k (info)".into() } else { String::new() });
                if pt == "attn_out_own" { if let Some(r) = self.rows.last_mut() { r.pass = None; } }
            }
            // ---- x_mid = hc_post(dumped attn_out, dumped x_in, our reduced attn mix)
            let da = keep.bf16(g, &aref)?;
            let xm = m.f.hc_post(da, dx, rec_a, rows)?;
            let xmid = rows_at(&d.layer(l, "x_mid")?, &pos)?;
            let got = g.down_bf16(xm, rows * hcd)?;
            self.push(name, l, "x_mid", cmp::cmp(&got, &xmid), String::new());
            // ---- FFN-side mix on the dumped x_mid; ffn_in with the dumped attn_pre
            let dxm = keep.bf16(g, &xmid)?;
            m.mix_partial(m.layers[l].as_ref().unwrap().mix_f, dxm, rows, fmx)?;
            m.comm.all_reduce_raw(m.cs, fmx, rows * MIX_STRIDE)?;
            let rec_f = m.mix_finish(l, false, fmx, rows)?;
            let got = g.down::<f32>(rec_f, rows * 24)?;
            self.push(name, l, "ffn_mix", cmp::cmp(&got, &mix_ref(d, l, "ffn", &pos)?), String::new());
            let apre = keep.f32(g, &rows_at(&d.layer(l, "attn_pre")?, &pos)?)?;
            let fnm = m.layers[l].as_ref().unwrap().w.param(C::FfnNorm)?.clone();
            let fin = m.f.hc_pre_norm(dxm, apre, 4, &fnm, rows)?;
            let fin_ref = rows_at(&d.layer(l, "ffn_in")?, &pos)?;
            let got = g.down_bf16(fin, rows * dm)?;
            self.push(name, l, "ffn_in", cmp::cmp(&got, &fin_ref), String::new());
            // ---- router + MoE on the dumped ffn_in
            let dfin = keep.bf16(g, &fin_ref)?;
            let (ids, wts) = m.f.router(&m.layers[l].as_ref().unwrap().w, dfin, rows)?;
            let (_, k) = cfg.moe_of(l);
            let gi = g.down::<i32>(ids, rows * k)?;
            let ridx = rows_at_i(&d.layer(l, "router_idx")?, &pos)?;
            let rw = rows_at(&d.layer(l, "router_w")?, &pos)?;
            let mut same = 0usize;
            for r in 0..rows {
                let mut a: Vec<i64> = gi[r * k..(r + 1) * k].iter().map(|&x| x as i64).collect();
                let mut b: Vec<i64> = ridx[r * k..(r + 1) * k].to_vec();
                a.sort_unstable(); b.sort_unstable();
                if a == b { same += 1; }
            }
            self.push(name, l, "router", Cmp { rel: 0.0, same: same as f64 / rows as f64, max_abs: 0.0, n: rows }, format!("{same}/{rows} rows same top-{k} set"));
            let dids = keep.i32(g, &ridx.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
            let dw = keep.f32(g, &rw)?;
            let fref = rows_at(&d.layer(l, "ffn_out")?, &pos)?;
            for (pt, (i_, w_)) in [("ffn_out", (dids, dw)), ("ffn_out_own", (ids, wts))] {
                m.ffn_core(l, dfin, rows, i_, w_, fh)?;
                m.comm.all_reduce_raw(m.cs, fh, rows * dm)?;
                let fo = m.f.b.take(rows * dm * 2)?;
                crate::dsv41::fwd::glue::add_round(g, fh, 0, fo, rows, dm)?;
                let got = g.down_bf16(fo, rows * dm)?;
                self.push(name, l, pt, cmp::cmp(&got, &fref), if pt == "ffn_out_own" { "own routing (info)".into() } else { String::new() });
                if pt == "ffn_out_own" { if let Some(r) = self.rows.last_mut() { r.pass = None; } }
            }
            // ---- block out = hc_post(dumped ffn_out, dumped x_mid, our reduced ffn mix) vs the next layer's input
            if l + 1 < cfg.layers && d.layer_opt(l + 1, "x_in")?.is_some() {
                let nx = if cfg.engram_layers.contains(&(l + 1)) { "x_pre_engram" } else { "x_in" };
                if let Some(nt) = d.layer_opt(l + 1, nx)? {
                    let df = keep.bf16(g, &fref)?;
                    let bo = m.f.hc_post(df, dxm, rec_f, rows)?;
                    let got = g.down_bf16(bo, rows * hcd)?;
                    self.push(name, l, "block_out", cmp::cmp(&got, &rows_at(&nt, &pos)?), String::new());
                }
            }
            g.sync()?;
            m.comm.check("the TP gate")?;
            for r in &self.rows[n0..] { say(Self::fmt(r)); }
            drop(frame);
            drop(keep);
        }
        Ok(())
    }
}

/// The dumped mixes of a sublayer as a [rows][24] record (pre 4, post 4, comb 16).
fn mix_ref(d: &RefbDump, l: usize, side: &str, pos: &[i64]) -> Result<Vec<f32>> {
    let pre = rows_at(&d.layer(l, &format!("{side}_pre"))?, pos)?;
    let post = rows_at(&d.layer(l, &format!("{side}_post"))?, pos)?;
    let comb = rows_at(&d.layer(l, &format!("{side}_comb"))?, pos)?;
    let mut v = Vec::with_capacity(pos.len() * 24);
    for r in 0..pos.len() {
        v.extend_from_slice(&pre[r * 4..r * 4 + 4]);
        v.extend_from_slice(&post[r * 4..r * 4 + 4]);
        v.extend_from_slice(&comb[r * 16..r * 16 + 16]);
    }
    Ok(v)
}
