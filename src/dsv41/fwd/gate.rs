//! SL gate (plan Revision 5): every layer of the slice, every class, against REFb's dumps, each piece fed its
//! DUMPED input (never a chain of layers):
//!
//! | point | input (dumped) | our computation | reference (dumped) |
//! |---|---|---|---|
//! | `engram` (1, 14) | `x_pre_engram`, `G.engram_hash_ids` | EIO rows -> wkv -> gate | `x_in` |
//! | `attn_mix` | `x_in` | hc_attn mixes | `attn_pre / post / comb` |
//! | `attn_in` | `x_in`, `pre_mix_in` | hc_pre + attn_norm | `attn_in` |
//! | `window_ring` | `attn_in` (all rows) | wkv, kv_norm, RoPE, FP8, ring of 134 | `window_ring` (128 slots) |
//! | `compress_kv` / `index_k` / `kv_state` / `score_state` (sources) | `attn_in` (all rows) | compressor, wk, k_norm, RoPE, FP4 | the caches |
//! | `topk` (index sources) | `attn_in`, the source's caches above | indexer (+ layer 20's candidates) | `topk_compressed_pos` (up to ties) |
//! | `attn_out` | `attn_in`; the source's dumped top-k | q path, window, sparse attention, wo_a, wo_b | `attn_out` |
//! | `attn_out_own` (info) | as above with OUR top-k | | `attn_out` |
//! | `x_mid` | `attn_out`, `x_in` | hc_post with our attn mix | `x_mid` |
//! | `ffn_mix`, `ffn_in` | `x_mid`, `attn_pre` | hc_ffn mixes, hc_pre + ffn_norm | dumped |
//! | `router` | `ffn_in` | gate scores + top-6 | `router_idx` (sets) |
//! | `ffn_out` | `ffn_in`, `router_idx / router_w` | routed experts + shared expert | `ffn_out` |
//! | `ffn_out_own` (info) | `ffn_in` with OUR routing | | `ffn_out` |
//! | `block_out` | `ffn_out`, `x_mid` | hc_post with our ffn mix | the next layer's `x_in` (or `x_pre_engram`) |
//!
//! A reuser layer's KV-source caches are computed from the SOURCE layer's dumped `attn_in` (state of a
//! function of a dumped input, not a chain of streams).

use super::cmp::{self, Cmp, Pass, Tol};
use super::dev::{dp, Gpu};
use super::layer::{bytemuck_i32, bytemuck_u16, Fwd, SrcState, Union, Window, HC_DIM};
use super::weights::{load_layer, LayerW, Pack};
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::dump::{RefTensor, RefbDump};
use crate::dsv41::engram::layout::{EngramConfig, EngramLayout};
use crate::dsv41::engram::prefetch::{EngramIo, EngramIoConfig};
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// One result row of the gate table.
#[derive(Clone, Debug)]
pub struct Row { pub fixture: String, pub layer: usize, pub point: &'static str, pub c: Cmp, pub extra: String, pub tol: Tol, pub pass: Option<bool> }

pub struct Fix { pub name: String, pub d: RefbDump, pub p: usize }

/// Per fixture: the KV sources' caches and layer 20's candidate union (reusers above read them).
#[derive(Default)]
pub struct FixState { pub src: HashMap<usize, SrcState>, pub uni: Option<Union> }

/// Inputs uploaded for one layer's checks (dropped at the next layer).
#[derive(Default)]
pub struct Keep(Vec<cudarc::driver::CudaSlice<u8>>);
impl Keep {
    pub(super) fn bytes(&mut self, g: &Gpu, b: &[u8]) -> Result<u64> { let s = g.up(b)?; let p = dp(&s); self.0.push(s); Ok(p) }
    fn bf16(&mut self, g: &Gpu, v: &[f32]) -> Result<u64> {
        let b: Vec<u16> = v.iter().map(|&x| cmp::f32_to_bf16_bits(x)).collect();
        self.bytes(g, bytemuck_u16(&b))
    }
    fn f32(&mut self, g: &Gpu, v: &[f32]) -> Result<u64> { self.bytes(g, super::layer::bytemuck_f32(v)) }
    fn i32(&mut self, g: &Gpu, v: &[i32]) -> Result<u64> { self.bytes(g, bytemuck_i32(v)) }
}

/// Rows of `t` (a per-row stream) at the positions `pos`, in that order.
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

pub struct Gate<'a> { pub f: Fwd<'a>, pub pass: Pass, pub rows: Vec<Row>, pub io: Option<EngramIo>, pub io_max: usize, pub keep: Keep }

impl<'a> Gate<'a> {
    fn push(&mut self, fx: &str, layer: usize, point: &'static str, c: Cmp, extra: String) {
        let t = cmp::tol(self.pass, point, layer);
        let pass = if t.rel.is_infinite() { None } else { Some(c.rel <= t.rel && c.same >= t.same) };
        self.rows.push(Row { fixture: fx.to_string(), layer, point, c, extra, tol: t, pass });
    }

    /// Open the Engram runtime on every head (one box serves all heads at TP1).
    pub fn open_engram(&mut self, model_dir: &std::path::Path, dir: &std::path::Path, max_tok: usize) -> Result<()> {
        let ec = EngramConfig::from_model_dir(model_dir)?.context("config has no Engram layers")?;
        let layout = Arc::new(EngramLayout::new(ec)?);
        let mut c = EngramIoConfig::new(dir.to_path_buf(), (0..layout.n_heads()).collect());
        c.prefill_max_tok = max_tok;
        c.decode_max_tok = 24;
        c.cache_mib = 1024;
        self.io = Some(EngramIo::open(&self.f.g.dev, layout, c)?);
        self.io_max = max_tok;
        Ok(())
    }

    /// The decoded n-gram rows of `pos` for Engram layer index `li` -> [rows][cols * 256] bf16 (device).
    fn engram_rows(&mut self, ids_all: &RefTensor, pos: &[i64], li: usize) -> Result<u64> {
        let io = self.io.as_mut().context("Engram runtime not open (--engram-dir)")?;
        let per = ids_all.stride();   // 2 layers x 24 columns
        let cols = per / 2;
        let rows = pos.len();
        ensure!(rows <= self.io_max, "{rows} rows > the prefill channel's {}", self.io_max);
        let ids: Vec<u32> = rows_at_i(ids_all, pos)?.into_iter().map(|x| x as u32).collect();
        let out = self.f.b.take(2 * self.io_max * cols * 256 * 2)?;
        let ch = io.ch_prefill();
        io.submit_prefill(&ids, None)?;
        io.launch(ch, out, 0, crate::dsv41::gpu::engram::MODE_DECODE)?;
        super::receipts::kernel("engram.rows", "d41_engram_step_decode", rows);
        self.f.g.dev.synchronize()?;
        let e = io.gpu_ctrl(ch).err;
        ensure!(e == 0, "Engram step error {} ({})", e, crate::dsv41::gpu::engram::err_name(e));
        Ok(out + (li * self.io_max * cols * 256 * 2) as u64)
    }

    /// Every check of layer `l` on one fixture.
    pub fn layer(&mut self, w: &LayerW, fx: &Fix, st: &mut FixState, cfg: &crate::dsv41::config::Dsv41Config, next_in: Option<RefTensor>) -> Result<()> {
        let (l, d, name) = (w.layer, cfg.hidden, fx.name.clone());
        self.f.g.sync()?;
        self.f.b.reset();
        self.keep = Keep::default();
        let x_in = fx.d.layer(l, "x_in")?;
        let pos: Vec<i64> = x_in.positions();
        let rows = pos.len();
        let pos32: Vec<i32> = pos.iter().map(|&p| p as i32).collect();
        let dpos = self.keep.i32(self.f.g, &pos32)?;
        let dx = self.keep.bf16(self.f.g, &x_in.f)?;
        // ---- Engram (dumped x_pre_engram -> x_in)
        if let Some(li) = cfg.engram_layers.iter().position(|&e| e == l) {
            let h = fx.d.layer(l, "x_pre_engram")?;
            let hp = rows_at(&h, &pos)?;
            let dh = self.keep.bf16(self.f.g, &hp)?;
            let ids = fx.d.global("engram_hash_ids")?;
            let e = self.engram_rows(&ids, &pos, li)?;
            let out = self.f.engram(w, dh, e, rows)?;
            let got = self.f.g.down_bf16(out, rows * HC_DIM)?;
            self.push(&name, l, "engram", cmp::cmp(&got, &x_in.f), String::new());
        }
        // ---- attention-side mHC
        let pre_in = fx.d.layer(l, "pre_mix_in")?;
        let rec_a = self.f.mix(w, true, dx, rows)?;
        let got = self.f.g.down::<f32>(rec_a, rows * 24)?;
        let refm = mix_ref(fx, l, "attn", &pos)?;
        self.push(&name, l, "attn_mix", cmp::cmp(&got, &refm), String::new());
        let dpre = self.keep.f32(self.f.g, &rows_at(&pre_in, &pos)?)?;
        let ain = self.f.hc_pre_norm(dx, dpre, 4, w.param(C::AttnNorm)?, rows)?;
        let ain_ref = fx.d.layer(l, "attn_in")?;
        let ain_rows = rows_at(&ain_ref, &pos)?;
        let got = self.f.g.down_bf16(ain, rows * d)?;
        self.push(&name, l, "attn_in", cmp::cmp(&got, &ain_rows), String::new());
        // ---- attention on the dumped attn_in
        let attn_out = self.attention(w, fx, st, cfg, &ain_ref, &pos, dpos)?;
        let aref = rows_at(&fx.d.layer(l, "attn_out")?, &pos)?;
        for (pt, o) in attn_out {
            let got = self.f.g.down_bf16(o, rows * d)?;
            self.push(&name, l, pt, cmp::cmp(&got, &aref), String::new());
        }
        // ---- x_mid = hc_post(dumped attn_out, dumped x_in, our attn mix)
        let da = self.keep.bf16(self.f.g, &aref)?;
        let xm = self.f.hc_post(da, dx, rec_a, rows)?;
        let xmid = rows_at(&fx.d.layer(l, "x_mid")?, &pos)?;
        let got = self.f.g.down_bf16(xm, rows * HC_DIM)?;
        self.push(&name, l, "x_mid", cmp::cmp(&got, &xmid), String::new());
        // ---- FFN-side mHC on the dumped x_mid; ffn_in with the dumped (delayed) attn_pre
        let dxm = self.keep.bf16(self.f.g, &xmid)?;
        let rec_f = self.f.mix(w, false, dxm, rows)?;
        let got = self.f.g.down::<f32>(rec_f, rows * 24)?;
        self.push(&name, l, "ffn_mix", cmp::cmp(&got, &mix_ref(fx, l, "ffn", &pos)?), String::new());
        let apre = self.keep.f32(self.f.g, &rows_at(&fx.d.layer(l, "attn_pre")?, &pos)?)?;
        let fin = self.f.hc_pre_norm(dxm, apre, 4, w.param(C::FfnNorm)?, rows)?;
        let fin_ref = rows_at(&fx.d.layer(l, "ffn_in")?, &pos)?;
        let got = self.f.g.down_bf16(fin, rows * d)?;
        self.push(&name, l, "ffn_in", cmp::cmp(&got, &fin_ref), String::new());
        // ---- router + MoE on the dumped ffn_in
        let dfin = self.keep.bf16(self.f.g, &fin_ref)?;
        let (ids, wts) = self.f.router(w, dfin, rows)?;
        let (_, k) = cfg.moe_of(l);
        let gi = self.f.g.down::<i32>(ids, rows * k)?;
        let ridx = rows_at_i(&fx.d.layer(l, "router_idx")?, &pos)?;
        let rw = rows_at(&fx.d.layer(l, "router_w")?, &pos)?;
        let mut same = 0usize;
        for r in 0..rows {
            let mut a: Vec<i64> = gi[r * k..(r + 1) * k].iter().map(|&x| x as i64).collect();
            let mut b: Vec<i64> = ridx[r * k..(r + 1) * k].to_vec();
            a.sort_unstable(); b.sort_unstable();
            if a == b { same += 1; }
        }
        let c = Cmp { rel: 0.0, same: same as f64 / rows as f64, max_abs: 0.0, n: rows };
        self.push(&name, l, "router", c, format!("{same}/{rows} rows same top-{k} set"));
        let dids = self.keep.i32(self.f.g, &ridx.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
        let dw = self.keep.f32(self.f.g, &rw)?;
        let fo = self.f.ffn(w, dfin, rows, dids, dw)?;
        let fref = rows_at(&fx.d.layer(l, "ffn_out")?, &pos)?;
        let got = self.f.g.down_bf16(fo, rows * d)?;
        self.push(&name, l, "ffn_out", cmp::cmp(&got, &fref), String::new());
        let fo2 = self.f.ffn(w, dfin, rows, ids, wts)?;
        let got = self.f.g.down_bf16(fo2, rows * d)?;
        self.push(&name, l, "ffn_out_own", cmp::cmp(&got, &fref), "own routing (info)".into());
        // ---- block out = hc_post(dumped ffn_out, dumped x_mid, our ffn mix) vs the next layer's input
        if let Some(nx) = next_in {
            let df = self.keep.bf16(self.f.g, &fref)?;
            let bo = self.f.hc_post(df, dxm, rec_f, rows)?;
            let got = self.f.g.down_bf16(bo, rows * HC_DIM)?;
            self.push(&name, l, "block_out", cmp::cmp(&got, &rows_at(&nx, &pos)?), String::new());
        }
        self.f.g.sync()
    }

    /// The attention of layer l over the dumped attn_in: window, compressor (sources), indexer (index
    /// sources), sparse attention with the dumped top-k (and with ours), wo. Returns the attn_out variants.
    fn attention(&mut self, w: &LayerW, fx: &Fix, st: &mut FixState, cfg: &crate::dsv41::config::Dsv41Config, ain_ref: &RefTensor, pos: &[i64], dpos: u64)
                 -> Result<Vec<(&'static str, u64)>> {
        let (l, d, name, p) = (w.layer, cfg.hidden, fx.name.clone(), fx.p);
        ensure!(ain_ref.n_rows() == p, "layer {l}: attn_in holds {} rows, the prompt {p} (an exact all-rows fixture is needed)", ain_ref.n_rows());
        let rows = pos.len();
        let ratio = cfg.ratio(l);
        let compressed = ratio > 0;
        let dall = self.keep.bf16(self.f.g, &ain_ref.f)?;
        let qx = self.keep.bf16(self.f.g, &rows_at(ain_ref, pos)?)?;
        // window over every prompt row
        let win: Window = self.f.window(w, dall, 0, p, compressed)?;
        self.check_ring(fx, l, &win, p)?;
        // KV source: compressor + index keys over every row
        if cfg.kv_source_of(l) == Some(l) && ratio > 0 {
            let cap = p / ratio as usize + 1;
            let s = self.f.compressor(w, dall, p, cap)?;
            self.check_src(fx, l, &s, p)?;
            st.src.insert(l, s);
        }
        let (qr, q) = self.f.q_path(w, qx, dpos, rows, compressed)?;
        let mut outs = Vec::new();
        if !compressed {
            let o = self.f.attention(w, q, dpos, rows, &win, None, None, None)?;
            outs.push(("attn_out", self.f.wo(w, o, rows)?));
            return Ok(outs);
        }
        let src_l = cfg.kv_source_of(l).context("compressed layer without a KV source")?;
        let src = st.src.get(&src_l).with_context(|| format!("layer {l}: KV source {src_l} not computed on this fixture (run it first)"))?;
        // our selection (index sources)
        let mut own: Option<u64> = None;
        if cfg.index_source_of(l) == Some(l) {
            let cand_out = cfg.candidate_source == Some(l);
            let cand_in = if cfg.uses_candidates(l) { Some(st.uni.as_ref().with_context(|| format!("layer {l}: layer 20's candidates not computed"))?) } else { None };
            let (sel, _cnt, un) = self.f.indexer(w, qr, qx, dpos, rows, src, cand_out, cand_in)?;
            if let Some(u) = un { st.uni = Some(u); }
            let tk = fx.d.layer(l, "topk_compressed_pos")?;
            let refk = rows_at_i(&tk, pos)?;
            let ours = self.f.g.down::<i32>(sel, rows * 512)?;
            let (same, ov, worst) = cmp::topk_agree(&ours, 512, &refk, tk.stride(), rows);
            let ok = ov >= cmp::topk_overlap_min(self.pass);
            let c = Cmp { rel: 0.0, same: same as f64 / rows as f64, max_abs: 0.0, n: rows };
            self.push(&name, l, "topk", c, format!("{same}/{rows} rows identical, mean overlap {ov:.5}, worst {worst:.4}"));
            if let Some(r) = self.rows.last_mut() { r.pass = Some(r.pass.unwrap_or(true) && ok); }
            own = Some(sel);
        }
        // the dumped top-k of the layer's index source (the reference's published selection for these rows)
        let isrc = cfg.index_source_of(l).context("compressed layer without an index source")?;
        let tk = fx.d.layer(isrc, "topk_compressed_pos")?;
        let ks = tk.stride();
        let refk: Vec<i32> = rows_at_i(&tk, pos)?.into_iter().map(|x| x as i32).collect();
        let mut padded = vec![-1i32; rows * 512];
        for r in 0..rows { padded[r * 512..r * 512 + ks.min(512)].copy_from_slice(&refk[r * ks..r * ks + ks.min(512)]); }
        let dsel = self.keep.i32(self.f.g, &padded)?;
        let cap = ks.min(512);
        let o = self.f.attention(w, q, dpos, rows, &win, Some(src), Some((dsel, 512, cap)), None)?;
        outs.push(("attn_out", self.f.wo(w, o, rows)?));
        if let Some(sel) = own {
            let o2 = self.f.attention(w, q, dpos, rows, &win, Some(src), Some((sel, 512, 512.min(src.n_groups.max(1)))), None)?;
            outs.push(("attn_out_own", self.f.wo(w, o2, rows)?));
        }
        let _ = d;
        Ok(outs)
    }

    fn check_ring(&mut self, fx: &Fix, l: usize, win: &Window, p: usize) -> Result<()> {
        let hd = 512;
        let Some(rr) = fx.d.layer_opt(l, "window_ring")? else { return Ok(()) };
        let slots = 134usize;
        let (q, s) = (self.f.g.down::<u8>(win.ring_q.ptr(), slots * hd)?, self.f.g.down::<u8>(win.ring_s.ptr(), slots * hd / 32)?);
        let ours = cmp::decode_fp8_rows(&q, &s, slots, hd);
        let n = p.min(128);
        let (mut a, mut b) = (Vec::with_capacity(n * hd), Vec::with_capacity(n * hd));
        for t in p - n..p {
            a.extend_from_slice(&ours[(t % slots) * hd..(t % slots + 1) * hd]);
            b.extend_from_slice(rr.row_f(t % 128));
        }
        let name = fx.name.clone();
        self.push(&name, l, "window_ring", cmp::cmp(&a, &b), format!("last {n} positions"));
        Ok(())
    }

    fn check_src(&mut self, fx: &Fix, l: usize, s: &SrcState, p: usize) -> Result<()> {
        let name = fx.name.clone();
        let ng = s.n_groups;
        if let Some(t) = fx.d.layer_opt(l, "compress_kv")? {
            let ours = cmp::decode_kv_records(&self.f.g.down::<u8>(s.kv_main.ptr(), ng * 288)?, ng);
            self.push(&name, l, "compress_kv", cmp::cmp(&ours, &t.f[..ng * 512]), format!("{ng} groups"));
        }
        if let Some(t) = fx.d.layer_opt(l, "index_k")? {
            let ours = cmp::decode_index_k(&self.f.g.down::<u8>(s.iq.ptr(), ng * 64)?, &self.f.g.down::<u8>(s.is.ptr(), ng * 4)?, ng);
            self.push(&name, l, "index_k", cmp::cmp(&ours, &t.f[..ng * 128]), format!("{ng} groups"));
        }
        if let (Some((fk, fs)), Some(tk), Some(ts)) = (&s.front, fx.d.layer_opt(l, "kv_state")?, fx.d.layer_opt(l, "score_state")?) {
            let r = s.ratio as usize;
            let rem = p % r;
            if rem > 0 {
                let rr = r + 5;
                let (ok, os) = (self.f.g.down::<f32>(fk.ptr(), rr * 512)?, self.f.g.down::<f32>(fs.ptr(), rr * 512)?);
                let (mut a, mut b, mut c2, mut d2) = (vec![], vec![], vec![], vec![]);
                for j in 0..rem {
                    let q = p - rem + j;
                    a.extend_from_slice(&ok[(q % rr) * 512..(q % rr + 1) * 512]);
                    c2.extend_from_slice(&os[(q % rr) * 512..(q % rr + 1) * 512]);
                    b.extend_from_slice(tk.row_f(j));
                    d2.extend_from_slice(ts.row_f(j));
                }
                self.push(&name, l, "kv_state", cmp::cmp(&a, &b), format!("{rem} open rows"));
                self.push(&name, l, "score_state", cmp::cmp(&c2, &d2), format!("{rem} open rows"));
            }
        }
        Ok(())
    }
}

/// The dumped mixes of a sublayer as a [rows][24] record (pre 4, post 4, comb 16).
fn mix_ref(fx: &Fix, l: usize, side: &str, pos: &[i64]) -> Result<Vec<f32>> {
    let pre = rows_at(&fx.d.layer(l, &format!("{side}_pre"))?, pos)?;
    let post = rows_at(&fx.d.layer(l, &format!("{side}_post"))?, pos)?;
    let comb = rows_at(&fx.d.layer(l, &format!("{side}_comb"))?, pos)?;
    let mut v = Vec::with_capacity(pos.len() * 24);
    for r in 0..pos.len() {
        v.extend_from_slice(&pre[r * 4..r * 4 + 4]);
        v.extend_from_slice(&post[r * 4..r * 4 + 4]);
        v.extend_from_slice(&comb[r * 16..r * 16 + 16]);
    }
    Ok(v)
}

/// The input stream of the layer after `l` (its `x_pre_engram` when it is an Engram layer), for `block_out`.
pub fn next_input(fx: &Fix, cfg: &crate::dsv41::config::Dsv41Config, l: usize) -> Result<Option<RefTensor>> {
    let n = l + 1;
    if n >= cfg.layers { return Ok(None); }
    let point = if cfg.engram_layers.contains(&n) { "x_pre_engram" } else { "x_in" };
    fx.d.layer_opt(n, point)
}

/// Load the layer, run it on every fixture, drop it.
pub fn run_layer(gate: &mut Gate, pack: &Pack, l: usize, fixes: &[Fix], states: &mut [FixState]) -> Result<f64> {
    let t0 = std::time::Instant::now();
    let w = load_layer(gate.f.g, pack, l)?;
    let load_s = t0.elapsed().as_secs_f64();
    for (fx, st) in fixes.iter().zip(states.iter_mut()) {
        let nx = next_input(fx, &pack.c.cfg, l)?;
        gate.layer(&w, fx, st, &pack.c.cfg, nx).with_context(|| format!("layer {l} on {}", fx.name))?;
    }
    crate::dsv41::weights::exl3::moe::clear_tables();
    drop(w);
    Ok(load_s)
}
