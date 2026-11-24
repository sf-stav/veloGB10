//! CE: consecutive slice layers CHAINED through our own forward, chunk-major (every layer of a chunk, then the
//! next chunk: the order a served prefill runs), for every variant of `probe::variants` (CED, exact, chunkings,
//! encoder-only and full prefix resumes with the whole lane snapshotted, poisoned and restored). Compared bit for
//! bit with `exact@one`: every computed row of every chain layer (block_out and the next layer's pre-mix) and every
//! layer's decode state. Chains are ours-vs-ours only (REFb §6.2: a chained comparison with the reference is noise).
//!
//! * chain `0-1-2`: the real model from the token ids: embedding gather, identity pre-mix, Engram at layer 1 with
//!   the row ids hashed by ENGa's `EngramHasher` per chunk (resumes rebuild the hasher state from the last token ids,
//!   no n-gram state is kept) and read through EIO's prefill channel. Info against REFb: layer 0's attn_in and the
//!   Engram row ids (exact).
//! * chain `20-21`: the first two decoder layers; layer 20's input stream on every row is SYNTHETIC (the nearest
//!   dumped row of `L20.x_in` / `pre_mix_in`, scaled by a small per-position factor so no two rows are equal):
//!   the slice has no layer 19, and only the bit identity of the variants is gated here.

use super::chunk::{self, Cands};
use super::lane::{Lane, LaneSnap, LayerLane};
use super::probe::{fnv, Fix, Gate, Variant, ONE_MAX};
use super::schedule::{n_of, Mode, Schedule, StateKind};
use crate::dsv41::serve::prefix::{PrefixPool, MIN_KEEP};
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::engram::hash::{EngramHasher, NgramHashState};
use crate::dsv41::engram::layout::{EngramConfig, EngramLayout};
use crate::dsv41::engram::prefetch::{EngramIo, EngramIoConfig};
use crate::dsv41::fwd::cmp;
use crate::dsv41::fwd::dev::Buf;
use crate::dsv41::fwd::layer::{bytemuck_u16, HC_DIM};
use crate::dsv41::fwd::weights::{load_layer, LayerW, Pack};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

/// Embedding rows of `ids` (BF16 table) read from the pack's file, expanded to the hc streams: [n][hc][dim] bf16.
fn embed_stream(pack: &Pack, ids: &[u32], hc: usize) -> Result<Vec<u16>> {
    let c = &pack.c;
    let e = c.entries.iter().find(|e| e.class == C::Embed).context("pack has no embedding")?;
    ensure!(e.dtype == "BF16" && e.shape.len() == 2, "embedding {} {:?}: BF16 [vocab, dim] expected", e.dtype, e.shape);
    let dim = e.shape[1] as usize;
    let f = std::fs::File::open(c.dir.join(&c.files[e.file as usize]))?;
    let mut out = Vec::with_capacity(ids.len() * hc * dim);
    let mut row = vec![0u8; dim * 2];
    for &id in ids {
        ensure!((id as u64) < e.shape[0], "token {id} outside the embedding");
        f.read_exact_at(&mut row, e.offset + id as u64 * dim as u64 * 2)?;
        let r: Vec<u16> = row.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        for _ in 0..hc { out.extend_from_slice(&r); }
    }
    Ok(out)
}

/// The synthetic input of a chain that starts at layer `l`: the nearest dumped row at or below each position.
pub(crate) fn synth_stream(fx: &Fix, l: usize) -> Result<(Vec<u16>, Vec<f32>)> {
    let x = fx.d.layer(l, "x_in")?;
    let pm = fx.d.layer(l, "pre_mix_in")?;
    let pos = x.positions();
    let s = x.stride();
    let mut xs = Vec::with_capacity(fx.p * s);
    let mut ps = Vec::with_capacity(fx.p * 4);
    let mut j = 0usize;
    for t in 0..fx.p {
        while j + 1 < pos.len() && pos[j + 1] as usize <= t { j += 1; }
        let scale = 1.0 + ((t as u64 * 2654435761) % 13) as f32 * (1.0 / 1024.0);
        xs.extend(x.row_f(j).iter().map(|&v| cmp::f32_to_bf16_bits(f32_round_bf16(v * scale))));
        ps.extend_from_slice(pm.row_f(j));
    }
    Ok((xs, ps))
}
fn f32_round_bf16(v: f32) -> f32 {
    let b = v.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// One chain run's results: per layer, row -> hash of (block_out, ffn pre-mix), and the decode states.
#[derive(Default)]
struct ChainOut { rows: BTreeMap<usize, BTreeMap<usize, u64>>, state: BTreeMap<usize, Vec<(&'static str, Vec<u8>)>>, ain0: BTreeMap<usize, Vec<u16>>,
                  /// the prefix-pool round trips of a resume variant: (check, pass, text)
                  pool: Vec<(String, bool, String)> }

struct Ctx<'p> { ws: Vec<LayerW>, x0: Buf, pre0: Buf, pre0_stride: usize, io: Option<EngramIo>, io_max: usize, hasher: Option<EngramHasher>, ids: Vec<u32>, keep0: Vec<usize>, _p: std::marker::PhantomData<&'p ()> }

impl<'a> Gate<'a> {
    /// The chain `layers` on one fixture. `embed`: the input is the embedding of the prompt (layer 0 first);
    /// else synthetic from the first layer's dumps.
    pub fn chain_gate(&mut self, pack: &Pack, fx: &Fix, layers: &[usize], embed: bool, chunks: &[usize], engram_dir: &Path) -> Result<()> {
        let cfg = self.f.cfg;
        let p = fx.p;
        let tag = format!("chain {}", layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join("-"));
        for w in layers.windows(2) { ensure!(w[1] == w[0] + 1, "{tag}: layers must be consecutive"); }
        let t0 = std::time::Instant::now();
        let ws: Vec<LayerW> = layers.iter().map(|&l| load_layer(self.f.g, pack, l)).collect::<Result<_>>()?;
        let ids: Vec<u32> = fx.d.global("prompt_ids")?.i.iter().map(|&x| x as u32).collect();
        ensure!(ids.len() == p, "prompt_ids holds {} of {p}", ids.len());
        let hc = cfg.hc;
        let (xs, ps, pstride) = if embed {
            let mut pre = vec![0f32; p * 4];
            for r in 0..p { pre[r * 4] = 1.0; }
            (embed_stream(pack, &ids, hc)?, pre, 4)
        } else { let (x, pm) = synth_stream(fx, layers[0])?; (x, pm, 4) };
        let x0 = self.f.g.buf(xs.len() * 2)?;
        self.f.g.h2d(x0.ptr(), bytemuck_u16(&xs))?;
        let pre0 = self.f.g.buf(ps.len() * 4)?;
        self.f.g.h2d(pre0.ptr(), crate::dsv41::fwd::layer::bytemuck_f32(&ps))?;
        // Engram: the hasher over the token ids (checked against REFb's ids) and the EIO prefill channel
        let max_rows = p.min(ONE_MAX).max(*chunks.iter().max().unwrap_or(&1)).min(p);
        let (mut io, mut hasher) = (None, None);
        if layers.iter().any(|l| cfg.engram_layers.contains(l)) {
            let h = EngramHasher::from_model_dir(&pack.c.dir)?.context("no Engram config")?;
            let per = h.ids_per_token();
            let mut st = NgramHashState::new();
            let mut got = vec![0u32; p * per];
            for (c0, c1) in super::schedule::chunks(0, p, 1001) { h.hash_chunk(&mut st, &ids[c0..c1], None, &mut got[c0 * per..c1 * per])?; }
            let refids: Vec<u32> = fx.d.global("engram_hash_ids")?.i.iter().map(|&x| x as u32).collect();
            let bad = got.iter().zip(&refids).filter(|(a, b)| a != b).count() + got.len().abs_diff(refids.len());
            self.push(&fx.name, layers[0], format!("{tag}: engram row ids"), Some(bad == 0),
                      format!("ENGa's hasher in 1001-token chunks over the {p} prompt ids vs REFb's G.engram_hash_ids [{p} x {per}]: {bad} differ"));
            let ec = EngramConfig::from_model_dir(&pack.c.dir)?.context("config has no Engram layers")?;
            let layout = Arc::new(EngramLayout::new(ec)?);
            let mut c = EngramIoConfig::new(engram_dir.to_path_buf(), (0..layout.n_heads()).collect());
            c.prefill_max_tok = max_rows;
            c.decode_max_tok = 24;
            c.cache_mib = 1024;
            io = Some(EngramIo::open(&self.f.g.dev, layout, c)?);
            hasher = Some(h);
        }
        let keep0 = super::probe::positions(&fx.d.layer(layers[0], "x_in")?);
        let mut ctx = Ctx { ws, x0, pre0, pre0_stride: pstride, io, io_max: max_rows, hasher, ids, keep0, _p: std::marker::PhantomData };
        let mut vars = super::probe::variants(cfg, p, chunks)?;
        vars.retain(|v| v.pad);
        let mut outs = Vec::new();
        for v in vars.iter_mut() { outs.push(self.chain_variant(&mut ctx, v, layers, max_rows).with_context(|| format!("{tag} variant {}", v.name))?); }
        // ---- bit identity with exact@one
        let r0 = &outs[0];
        let ced = Schedule::new(cfg, p, Mode::Ced)?;
        for (vi, o) in outs.iter().enumerate().skip(1) {
            if !vars[vi].pad { continue; }
            let mut parts = Vec::new();
            let mut ok = true;
            for &l in layers {
                let (a, b) = (&o.rows[&l], &r0.rows[&l]);
                // a replay: decoder layer 20 + j is exact from P - keep + 127 (j + 1) on (the window truncation adds up)
                let bound = match (vars[vi].replay, cfg.first_decoder_layer()) {
                    (Some(k), Some(d)) if l >= d => super::probe::replay_exact_from(p, k, l - d + 1, cfg.window), _ => 0 };
                let common = a.keys().filter(|k| **k >= bound && b.contains_key(k)).count();
                let diff = a.iter().filter(|(k, h)| **k >= bound && b.get(k).map_or(false, |x| x != *h)).count();
                let approx = a.iter().filter(|(k, h)| **k < bound && b.get(k).map_or(false, |x| x != *h)).count();
                let missing = if vars[vi].replay.is_some() { 0 } else { (ced.layers[l].q0..p).filter(|r| !a.contains_key(r)).count() };
                let sd: Vec<&str> = o.state[&l].iter().zip(&r0.state[&l]).filter(|(x, y)| x.1 != y.1).map(|(x, _)| x.0).collect();
                ok &= diff == 0 && missing == 0 && sd.is_empty() && common > 0;
                let rp = if bound > 0 { format!(" ({approx} approximate rows below {bound} differ)") } else { String::new() };
                parts.push(format!("L{l}: {common} rows, {diff} differ{rp}, {missing} missing, state {}", if sd.is_empty() { "equal".to_string() } else { format!("DIFFERS {sd:?}") }));
            }
            let gated = vars[vi].replay.map_or(true, |k| k >= 2541);
            self.push(&fx.name, layers[0], format!("{tag} bitwise {}", vars[vi].name), if gated { Some(ok) } else { None }, parts.join("; "));
            for (what, ok, text) in &o.pool { self.push(&fx.name, layers[0], format!("{tag} {}: {what}", vars[vi].name), Some(*ok), text.clone()); }
        }
        // ---- info: the chain's first attn_in against REFb on the dumped rows (embedding path + identity pre-mix)
        if embed {
            let aref = fx.d.layer(layers[0], "attn_in")?;
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for (pos, row) in &r0.ain0 { a.extend(row.iter().map(|&x| cmp::bf16_to_f32(x))); b.extend_from_slice(aref.row_f(aref.index_of(*pos as i64).context("row")?)); }
            self.push_cmp(&fx.name, layers[0], "attn_in", cmp::cmp(&a, &b), format!("{tag}: from the embedding gather + identity pre-mix, {} dumped rows", r0.ain0.len()));
        }
        self.push(&fx.name, layers[0], format!("{tag} time"), None, format!("{} variants in {:.1} s", vars.len(), t0.elapsed().as_secs_f64()));
        crate::dsv41::weights::exl3::moe::clear_tables();
        Ok(())
    }

    fn chain_variant(&mut self, ctx: &mut Ctx, var: &mut Variant, layers: &[usize], max_rows: usize) -> Result<ChainOut> {
        let (cfg, d) = (self.f.cfg, self.f.cfg.hidden);
        let p = var.phases.last().context("no phase")?.sched.p;
        let mut lane = Lane::default();
        for &l in layers { lane.layers.insert(l, LayerLane::new(self.f.g, cfg, l, max_rows, p)?); }
        let mut cands = Cands::default();
        let mut out = ChainOut::default();
        for &l in layers { out.rows.insert(l, BTreeMap::new()); }
        // ping-pong buffers for a layer's outputs (block_out stream, ffn mix records)
        let bx = [self.f.g.buf(max_rows * HC_DIM * 2)?, self.f.g.buf(max_rows * HC_DIM * 2)?];
        let bp = [self.f.g.buf(max_rows * 96)?, self.f.g.buf(max_rows * 96)?];
        let mut hstate = NgramHashState::new();
        for ph in 0..var.phases.len() {
            let sched = var.phases[ph].sched.clone();
            if let (Some(h), true) = (&ctx.hasher, ph > 0) {
                // a resume: the n-gram state comes from the token ids alone
                let k = sched.from;
                let tail = &ctx.ids[k.saturating_sub(h.layout().cfg.max_ngram_size - 1)..k];
                hstate = h.state_at(k as u64, tail, None)?;
            }
            for &(c0, c1) in &var.phases[ph].chunks.clone() {
                let mut ids_chunk: Vec<u32> = Vec::new();
                if let Some(h) = &ctx.hasher {
                    ids_chunk = vec![0u32; (c1 - c0) * h.ids_per_token()];
                    h.hash_chunk(&mut hstate, &ctx.ids[c0..c1], None, &mut ids_chunk)?;
                }
                // (row0, slot) of the previous layer's outputs
                let mut prev: Option<(usize, usize)> = None;
                for (li, &l) in layers.iter().enumerate() {
                    let w = &ctx.ws[li];
                    let cr = sched.chunk(l, c0, c1);
                    let na = n_of(cr.a);
                    self.f.g.sync()?;
                    self.f.b.reset();
                    if na == 0 && cr.kv.map_or(true, |k| n_of(k) == 0) { prev = None; continue; }
                    let (mut x, pre, pstride) = match prev {
                        None => {
                            ensure!(li == 0, "layer {l}: needs rows {:?} the layer below did not compute", cr.a);
                            (ctx.x0.ptr() + (cr.a.0 * HC_DIM * 2) as u64, ctx.pre0.ptr() + (cr.a.0 * ctx.pre0_stride * 4) as u64, ctx.pre0_stride)
                        }
                        Some((row0, s)) => {
                            ensure!(cr.a.0 >= row0, "layer {l}: needs rows from {} but the layer below computed from {row0}", cr.a.0);
                            (bx[s].ptr() + ((cr.a.0 - row0) * HC_DIM * 2) as u64, bp[s].ptr() + ((cr.a.0 - row0) * 96) as u64, 24)
                        }
                    };
                    if let Some(li_e) = cfg.engram_layers.iter().position(|&e| e == l) {
                        let io = ctx.io.as_mut().context("Engram runtime")?;
                        let per = ids_chunk.len() / (c1 - c0);
                        let rows_ids = &ids_chunk[(cr.a.0 - c0) * per..(c1 - c0) * per];
                        let cols = per / 2;
                        let e = self.f.b.take(2 * ctx.io_max * cols * 256 * 2)?;
                        let ch = io.ch_prefill();
                        io.submit_prefill(rows_ids, None)?;
                        io.launch(ch, e, 0, crate::dsv41::gpu::engram::MODE_DECODE)?;
                        self.f.g.dev.synchronize()?;
                        let er = io.gpu_ctrl(ch).err;
                        ensure!(er == 0, "Engram step error {er}");
                        x = self.f.engram(w, x, e + (li_e * ctx.io_max * cols * 256 * 2) as u64, na)?;
                    }
                    let ain = chunk::attn_in_rows(&mut self.f, w, x, pre, pstride, na)?;
                    if li == 0 && ph == 0 {
                        let a = self.f.g.down::<u16>(ain, na * d)?;
                        for (i, r) in (cr.a.0..cr.a.1).enumerate() { if ctx.keep0.binary_search(&r).is_ok() { out.ain0.insert(r, a[i * d..(i + 1) * d].to_vec()); } }
                    }
                    let r = chunk::attn_chunk(&mut self.f, w, &mut lane, &cr, ain, c0, &mut cands)?;
                    let nq = n_of(r.q);
                    if nq == 0 { prev = None; continue; }
                    let xq = x + ((r.q.0 - cr.a.0) * HC_DIM * 2) as u64;
                    let rec_a = self.f.mix(w, true, xq, nq)?;
                    let o = chunk::ffn_rows(&mut self.f, w, xq, r.out, rec_a, nq, true, None)?;
                    let s = li % 2;
                    self.f.g.d2d(bx[s].ptr(), o.block_out, nq * HC_DIM * 2)?;
                    self.f.g.d2d(bp[s].ptr(), o.rec_f, nq * 96)?;
                    let (bo, rf) = (self.f.g.down::<u8>(bx[s].ptr(), nq * HC_DIM * 2)?, self.f.g.down::<u8>(bp[s].ptr(), nq * 96)?);
                    let m = out.rows.get_mut(&l).context("rows")?;
                    for i in 0..nq {
                        let mut h = fnv(&bo[i * HC_DIM * 2..(i + 1) * HC_DIM * 2]);
                        h ^= fnv(&rf[i * 96..(i + 1) * 96]).rotate_left(1);
                        m.insert(r.q.0 + i, h);
                    }
                    prev = Some((r.q.0, s));
                }
                cands.drop_chunk(c0);
            }
            if let Some(kind) = var.phases[ph].snap {
                // through the serve shell's prefix pool: keep the state, poison the lane, look the prompt up, restore
                let k = var.phases[ph].chunks.last().map_or(0, |c| c.1);
                lane.pos = k;
                let min_suffix = Schedule::new(cfg, p, Mode::Ced)?.min_suffix(kind);
                let snap = LaneSnap::take(self.f.g, &lane, kind)?;
                let mut pool = PrefixPool::new(64 << 30, MIN_KEEP);
                ensure!(pool.insert(&ctx.ids[..k], snap.into_serve(min_suffix)), "the pool did not keep the {kind:?} state at {k}");
                lane.poison(self.f.g, 0x5A)?;
                if kind == StateKind::Encoder && k + min_suffix - 1 <= p {
                    // a prompt that stops one row short of the decoder cone after the state must not resume from it
                    let short = pool.lookup(&ctx.ids[..k + min_suffix - 1]).map(|x| x.1);
                    out.pool.push((format!("pool refuses an encoder-only state at {k} for a {}-token prompt", k + min_suffix - 1), short.is_none(),
                                   format!("lookup -> {short:?} (min_suffix {min_suffix})")));
                }
                let (id, hit) = pool.lookup(&ctx.ids[..p]).with_context(|| format!("the pool found no state for the {p}-token prompt"))?;
                let s = pool.snapshot(id).context("pool entry")?;
                LaneSnap::of_serve(s).context("not a CE lane snapshot")?.restore(self.f.g, &mut lane)?;
                out.pool.push((format!("pool hit for the {p}-token prompt"), hit == k, format!("{kind:?} state, hit at {hit} (kept at {k}, min_suffix {min_suffix}, cap {})", p.saturating_sub(min_suffix.max(MIN_KEEP)))));
            }
        }
        self.f.g.sync()?;
        for &l in layers { out.state.insert(l, super::probe::state_of(self.f.g, lane.get(l)?, p)?); }
        Ok(out)
    }
}
