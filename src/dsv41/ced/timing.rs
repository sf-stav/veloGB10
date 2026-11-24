//! CE: the measured work-saved table on the slice (informative: one box, TP1, host launches, no graphs; not the TP
//! posture). Per fixture length P, every slice trunk layer runs a whole layer (attn_in, attention side, FFN side)
//! over a synthetic input stream on its rows, chunked at 2,048:
//! * every slice layer on all P rows (the exact prefill's cost of that layer class);
//! * the first decoder layer on its CED rows (compressor + index keys on every row, the rest on its cone);
//! * the decoder classes on EVERY decoder layer's own cone (layer 21's weights for the ratio-1 reusers, layer 24's
//!   for the candidate-using index sources), so the projected CED decoder is the measured sum of 20 cones.
//! Projection to the 40-layer trunk: each layer takes the time of its class's slice layer: SWA-only 0 / 1; the ratio-2
//! sources 2 / 14 (8 = layer 2's); the ratio-2 REUSERS 3-7, 9-13, 15-19 = layer 21 on every row (a reuser of either
//! ratio runs the same work: q path, window + <= 512 selected compressed keys, wo, MoE; no compressor, no indexer);
//! the decoder from the measured cones (ratio-1 reusers with 21's weights, index sources with candidates with 24's).

use super::chunk::{self, Cands};
use super::lane::{Lane, LayerLane};
use super::probe::{Fix, Gate};
use super::schedule::{self, ChunkRows, Mode, Schedule};
use crate::dsv41::config::{AttnClass, Dsv41Config};
use crate::dsv41::fwd::dev::Buf;
use crate::dsv41::fwd::layer::{bytemuck_u16, HC_DIM};
use crate::dsv41::fwd::weights::{load_layer, LayerW, Pack};
use anyhow::{Context, Result};
use std::collections::BTreeMap;

/// Rows of one timed run of one layer: query from q0, window from w0, KV source on every row when kv_all.
#[derive(Clone, Copy)]
struct Rows { q0: usize, w0: usize, kv_all: bool }

const CHUNK: usize = 2048;

impl<'a> Gate<'a> {
    /// Run layer `w` over `rows` of a P-row prompt (x / pre: the layer's input stream on every row), chunked;
    /// returns wall seconds (stream synchronised before and after).
    fn time_layer(&mut self, w: &LayerW, lane: &mut Lane, cands: &mut Cands, p: usize, r: Rows, x: &Buf, pre: &Buf) -> Result<f64> {
        let cfg: &Dsv41Config = self.f.cfg;
        let l = w.layer;
        if !lane.layers.contains_key(&l) { lane.layers.insert(l, LayerLane::new(self.f.g, cfg, l, CHUNK, p)?); }
        self.f.g.sync()?;
        let t0 = std::time::Instant::now();
        for (c0, c1) in schedule::chunks(0, p, CHUNK) {
            let clip = |a: usize| (a.max(c0), c1);
            let cr = ChunkRows { q: clip(r.q0), w: clip(r.w0), a: clip(if r.kv_all { 0 } else { r.w0 }), kv: if r.kv_all { Some(clip(0)) } else { None }, win_lo: 0 };
            let na = schedule::n_of(cr.a);
            if na == 0 { continue; }
            self.f.b.reset();
            let xa = x.ptr() + (cr.a.0 * HC_DIM * 2) as u64;
            let ain = chunk::attn_in_rows(&mut self.f, w, xa, pre.ptr() + (cr.a.0 * 16) as u64, 4, na)?;
            let o = chunk::attn_chunk(&mut self.f, w, lane, &cr, ain, c0, cands)?;
            let nq = schedule::n_of(o.q);
            if nq > 0 {
                let xq = xa + ((o.q.0 - cr.a.0) * HC_DIM * 2) as u64;
                let rec_a = self.f.mix(w, true, xq, nq)?;
                chunk::ffn_rows(&mut self.f, w, xq, o.out, rec_a, nq, true, None)?;
            }
            self.f.g.sync()?;
        }
        Ok(t0.elapsed().as_secs_f64())
    }

    /// The table for one fixture length (inputs: the layers' synthetic streams from the fixture's dumps).
    pub fn timing(&mut self, pack: &Pack, fx: &Fix, trunk: &[usize]) -> Result<Vec<String>> {
        let cfg = self.f.cfg;
        let p = fx.p;
        let ced = Schedule::new(cfg, p, Mode::Ced)?;
        let d = cfg.first_decoder_layer().context("no decoder layer")?;
        let mut exact: BTreeMap<usize, f64> = BTreeMap::new();
        let mut cone: BTreeMap<usize, f64> = BTreeMap::new(); // decoder layer M -> its cone measured with its class's slice weights
        let ws: Vec<LayerW> = trunk.iter().map(|&l| load_layer(self.f.g, pack, l)).collect::<Result<_>>()?;
        let mut streams = BTreeMap::new();
        for &l in trunk {
            let (xs, ps) = super::chain::synth_stream(fx, l)?;
            let x = self.f.g.buf(xs.len() * 2)?;
            self.f.g.h2d(x.ptr(), bytemuck_u16(&xs))?;
            let pre = self.f.g.buf(ps.len() * 4)?;
            self.f.g.h2d(pre.ptr(), crate::dsv41::fwd::layer::bytemuck_f32(&ps))?;
            streams.insert(l, (x, pre));
        }
        let wl = |l: usize| ws.iter().find(|w| w.layer == l);
        // exact: every slice layer on every row (one lane: the reusers read 20's caches and selections)
        let (mut lane, mut cands) = (Lane::default(), Cands::default());
        for (i, &l) in trunk.iter().enumerate() {
            let (x, pre) = &streams[&l];
            let t = self.time_layer(&ws[i], &mut lane, &mut cands, p, Rows { q0: 0, w0: 0, kv_all: cfg.kv_sources.contains(&l) && cfg.ratio(l) > 0 }, x, pre)?;
            exact.insert(l, t);
        }
        // CED: layer 20 on its CED rows, then every decoder layer's cone with its class's weights
        if let (Some(w20), Some(w21), Some(w24)) = (wl(d), wl(d + 1), wl(24)) {
            let (mut lane, mut cands) = (Lane::default(), Cands::default());
            let r20 = ced.layers[d];
            let (x, pre) = &streams[&d];
            cone.insert(d, self.time_layer(w20, &mut lane, &mut cands, p, Rows { q0: r20.q0, w0: r20.w0, kv_all: true }, x, pre)?);
            for m in d + 1..cfg.layers {
                let rm = ced.layers[m];
                let (w, src) = match cfg.attn_class(m) {
                    AttnClass::Compressed { index_source: true, uses_candidates: true, .. } => (w24, 24),
                    _ => (w21, d + 1),
                };
                // a fresh lane per run (each layer writes its own ring); 20's stores in `cands` stay
                let mut l2 = Lane::default();
                l2.layers.insert(d, lane.layers.remove(&d).context("layer 20 state")?);
                let (x, pre) = &streams[&src];
                let t = self.time_layer(w, &mut l2, &mut cands, p, Rows { q0: rm.q0, w0: rm.w0, kv_all: false }, x, pre)?;
                lane.layers.insert(d, l2.layers.remove(&d).context("layer 20 state")?);
                cone.insert(m, t);
            }
        }
        // projection to the trunk
        let class_of = |l: usize| -> usize {
            match cfg.attn_class(l) {
                AttnClass::SwaOnly => if trunk.contains(&l) { l } else { 0 },
                AttnClass::Compressed { ratio: 2, kv_source: true, .. } => if trunk.contains(&l) { l } else if l < 14 { 2 } else { 14 },
                AttnClass::Compressed { ratio: 2, .. } => d + 1,
                AttnClass::Compressed { candidate_source: true, .. } => d,
                AttnClass::Compressed { index_source: true, uses_candidates: true, .. } => 24,
                _ => d + 1,
            }
        };
        let tex: f64 = (0..cfg.layers).map(|l| exact.get(&class_of(l)).copied().unwrap_or(f64::NAN)).sum();
        let tced: f64 = (0..cfg.layers).map(|l| if l < d { exact.get(&class_of(l)).copied().unwrap_or(f64::NAN) } else { cone.get(&l).copied().unwrap_or(f64::NAN) }).sum();
        let enc: f64 = (0..d).map(|l| exact.get(&class_of(l)).copied().unwrap_or(f64::NAN)).sum();
        let mut out = vec![format!("measured (slice, one box, TP1, host launches, chunk {CHUNK}): P = {p}")];
        out.push(format!("  per layer, every row: {}", exact.iter().map(|(l, t)| format!("L{l} {:.0} ms", t * 1e3)).collect::<Vec<_>>().join(", ")));
        out.push(format!("  decoder cones (CED rows): {}", cone.iter().map(|(l, t)| format!("L{l} {:.0}", t * 1e3)).collect::<Vec<_>>().join(", ")));
        out.push(format!("  projected 40-layer trunk: exact {:.2} s ({:.0} tok/s), CED {:.2} s ({:.0} tok/s) = {:.3} of exact; encoder {:.2} s, decoder exact {:.2} s vs CED {:.2} s",
                         tex, p as f64 / tex, tced, p as f64 / tced, tced / tex, enc, tex - enc, tced - enc));
        crate::dsv41::weights::exl3::moe::clear_tables();
        Ok(out)
    }
}
