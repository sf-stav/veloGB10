//! CE: the row-local pieces of a layer (attention mix, attn_in, x_mid, FFN mix, ffn_in, router, routed + shared
//! experts, block_out) on row SUBSETS, fed the DUMPED inputs of the fixture's dumped rows.
//!
//! CED computes a decoder layer only on its cone rows, and a chunk tail can have any row count, so every
//! row-local kernel must give a row the same bits whatever other rows share its call. Calls compared bit for
//! bit, per row: (a) every dumped row in one call, (b) the dumped rows inside the CED cone in one call, (c) the
//! same cone rows in pieces of 1, 5, 33, 100 rows and the rest. Then (b) against REFb with SL's tolerances
//! (mixes, attn_in, ffn_in, the router's sets, ffn_out with OUR routing: the served path).

use super::chunk;
use super::probe::{bf16_bits, Fix, Gate};
use super::schedule::{Mode, Schedule};
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::cpu::mhc::refdump::RefTensor;
use crate::dsv41::fwd::cmp::{self, Cmp};
use crate::dsv41::fwd::dev::dp;
use crate::dsv41::fwd::layer::HC_DIM;
use crate::dsv41::fwd::weights::LayerW;
use anyhow::{Context, Result};

fn rows_at(t: &RefTensor, pos: &[usize]) -> Result<Vec<f32>> {
    let s = t.stride();
    let mut v = Vec::with_capacity(pos.len() * s);
    for &p in pos { let i = t.index_of(p as i64).with_context(|| format!("{}: no row {p}", t.name))?; v.extend_from_slice(t.row_f(i)); }
    Ok(v)
}

/// Per row: the bytes of every piece (downloaded), for bit comparisons.
struct Pieces { per_row: Vec<Vec<u8>>, ffn_out: Vec<f32>, ffn_in: Vec<f32>, rec_a: Vec<f32>, rec_f: Vec<f32>, ain: Vec<f32>, ids: Vec<i32> }

impl<'a> Gate<'a> {
    fn pieces(&mut self, w: &LayerW, fx: &Fix, pos: &[usize]) -> Result<Pieces> {
        let (l, d) = (w.layer, self.f.cfg.hidden);
        let k = self.f.cfg.moe_of(l).1;
        let n = pos.len();
        self.f.g.sync()?;
        self.f.b.reset();
        let x = self.f.g.up(&bf16_bits(&rows_at(&fx.d.layer(l, "x_in")?, pos)?))?;
        let ao = self.f.g.up(&bf16_bits(&rows_at(&fx.d.layer(l, "attn_out")?, pos)?))?;
        let pre = self.f.g.up(&rows_at(&fx.d.layer(l, "pre_mix_in")?, pos)?)?;
        let rec_a = self.f.mix(w, true, dp(&x), n)?;
        let ain = chunk::attn_in_rows(&mut self.f, w, dp(&x), dp(&pre), 4, n)?;
        let o = chunk::ffn_rows(&mut self.f, w, dp(&x), dp(&ao), rec_a, n, true, None)?;
        let g = self.f.g;
        let dl = |p: u64, b: usize| g.down::<u8>(p, b);
        let bufs = [(rec_a, 96), (ain, d * 2), (o.x_mid, HC_DIM * 2), (o.rec_f, 96), (o.ffn_in, d * 2), (o.ids, k * 4), (o.ffn_out, d * 2), (o.block_out, HC_DIM * 2)];
        let all: Vec<Vec<u8>> = bufs.iter().map(|&(p, b)| dl(p, n * b)).collect::<Result<_>>()?;
        let per_row = (0..n).map(|r| bufs.iter().zip(&all).flat_map(|(&(_, b), v)| v[r * b..(r + 1) * b].iter().copied()).collect()).collect();
        let f32s = |v: &[u8]| v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
        let bf = |v: &[u8]| v.chunks_exact(2).map(|c| cmp::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect::<Vec<f32>>();
        Ok(Pieces { per_row, rec_a: f32s(&all[0]), ain: bf(&all[1]), rec_f: f32s(&all[3]), ffn_in: bf(&all[4]),
                    ids: all[5].chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), ffn_out: bf(&all[6]) })
    }

    /// The row-local checks of one layer on one fixture.
    pub fn rows_gate(&mut self, w: &LayerW, fx: &Fix) -> Result<()> {
        let (l, cfg) = (w.layer, self.f.cfg);
        let all: Vec<usize> = super::probe::positions(&fx.d.layer(l, "x_in")?);
        let q0 = Schedule::new(cfg, fx.p, Mode::Ced)?.layers[l].q0;
        let cone: Vec<usize> = all.iter().copied().filter(|&r| r >= q0).collect();
        if cone.is_empty() { return Ok(()); }
        let a = self.pieces(w, fx, &all)?;
        let b = self.pieces(w, fx, &cone)?;
        // (c) the cone rows in pieces
        let mut c_rows: Vec<Vec<u8>> = Vec::new();
        let mut i = 0;
        for want in [1usize, 5, 33, 100, usize::MAX] {
            if i >= cone.len() { break; }
            let n = want.min(cone.len() - i);
            c_rows.extend(self.pieces(w, fx, &cone[i..i + n])?.per_row);
            i += n;
        }
        let off = all.len() - cone.len();
        let bad_ab = (0..cone.len()).filter(|&r| a.per_row[off + r] != b.per_row[r]).count();
        let bad_bc = (0..cone.len()).filter(|&r| b.per_row[r] != c_rows[r]).count();
        self.push(&fx.name, l, "bitwise rows: cone vs all", Some(bad_ab == 0),
                  format!("{} cone rows (of {} dumped) in one call vs all dumped rows in one call: {bad_ab} rows differ (mixes, attn_in, x_mid, ffn_in, route, ffn_out, block_out)", cone.len(), all.len()));
        self.push(&fx.name, l, "bitwise rows: pieces 1/5/33/100", Some(bad_bc == 0), format!("the cone rows in pieces of 1, 5, 33, 100 rows and the rest vs one call: {bad_bc} rows differ"));
        // (b) against REFb
        let refm = |side: &str| -> Result<Vec<f32>> {
            let (p4, q4, c16) = (rows_at(&fx.d.layer(l, &format!("{side}_pre"))?, &cone)?, rows_at(&fx.d.layer(l, &format!("{side}_post"))?, &cone)?,
                                 rows_at(&fx.d.layer(l, &format!("{side}_comb"))?, &cone)?);
            Ok((0..cone.len()).flat_map(|r| p4[r * 4..r * 4 + 4].iter().chain(&q4[r * 4..r * 4 + 4]).chain(&c16[r * 16..r * 16 + 16]).copied().collect::<Vec<_>>()).collect())
        };
        let x_mid_ok = fx.d.layer(l, "x_mid").is_ok();
        let pts: Vec<(&'static str, Cmp)> = vec![
            ("attn_mix", cmp::cmp(&b.rec_a, &refm("attn")?)),
            ("attn_in", cmp::cmp(&b.ain, &rows_at(&fx.d.layer(l, "attn_in")?, &cone)?)),
            ("ffn_in", cmp::cmp(&b.ffn_in, &rows_at(&fx.d.layer(l, "ffn_in")?, &cone)?)),
            ("ffn_out", cmp::cmp(&b.ffn_out, &rows_at(&fx.d.layer(l, "ffn_out")?, &cone)?)),
        ];
        let _ = x_mid_ok;
        for (pt, c) in pts { self.push_cmp(&fx.name, l, pt, c, format!("{} cone rows (own routing; ffn_in from our x_mid, ffn mix)", cone.len())); }
        let fm = cmp::cmp(&b.rec_f, &refm("ffn")?);
        self.push_cmp(&fx.name, l, "ffn_mix", fm, format!("{} cone rows (from our x_mid)", cone.len()));
        let k = cfg.moe_of(l).1;
        let ridx = fx.d.layer(l, "router_idx")?;
        let mut same = 0;
        for (r, &p) in cone.iter().enumerate() {
            let i = ridx.index_of(p as i64).context("router row")?;
            let mut x: Vec<i64> = b.ids[r * k..(r + 1) * k].iter().map(|&v| v as i64).collect();
            let mut y = ridx.row_i(i).to_vec();
            x.sort_unstable(); y.sort_unstable();
            if x == y { same += 1; }
        }
        let rc = Cmp { rel: 0.0, same: same as f64 / cone.len() as f64, max_abs: 0.0, n: cone.len() };
        self.push_cmp(&fx.name, l, "router", rc, format!("{same}/{} cone rows same top-{k} set (on our ffn_in)", cone.len()));
        // the FFN recipes side by side (information): which input moves ffn_out against REFb
        {
            self.f.g.sync()?;
            self.f.b.reset();
            let n = cone.len();
            let dfin = self.f.g.up(&bf16_bits(&rows_at(&fx.d.layer(l, "ffn_in")?, &cone)?))?;
            let ridx: Vec<i32> = cone.iter().map(|&p| fx.d.layer(l, "router_idx").ok().and_then(|t| t.index_of(p as i64).map(|i| t.row_i(i).iter().map(|&x| x as i32).collect::<Vec<_>>())).unwrap_or_default()).flatten().collect();
            let rw = rows_at(&fx.d.layer(l, "router_w")?, &cone)?;
            let (di, dw) = (self.f.g.up(&ridx)?, self.f.g.up(&rw)?);
            let fref = rows_at(&fx.d.layer(l, "ffn_out")?, &cone)?;
            let a = self.f.ffn(w, dp(&dfin), n, dp(&di), dp(&dw))?;
            let ra = cmp::cmp(&self.f.g.down_bf16(a, n * cfg.hidden)?, &fref);
            let (oi, ow) = self.f.router(w, dp(&dfin), n)?;
            let b2 = self.f.ffn(w, dp(&dfin), n, oi, ow)?;
            let rb = cmp::cmp(&self.f.g.down_bf16(b2, n * cfg.hidden)?, &fref);
            self.push(&fx.name, l, "ffn_out recipes (info)", None, format!("dumped ffn_in + dumped route {:.3e}; dumped ffn_in + own route {:.3e}; ours (own ffn_in + route) {:.3e}", ra.rel, rb.rel,
                                                                      cmp::cmp(&b.ffn_out, &fref).rel));
        }
        let _ = C::FfnNorm;
        Ok(())
    }
}
