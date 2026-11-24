//! SL: the slice's step time at 1 / 6 / 24 rows (`--dsv41-kernel-timing 1,6,24`), one box, decode / verify
//! shape: per layer, the rows are the LAST n positions of a fixture prompt (agent_4k when given), the lane's
//! window ring and the KV sources' caches are the ones the gate built from that prompt, attention runs on the
//! decode entry (`d41_attn_decode`), and every piece of the layer runs back to back on the stream:
//! [Engram] -> attn mix -> hc_pre + norm -> q path -> window write -> [indexer] -> attention -> wo_a / wo_b
//! -> hc_post -> ffn mix -> hc_pre + norm -> router -> routed + shared experts -> hc_post.
//! Host launches, no CUDA graph, no PDL (the served path will graph it): INFORMATIVE, not the TP posture.
//! Not in the timed step: the compressor's decode update (KC measured ~12 us per call) and layer 20's
//! candidate pass (a no-op below 16,384 positions: the 4K fixture never binds it).

use super::dev::{dp, Ev};
use super::gate::{Fix, FixState, Gate};
use super::layer::HC_DIM;
use super::weights::{load_layer, Pack};
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::gpu::attn;
use anyhow::Result;

pub fn run(gt: &mut Gate, pack: &Pack, layers: &[usize], fixes: &[Fix], states: &mut [FixState], spec: &str) -> Result<Vec<String>> {
    let widths: Vec<usize> = spec.split(',').filter_map(|s| s.trim().parse().ok()).filter(|&n| n > 0).collect();
    let widths = if widths.is_empty() { vec![1, 6, 24] } else { widths };
    let fi = fixes.iter().position(|f| f.name.contains("4k")).unwrap_or(0);
    let fx = &fixes[fi];
    let cfg = pack.c.cfg.clone();
    let p = fx.p;
    let mut lines = vec![format!("SL step time (one box, host launches, no graph; rows = the last n positions of {} at P {p}; ms per layer, median of 10):", fx.name)];
    lines.push(format!("{:<6} {:<40} {}", "layer", "class", widths.iter().map(|n| format!("{:>9}", format!("{n} rows"))).collect::<String>()));
    let mut totals = vec![0f64; widths.len()];
    gt.f.checks = false;
    for &l in layers.iter().filter(|&&l| l < cfg.layers) {
        let w = load_layer(gt.f.g, pack, l)?;
        let ain = fx.d.layer(l, "attn_in")?;
        let ratio = cfg.ratio(l);
        let compressed = ratio > 0;
        // the lane's window: the prompt's rows, so the ring holds its last 134 positions
        gt.f.b.reset();
        let dall = {
            let b: Vec<u16> = ain.f.iter().map(|&x| super::cmp::f32_to_bf16_bits(x)).collect();
            gt.f.g.up(&b)?
        };
        let mut win = gt.f.window(&w, dp(&dall), 0, p, compressed)?;
        let src = cfg.kv_source_of(l).and_then(|s| states[fi].src.get(&s));
        let mut row = format!("{:<6} {:<40}", l, format!("{:?}", cfg.attn_class(l)).chars().take(40).collect::<String>());
        for (wi, &n) in widths.iter().enumerate() {
            let pos: Vec<i32> = ((p - n) as i32..p as i32).collect();
            let dpos = gt.f.g.up(&pos)?;
            let x_in = fx.d.layer(l, "x_in")?;
            let last = &x_in.f[x_in.f.len().saturating_sub(n * HC_DIM)..];
            let xs: Vec<f32> = (0..n).flat_map(|i| last[(i % (last.len() / HC_DIM)) * HC_DIM..(i % (last.len() / HC_DIM) + 1) * HC_DIM].to_vec()).collect();
            let dx = gt.f.g.up(&xs.iter().map(|&v| super::cmp::f32_to_bf16_bits(v)).collect::<Vec<u16>>())?;
            let pre: Vec<f32> = (0..n).flat_map(|_| [1.0f32, 0.0, 0.0, 0.0]).collect();
            let dpre = gt.f.g.up(&pre)?;
            let (pp, gq) = (attn::DecodeScratch::bytes(n, cfg.heads, attn::MAX_SPLITS), cfg.heads.div_ceil(attn::HG));
            let (part, cnt) = (gt.f.g.buf(pp.0)?, gt.f.g.buf(pp.1)?);
            let ds = attn::DecodeScratch { part: part.p(), cnt: cnt.p(), rows: n, groups: gq, nsplit: attn::MAX_SPLITS };
            let mut step = |gt: &mut Gate| -> Result<()> {
                gt.f.b.reset();
                let x = dp(&dx);
                let rec_a = gt.f.mix(&w, true, x, n)?;
                let ain_n = gt.f.hc_pre_norm(x, dp(&dpre), 4, w.param(C::AttnNorm)?, n)?;
                let (qr, q) = gt.f.q_path(&w, ain_n, dp(&dpos), n, compressed)?;
                gt.f.window_into(&w, ain_n, (p - n) as i64, n, compressed, &mut win)?;
                let sel = match (compressed, src) {
                    (true, Some(s)) if cfg.index_source_of(l) == Some(l) => { let (sel, _, _) = gt.f.indexer(&w, qr, ain_n, dp(&dpos), n, s, false, None)?; Some((sel, 512, 512.min(s.n_groups.max(1)))) }
                    (true, Some(s)) => { let b = gt.f.b.take(n * 512 * 4)?; gt.f.g.memset(b, n * 512 * 4, 0xff)?; Some((b, 512, 512.min(s.n_groups.max(1)))) }
                    _ => None,
                };
                let o = gt.f.attention(&w, q, dp(&dpos), n, &win, src, sel, Some(&ds))?;
                let ao = gt.f.wo(&w, o, n)?;
                let xm = gt.f.hc_post(ao, x, rec_a, n)?;
                let rec_f = gt.f.mix(&w, false, xm, n)?;
                let fin = gt.f.hc_pre_norm(xm, rec_a, 24, w.param(C::FfnNorm)?, n)?;
                let (ids, wts) = gt.f.router(&w, fin, n)?;
                let fo = gt.f.ffn(&w, fin, n, ids, wts)?;
                gt.f.hc_post(fo, xm, rec_f, n)?;
                Ok(())
            };
            for _ in 0..2 { step(gt)?; }
            gt.f.g.sync()?;
            let mut ms = Vec::new();
            for _ in 0..10 {
                let (a, b) = (Ev::new()?, Ev::new()?);
                a.rec(gt.f.g)?;
                step(gt)?;
                b.rec(gt.f.g)?;
                ms.push(b.ms_since(&a)? as f64);
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ms[ms.len() / 2];
            totals[wi] += med;
            row.push_str(&format!("{med:>9.3}"));
        }
        lines.push(row);
        crate::dsv41::weights::exl3::moe::clear_tables();
        drop(w);
    }
    lines.push(format!("{:<6} {:<40}{}", "sum", "(the slice's trunk layers)", totals.iter().map(|t| format!("{t:>9.3}")).collect::<String>()));
    gt.f.checks = true;
    Ok(lines)
}
