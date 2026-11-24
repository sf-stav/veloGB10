//! DSV4.1 CPU oracle (leg OF): the probe behind `--probe-dsv41-oracle --dsv41-oracle-part moe|engram`.
//!
//! `--model-dir` = a native checkpoint or a native slice (the FFN / Engram weights), `--dsv41-dump-ref` = a REFb
//! `prefill_exact` directory (`L{NN}.safetensors`, `G.safetensors`), optional `--layers 1,14` and
//! `--dsv41-oracle-rows N` (evenly spaced subset of the dumped rows; the oracle is a slow CPU implementation).
//! The Engram tables come from `--model-dir` when it has them, else from the slice's `dsv41_slice.source_dir`.

use super::engram_mod as eg;
use super::load::{RefFile, Store};
use super::{LayerMoe, Routing, DIM, TOPK};
use super::quant::*;
use anyhow::{bail, ensure, Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { n += ((x - y) as f64).powi(2); d += (*y as f64).powi(2); }
    (n / d.max(1e-300)).sqrt()
}
pub fn bit_equal_frac(a: &[f32], b: &[f32]) -> f64 { a.iter().zip(b).filter(|(x, y)| x.to_bits() == y.to_bits()).count() as f64 / a.len().max(1) as f64 }
pub fn max_abs(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) }

fn pick_rows(total: usize, max: Option<usize>) -> Vec<usize> {
    match max { Some(m) if m < total && m > 0 => (0..m).map(|i| i * total / m).collect(), _ => (0..total).collect() }
}
fn gather(x: &[f32], width: usize, rows: &[usize]) -> Vec<f32> { rows.iter().flat_map(|&r| x[r * width..(r + 1) * width].iter().copied()).collect() }

fn out(s: &str) { let _ = writeln!(std::io::stdout(), "{s}"); }

pub fn run(args: &[String]) -> Result<()> {
    let model = PathBuf::from(crate::arg_value(args, "--model-dir").context("--model-dir <native checkpoint or slice> is required")?);
    // the one REFb resolution: a fixture directory or its prefill_exact variant both name the dump
    let refd = crate::dsv41::dump::RefbDump::variant_dir(Path::new(crate::arg_value(args, "--dsv41-dump-ref").context("--dsv41-dump-ref <REFb fixture dir or its prefill_exact dir> is required")?));
    let part = crate::arg_value(args, "--dsv41-oracle-part").unwrap_or("moe");
    let max_rows = crate::arg_value(args, "--dsv41-oracle-rows").map(|s| s.parse::<usize>()).transpose().context("--dsv41-oracle-rows")?;
    let layers: Option<Vec<usize>> = crate::arg_value(args, "--layers").map(|s| s.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>()).transpose().context("--layers")?;
    let st = Store::open(&model)?;
    match part {
        "moe" => run_moe(&st, &refd, &layers.unwrap_or_else(|| vec![0, 1, 2]), max_rows),
        "engram" => {
            let tables = tables_store(&model, &st)?;
            run_engram(&st, tables.as_ref().unwrap_or(&st), &refd, &layers.unwrap_or_else(|| vec![1, 14]), max_rows)
        }
        p => bail!("part {p} is not an OF part"),
    }
}

fn tables_store(model: &Path, st: &Store) -> Result<Option<Store>> {
    if st.has("layers.1.engram.embed.weight") { return Ok(None); }
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?;
    let src = cfg["dsv41_slice"]["source_dir"].as_str().context("the slice's config.json has no dsv41_slice.source_dir and holds no Engram tables")?;
    Ok(Some(Store::open(Path::new(src))?))
}

fn rows_of(f: &RefFile, point: &str) -> Result<Vec<i64>> { f.i64s(&format!("{point}.rows")) }

pub fn run_moe(st: &Store, refd: &Path, layers: &[usize], max_rows: Option<usize>) -> Result<()> {
    let mut all_ok = true;
    for &l in layers {
        let f = RefFile::open(refd, &format!("L{l:02}"))?;
        let sh = f.shape(&format!("L{l:02}.ffn_in"))?;
        ensure!(sh.len() == 2 && sh[1] == DIM, "ffn_in shape {sh:?}");
        let rows = pick_rows(sh[0], max_rows);
        let n = rows.len();
        let x = gather(&f.f32(&format!("L{l:02}.ffn_in"))?, DIM, &rows);
        let want = gather(&f.f32(&format!("L{l:02}.ffn_out"))?, DIM, &rows);
        // the router dump covers every row, ffn_in a subset: align by absolute position
        let in_rows = rows_of(&f, &format!("L{l:02}.ffn_in"))?;
        let r_rows = rows_of(&f, &format!("L{l:02}.router_idx"))?;
        let rpos: Vec<usize> = rows.iter().map(|&r| r_rows.iter().position(|&p| p == in_rows[r]).with_context(|| format!("no router row for position {}", in_rows[r]))).collect::<Result<_>>()?;
        let ridx = f.i64s(&format!("L{l:02}.router_idx"))?;
        let rw = f.f32(&format!("L{l:02}.router_w"))?;
        let dump_rt = Routing {
            idx: rpos.iter().map(|&r| { let mut a = [0usize; TOPK]; for j in 0..TOPK { a[j] = ridx[r * TOPK + j] as usize; } a }).collect(),
            w: rpos.iter().map(|&r| { let mut a = [0f32; TOPK]; for j in 0..TOPK { a[j] = rw[r * TOPK + j]; } a }).collect(),
            margin: vec![0.0; n],
        };
        let mut lm = LayerMoe::new(st, l)?;
        // 1. the router
        let rt = super::route(&lm.gate, &x, n);
        let (mut set_mis, mut tie_mis, mut wmax) = (0usize, 0usize, 0f32);
        for i in 0..n {
            let mut a = rt.idx[i]; a.sort();
            let mut b = dump_rt.idx[i]; b.sort();
            if a != b { set_mis += 1; if rt.margin[i] < 1e-6 { tie_mis += 1; } }
            else { for j in 0..TOPK { let k = dump_rt.idx[i].iter().position(|&e| e == rt.idx[i][j]).unwrap(); wmax = wmax.max((rt.w[i][j] - dump_rt.w[i][k]).abs() / dump_rt.w[i][k].abs().max(1e-9)); } }
        }
        out(&format!("L{l:02} router: {n} rows, expert-set mismatches {set_mis} (of which near-ties {tie_mis}), max relative weight diff on agreeing rows {wmax:.2e}"));
        // 2. the experts on the dump's own routing (isolates the FFN math from the router)
        let t0 = std::time::Instant::now();
        let y = lm.forward(&x, n, &dump_rt, None)?;
        out(&format!("L{l:02} ffn_out (dump routing): rel-L2 {:.3e}, bit-equal {:.4}, max|d| {:.3e}  [{:.1}s]", rel_l2(&y, &want), bit_equal_frac(&y, &want), max_abs(&y, &want), t0.elapsed().as_secs_f64()));
        all_ok &= rel_l2(&y, &want) < 2e-2;
        // 3. the whole chain on the oracle's own routing
        if set_mis < n { let y2 = lm.forward(&x, n, &rt, None)?; out(&format!("L{l:02} ffn_out (own routing):  rel-L2 {:.3e}, bit-equal {:.4}", rel_l2(&y2, &want), bit_equal_frac(&y2, &want))); }
        // 4. the T0 inner-width split of every expert, at TP2 / TP4 / TP8 (1/2, 1/4, 1/8 of 2304)
        for parts in [2usize, 4, 8] {
            let ys = lm.forward(&x, n, &dump_rt, Some(parts))?;
            out(&format!("L{l:02} inner-width split 1/{parts}: vs dump rel-L2 {:.3e} (bit-equal {:.4}); vs unsplit oracle rel-L2 {:.3e}", rel_l2(&ys, &want), bit_equal_frac(&ys, &want), rel_l2(&ys, &y)));
        }
    }
    ensure!(all_ok, "an ffn_out rel-L2 exceeded the 2e-2 bar");
    out("DSV41_ORACLE_MOE_OK");
    Ok(())
}

pub fn run_engram(weights: &Store, tables: &Store, refd: &Path, layers: &[usize], max_rows: Option<usize>) -> Result<()> {
    let g = RefFile::open(refd, "G")?;
    let ids_all = g.i64s("G.engram_hash_ids")?;
    let grows = rows_of(&g, "G.engram_hash_ids")?;
    let mut ok = true;
    for &l in layers {
        let lp = match l { 1 => 0usize, 14 => 1, _ => bail!("Engram layers are 1 and 14, not {l}") };
        let f = RefFile::open(refd, &format!("L{l:02}"))?;
        let lrows = rows_of(&f, &format!("L{l:02}.x_pre_engram"))?;
        let sel = pick_rows(lrows.len(), max_rows);
        let n = sel.len();
        let xin = gather(&f.f32(&format!("L{l:02}.x_pre_engram"))?, eg::HC * eg::DIM, &sel);
        let want = gather(&f.f32(&format!("L{l:02}.x_in"))?, eg::HC * eg::DIM, &sel);
        let mut ids = Vec::with_capacity(n * eg::NCOL);
        for &s in &sel {
            let pos = lrows[s];
            let gi = grows.iter().position(|&p| p == pos).with_context(|| format!("no hash ids for row {pos}"))?;
            ids.extend_from_slice(&ids_all[(gi * 2 + lp) * eg::NCOL..(gi * 2 + lp + 1) * eg::NCOL]);
        }
        let w = eg::EngramW::load(weights, l)?;
        let e = eg::embed_rows(tables, l, &ids, n)?;
        let t0 = std::time::Instant::now();
        let a = act_quant(&e, n, eg::KIN);
        let kv_ref: Vec<f32> = qgemm_f32(&a, &w.wkv, None).into_iter().map(bf16r).collect();
        let y = eg::apply(&w, &xin, &kv_ref, n);
        out(&format!("L{l:02} engram out vs x_in: {n} rows, rel-L2 {:.3e}, bit-equal {:.4}, max|d| {:.3e}  [{:.1}s]", rel_l2(&y, &want), bit_equal_frac(&y, &want), max_abs(&y, &want), t0.elapsed().as_secs_f64()));
        ok &= rel_l2(&y, &want) < 1e-2;
        // the split experiment: kv partials per head shard, reduced in fp32 in rank order, one bf16 rounding
        let truth = qgemm_f64(&a, &w.wkv);
        let t32: Vec<f32> = truth.iter().map(|&v| v as f32).collect();
        let kv_ref_f32 = qgemm_f32(&a, &w.wkv, None);
        out(&format!("L{l:02}   reference f32 accumulator vs f64 truth: rel-L2 {:.3e}", rel_l2(&kv_ref_f32, &t32)));
        let mut variants: Vec<(String, Vec<Vec<usize>>)> = Vec::new();
        for tp in [2usize, 4, 8] { variants.push((format!("TP{tp} head shards"), (0..tp).map(|r| eg::rank_heads(tp, r)).collect())); }
        variants.push(("TP8 reduced in REVERSE rank order (control)".into(), (0..8).rev().map(|r| eg::rank_heads(8, r)).collect()));
        for (name, shards) in &variants {
            let kvf = eg::kv_f32(&w, &e, n, Some(shards));
            let kvb: Vec<f32> = kvf.iter().map(|&v| bf16r(v)).collect();
            let flips = kvb.iter().zip(&kv_ref).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let ys = eg::apply(&w, &xin, &kvb, n);
            out(&format!("L{l:02}   {name}: f32 kv vs truth rel-L2 {:.3e}; bf16 kv differs from sequential in {flips} of {} elements ({:.3}%); engram out vs unsplit oracle rel-L2 {:.3e} (bit-equal {:.4}), vs dump rel-L2 {:.3e}",
                         rel_l2(&kvf, &t32), kvb.len(), 100.0 * flips as f64 / kvb.len() as f64, rel_l2(&ys, &y), bit_equal_frac(&ys, &y), rel_l2(&ys, &want)));
        }
    }
    ensure!(ok, "an Engram output rel-L2 exceeded the 1e-2 bar");
    out("DSV41_ORACLE_ENGRAM_OK");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLICE: &str = "/mnt/models/dsv41/slices/native-fp8-L0-1-2-14-20-21-24-40-41-42";
    const REF: &str = "/mnt/models/dsv41/ref/short_chat/prefill_exact";

    /// Lab fixtures only: skips cleanly when the slice or the REFb dump is absent (public repo suite).
    #[test]
    fn moe_and_engram_match_the_refb_dump_on_short_chat() {
        if !Path::new(SLICE).join("model.safetensors.index.json").exists() || !Path::new(REF).join("L01.safetensors").exists() { return; }
        let native = ["/models/DeepSeek-V4.1-Flash-1017", "/models/DeepSeek-V4.1-Flash-1017"]
            .iter().map(Path::new).find(|p| p.join("model.safetensors.index.json").exists());
        let Some(native) = native else { return; };
        let st = Store::open(Path::new(SLICE)).unwrap();
        run_moe(&st, Path::new(REF), &[1], Some(4)).unwrap();
        let src = Store::open(native).unwrap();
        run_engram(&st, &src, Path::new(REF), &[1], Some(4)).unwrap();
    }
}
