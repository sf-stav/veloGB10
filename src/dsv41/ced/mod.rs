//! DSV4.1 leg CE: the exact CED prefill (compressed-encoder prefill), chunking, prefix states and the score mode.
//!
//! * `schedule.rs` — which (layer, row) pairs a prefill computes: exact / CED (REFb's exact minimal cone, the
//!   default) / the opt-in truncated replay; resumes from a prefix state (full or encoder-only) and the caps they
//!   imply; the work model (REFb's MAC cost model) for the work-saved table. Pure, no GPU.
//! * `lane.rs` — a lane's per-layer device state (SWA ring + chunk planes, KV-source caches) and its snapshots.
//! * `chunk.rs` — one layer of one lane over one prefill chunk (attention side with the ring written AFTER the
//!   chunk's queries read it; the row-local FFN side), on SL's forward pieces in the wide row family.
//! * `probe.rs` — `--probe-dsv41-ced`: the per-layer gate (variants bit for bit, CED against REFb).
//! * `rows.rs`, `chain.rs`, `score.rs` — the row-local subsets, the chained layers, the score head.
//!
//! Mode: `--probe-dsv41-ced --model-dir <slice> --dsv41-dump-ref <REFb fixture>[,...] [--layers ..] [--engram-dir DIR]
//! [--dsv41-ced-chunks 2048,4095,1001] [--dsv41-ced-parts work,gate,rows,chain,score] [--dsv41-score-out FILE]`
//! (RESULT: DSV41_CE_OK).

pub mod chain;
pub mod chunk;
pub mod lane;
pub mod probe;
pub mod rows;
pub mod schedule;
pub mod score;
pub mod timing;

use crate::opts;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

fn flag_val(name: &str) -> Option<String> { opts::lookup(name).and_then(opts::get) }

/// Result lines go out as they are produced (a run takes minutes; a crash keeps the rows so far).
fn say(lines: &mut Vec<String>, l: String) { println!("{l}"); lines.push(l); }

pub fn fmt_row(r: &probe::Row) -> String {
    let v = match r.pass { Some(true) => "PASS", Some(false) => "FAIL", None => "info" };
    format!("{:<10} L{:02} {:<34} {v}  {}", r.fixture, r.layer, r.point, r.text)
}

/// The work-saved table (schedule arithmetic: layer-rows and REFb's MAC cost model), no GPU.
pub fn work_table(cfg: &crate::dsv41::config::Dsv41Config) -> Result<Vec<String>> {
    use schedule::{prefill_macs, Mode, Schedule};
    let mut out = vec!["work saved by CED (layer-rows of full layers; MACs by REFb's cost model, ref_forward.py prefill_macs); 1.000 = the exact every-layer prefill".to_string(),
                       format!("{:>9} | {:>22} | {:>22} | {:>22} | {:>22}", "prompt", "CED (exact, default)", "replay 2560 (exact)", "replay 140 (approx.)", "exact"),
                       "".into()];
    for p in [2048usize, 4096, 4197, 8192, 16384, 32768, 33467, 131072, 1_048_576] {
        let e = Schedule::new(cfg, p, Mode::Exact)?;
        let (er, em) = (e.layer_rows() as f64, prefill_macs(cfg, &e));
        let cell = |m: Mode| -> Result<String> {
            let s = Schedule::new(cfg, p, m)?;
            Ok(format!("rows {:.3} MACs {:.3}", s.layer_rows() as f64 / er, prefill_macs(cfg, &s) / em))
        };
        out.push(format!("{p:>9} | {:>22} | {:>22} | {:>22} | {:.2e} MACs", cell(Mode::Ced)?, cell(Mode::Replay { keep: 2560 })?, cell(Mode::Replay { keep: 140 })?, em));
    }
    out[2] = "-".repeat(out[1].len());
    Ok(out)
}

/// The mode.
pub fn run(args: &[String]) -> Result<Vec<String>> {
    use crate::dsv41::fwd::{cmp::Pass, dev::Gpu, glue, layer, weights};
    let pack_dir = PathBuf::from(crate::arg_value(args, "--model-dir").context("--model-dir <slice or pack> is required")?);
    let parts = flag_val("dsv41-ced-parts").unwrap_or_else(|| "work,gate,rows,chain,score".into());
    let has = |p: &str| parts.split(',').any(|x| x.trim() == p);
    let chunks: Vec<usize> = flag_val("dsv41-ced-chunks").unwrap_or_else(|| "2048,4095,1001".into()).split(',').filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<usize>().with_context(|| format!("--dsv41-ced-chunks: {s}"))).collect::<Result<_>>()?;
    let mut lines = Vec::new();
    let cfg = crate::dsv41::config::Dsv41Config::load(&pack_dir)?;
    let served = schedule::mode_from_flags()?;
    say(&mut lines, format!("served prefill mode (--dsv41-prefill / --dsv41-ced-keep): {} (the probe runs every variant below regardless)", served.name()));
    if has("work") { for l in work_table(&cfg)? { say(&mut lines, l); } }
    if !(has("gate") || has("rows") || has("chain") || has("score") || has("timing")) { say(&mut lines, "DSV41_CE_OK (work only)".into()); return Ok(lines); }
    let dumps = flag_val("dsv41-dump-ref").context("--dsv41-dump-ref <REFb fixture dir>[,<dir>...] is required")?;
    let g = Gpu::new()?;
    let pack = weights::Pack::open(&pack_dir)?;
    let cfg = pack.c.cfg.clone();
    let pass = if pack.is_exl3() { Pass::Exl3 } else { Pass::Native };
    let mut fixes = Vec::new();
    for d in dumps.split(',').filter(|s| !s.is_empty()) {
        let dd = crate::dsv41::cpu::mhc::refdump::RefDump::open(Path::new(d))?;
        let name = crate::dsv41::dump::RefbDump::fixture_name(Path::new(d));
        let p = dd.p;
        fixes.push(probe::Fix { name, d: dd, p });
    }
    if fixes.is_empty() { bail!("no fixture"); }
    let layers: Vec<usize> = match crate::arg_value(args, "--layers") {
        Some(s) => crate::dsv41::slice::parse_layers(s)?,
        None => (0..cfg.layers).filter(|&l| pack.has_layer(l)).collect(),
    };
    let trunk: Vec<usize> = layers.iter().copied().filter(|&l| l < cfg.layers).collect();
    say(&mut lines, format!("CE: pack {} ({pass:?}), trunk layers {trunk:?}, fixtures {}, chunks {chunks:?}, parts {parts}", pack_dir.display(),
                            fixes.iter().map(|f| format!("{} (P {})", f.name, f.p)).collect::<Vec<_>>().join(", ")));
    let scr_bytes = 2usize << 30;
    let scr = g.buf(scr_bytes)?;
    let f = layer::Fwd { g: &g, cfg: &cfg, scr: glue::Scr { ptr: scr.ptr(), bytes: scr_bytes }, b: layer::Bump::new(&g, 6usize << 30)?,
                         rope: crate::dsv41::fwd::rope_set(&cfg)?, checks: true, wide_min: 33 };
    let mut gt = probe::Gate { f, pass, rows: Vec::new() };
    if has("gate") {
        let mut vars: Vec<Vec<probe::Variant>> = fixes.iter().map(|fx| probe::variants(&cfg, fx.p, &chunks)).collect::<Result<_>>()?;
        for &l in &trunk {
            let t0 = std::time::Instant::now();
            let w = weights::load_layer(&g, &pack, l)?;
            let load_s = t0.elapsed().as_secs_f64();
            let n0 = gt.rows.len();
            for (fx, vs) in fixes.iter().zip(vars.iter_mut()) {
                gt.layer_gate(&w, fx, vs).with_context(|| format!("layer {l} on {}", fx.name))?;
            }
            crate::dsv41::weights::exl3::moe::clear_tables();
            drop(w);
            say(&mut lines, format!("-- layer {l} ({:?}, {:?}; weights loaded in {load_s:.1} s)", cfg.attn_class(l), cfg.layer_role(l)));
            for r in &gt.rows[n0..] { say(&mut lines, fmt_row(r)); }
        }
    }
    if has("rows") {
        for &l in &trunk {
            let w = weights::load_layer(&g, &pack, l)?;
            let n0 = gt.rows.len();
            for fx in &fixes { gt.rows_gate(&w, fx).with_context(|| format!("rows: layer {l} on {}", fx.name))?; }
            crate::dsv41::weights::exl3::moe::clear_tables();
            drop(w);
            say(&mut lines, format!("-- rows: layer {l}"));
            for r in &gt.rows[n0..] { say(&mut lines, fmt_row(r)); }
        }
    }
    if has("chain") {
        let engram_dir = PathBuf::from(crate::arg_value(args, "--engram-dir").unwrap_or("/models/DeepSeek-V4.1-Flash-1017/engram"));
        for (chain, embed) in [(vec![0usize, 1, 2], true), (vec![20usize, 21], false)] {
            if !chain.iter().all(|l| trunk.contains(l)) { continue; }
            for fx in &fixes {
                let n0 = gt.rows.len();
                gt.chain_gate(&pack, fx, &chain, embed, &chunks, &engram_dir).with_context(|| format!("chain {chain:?} on {}", fx.name))?;
                for r in &gt.rows[n0..] { say(&mut lines, fmt_row(r)); }
            }
        }
    }
    if has("score") {
        let n0 = gt.rows.len();
        let out = flag_val("dsv41-score-out").map(PathBuf::from);
        gt.score_gate(&pack, &fixes, out.as_deref())?;
        for r in &gt.rows[n0..] { say(&mut lines, fmt_row(r)); }
    }
    if has("timing") {
        for fx in &fixes {
            if fx.p < 1024 { continue; }
            for l in gt.timing(&pack, fx, &trunk)? { say(&mut lines, l); }
        }
    }
    for l in crate::dsv41::fwd::receipts::boot_lines() { say(&mut lines, l); }
    let fails: Vec<String> = gt.rows.iter().filter(|r| r.pass == Some(false)).map(|r| format!("{} L{} {}", r.fixture, r.layer, r.point)).collect();
    let gated = gt.rows.iter().filter(|r| r.pass.is_some()).count();
    say(&mut lines, format!("CE gate ({pass:?}): {gated} gated points, {} pass, {} fail", gated - fails.len(), fails.len()));
    if fails.is_empty() { say(&mut lines, "DSV41_CE_OK".into()); } else { say(&mut lines, format!("DSV41_CE_FAIL: {}", fails.join("; "))); }
    Ok(lines)
}
