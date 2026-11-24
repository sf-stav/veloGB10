//! DSV4.1 leg SL: the per-layer GPU forward of the trunk on ONE box (no TP split), assembled from the merged
//! kernel bands, plus its gate against REFb's dumps (plan Revision 5: every layer and class fed its DUMPED
//! input; never a chain of layers), the engagement receipts and a step-time table.
//!
//! * `dev.rs` — the device context: one blocking stream, raw launches, buffers, events.
//! * `weights.rs` — one layer of ANY pack (slice or full; EXL3, native, mixed) on the device through the census.
//! * `glue.rs` + `kernels/gpu_dsv41/02_fwd.cuh` (band 02-04) — row RoPE, SwiGLU, add-and-round; the backend dispatch.
//! * `layer.rs` — the forward pieces (mHC, q path, window, compressor, indexer, attention, wo, router, FFN, Engram).
//! * `gate.rs` + `cmp.rs` — the REFb gate, tolerances per backend, cache decoders.
//! * `receipts.rs` — which kernel entry served each path and width: one block per process + `/metrics`.
//! * `timing.rs` — the slice's step time at 1 / 6 / 24 rows (one box; informative, not the TP posture).
//!
//! * `dspark.rs` — the DSpark stages (40-42) per op against OD's oracle traces.
//!
//! Mode: `--probe-dsv41-slice-model --model-dir <pack> --dsv41-dump-ref <REFb fixture>[,<fixture>...]
//! [--layers 0,1,2,...] [--engram-dir DIR] [--dsv41-kernel-fixture <OD trace dir>[,...]] [--dsv41-kernel-timing <rows,..>]`
//! (RESULT: DSV41_SL_OK). The OD traces are the `--dsv41-oracle-trace` output of `--probe-dsv41-oracle
//! --dsv41-oracle-part dspark`, one directory per fixture named like the fixture.

pub mod cmp;
pub mod dev;
pub mod dspark;
pub mod gate;
pub mod glue;
pub mod layer;
pub mod receipts;
pub mod timing;
pub mod weights;

use crate::opts;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Entry kernels of band 02-04 (the gpu_dsv41 loader collects them).
pub const FNS: &[&str] = glue::FNS;

fn flag_val(name: &str) -> Option<String> { opts::lookup(name).and_then(opts::get) }

/// The mode's result lines go out as they are produced (a fixture run takes minutes; a crash keeps the rows so far).
fn say(lines: &mut Vec<String>, l: String) { println!("{l}"); lines.push(l); }

/// The RoPE tables each consumer uses: since the SL RoPE unification, the ONE builder (`crate::dsv41::rope`)
/// for all of them (the per-consumer split stays so a variant can be compared).
pub fn rope_set(cfg: &crate::dsv41::config::Dsv41Config) -> Result<layer::RopeSet> {
    use crate::dsv41::rope::{table_for, Class};
    let t = layer::Tabs { compressed: table_for(cfg, Class::Compressed)?, plain: table_for(cfg, Class::Plain)? };
    Ok(layer::RopeSet { q: t, kc: t, ka: t, ki: t.compressed })
}

/// The mode.
pub fn run(args: &[String]) -> Result<Vec<String>> {
    let pack_dir = PathBuf::from(crate::arg_value(args, "--model-dir").context("--model-dir <pack: a slice or a full pack> is required")?);
    let dumps = flag_val("dsv41-dump-ref").context("--dsv41-dump-ref <REFb fixture dir>[,<dir>...] is required")?;
    let engram_dir = PathBuf::from(crate::arg_value(args, "--engram-dir").unwrap_or("/models/DeepSeek-V4.1-Flash-1017/engram"));
    let g = dev::Gpu::new()?;
    let pack = weights::Pack::open(&pack_dir)?;
    let cfg = pack.c.cfg.clone();
    let pass = if pack.is_exl3() { cmp::Pass::Exl3 } else { cmp::Pass::Native };
    let mut fixes = Vec::new();
    for d in dumps.split(',').filter(|s| !s.is_empty()) {
        let dd = crate::dsv41::dump::RefbDump::open(Path::new(d))?;
        let name = crate::dsv41::dump::RefbDump::fixture_name(Path::new(d));
        let p = dd.p;
        fixes.push(gate::Fix { name, d: dd, p });
    }
    if fixes.is_empty() { bail!("no fixture"); }
    let layers: Vec<usize> = match crate::arg_value(args, "--layers") {
        Some(s) => crate::dsv41::slice::parse_layers(s)?,
        None => (0..cfg.layers).filter(|&l| pack.has_layer(l)).collect(),
    };
    let mut lines = Vec::new();
    say(&mut lines, format!("SL slice model: pack {} ({:?}, {} layers {:?}), fixtures {}", pack_dir.display(), pass, layers.len(), layers,
                            fixes.iter().map(|f| format!("{} (P {})", f.name, f.p)).collect::<Vec<_>>().join(", ")));
    let scr_bytes = 1usize << 30;
    let scr = g.buf(scr_bytes)?;
    let f = layer::Fwd { g: &g, cfg: &cfg, scr: glue::Scr { ptr: scr.ptr(), bytes: scr_bytes }, b: layer::Bump::new(&g, 3usize << 30)?, rope: rope_set(&cfg)?, checks: true, wide_min: 0 };
    let mut gt = gate::Gate { f, pass, rows: Vec::new(), io: None, io_max: 0, keep: gate::Keep::default() };
    if layers.iter().any(|l| cfg.engram_layers.contains(l)) {
        let max_tok = fixes.iter().map(|f| f.d.layer(1, "x_in").map(|t| t.n_rows()).unwrap_or(1024)).max().unwrap_or(1024).max(1);
        gt.open_engram(&pack_dir, &engram_dir, max_tok)?;
    }
    let mut states: Vec<gate::FixState> = fixes.iter().map(|_| gate::FixState::default()).collect();
    // OD's oracle traces of the DSpark stages (dsv41 dump format, one directory per fixture, named like the fixture)
    let ds_fixes: Vec<dspark::DsFix> = match flag_val("dsv41-kernel-fixture") {
        Some(s) => s.split(',').filter(|x| !x.is_empty()).map(|d| dspark::DsFix::open(Path::new(d))).collect::<Result<_>>()?,
        None => Vec::new(),
    };
    for &l in &layers {
        if l >= cfg.layers {
            if ds_fixes.is_empty() { say(&mut lines, format!("layer {l}: DSpark stage: no OD trace given (--dsv41-kernel-fixture), not gated")); continue; }
            let n0 = gt.rows.len();
            let t0 = std::time::Instant::now();
            let w = weights::load_layer(&g, &pack, l)?;
            let load_s = t0.elapsed().as_secs_f64();
            for dfx in &ds_fixes {
                let refb = fixes.iter().find(|f| f.name == dfx.name);
                gt.dspark_stage(&w, dfx, refb, &cfg).with_context(|| format!("DSpark stage {l} on {}", dfx.name))?;
            }
            crate::dsv41::weights::exl3::moe::clear_tables();
            drop(w);
            say(&mut lines, format!("-- layer {l} (DSpark stage {}; weights loaded in {load_s:.1} s; per op against OD's oracle traces)", l - cfg.layers));
            for r in &gt.rows[n0..] { say(&mut lines, fmt_row(r)); }
            continue;
        }
        let n0 = gt.rows.len();
        let load_s = gate::run_layer(&mut gt, &pack, l, &fixes, &mut states)?;
        say(&mut lines, format!("-- layer {l} ({:?}; weights loaded in {load_s:.1} s)", cfg.attn_class(l)));
        for r in &gt.rows[n0..] { say(&mut lines, fmt_row(r)); }
    }
    if let Some(t) = flag_val("dsv41-kernel-timing") {
        for l in timing::run(&mut gt, &pack, &layers, &fixes, &mut states, &t)? { say(&mut lines, l); }
    }
    for l in receipts::boot_lines() { say(&mut lines, l); }
    // the same counters through the process's /metrics body (the served path renders this one)
    let body = crate::metrics::render(true);
    let series: Vec<&str> = body.lines().filter(|l| l.starts_with("velogb10_dsv41_fwd_calls_total{")).collect();
    say(&mut lines, format!("/metrics: {} velogb10_dsv41_fwd_calls_total series, e.g. {}", series.len(), series.first().copied().unwrap_or("(none)")));
    let fails: Vec<String> = gt.rows.iter().filter(|r| r.pass == Some(false)).map(|r| format!("{} L{} {}", r.fixture, r.layer, r.point)).collect();
    let gated = gt.rows.iter().filter(|r| r.pass.is_some()).count();
    say(&mut lines, format!("SL gate ({pass:?}): {gated} gated points, {} within tolerance, {} not", gated - fails.len(), fails.len()));
    if fails.is_empty() { say(&mut lines, "DSV41_SL_OK".into()); } else { say(&mut lines, format!("DSV41_SL_FAIL: {}", fails.join("; "))); }
    Ok(lines)
}

fn fmt_row(r: &gate::Row) -> String {
    let v = match r.pass { Some(true) => "PASS", Some(false) => "FAIL", None => "info" };
    let tol = if r.tol.rel.is_infinite() { "-".to_string() } else { format!("rel<={:.0e} same>={:.2}", r.tol.rel, r.tol.same) };
    format!("{:<11} L{:02} {:<13} rel-L2 {:.3e}  same {:7.3}%  max|d| {:.3e}  n {:>9}  [{tol}] {v} {}", r.fixture, r.layer, r.point, r.c.rel, 100.0 * r.c.same, r.c.max_abs, r.c.n, r.extra)
}
