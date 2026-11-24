//! PK3: the numeric cost of `--dsv41-prefill-bf16-partials` per layer, against REFb, on the served prefill kernels.
//!
//! `--probe-dsv41-kernel --dsv41-kernel-set trellis|mxfp4 --model-dir <V4.1 pack or slice> --dsv41-dump-ref <REFb fixture>[,..]
//! [--layers L,..]` (the pack's format picks the backend). For every MoE layer of the pack and every fixture whose dump holds more
//! than 32 rows of that layer (a prefill-shaped call: the wide kernels), the layer's DUMPED `ffn_in` and DUMPED routing go through
//! the routed (+ shared) experts as each rank of a TP1 / TP2 / TP4 / TP8 inner-width split computes them on its slices (the
//! loader's cut: EXL3 128-blocks rotated by expert id, native 32-groups), the rank partials are summed in f32 in rank order (the
//! reduce), and the bf16 `ffn_out` is compared with REFb's, with the opt-in off and on side by side:
//! * EXL3: the TP forward's call, routed + shared expert fused per rank (`moe_ex`), `ffn_out` = bf16(sum of the rank partials);
//! * native: the routed experts per rank (`native::moe::moe` on slices packed by `pack_layer`), the FP8 shared expert once (whole,
//!   bf16, as SL's forward: w1 / w3, SwiGLU, w2), `ffn_out` = bf16(sum + shared) (`d41f_add_round`'s arithmetic).
//! Success (`DSV41_PK3_PARTIALS_OK`): every cell inside SL's `ffn_out` tolerance for the backend, the opt-in engaged wherever it
//! can change bits (some output differs on vs off: EXL3 every TP, native TP >= 2), and the native TP1 control bit-equal on vs off
//! (whole experts already round per expert; the opt-in changes nothing there).

use crate::dsv41::census::classify::{Role, TensorClass as C};
use crate::dsv41::fwd::cmp::{self, Cmp, Pass};
use crate::dsv41::fwd::dev::{bf16, f32a, Gpu};
use crate::dsv41::fwd::glue;
use crate::dsv41::fwd::weights::{load_layer, LayerW, Pack};
use crate::dsv41::weights::{Accum, BackendKind, DenseWeight, DevPtr, DevTensor, MoeWeight, Route, Scratch, Shard};
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::Path;

const WORLDS: [usize; 4] = [1, 2, 4, 8];

/// The probe runs instead of the kernel set's own battery when a REFb fixture and a pack (config.json) are given.
pub fn wanted(args: &[String]) -> bool {
    let has_ref = crate::opts::lookup("dsv41-dump-ref").and_then(crate::opts::get).is_some();
    let pack = crate::arg_value(args, "--model-dir").is_some_and(|d| Path::new(d).join("config.json").exists());
    has_ref && pack
}

fn say(lines: &mut Vec<String>, l: String) { println!("{l}"); lines.push(l); }

fn bf16_round(v: f32) -> f32 { half::bf16::from_f32(v).to_f32() }

/// One fixture's inputs for one layer (the dumped row subset).
struct Rows { n: usize, x: Vec<u16>, ids: Vec<i32>, wts: Vec<f32>, refo: Vec<f32> }

fn rows_of(d: &crate::dsv41::dump::RefbDump, l: usize, top_k: usize) -> Result<Option<Rows>> {
    if !d.has(&crate::dsv41::dump::RefbDump::key(l, "ffn_in")) { return Ok(None); }
    let fin = d.layer(l, "ffn_in")?;
    let pos = fin.positions();
    let (ri, rw, ro) = (d.layer(l, "router_idx")?, d.layer(l, "router_w")?, d.layer(l, "ffn_out")?);
    let mut r = Rows { n: pos.len(), x: Vec::new(), ids: Vec::new(), wts: Vec::new(), refo: Vec::new() };
    for (k, &p) in pos.iter().enumerate() {
        r.x.extend(fin.row_f(k).iter().map(|&v| cmp::f32_to_bf16_bits(v)));
        let (a, b, c) = (ri.index_of(p).context("router_idx row")?, rw.index_of(p).context("router_w row")?, ro.index_of(p).context("ffn_out row")?);
        ensure!(ri.stride() == top_k && rw.stride() == top_k, "layer {l}: router dump stride {} / {} != top-{top_k}", ri.stride(), rw.stride());
        r.ids.extend(ri.row_i(a).iter().map(|&v| v as i32));
        r.wts.extend_from_slice(rw.row_f(b));
        r.refo.extend_from_slice(ro.row_f(c));
    }
    Ok(Some(r))
}

/// A copy of `w` with every expert cut to rank `rank` of `world` (EXL3: 128-blocks rotated by the expert id, as the loader cuts).
fn exl3_rank(w: &MoeWeight, world: usize, rank: usize, id_of: &dyn Fn(usize) -> usize) -> MoeWeight {
    let blocks = (w.inter / 128) as u64;
    let experts = w.experts.iter().enumerate().map(|(e, x)| {
        let (a, n) = crate::dsv41::load::splits::part(blocks, world, rank, id_of(e));
        let (lo, hi) = (a * 128, (a + n) * 128);
        let whole = world == 1;
        let mut y = x.clone();
        y[0].shard = if whole { Shard::Whole } else { Shard::Out(lo..hi) };
        y[2].shard = if whole { Shard::Whole } else { Shard::Out(lo..hi) };
        y[1].shard = if whole { Shard::Whole } else { Shard::In(lo..hi) };
        y
    }).collect();
    MoeWeight { experts, ..w.clone() }
}

/// The raw checkpoint bytes of a native layer's routed experts (w1, s1, w3, s3, w2, s2 per expert).
fn native_raw(p: &Pack, l: usize, n: usize) -> Result<Vec<[Vec<u8>; 6]>> {
    let c = &p.c;
    let mut files: HashMap<u32, std::fs::File> = HashMap::new();
    let mut raw: Vec<[Vec<u8>; 6]> = (0..n).map(|_| Default::default()).collect();
    for e in c.entries.iter().filter(|e| e.layer == Some(l as u16) && e.class.is_routed_expert()) {
        let slot = match (e.class, e.role) {
            (C::ExpertW1, Role::Weight) => 0, (C::ExpertW1, Role::Scale) => 1, (C::ExpertW3, Role::Weight) => 2, (C::ExpertW3, Role::Scale) => 3,
            (C::ExpertW2, Role::Weight) => 4, (C::ExpertW2, Role::Scale) => 5, (cl, r) => bail!("{}: unexpected expert tensor {cl:?} / {r:?}", e.name),
        };
        if !files.contains_key(&e.file) {
            let path = c.dir.join(&c.files[e.file as usize]);
            files.insert(e.file, std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?);
        }
        let mut v = vec![0u8; e.bytes as usize];
        files[&e.file].read_exact_at(&mut v, e.offset).with_context(|| format!("read {}", e.name))?;
        raw[e.sub.context("expert id")? as usize][slot] = v;
    }
    ensure!(raw.iter().all(|r| r.iter().all(|v| !v.is_empty())), "layer {l}: not every native expert tensor was found");
    Ok(raw)
}

/// A native layer's routed experts packed for rank `rank` of `world` (mx1 layout on the device) as a `MoeWeight`.
struct NativeRank { mw: MoeWeight, _b13: crate::dsv41::fwd::dev::Buf, _b2: crate::dsv41::fwd::dev::Buf }
fn native_rank(g: &Gpu, raw: &[[Vec<u8>; 6]], like: &MoeWeight, world: usize, rank: usize) -> Result<NativeRank> {
    use crate::dsv41::weights::native::moe as nm;
    let geom = if world == 1 { nm::pack::ExpertGeom::whole(like.dim, like.inter) } else { nm::pack::ExpertGeom::slice(like.dim, like.inter, rank, world)? };
    let rx: Vec<nm::reference::RawExpert> = raw.iter().map(|r| nm::reference::RawExpert { w1: &r[0], s1: &r[1], w3: &r[2], s3: &r[3], w2: &r[4], s2: &r[5] }).collect();
    let (h13, h2) = nm::pack_layer(&rx, &geom)?;
    let (b13, b2) = (g.buf(h13.len())?, g.buf(h2.len())?);
    g.h2d(b13.ptr(), &h13)?;
    g.h2d(b2.ptr(), &h2)?;
    let (e13, e2) = (geom.w13_bytes() as u64, geom.w2_bytes() as u64);
    let (s13, s2) = if geom.is_whole() { (Shard::Whole, Shard::Whole) } else {
        let r = geom.i0 as u64..(geom.i0 + geom.ns) as u64;
        (Shard::Out(r.clone()), Shard::In(r))
    };
    let fmt = like.experts[0][0].format.clone();
    let dw = |cl: C, out: u64, inn: u64, sh: Shard, t: Vec<DevTensor>| DenseWeight { key: format!("pk3.{cl:?}"), class: cl, layer: Some(like.layer), sub: None,
                                                                                   format: fmt.clone(), out, inn, shard: sh, tensors: t };
    let (dim, inter) = (like.dim as u64, like.inter as u64);
    let experts = (0..raw.len() as u64).map(|e| {
        let t13 = DevTensor { role: Role::Weight, ptr: DevPtr(b13.ptr() + e * e13), dtype: nm::DT_W13.into(), shape: vec![e13], bytes: e13 };
        let t2 = DevTensor { role: Role::Weight, ptr: DevPtr(b2.ptr() + e * e2), dtype: nm::DT_W2.into(), shape: vec![e2], bytes: e2 };
        [dw(C::ExpertW1, inter, dim, s13.clone(), vec![t13]), dw(C::ExpertW2, dim, inter, s2.clone(), vec![t2]), dw(C::ExpertW3, inter, dim, s13.clone(), vec![])]
    }).collect();
    Ok(NativeRank { mw: MoeWeight { experts, ..like.clone() }, _b13: b13, _b2: b2 })
}

/// The partial precision for the calls that follow (the option store: the launchers read it at every call).
fn set_mode(on: bool) {
    let o = crate::opt!("dsv41-prefill-bf16-partials");
    if on { crate::opts::set(o, "on") } else { crate::opts::unset(o) }
}

struct Cell { world: usize, off: Cmp, on: Cmp, onoff: Cmp }

pub fn run(args: &[String]) -> Result<Vec<String>> {
    let dir = crate::arg_value(args, "--model-dir").context("--model-dir <V4.1 pack or slice>")?;
    let dumps = crate::opts::lookup("dsv41-dump-ref").and_then(crate::opts::get).context("--dsv41-dump-ref")?;
    let g = Gpu::new()?;
    let pack = Pack::open(Path::new(dir))?;
    let cfg = pack.c.cfg.clone();
    let exl3 = pack.is_exl3();
    let pass = if exl3 { Pass::Exl3 } else { Pass::Native };
    let fixes: Vec<(String, crate::dsv41::dump::RefbDump)> = dumps.split(',').filter(|s| !s.is_empty())
        .map(|d| Ok((crate::dsv41::dump::RefbDump::fixture_name(Path::new(d)), crate::dsv41::dump::RefbDump::open(Path::new(d))?))).collect::<Result<_>>()?;
    let layers: Vec<usize> = match crate::arg_value(args, "--layers") {
        Some(s) => crate::dsv41::slice::parse_layers(s)?,
        None => (0..cfg.layers).filter(|&l| pack.has_layer(l)).collect(),
    };
    let was_on = crate::dsv41::weights::partials::on();
    let mut lines = Vec::new();
    say(&mut lines, format!("PK3 partials probe: pack {dir} ({}), fixtures {}, layers {layers:?}, TP {WORLDS:?}; per cell: ffn_out vs REFb with \
                             --dsv41-prefill-bf16-partials off | on (rel-L2, bf16-equal share, max |d|) and on vs off",
                            if exl3 { "EXL3" } else { "native" }, fixes.iter().map(|f| f.0.clone()).collect::<Vec<_>>().join(", ")));
    let d = cfg.hidden;
    let scr_bytes = 2usize << 30;
    let scr = g.buf(scr_bytes)?;
    let s = Scratch { ptr: DevPtr(scr.ptr()), bytes: scr_bytes };
    let mut cells: Vec<Cell> = Vec::new();
    let mut fails: Vec<String> = Vec::new();
    for &l in &layers {
        if l >= cfg.layers { continue; }
        let (_, top_k) = cfg.moe_of(l);
        let mut work: Vec<(String, Rows)> = Vec::new();
        for (name, dd) in &fixes {
            match rows_of(dd, l, top_k)? {
                Some(r) if r.n > 32 => work.push((name.clone(), r)),
                Some(r) => say(&mut lines, format!("  L{l:02} {name}: {} dumped rows (a decode-class call; the opt-in does not apply): skipped", r.n)),
                None => {}
            }
        }
        if work.is_empty() { continue; }
        let t0 = std::time::Instant::now();
        let w: LayerW = load_layer(&g, &pack, l)?;
        let (mw, kind) = w.moe.as_ref().context("layer without routed experts")?;
        ensure!((*kind == BackendKind::Exl3) == exl3, "layer {l}: MoE backend {kind:?} on a {} pack", if exl3 { "EXL3" } else { "native" });
        let raw = if exl3 { None } else { Some(native_raw(&pack, l, mw.n_experts)?) };
        say(&mut lines, format!("-- layer {l:02} (weights loaded in {:.1} s)", t0.elapsed().as_secs_f64()));
        // device inputs per fixture; native: the shared expert's bf16 output once per fixture
        let mut dev: Vec<(u64, u64, u64, Vec<f32>)> = Vec::new();   // x, ids, wts, shared (native)
        let mut keep = Vec::new();
        for (_, r) in &work {
            let (x, i, wt) = (g.up(&r.x)?, g.up(&r.ids)?, g.up(&r.wts)?);
            let (px, pi, pw) = (crate::dsv41::fwd::dev::dp(&x), crate::dsv41::fwd::dev::dp(&i), crate::dsv41::fwd::dev::dp(&wt));
            let mut sh = Vec::new();
            if !exl3 {
                let inter = cfg.moe_inter;
                let (g1, g3, act, so) = (g.buf(r.n * inter * 2)?, g.buf(r.n * inter * 2)?, g.buf(r.n * inter * 2)?, g.buf(r.n * d * 2)?);
                let sc = glue::Scr { ptr: scr.ptr(), bytes: scr_bytes };
                let lin = |c: C, xin: u64, k: usize, y: u64, n: usize| -> Result<()> {
                    let lw = w.lin(c)?;
                    let mut ya = bf16(y, r.n, n, n);
                    ya.cols = lw.0.out as usize;
                    glue::linear(&g, "pk3.shared", lw, &bf16(xin, r.n, k, k), &ya, Accum::Overwrite, &sc).map(|_| ())
                };
                lin(C::SharedW1, px, d, g1.ptr(), inter)?;
                lin(C::SharedW3, px, d, g3.ptr(), inter)?;
                glue::swiglu(&g, g1.ptr(), g3.ptr(), act.ptr(), inter, r.n, inter, cfg.swiglu_limit as f32)?;
                lin(C::SharedW2, act.ptr(), inter, so.ptr(), d)?;
                sh = g.down_bf16(so.ptr(), r.n * d)?;
            }
            keep.push((x, i, wt));
            dev.push((px, pi, pw, sh));
        }
        // sums[fixture][world][mode] (f32 sum of the rank partials, rank order)
        let mut sums: Vec<Vec<[Vec<f32>; 2]>> = work.iter().map(|(_, r)| WORLDS.iter().map(|_| [vec![0f32; r.n * d], vec![0f32; r.n * d]]).collect()).collect();
        let shared_whole = if exl3 {
            Some(MoeWeight { layer: l as u16, n_experts: 1, local: 0..1, top_k: 1, dim: d, inter: cfg.moe_inter, swiglu_limit: cfg.swiglu_limit as f32,
                             experts: vec![[w.lin(C::SharedW1)?.0.clone(), w.lin(C::SharedW2)?.0.clone(), w.lin(C::SharedW3)?.0.clone()]] })
        } else { None };
        for (wi, &world) in WORLDS.iter().enumerate() {
            for rank in 0..world {
                let nat = match &raw { Some(rw) => Some(native_rank(&g, rw, mw, world, rank)?), None => None };
                let (mw_r, sh_r) = match &nat {
                    Some(n) => (n.mw.clone(), None),
                    None => (exl3_rank(mw, world, rank, &|e| e), Some(exl3_rank(shared_whole.as_ref().unwrap(), world, rank, &|_| 0))),
                };
                for (fi, (_, r)) in work.iter().enumerate() {
                    let (px, pi, pw, _) = &dev[fi];
                    let y = g.buf(r.n * d * 4)?;
                    let route = Route { ids: DevPtr(*pi), weights: DevPtr(*pw), top_k };
                    for (mi, on) in [false, true].into_iter().enumerate() {
                        set_mode(on);
                        g.memset(y.ptr(), r.n * d * 4, 0)?;
                        let xa = bf16(*px, r.n, d, d);
                        let ya = f32a(y.ptr(), r.n, d, d);
                        if exl3 {
                            crate::dsv41::weights::exl3::moe::moe_ex(&g.l(), &mw_r, sh_r.as_ref(), &xa, &route, &ya, s, crate::dsv41::weights::exl3::moe::MoePath::Auto)?;
                        } else {
                            crate::dsv41::weights::native::moe::moe(&g.l(), &mw_r, &xa, &route, &ya, s)?;
                        }
                        let part = g.down::<f32>(y.ptr(), r.n * d)?;
                        for (a, b) in sums[fi][wi][mi].iter_mut().zip(&part) { *a += *b; }
                    }
                }
                crate::dsv41::weights::exl3::moe::clear_tables();
            }
        }
        set_mode(was_on);
        for (fi, (name, r)) in work.iter().enumerate() {
            for (wi, &world) in WORLDS.iter().enumerate() {
                let out = |mi: usize| -> Vec<f32> {
                    let sm = &sums[fi][wi][mi];
                    if exl3 { sm.iter().map(|&v| bf16_round(v)).collect() } else { sm.iter().zip(&dev[fi].3).map(|(&v, &sv)| bf16_round(v + sv)).collect() }
                };
                let (o0, o1) = (out(0), out(1));
                let c = Cell { world, off: cmp::cmp(&o0, &r.refo), on: cmp::cmp(&o1, &r.refo), onoff: cmp::cmp(&o1, &o0) };
                let tol = cmp::tol(pass, "ffn_out", l);
                let engaged = c.onoff.same < 1.0;
                let must_engage = exl3 || world > 1;
                let mut flags = Vec::new();
                if c.off.rel > tol.rel || c.on.rel > tol.rel { flags.push(format!("OUT OF TOLERANCE (rel {:.0e})", tol.rel)); }
                if must_engage && !engaged { flags.push("NOT ENGAGED (on == off)".into()); }
                if !must_engage && engaged { flags.push("CONTROL MOVED (native TP1 must not change)".into()); }
                say(&mut lines, format!("  L{l:02} {name:<11} rows {:>4} TP{world}: off rel {:.3e} eq {:.4} max {:.2e} | on rel {:.3e} eq {:.4} max {:.2e} | on vs off rel {:.3e} eq {:.4}{}",
                                        r.n, c.off.rel, c.off.same, c.off.max_abs, c.on.rel, c.on.same, c.on.max_abs, c.onoff.rel, c.onoff.same,
                                        if flags.is_empty() { String::new() } else { format!("  {}", flags.join("; ")) }));
                if !flags.is_empty() { fails.push(format!("L{l} {name} TP{world}: {}", flags.join("; "))); }
                cells.push(c);
            }
        }
        drop(keep);
        crate::dsv41::weights::exl3::moe::clear_tables();
        drop(w);
    }
    // summary per TP: means over the cells
    say(&mut lines, "summary (means over layers x fixtures): TP | cells | rel-L2 vs REFb off | on | increase | bf16-equal vs REFb off | on | on vs off rel | on vs off equal".into());
    for &world in &WORLDS {
        let cs: Vec<&Cell> = cells.iter().filter(|c| c.world == world).collect();
        if cs.is_empty() { continue; }
        let m = |f: &dyn Fn(&Cell) -> f64| cs.iter().map(|c| f(c)).sum::<f64>() / cs.len() as f64;
        let (ro, rn) = (m(&|c| c.off.rel), m(&|c| c.on.rel));
        say(&mut lines, format!("  TP{world} | {} | {:.3e} | {:.3e} | {:+.1} % | {:.4} | {:.4} | {:.3e} | {:.4}", cs.len(), ro, rn, (rn / ro - 1.0) * 100.0,
                                m(&|c| c.off.same), m(&|c| c.on.same), m(&|c| c.onoff.rel), m(&|c| c.onoff.same)));
    }
    ensure!(!cells.is_empty(), "no prefill-shaped cell ran (no layer of the pack has more than 32 dumped rows in the fixtures)");
    if !fails.is_empty() { bail!("{} cell(s) failed: {}", fails.len(), fails.join(" | ")); }
    say(&mut lines, "DSV41_PK3_PARTIALS_OK".into());
    Ok(lines)
}
