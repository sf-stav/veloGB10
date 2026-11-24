//! CE: `--probe-dsv41-ced`, the CED / chunked / resumed prefill on the slice (plan Revision 5: every layer is
//! fed its DUMPED input, never a chain, when compared with the reference; bit-identity is checked only between
//! variants of our own batch-invariant forward, REFb §6.2).
//!
//! Per layer and fixture ([`layer_gate`]), the attention side over the dumped `attn_in` (every prompt row), as
//! a set of VARIANTS that must agree bit for bit with the reference variant `exact@one` (every row, one chunk):
//!
//! | variant | what |
//! |---|---|
//! | `exact@one` | every layer on every row, single shot (8,192-row chunks past 8K: the largest chunk T0 plans at TP2) |
//! | `ced@one`, `ced@<c>` | the CED rows, single shot and every `--dsv41-ced-chunks` chunking |
//! | `exact@<c0>` | every row, the first chunking |
//! | `ced@<c0>+enc@K` | chunks to K (a 128 multiple at or below the decoder cone), snapshot, poison, restore, continue: an ENCODER-only state |
//! | `ced:pfx<K>+full` | the CED prefill of the first K rows as its own prompt, snapshot at its end (a FULL state), poison, restore, then the rest of the prompt resumed |
//!
//! Compared: every row a variant computed (attn_out, a 64-bit hash per row) against the reference, every row the
//! CED schedule needs is present, and the decode state (the ring's last 128 positions, a KV source's main KV /
//! index K / frontier). Then `ced@<c0>` against REFb: attn_out on the dumped rows it computed, the ring, the caches,
//! the top-k (index sources), with SL's tolerances (`fwd::cmp::tol`).

use super::chunk::{self, Cands};
use super::lane::{Lane, LaneSnap, LayerLane};
use super::schedule::{self, n_of, Mode, Schedule, StateKind};
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::cpu::mhc::refdump::{RefDump, RefTensor};
use crate::dsv41::fwd::cmp::{self, Cmp, Pass, Tol};
use crate::dsv41::fwd::dev::Gpu;
use crate::dsv41::fwd::layer::{bytemuck_u16, Fwd};
use crate::dsv41::fwd::weights::LayerW;
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// One result line of the probe.
#[derive(Clone, Debug)]
pub struct Row { pub fixture: String, pub layer: usize, pub point: String, pub pass: Option<bool>, pub text: String }

pub struct Fix { pub name: String, pub d: RefDump, pub p: usize }

/// The largest single chunk a variant uses (memory of the probe's bump arena; T0's TP2 chunk).
pub const ONE_MAX: usize = 8192;

pub fn fnv(b: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &x in b { h ^= x as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

/// A variant: phases of (schedule, chunks); after a phase marked `snap`, the layer's state is snapshotted,
/// poisoned and restored before the next phase (`kind` = what the snapshot claims to be).
pub struct Phase { pub sched: Schedule, pub chunks: Vec<(usize, usize)>, pub snap: Option<StateKind> }
pub struct Variant {
    pub name: String, pub phases: Vec<Phase>, pub lane: Lane, pub cands: Cands,
    /// run the linears in the wide row family (`Fwd::wide_min`); false = information only (shows why it is needed)
    pub pad: bool,
    /// the opt-in truncated replay of the last `keep` rows: rows whose window reaches below the replay are
    /// approximate by design; the rest and the decode state must equal the exact prefill (keep >= 2,541: exact)
    pub replay: Option<usize>,
}

/// First row of decoder layer `l` that a truncated replay of `keep` rows leaves exact, when the layers below it are
/// exact on their inputs (`chained` = how many decoder layers up to and including `l` truncate: 1 when the layer is
/// fed its dumped input): each truncating layer pushes the boundary 127 rows up.
pub fn replay_exact_from(p: usize, keep: usize, chained: usize, window: usize) -> usize { p.saturating_sub(keep) + (window - 1) * chained }

/// What a variant produced for one layer.
#[derive(Default)]
pub struct LayerOut {
    /// absolute row -> FNV-64 of its attn_out (bf16 bytes)
    pub rows: BTreeMap<usize, u64>,
    /// the dumped rows' attn_out (bf16 bits) and top-k
    pub keep: BTreeMap<usize, Vec<u16>>,
    pub topk: BTreeMap<usize, Vec<i32>>,
    /// named state bytes
    pub state: Vec<(&'static str, Vec<u8>)>,
}

pub fn bf16_bits(v: &[f32]) -> Vec<u16> { v.iter().map(|&x| cmp::f32_to_bf16_bits(x)).collect() }

/// The variants of one fixture (`chunks` = the chunkings to compare; the first one is the default).
pub fn variants(cfg: &Dsv41Config, p: usize, chunks: &[usize]) -> Result<Vec<Variant>> {
    let one = p.min(ONE_MAX);
    let ex = Schedule::new(cfg, p, Mode::Exact)?;
    let ced = Schedule::new(cfg, p, Mode::Ced)?;
    let mk = |name: String, phases: Vec<Phase>| Variant { name, phases, lane: Lane::default(), cands: Cands::default(), pad: true, replay: None };
    let tag = |c: usize| if c >= p { "one".to_string() } else { c.to_string() };
    let mut v = vec![mk(format!("exact@{}", tag(one)), vec![Phase { sched: ex.clone(), chunks: schedule::chunks(0, p, one), snap: None }]),
                     mk(format!("ced@{}", tag(one)), vec![Phase { sched: ced.clone(), chunks: schedule::chunks(0, p, one), snap: None }])];
    for &c in chunks { if c < p && c != one { v.push(mk(format!("ced@{c}"), vec![Phase { sched: ced.clone(), chunks: schedule::chunks(0, p, c), snap: None }])); } }
    let c0 = chunks.first().copied().unwrap_or(2048);
    if c0 < p { v.push(mk(format!("exact@{c0}"), vec![Phase { sched: ex.clone(), chunks: schedule::chunks(0, p, c0), snap: None }])); }
    // a 1-row last chunk (CED's layer-39 shape, a 1-row resume): every linear of it runs on padded rows
    if p > 2 {
        let mut cs = schedule::chunks(0, p - 1, c0.min(ONE_MAX));
        cs.push((p - 1, p));
        v.push(mk(format!("ced@{c0}+1"), vec![Phase { sched: ced.clone(), chunks: cs.clone(), snap: None }]));
        let mut np = mk(format!("ced@{c0}+1 unpadded (info)"), vec![Phase { sched: ced.clone(), chunks: cs, snap: None }]);
        np.pad = false;
        v.push(np);
    }
    // the opt-in truncated replay: 2,560 (DeepSeek's exact statement) and 140 (their recommended approximate one)
    for keep in [2560usize, 140] {
        if keep < p {
            let mut r = mk(format!("replay{keep}@{c0}"), vec![Phase { sched: Schedule::new(cfg, p, Mode::Replay { keep })?, chunks: schedule::chunks(0, p, c0.min(ONE_MAX)), snap: None }]);
            r.replay = Some(keep);
            v.push(r);
        }
    }
    // an encoder-only state at a 128 multiple at or below the decoder cone (and inside the prompt's first half)
    let k_enc = (ced.encoder_resume_cap().min(p / 2)) / 128 * 128;
    if k_enc >= 128 {
        let mut first = schedule::chunks(0, k_enc, c0);
        let rest = schedule::chunks(k_enc, p, c0);
        ensure!(ced.state_kind_at(k_enc) == StateKind::Encoder || p <= 2541, "a mid-prompt CED state at {k_enc} should be encoder-only");
        let kind = ced.state_kind_at(k_enc);
        if first.is_empty() { first.push((0, k_enc)); }
        let resumed = Schedule::resumed(cfg, p, Mode::Ced, k_enc, kind)?;
        v.push(mk(format!("ced@{c0}+enc@{k_enc}"), vec![Phase { sched: ced.clone(), chunks: first, snap: Some(kind) }, Phase { sched: resumed, chunks: rest, snap: None }]));
    }
    // a full state: the prompt's first K rows prefilled as their own prompt (CED), then the rest resumed
    let k_full = p.saturating_sub(schedule::MIN_KEEP) / 128 * 128;
    if k_full >= 128 {
        let pre = Schedule::new(cfg, k_full, Mode::Ced)?;
        ensure!(pre.state_kind_at(k_full) == StateKind::Full, "the prompt end is a full state");
        let resumed = Schedule::resumed(cfg, p, Mode::Ced, k_full, StateKind::Full)?;
        v.push(mk(format!("ced:pfx{k_full}+full"), vec![Phase { sched: pre, chunks: schedule::chunks(0, k_full, c0), snap: Some(StateKind::Full) },
                                                       Phase { sched: resumed, chunks: schedule::chunks(k_full, p, c0), snap: None }]));
    }
    Ok(v)
}

/// The attention side of layer `w.layer` for one variant over the dumped attn_in (all rows, device bf16).
fn run_variant(f: &mut Fwd, w: &LayerW, var: &mut Variant, ain_all: u64, keep_rows: &[usize], lin_cap: usize, max_seq: usize) -> Result<LayerOut> {
    let (l, d) = (w.layer, f.cfg.hidden);
    if !var.lane.layers.contains_key(&l) { var.lane.layers.insert(l, LayerLane::new(f.g, f.cfg, l, lin_cap, max_seq)?); }
    let wide = f.wide_min;
    if !var.pad { f.wide_min = 0; }
    let r = run_variant_inner(f, w, var, ain_all, keep_rows, l, d);
    f.wide_min = wide;
    r
}

fn run_variant_inner(f: &mut Fwd, w: &LayerW, var: &mut Variant, ain_all: u64, keep_rows: &[usize], l: usize, d: usize) -> Result<LayerOut> {
    let mut out = LayerOut::default();
    let index_src = f.cfg.index_source_of(l) == Some(l);
    for ph in 0..var.phases.len() {
        for ci in 0..var.phases[ph].chunks.len() {
            let (c0, c1) = var.phases[ph].chunks[ci];
            let cr = var.phases[ph].sched.chunk(l, c0, c1);
            if n_of(cr.a) == 0 && cr.kv.map_or(true, |k| n_of(k) == 0) { continue; }
            f.g.sync()?;
            f.b.reset();
            let ain = ain_all + (cr.a.0 * d * 2) as u64;
            let r = chunk::attn_chunk(f, w, &mut var.lane, &cr, ain, c0, &mut var.cands)?;
            let nq = n_of(r.q);
            if nq == 0 { continue; }
            let o = f.g.down::<u16>(r.out, nq * d)?;
            let sel = if index_src { Some(f.g.down::<i32>(r.sel.context("index source without a selection")?, nq * 512)?) } else { None };
            for i in 0..nq {
                let pos = r.q.0 + i;
                let row = &o[i * d..(i + 1) * d];
                out.rows.insert(pos, fnv(bytemuck_u16(row)));
                if keep_rows.binary_search(&pos).is_ok() {
                    out.keep.insert(pos, row.to_vec());
                    if let Some(s) = &sel { out.topk.insert(pos, s[i * 512..(i + 1) * 512].to_vec()); }
                }
            }
        }
        if let Some(kind) = var.phases[ph].snap {
            // snapshot this layer, poison it, restore it: the next phase must see exactly the saved state
            let mut one = Lane::default();
            let ll = var.lane.layers.remove(&l).context("layer state")?;
            one.layers.insert(l, ll);
            let snap = LaneSnap::take(f.g, &one, kind)?;
            one.poison(f.g, 0xA5)?;
            snap.restore(f.g, &mut one)?;
            let ll = one.layers.remove(&l).context("layer state")?;
            var.lane.layers.insert(l, ll);
        }
    }
    f.g.sync()?;
    let p = var.phases.last().map(|ph| ph.sched.p).unwrap_or(0);
    out.state = state_of(f.g, var.lane.get(l)?, p)?;
    Ok(out)
}

/// The decode state of a layer after a prefill of `p` rows: the ring slots of the last min(p, 128) positions,
/// the KV source's main KV / index-K planes / frontier ring.
pub fn state_of(g: &Gpu, ll: &LayerLane, p: usize) -> Result<Vec<(&'static str, Vec<u8>)>> {
    let slots = ll.win.ring_q.bytes / 512;
    let (rq, rs) = (g.down::<u8>(ll.win.ring_q.ptr(), slots * 512)?, g.down::<u8>(ll.win.ring_s.ptr(), slots * 16)?);
    let mut ring = Vec::new();
    for t in p.saturating_sub(128)..p {
        let s = t % slots;
        ring.extend_from_slice(&rq[s * 512..(s + 1) * 512]);
        ring.extend_from_slice(&rs[s * 16..(s + 1) * 16]);
    }
    let mut v = vec![("window_ring", ring)];
    if let Some(s) = &ll.src {
        let n = s.n_groups;
        v.push(("compress_kv", g.down::<u8>(s.kv_main.ptr(), n * 288)?));
        let mut ik = g.down::<u8>(s.iq.ptr(), n * 64)?;
        ik.extend(g.down::<u8>(s.is.ptr(), n * 4)?);
        v.push(("index_k", ik));
        if let Some((a, b)) = &s.front {
            let mut fr = g.down::<u8>(a.ptr(), a.bytes)?;
            fr.extend(g.down::<u8>(b.ptr(), b.bytes)?);
            v.push(("frontier", fr));
        }
    }
    Ok(v)
}

/// The probe's context for the per-layer gate.
pub struct Gate<'a> { pub f: Fwd<'a>, pub pass: Pass, pub rows: Vec<Row> }

impl<'a> Gate<'a> {
    pub fn push(&mut self, fx: &str, layer: usize, point: impl Into<String>, pass: Option<bool>, text: String) {
        self.rows.push(Row { fixture: fx.to_string(), layer, point: point.into(), pass, text });
    }
    pub fn push_cmp(&mut self, fx: &str, layer: usize, point: &'static str, c: Cmp, extra: String) {
        let t: Tol = cmp::tol(self.pass, point, layer);
        let pass = if t.rel.is_infinite() { None } else { Some(c.rel <= t.rel && c.same >= t.same) };
        let tol = if t.rel.is_infinite() { "-".to_string() } else { format!("rel<={} same>={}", t.rel, t.same) };
        self.push(fx, layer, format!("refb.{point}"), pass, format!("rel-L2 {:.3e} same {:.3}% max|d| {:.3e} n {} [{tol}] {extra}", c.rel, 100.0 * c.same, c.max_abs, c.n));
    }

    /// Every check of layer `w.layer` on one fixture: the variants against `exact@one`, then the default CED
    /// variant against REFb.
    pub fn layer_gate(&mut self, w: &LayerW, fx: &Fix, vars: &mut [Variant]) -> Result<()> {
        let (l, cfg) = (w.layer, self.f.cfg);
        let p = fx.p;
        let ain_ref = fx.d.layer(l, "attn_in")?;
        ensure!(ain_ref.n_rows() == p, "layer {l}: attn_in holds {} rows of {p} (an all-rows fixture is needed)", ain_ref.n_rows());
        let keep_rows: Vec<usize> = fx.d.layer(l, "x_in")?.positions().into_iter().map(|x| x as usize).collect();
        let ain_all = self.f.g.up(&bf16_bits(&ain_ref.f))?;
        let ain_ptr = crate::dsv41::fwd::dev::dp(&ain_all);
        let lin_cap = p.min(ONE_MAX);
        let mut outs: Vec<LayerOut> = Vec::new();
        let t0 = std::time::Instant::now();
        for v in vars.iter_mut() {
            outs.push(run_variant(&mut self.f, w, v, ain_ptr, &keep_rows, lin_cap, p).with_context(|| format!("variant {}", v.name))?);
        }
        let secs = t0.elapsed().as_secs_f64();
        // ---- bit identity against exact@one
        let reference = &outs[0];
        let ced_rows: Vec<usize> = (Schedule::new(cfg, p, Mode::Ced)?.layers[l].q0..p).collect();
        for (vi, o) in outs.iter().enumerate().skip(1) {
            let name = vars[vi].name.clone();
            let mut diff = 0usize;
            let mut common = 0usize;
            let mut first_bad = None;
            // a replay's rows below its exactness boundary are approximate by design: counted apart
            let bound = match vars[vi].replay { Some(k) if cfg.layer_role(l) != crate::dsv41::config::LayerRole::Encoder => replay_exact_from(p, k, 1, cfg.window), _ => 0 };
            let (mut approx, mut approx_diff) = (0usize, 0usize);
            for (pos, h) in &o.rows {
                if let Some(rh) = reference.rows.get(pos) {
                    if *pos < bound { approx += 1; if rh != h { approx_diff += 1; } continue; }
                    common += 1; if rh != h { diff += 1; first_bad.get_or_insert(*pos); }
                }
            }
            let missing = if vars[vi].replay.is_some() { 0 } else { ced_rows.iter().filter(|r| !o.rows.contains_key(r)).count() };
            let mut sdiff = Vec::new();
            for ((n, a), (_, b)) in o.state.iter().zip(reference.state.iter()) {
                if a != b { sdiff.push(format!("{n} ({} of {} bytes differ)", a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len()), b.len())); }
            }
            let ok = diff == 0 && missing == 0 && sdiff.is_empty() && common > 0;
            let gated = vars[vi].pad && vars[vi].replay.map_or(true, |k| k >= 2541);
            let rp = if vars[vi].replay.is_some() && bound > 0 { format!(" [replay: {approx} rows below {bound} with a truncated window, {approx_diff} of them differ (approximate by design)]") } else { String::new() };
            self.push(&fx.name, l, format!("bitwise {name}"), if gated { Some(ok) } else { None },
                      format!("{common} rows in common with exact@one, {diff} differ{}{rp}; {missing} CED rows missing; decode state {}",
                              first_bad.map(|b| format!(" (first {b})")).unwrap_or_default(),
                              if sdiff.is_empty() { format!("equal ({} buffers)", o.state.len()) } else { format!("DIFFERS: {}", sdiff.join(", ")) }));
        }
        // ---- the default CED variant (first chunking) against REFb
        let vi = vars.iter().position(|v| v.phases.len() == 1 && v.pad && v.replay.is_none() && v.name.starts_with("ced@") && !v.name.ends_with("one")).unwrap_or(1);
        let o = &outs[vi];
        let vname = vars[vi].name.clone();
        let aref = fx.d.layer(l, "attn_out")?;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let mut n = 0usize;
        for (pos, row) in &o.keep {
            let i = aref.index_of(*pos as i64).context("attn_out row")?;
            a.extend(row.iter().map(|&x| cmp::bf16_to_f32(x)));
            b.extend_from_slice(aref.row_f(i));
            n += 1;
        }
        let c = cmp::cmp(&a, &b);
        self.push_cmp(&fx.name, l, "attn_out", c, format!("{vname}: {n} dumped rows of the CED rows (own top-k)"));
        self.refb_state(fx, l, o, p, &vname)?;
        if !o.topk.is_empty() {
            if let Some(tk) = fx.d.layer_opt(l, "topk_compressed_pos")? {
                let pos: Vec<i64> = o.topk.keys().map(|&x| x as i64).collect();
                let ours: Vec<i32> = o.topk.values().flatten().copied().collect();
                let mut refk = Vec::new();
                for &q in &pos { let i = tk.index_of(q).context("topk row")?; refk.extend_from_slice(tk.row_i(i)); }
                let (same, ov, worst) = cmp::topk_agree(&ours, 512, &refk, tk.stride(), pos.len());
                let ok = ov >= cmp::topk_overlap_min(self.pass);
                self.push(&fx.name, l, "refb.topk", Some(ok), format!("{vname}: {same}/{} rows identical, mean overlap {ov:.5}, worst {worst:.4}", pos.len()));
            }
        }
        self.push(&fx.name, l, "time", None, format!("{} variants in {secs:.1} s", vars.len()));
        Ok(())
    }

    fn refb_state(&mut self, fx: &Fix, l: usize, o: &LayerOut, p: usize, vname: &str) -> Result<()> {
        for (name, bytes) in &o.state {
            match *name {
                "window_ring" => {
                    let Some(rr) = fx.d.layer_opt(l, "window_ring")? else { continue };
                    let n = p.min(128);
                    let mut ours = Vec::with_capacity(n * 512);
                    let mut refv = Vec::with_capacity(n * 512);
                    for (k, t) in (p - n..p).enumerate() {
                        let rec = &bytes[k * 528..(k + 1) * 528];
                        ours.extend(cmp::decode_fp8_rows(&rec[..512], &rec[512..], 1, 512));
                        refv.extend_from_slice(rr.row_f(t % 128));
                    }
                    self.push_cmp(&fx.name, l, "window_ring", cmp::cmp(&ours, &refv), format!("{vname}: last {n} positions"));
                }
                "compress_kv" => {
                    let Some(t) = fx.d.layer_opt(l, "compress_kv")? else { continue };
                    let ng = bytes.len() / 288;
                    let ours = cmp::decode_kv_records(bytes, ng);
                    self.push_cmp(&fx.name, l, "compress_kv", cmp::cmp(&ours, &t.f[..ng * 512]), format!("{vname}: {ng} groups (every row)"));
                }
                "index_k" => {
                    let Some(t) = fx.d.layer_opt(l, "index_k")? else { continue };
                    let ng = bytes.len() / 68;
                    let ours = cmp::decode_index_k(&bytes[..ng * 64], &bytes[ng * 64..], ng);
                    self.push_cmp(&fx.name, l, "index_k", cmp::cmp(&ours, &t.f[..ng * 128]), format!("{vname}: {ng} groups (every row)"));
                }
                "frontier" => {
                    let (Some(tk), Some(ts)) = (fx.d.layer_opt(l, "kv_state")?, fx.d.layer_opt(l, "score_state")?) else { continue };
                    let r = self.f.cfg.ratio(l) as usize;
                    let rr = self.f.cfg.frontier_rows(r as u32);
                    let rem = p % r;
                    if rem == 0 { continue; }
                    let f32s: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    let (kv, sc) = f32s.split_at(rr * 512);
                    let (mut a, mut b, mut c2, mut d2) = (vec![], vec![], vec![], vec![]);
                    for j in 0..rem {
                        let q = p - rem + j;
                        a.extend_from_slice(&kv[(q % rr) * 512..(q % rr + 1) * 512]);
                        c2.extend_from_slice(&sc[(q % rr) * 512..(q % rr + 1) * 512]);
                        b.extend_from_slice(tk.row_f(j));
                        d2.extend_from_slice(ts.row_f(j));
                    }
                    self.push_cmp(&fx.name, l, "kv_state", cmp::cmp(&a, &b), format!("{vname}: {rem} open rows"));
                    self.push_cmp(&fx.name, l, "score_state", cmp::cmp(&c2, &d2), format!("{vname}: {rem} open rows"));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Positions of a dumped per-row stream.
pub fn positions(t: &RefTensor) -> Vec<usize> { t.positions().into_iter().map(|x| x as usize).collect() }
