//! `--probe-dsv41-dspark` (leg DS): DSpark on the GPU, one box.
//!
//! ```text
//! --probe-dsv41-dspark [--model-dir <pack>] [--dsv41-dump-ref <REFb fixture>[,...]] [--dsv41-kernel-fixture <OD trace dir>[,...]]
//!                      [--engram-dir DIR] [--dsv41-kernel-timing 1,2,4]
//! ```
//! 1. ABI: the band's struct sizes, device vs host.
//! 2. The sampler (no model): the chi-square cells of [`super::chisq`] (+ REFb's real target rows when a fixture is given).
//! 3. With `--model-dir` + fixtures: the drafter against REFb's drafts at every dumped decode step, fed the DUMPED main
//!    hidden (plan Revision 5): the rings seeded from the prompt's last rows' taps, each step's tap written, the block
//!    drafted (free greedy chain) and again with REFb's tokens as the Markov inputs (forced chain, so one flip does not
//!    move the later rows). Gate = leg OD's rule (`cpu::dspark::{RANK_TOL, GAP_TOL, SIG_TOL}`): a row's draft equals
//!    REFb's, or REFb's token ranks <= 2 here within 1.25 logits of the argmax; sigmoid(confidence) within 0.25. For an
//!    EXL3 pack the bounds are a-priori (`EXL3_FLOOR`, set before the first EXL3 run; pack quality is GA's NLL).
//! 4. With OD's traces: the head side per op, fed the trace's own inputs: hc_pre(hc_out, ffn_pre) vs `Lhead.x`; the
//!    head + Markov chain (forced with OD's ids) + confidence from `Lhead.x` vs OD's drafts and confidence.
//! 5. Graph capture: one draft step (main row + block + chain, sampled) captured and replayed, bitwise equal to eager.
//! 6. Engagement: one demo round's counters through the process `/metrics` body; the draft-token Engram prefetch
//!    (`--engram-dir`): reads issued, then none on a repeat. 7. Timing (`--dsv41-kernel-timing`): the draft step at
//!    1 / 2 / 4 lanes, eager and replayed, written as the policy's drafter costs (calibration cache round trip).
//! RESULT: DSV41_DSPARK_OK.

use super::drafter::Drafter;
use super::vocab::{self, Round, Rows, Shard};
use super::xchg::Local;
use super::{chisq, metrics, policy, Lane, BLOCK, IDS_ROWS};
use crate::dsv41::cpu::dspark::dspark_fwd::DraftOut;
use crate::dsv41::cpu::dspark::dspark_num::{sigmoid32, topk_desc};
use crate::dsv41::cpu::dspark::{self as od, Fixture, StepCmp};
use crate::dsv41::dump::{Dump, DumpLayer};
use crate::dsv41::fwd::dev::{Buf, Ev, Gpu};
use crate::dsv41::fwd::glue::Scr;
use crate::dsv41::fwd::layer::{Bump, Fwd};
use crate::dsv41::fwd::weights::Pack;
use crate::dsv41::gpu::mixers as km;
use crate::opts;
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::sys;
use std::path::{Path, PathBuf};

/// (rank, gap, sigmoid) bounds of a draft-row comparison.
#[derive(Clone, Copy, Debug)]
pub struct Floor { pub rank: usize, pub gap: f32, pub sig: f32, pub what: &'static str }
pub const NATIVE_FLOOR: Floor = Floor { rank: od::RANK_TOL, gap: od::GAP_TOL, sig: od::SIG_TOL, what: "OD's floor rule (the reference's own floor)" };
/// EXL3: a-priori (declared before the first EXL3 run): the 2.9-bpw pack's K4 stage experts and K6 head move the
/// logits by up to ~1-2 (SL: stage outputs at 7-14 % rel-L2), so a flip may rank up to 4 within 3 logits.
pub const EXL3_FLOOR: Floor = Floor { rank: 4, gap: 3.0, sig: 0.35, what: "a-priori EXL3 bounds (rank <= 4, gap <= 3.0, sigmoid <= 0.35)" };

fn flag_val(name: &str) -> Option<String> { opts::lookup(name).and_then(opts::get) }
fn say(lines: &mut Vec<String>, l: String) { println!("{l}"); lines.push(l); }
fn bits(v: &[f32]) -> Vec<u16> { v.iter().map(|x| (x.to_bits() >> 16) as u16).collect() }

/// The mode.
pub fn run(args: &[String]) -> Result<Vec<String>> {
    let mut lines = Vec::new();
    let mut fails: Vec<String> = Vec::new();
    let g = Gpu::new()?;
    // 1. ABI
    let out = g.buf(64)?;
    g.launch1("d41_ds_sizes", (1, 1, 1), (32, 1, 1), 0, &out.ptr())?;
    let dev: Vec<u32> = g.down::<u32>(out.ptr(), super::host_sizes().len())?;
    let host = super::host_sizes();
    let abi_ok = dev.iter().zip(host.iter()).all(|(a, b)| *a as usize == *b);
    say(&mut lines, format!("DSpark ABI: device struct sizes {dev:?}, host {host:?}: {}", if abi_ok { "equal" } else { "DIFFER" }));
    if !abi_ok { fails.push("ABI".into()); }
    let fixtures: Vec<PathBuf> = flag_val("dsv41-dump-ref").map(|s| s.split(',').filter(|x| !x.is_empty()).map(PathBuf::from).collect()).unwrap_or_default();
    // 2. the sampler
    let real = real_rows(&fixtures)?;
    chisq::run(&g, real, &mut lines, &mut fails)?;
    for l in lines.iter().skip(1) { println!("{l}"); }
    // 3-7. the drafter
    if let Some(md) = crate::arg_value(args, "--model-dir") {
        drafter_part(&g, args, Path::new(md), &fixtures, &mut lines, &mut fails)?;
    } else {
        say(&mut lines, "drafter: no --model-dir (sampler parts only)".into());
    }
    say(&mut lines, format!("DSpark probe: {} failure(s){}", fails.len(), if fails.is_empty() { String::new() } else { format!(": {}", fails.join("; ")) }));
    if fails.is_empty() { say(&mut lines, "DSV41_DSPARK_OK".into()); } else { bail!("{}", fails.join("; ")); }
    Ok(lines)
}

/// REFb's real target rows for the sampler gate: the 6 decode rows of highest entropy (at T = 1) over the given
/// fixtures' `G.decode_logits` (peaked rows put all mass on one token and test nothing).
fn real_rows(fixtures: &[PathBuf]) -> Result<Option<Vec<f32>>> {
    let mut rows: Vec<(f64, Vec<f32>)> = Vec::new();
    for d in fixtures {
        // the one REFb reader: a fixture directory or its prefill_exact variant, both work
        let Some(t) = crate::dsv41::dump::RefbDump::open(d)?.global_opt("decode_logits")? else { continue };
        let v = t.shape.get(1).copied().unwrap_or(0);
        if v < 1000 { continue; }
        for r in t.f.chunks_exact(v) {
            let m = r.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x)) as f64;
            let z: f64 = r.iter().map(|&x| (x as f64 - m).exp()).sum();
            let h: f64 = r.iter().map(|&x| { let p = (x as f64 - m).exp() / z; if p > 0.0 { -p * p.ln() } else { 0.0 } }).sum();
            rows.push((h, r.to_vec()));
        }
    }
    if rows.len() < 6 { return Ok(None); }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("sampler real rows: the 6 of highest entropy among {} REFb decode rows: {:?} nats", rows.len(), rows.iter().take(6).map(|r| (r.0 * 100.0).round() / 100.0).collect::<Vec<_>>());
    Ok(Some(rows.into_iter().take(6).flat_map(|r| r.1).collect()))
}

struct Ctx<'a> { g: &'a Gpu, dr: Drafter, rd: Round, x: Local, keep: Vec<Buf> }

impl Ctx<'_> {
    fn up<T: Copy>(&mut self, v: &[T]) -> Result<u64> {
        let b = self.g.buf(std::mem::size_of_val(v))?;
        self.g.h2d(b.ptr(), vocab::as_bytes(v))?;
        let p = b.ptr();
        self.keep.push(b);
        Ok(p)
    }
}

fn drafter_part(g: &Gpu, args: &[String], md: &Path, fixtures: &[PathBuf], lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    let t0 = std::time::Instant::now();
    let pack = Pack::open(md)?;
    let cfg = pack.c.cfg.clone();
    let floor = if pack.is_exl3() { EXL3_FLOOR } else { NATIVE_FLOOR };
    let sh = Shard { v0: 0, vlocal: cfg.vocab };
    let dr = Drafter::load(g, &pack, 4, 512, sh)?;
    say(lines, format!("drafter: pack {} ({}), stages {}..{}, head {} ({:?}), Markov rank {}, block {}, noise {}; loaded in {:.1} s; gate: {}",
                       md.display(), if pack.is_exl3() { "EXL3" } else { "native" }, cfg.layers, cfg.layers + 2, dr.head.w.0.key, dr.head.w.1, dr.rank, BLOCK,
                       cfg.dspark_noise_token, t0.elapsed().as_secs_f64(), floor.what));
    let rd = Round::new(g, 4, 6, sh.nblk(), 1)?;
    let mut c = Ctx { g, dr, rd, x: Local { max_bytes: 64 * 1024 }, keep: Vec::new() };
    let scr = g.buf(512 << 20)?;
    let mut f = Fwd { wide_min: 0, g, cfg: &cfg, scr: Scr { ptr: scr.ptr(), bytes: 512 << 20 }, b: Bump::new(g, 1 << 30)?, rope: crate::dsv41::fwd::rope_set(&cfg)?, checks: true };
    for d in fixtures { fixture_gate(&mut c, &mut f, d, floor, lines, fails).with_context(|| format!("fixture {}", d.display()))?; }
    if let Some(t) = flag_val("dsv41-kernel-fixture") {
        for d in t.split(',').filter(|x| !x.is_empty()) { head_ops(&mut c, &mut f, Path::new(d), floor, lines, fails).with_context(|| format!("trace {d}"))?; }
    }
    graph_check(&mut c, &mut f, lines, fails)?;
    {
        // leg PK5: the fused chain step / confidence head / one-pass plain GEMV against the sequences they replace
        use crate::dsv41::census::classify::TensorClass as C;
        let w2 = &c.dr.stages[2];
        let (cl, cf) = super::chain::selfcheck(g, &f.scr, w2.lin(C::DsMarkovEmbed)?, w2.lin(C::DsMarkovHead)?, w2.lin(C::DsConfidence)?, cfg.vocab, cfg.hidden)?;
        for l in cl { say(lines, l); }
        fails.extend(cf);
    }
    demo_round(&mut c, &mut f, lines, fails)?;
    if let Some(e) = crate::arg_value(args, "--engram-dir").map(PathBuf::from) { prefetch_check(g, md, &e, lines, fails)?; }
    if let Some(t) = flag_val("dsv41-kernel-timing") { timing(&mut c, &mut f, &t, md, lines)?; }
    for l in crate::dsv41::fwd::receipts::boot_lines() { say(lines, l); }
    Ok(())
}

/// The block for one lane at window end `p` with token `tok`; returns (ids [6], forced-chain DraftOut).
fn draft_step(c: &mut Ctx, f: &mut Fwd, p: usize, tok: u32, forced_ids: Option<&[u32]>, inv_t: f32, seed: u64) -> Result<(Vec<u32>, Option<DraftOut>, Vec<(u64, u64)>)> {
    let lane = Lane { seed, inv_t, vbase: p as i32 + 1, k: 0, wend: p as i32, tok: tok as i32, active: 1, ..Default::default() };
    c.rd.set_lanes(c.g, &[lane])?;
    let outs = c.dr.block(f, &c.rd, &c.x, 1, BLOCK, false)?;
    let free: Vec<u32> = c.rd.ids_table(c.g)?[..IDS_ROWS].iter().map(|&x| x as u32).collect();
    let Some(fi) = forced_ids else { return Ok((free, None, outs)) };
    let fids: Vec<i32> = fi.iter().map(|&x| x as i32).collect();
    c.g.h2d(c.rd.fids.ptr(), vocab::as_bytes(&fids))?;
    c.dr.head_and_chain(f, &c.rd, &c.x, 1, BLOCK, true)?;
    Ok((free, Some(read_out(c, fi)?), outs))
}

/// OD's trace of a fixture (stage outputs and window rings per step), for the drafter gate's per-stage diagnosis.
struct StageTrace { hc_out: Vec<Vec<f32>>, ring: Vec<Vec<f32>>, router: Vec<Vec<f32>> }
fn stage_trace(dir: &Path) -> Result<StageTrace> {
    let tr = Dump::open(dir)?;
    let mut t = StageTrace { hc_out: Vec::new(), ring: Vec::new(), router: Vec::new() };
    for l in 40u16..43 {
        t.hc_out.push(tr.read(DumpLayer::Layer(l), "hc_out")?.to_f32()?);
        t.ring.push(tr.read(DumpLayer::Layer(l), "attn_ring")?.to_f32()?);
        t.router.push(tr.read(DumpLayer::Layer(l), "router_ids")?.to_f32()?);
    }
    Ok(t)
}

/// Our stage outputs and rings at step `i` (window end `p`) against OD's trace: per stage (hc_out rel-L2, ring rows equal / rows,
/// per-row hc_out rel-L2, the rows whose top-k router set differs from OD's).
#[allow(clippy::type_complexity)]
fn stage_diag(c: &Ctx, outs: &[(u64, u64)], t: &StageTrace, i: usize, p: usize) -> Result<Vec<(f64, usize, usize, Vec<f64>, Vec<usize>)>> {
    let (d, win, hd) = (c.dr.dim, c.dr.cfg.window, c.dr.cfg.head_dim);
    let tk = c.dr.cfg.dspark_top_k;
    let mut v = Vec::new();
    for s in 0..3 {
        let n = BLOCK * 4 * d;
        let ours = c.g.down_bf16(outs[s].0, n)?;
        let mut cm = od::Cmp::default();
        cm.add(&ours, &t.hc_out[s][i * n..(i + 1) * n]);
        let per_row: Vec<f64> = (0..BLOCK).map(|r| { let mut x = od::Cmp::default(); x.add(&ours[r * 4 * d..(r + 1) * 4 * d], &t.hc_out[s][i * n + r * 4 * d..i * n + (r + 1) * 4 * d]); x.rel_l2() }).collect();
        let rid = c.g.down::<i32>(outs[s].1, BLOCK * tk)?;
        let flips: Vec<usize> = (0..BLOCK).filter(|&r| {
            let mut a: Vec<i64> = rid[r * tk..(r + 1) * tk].iter().map(|&x| x as i64).collect();
            let mut b: Vec<i64> = t.router[s][(i * BLOCK + r) * tk..(i * BLOCK + r + 1) * tk].iter().map(|&x| x as i64).collect();
            a.sort_unstable(); b.sort_unstable();
            a != b
        }).collect();
        let (rq, rs) = c.dr.ring(s, 0);
        let q = c.g.down::<u8>(rq, 134 * hd)?;
        let sc = c.g.down::<u8>(rs, 134 * hd / 32)?;
        let dec = crate::dsv41::fwd::cmp::decode_fp8_rows(&q, &sc, 134, hd);
        let (mut eq, mut tot) = (0usize, 0usize);
        for pos in p.saturating_sub(win - 1)..=p {
            let a = &dec[(pos % 134) * hd..(pos % 134 + 1) * hd];
            let b = &t.ring[s][(i * win + pos % win) * hd..(i * win + pos % win + 1) * hd];
            tot += 1;
            if a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) { eq += 1; }
        }
        v.push((cm.rel_l2(), eq, tot, per_row, flips));
    }
    Ok(v)
}

/// The forced chain's outputs as OD's `DraftOut` (probe = the reference's ids).
fn read_out(c: &Ctx, ref_ids: &[u32]) -> Result<DraftOut> {
    let v = c.dr.shard.vlocal;
    let lg = c.g.down::<f32>(c.dr.logits.ptr(), BLOCK * v)?;
    let conf = c.g.down::<f32>(c.dr.conf.ptr(), BLOCK)?;
    let picks: Vec<u32> = c.rd.ids_table(c.g)?[..IDS_ROWS].iter().map(|&x| x as u32).collect();
    let mut top8 = Vec::new();
    let mut margin = Vec::new();
    let mut probe = Vec::new();
    for i in 0..BLOCK {
        let l = &lg[i * v..(i + 1) * v];
        let t = topk_desc(l, 8);
        margin.push(l[t[0]] - l[t[1]]);
        let lt = l[ref_ids[i + 1] as usize];
        probe.push((l.iter().filter(|&&x| x > lt).count(), l[t[0]] - lt));
        top8.push(t.into_iter().map(|x| x as u32).collect());
    }
    let mut ids = vec![ref_ids[0]];
    ids.extend_from_slice(&picks[1..]);
    Ok(DraftOut { ids, confidence: conf, top8, margin, probe })
}

fn within(s: &StepCmp, fl: Floor) -> bool { s.flips().iter().all(|&(_, r, g)| r <= fl.rank && g <= fl.gap) && s.sig_diff() <= fl.sig }

fn fixture_gate(c: &mut Ctx, f: &mut Fwd, dir: &Path, fl: Floor, lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    let t0 = std::time::Instant::now();
    let fx = Fixture::open(dir)?;
    let name = fx.name();
    // OD's trace of the same fixture (when given): the per-stage diagnosis of every step
    let trace = flag_val("dsv41-kernel-fixture").and_then(|t| t.split(',').map(PathBuf::from).find(|p| p.file_name().map(|x| x.to_string_lossy() == name.as_str()).unwrap_or(false)));
    let st = match &trace { Some(p) => Some(stage_trace(p)?), None => None };
    let mut worst = [(0f64, usize::MAX, 0usize); 3];
    let mut rflips = [0usize; 3];
    let dmh = fx.decode_mh.as_ref().context("the fixture has no G.decode_main_hidden (REFb before e6701f81)")?;
    c.dr.reset_rings(c.g)?;
    // the prompt's last rows' taps into the rings (forward_spec at start_pos 0)
    let n = fx.prefill_mh.len();
    let taps: Vec<u16> = fx.prefill_mh.iter().flat_map(|(_, t)| bits(t)).collect();
    let pos: Vec<i32> = fx.prefill_mh.iter().map(|(p, _)| *p as i32).collect();
    let (tp, pp, lp) = (c.up(&taps)?, c.up(&pos)?, c.up(&vec![0i32; n])?);
    f.b.reset();
    c.dr.main_rows(f, tp, n, pp, lp)?;
    let mut steps = Vec::new();
    for (i, mh) in dmh.iter().enumerate().take(fx.drafts.len()) {
        let p = fx.p + i;
        let tok = fx.decode_tokens[i + 1];
        ensure!(fx.drafts[i][0] == tok, "step {i}: REFb's draft row starts with {} but the token is {tok}", fx.drafts[i][0]);
        f.b.reset();
        let (t1, p1, l1) = (c.up(&bits(mh))?, c.up(&[p as i32])?, c.up(&[0i32])?);
        c.dr.main_rows(f, t1, 1, p1, l1)?;
        let (free, forced, outs) = draft_step(c, f, p, tok, Some(&fx.drafts[i]), 0.0, 0)?;
        let diag = match &st { Some(t) => Some(stage_diag(c, &outs, t, i, p)?), None => None };
        if let Some(dg) = &diag {
            for (s, x) in dg.iter().enumerate() { worst[s].0 = worst[s].0.max(x.0); worst[s].1 = worst[s].1.min(x.1); worst[s].2 = worst[s].2.max(x.2); rflips[s] += x.4.len(); }
        }
        let free = DraftOut { ids: free, confidence: vec![], top8: vec![], margin: vec![], probe: vec![] };
        let sc = StepCmp { step: i, pos: p, ref_ids: fx.drafts[i].clone(), free, forced: forced.context("forced chain")?, ref_conf: fx.confidence[i].clone(), ref_top8: fx.top8[i].clone() };
        if !sc.drafts_equal() || !within(&sc, fl) {
            say(lines, format!("  {name} step {i:2} pos {p}: drafts {} (ref {:?}, gpu {:?}), flips (row, ref rank, gap) {:?} -> {}, sigmoid diff {:.3}{}",
                               if sc.drafts_equal() { "IDENTICAL" } else { "DIFFER" }, &sc.ref_ids[1..], &sc.free.ids[1..],
                               sc.flips().iter().map(|(j, r, g)| (*j, *r, (g * 1000.0).round() / 1000.0)).collect::<Vec<_>>(),
                               if within(&sc, fl) { "within" } else { "OUTSIDE" }, sc.sig_diff(),
                               diag.map(|d| format!("; vs OD's trace per stage (hc_out rel-L2 [per row], ring rows equal, rows whose router set differs): {:?}",
                                   d.iter().map(|x| (format!("{:.2e} {:?}", x.0, x.3.iter().map(|r| format!("{r:.1e}")).collect::<Vec<_>>()), format!("{}/{}", x.1, x.2), x.4.clone())).collect::<Vec<_>>())).unwrap_or_default()));
        }
        steps.push(sc);
    }
    let ident = steps.iter().filter(|s| s.drafts_equal()).count();
    let rows: usize = steps.len() * BLOCK;
    let flips: Vec<(usize, usize, f32)> = steps.iter().flat_map(|s| s.flips()).collect();
    let ok = steps.iter().all(|s| within(s, fl));
    let gap = flips.iter().map(|f| f.2).fold(0.0f32, f32::max);
    let sig = steps.iter().map(|s| s.sig_diff()).fold(0.0f32, f32::max);
    let t8 = steps.iter().map(|s| s.top8_same_set()).sum::<usize>();
    say(lines, format!("  {name:<10} P {:>6}: {} steps; drafts identical (free chain) {ident}/{}; forced-chain rows identical {}/{rows}; flips: max rank {} max gap {gap:.3}; \
                        sigmoid(confidence) max diff {sig:.4}; top-8 same set {t8}/{rows} -> {} ({:.1} s){}",
                       fx.p, steps.len(), steps.len(), rows - flips.len(), flips.iter().map(|f| f.1).max().unwrap_or(0), if ok { "PASS" } else { "FAIL" }, t0.elapsed().as_secs_f64(),
                       if st.is_some() { format!("; vs OD's trace, worst over steps: stage hc_out rel-L2 {:?}, window ring rows bitwise equal (min) {:?}, block rows whose router set differs per stage {:?} of {rows}",
                                                 worst.iter().map(|w| format!("{:.2e}", w.0)).collect::<Vec<_>>(), worst.iter().map(|w| format!("{}/{}", w.1, w.2)).collect::<Vec<_>>(), rflips) } else { String::new() }));
    if !ok { fails.push(format!("drafter vs REFb on {name}")); }
    Ok(())
}

/// The head side per op against OD's trace (fed the trace's own inputs).
fn head_ops(c: &mut Ctx, f: &mut Fwd, dir: &Path, fl: Floor, lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    let tr = Dump::open(dir)?;
    let name = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let d = c.dr.dim;
    let xh = tr.read(DumpLayer::Head, "x")?.to_f32()?;
    let ids = tr.read(DumpLayer::Layer(42), "draft_ids")?.to_f32()?;
    let conf = tr.read(DumpLayer::Layer(42), "confidence")?.to_f32()?;
    let hc_out = tr.read(DumpLayer::Layer(42), "hc_out")?.to_f32()?;
    let pre = tr.read(DumpLayer::Layer(42), "ffn_pre")?.to_f32()?;
    let rows = xh.len() / d;
    let steps = rows / BLOCK;
    ensure!(ids.len() == steps * IDS_ROWS && conf.len() == steps * BLOCK && pre.len() == rows * 4, "{name}: trace shapes");
    // (a) hc_pre(hc_out, ffn_pre) vs Lhead.x, in chunks of 20 rows (the drafter's buffers)
    let mut cm = od::Cmp::default();
    for r0 in (0..rows).step_by(c.dr.cap * BLOCK) {
        let n = (rows - r0).min(c.dr.cap * BLOCK);
        f.b.reset();
        let x = c.up(&bits(&hc_out[r0 * 4 * d..(r0 + n) * 4 * d]))?;
        let pp = c.up(&pre[r0 * 4..(r0 + n) * 4])?;
        let y = f.b.take(n * d * 2)?;
        let nw = c.dr.stages[2].param(crate::dsv41::census::classify::TensorClass::DsNorm)?;
        let a = km::HcPreNormArgs { x, x_ld: (4 * d) as i64, pre: pp, pre_stride: 4, w: nw.ptr.0, y_norm: 0, yn_ld: 0, y_raw: y, yr_ld: d as i64,
                                    w_fmt: crate::dsv41::fwd::layer::fmt_of(nw)?, dim: d as i32, rows: n as i32, eps: c.dr.cfg.rms_eps as f32, inv_dim: 1.0 / d as f32 };
        c.g.launch1("d41_hc_pre_norm", (n as u32, 1, 1), (256, 1, 1), 0, &a)?;
        cm.add(&c.g.down_bf16(y, n * d)?, &xh[r0 * d..(r0 + n) * d]);
        c.keep.clear();
    }
    // (b) head + Markov chain (forced with OD's ids) + confidence from Lhead.x
    let (mut flips, mut rows_eq, mut sig, mut cabs) = (Vec::new(), 0usize, 0f32, 0f32);
    for s in 0..steps {
        f.b.reset();
        c.dr.set_head_input(f, &bits(&xh[s * BLOCK * d..(s + 1) * BLOCK * d]), BLOCK)?;
        let rid: Vec<u32> = ids[s * IDS_ROWS..(s + 1) * IDS_ROWS].iter().map(|&x| x as u32).collect();
        c.rd.set_lanes(c.g, &[Lane { inv_t: 0.0, vbase: 1, wend: 0, tok: rid[0] as i32, active: 1, ..Default::default() }])?;
        let fids: Vec<i32> = rid.iter().map(|&x| x as i32).collect();
        c.g.h2d(c.rd.fids.ptr(), vocab::as_bytes(&fids))?;
        c.dr.head_and_chain(f, &c.rd, &c.x, 1, BLOCK, true)?;
        let o = read_out(c, &rid)?;
        for j in 0..BLOCK {
            if o.ids[j + 1] == rid[j + 1] { rows_eq += 1; } else { flips.push((s, j, o.probe[j].0, o.probe[j].1)); }
            let rc = conf[s * BLOCK + j];
            sig = sig.max((sigmoid32(o.confidence[j]) - sigmoid32(rc)).abs());
            cabs = cabs.max((o.confidence[j] - rc).abs());
        }
    }
    let ok_flips = flips.iter().all(|&(_, _, r, g)| r <= fl.rank && g <= fl.gap);
    let hc_ok = cm.rel_l2() <= 1e-3 && cm.equal as f64 >= 0.99 * cm.n as f64;
    let ok = ok_flips && sig <= fl.sig && hc_ok;
    say(lines, format!("  head ops {name:<10}: hc_pre(hc_out, ffn_pre) vs Lhead.x: {}; head + Markov (forced) + confidence from Lhead.x: rows equal {rows_eq}/{} \
                        (flips (step, row, rank, gap) {:?}), confidence max |diff| {cabs:.3e} logit, {sig:.2e} after the sigmoid -> {}",
                       cm.line(), steps * BLOCK, flips.iter().take(6).map(|&(s, j, r, g)| (s, j, r, (g * 1000.0).round() / 1000.0)).collect::<Vec<_>>(), if ok { "PASS" } else { "FAIL" }));
    if !ok { fails.push(format!("head ops on {name}")); }
    Ok(())
}

/// A CUDA graph of one draft step (main row + block + chain, sampled), replayed: bitwise equal to the eager step.
fn graph_check(c: &mut Ctx, f: &mut Fwd, lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    let (d3, p) = (3 * c.dr.dim, 200usize);
    let tap: Vec<u16> = (0..d3).map(|i| crate::dsv41::fwd::cmp::f32_to_bf16_bits(((i * 7919 % 1000) as f32 - 500.0) * 2e-3)).collect();
    let (tp, pp, lp) = (c.up(&tap)?, c.up(&[p as i32])?, c.up(&[0i32])?);
    let lane = Lane { seed: 0xD5EED, inv_t: 1.0, vbase: p as i32 + 1, k: 0, wend: p as i32, tok: 1000, active: 1, ..Default::default() };
    let v = c.dr.shard.vlocal;
    f.checks = false;
    let step = |c: &mut Ctx, f: &mut Fwd| -> Result<()> { f.b.reset(); c.dr.main_rows(f, tp, 1, pp, lp)?; c.dr.block(f, &c.rd, &c.x, 1, BLOCK, false).map(|_| ()) };
    c.rd.set_lanes(c.g, &[lane])?;
    step(c, f)?;
    let e_ids = c.rd.ids_table(c.g)?;
    let e_lg = c.g.down::<f32>(c.dr.logits.ptr(), BLOCK * v)?;
    let e_cf = c.g.down::<f32>(c.dr.conf.ptr(), BLOCK)?;
    c.rd.set_lanes(c.g, &[lane])?;
    c.g.sync()?;
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    let st = c.g.stream as sys::CUstream;
    ensure!(unsafe { sys::cuStreamBeginCapture_v2(st, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) } == sys::CUresult::CUDA_SUCCESS, "begin capture");
    let r = step(c, f);
    let e = unsafe { sys::cuStreamEndCapture(st, &mut graph) };
    r.context("capturing the draft step")?;
    ensure!(e == sys::CUresult::CUDA_SUCCESS, "end capture: {e:?}");
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    ensure!(unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) } == sys::CUresult::CUDA_SUCCESS, "instantiate");
    let mut nn = 0usize;
    unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut nn) };
    ensure!(unsafe { sys::cuGraphLaunch(exec, st) } == sys::CUresult::CUDA_SUCCESS, "graph launch");
    let g_ids = c.rd.ids_table(c.g)?;
    let g_lg = c.g.down::<f32>(c.dr.logits.ptr(), BLOCK * v)?;
    let g_cf = c.g.down::<f32>(c.dr.conf.ptr(), BLOCK)?;
    // timing: eager vs replay (one lane)
    let (e0, e1) = (Ev::new()?, Ev::new()?);
    let mut te = Vec::new();
    let mut tg = Vec::new();
    for _ in 0..8 {
        c.rd.set_lanes(c.g, &[lane])?;
        e0.rec(c.g)?; step(c, f)?; e1.rec(c.g)?; te.push(e1.ms_since(&e0)?);
        c.rd.set_lanes(c.g, &[lane])?;
        e0.rec(c.g)?;
        ensure!(unsafe { sys::cuGraphLaunch(exec, st) } == sys::CUresult::CUDA_SUCCESS, "graph launch");
        e1.rec(c.g)?; tg.push(e1.ms_since(&e0)?);
    }
    unsafe { sys::cuGraphExecDestroy(exec); sys::cuGraphDestroy(graph); }
    f.checks = true;
    te.sort_by(|a, b| a.total_cmp(b));
    tg.sort_by(|a, b| a.total_cmp(b));
    let same = e_ids[..IDS_ROWS] == g_ids[..IDS_ROWS] && e_lg.iter().zip(&g_lg).all(|(a, b)| a.to_bits() == b.to_bits()) && e_cf.iter().zip(&g_cf).all(|(a, b)| a.to_bits() == b.to_bits());
    say(lines, format!("  graph capture: one sampled draft step (1 main row + block + 5-step chain) = {nn} graph nodes; replay vs eager: drafts {:?} vs {:?}, logits and confidence {} \
                        -> {}; step median {:.3} ms eager, {:.3} ms replayed", &g_ids[1..IDS_ROWS], &e_ids[1..IDS_ROWS], if same { "bitwise equal" } else { "DIFFER" },
                       if same { "PASS" } else { "FAIL" }, te[te.len() / 2], tg[tg.len() / 2]));
    if !same { fails.push("graph capture".into()); }
    // the verify side of the round (statistics, accept, residual, finish) on target rows = the drafter's own rows with
    // two of them perturbed (a rejection and a residual draw happen), captured and replayed against the eager result
    let mut tl = c.g.down::<f32>(c.dr.logits.ptr(), BLOCK * v)?;
    tl.extend_from_slice(&tl[(BLOCK - 1) * v..].to_vec());
    for x in tl[v..2 * v].iter_mut() { *x *= 0.97; }
    for x in tl[2 * v..3 * v].iter_mut() { *x *= 0.5; }
    let td = c.up(&tl)?;
    let vl = Lane { seed: 0xA5A5, inv_t: 1.0, vbase: p as i32 + 1, k: BLOCK as i32, wend: p as i32, tok: 1000, active: 1, ..Default::default() };
    let tr = Rows { ptr: td, lane_stride: 6 * v, row_stride: v };
    let dr_ = Rows { ptr: c.dr.logits.ptr(), lane_stride: BLOCK * v, row_stride: v };
    c.rd.set_lanes(c.g, &[vl])?;
    c.rd.verify(c.g, &c.x, c.dr.shard, tr, dr_, 1, None, None, 0)?;
    let (e_out, e_ln) = (c.rd.out_tokens(c.g, 1)?[0], c.rd.get_lanes(c.g, 1)?[0]);
    c.rd.set_lanes(c.g, &[vl])?;
    c.g.sync()?;
    let mut vgraph: sys::CUgraph = std::ptr::null_mut();
    ensure!(unsafe { sys::cuStreamBeginCapture_v2(st, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) } == sys::CUresult::CUDA_SUCCESS, "begin capture (verify)");
    let r = c.rd.verify(c.g, &c.x, c.dr.shard, tr, dr_, 1, None, None, 0);
    let e = unsafe { sys::cuStreamEndCapture(st, &mut vgraph) };
    r.context("capturing the verify")?;
    ensure!(e == sys::CUresult::CUDA_SUCCESS, "end capture (verify): {e:?}");
    let mut vexec: sys::CUgraphExec = std::ptr::null_mut();
    ensure!(unsafe { sys::cuGraphInstantiateWithFlags(&mut vexec, vgraph, 0) } == sys::CUresult::CUDA_SUCCESS, "instantiate (verify)");
    let mut vn = 0usize;
    unsafe { sys::cuGraphGetNodes(vgraph, std::ptr::null_mut(), &mut vn) };
    ensure!(unsafe { sys::cuGraphLaunch(vexec, st) } == sys::CUresult::CUDA_SUCCESS, "graph launch (verify)");
    let (g_out, g_ln) = (c.rd.out_tokens(c.g, 1)?[0], c.rd.get_lanes(c.g, 1)?[0]);
    unsafe { sys::cuGraphExecDestroy(vexec); sys::cuGraphDestroy(vgraph); }
    let vsame = e_out == g_out && e_ln == g_ln;
    say(lines, format!("  graph capture: the verify (k 5, sampled: statistics, accept, residual, finish) = {vn} graph nodes; replay vs eager: accepted {} / {}, emitted {:?} / {:?}, \
                        lane state {} -> {}", g_ln.n_acc, e_ln.n_acc, &g_out[..g_ln.n_emit.max(0) as usize], &e_out[..e_ln.n_emit.max(0) as usize],
                       if e_ln == g_ln { "bitwise equal" } else { "DIFFERS" }, if vsame { "PASS" } else { "FAIL" }));
    if !vsame { fails.push("graph capture (verify)".into()); }
    Ok(())
}

/// One drafted + verified round on synthetic target rows (a 1-lane sampled round through the drafter's own drafts
/// and logits), its counters through the process `/metrics` body, and the once-per-boot line.
fn demo_round(c: &mut Ctx, f: &mut Fwd, lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    if let Some(l) = metrics::announce("dspark (3 stages, block 5, Markov rank 256)", "auto (measured round costs, --dsv41-draft-len)") { say(lines, l); }
    let v = c.dr.shard.vlocal;
    let p = 300usize;
    let k = BLOCK;
    let lane = Lane { seed: 4242, inv_t: 1.0, vbase: p as i32 + 1, k: k as i32, wend: p as i32, tok: 1000, active: 1, ..Default::default() };
    c.rd.set_lanes(c.g, &[lane])?;
    f.b.reset();
    c.dr.block(f, &c.rd, &c.x, 1, k, false)?;
    // the target rows: the drafter's own rows (so most drafts are accepted) with the last one perturbed
    let mut tl = c.g.down::<f32>(c.dr.logits.ptr(), BLOCK * v)?;
    tl.extend_from_slice(&tl[(BLOCK - 1) * v..].to_vec());
    for x in tl[3 * v..4 * v].iter_mut() { *x *= 0.5; }
    let tdev = c.up(&tl)?;
    c.rd.verify(c.g, &c.x, c.dr.shard, Rows { ptr: tdev, lane_stride: 6 * v, row_stride: v }, Rows { ptr: c.dr.logits.ptr(), lane_stride: BLOCK * v, row_stride: v }, 1, None, None, 0)?;
    let ln = c.rd.get_lanes(c.g, 1)?[0];
    let out = c.rd.out_tokens(c.g, 1)?[0];
    metrics::round(k, ln.n_acc as usize, true, ln.n_acc < k as i32, false);
    let body = crate::metrics::render(true);
    let ours: Vec<&str> = body.lines().filter(|l| l.starts_with("velogb10_dsv41_spec_")).collect();
    let ok = ln.n_emit == ln.n_acc + 1 && ln.wend == p as i32 + 1 + ln.n_acc && ours.iter().any(|l| l.starts_with("velogb10_dsv41_spec_rounds_total "));
    say(lines, format!("  demo round (sampled, 1 lane, k {k}): accepted {} of {k}, emitted {:?}; lane state wend {} tok {}; /metrics: {} velogb10_dsv41_spec_* series ({}) -> {}",
                       ln.n_acc, &out[..ln.n_emit as usize], ln.wend, ln.tok, ours.len(), ours.iter().take(3).copied().collect::<Vec<_>>().join("; "), if ok { "PASS" } else { "FAIL" }));
    if !ok { fails.push("demo round".into()); }
    Ok(())
}

/// The draft-token Engram prefetch through EIO's `warm` on this box's head files.
fn prefetch_check(g: &Gpu, md: &Path, edir: &Path, lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    use crate::dsv41::engram::{hash::EngramHasher, layout::{EngramConfig, EngramLayout}, prefetch::{EngramIo, EngramIoConfig}};
    let Some(h) = EngramHasher::from_model_dir(md)? else { say(lines, "  engram prefetch: the pack has no Engram".into()); return Ok(()) };
    let ec = EngramConfig::from_model_dir(md)?.context("Engram config")?;
    let layout = std::sync::Arc::new(EngramLayout::new(ec)?);
    let heads: Vec<usize> = (0..layout.n_heads()).filter(|hh| edir.join(format!("engram_head{hh}.bin")).exists()).collect();
    ensure!(!heads.is_empty(), "no engram_head*.bin in {}", edir.display());
    let mut c = EngramIoConfig::new(edir.to_path_buf(), heads.clone());
    c.cache_mib = 64;
    let mut io = EngramIo::open(&g.dev, layout, c)?;
    let st = h.state_at(0, &[], None)?;
    let tok = 1000u32;
    let drafts = [1001u32, 1002, 1003, 1004, 1005];
    let a = super::prefetch::warm_token(&mut io, &h, &st, tok)?;
    let b = super::prefetch::warm_drafts(&mut io, &h, &st, tok, &drafts)?;
    let t0 = std::time::Instant::now();
    while io.warm_pending() > 0 && t0.elapsed().as_secs() < 10 { io.housekeep(); std::thread::sleep(std::time::Duration::from_millis(2)); }
    let again = super::prefetch::warm_drafts(&mut io, &h, &st, tok, &drafts)?;
    let ok = a > 0 && b > 0 && again == 0;
    say(lines, format!("  engram draft prefetch (heads {heads:?}): the token's rows {a} read(s) when the drafter starts, the drafts' {b} more when they are known, \
                        a repeat {again} (resident or in flight) -> {}", if ok { "PASS" } else { "FAIL" }));
    if !ok { fails.push("engram draft prefetch".into()); }
    Ok(())
}

/// The draft step at 1 / 2 / 4 lanes (main rows = 6 per lane, the verify's taps; block; k = 5), eager and replayed,
/// then the policy's drafter costs written to and read back from the calibration cache.
fn timing(c: &mut Ctx, f: &mut Fwd, list: &str, md: &Path, lines: &mut Vec<String>) -> Result<()> {
    if list.split(',').any(|s| s.trim() == "budget") {
        // leg PK5: the per-kernel budget at TP slice shapes (rank 0 of each world; `super::tpbench`)
        let worlds: Vec<usize> = list.split(',').filter_map(|s| s.trim().strip_prefix("tp").and_then(|n| n.parse().ok())).collect();
        let lanes: Vec<usize> = list.split(',').filter_map(|s| s.trim().parse().ok()).filter(|&n| (1..=4).contains(&n)).collect();
        let worlds = if worlds.is_empty() { vec![1, 2, 4] } else { worlds };
        let lanes = if lanes.is_empty() { vec![1, 2, 4] } else { lanes };
        return super::tpbench::run(c.g, md, &worlds, &lanes, &mut |l| say(lines, l));
    }
    let lanes_list: Vec<usize> = list.split(',').filter_map(|s| s.trim().parse().ok()).filter(|&n| n >= 1 && n <= c.dr.cap).collect();
    let d3 = 3 * c.dr.dim;
    let v = c.dr.shard.vlocal;
    let mut costs = policy::Costs::default();
    say(lines, "  draft step timing (one box, TP1, whole-vocab head and Markov head, host launches; main rows = 6 per lane; median of 8 after a warm-up):".into());
    f.checks = false;
    let med = |mut x: Vec<f32>| -> f64 { x.sort_by(|a, b| a.total_cmp(b)); x[x.len() / 2] as f64 };
    for &nl in &lanes_list {
        let rows = 6 * nl;
        let tap: Vec<u16> = (0..rows * d3).map(|i| crate::dsv41::fwd::cmp::f32_to_bf16_bits(((i * 7919 % 1000) as f32 - 500.0) * 2e-3)).collect();
        let pos: Vec<i32> = (0..rows).map(|r| 500 + (r % 6) as i32).collect();
        let ln: Vec<i32> = (0..rows).map(|r| (r / 6) as i32).collect();
        let (tp, pp, lp) = (c.up(&tap)?, c.up(&pos)?, c.up(&ln)?);
        let lanes: Vec<Lane> = (0..nl).map(|l| Lane { seed: 7 + l as u64, inv_t: 1.0, vbase: 501, k: 5, wend: 505, tok: 1000 + l as i32, active: 1, ..Default::default() }).collect();
        let (e0, e1, e2, e3) = (Ev::new()?, Ev::new()?, Ev::new()?, Ev::new()?);
        // phases: main rows | stages + head (block with k = 0) | head + confidence alone | head + 5-step chain
        let (mut tm, mut tb0, mut th0, mut th5) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for it in 0..9 {
            c.rd.set_lanes(c.g, &lanes)?;
            f.b.reset();
            e0.rec(c.g)?;
            c.dr.main_rows(f, tp, rows, pp, lp)?;
            e1.rec(c.g)?;
            c.dr.block(f, &c.rd, &c.x, nl, 0, false)?;
            e2.rec(c.g)?;
            c.dr.head_and_chain(f, &c.rd, &c.x, nl, 0, false)?;
            e3.rec(c.g)?;
            if it > 0 { tm.push(e1.ms_since(&e0)?); tb0.push(e2.ms_since(&e1)?); th0.push(e3.ms_since(&e2)?); }
            e0.rec(c.g)?;
            c.dr.head_and_chain(f, &c.rd, &c.x, nl, 5, false)?;
            e1.rec(c.g)?;
            if it > 0 { th5.push(e1.ms_since(&e0)?); }
        }
        let (m, b0, h0, h5) = (med(tm), med(tb0), med(th0), med(th5));
        let stages = b0 - h0;
        let chain_step = (h5 - h0) / 5.0;
        for k in 1..=5usize { costs.observe_draft(nl, k, m + stages + h0 + k as f64 * chain_step); }
        say(lines, format!("    lanes {nl}: main rows ({rows}) {m:.3} ms | 3 stages on {} block rows {stages:.3} ms | head + confidence {h0:.3} ms | Markov chain {chain_step:.3} ms per draft \
                            => draft step k=5 {:.3} ms, k=2 {:.3} ms", nl * BLOCK, m + stages + h0 + 5.0 * chain_step, m + stages + h0 + 2.0 * chain_step));
        // the verify's sampling side alone: target rows (k + 1 = 6 per lane) and the draft rows, full vocabulary
        let tl: Vec<f32> = (0..nl * 6 * v).map(|i| ((i * 2_654_435_761usize % 10_007) as f32) * 1e-3).collect();
        let td = c.up(&tl)?;
        let mut tv = Vec::new();
        for it in 0..9 {
            c.rd.set_lanes(c.g, &lanes)?;
            e0.rec(c.g)?;
            c.rd.verify(c.g, &c.x, c.dr.shard, Rows { ptr: td, lane_stride: 6 * v, row_stride: v }, Rows { ptr: c.dr.logits.ptr(), lane_stride: BLOCK * v, row_stride: v }, nl, None, None, 0)?;
            e1.rec(c.g)?;
            if it > 0 { tv.push(e1.ms_since(&e0)?); }
        }
        say(lines, format!("    lanes {nl}: verify sampling ({} target rows x {v} ids: statistics, accept, residual) {:.3} ms", nl * 6, med(tv)));
        c.keep.clear();
    }
    f.checks = true;
    if let Some(path) = policy::calib_path(md) {
        let stamp = policy::calib_stamp(&policy::pack_fingerprint(md), 1, 1);
        policy::write_calib(&path, &stamp, &costs)?;
        let back = policy::read_calib(&path, &stamp);
        say(lines, format!("  drafter costs -> calibration cache {} ({}; the verify costs come from the TP forward's calibration)", path.display(),
                           if back.as_ref() == Some(&costs) { "read back equal" } else { "READ BACK DIFFERS" }));
    }
    Ok(())
}
