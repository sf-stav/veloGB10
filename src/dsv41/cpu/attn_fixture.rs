//! DSV4.1 attention oracle (leg OA): REFb's reference dumps, the per-point comparisons, and the
//! `--probe-dsv41-oracle --dsv41-oracle-part attn` driver.
//!
//! REFb's layout (tools/dsv41_ref/refdump.py, NOT I0's one-file-per-point `dump.rs` format): a
//! directory with `meta.json` (prompt length `P`, `rowsel`, and `tensors`: name -> file / dtype /
//! shape) and one safetensors per layer `L{LL}.safetensors` holding `L{LL}.{point}` plus
//! `L{LL}.{point}.rows` (absolute positions of the first dim). Attention points: `attn_in` (every
//! row), `attn_out` (the `rowsel` rows), `window_ring` [128, 512], `compress_kv`, `index_k`,
//! `kv_state` / `score_state` (ratio-2 sources), `topk_compressed_pos` (index sources: compressed
//! positions, -1 = none).

use super::layer::{wo_b_split, LayerIn, Oracle, OracleCfg, Order, SrcState, TopK};
use super::num::bf16_val;
use super::weights::{AttnWeights, Ckpt};
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::dump::RefbDump;
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

// ------------------------------------------------------------------------------------------ reader

/// One REFb dump directory (e.g. `/mnt/models/dsv41/ref/short_chat/prefill_exact`). A thin wrapper over the shared
/// reader `crate::dsv41::dump::RefbDump` (everything else, `has` / `raw` / `f32` / `ints` / `rows` / `meta` / `dir`,
/// is the shared reader's, through `Deref`); it keeps OA's `p()` (a Result) and `layers()` (layers with `attn_in`).
pub struct RefDump(RefbDump);

impl std::ops::Deref for RefDump {
    type Target = RefbDump;
    fn deref(&self) -> &RefbDump { &self.0 }
}

impl RefDump {
    /// Open a dump; a fixture root holding `prefill_exact/` is accepted too. OA needs the finished dump
    /// (`meta.json` written), as before.
    pub fn open(dir: &Path) -> Result<RefDump> {
        let d = RefbDump::open(dir)?;
        ensure!(d.meta.get("tensors").and_then(Value::as_object).is_some(), "{}: no tensors map (not a REFb dump?)", d.dir.join("meta.json").display());
        Ok(RefDump(d))
    }

    /// Prompt length.
    pub fn p(&self) -> Result<usize> { self.0.prompt_len() }

    /// Layers that have an `attn_in` dump.
    pub fn layers(&self) -> Vec<usize> { self.0.layers_with("attn_in") }
}

// ------------------------------------------------------------------------------------------ compare

/// Elementwise comparison of two value tensors.
#[derive(Clone, Debug, Default)]
pub struct Cmp {
    pub n: usize,
    /// identical values (f32 equality; -0 == +0; -inf == -inf)
    pub equal: usize,
    /// |mine - ref| / |ref| over the elements finite on both sides
    pub rel_l2: f64,
    pub max_abs: f64,
    /// elements finite on one side only
    pub finite_mismatch: usize,
}

impl Cmp {
    pub fn of(mine: &[f32], refv: &[f32]) -> Cmp {
        assert_eq!(mine.len(), refv.len(), "compare: lengths differ");
        let (mut num, mut den, mut c) = (0f64, 0f64, Cmp { n: mine.len(), ..Default::default() });
        for (&a, &b) in mine.iter().zip(refv) {
            if a == b || (a.is_nan() && b.is_nan()) {
                c.equal += 1;
            }
            if a.is_finite() != b.is_finite() {
                c.finite_mismatch += 1;
                continue;
            }
            if !a.is_finite() {
                continue;
            }
            let d = a as f64 - b as f64;
            num += d * d;
            den += (b as f64) * (b as f64);
            c.max_abs = c.max_abs.max(d.abs());
        }
        c.rel_l2 = if den > 0.0 { (num / den).sqrt() } else if num > 0.0 { f64::INFINITY } else { 0.0 };
        c
    }
    pub fn equal_frac(&self) -> f64 { if self.n == 0 { 1.0 } else { self.equal as f64 / self.n as f64 } }
    pub fn line(&self) -> String {
        format!("rel-L2 {:.2e}, equal {:.4}% ({} of {}), max |d| {:.3e}{}", self.rel_l2, 100.0 * self.equal_frac(), self.equal, self.n, self.max_abs,
                if self.finite_mismatch > 0 { format!(", {} FINITE MISMATCHES", self.finite_mismatch) } else { String::new() })
    }
}


/// A differing selection is a NEAR TIE when every position the reference picked and the oracle did
/// not lies within this many score-scale ulps below the oracle's cut (score-scale ulp = the bf16 ulp
/// of the row's largest |score|: the rounding step of the bf16 score chain at the row's scale).
/// Calibrated by the tolerance study: the oracle's recipe stays within 2 against REFb, the least
/// accurate valid order (naive sequential fp32 sums) reaches 6, the selection-moving mutation (no
/// YaRN) puts 846 agent_4k rows beyond it.
pub const NEAR_TIE_ULPS: f64 = 8.0;

/// Selection comparison (rows of compressed positions, -1 = none), against the oracle's own scores.
#[derive(Clone, Debug, Default)]
pub struct TopkCmp {
    pub rows: usize,
    pub identical: usize,
    /// sets differ only between positions whose oracle scores equal the cut (exact ties)
    pub up_to_ties: usize,
    /// sets differ, every reference-only pick within NEAR_TIE_ULPS score-scale ulps of the cut
    pub near_ties: usize,
    /// anything else (a reference pick far below the cut, unreachable, or a different count)
    pub beyond: usize,
    /// positions in the symmetric differences, all rows
    pub diff_positions: usize,
    /// the largest gap (cut - reference-only pick) over differing rows, in score-scale ulps
    pub max_gap_ulps: f64,
    /// differing rows explained by the candidate cut one level up: a reference pick inside a block
    /// the oracle's candidate set excluded, that block within NEAR_TIE_ULPS of the block cut
    pub near_ties_blocks: usize,
}

/// The candidate source's verdict for one query row (for near-tie analysis at consumers).
pub struct CandRow<'a> {
    pub keep: &'a [bool],
    pub block_scores: &'a [f32],
    pub block: usize,
}

/// Gap of `v` below `cut` in ulps of the bf16 step at `scale`.
fn gap_ulps(cut: f32, v: f32, scale: f32) -> f64 {
    if !(cut.is_finite() && v.is_finite()) {
        return f64::INFINITY;
    }
    let ulp = if scale > 0.0 { 2f64.powi(scale.log2().floor() as i32 - 7) } else { f64::MIN_POSITIVE };
    (cut as f64 - v as f64).max(0.0) / ulp
}

impl TopkCmp {
    pub fn add(&mut self, mine: &TopK, refrow: &[i64], cand: Option<CandRow>) {
        self.rows += 1;
        let a: std::collections::BTreeSet<i64> = mine.pos.iter().filter(|&&p| p >= 0).map(|&p| p as i64).collect();
        let b: std::collections::BTreeSet<i64> = refrow.iter().copied().filter(|&p| p >= 0).collect();
        if a == b {
            self.identical += 1;
            return;
        }
        self.diff_positions += a.symmetric_difference(&b).count();
        let sc = |p: i64| mine.scores.get(p as usize).copied().unwrap_or(f32::NEG_INFINITY);
        let cut = a.iter().map(|&p| sc(p)).fold(f32::INFINITY, f32::min);
        let scale = mine.scores.iter().filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()));
        let mut gap = 0f64;
        let mut block_gap = 0f64;
        for &p in b.difference(&a) {
            let s = sc(p);
            match (&cand, s == f32::NEG_INFINITY) {
                // masked by the oracle's candidate set: measure the block's gap at the block cut
                (Some(c), true) if !c.keep.get(p as usize / c.block).copied().unwrap_or(true) => {
                    let bcut = c.block_scores.iter().zip(c.keep).filter(|(v, k)| **k && v.is_finite()).map(|(v, _)| *v).fold(f32::INFINITY, f32::min);
                    let bscale = c.block_scores.iter().filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()));
                    block_gap = block_gap.max(gap_ulps(bcut, c.block_scores[p as usize / c.block], bscale));
                }
                _ => gap = gap.max(gap_ulps(cut, s, scale)),
            }
        }
        self.max_gap_ulps = self.max_gap_ulps.max(gap).max(block_gap);
        if a.len() != b.len() || gap > NEAR_TIE_ULPS || block_gap > NEAR_TIE_ULPS {
            self.beyond += 1;
        } else if block_gap > 0.0 || (block_gap == 0.0 && gap == 0.0 && b.difference(&a).any(|&p| sc(p) == f32::NEG_INFINITY)) {
            self.near_ties_blocks += 1;
        } else if gap == 0.0 {
            self.up_to_ties += 1;
        } else {
            self.near_ties += 1;
        }
    }
    pub fn line(&self) -> String {
        format!("{} rows: {} identical, {} equal up to exact ties, {} near ties (<= {} score-scale ulp), {} near ties at the candidate-block cut, {} beyond ({} positions differ; largest gap {:.2} score-scale ulp)",
                self.rows, self.identical, self.up_to_ties, self.near_ties, NEAR_TIE_ULPS, self.near_ties_blocks, self.beyond, self.diff_positions, self.max_gap_ulps)
    }
}

// ------------------------------------------------------------------------------------------ tolerances

/// The stated tolerances (report §2). A point passes when rel-L2 <= `rel_l2` AND the share of
/// identical elements >= `min_equal` AND no element is finite on one side only.
#[derive(Clone, Copy, Debug)]
pub struct Tol {
    pub rel_l2: f64,
    pub min_equal: f64,
}

/// Tolerance per dumped point (see the report for the derivation).
pub fn tol_of(point: &str) -> Tol {
    match point {
        // fp32 compressor state: fp32 matmul summation order only
        "kv_state" | "score_state" => Tol { rel_l2: 1e-5, min_equal: 0.0 },
        // FP8 (window) / FP4 (compressed, index) grid values: a few elements flip one grid step
        "window_ring" => Tol { rel_l2: 1e-2, min_equal: 0.99 },
        "compress_kv" | "index_k" => Tol { rel_l2: 1e-2, min_equal: 0.99 },
        // bf16 output of wo_b over an act_quant-ed input: one upstream FP8 / FP4 flip moves many
        // outputs by an ulp, so rel-L2 is the gate and the identical share only a floor
        "attn_out" => Tol { rel_l2: 1e-2, min_equal: 0.40 },
        _ => Tol { rel_l2: 1e-2, min_equal: 0.0 },
    }
}

fn verdict(point: &str, c: &Cmp) -> bool {
    let t = tol_of(point);
    c.rel_l2 <= t.rel_l2 && c.equal_frac() >= t.min_equal && c.finite_mismatch == 0
}

// ------------------------------------------------------------------------------------------ probe

/// One layer's results.
#[derive(Clone, Debug, Default)]
pub struct LayerReport {
    pub layer: usize,
    pub class: String,
    pub points: Vec<(String, Cmp, bool)>,
    pub topk: Option<TopkCmp>,
    pub extra: Vec<String>,
    pub ok: bool,
}

/// Seed a KV source's state from the dump (when the source layer is not part of the run).
fn seed_source(o: &mut Oracle, d: &RefDump, s: usize, p: usize) -> Result<()> {
    let ratio = o.cfg.ratio(s);
    let mut st = SrcState { ratio, seen: p, ..Default::default() };
    st.compress_kv = d.f32(&format!("L{s:02}.compress_kv"))?.1;
    if d.has(&format!("L{s:02}.index_k")) {
        st.index_k = d.f32(&format!("L{s:02}.index_k"))?.1;
    }
    if ratio > 1 {
        st.kv_state = d.f32(&format!("L{s:02}.kv_state"))?.1;
        st.score_state = d.f32(&format!("L{s:02}.score_state"))?.1;
    } else {
        st.kv_state = vec![0.0; o.cfg.head_dim];
        st.score_state = vec![f32::NEG_INFINITY; o.cfg.head_dim];
    }
    o.src.insert(s, st);
    Ok(())
}

/// The dumped selections of an index source, by query row.
fn dumped_topk(d: &RefDump, l: usize) -> Result<BTreeMap<usize, Vec<i32>>> {
    let name = format!("L{l:02}.topk_compressed_pos");
    let (sh, v) = d.ints(&name)?;
    let rows = d.rows(&name)?;
    let k = if sh.len() == 2 { sh[1] } else { 0 };
    ensure!(rows.len() * k == v.len(), "{name}: shape {sh:?} vs {} rows", rows.len());
    Ok(rows.iter().enumerate().map(|(i, &r)| (r, v[i * k..(i + 1) * k].iter().map(|&x| x as i32).collect())).collect())
}

/// A probe run: printable lines, the verdict, and the per-layer reports.
#[derive(Clone, Debug, Default)]
pub struct ProbeResult {
    pub lines: Vec<String>,
    pub ok: bool,
    pub reports: Vec<LayerReport>,
}

impl ProbeResult {
    /// Worst value of one point's metric over the layers (rel-L2: max; identical share: min).
    pub fn worst(&self, point: &str) -> Option<(f64, f64)> {
        let v: Vec<&Cmp> = self.reports.iter().flat_map(|r| r.points.iter().filter(|(n, _, _)| n == point).map(|(_, c, _)| c)).collect();
        if v.is_empty() { return None; }
        Some((v.iter().map(|c| c.rel_l2).fold(0.0, f64::max), v.iter().map(|c| c.equal_frac()).fold(1.0, f64::min)))
    }
}

/// Run the attention oracle over `layers` of a REFb dump and compare every attention point. Each
/// layer runs on its own dumped `attn_in` and the dumped cross-layer state (isolation); the
/// candidate blocks (not dumped) come from this run's candidate-source layer, itself run on its
/// dumped input, and only shape the oracle's own selections (attention uses the dumped top-k).
pub fn probe(model_dir: &Path, dump_dir: &Path, layers: Option<Vec<usize>>) -> Result<ProbeResult> {
    let t_all = Instant::now();
    let cfg = Dsv41Config::load(model_dir)?;
    let ck = Ckpt::open(model_dir)?;
    let d = RefDump::open(dump_dir)?;
    let p = d.p()?;
    let oc = OracleCfg::from_config(&cfg);
    let mut want: Vec<usize> = layers.unwrap_or_else(|| d.layers()).into_iter().filter(|&l| l < cfg.layers).collect();
    // a candidate consumer needs its candidate source run (the candidate blocks are not dumped)
    if let Some(cs) = oc.cand_source {
        if want.iter().any(|&l| oc.uses_candidates(l)) && !want.contains(&cs) {
            want.push(cs);
        }
    }
    want.sort_unstable();
    want.dedup();
    let mut lines = vec![format!("attention oracle: dump {} (P = {p}, kernels {}), weights {}, layers {:?}, {} threads",
                                 d.dir.display(), d.meta["kernels"].as_str().unwrap_or("?"), model_dir.display(), want, super::num::threads())];
    let mut o = Oracle::new(oc.clone());
    // a short prompt also gets the self-consistency checks: decode of the last row, and a 2-chunk prefill
    let small = p <= 512 && p >= 2;
    let mut o_dec = Oracle::new(oc.clone());
    let mut o_chunk = Oracle::new(oc.clone());
    let mut reports = Vec::new();
    for &l in &want {
        let t0 = Instant::now();
        let w = AttnWeights::load(&ck, &cfg, l)?;
        let t_load = t0.elapsed().as_secs_f64();
        let ll = format!("L{l:02}");
        let (_, x) = d.f32(&format!("{ll}.attn_in"))?;
        let xr = d.rows(&format!("{ll}.attn_in"))?;
        ensure!(xr.len() == p && xr.iter().enumerate().all(|(i, &r)| i == r), "{ll}.attn_in does not hold every row 0..{p} (not an exact-mode dump?)");
        let queries = d.rows(&format!("{ll}.attn_out"))?;
        // per-layer isolation (REFb §6.2: rounding-order noise compounds across layers, so a layer is
        // checked on ITS dumped inputs only): everything this layer reads from an earlier layer (the
        // KV source's compressed KV, partial group and index keys; the index source's top-k) is
        // taken from the dump, never from this run's own earlier layers
        if oc.ratio(l) > 0 {
            let s = oc.kv_source_of(l).context("compressing layer without a KV source")?;
            if s != l {
                seed_source(&mut o, &d, s, p)?;
            }
            let is = oc.index_source_of(l).context("compressing layer without an index source")?;
            if is != l {
                let m = dumped_topk(&d, is)?;
                o.topk.insert(is, m.into_iter().map(|(r, pos)| (r, TopK { pos, scores: Vec::new() })).collect());
            }
            // an index source above the candidate source owns no keys: they are the KV source's (dumped)
        }
        let ov = if oc.is_index_source(l) { Some(dumped_topk(&d, l)?) } else { None };
        let out = o.forward_layer(&w, LayerIn { layer: l, s0: 0, x: &x, queries: &queries, order: Order::Prefill { p_total: p }, topk_override: ov.as_ref() })?;
        let t_fwd = t0.elapsed().as_secs_f64() - t_load;
        let mut rep = LayerReport { layer: l, class: oc.class_name(l), ok: true, ..Default::default() };
        let mut point = |name: &str, mine: &[f32], refv: &[f32]| {
            let c = Cmp::of(mine, refv);
            let ok = verdict(name, &c);
            rep.ok &= ok;
            rep.points.push((name.to_string(), c, ok));
        };
        let (_, ao) = d.f32(&format!("{ll}.attn_out"))?;
        point("attn_out", &out.attn_out, &ao);
        if d.has(&format!("{ll}.window_ring")) {
            point("window_ring", &o.window_ring(l), &d.f32(&format!("{ll}.window_ring"))?.1);
        }
        if oc.is_kv_source(l) {
            let st = &o.src[&l];
            point("compress_kv", &st.compress_kv, &d.f32(&format!("{ll}.compress_kv"))?.1);
            if d.has(&format!("{ll}.index_k")) {
                point("index_k", &st.index_k, &d.f32(&format!("{ll}.index_k"))?.1);
            }
            if d.has(&format!("{ll}.kv_state")) {
                point("kv_state", &st.kv_state, &d.f32(&format!("{ll}.kv_state"))?.1);
                point("score_state", &st.score_state, &d.f32(&format!("{ll}.score_state"))?.1);
            }
        }
        if let Some(ov) = &ov {
            let mut tc = TopkCmp::default();
            for (q, t) in queries.iter().zip(&out.topk) {
                let r: Vec<i64> = ov[q].iter().map(|&v| v as i64).collect();
                let cand = if oc.uses_candidates(l) {
                    match (o.cand.get(q), o.cand_scores.get(q)) {
                        (Some(k), Some(bs)) => Some(CandRow { keep: k, block_scores: bs, block: oc.cand_block }),
                        _ => None,
                    }
                } else {
                    None
                };
                tc.add(t, &r, cand);
            }
            rep.ok &= tc.beyond == 0;
            rep.topk = Some(tc);
        }
        // TP: wo_b with its input split over 2 / 4 / 8 ranks (fp32 partials summed in rank order)
        let full = &out.attn_out;
        let mut tp = Vec::new();
        for ranks in [2usize, 4, 8] {
            if (w.wo_b.k / 32) % ranks == 0 {
                let s = wo_b_split(&w.wo_b, &out.wo_b_in, queries.len(), ranks);
                tp.push(format!("TP{ranks} {:.4}%", 100.0 * Cmp::of(&s, full).equal_frac()));
            }
        }
        rep.extra.push(format!("wo_b split by o_group, rank-order fp32 sum vs one pass, identical bf16 outputs: {}", tp.join(", ")));
        // self-consistency on short prompts
        if small {
            let last = p - 1;
            // decode: prefill rows [0, P-1) then one decode step at P-1
            let pre = o_dec.forward_layer(&w, LayerIn { layer: l, s0: 0, x: &x[..last * cfg.hidden], queries: &[], order: Order::Prefill { p_total: last }, topk_override: None });
            let dec = pre.and_then(|_| o_dec.forward_layer(&w, LayerIn { layer: l, s0: last, x: &x[last * cfg.hidden..], queries: &[last], order: Order::Decode, topk_override: None }));
            match dec {
                Ok(dec) => {
                    if let Some(qi) = queries.iter().position(|&q| q == last) {
                        let h = cfg.hidden;
                        let c = Cmp::of(&dec.attn_out, &ao[qi * h..(qi + 1) * h]);
                        let c2 = Cmp::of(&dec.attn_out, &out.attn_out[qi * h..(qi + 1) * h]);
                        rep.extra.push(format!("decode step at row {last} (prefill {last} rows + 1 decode) vs the dump's prefill row: {}", c.line()));
                        rep.extra.push(format!("    same decode step vs this run's prefill row (the reference's own decode/prefill difference: ring order regroups the softmax blocks): {}", c2.line()));
                    }
                }
                Err(e) => rep.extra.push(format!("decode check not run: {e:#}")),
            }
            // chunked prefill at an odd boundary, against this run's single-shot rows (bitwise)
            let cut = (p / 2) | 1;
            let a = o_chunk.forward_layer(&w, LayerIn { layer: l, s0: 0, x: &x[..cut * cfg.hidden], queries: &queries.iter().copied().filter(|&q| q < cut).collect::<Vec<_>>(), order: Order::Prefill { p_total: p }, topk_override: None });
            let b = a.and_then(|a| {
                let qb: Vec<usize> = queries.iter().copied().filter(|&q| q >= cut).collect();
                o_chunk.forward_layer(&w, LayerIn { layer: l, s0: cut, x: &x[cut * cfg.hidden..], queries: &qb, order: Order::Prefill { p_total: p }, topk_override: None })
                    .map(|b| (a, b))
            });
            match b {
                Ok((a, b)) => {
                    let mut joined = a.attn_out.clone();
                    joined.extend_from_slice(&b.attn_out);
                    // the single-shot run used the dumped selections at index sources; compare chunked
                    // against a single-shot run with the oracle's own selections where they differ
                    let c = Cmp::of(&joined, &out.attn_out);
                    let note = if oc.ratio(l) > 0 { " (index layers: chunked uses own selections, single-shot the dumped ones)" } else { "" };
                    rep.extra.push(format!("2-chunk prefill (cut {cut}) vs single-shot: {}{note}", c.line()));
                    let ring_eq = o_chunk.window_ring(l) == o.window_ring(l);
                    let st_eq = match (o_chunk.src.get(&l), o.src.get(&l)) {
                        (Some(a), Some(b)) => a.same_live_state(b, cfg.head_dim),
                        (None, None) => true,
                        _ => false,
                    };
                    rep.extra.push(format!("2-chunk prefill state (ring, compressed KV, index keys, partial group) bitwise equal to single-shot: {}", ring_eq && st_eq));
                    rep.ok &= ring_eq && st_eq;
                }
                Err(e) => rep.extra.push(format!("chunk check not run: {e:#}")),
            }
        }
        lines.push(format!("L{l:02} [{}] {} query rows, weights {:.1} s, forward {:.1} s: {}", rep.class, queries.len(), t_load, t_fwd, if rep.ok { "PASS" } else { "FAIL" }));
        for (n, c, ok) in &rep.points {
            lines.push(format!("    {n:12} {} {}", c.line(), if *ok { "" } else { "<-- OUT OF TOLERANCE" }));
        }
        if let Some(t) = &rep.topk {
            lines.push(format!("    {:12} {}", "topk", t.line()));
        }
        for e in &rep.extra {
            lines.push(format!("    {e}"));
        }
        reports.push(rep);
    }
    // summary by class
    let mut by: BTreeMap<String, (usize, usize, f64)> = BTreeMap::new();
    for r in &reports {
        let e = by.entry(r.class.clone()).or_default();
        e.0 += 1;
        e.1 += r.ok as usize;
        let ao = r.points.iter().find(|(n, _, _)| n == "attn_out").map(|(_, c, _)| c.rel_l2).unwrap_or(0.0);
        e.2 = e.2.max(ao);
    }
    lines.push("summary by layer class (layers passing / run, worst attn_out rel-L2):".into());
    for (k, (n, ok, w)) in &by {
        lines.push(format!("    {k:60} {ok} / {n}   {w:.2e}"));
    }
    let all_ok = reports.iter().all(|r| r.ok);
    lines.push(format!("total {:.0} s", t_all.elapsed().as_secs_f64()));
    Ok(ProbeResult { lines, ok: all_ok, reports })
}
