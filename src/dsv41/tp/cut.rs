//! DSV4.1 TP (leg SE2): top-k / top-p at TP, through DS's sampler. DS's kernels apply any per-row cut exactly (kept = a
//! prefix of the (logit desc, id asc) order, `Round::verify`'s `tcut`); this file computes the cut of every verify row on
//! TPC's byte-exact exchange:
//!
//! 1. one STATS pass over the target rows (no cut) gives every row's log-sum-exp over the whole vocabulary;
//! 2. each rank takes the top `C` candidates of its own shard of every cut row (host, from the logits it holds: unified
//!    memory, ~1 ms for 24 rows), the lists travel over the exchange (`TpXchg::all_gather`: exact bits), and every rank
//!    merges them to the global top `C` (a rank's top `C` always contains its share of the global top `C`);
//! 3. the cut is the last kept element of `kept_set`'s rule on that list: top-k first, then top-p over the renormalised
//!    top-k mass (or over the whole vocabulary's mass, from the log-sum-exp, when there is no top-k). A nucleus that does
//!    not fit in `C1` candidates is retried with `C2`; one that does not fit in `C2` is capped at `C2` tokens (counted).
//!
//! The drafts are drawn from the drafter's UNCUT distribution: speculative sampling is exact for any proposal
//! distribution (acceptance `min(1, q / p)` with the p the draft was drawn from, residual `max(q - alpha p, 0)`), so only
//! the target rows need a cut; a draft outside the target's kept set is rejected, which costs acceptance, not exactness.
//! Penalties are already in the logits (`pen.rs` runs first), as the host sampler applies them before its cut.

use super::spec::TpSpec;
use crate::dsv41::fwd::dev::Gpu;
use crate::dsv41::gpu::dspark::vocab::{Cut, Rows};
use crate::dsv41::gpu::dspark::xchg::VocabExchange;
use crate::dsv41::gpu::dspark::WIDE_RPL;
use anyhow::{ensure, Result};

/// Candidates per row and rank, first try / wide retry.
pub const C1: usize = 256;
pub const C2: usize = 2048;
/// The most rows a call cuts (up to MAX_LANES lanes x WIDE_RPL rows).
pub const MAX_CUT_ROWS: usize = super::fwd::MAX_LANES * WIDE_RPL;

/// A lane's cut parameters (`top_p` 1.0 and `top_k` 0 = none).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutCfg { pub top_k: usize, pub top_p: f32 }
impl Default for CutCfg { fn default() -> Self { CutCfg { top_k: 0, top_p: 1.0 } } }
impl CutCfg { pub fn active(&self) -> bool { self.top_k > 0 || self.top_p < 1.0 } }

pub const NO_CAND: (f32, i32) = (f32::NEG_INFINITY, i32::MAX);

/// `a` before `b` in the cut order (logit desc, id asc).
fn before(a: (f32, i32), b: (f32, i32)) -> bool { a.0 > b.0 || (a.0 == b.0 && a.1 < b.1) }
fn by(a: &(f32, i32), b: &(f32, i32)) -> std::cmp::Ordering { b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)) }

/// The top `c` entries of a shard row (ids `v0 + i`) in the cut order, padded to `c` with [`NO_CAND`]. NaN and -inf entries
/// never qualify (the kernels treat NaN as -inf and a -inf token is never kept).
pub fn shard_candidates(row: &[f32], v0: usize, c: usize) -> Vec<(f32, i32)> {
    let mut cand: Vec<(f32, i32)> = row.iter().enumerate().filter(|(_, a)| a.is_finite()).map(|(i, &a)| (a, (v0 + i) as i32)).collect();
    if cand.len() > c {
        cand.select_nth_unstable_by(c - 1, by);
        cand.truncate(c);
    }
    cand.sort_unstable_by(by);
    cand.resize(c, NO_CAND);
    cand
}

/// The global top `c` of the ranks' sorted lists (each padded with [`NO_CAND`]).
pub fn merge_lists(lists: &[Vec<(f32, i32)>], c: usize) -> Vec<(f32, i32)> {
    let mut idx = vec![0usize; lists.len()];
    let mut out = Vec::with_capacity(c);
    while out.len() < c {
        let mut best: Option<usize> = None;
        for (r, l) in lists.iter().enumerate() {
            let Some(&x) = l.get(idx[r]) else { continue };
            if x == NO_CAND { continue; }
            if best.map_or(true, |b| before(x, lists[b][idx[b]])) { best = Some(r); }
        }
        let Some(b) = best else { break };
        out.push(lists[b][idx[b]]);
        idx[b] += 1;
    }
    out
}

/// What a row's cut needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RowCut {
    /// the cut: kept iff `a > cut.0 || (a == cut.0 && id <= cut.1)`
    Cut((f32, i32)),
    /// no cut (every finite entry is kept)
    None,
    /// the candidates do not reach the nucleus (ask for more)
    NeedMore,
}

/// The cut of one row from the global top candidates `cand` (valid entries only, in the cut order). `truncated` = the list
/// was cut at its capacity (more finite entries exist); `lse` = the row's log-sum-exp of `a * inv_t` over the WHOLE vocabulary.
/// The rule is `serve::sampler::kept_set`'s: top-k, then top-p over the renormalised top-k mass.
pub fn row_cut(cand: &[(f32, i32)], truncated: bool, lse: f32, inv_t: f32, cfg: &CutCfg) -> RowCut {
    if cand.is_empty() || !cfg.active() { return RowCut::None; }
    let n = cand.len();
    let k = if cfg.top_k > 0 {
        if cfg.top_k > n && truncated { return RowCut::NeedMore; }
        cfg.top_k.min(n)
    } else { n };
    if cfg.top_p >= 1.0 { return if cfg.top_k > 0 && (k < n || truncated) { RowCut::Cut(cand[k - 1]) } else { RowCut::None }; }
    let a0 = cand[0].0 as f64;
    let w: Vec<f64> = cand[..k].iter().map(|c| ((c.0 as f64 - a0) * inv_t as f64).exp()).collect();
    let z = if cfg.top_k > 0 { w.iter().sum::<f64>() } else { (lse as f64 - a0 * inv_t as f64).exp() };
    let mut cum = 0.0;
    for (i, wi) in w.iter().enumerate() {
        cum += wi / z;
        if cum >= cfg.top_p as f64 { return RowCut::Cut(cand[i]); }
    }
    if cfg.top_k > 0 || !truncated { return RowCut::Cut(cand[k - 1]); }
    RowCut::NeedMore
}

impl TpSpec<'_> {
    /// The target cut of the verify rows `0..=host[l].k` of every active lane of `0..nl` that has a top-k / top-p (sampled
    /// lanes only), for rows laid out `[lane][row]` at `lane_stride` floats. None when no lane needs one.
    pub fn target_cut(&mut self, g: &Gpu, nl: usize, rpl: usize, lane_stride: usize) -> Result<Option<Cut>> {
        let want: Vec<usize> = (0..nl).filter(|&l| self.host[l].active != 0 && self.host[l].inv_t > 0.0 && self.cutcfg[l].active()).collect();
        if want.is_empty() { return Ok(None); }
        let (vl, v0) = (self.sh.vlocal, self.sh.v0);
        let rows_of = |l: usize| (self.host[l].k.max(0) as usize + 1).min(rpl);
        // 1. the rows' log-sum-exp over the whole vocabulary (no cut)
        let sres = self.round.stats(g, &self.x, self.sh, Rows { ptr: self.tlog.ptr(), lane_stride, row_stride: vl }, nl, rpl, None)?;
        let recs = self.round.recs(g, sres, nl * rpl)?;
        // 2. this rank's rows of the shard
        let mut rows: Vec<(usize, usize)> = Vec::new();
        for &l in &want { for j in 0..rows_of(l) { rows.push((l, j)); } }
        ensure!(rows.len() <= MAX_CUT_ROWS, "{} cut rows", rows.len());
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
        for &(l, j) in &rows {
            let row_off = if self.host[l].trow0_p1 > 0 {
                (self.host[l].trow0_p1 as usize - 1 + j) * vl
            } else {
                l * lane_stride + j * vl
            };
            let mut row = g.down::<f32>(self.tlog.ptr() + (row_off * 4) as u64, vl)?;
            if (j as i32) < self.host[l].suppress_eos {
                for &sid in &self.round.stop_ids {
                    if sid >= 0 && (sid as usize) >= v0 && (sid as usize) < v0 + vl {
                        row[sid as usize - v0] = f32::NEG_INFINITY;
                    }
                }
            }
            data.push(row);
        }
        let mut c = C1;
        let mut out: Vec<(usize, usize, RowCut)>;
        loop {
            // 3. candidates -> exchange -> merge -> cut
            let mine: Vec<Vec<(f32, i32)>> = std::thread::scope(|sc| {
                let hs: Vec<_> = data.chunks(rows.len().div_ceil(8).max(1)).map(|part| sc.spawn(move || part.iter().map(|r| shard_candidates(r, v0, c)).collect::<Vec<_>>())).collect();
                hs.into_iter().flat_map(|h| h.join().expect("candidate thread")).collect()
            });
            let world = self.x.ranks();
            let bytes = rows.len() * c * 8;
            ensure!(bytes <= self.cut_bytes, "{bytes} B of candidates exceed the exchange ({})", self.cut_bytes);
            let mut send = Vec::with_capacity(bytes);
            for r in &mine { for &(a, id) in r { send.extend_from_slice(&a.to_bits().to_le_bytes()); send.extend_from_slice(&id.to_le_bytes()); } }
            g.h2d(self.cut_send.ptr(), &send)?;
            self.cx.all_gather(g, self.cut_send.ptr(), bytes, self.cut_recv.ptr())?;
            let got = g.down::<u8>(self.cut_recv.ptr(), world * bytes)?;
            let rd = |off: usize| (f32::from_bits(u32::from_le_bytes(got[off..off + 4].try_into().unwrap())), i32::from_le_bytes(got[off + 4..off + 8].try_into().unwrap()));
            out = Vec::with_capacity(rows.len());
            let mut need_more = false;
            for (ri, &(l, j)) in rows.iter().enumerate() {
                let lists: Vec<Vec<(f32, i32)>> = (0..world).map(|r| (0..c).map(|i| rd(r * bytes + (ri * c + i) * 8)).collect()).collect();
                let merged = merge_lists(&lists, c);
                let lse = recs[l * rpl + j].lse();
                let mut rc = row_cut(&merged, merged.len() == c, lse, self.host[l].inv_t, &self.cutcfg[l]);
                if rc == RowCut::NeedMore {
                    if c < C2 { need_more = true; }
                    else if let Some(&last) = merged.last() {
                        // a nucleus wider than C2 tokens: the C2 best are kept (counted)
                        crate::dsv41::serve::metrics::inc(&crate::dsv41::serve::metrics::CUT_CAPPED);
                        rc = RowCut::Cut(last);
                    } else { rc = RowCut::None; }
                }
                out.push((l, j, rc));
            }
            if need_more { c = C2; crate::dsv41::serve::metrics::inc(&crate::dsv41::serve::metrics::CUT_WIDE); continue; }
            break;
        }
        // 4. the cut arrays (rows without a cut keep everything)
        ensure!(nl <= self.cap && rpl >= 1 && nl * rpl <= self.cap * WIDE_RPL, "cut array: {nl} lanes x {rpl} rows");
        let (mut ca, mut ci) = (vec![f32::NEG_INFINITY; nl * rpl], vec![i32::MAX; nl * rpl]);
        for &(l, j, rc) in &out {
            if let RowCut::Cut((a, id)) = rc { ca[l * rpl + j] = a; ci[l * rpl + j] = id; }
        }
        crate::dsv41::serve::metrics::add(&crate::dsv41::serve::metrics::CUT_ROWS, rows.len() as u64);
        g.h2d(self.cut_a.ptr(), crate::dsv41::gpu::dspark::vocab::as_bytes(&ca))?;
        g.h2d(self.cut_id.ptr(), crate::dsv41::gpu::dspark::vocab::as_bytes(&ci))?;
        Ok(Some(Cut { a: self.cut_a.ptr(), id: self.cut_id.ptr(), rpl }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::serve::sampler::{kept_set, Sampling};

    fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 33 }

    /// A vocabulary row with ties, a peaked head and a long tail.
    fn row(seed: u64, n: usize, flat: bool) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|i| {
            let r = (lcg(&mut s) % 4000) as f32 / 400.0 - 5.0;
            let v = if flat { r * 0.2 } else if i % 97 == 0 { r + 6.0 } else { r };
            (v * 8.0).round() / 8.0   // coarse grid: many exact ties
        }).collect()
    }

    /// The cut `kept_set` implies (the oracle `chisq::host_cut` uses).
    fn oracle(row: &[f32], s: &Sampling) -> Option<(f32, i32)> { crate::dsv41::gpu::dspark::chisq::host_cut(row, s) }

    fn lse_of(row: &[f32], inv_t: f32) -> f32 {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64 * inv_t as f64;
        (m + row.iter().map(|&a| (a as f64 * inv_t as f64 - m).exp()).sum::<f64>().ln()) as f32
    }

    /// The whole host pipeline over `ranks` shards equals the oracle on every kept-set boundary.
    fn pipeline(row: &[f32], ranks: usize, c: usize, s: &Sampling) -> RowCut {
        let per = row.len() / ranks;
        let lists: Vec<Vec<(f32, i32)>> = (0..ranks).map(|r| shard_candidates(&row[r * per..(r + 1) * per], r * per, c)).collect();
        let merged = merge_lists(&lists, c);
        let inv_t = 1.0 / s.temperature;
        row_cut(&merged, merged.len() == c, lse_of(row, inv_t), inv_t, &CutCfg { top_k: s.top_k, top_p: s.top_p })
    }

    #[test]
    fn the_candidate_pipeline_equals_kept_set_for_top_k_top_p_and_both() {
        let mut checked = 0;
        for (seed, flat) in [(1u64, false), (2, false), (3, true), (4, true), (5, false)] {
            let r = row(seed, 4096, flat);
            for (top_k, top_p, t) in [(40usize, 1.0f32, 1.0f32), (0, 0.9, 1.0), (0, 0.5, 0.7), (50, 0.9, 1.0), (1, 1.0, 1.0), (7, 0.2, 1.3), (0, 0.99, 1.0)] {
                let s = Sampling { temperature: t, top_p, top_k, ..Default::default() };
                for ranks in [1usize, 2, 4] {
                    let want = oracle(&r, &s);
                    match pipeline(&r, ranks, 4096, &s) {
                        RowCut::Cut(c) => { assert_eq!(Some(c), want, "seed {seed} flat {flat} k {top_k} p {top_p} T {t} ranks {ranks}"); checked += 1; }
                        RowCut::None => assert_eq!(want, None, "seed {seed} k {top_k} p {top_p}"),
                        RowCut::NeedMore => panic!("4096 candidates of a 4096 row must reach top_p {top_p} (seed {seed})"),
                    }
                }
            }
        }
        assert!(checked > 60);
    }

    #[test]
    fn a_nucleus_beyond_the_candidates_asks_for_more() {
        // 4096 equal logits: top_p 0.9 needs 3687 tokens
        let r = vec![1.0f32; 4096];
        let s = Sampling { top_p: 0.9, ..Default::default() };
        assert_eq!(pipeline(&r, 2, 256, &s), RowCut::NeedMore);
        match pipeline(&r, 2, 4096, &s) { RowCut::Cut(c) => assert_eq!(Some(c), oracle(&r, &s)), x => panic!("{x:?}") }
    }

    #[test]
    fn nan_and_neg_inf_never_qualify_and_the_lists_are_padded() {
        let r = [f32::NAN, 2.0, f32::NEG_INFINITY, 1.0, 2.0];
        let l = shard_candidates(&r, 10, 4);
        assert_eq!(l, vec![(2.0, 11), (2.0, 14), (1.0, 13), NO_CAND]);
        let m = merge_lists(&[l.clone(), shard_candidates(&[3.0], 100, 4)], 4);
        assert_eq!(m, vec![(3.0, 100), (2.0, 11), (2.0, 14), (1.0, 13)]);
    }
}
