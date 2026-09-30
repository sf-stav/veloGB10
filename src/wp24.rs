//! WP24 — exact speculative sampling with STOCHASTIC drafts (real-q), host side.
//!
//! The device half lives in kernels/exl3_bench.cu (`xq_rq_draft`, `xq_sample_rows_rq`) and is
//! wired into the EXL3 MTP round behind `--spec-sampling ratio` (default `match` = today's
//! kernels and graphs exactly). This module holds:
//! * bit-exact host replicas of the device RNG (`u24`) and float-order replicas of the draft
//!   proposal q' (`qprime`) and the target law p (`target_law`, xq_sample_rows' semantics);
//! * rung 0 — the teacher-forced JSONL dump (`--wp24-dump=<path>`): per verify row the
//!   target's processed top-32 (after T / top-k / min-p / top-p and the device penalties) and
//!   the draft head's top-32, analysed offline by `scripts/wp24_accept.py` (A0 vs A_tau);
//! * the real-q round cross-check (`--wp24-xcheck=1`): every verify row of a served real-q
//!   round re-derived on the host (draft draw, q', accept test, residual support, bonus row);
//! * the distribution gate `--probe-spec-sampling`, two stages:
//!   A. kernel level (no model, ~1-3 min): reference-free two-sample chi-square of the real-q
//!      emission (xq_rq_draft + xq_sample_rows_rq, one verify row) vs the plain sampler on
//!      synthetic rows shaped like the served pair (+ a plain-vs-plain control, a negative
//!      control, the analytic acceptance, determinism and bonus-row identity);
//!   B. served level (`--model-dir <pack>`, exl3_serve::wp24_served_gate): the REAL scheduler
//!      (admit -> prefix-cache restore -> seam step -> prime -> DDS rounds with the HOST
//!      speculative pass, device re-prime, graphs) runs many seeded requests of one prompt in
//!      three arms — ratio MTP, match MTP (the control) and plain sampled decode (MTP off, the
//!      reference) — and compares the per-position histograms of the first N generated tokens
//!      (+ the joint of positions 2-3) with the same two-sample chi-square. This covers what
//!      stage A cannot: rows >= 1, list/counter routing per pass, DDS early stop, the bonus row
//!      after w < k.
//!
//! w5 port note (DHEAD line): a real-q draft pass runs the FULL pruned-slice head (the old tail:
//! exl3_chain -> [xq_pen_draft] -> xq_rq_draft -> xq_rowmax_f16), because DHEAD's screen keeps
//! only the blocks that can hold the ARGMAX; q' needs the exact top-K. The dump rung keeps the
//! served DHEAD/DHEADP draft and recomputes the slice logits only for its list.
//! A5-L4 (spec.ratio_dhead=1, opt-in): unpenalized real-q passes run DHEAD + xq_rq_draft_dh, q' over
//! the rescored candidate set (see exl3_forward/dhead.rs); the list row format, the verify kernel
//! and the host replicas here are unchanged — XCHECK reads the same list rows.

use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::f16;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Candidate list width (xq_rq_draft XQ_RQ_K) and the u32 words per list row (ids | logits | q').
pub const RQ_K: usize = 32;
pub const RQ_ROW: usize = 96;
/// xq_rq_draft stage-1 geometry (XQ_RQ_NB_MAX blocks per row, XQ_RQ_SLICE fp16 per block).
pub const RQ_NB_MAX: usize = 32;
pub const RQ_SLICE: usize = 8192;
/// RNG domains (XQ_RQ_DOM_*); domain 0 = xq_sample_rows' own draw.
pub const DOM_ACC: u32 = 1;
pub const DOM_RES: u32 = 2;
pub const DOM_DRAFT: u32 = 3;

/// xq_rq_draft blocks per row: ~4K logits per block, at most RQ_NB_MAX.
pub fn draft_nb(v: usize) -> usize {
    v.div_ceil(4096).clamp(1, RQ_NB_MAX)
}

/// The kernel's per-block slice (ceil(v / nb) rounded up to 8) fits its shared-memory stage.
pub fn draft_fits(v: usize) -> bool {
    let nb = draft_nb(v);
    v > 0 && ((v.div_ceil(nb) + 7) & !7) <= RQ_SLICE
}

/// --wp24-dump=<path>: append the rung-0 JSONL rows there (diagnostics; off when unset/empty).
pub fn dump_path() -> Option<&'static str> {
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| crate::opts::var(crate::opt!("wp24-dump")).ok().filter(|s| !s.is_empty())).as_deref()
}

pub fn dump_on() -> bool {
    dump_path().is_some()
}

/// --wp24-xcheck=1: re-derive every real-q verify row on the host (diagnostics; slow).
pub fn xcheck_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("wp24-xcheck")).as_deref() == Ok("1"))
}

// ---------------------------------------------------------------------------------------------
// device-math replicas

/// xq_mix64 (splitmix64 finaliser).
pub fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xq_rq_u24 bit for bit: a uniform in (0, 1) from (seed, counter, domain). Domain 0 is
/// xq_sample_rows' draw `(mix64(seed ^ mix64(ctr + 1)) >> 40 + 0.5) / 2^24`.
pub fn u24(seed: u64, ctr: u32, dom: u32) -> f32 {
    let h = mix64(seed ^ mix64(((dom as u64) << 32).wrapping_add(ctr as u64).wrapping_add(1)));
    ((h >> 40) as f32 + 0.5) * (1.0 / 16_777_216.0)
}

/// xq_skey: order-preserving 16-bit key of an fp16 bit pattern.
pub fn skey(b: u16) -> u32 {
    if b & 0x8000 != 0 { (!(b as u32)) & 0xFFFF } else { (b as u32) | 0x8000 }
}

/// xq_skey_val: the fp16 value behind a key.
pub fn skey_val(k: u32) -> f32 {
    let b = if k & 0x8000 != 0 { (k & 0x7FFF) as u16 } else { ((!k) & 0xFFFF) as u16 };
    f16::from_bits(b).to_f32()
}

/// The draft proposal q' over one pass's sorted candidate list (xq_rq_draft mode 2): the
/// REQUEST's top-k (first `top_k` entries), tau-softmax against list[0], the smallest prefix
/// whose mass reaches top_p, renormalised. Invalid entries (id < 0) trail and get 0. f64 here,
/// f32 on the device: compare with a tolerance.
pub fn qprime(ids: &[i32], lg: &[f32], tau: f32, top_k: u32, top_p: f32) -> Vec<f64> {
    let n = ids.len().min(lg.len());
    let mut q = vec![0f64; n];
    if n == 0 {
        return q;
    }
    if ids[0] < 0 {
        q[0] = 1.0; // the device's no-candidate delta (it rewrites id 0 there)
        return q;
    }
    let invt = 1.0 / tau.max(1e-6) as f64;
    let l0 = lg[0] as f64;
    let mut w: Vec<f64> = (0..n)
        .map(|j| {
            if ids[j] >= 0 && (top_k == 0 || (j as u32) < top_k) { ((lg[j] as f64 - l0) * invt).exp() } else { 0.0 }
        })
        .collect();
    let tot: f64 = w.iter().sum();
    if top_p < 1.0 && tot > 0.0 {
        let target = top_p as f64 * tot;
        let mut excl = 0.0;
        for x in w.iter_mut() {
            let wj = *x;
            if !(excl < target) {
                *x = 0.0;
            }
            excl += wj;
        }
    }
    let z: f64 = w.iter().sum();
    if !(z > 0.0 && z.is_finite()) {
        q[0] = 1.0;
        return q;
    }
    for j in 0..n {
        q[j] = w[j] / z;
    }
    q
}

/// True when q''s top-p cut is within float rounding of flipping an entry: the device decides
/// `excl_j < top_p * tot` in f32 (a warp scan over __expf weights), `qprime` in f64, so an entry
/// whose exclusive prefix mass sits within 1e-5 * tot of the cut can land on either side — a
/// q' difference there is a boundary tie ("near"), not a defect (review note on w2/WP24).
pub fn qprime_near(ids: &[i32], lg: &[f32], tau: f32, top_k: u32, top_p: f32) -> bool {
    let n = ids.len().min(lg.len());
    if n == 0 || ids[0] < 0 || !(top_p < 1.0) {
        return false;
    }
    let invt = 1.0 / tau.max(1e-6) as f64;
    let l0 = lg[0] as f64;
    let w: Vec<f64> = (0..n)
        .map(|j| if ids[j] >= 0 && (top_k == 0 || (j as u32) < top_k) { ((lg[j] as f64 - l0) * invt).exp() } else { 0.0 })
        .collect();
    let tot: f64 = w.iter().sum();
    let cut = top_p as f64 * tot;
    let mut excl = 0.0;
    for &wj in &w {
        if wj > 0.0 && (excl - cut).abs() <= 1e-5 * tot {
            return true;
        }
        excl += wj;
    }
    false
}

/// The target law of one fp16 logits row under xq_sample_rows' semantics (keys -> top-k with
/// ties -> min-p floor -> top-p over key-mass buckets with boundary ties -> softmax / T), dense.
pub fn target_law(lg: &[u16], t: f32, top_k: u32, top_p: f32, min_p: f32) -> Vec<f64> {
    let v = lg.len();
    if v == 0 {
        return Vec::new();
    }
    let keys: Vec<u32> = lg.iter().map(|&b| skey(b)).collect();
    let kmax = keys.iter().copied().max().unwrap_or(0);
    let lmax = skey_val(kmax);
    let tau = if top_k > 0 && (top_k as usize) < v {
        let mut ks = keys.clone();
        *ks.select_nth_unstable_by(top_k as usize - 1, |a, b| b.cmp(a)).1
    } else {
        0
    };
    let t = t.max(1e-6);
    let mfloor = if min_p > 0.0 { lmax + t * min_p.ln() } else { f32::NEG_INFINITY };
    let val = |i: usize| skey_val(keys[i]);
    let cand = |i: usize| keys[i] >= tau && val(i) >= mfloor;
    let w = |i: usize| (((val(i) - lmax) / t) as f64).exp();
    let s: f64 = (0..v).filter(|&i| cand(i)).map(w).sum();
    let mut thr = tau;
    if top_p < 1.0 && s > 0.0 {
        let mut by: std::collections::BTreeMap<u32, f64> = Default::default();
        for i in 0..v {
            if cand(i) {
                *by.entry(keys[i]).or_insert(0.0) += w(i);
            }
        }
        let target = top_p as f64 * s;
        let mut cum = 0.0;
        for (&k, &m) in by.iter().rev() {
            cum += m;
            if cum >= target {
                thr = thr.max(k);
                break;
            }
        }
    }
    let mut p: Vec<f64> = (0..v).map(|i| if cand(i) && keys[i] >= thr { w(i) } else { 0.0 }).collect();
    let z: f64 = p.iter().sum();
    if z > 0.0 {
        for x in p.iter_mut() {
            *x /= z;
        }
    }
    p
}

/// One list row of the device buffer: (ids, logits, q').
pub fn list_row(list: &[u32], i: usize) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    let r = &list[i * RQ_ROW..(i + 1) * RQ_ROW];
    (
        r[..RQ_K].iter().map(|&x| x as i32).collect(),
        r[RQ_K..2 * RQ_K].iter().map(|&x| f32::from_bits(x)).collect(),
        r[2 * RQ_K..].iter().map(|&x| f32::from_bits(x)).collect(),
    )
}

/// One sampler row (sc.samp [8] u32: flag, T, top_p, top_k, seed_lo, seed_hi, ctr, min_p).
#[derive(Clone, Copy, Debug)]
pub struct RowParams {
    pub flag: u32,
    pub t: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub seed: u64,
    pub ctr: u32,
    pub min_p: f32,
}

pub fn row_params(samp: &[u32], r: usize) -> RowParams {
    let p = &samp[r * 8..r * 8 + 8];
    RowParams {
        flag: p[0],
        t: f32::from_bits(p[1]),
        top_p: f32::from_bits(p[2]),
        top_k: p[3],
        seed: ((p[5] as u64) << 32) | p[4] as u64,
        ctr: p[6],
        min_p: f32::from_bits(p[7]),
    }
}

/// A completed verify, as read back by `FwdModel::wp24_after_verify`: w drafts, m = w + 1 rows.
pub struct Round<'a> {
    pub slot: usize,
    pub pos: usize,
    pub w: usize,
    pub a: usize,
    pub mode: u8,
    pub v: usize,
    pub logits: &'a [u16],
    pub emitted: &'a [i32],
    pub samp: &'a [u32],
    pub list: &'a [u32],
    pub drafts: &'a [i32],
    pub dsamp: [u32; 8],
    pub dctr: u32,
}

// ---------------------------------------------------------------------------------------------
// rung 0: teacher-forced dump

/// Append one JSONL line per sampled verify row i < w: the target's processed top-32 (p desc,
/// id asc), p at every draft candidate (exact: q_tau lives inside the draft's top-32), and the
/// draft's top-32 (raw logits, after the WP15 draft mirror when penalties are live). `on` =
/// the row's context is the committed trajectory (i <= a: every earlier draft was accepted).
pub fn dump_round(r: &Round) -> Result<()> {
    use std::io::Write;
    let path = match dump_path() {
        Some(p) => p,
        None => return Ok(()),
    };
    let mut out = String::new();
    for i in 0..r.w {
        let pp = row_params(r.samp, i);
        if pp.flag == 0 {
            continue; // greedy row: p is a delta, nothing to analyse
        }
        let row = &r.logits[i * r.v..(i + 1) * r.v];
        let p = target_law(row, pp.t, pp.top_k, pp.top_p, pp.min_p);
        let mut sup: Vec<(u32, f64)> =
            p.iter().enumerate().filter(|(_, &x)| x > 0.0).map(|(j, &x)| (j as u32, x)).collect();
        let p_n = sup.len();
        let by = |a: &(u32, f64), b: &(u32, f64)| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0));
        if sup.len() > RQ_K {
            sup.select_nth_unstable_by(RQ_K - 1, by);
            sup.truncate(RQ_K);
        }
        sup.sort_by(by);
        let (ids, lg, _) = list_row(r.list, i);
        let nq = ids.iter().take_while(|&&x| x >= 0).count();
        let p_at_q: Vec<f64> = ids[..nq].iter().map(|&id| if (id as usize) < r.v { p[id as usize] } else { 0.0 }).collect();
        let line = serde_json::json!({
            "slot": r.slot, "pos": r.pos, "row": i, "w": r.w, "a": r.a, "on": i <= r.a, "mode": r.mode,
            "T": pp.t, "top_p": pp.top_p, "top_k": pp.top_k, "min_p": pp.min_p,
            "d": r.drafts.get(i).copied().unwrap_or(-1), "e": r.emitted.get(i).copied().unwrap_or(-1),
            "p_n": p_n,
            "p_ids": sup.iter().map(|x| x.0).collect::<Vec<_>>(),
            "p": sup.iter().map(|x| x.1).collect::<Vec<_>>(),
            "q_ids": &ids[..nq],
            "q_lg": &lg[..nq],
            "p_at_q": p_at_q,
        });
        out.push_str(&line.to_string());
        out.push('\n');
    }
    if !out.is_empty() {
        static LOCK: Mutex<()> = Mutex::new(());
        let _g = LOCK.lock().unwrap();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)
            .with_context(|| format!("--wp24-dump: open {path}"))?;
        f.write_all(out.as_bytes())?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// real-q round cross-check

#[derive(Default, Debug)]
struct Xc {
    rounds: u64,
    rows: u64,
    acc: u64,
    rej: u64,
    list_miss: u64,
    draft_mism: u64,
    draft_near: u64,
    q_maxdiff: f64,
    q_near: u64,
    acc_mism: u64,
    acc_near: u64,
    resid_bad: u64,
    bonus_bad: u64,
}

/// Re-derive one real-q round on the host. Hard mismatches (counted, printed at once):
/// a draft outside its own list (list_miss), a draft that is not the device draw's replica
/// (draft_mism), an accept decision that contradicts u * q'(d) < p(d) (acc_mism), a rejected
/// row that emitted outside supp(max(0, p - q')) or the draft itself (resid_bad), a bonus id
/// outside supp(p) (bonus_bad). `near` = within float rounding of the decision boundary.
pub fn xcheck_round(r: &Round) {
    static XC: OnceLock<Mutex<Xc>> = OnceLock::new();
    let tau = f32::from_bits(r.dsamp[1]);
    let (top_p_d, top_k_d) = (f32::from_bits(r.dsamp[2]), r.dsamp[3]);
    let seed_d = ((r.dsamp[5] as u64) << 32) | r.dsamp[4] as u64;
    let mut hard = Vec::<String>::new();
    let mut xc = XC.get_or_init(|| Mutex::new(Xc::default())).lock().unwrap();
    xc.rounds += 1;
    for i in 0..r.w {
        let pp = row_params(r.samp, i);
        if pp.flag != 2 {
            continue;
        }
        xc.rows += 1;
        let (ids, lg, qd) = list_row(r.list, i);
        let qh = qprime(&ids, &lg, tau, top_k_d, top_p_d);
        let diff = qh.iter().zip(qd.iter()).map(|(a, &b)| (a - b as f64).abs()).fold(0.0, f64::max);
        if diff > 1e-4 && qprime_near(&ids, &lg, tau, top_k_d, top_p_d) {
            xc.q_near += 1; // a top-p boundary tie (f32 device cut vs f64 replica), not a defect
        } else if diff > xc.q_maxdiff {
            xc.q_maxdiff = diff;
        }
        let d = r.drafts[i];
        // the draft draw: inverse CDF over the device q' in list order
        let u = u24(seed_d, r.dctr.wrapping_add(i as u32), DOM_DRAFT) as f64;
        let zq: f64 = qd.iter().filter(|&&x| x > 0.0).map(|&x| x as f64).sum();
        let (mut cum, mut exp, mut near) = (0.0f64, None, false);
        for j in 0..RQ_K {
            if qd[j] > 0.0 {
                let c2 = cum + qd[j] as f64 / zq;
                if exp.is_none() && c2 > u {
                    exp = Some(ids[j]);
                    near = (c2 - u).abs() < 1e-5 || (u - cum).abs() < 1e-5;
                }
                cum = c2;
            }
        }
        if exp != Some(d) {
            if near { xc.draft_near += 1; } else {
                xc.draft_mism += 1;
                hard.push(format!("row {i}: draft {d} but the replica draws {exp:?} (u {u:.6})"));
            }
        }
        let jd = (0..RQ_K).find(|&j| ids[j] == d && qd[j] > 0.0);
        let q_d = match jd {
            Some(j) => qd[j] as f64,
            None => {
                xc.list_miss += 1;
                hard.push(format!("row {i}: draft {d} outside its pass list"));
                1.0
            }
        };
        let p = target_law(&r.logits[i * r.v..(i + 1) * r.v], pp.t, pp.top_k, pp.top_p, pp.min_p);
        let pd = if d >= 0 && (d as usize) < r.v { p[d as usize] } else { 0.0 };
        let ua = u24(pp.seed, pp.ctr, DOM_ACC) as f64;
        let exp_acc = ua * q_d < pd;
        let e = r.emitted[i];
        let dev_acc = e == d;
        if exp_acc != dev_acc {
            if (ua * q_d - pd).abs() <= 1e-4 * q_d.max(pd).max(1e-6) { xc.acc_near += 1; } else {
                xc.acc_mism += 1;
                hard.push(format!("row {i}: accept {dev_acc} vs replica {exp_acc} (u {ua:.6} q' {q_d:.6} p {pd:.6})"));
            }
        }
        if dev_acc {
            xc.acc += 1;
        } else {
            xc.rej += 1;
            let pe = if e >= 0 && (e as usize) < r.v { p[e as usize] } else { 0.0 };
            let qe = (0..RQ_K).find(|&j| ids[j] == e).map_or(0.0, |j| qd[j] as f64);
            if pe <= 0.0 || (qe > 0.0 && pe < qe * (1.0 - 1e-4)) {
                xc.resid_bad += 1;
                hard.push(format!("row {i}: rejected {d} -> {e} outside max(0, p - q') (p {pe:.3e} q' {qe:.3e})"));
            }
        }
    }
    let pb = row_params(r.samp, r.w);
    if pb.flag != 0 {
        let p = target_law(&r.logits[r.w * r.v..(r.w + 1) * r.v], pb.t, pb.top_k, pb.top_p, pb.min_p);
        let e = r.emitted[r.w];
        if !(e >= 0 && (e as usize) < r.v && p[e as usize] > 0.0) {
            xc.bonus_bad += 1;
            hard.push(format!("bonus row {}: id {e} outside supp(p)", r.w));
        }
    }
    if !hard.is_empty() || xc.rounds % 32 == 0 {
        println!("WP24_XCHECK rounds={} rows={} acc={} rej={} list_miss={} draft_mism={} (near {}) q_maxdiff={:.2e} \
                  (top-p ties {}) accept_mism={} (near {}) resid_bad={} bonus_bad={}{}",
                 xc.rounds, xc.rows, xc.acc, xc.rej, xc.list_miss, xc.draft_mism, xc.draft_near, xc.q_maxdiff,
                 xc.q_near, xc.acc_mism, xc.acc_near, xc.resid_bad, xc.bonus_bad,
                 if hard.is_empty() { String::new() } else { format!(" | slot {} pos {}: {}", r.slot, r.pos, hard.join("; ")) });
    }
}

// ---------------------------------------------------------------------------------------------
// the distribution gate

/// Two-sample chi-square of homogeneity (reference-free): bins with expected >= 5 in both arms,
/// the rest pooled into one tail bin; Wilson-Hilferty z. Returns (chi2/df, z, bins).
pub fn gof2(a: &[u64], b: &[u64]) -> (f64, f64, usize) {
    let na: u64 = a.iter().sum();
    let nb: u64 = b.iter().sum();
    let (naf, nbf) = (na as f64, nb as f64);
    let tot = naf + nbf;
    if na == 0 || nb == 0 {
        return (0.0, 0.0, 0);
    }
    let mut bins: Vec<(f64, f64)> = Vec::new();
    let (mut ta, mut tb) = (0.0f64, 0.0f64);
    for t in 0..a.len().min(b.len()) {
        let (oa, ob) = (a[t] as f64, b[t] as f64);
        let pooled = oa + ob;
        if pooled == 0.0 {
            continue;
        }
        if naf * pooled / tot >= 5.0 && nbf * pooled / tot >= 5.0 {
            bins.push((oa, ob));
        } else {
            ta += oa;
            tb += ob;
        }
    }
    let tp = ta + tb;
    if naf * tp / tot >= 5.0 && nbf * tp / tot >= 5.0 {
        bins.push((ta, tb));
    }
    let mut chi2 = 0.0;
    for &(oa, ob) in &bins {
        let pooled = oa + ob;
        let (ea, eb) = (naf * pooled / tot, nbf * pooled / tot);
        chi2 += (oa - ea) * (oa - ea) / ea + (ob - eb) * (ob - eb) / eb;
    }
    let df = bins.len().saturating_sub(1);
    if df == 0 {
        return (0.0, 0.0, bins.len());
    }
    let dff = df as f64;
    let z = ((chi2 / dff).cbrt() - (1.0 - 2.0 / (9.0 * dff))) / (2.0 / (9.0 * dff)).sqrt();
    (chi2 / dff, z, bins.len())
}

/// `gof2` over two sparse histograms (key -> count), aligned on the union of their keys.
pub fn gof2_maps(a: &HashMap<u64, u64>, b: &HashMap<u64, u64>) -> (f64, f64, usize) {
    let mut keys: Vec<u64> = a.keys().chain(b.keys()).copied().collect();
    keys.sort_unstable();
    keys.dedup();
    let va: Vec<u64> = keys.iter().map(|k| a.get(k).copied().unwrap_or(0)).collect();
    let vb: Vec<u64> = keys.iter().map(|k| b.get(k).copied().unwrap_or(0)).collect();
    gof2(&va, &vb)
}

fn parse<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok())
}

/// Synthetic (draft, target) pair shaped like the served one: the target over the full vocab
/// (Gaussian bulk sd 2 + a 40-token head in [6, 11], some of it outside the pruned draft slice),
/// the draft over the leading `vd` ids = the target's logits perturbed (head noise sd 1, a
/// draft-only favourite the target does not share) — so q' != p in both directions.
fn synthetic_rows(v: usize, vd: usize) -> (Vec<u16>, Vec<u16>) {
    let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        (st >> 11) as f64 / (1u64 << 53) as f64
    };
    let gauss = |sd: f64, rnd: &mut dyn FnMut() -> f64| {
        let (u1, u2) = (rnd().max(1e-12), rnd());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * sd
    };
    let mut tgt: Vec<f64> = (0..v).map(|_| gauss(2.0, &mut rnd)).collect();
    let mut head: Vec<usize> = Vec::new();
    for j in 0..40usize {
        let idx = (j * 6151 + 17) % v;
        tgt[idx] = 6.0 + 5.0 * rnd();
        head.push(idx);
    }
    let mut dr: Vec<f64> = tgt[..vd].to_vec();
    for &idx in &head {
        if idx < vd {
            dr[idx] += gauss(1.0, &mut rnd);
        }
    }
    dr[1234 % vd] = 10.5; // the draft's own favourite (the target keeps its bulk logit there)
    (
        dr.iter().map(|&x| f16::from_f64(x).to_bits()).collect(),
        tgt.iter().map(|&x| f16::from_f64(x).to_bits()).collect(),
    )
}

struct ProbeFns {
    dev: Arc<CudaDevice>,
    draft: cudarc::driver::CudaFunction,
    rq: cudarc::driver::CudaFunction,
    plain: cudarc::driver::CudaFunction,
}

struct Cond {
    gate: (f64, f64, usize),
    control: (f64, f64, usize),
    neg: (f64, f64, usize),
    acc_rate: f64,
    acc_e: f64,
    acc_z: f64,
    outside: u64,
    unwritten: u64,
    q_maxdiff: f64,
    q_near: bool,
    determinism: bool,
    bonus_identity: bool,
    n: u64,
}

#[allow(clippy::too_many_arguments)]
fn run_condition(g: &ProbeFns, draft_dev: &CudaSlice<u16>, tgt_dev: &CudaSlice<u16>, vd: usize, v: usize,
                 target_row: &[u16], rows: usize, launches: usize,
                 t: f32, top_p: f32, top_k: u32, tau: f32, seed: u64) -> Result<Cond> {
    let dev = &g.dev;
    let nb = draft_nb(vd);
    let cfg_draft = LaunchConfig { grid_dim: (nb as u32, rows as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let cfg_rows = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    let mut d_buf = dev.htod_sync_copy(&vec![-1i32; rows])?;
    let mut part = dev.alloc_zeros::<u64>(rows * nb * RQ_K)?; // fully written before any read
    let mut cnt = dev.htod_sync_copy(&vec![0u32; rows])?;   // a REAL zero (AGENTS §2.2)
    let mut list = dev.htod_sync_copy(&vec![u32::MAX; rows * RQ_ROW])?;
    let dsamp = dev.htod_sync_copy(&[1u32, tau.to_bits(), top_p.to_bits(), top_k,
                                     seed as u32, (seed >> 32) as u32, 0, 0])?;
    let p = target_law(target_row, t, top_k, top_p, 0.0);
    let (seed_p, seed_c) = (seed ^ 0x5BD1_E995_0000_0001, seed ^ 0x27D4_EB2F_1656_67C5);
    let samp_rows = |flag: u32, sd: u64, base: u32| -> Vec<u32> {
        let mut s = vec![0u32; rows * 8];
        for r in 0..rows {
            s[r * 8..r * 8 + 8].copy_from_slice(&[flag, t.to_bits(), top_p.to_bits(), top_k,
                sd as u32, (sd >> 32) as u32, base + r as u32, 0]);
        }
        s
    };
    let (mut h_rq, mut h_plain, mut h_ctl, mut h_draft) = (vec![0u64; v], vec![0u64; v], vec![0u64; v], vec![0u64; v]);
    let (mut acc, mut unwritten, mut outside) = (0u64, 0u64, 0u64);
    let mut first: Option<(Vec<i32>, Vec<i32>)> = None;
    let mut q_maxdiff = 0.0f64;
    let mut q_near = false;
    let mut bonus_identity = true;
    let mut acc_e = 0.0f64;
    // one real-q launch pair: drafts for `rows` independent trials (counter base + r), then the
    // ratio verify of each trial's row (nratio = rows: every row is a draft row)
    let mut rq_launch = |base: u32, d_buf: &mut CudaSlice<i32>, list: &mut CudaSlice<u32>| -> Result<(Vec<i32>, Vec<i32>)> {
        let meta = dev.htod_sync_copy(&[0i32, 0, 0, base as i32])?;
        unsafe {
            g.draft.clone().launch(cfg_draft, (&mut *d_buf, draft_dev, vd as i32, &mut part, &mut cnt,
                                               &mut *list, &dsamp, &meta, 0i32, 2i32))?;
        }
        let samp = dev.htod_sync_copy(&samp_rows(2, seed, base))?;
        let mut e_buf = dev.htod_sync_copy(&vec![-1i32; rows])?;
        unsafe {
            g.rq.clone().launch(cfg_rows, (&mut e_buf, tgt_dev, v as i32, &samp, &*d_buf, &*list, rows as i32))?;
        }
        Ok((dev.dtoh_sync_copy(d_buf)?, dev.dtoh_sync_copy(&e_buf)?))
    };
    for l in 0..launches {
        let base = (l * rows) as u32;
        let (d, e) = rq_launch(base, &mut d_buf, &mut list)?;
        if l == 0 {
            // q' vs the host replica, and the analytic acceptance sum_x min(p(x), q'(x))
            let lh = dev.dtoh_sync_copy(&list)?;
            let (ids, lg, qd) = list_row(&lh, 0);
            let qh = qprime(&ids, &lg, tau, top_k, top_p);
            q_maxdiff = qh.iter().zip(qd.iter()).map(|(a, &b)| (a - b as f64).abs()).fold(0.0, f64::max);
            q_near = q_maxdiff >= 1e-4 && qprime_near(&ids, &lg, tau, top_k, top_p);
            acc_e = (0..RQ_K).filter(|&j| ids[j] >= 0).map(|j| p[ids[j] as usize].min(qd[j] as f64)).sum();
            first = Some((d.clone(), e.clone()));
        }
        for r in 0..rows {
            if e[r] < 0 || e[r] as usize >= v {
                unwritten += 1;
                continue;
            }
            h_rq[e[r] as usize] += 1;
            if p[e[r] as usize] == 0.0 {
                outside += 1;
            }
            if d[r] >= 0 && (d[r] as usize) < v {
                h_draft[d[r] as usize] += 1;
            }
            if e[r] == d[r] {
                acc += 1;
            }
        }
        // the plain sampler (reference arm) and a second plain stream (control: the match rule's
        // emitted law at a verify row IS the plain sample) — through xq_sample_rows_rq's flag-1
        // branch, which must equal xq_sample_rows id for id (checked on launch 0)
        let sp = dev.htod_sync_copy(&samp_rows(1, seed_p, base))?;
        let mut e1 = dev.htod_sync_copy(&vec![-1i32; rows])?;
        unsafe { g.plain.clone().launch(cfg_rows, (&mut e1, tgt_dev, v as i32, &sp))?; }
        let sc = dev.htod_sync_copy(&samp_rows(1, seed_c, base))?;
        let mut e2 = dev.htod_sync_copy(&vec![-1i32; rows])?;
        unsafe { g.rq.clone().launch(cfg_rows, (&mut e2, tgt_dev, v as i32, &sc, &d_buf, &list, 0i32))?; }
        let (h1, h2) = (dev.dtoh_sync_copy(&e1)?, dev.dtoh_sync_copy(&e2)?);
        for r in 0..rows {
            if h1[r] >= 0 && (h1[r] as usize) < v { h_plain[h1[r] as usize] += 1; } else { unwritten += 1; }
            if h2[r] >= 0 && (h2[r] as usize) < v { h_ctl[h2[r] as usize] += 1; } else { unwritten += 1; }
        }
        if l == 0 {
            let mut e3 = dev.htod_sync_copy(&vec![-1i32; rows])?;
            unsafe { g.plain.clone().launch(cfg_rows, (&mut e3, tgt_dev, v as i32, &sc))?; }
            bonus_identity = dev.dtoh_sync_copy(&e3)? == h2;
        }
    }
    // determinism: launch 0 again (graph replays must reproduce a seeded round)
    let again = rq_launch(0, &mut d_buf, &mut list)?;
    let determinism = first.as_ref().map_or(false, |f| *f == again);
    let n = (rows * launches) as u64;
    let acc_rate = acc as f64 / n as f64;
    let acc_z = if acc_e > 0.0 && acc_e < 1.0 { (acc_rate - acc_e) / (acc_e * (1.0 - acc_e) / n as f64).sqrt() } else { 0.0 };
    Ok(Cond {
        gate: gof2(&h_rq, &h_plain),
        control: gof2(&h_ctl, &h_plain),
        neg: gof2(&h_draft, &h_plain),
        acc_rate, acc_e, acc_z, outside, unwritten, q_maxdiff, q_near, determinism, bonus_identity, n,
    })
}

/// `--probe-spec-sampling`: the WP24 distribution gate (reference-free two-sample chi-square,
/// AGENTS §3: a control arm, a negative control, independent RNG streams per condition).
///
/// Stage A (always; no model): per (T in {0.7, 1.0} x top_p in {0.8, 0.95}, top_k 20, >= 2 seeds)
/// `launches x rows` independent real-q trials (a draft sampled from q' by xq_rq_draft + the
/// ratio verify row of xq_sample_rows_rq) vs the plain sampler (xq_sample_rows) on the same
/// synthetic target row. PASS iff real-q vs plain |z| < 4 (THE gate), the plain-vs-plain control
/// |z| < 4, the acceptance rate matches sum min(p, q') (|z| < 4), no emission outside supp(p), no
/// unwritten row, q' == the host replica (< 1e-4, top-p boundary ties excepted), and a re-launch
/// reproduces the trials bit for bit. The negative control (the raw drafts scored as emissions)
/// must be REJECTED (the gate has power). Reported, not gated: whether the flag-1 (bonus) branch
/// of xq_sample_rows_rq equals xq_sample_rows id for id (expected: the same code).
///
/// Stage B (`--model-dir <EXL3 pack>`): the served-level gate, exl3_serve::wp24_served_gate —
/// ratio MTP vs plain decode (MTP off), with match MTP as the control, over the real scheduler.
/// `--skip-kernel` runs stage B only. Exit status: non-zero on any FAIL.
pub fn probe_spec_sampling(args: &[String]) -> Result<()> {
    let rows: usize = parse(args, "--trials-per-launch").unwrap_or(512);
    let launches: usize = parse(args, "--launches").unwrap_or(200);
    let n_seeds: usize = parse::<usize>(args, "--seeds").unwrap_or(2).max(2);
    let top_k: u32 = parse(args, "--top-k").unwrap_or(20);
    let tau_fixed: Option<f32> = parse(args, "--draft-temperature");
    let model_dir: Option<String> = parse(args, "--model-dir");
    let mut all_ok = true;
    if !args.iter().any(|a| a == "--skip-kernel") {
        all_ok &= kernel_gate(rows, launches, n_seeds, top_k, tau_fixed)?;
    }
    if let Some(dir) = model_dir.as_deref() {
        let ok = crate::exl3_serve::wp24_served_gate(args, dir)?;
        all_ok &= ok;
    } else {
        println!("WP24 gate: stage B (served level) skipped — pass --model-dir <EXL3 pack> to run it");
    }
    if all_ok {
        println!("RESULT: WP24_GATE_OK");
        Ok(())
    } else {
        println!("RESULT: WP24_GATE_FAIL");
        bail!("real-q distribution gate failed")
    }
}

/// Stage A of `--probe-spec-sampling` (see there). Its own device handle and module, dropped on
/// return (before stage B loads the model).
fn kernel_gate(rows: usize, launches: usize, n_seeds: usize, top_k: u32, tau_fixed: Option<f32>) -> Result<bool> {
    let v_full: usize = 248_320;
    let (dr, tr) = synthetic_rows(v_full, 65_536);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/exl3_bench.ptx")
        .context("src/ptx/exl3_bench.ptx missing (run from the deploy dir)")?);
    dev.load_ptx(ptx, "wp24_probe", &["xq_rq_draft", "xq_sample_rows_rq", "xq_sample_rows"])?;
    let g = ProbeFns {
        draft: dev.get_func("wp24_probe", "xq_rq_draft").context("xq_rq_draft missing")?,
        rq: dev.get_func("wp24_probe", "xq_sample_rows_rq").context("xq_sample_rows_rq missing")?,
        plain: dev.get_func("wp24_probe", "xq_sample_rows").context("xq_sample_rows missing")?,
        dev: dev.clone(),
    };
    println!("WP24 real-q gate, stage A (kernel level, synthetic rows): {launches} launches x {rows} trials per condition, \
              top_k {top_k}, {n_seeds} seeds, tau_d {}", tau_fixed.map_or("= T".to_string(), |x| x.to_string()));
    let (vd, v) = (dr.len(), tr.len());
    anyhow::ensure!(draft_fits(vd) && vd <= v, "draft row {vd} outside the xq_rq_draft envelope");
    let draft_dev = dev.htod_sync_copy(&dr.iter().cycle().take(vd * rows).cloned().collect::<Vec<u16>>())?;
    let tgt_dev = dev.htod_sync_copy(&tr.iter().cycle().take(v * rows).cloned().collect::<Vec<u16>>())?;
    let mut all_ok = true;
    let mut neg_rejected = false;
    for &(t, top_p) in &[(0.7f32, 0.8f32), (0.7, 0.95), (1.0, 0.8), (1.0, 0.95)] {
        for si in 0..n_seeds {
            // an independent RNG stream per condition (AGENTS §3)
            let seed = 0xC0FF_EE00_0000_0000u64 ^ mix64((t.to_bits() as u64) << 32 | (top_p.to_bits() as u64) ^ (si as u64 + 1));
            let tau = tau_fixed.unwrap_or(t);
            let c = run_condition(&g, &draft_dev, &tgt_dev, vd, v, &tr, rows, launches, t, top_p, top_k, tau, seed)?;
            // bonus identity (flag-1 rows of xq_sample_rows_rq == xq_sample_rows id for id) is
            // reported, not gated: the bonus row's LAW is p either way (the gate's plain arm)
            let ok = c.gate.1.abs() < 4.0 && c.control.1.abs() < 4.0 && c.acc_z.abs() < 4.0
                && c.outside == 0 && c.unwritten == 0 && (c.q_maxdiff < 1e-4 || c.q_near) && c.determinism;
            neg_rejected |= c.neg.1 >= 4.0;
            all_ok &= ok;
            println!("  [synthetic] T={t} top_p={top_p} seed#{si}: n={} GATE realq-vs-plain z={:+.2} (chi2/df {:.3}, {} bins) | \
                      control plain-vs-plain z={:+.2} | NEG drafts-vs-plain z={:+.1} | accept {:.4} vs E {:.4} (z={:+.2}) | \
                      outside={} unwritten={} q'diff={:.1e}{} determinism={} bonus-identity={} -> {}",
                     c.n, c.gate.1, c.gate.0, c.gate.2, c.control.1, c.neg.1, c.acc_rate, c.acc_e, c.acc_z,
                     c.outside, c.unwritten, c.q_maxdiff, if c.q_near { " (top-p boundary tie)" } else { "" },
                     c.determinism, c.bonus_identity, if ok { "PASS" } else { "FAIL" });
        }
    }
    if !neg_rejected {
        println!("  negative control never rejected: the gate has no power on these rows -> FAIL");
        all_ok = false;
    }
    println!("RESULT: {}", if all_ok {
        "DISTRIBUTION_OK (stage A: real-q speculative sampling is distribution-exact vs the plain sampler)"
    } else {
        "DISTRIBUTION_MISMATCH (stage A)"
    });
    Ok(all_ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u24_domain0_is_the_plain_sampler_draw() {
        for &(seed, ctr) in &[(0u64, 0u32), (0x1234_5678_9ABC, 7), (u64::MAX, u32::MAX)] {
            let old = ((mix64(seed ^ mix64(ctr as u64 + 1)) >> 40) as f32 + 0.5) * (1.0 / 16_777_216.0);
            assert_eq!(u24(seed, ctr, 0).to_bits(), old.to_bits());
            let u = u24(seed, ctr, DOM_DRAFT);
            assert!(u > 0.0 && u < 1.0);
        }
        // domains are different streams
        assert_ne!(u24(5, 9, DOM_ACC).to_bits(), u24(5, 9, DOM_RES).to_bits());
    }

    #[test]
    fn skey_roundtrip_and_order() {
        let vals = [-65504.0f32, -3.5, -0.0, 0.0, 1e-3, 2.0, 65504.0];
        let mut last = 0u32;
        for (i, &x) in vals.iter().enumerate() {
            let b = f16::from_f32(x).to_bits();
            assert_eq!(skey_val(skey(b)).to_bits(), f16::from_bits(b).to_f32().to_bits());
            if i > 0 { assert!(skey(b) >= last); }
            last = skey(b);
        }
    }

    #[test]
    fn qprime_truncation_and_renorm() {
        let ids: Vec<i32> = (0..32).collect();
        let lg: Vec<f32> = (0..32).map(|j| 10.0 - j as f32 * 0.25).collect();
        // top_k 20 -> support <= 20; top_p 1 -> exactly 20; sums to 1
        let q = qprime(&ids, &lg, 1.0, 20, 1.0);
        assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(q.iter().filter(|&&x| x > 0.0).count(), 20);
        // top_p cuts the smallest prefix reaching the mass; lower tau sharpens
        let qp = qprime(&ids, &lg, 1.0, 20, 0.5);
        let n = qp.iter().filter(|&&x| x > 0.0).count();
        let w: Vec<f64> = (0..20).map(|j| (-(j as f64) * 0.25).exp()).collect();
        let tot: f64 = w.iter().sum();
        let mut c = 0.0;
        let mut want = 0;
        for x in &w { if c < 0.5 * tot { want += 1; } c += x; }
        assert_eq!(n, want);
        let qs = qprime(&ids, &lg, 0.3, 20, 1.0);
        assert!(qs[0] > q[0]);
        // invalid tail entries never get mass; no valid candidate -> the delta on entry 0
        let mut ids2 = ids.clone();
        for x in ids2.iter_mut().skip(3) { *x = -1; }
        let q2 = qprime(&ids2, &lg, 1.0, 0, 1.0);
        assert_eq!(q2.iter().filter(|&&x| x > 0.0).count(), 3);
        let q3 = qprime(&[-1; 32], &lg, 1.0, 0, 1.0);
        assert_eq!(q3[0], 1.0);
    }

    #[test]
    fn target_law_topk_ties_and_topp() {
        // 4 tokens: logits 3, 2, 2, 1 -> top_k 2 keeps the tie (3 survivors)
        let lg: Vec<u16> = [3.0f32, 2.0, 2.0, 1.0].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let p = target_law(&lg, 1.0, 2, 1.0, 0.0);
        assert_eq!(p.iter().filter(|&&x| x > 0.0).count(), 3);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // top_p small -> only the max survives
        let p2 = target_law(&lg, 1.0, 0, 0.1, 0.0);
        assert_eq!(p2.iter().filter(|&&x| x > 0.0).count(), 1);
    }

    #[test]
    fn residual_rule_is_exact_on_paper() {
        // P(emit x) = min(p, q)(x) + (1 - sum min) * max(0, p - q)(x) / sum max(0, p - q) = p(x)
        let p = [0.5, 0.3, 0.2, 0.0];
        let q = [0.2, 0.2, 0.1, 0.5];
        let acc: f64 = p.iter().zip(q.iter()).map(|(a, b): (&f64, &f64)| a.min(*b)).sum();
        let r: Vec<f64> = p.iter().zip(q.iter()).map(|(a, b)| (a - b).max(0.0)).collect();
        let rs: f64 = r.iter().sum();
        for x in 0..4 {
            let e = p[x].min(q[x]) + (1.0 - acc) * r[x] / rs;
            assert!((e - p[x]).abs() < 1e-12);
        }
    }

    #[test]
    fn gof2_same_law_passes_and_different_fails() {
        let mut st = 1u64;
        let mut draw = |pr: &[f64]| {
            st = mix64(st);
            let u = (st >> 11) as f64 / (1u64 << 53) as f64;
            let mut c = 0.0;
            for (i, &x) in pr.iter().enumerate() { c += x; if u < c { return i; } }
            pr.len() - 1
        };
        let pa = [0.4, 0.3, 0.2, 0.1];
        let pb = [0.35, 0.35, 0.2, 0.1];
        let (mut a, mut b, mut c) = (vec![0u64; 4], vec![0u64; 4], vec![0u64; 4]);
        for _ in 0..50_000 { a[draw(&pa)] += 1; b[draw(&pa)] += 1; c[draw(&pb)] += 1; }
        assert!(gof2(&a, &b).1.abs() < 4.0);
        assert!(gof2(&a, &c).1 > 4.0);
    }

    /// The real-q scheme exactly as the kernels compose it (host replica): d ~ q' (DOM_DRAFT draw,
    /// list order), accept iff u_acc * q'(d) < p(d), else the residual max(0, p - q') with d
    /// forced to 0 (DOM_RES draw, id order). The emissions must follow p (two-sample test vs
    /// direct draws from p) and the acceptance rate must be sum min(p, q'); the drafts alone
    /// (the negative control) must NOT follow p.
    #[test]
    fn host_realq_scheme_is_distribution_exact() {
        let (v, vd) = (64usize, 48usize);
        let mut st = 7u64;
        let mut g = || { st = mix64(st); (st >> 11) as f64 / (1u64 << 53) as f64 * 6.0 - 3.0 };
        let tgt: Vec<u16> = (0..v).map(|_| f16::from_f64(g()).to_bits()).collect();
        let mut drf: Vec<u16> = tgt[..vd].iter().map(|&b| f16::from_f64(f16::from_bits(b).to_f64() + 0.8 * g() / 3.0).to_bits()).collect();
        drf[5] = f16::from_f64(3.5).to_bits(); // a draft favourite
        let p = target_law(&tgt, 1.0, 20, 0.95, 0.0);
        let mut idx: Vec<usize> = (0..vd).collect();
        idx.sort_by(|&a, &b| skey(drf[b]).cmp(&skey(drf[a])).then(a.cmp(&b)));
        let ids: Vec<i32> = idx[..RQ_K].iter().map(|&x| x as i32).collect();
        let lg: Vec<f32> = idx[..RQ_K].iter().map(|&x| f16::from_bits(drf[x]).to_f32()).collect();
        let q = qprime(&ids, &lg, 0.8, 20, 0.95);
        let inv = |w: &[f64], u: f64| -> usize {
            let tot: f64 = w.iter().sum();
            let mut c = 0.0;
            let mut last = 0;
            for (j, &x) in w.iter().enumerate() {
                if x > 0.0 { last = j; }
                c += x;
                if x > 0.0 && c > u * tot { return j; }
            }
            last
        };
        let (n, s1, s2, s3) = (200_000u32, 0xABCDu64, 0x1234_5678u64, 0x9999_0000_1111u64);
        let (mut he, mut hp, mut hd) = (vec![0u64; v], vec![0u64; v], vec![0u64; v]);
        let mut acc = 0u64;
        for t in 0..n {
            let jd = inv(&q, u24(s1, t, DOM_DRAFT) as f64);
            let d = ids[jd] as usize;
            hd[d] += 1;
            let e = if (u24(s2, t, DOM_ACC) as f64) * q[jd] < p[d] {
                acc += 1;
                d
            } else {
                let mut r = p.clone();
                for (j, &id) in ids.iter().enumerate() { r[id as usize] = (p[id as usize] - q[j]).max(0.0); }
                r[d] = 0.0;
                inv(&r, u24(s2, t, DOM_RES) as f64)
            };
            he[e] += 1;
            hp[inv(&p, u24(s3, t, 0) as f64)] += 1;
        }
        let z = gof2(&he, &hp).1;
        assert!(z.abs() < 4.0, "real-q emissions vs p: z {z}");
        assert!(gof2(&hd, &hp).1 > 4.0, "negative control must reject");
        let e_acc: f64 = ids.iter().enumerate().map(|(j, &id)| p[id as usize].min(q[j])).sum();
        let rate = acc as f64 / n as f64;
        let za = (rate - e_acc) / (e_acc * (1.0 - e_acc) / n as f64).sqrt();
        assert!(za.abs() < 4.0, "acceptance {rate} vs {e_acc} (z {za})");
    }

    /// A top-p cut that lands exactly on an entry boundary is flagged "near" (the device's f32
    /// scan and the f64 replica may disagree there); a cut well inside an entry is not.
    #[test]
    fn qprime_near_flags_boundary_ties_only() {
        let ids: Vec<i32> = (0..32).collect();
        // 4 equal weights (logit 0) then -inf-ish tail: prefix masses 0, .25, .5, .75 of tot
        let lg: Vec<f32> = (0..32).map(|j| if j < 4 { 0.0 } else { -60.0 }).collect();
        assert!(qprime_near(&ids, &lg, 1.0, 20, 0.5));   // excl(entry 2) == 0.5 * tot exactly
        assert!(!qprime_near(&ids, &lg, 1.0, 20, 0.6));
        assert!(!qprime_near(&ids, &lg, 1.0, 20, 1.0));  // no top-p cut at all
        assert!(!qprime_near(&[-1; 32], &lg, 1.0, 20, 0.5));
    }

    /// gof2_maps aligns sparse histograms on the union of their keys (a key present in one arm
    /// only is a real bin, not dropped).
    #[test]
    fn gof2_maps_aligns_keys() {
        let mut a: HashMap<u64, u64> = HashMap::new();
        let mut b: HashMap<u64, u64> = HashMap::new();
        for (k, n) in [(3u64, 500u64), (7, 300), (11, 200)] { a.insert(k, n); b.insert(k, n); }
        let same = gof2_maps(&a, &b);
        assert!(same.1.abs() < 4.0 && same.2 == 3, "{same:?}");
        b.insert(99, 400); // a token only one arm ever emits
        let diff = gof2_maps(&a, &b);
        assert!(diff.1 > 4.0, "{diff:?}");
        assert_eq!(gof2_maps(&a, &b), gof2(&[500, 300, 200, 0], &[500, 300, 200, 400]));
    }

    #[test]
    fn draft_envelope() {
        assert!(draft_fits(65_536));
        assert!(draft_fits(248_320));
        assert_eq!(draft_nb(65_536), 16);
        assert_eq!(draft_nb(248_320), 32);
    }
}
