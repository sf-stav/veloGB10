//! Leg PK5: the drafter's head side (inference/model.py DSparkBlock forward_head): the Markov chain (for i in
//! 0..k, `logits[i] += markov_head(markov_embed[ids[i]])`, `ids[i + 1]` = a draw from the processed distribution of
//! `logits[i]`), the confidence head and the tied head over many rows. Shared by the TP1 drafter (`drafter.rs`),
//! FW's TP drafter and the budget harness (`tpbench.rs`), so every caller runs the same kernels.
//!
//! * [`run`]: three launches per chain step (`d41_ds_mstep_l*` + `d41_ds_vscan` + `d41_ds_vfin`, band 93; at TP > 1
//!   the last is `d41_ds_vrank` + the exchange + `d41_ds_vmerge`).
//! * [`run_reference`]: DS's per-step sequence (gather, plain GEMV, vpart, vrank, exchange, vmerge), kept as the
//!   bit-identity reference of [`selfcheck`].
//! * [`confidence`]: `d41_ds_conf` (one warp per row) for a plain confidence head.
//! * [`plain_rows_gemv`]: `d41_ds_plain_r{16,24,32}`, a plain BF16 / F16 head over up to 32 rows in one pass.

use super::vocab::{MstepIn, Round, Rows, Shard};
use super::xchg::VocabExchange;
use super::{ConfArgs, PlainArgs, VRec, BLOCK};
use crate::dsv41::census::classify::Role;
use crate::dsv41::fwd::dev::{bf16, f32a, Gpu};
use crate::dsv41::fwd::glue::{self, Scr};
use crate::dsv41::weights::{Accum, Act, ActDtype, BackendKind, DenseWeight, DevPtr, Shard as WShard};
use anyhow::{ensure, Context, Result};

/// The entries a chain step runs (the engagement receipt).
pub const ENTRY: &str = "d41_ds_mstep_l*+d41_ds_vscan+d41_ds_vfin (TP > 1: +d41_ds_vrank+exchange+d41_ds_vmerge)";
/// The reference sequence's entries.
pub const ENTRY_REF: &str = "d41x_gather+d41x_plain_gemv+d41_ds_vpart+d41_ds_vrank+d41_ds_vmerge";

/// One lane set's chain operands.
pub struct Chain<'a> {
    /// the Markov embedding [vocab][rank] (replicated) and head [vocab rows of this rank][rank]
    pub me: &'a (DenseWeight, BackendKind),
    pub mh: &'a (DenseWeight, BackendKind),
    /// [lanes][5][ld] bf16: the confidence head's input; columns d .. d + r of row i receive markov_embed(ids[i])
    pub cin: u64,
    pub ld: usize,
    pub d: usize,
    pub r: usize,
    /// [lanes][5][vl] f32: the head's logits of the block rows (this rank's vocab columns); row i gets the Markov bias
    pub logits: u64,
    pub vl: usize,
    pub sh: Shard,
    /// the chain's input ids table ([6][lanes] i32: the round's ids, or the gate's forced ids)
    pub table: u64,
}

/// (pointer to the call's first row, K, rows, code 0 bf16 / 1 f16) of a plain matrix (`weights::plain`'s rule).
pub fn plain_rows(w: &DenseWeight) -> Result<(u64, usize, usize, i32)> {
    let t = w.tensors.iter().find(|t| t.role == Role::Weight).with_context(|| format!("{}: no weight tensor", w.key))?;
    let code = match t.dtype.as_str() { "BF16" => 0, "F16" => 1, d => anyhow::bail!("plain {}: dtype {d} (BF16 / F16 only)", w.key) };
    ensure!(t.shape.len() == 2 && t.shape[1] == w.inn, "plain {}: shape {:?} (K {})", w.key, t.shape, w.inn);
    let (n, k) = (t.shape[0] as usize, t.shape[1] as usize);
    Ok(match &w.shard {
        WShard::Whole => (t.ptr.0, k, n, code),
        WShard::Out(r) => {
            let (lo, hi) = (r.start as usize, r.end as usize);
            ensure!(hi <= w.out as usize && lo < hi, "plain {}: shard {lo}..{hi}", w.key);
            if n == w.out as usize { (t.ptr.0 + (lo * k * 2) as u64, k, hi - lo, code) } else { ensure!(n == hi - lo, "plain {}: resident rows {n}", w.key); (t.ptr.0, k, n, code) }
        }
        WShard::In(_) => anyhow::bail!("plain {}: an input slice is not supported", w.key),
    })
}

/// Can [`run`] serve this chain (plain Markov pieces of rank 256: a BF16 embedding, whole; a BF16 / F16 head; <= 4
/// lanes)? Err = why not.
pub fn fused_why(c: &Chain, nl: usize) -> Result<()> {
    let (_, ke, ne, ce) = plain_rows(&c.me.0)?;
    let (_, kh, nh, chd) = plain_rows(&c.mh.0)?;
    ensure!(c.me.1 == BackendKind::Plain && c.mh.1 == BackendKind::Plain, "Markov backends {:?} / {:?} (plain expected)", c.me.1, c.mh.1);
    ensure!(ce == 0, "Markov embedding dtype code {ce} (BF16 expected; the head may be BF16 or F16, code {chd})");
    ensure!(ke == 256 && kh == 256 && c.r == 256, "Markov rank: embed K {ke}, head K {kh}, r {} (256 expected)", c.r);
    ensure!(nh == c.vl, "Markov head rows {nh} != the shard's {} vocab rows", c.vl);
    ensure!(ne as u64 == c.me.0.out, "Markov embedding rows {ne} of {} (whole expected)", c.me.0.out);
    ensure!(nl <= 4, "{nl} lanes (<= 4)");
    Ok(())
}
pub fn fused_ok(c: &Chain, nl: usize) -> bool { fused_why(c, nl).is_ok() }

/// k chain steps of `nl` lanes ([`Round::mstep`]); falls back to [`run_reference`] when the
/// operands are not the fused kernel's (counted in the receipt). `mark` is called after each step.
pub fn run(g: &Gpu, scr: &Scr, c: &Chain, round: &Round, x: &dyn VocabExchange, nl: usize, k: usize, mark: &mut dyn FnMut(&'static str) -> Result<()>) -> Result<()> {
    if !fused_ok(c, nl) {
        crate::dsv41::fwd::receipts::kernel("ds.markov_chain (reference sequence: operands not fusable)", ENTRY_REF, nl);
        return run_reference(g, scr, c, round, x, nl, k, mark);
    }
    let (me, _, _, _) = plain_rows(&c.me.0)?;
    let (mh, _, _, mh_code) = plain_rows(&c.mh.0)?;
    let m = MstepIn { me, mh, mh_code, table: c.table, cin: c.cin, cin_ls: BLOCK * c.ld, ld: c.ld, d: c.d, logits: Rows { ptr: c.logits, lane_stride: BLOCK * c.vl, row_stride: c.vl } };
    for i in 0..k {
        round.mstep(g, x, c.sh, &m, nl, i)?;
        mark("markov chain step (mstep + vscan + vfin)")?;
    }
    Ok(())
}

/// DS's per-step sequence: gather, plain GEMV (+= logits), vpart, vrank, exchange, vmerge (PICK).
pub fn run_reference(g: &Gpu, scr: &Scr, c: &Chain, round: &Round, x: &dyn VocabExchange, nl: usize, k: usize, mark: &mut dyn FnMut(&'static str) -> Result<()>) -> Result<()> {
    let meb = glue::backend(c.me.1);
    for i in 0..k {
        let out = bf16(c.cin + ((i * c.ld + c.d) * 2) as u64, nl, c.r, BLOCK * c.ld);
        meb.gather(&g.l(), &c.me.0, DevPtr(c.table + (i * nl * 4) as u64), nl, &out)?;
        mark("markov embed gather")?;
        let mut ya = f32a(c.logits + (i * c.vl * 4) as u64, nl, c.vl, BLOCK * c.vl);
        ya.cols = c.vl;
        glue::linear(g, "ds.markov_head", c.mh, &out, &ya, Accum::Add, scr)?;
        mark("markov head GEMV (+= logits)")?;
        round.draft_pick(g, x, c.sh, Rows { ptr: c.logits, lane_stride: BLOCK * c.vl, row_stride: c.vl }, nl, i, 1, None)?;
        mark("vocab pass + pick (vpart, vrank, vmerge)")?;
    }
    Ok(())
}

/// conf[r] = cw . cin[r] (f32) for `rows` rows of `ld` columns: `d41_ds_conf` for a plain [1][ld] head, the
/// backend's linear otherwise.
/// Returns the entry that ran (the engagement receipt).
pub fn confidence(g: &Gpu, scr: &Scr, cw: &(DenseWeight, BackendKind), cin: u64, rows: usize, ld: usize, conf: u64) -> Result<&'static str> {
    if cw.1 == BackendKind::Plain {
        if let Ok((w, k, n, code)) = plain_rows(&cw.0) {
            if n == 1 && k == ld && k % 256 == 0 {
                g.launch1("d41_ds_conf", (rows as u32, 1, 1), (32, 1, 1), 0,
                          &ConfArgs { w, x: cin, y: conf, wcode: code, k: k as i32, ldx: ld as i32, ldy: 1, rows: rows as i32, pad: 0 })?;
                return Ok("d41_ds_conf");
            }
        }
    }
    glue::linear(g, "ds.confidence", cw, &bf16(cin, rows, ld, ld), &f32a(conf, rows, 1, 1), Accum::Overwrite, scr)?;
    Ok("the backend's linear")
}

fn act_code(a: &Act) -> Option<i32> { match a.dtype { ActDtype::Bf16 => Some(0), ActDtype::F16 => Some(1), ActDtype::F32 => Some(2), _ => None } }

/// y (+)= x . W^T for a plain matrix over 9..=32 rows in ONE pass (`d41_ds_plain_r{16,24,32}`; the plain backend's
/// GEMV re-reads the matrix for every 8 rows). Returns false (nothing launched) when the call is not one: a
/// non-plain weight, <= 8 rows (the backend's single pass), > 32 rows.
pub fn plain_rows_gemv(g: &Gpu, w: &(DenseWeight, BackendKind), x: &Act, y: &Act, acc: Accum) -> Result<bool> {
    if w.1 != BackendKind::Plain || x.rows <= 8 || x.rows > 32 { return Ok(false); }
    let (Some(xc), Some(yc)) = (act_code(x), act_code(y)) else { return Ok(false) };
    let (wp, k, nn, code) = plain_rows(&w.0)?;
    ensure!(x.cols == k && y.cols == nn && x.rows == y.rows && k % 256 == 0, "plain {}: x [{}, {}] y [{}, {}] K {k} N {nn}", w.0.key, x.rows, x.cols, y.rows, y.cols);
    // bf16 rows are read 16 B at a time: 16-B aligned rows
    if xc == 0 && (x.ptr.0 % 16 != 0 || x.ld % 8 != 0) { return Ok(false); }
    let name = if x.rows <= 16 { "d41_ds_plain_r16" } else if x.rows <= 24 { "d41_ds_plain_r24" } else { "d41_ds_plain_r32" };
    let a = PlainArgs { w: wp, x: x.ptr.0, y: y.ptr.0, wcode: code, k: k as i32, nn: nn as i32, xcode: xc, ldx: x.ld as i32, rows: x.rows as i32,
                        ycode: yc, ldy: y.ld as i32, accum: (acc == Accum::Add) as i32, pad: 0 };
    g.launch1(name, (nn.div_ceil(8) as u32, 1, 1), (256, 1, 1), 0, &a)?;
    Ok(true)
}

/// A 2-rank test exchange: this rank's records in slot 0, empty records (the merge's identity) in slot 1, so the
/// TP > 1 branch of [`Round::mstep`] (exchange + PICK merge) must reproduce the TP1 result bit for bit.
struct PadEmpty { empty: u64, max_bytes: usize }
impl VocabExchange for PadEmpty {
    fn ranks(&self) -> usize { 2 }
    fn rank(&self) -> usize { 0 }
    fn max_bytes(&self) -> usize { self.max_bytes }
    fn all_gather(&self, g: &Gpu, send: u64, bytes: usize, recv: u64) -> Result<()> {
        g.d2d(recv, send, bytes)?;
        g.d2d(recv + bytes as u64, self.empty, bytes)
    }
}

fn empty_rec() -> VRec {
    VRec { amax: f32::NEG_INFINITY, aid: i32::MAX, m: f32::NEG_INFINITY, s: 0.0, gmax: f32::NEG_INFINITY, gid: i32::MAX, gx: f32::NEG_INFINITY,
           px: f32::NEG_INFINITY, pheld: 0, rsum: 0.0, rmax: f32::NEG_INFINITY, rid: i32::MAX, pad: [0; 4] }
}

/// The fused kernels against the sequences they replace, on the drafter's real Markov / confidence weights and
/// synthetic logits: the chain (TP1 whole vocabulary, a TP4-sized shard, the TP > 1 exchange branch; 1 / 2 / 4
/// lanes; greedy and sampled lanes; k = 5) bit for bit on the logits rows, the drafts, their log-probabilities and
/// log-sum-exps and the confidence input; the confidence head; the one-pass plain GEMV at 12 / 20 / 28 rows.
/// Returns (lines, failures).
pub fn selfcheck(g: &Gpu, scr: &Scr, me: &(DenseWeight, BackendKind), mh_whole: &(DenseWeight, BackendKind), cw: &(DenseWeight, BackendKind), vocab: usize, d: usize)
                 -> Result<(Vec<String>, Vec<String>)> {
    let mut lines = Vec::new();
    let mut fails = Vec::new();
    let r = 256usize;
    let ld = d + r;
    let cap = 4usize;
    let up = |v: &[u8]| -> Result<crate::dsv41::fwd::dev::Buf> { let b = g.buf(v.len())?; g.h2d(b.ptr(), v)?; Ok(b) };
    for (label, world, rank) in [("TP1 whole vocabulary", 1usize, 0usize), ("TP4-sized shard (rank 1)", 4, 1), ("TP4-sized shard, exchange branch", 4, 3)] {
        let vr = crate::dsv41::load::splits::vocab_range(vocab, world, rank);
        let sh = Shard { v0: vr.start, vlocal: vr.len() };
        let mut mh = mh_whole.clone();
        if world > 1 { mh.0.shard = WShard::Out(vr.start as u64..vr.end as u64); }
        let vl = sh.vlocal;
        let exch = label.contains("exchange");
        let ranks = if exch { 2 } else { 1 };
        let round = Round::new(g, cap, BLOCK + 1, sh.nblk(), ranks)?;
        let emp = up(super::vocab::as_bytes(&vec![empty_rec(); cap * BLOCK]))?;
        let pad = PadEmpty { empty: emp.ptr(), max_bytes: 64 * 1024 };
        let local = super::xchg::Local { max_bytes: 64 * 1024 };
        let x: &dyn VocabExchange = if exch { &pad } else { &local };
        let logits0: Vec<f32> = (0..cap * BLOCK * vl).map(|i| (((i as u64).wrapping_mul(2_654_435_761) % 20_011) as f32 - 10_000.0) * 6e-4).collect();
        let lg = g.buf(logits0.len() * 4)?;
        let cin = g.buf(cap * BLOCK * ld * 2)?;
        for nl in [1usize, 2, 4] {
            let lanes: Vec<super::Lane> = (0..nl).map(|l| super::Lane { seed: 0x5EED + 977 * l as u64, inv_t: if l == 1 { 0.0 } else { 1.0 / (0.7 + 0.3 * l as f32) },
                                                                         vbase: 700 + l as i32, k: 5, wend: 703 + 2 * l as i32, tok: 0, active: 1,
                                                                         rounds: 7 + 3 * l as u32, ..Default::default() }).collect();   // PH1: a later round (PS1 draft noise)
            let tok0: Vec<i32> = (0..nl).map(|l| (1000 + 31_337 * l) as i32 % vocab as i32).collect();
            let mut got = Vec::new();
            for fused in [false, true] {
                g.h2d(lg.ptr(), super::vocab::as_bytes(&logits0))?;
                g.memset(cin.ptr(), cin.bytes, 0)?;
                round.set_lanes(g, &lanes)?;
                g.memset(round.ids.ptr(), round.ids.bytes, 0)?;
                g.h2d(round.ids.ptr(), super::vocab::as_bytes(&tok0))?;
                let c = Chain { me, mh: &mh, cin: cin.ptr(), ld, d, r, logits: lg.ptr(), vl, sh, table: round.ids.ptr() };
                if fused {
                    fused_why(&c, nl).context("chain selfcheck: the drafter's Markov pieces are not the fused kernel's operands")?;
                    run(g, scr, &c, &round, x, nl, BLOCK, &mut |_| Ok(()))?;
                } else {
                    run_reference(g, scr, &c, &round, x, nl, BLOCK, &mut |_| Ok(()))?;
                }
                let ids = round.ids_table(g)?;
                let dlp = g.down::<f32>(round.dlp.ptr(), cap * BLOCK)?;
                let dlse = g.down::<f32>(round.dlse.ptr(), cap * BLOCK)?;
                let lgo = g.down::<u32>(lg.ptr(), nl * BLOCK * vl)?;
                let ci = g.down::<u16>(cin.ptr(), nl * BLOCK * ld)?;
                got.push((ids, dlp, dlse, lgo, ci));
            }
            let (a, b) = (&got[0], &got[1]);
            // the ids table is [6][nl] for an nl-lane call (row j at j * nl)
            let ids_eq = (0..=BLOCK).all(|j| (0..nl).all(|l| a.0[j * nl + l] == b.0[j * nl + l]));
            let lp_eq = (0..nl * BLOCK).all(|i| a.1[i].to_bits() == b.1[i].to_bits() && a.2[i].to_bits() == b.2[i].to_bits());
            let lg_eq = a.3 == b.3;
            let ci_eq = a.4 == b.4;
            let ok = ids_eq && lp_eq && lg_eq && ci_eq;
            let drafts: Vec<Vec<i32>> = (0..nl).map(|l| (1..=BLOCK).map(|j| b.0[j * nl + l]).collect()).collect();
            lines.push(format!("  chain fused vs reference, {label}, {nl} lane(s) (lane 1 greedy), k 5: drafts {} {drafts:?}, log-probs / lse {}, logits rows {}, confidence input {} -> {}",
                               if ids_eq { "equal" } else { "DIFFER" }, if lp_eq { "bitwise" } else { "DIFFER" }, if lg_eq { "bitwise" } else { "DIFFER" },
                               if ci_eq { "bitwise" } else { "DIFFERS" }, if ok { "PASS" } else { "FAIL" }));
            if !ok { fails.push(format!("chain fused vs reference ({label}, {nl} lanes)")); }
        }
    }
    // the confidence head: d41_ds_conf vs the plain GEMV
    let rows = cap * BLOCK;
    let cv: Vec<u16> = (0..rows * ld).map(|i| crate::dsv41::fwd::cmp::f32_to_bf16_bits((((i * 7919) % 2003) as f32 - 1001.0) * 1e-3)).collect();
    let cb = up(super::vocab::as_bytes(&cv))?;
    let (c0, c1) = (g.buf(rows * 4)?, g.buf(rows * 4)?);
    glue::linear(g, "ds.confidence", cw, &bf16(cb.ptr(), rows, ld, ld), &f32a(c0.ptr(), rows, 1, 1), Accum::Overwrite, scr)?;
    confidence(g, scr, cw, cb.ptr(), rows, ld, c1.ptr())?;
    let (v0, v1) = (g.down::<u32>(c0.ptr(), rows)?, g.down::<u32>(c1.ptr(), rows)?);
    let ok = v0 == v1;
    lines.push(format!("  confidence head d41_ds_conf vs the plain GEMV, {rows} rows: {} -> {}", if ok { "bitwise" } else { "DIFFER" }, if ok { "PASS" } else { "FAIL" }));
    if !ok { fails.push("confidence head".into()); }
    // the one-pass plain GEMV vs the plain backend (the Markov head as the matrix: K 256, 129,280 columns)
    let (_, k, nn, _) = plain_rows(&mh_whole.0)?;
    for rows in [12usize, 20, 28] {
        let xv: Vec<u16> = (0..rows * k).map(|i| crate::dsv41::fwd::cmp::f32_to_bf16_bits((((i * 104_729) % 4001) as f32 - 2000.0) * 5e-4)).collect();
        let xb = up(super::vocab::as_bytes(&xv))?;
        let y0: Vec<f32> = (0..rows * nn).map(|i| ((i % 977) as f32 - 488.0) * 1e-2).collect();
        let (ya, yb) = (up(super::vocab::as_bytes(&y0))?, up(super::vocab::as_bytes(&y0))?);
        let xa = bf16(xb.ptr(), rows, k, k);
        glue::linear(g, "ds.plain_check", mh_whole, &xa, &f32a(ya.ptr(), rows, nn, nn), Accum::Add, scr)?;
        let ran = plain_rows_gemv(g, mh_whole, &xa, &f32a(yb.ptr(), rows, nn, nn), Accum::Add)?;
        let ok = ran && g.down::<u32>(ya.ptr(), rows * nn)? == g.down::<u32>(yb.ptr(), rows * nn)?;
        lines.push(format!("  one-pass plain GEMV d41_ds_plain_r{} vs the plain backend, {rows} rows x {nn} columns (+= f32): {} -> {}",
                           if rows <= 16 { 16 } else if rows <= 24 { 24 } else { 32 }, if ok { "bitwise" } else { "DIFFER" }, if ok { "PASS" } else { "FAIL" }));
        if !ok { fails.push(format!("one-pass plain GEMV ({rows} rows)")); }
    }
    Ok((lines, fails))
}
