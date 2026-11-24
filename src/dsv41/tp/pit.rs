//! DSV4.1 TP (leg PS3): the served-path sampled-exactness probe, `--dsv41-spec-pit-log <path>` (a debug switch, off by
//! default; every rank takes part, rank 0 writes the file).
//!
//! For every token a SAMPLED lane emits in a speculative round, the engine's own target distribution at that position
//! is the verify row it was decided on (row i of the lane's verify = p(. | the emitted prefix), after penalties, the
//! temperature and the top-k / top-p cut, exactly as the accept / residual kernels read it). Each rank sums, over its
//! vocabulary shard, the processed probability of the kept ids BELOW the emitted token and the token's own probability;
//! one fp32 all-reduce gives F(x-) and p(x); rank 0 writes one JSON line per token with the randomized PIT value
//! `u = F(x-) + U p(x)` (U an independent keyed uniform). If every token is an exact draw from its row, the u values are
//! i.i.d. uniform on [0, 1) over ALL tokens of ALL replies (Rosenblatt), whatever the draft counts were:
//! `tools/dsv41_gates/pit_check.py` tests that overall and per bucket (row in the round, drafts verified, lanes).
//! DeepSeek's residual uses alpha = exp(-5e-5): a known deviation below 5e-5 per token, far under the check's resolution.
//! Cost: one download of every emitted row (<= 24 x vocab / ranks floats) and one tiny all-reduce per round: debug only.

use super::fwd::TpModel;
use crate::dsv41::fwd::dev::Buf;
use crate::dsv41::gpu::dspark::vocab::{uniform, Cut, Round};
use crate::dsv41::gpu::dspark::{Lane, BLOCK, OUT_TOK};
use anyhow::Result;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Rows a round can emit at most (`tp::frame::MAX_ROWS`).
const MAX_TOK: usize = super::frame::MAX_ROWS;

/// The TP all-reduce moves whole 16-byte units (`rdma.rs`: "N floats is not a 16 B multiple"): every length this probe
/// reduces is padded to a multiple of `REDUCE_UNIT` floats (leg PS4: an odd count of sampled tokens killed both ranks).
pub const REDUCE_UNIT: usize = 4;
/// The host-side twin of the device path's rule, checked where the length is computed (so the unit suite sees this class).
pub fn check_reduce_len(floats: usize) -> Result<()> {
    anyhow::ensure!(floats % REDUCE_UNIT == 0, "dsv41 PIT: {floats} floats is not a 16 B multiple (the TP all-reduce moves whole 16-byte units)");
    Ok(())
}

/// The reduce frame of `k` tokens' (below, own) partials: 2 k floats, zero-padded to the 16-byte unit (the padding sums to 0).
pub fn pack(parts: &[(f64, f64)]) -> Vec<f32> {
    let n = (2 * parts.len()).div_ceil(REDUCE_UNIT) * REDUCE_UNIT;
    let mut v = vec![0f32; n];
    for (k, &(b, o)) in parts.iter().enumerate() { v[2 * k] = b as f32; v[2 * k + 1] = o as f32; }
    v
}

/// Receipts: tokens logged, and times the probe turned itself off (a debug facility never takes serving down).
pub static TOKENS: AtomicU64 = AtomicU64::new(0);
pub static DISABLED: AtomicU64 = AtomicU64::new(0);

/// `/metrics` rows (only once the probe was configured in this process).
pub fn append(o: &mut String) {
    if !CONFIGURED.load(Relaxed) { return; }
    o.push_str(&format!("# HELP velogb10_dsv41_spec_pit_tokens_total Sampled tokens the --dsv41-spec-pit-log probe recorded.\n# TYPE velogb10_dsv41_spec_pit_tokens_total counter\nvelogb10_dsv41_spec_pit_tokens_total {}\n", TOKENS.load(Relaxed)));
    o.push_str(&format!("# HELP velogb10_dsv41_spec_pit_disabled_total Times the --dsv41-spec-pit-log probe disabled itself (a bad reduce length or a log write error) instead of stopping the server.\n# TYPE velogb10_dsv41_spec_pit_disabled_total counter\nvelogb10_dsv41_spec_pit_disabled_total {}\n", DISABLED.load(Relaxed)));
}
static CONFIGURED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub struct Pit { out: Option<std::io::BufWriter<std::fs::File>>, dev: Buf, pub tokens: u64, pub off: bool }

/// Leg PS4: one lane's round as the head decided it: its count, tokens emitted, confidences, and the decision-sensitive
/// bits: `scan_diff` = the inexact scan would have chosen another count for this lane this round (computed from the same
/// confidences; nothing served changes), `sensitive` = the exact decision was one draft from flipping. None = not decided
/// by the auto rule (pinned length, or a rank that did not decide).
#[derive(Clone, Debug, Default)]
pub struct Ctx { pub lane: usize, pub c: usize, pub n_emit: usize, pub z: Vec<f32>, pub scan_diff: Option<bool>, pub sensitive: Option<bool> }

/// The JSON fields of a token's context (stable order; the bits as 0 / 1, absent when unknown).
pub fn ctx_json(x: Option<&Ctx>, lanes: usize) -> String {
    let Some(x) = x else { return format!("\"c\":0,\"n_emit\":0,\"lanes\":{lanes},\"z\":[]") };
    let mut s = format!("\"c\":{},\"n_emit\":{},\"lanes\":{lanes},\"z\":{:?}", x.c, x.n_emit, x.z);
    if let Some(b) = x.scan_diff { s.push_str(&format!(",\"scan_diff\":{}", u8::from(b))); }
    if let Some(b) = x.sensitive { s.push_str(&format!(",\"sensitive\":{}", u8::from(b))); }
    s
}

/// One emitted token's place: lane, row in the lane's verify (= index of the token in the round), the processed row.
pub struct Tok { pub lane: usize, pub row: usize, pub tok: u32, pub pos: i64, pub seed: u64, pub inv_t: f32 }

/// A rank's partial sums for one token over its shard `[v0, v0 + row.len())`: (mass of kept ids below `tok`, mass of `tok`).
/// `lse` = the row's processed log-sum-exp over the whole vocabulary (the merged verify record); kept iff
/// `a > ca || (a == ca && id <= cid)` (the cut; `(-inf, i32::MAX)` = no cut), NaN as -inf (the kernels' rule).
pub fn partials(row: &[f32], v0: usize, tok: u32, inv_t: f32, lse: f32, cut: (f32, i32)) -> (f64, f64) {
    let (mut below, mut own) = (0f64, 0f64);
    for (i, &a0) in row.iter().enumerate() {
        let a = if a0.is_nan() { f32::NEG_INFINITY } else { a0 };
        let id = (v0 + i) as u32;
        if a == f32::NEG_INFINITY || !(a > cut.0 || (a == cut.0 && id as i32 <= cut.1)) { continue; }
        if id > tok { break; }
        let pr = (((a * inv_t) - lse) as f64).exp();
        if id < tok { below += pr; } else { own = pr; }
    }
    (below, own)
}

/// The randomized PIT value of a token (U keyed by the lane's seed and the position, in a namespace no draw uses).
pub fn pit_u(below: f64, own: f64, seed: u64, pos: i64) -> f64 { below + uniform(seed ^ 0x5049_545F_4C4F_4721, pos as u64) * own }

impl Pit {
    /// The probe when `--dsv41-spec-pit-log` is set (rank 0 opens the file for appending).
    pub fn from_flags(g: &crate::dsv41::fwd::dev::Gpu, rank: usize) -> Result<Option<Pit>> {
        let Some(path) = crate::opts::lookup("dsv41-spec-pit-log").and_then(crate::opts::get) else { return Ok(None) };
        let out = if rank == 0 {
            Some(std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(&path)
                .map_err(|e| anyhow::anyhow!("--dsv41-spec-pit-log {path}: {e}"))?))
        } else { None };
        CONFIGURED.store(true, Relaxed);
        Ok(Some(Pit { out, dev: g.buf(2 * MAX_TOK * 4)?, tokens: 0, off: false }))
    }

    /// One round's sampled tokens. `tlog` / `lane_stride` / `vl` / `v0`: this rank's target rows as the verify read them;
    /// `cut`: the round's cut arrays ([cap][RPL] at `rpl`); `round`: its merged verify records; `lanes`: the round's results.
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self, m: &mut TpModel, round: &Round, tlog: u64, lane_stride: usize, vl: usize, v0: usize, cut: Option<Cut>, toks: &[Tok],
                  ctx: &[Ctx]) -> Result<()> {
        let g = m.f.g;
        if self.off { return Ok(()); }
        let toks: Vec<&Tok> = toks.iter().filter(|t| t.inv_t > 0.0).take(MAX_TOK).collect();
        if toks.is_empty() { return Ok(()); }
        // the frame's length is a function of the token count, the same on every rank: a bad length turns the probe off on
        // every rank alike before any collective (so the ranks' collective schedules never diverge)
        let nf = (2 * toks.len()).div_ceil(REDUCE_UNIT) * REDUCE_UNIT;
        if let Err(e) = check_reduce_len(nf).and_then(|_| { anyhow::ensure!(nf * 4 <= self.dev.bytes, "dsv41 PIT: {nf} floats > the probe's frame"); Ok(()) }) {
            self.disable(&format!("{e:#}"));
            return Ok(());
        }
        let rpl = round.rpl_max;
        let recs = round.recs(g, round.tres.ptr(), round.cap * rpl)?;
        let (ca, ci) = match cut {
            Some(c) => (g.down::<f32>(c.a, round.cap * c.rpl)?, g.down::<i32>(c.id, round.cap * c.rpl)?),
            None => (Vec::new(), Vec::new()),
        };
        let mut parts = Vec::with_capacity(toks.len());
        for t in &toks {
            let row = g.down::<f32>(tlog + ((t.lane * lane_stride + t.row * vl) * 4) as u64, vl)?;
            let cr = cut.map_or((f32::NEG_INFINITY, i32::MAX), |c| (ca[t.lane * c.rpl + t.row], ci[t.lane * c.rpl + t.row]));
            parts.push(partials(&row, v0, t.tok, t.inv_t, recs[t.lane * rpl + t.row].lse(), cr));
        }
        let part = pack(&parts);
        debug_assert_eq!(part.len(), nf);
        g.h2d(self.dev.ptr(), crate::dsv41::gpu::dspark::vocab::as_bytes(&part))?;
        m.reduce(self.dev.ptr(), part.len())?;
        let sum = g.down::<f32>(self.dev.ptr(), part.len())?;
        self.tokens += toks.len() as u64;
        TOKENS.fetch_add(toks.len() as u64, Relaxed);
        if self.out.is_some() {
            // rank 0 only, after the collective: a write error drops the file, never the round
            if let Err(e) = self.write(&toks, &sum, ctx) {
                self.out = None;
                DISABLED.fetch_add(1, Relaxed);
                crate::reprintln!("[dsv41 pit] the --dsv41-spec-pit-log file could not be written ({e:#}): the log is off from here; serving continues");
            }
        }
        Ok(())
    }

    /// Turn the probe off (every rank reaches this alike); one line, one count.
    fn disable(&mut self, why: &str) {
        self.off = true;
        DISABLED.fetch_add(1, Relaxed);
        crate::reprintln!("[dsv41 pit] --dsv41-spec-pit-log disabled ({why}); serving continues");
    }

    fn write(&mut self, toks: &[&Tok], sum: &[f32], ctx: &[Ctx]) -> Result<()> {
        if let Some(w) = self.out.as_mut() {
            for (k, t) in toks.iter().enumerate() {
                let (b, o) = (sum[2 * k] as f64, sum[2 * k + 1] as f64);
                writeln!(w, "{{\"lane\":{},\"pos\":{},\"row\":{},\"tok\":{},\"F\":{b:.9},\"p\":{o:.9},\"u\":{:.9},{}}}",
                         t.lane, t.pos, t.row, t.tok, pit_u(b, o, t.seed, t.pos), ctx_json(ctx.iter().find(|x| x.lane == t.lane), ctx.len()))?;
            }
            w.flush()?;
        }
        Ok(())
    }
}

/// The tokens a round emitted per lane (`out`, `lanes` after the round; `before` = the lanes' states before it), as `Tok`s.
pub fn tokens(active: &[usize], before: &[Lane], lanes: &[Lane], out: &[[i32; OUT_TOK]]) -> Vec<Tok> {
    let mut v = Vec::new();
    for &l in active {
        let n = lanes[l].n_emit.max(0) as usize;
        for i in 0..n.min(OUT_TOK) {
            v.push(Tok { lane: l, row: i, tok: out[l][i] as u32, pos: before[l].vbase as i64 + 1 + i as i64, seed: before[l].seed, inv_t: before[l].inv_t });
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shard partials of a split vocabulary sum to the whole row's F(x-) and p(x); the cut and NaN rules are the kernels'.
    #[test]
    fn dsv41_ps3_pit_partials_split_and_cut() {
        let row: Vec<f32> = (0..40).map(|i| ((i * 7919) % 23) as f32 * 0.3 - 2.0).collect();
        let inv_t = 1.25f32;
        let lse = { let m = row.iter().map(|&a| a * inv_t).fold(f32::NEG_INFINITY, f32::max); m + row.iter().map(|&a| ((a * inv_t) - m).exp()).sum::<f32>().ln() };
        for tok in [0u32, 7, 19, 39] {
            let (b, o) = partials(&row, 0, tok, inv_t, lse, (f32::NEG_INFINITY, i32::MAX));
            let (b1, o1) = partials(&row[..16], 0, tok, inv_t, lse, (f32::NEG_INFINITY, i32::MAX));
            let (b2, o2) = partials(&row[16..], 16, tok, inv_t, lse, (f32::NEG_INFINITY, i32::MAX));
            assert!((b - (b1 + b2)).abs() < 1e-9 && (o - (o1 + o2)).abs() < 1e-12, "tok {tok}");
            let exact_b: f64 = row[..tok as usize].iter().map(|&a| (((a * inv_t) - lse) as f64).exp()).sum();
            assert!((b - exact_b).abs() < 1e-9);
        }
        let total: f64 = (0..40).map(|t| partials(&row, 0, t, inv_t, lse, (f32::NEG_INFINITY, i32::MAX)).1).sum();
        assert!((total - 1.0).abs() < 1e-5, "{total}");
        // a cut keeps a > ca or (a == ca and id <= cid); a NaN never counts
        let mut r2 = row.clone();
        r2[3] = f32::NAN;
        let (b, _) = partials(&r2, 0, 10, inv_t, lse, (0.0, 5));
        let want: f64 = (0..10).filter(|&i| i != 3 && (r2[i] > 0.0 || (r2[i] == 0.0 && i <= 5))).map(|i| (((r2[i] * inv_t) - lse) as f64).exp()).sum();
        assert!((b - want).abs() < 1e-9);
        // leg PS4: every token count a round can produce (and beyond) packs to a 16-byte multiple that the reduce accepts,
        // the padding is zero, the payload is in place; the unpadded length (the bug) fails the same host check
        for k in 1..=33usize {
            let parts: Vec<(f64, f64)> = (0..k).map(|i| (i as f64 * 0.01, 0.5)).collect();
            let v = pack(&parts);
            assert!(check_reduce_len(v.len()).is_ok() && v.len() >= 2 * k && v.len() < 2 * k + REDUCE_UNIT, "k {k}: {}", v.len());
            assert!(v[2 * k..].iter().all(|&x| x == 0.0));
            assert!((0..k).all(|i| v[2 * i] == (i as f64 * 0.01) as f32 && v[2 * i + 1] == 0.5));
            assert_eq!(check_reduce_len(2 * k).is_ok(), k % 2 == 0, "the unpadded length of {k} tokens");
        }
        // leg PS4: the context fields, parseable JSON, the bits present only when known
        let x = Ctx { lane: 1, c: 3, n_emit: 2, z: vec![0.5, -1.0], scan_diff: Some(true), sensitive: Some(false) };
        let line = format!("{{\"u\":0.5,{}}}", ctx_json(Some(&x), 2));
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!((v["c"].as_u64(), v["lanes"].as_u64(), v["scan_diff"].as_u64(), v["sensitive"].as_u64()), (Some(3), Some(2), Some(1), Some(0)));
        let v: serde_json::Value = serde_json::from_str(&format!("{{{}}}", ctx_json(Some(&Ctx { scan_diff: None, ..x.clone() }), 1))).unwrap();
        assert!(v.get("scan_diff").is_none());
        // the PIT value of a sure token is uniform within its own mass
        let u = pit_u(0.25, 0.5, 7, 100);
        assert!((0.25..0.75).contains(&u));
    }
}
