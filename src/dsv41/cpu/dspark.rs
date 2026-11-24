//! DSV4.1 CPU oracle — leg OD: the DSpark drafter (3 stages: window-only MQA + 128-expert top-3 MoE +
//! shared expert + mHC each, `main_proj` over the inputs of target layers 37-39, the rank-256 Markov
//! head, the confidence head, block 5 with noise token 128799, `forward_spec`).
//!
//! Files (all `cpu/dspark*.rs`, so no other leg's module list changes):
//! * `dspark_num.rs` — formats and the reference's rounding points (act_quant, fp8 / fp4 GEMM, RMSNorm,
//!   RoPE, sparse attention, Sinkhorn);
//! * `dspark_weights.rs` — the native checkpoint reader and the drafter's weights;
//! * `dspark_fwd.rs` — the forward (`Dspark::{main_x, seed, draft_body, chain}`);
//! * this file — REFb fixture reading, the validation run, the CLI arm.
//!
//! Mode: `--probe-dsv41-oracle --dsv41-oracle-part dspark --model-dir <native checkpoint or slice with
//! mtp.* + embed + head> --dsv41-dump-ref <REFb fixture dir>[,<dir>...]`. Per fixture it checks
//! (1) `main_x = main_norm(main_proj(main_hidden))` on the prefill rows against REFb's `dspark_main_x`,
//! (2) the main-hidden tap (`mean over the hc copies of x_in` at layers 37 / 38 / 39) against REFb's
//! `dspark_main_hidden` where the layer dumps hold those rows, and (3) every decode step from the dumped
//! inputs (REFb's `decode_main_hidden` and `decode_tokens`, never a chained main model): the ring is seeded
//! from the prompt rows, each step writes its own row and drafts. Each step is compared with REFb's
//! `dspark_drafts` / `dspark_confidence` twice: the free argmax chain (bit-identical drafts are counted) and a
//! chain fed REFb's drafts (every row then sees REFb's Markov input). The gate is the reference's own floor
//! (`GAP_TOL`, `RANK_TOL`, `SIG_TOL`): a row's draft equals REFb's, or REFb's token ranks <= `RANK_TOL` here
//! within `GAP_TOL` logits of the argmax; the confidence after the sigmoid within `SIG_TOL`. Result line `DSV41_ORACLE_DSPARK_OK`
//! only when (3) ran on at least one fixture and held on every step (and main_x / taps equal).

#[path = "dspark_num.rs"]
pub mod dspark_num;
#[path = "dspark_weights.rs"]
pub mod dspark_weights;
#[path = "dspark_fwd.rs"]
pub mod dspark_fwd;

use crate::dsv41::dump::{RefTensor, RefbDump};
use anyhow::{bail, ensure, Context, Result};
use dspark_fwd::{DraftOut, Dspark, Ring};
use dspark_num as num;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The reference's own floor, measured by running DeepSeek's `model.py` DSpark code unchanged with REFb's
/// torch emulation of `kernel.py` on the CPU (`tools/dsv41_ref/od_dspark_check.py`) against REFb's GPU
/// drafts (TileLang kernels) on the same dumped inputs (PLAN/DSV41_OD_REPORT.md section 4): identical drafts
/// on 37 / 48 (short chat), 32 / 32 (agent 4K), 26 / 32 (document 33K) steps; at every disagreement REFb's
/// token was that run's 2nd or 3rd choice (`RANK_TOL` = 2), at most 1.09 logits below its argmax; the
/// confidence after the sigmoid (how DeepSeek's engine uses it) differed by up to 0.19. Bit-identical drafts
/// are therefore not reachable by any other kernel backend; the gate is: every row's draft equals REFb's, or
/// REFb's token ranks <= `RANK_TOL` here within `GAP_TOL` logits of the argmax (the floor's maxima + ~15%).
pub const GAP_TOL: f32 = 1.25;
pub const RANK_TOL: usize = 2;
/// max |sigmoid(confidence) - sigmoid(REFb's confidence)| (floor 0.19, see `GAP_TOL`).
pub const SIG_TOL: f32 = 0.25;

// ------------------------------------------------------------------------------------------------ fixture reading

/// One REFb fixture (`tools/dsv41_ref/ref_forward.py fixture`, layout in `refdump.py`), read through the one REFb
/// reader (`dsv41::dump::RefbDump`; this module used to carry its own copy of it).
pub struct Fixture {
    pub dir: PathBuf,
    pub p: usize,
    /// prompt rows whose main hidden REFb kept (the last `window` rows): (position, [targets x dim])
    pub prefill_mh: Vec<(usize, Vec<f32>)>,
    pub prefill_main_x: Vec<(usize, Vec<f32>)>,
    /// `decode_tokens[0]` sits at position P; step i feeds `decode_tokens[i]` at P + i
    pub decode_tokens: Vec<u32>,
    /// the main model's tap at each decode step (REFb commit e6701f81 and later)
    pub decode_mh: Option<Vec<Vec<f32>>>,
    pub drafts: Vec<Vec<u32>>,
    pub confidence: Vec<Vec<f32>>,
    pub top8: Vec<Vec<Vec<u32>>>,
    dump: RefbDump,
}

/// A float tensor split along its first dimension into rows.
fn rows_f32(t: &RefTensor) -> Result<Vec<Vec<f32>>> {
    ensure!(t.n_rows() > 0 && !t.f.is_empty(), "rows of {:?}", t.shape);
    Ok((0..t.n_rows()).map(|r| t.row_f(r).to_vec()).collect())
}

/// (absolute position, row) pairs of a per-row point and its `.rows` companion.
fn with_rows(t: &RefTensor) -> Result<Vec<(usize, Vec<f32>)>> {
    let r = t.rows.as_ref().with_context(|| format!("{}: no .rows companion", t.name))?;
    let v = rows_f32(t)?;
    ensure!(r.len() == v.len(), "rows companion length");
    Ok(r.iter().map(|&x| x as usize).zip(v).collect())
}

impl Fixture {
    pub fn open(dir: &Path) -> Result<Fixture> {
        let dump = RefbDump::open(dir)?;
        let prompt = dump.global("prompt_ids")?;
        let p = prompt.i.len();
        let mh = dump.global("dspark_main_hidden")?;
        let mx = dump.global("dspark_main_x")?;
        let decode_tokens: Vec<u32> = dump.global("decode_tokens")?.i.into_iter().map(|x| x as u32).collect();
        let drafts_t = dump.global("dspark_drafts")?;
        let n = drafts_t.shape[0];
        let blk1 = drafts_t.shape[1];
        let drafts: Vec<Vec<u32>> = drafts_t.i.chunks(blk1).map(|c| c.iter().map(|&x| x as u32).collect()).collect();
        let confidence = rows_f32(&dump.global("dspark_confidence")?)?;
        let t8 = dump.global("dspark_draft_top8")?;
        let (b, k) = (t8.shape[1], t8.shape[2]);
        let top8: Vec<Vec<Vec<u32>>> = t8.i.chunks(b * k).map(|s| s.chunks(k).map(|r| r.iter().map(|&x| x as u32).collect()).collect()).collect();
        let decode_mh = match dump.global_opt("decode_main_hidden")? { Some(t) => Some(rows_f32(&t)?), None => None };
        if let Some(d) = &decode_mh { ensure!(d.len() >= n, "decode_main_hidden has {} rows for {n} steps", d.len()); }
        ensure!(decode_tokens.len() > n, "decode_tokens");
        let (prefill_mh, prefill_main_x) = (with_rows(&mh)?, with_rows(&mx)?);
        Ok(Fixture { dir: dir.to_path_buf(), p, prefill_mh, prefill_main_x, decode_tokens, decode_mh, drafts, confidence, top8, dump })
    }

    pub fn name(&self) -> String { RefbDump::fixture_name(&self.dir) }

    /// REFb's dumped block input stream `x_in` [rows, hc, dim] of layer `l` (rows that were kept).
    pub fn x_in(&self, l: usize) -> Result<Option<Vec<(usize, Vec<f32>)>>> {
        match self.dump.layer_opt(l, "x_in")? { Some(t) if t.rows.is_some() => Ok(Some(with_rows(&t)?)), _ => Ok(None) }
    }
}

// ------------------------------------------------------------------------------------------------ comparison

/// Elementwise comparison of two bf16-valued vectors.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cmp { pub n: usize, pub equal: usize, pub max_abs: f32, pub num: f64, pub den: f64 }

impl Cmp {
    pub fn add(&mut self, a: &[f32], r: &[f32]) {
        for (x, y) in a.iter().zip(r) {
            self.n += 1;
            if x == y { self.equal += 1; }
            self.max_abs = self.max_abs.max((x - y).abs());
            self.num += ((x - y) as f64).powi(2);
            self.den += (*y as f64).powi(2);
        }
    }
    pub fn rel_l2(&self) -> f64 { if self.den > 0.0 { (self.num / self.den).sqrt() } else { self.num.sqrt() } }
    pub fn line(&self) -> String {
        format!("{} / {} elements equal ({:.4}%), rel-L2 {:.3e}, max |diff| {:.3e}", self.equal, self.n,
                100.0 * self.equal as f64 / self.n.max(1) as f64, self.rel_l2(), self.max_abs)
    }
}

/// One decode step's comparison.
#[derive(Clone, Debug)]
pub struct StepCmp {
    pub step: usize,
    pub pos: usize,
    pub ref_ids: Vec<u32>,
    pub free: DraftOut,
    pub forced: DraftOut,
    pub ref_conf: Vec<f32>,
    pub ref_top8: Vec<Vec<u32>>,
}

impl StepCmp {
    pub fn drafts_equal(&self) -> bool { self.free.ids == self.ref_ids }
    pub fn forced_equal(&self) -> bool { self.forced.ids[1..] == self.ref_ids[1..] }
    /// max |confidence - reference| (forced chain: the same Markov inputs as the reference)
    pub fn conf_diff(&self) -> f32 { self.forced.confidence.iter().zip(&self.ref_conf).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max) }
    /// the same after the sigmoid
    pub fn sig_diff(&self) -> f32 {
        self.forced.confidence.iter().zip(&self.ref_conf).map(|(a, b)| (num::sigmoid32(*a) - num::sigmoid32(*b)).abs()).fold(0.0, f32::max)
    }
    /// rows (forced chain, so every row sees REFb's Markov input) whose draft differs from REFb's:
    /// (row, REFb token's rank here, its gap below this oracle's argmax)
    pub fn flips(&self) -> Vec<(usize, usize, f32)> {
        (0..self.forced.probe.len()).filter(|&j| self.forced.ids[j + 1] != self.ref_ids[j + 1]).map(|j| (j, self.forced.probe[j].0, self.forced.probe[j].1)).collect()
    }
    /// every flip ranks <= `RANK_TOL` within `GAP_TOL` logits and the confidence (after the sigmoid) is within `SIG_TOL`
    pub fn within_floor(&self) -> bool { self.flips().iter().all(|&(_, r, g)| r <= RANK_TOL && g <= GAP_TOL) && self.sig_diff() <= SIG_TOL }
    pub fn top8_equal(&self) -> usize { self.forced.top8.iter().zip(&self.ref_top8).filter(|(a, b)| a == b).count() }
    pub fn top8_same_set(&self) -> usize {
        self.forced.top8.iter().zip(&self.ref_top8).filter(|(a, b)| { let mut x = (*a).clone(); let mut y = (*b).clone(); x.sort(); y.sort(); x == y }).count()
    }
}

/// The results of one fixture.
pub struct FixtureResult {
    pub name: String,
    pub p: usize,
    pub main_x: Cmp,
    pub taps: Vec<(usize, Cmp)>,
    pub steps: Vec<StepCmp>,
    pub drafts_checked: bool,
    pub seconds: f64,
}

impl FixtureResult {
    pub fn drafts_identical(&self) -> usize { self.steps.iter().filter(|s| s.drafts_equal()).count() }
    pub fn conf_max(&self) -> f32 { self.steps.iter().map(|s| s.conf_diff()).fold(0.0, f32::max) }
    pub fn sig_max(&self) -> f32 { self.steps.iter().map(|s| s.sig_diff()).fold(0.0, f32::max) }
    pub fn gap_max(&self) -> f32 { self.steps.iter().flat_map(|s| s.flips()).map(|f| f.2).fold(0.0, f32::max) }
    /// every draft bit-identical to REFb's (not reachable by another kernel backend; reported, not gated)
    pub fn identical(&self) -> bool { self.drafts_checked && self.drafts_identical() == self.steps.len() }
    /// the gate: every step within the reference's own floor, and main_x / taps equal
    pub fn ok(&self) -> bool {
        self.drafts_checked && self.steps.iter().all(|s| s.within_floor())
            && self.main_x.equal as f64 >= 0.9999 * self.main_x.n as f64 && self.taps.iter().all(|(_, t)| t.equal == t.n)
    }
}

/// Per-stage intermediates of every decode step, gathered for `write` (rows = step x block).
pub struct TraceAcc { stages: Vec<Vec<(&'static str, Vec<f32>)>>, rings: Vec<Vec<f32>>, router_ids: Vec<Vec<i32>>, router_w: Vec<Vec<f32>>, head_x: Vec<f32>,
                      main_x: Vec<f32>, draft_ids: Vec<i64>, confidence: Vec<f32>, positions: Vec<i64>, steps: usize }

/// The bf16-valued stream points of a stage, in `StageTrace` field order.
const TRACE_POINTS: [&str; 16] = ["hc_in", "attn_x", "attn_out", "hc_mid", "ffn_x", "ffn_out", "hc_out", "pre_mix_in", "attn_q", "attn_kv", "attn_o", "attn_mix", "ffn_mix",
                                   "attn_qr", "moe_h", "shared_h"];
/// trace points stored as fp32 (the rest are bf16-valued and stored as BF16)
const TRACE_F32: [&str; 4] = ["pre_mix_in", "ffn_pre", "attn_mix", "ffn_mix"];

impl TraceAcc {
    pub fn new(stages: usize) -> TraceAcc {
        TraceAcc { stages: (0..stages).map(|_| TRACE_POINTS.iter().map(|p| (*p, Vec::new())).chain(std::iter::once(("ffn_pre", Vec::new()))).collect()).collect(),
                   rings: vec![Vec::new(); stages], router_ids: vec![Vec::new(); stages], router_w: vec![Vec::new(); stages], head_x: Vec::new(), main_x: Vec::new(),
                   draft_ids: Vec::new(), confidence: Vec::new(), positions: Vec::new(), steps: 0 }
    }

    pub fn push(&mut self, p: usize, main_x: &[f32], body: &dspark_fwd::DraftBody, free: &DraftOut) {
        for (s, tr) in body.stages.iter().enumerate() {
            let src: [&Vec<Vec<f32>>; 17] = [&tr.x_in, &tr.attn_x, &tr.attn_out, &tr.hc_mid, &tr.ffn_x, &tr.ffn_out, &tr.x_out, &tr.pre_mix_in,
                                             &tr.attn_q, &tr.attn_kv, &tr.attn_o, &tr.attn_mix, &tr.ffn_mix, &tr.attn_qr, &tr.moe_h, &tr.shared_h,
                                             &tr.ffn_pre];
            for (slot, rows) in self.stages[s].iter_mut().zip(src) { for r in rows { slot.1.extend_from_slice(r); } }
            for r in &tr.ring { self.rings[s].extend_from_slice(r); }
            for r in &tr.router_ids { self.router_ids[s].extend(r.iter().map(|&x| x as i32)); }
            for r in &tr.router_w { self.router_w[s].extend_from_slice(r); }
        }
        for r in &body.x_head { self.head_x.extend_from_slice(r); }
        self.main_x.extend_from_slice(main_x);
        self.draft_ids.extend(free.ids.iter().map(|&x| x as i64));
        self.confidence.extend_from_slice(&free.confidence);
        self.positions.extend((0..body.x_head.len()).map(|r| (p + 1 + r) as i64));
        self.steps += 1;
    }

    /// Write the I0 dump format (`dump.rs`): layer slots 40.. for the stages, `head` for the head input.
    pub fn write(&self, dir: &Path, o: &Dspark, fx: &Fixture) -> Result<PathBuf> {
        use crate::dsv41::dump::{DumpLayer, DumpWriter};
        let c = &o.w.cfg;
        let (d, hc, blk, k) = (c.hidden as u64, c.hc as u64, c.dspark_block as u64, c.dspark_top_k as u64);
        let rows = self.steps as u64 * blk;
        let mut w = DumpWriter::create(dir, "dsv41 leg OD CPU oracle (src/dsv41/cpu/dspark.rs)", &o.w.ckpt.dir.display().to_string(), "native", "exact", &[], (fx.p as u64, fx.p as u64 + self.steps as u64))?;
        let bf = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect() };
        for (s, pts) in self.stages.iter().enumerate() {
            let l = DumpLayer::Layer((c.layers + s) as u16);
            for (name, v) in pts {
                let width = v.len() as u64 / rows.max(1);
                let shape: Vec<u64> = if width == hc * d { vec![rows, hc, d] } else { vec![rows, width] };
                if TRACE_F32.contains(name) { w.put_f32(l, name, &shape, v)?; } else { w.put(l, name, "BF16", &shape, &bf(v))?; }
            }
            let win = c.window as u64;
            w.put(l, "attn_ring", "BF16", &[self.steps as u64 * win, c.head_dim as u64], &bf(&self.rings[s]))?;
            let ib: Vec<u8> = self.router_ids[s].iter().flat_map(|x| x.to_le_bytes()).collect();
            w.put(l, "router_ids", "I32", &[rows, k], &ib)?;
            w.put_f32(l, "router_w", &[rows, k], &self.router_w[s])?;
        }
        let first = DumpLayer::Layer(c.layers as u16);
        let last = DumpLayer::Layer((c.layers + c.dspark_stages - 1) as u16);
        w.put(first, "main_x", "BF16", &[self.steps as u64, d], &bf(&self.main_x))?;
        w.put(DumpLayer::Head, "x", "BF16", &[rows, d], &bf(&self.head_x))?;
        let ib: Vec<u8> = self.draft_ids.iter().flat_map(|x| x.to_le_bytes()).collect();
        w.put(last, "draft_ids", "I64", &[self.steps as u64, blk + 1], &ib)?;
        w.put_f32(last, "confidence", &[self.steps as u64, blk], &self.confidence)?;
        let pb: Vec<u8> = self.positions.iter().flat_map(|x| x.to_le_bytes()).collect();
        w.put(DumpLayer::Head, "positions", "I64", &[rows], &pb)?;
        w.note("pre_mix_in", "the hc pre-mix this stage's attention uses (the previous stage's ffn_pre; one-hot at stage 0), f32 [rows, hc]");
        w.note("attn_q", "queries after RoPE (block rows), bf16 [rows, heads x head_dim]");
        w.note("attn_kv", "the block rows' kv (kv_norm, RoPE, fp8 quant-dequant), bf16 [rows, head_dim]");
        w.note("attn_o", "sparse-attention output before the inverse RoPE, bf16 [rows, heads x head_dim]");
        w.note("attn_ring", "the stage's window ring at each step after this step's main row was written, slot order (slot = position % window; unwritten slots zero), bf16 [steps x window, head_dim]");
        w.note("attn_mix", "the attention half's hc coefficients per row, fp32 [rows, pre (hc) | post (hc) | comb (hc x hc, comb[j][k] at j*hc+k)]");
        w.note("ffn_mix", "the FFN half's hc coefficients, same layout as attn_mix");
        w.note("attn_qr", "q_norm(wq_a(attn_x)), the input of wq_b, bf16 [rows, q_lora_rank]");
        w.note("moe_h", "the bf16 input of each routed expert's w2 (weight * silu(gate) * up), bf16 [rows, top_k x moe_inter] in top-k slot order");
        w.note("shared_h", "the bf16 input of the shared expert's w2, bf16 [rows, moe_inter]");
        w.note("ffn_pre", "this stage's ffn_pre (the next stage's pre_mix_in, the head's pre-mix after the last stage), f32 [rows, hc]");
        w.note("main_x", "main_norm(main_proj(main_hidden)) of the decode step's main row, one row per step");
        w.note("positions", "the rope position of every block row (p + 1 + r at step p)");
        w.note("draft_ids", "[token at p + 1, drafts for p + 2 .. p + 6] (argmax chain), one row per step");
        w.extra("fixture", serde_json::json!(fx.dir.display().to_string()));
        w.extra("rows", serde_json::json!("row = step * block + block row; step i is position P + i"));
        w.finish()
    }
}

/// Run the oracle over one fixture. `max_steps` limits the decode steps (0 = all). With `trace`, every
/// step's per-stage intermediates are gathered (`TraceAcc`) and returned.
pub fn run_fixture(o: &Dspark, fx: &Fixture, max_steps: usize, log: &mut dyn FnMut(String)) -> Result<FixtureResult> {
    run_fixture_traced(o, fx, max_steps, log, None)
}

pub fn run_fixture_traced(o: &Dspark, fx: &Fixture, max_steps: usize, log: &mut dyn FnMut(String), mut trace: Option<&mut TraceAcc>) -> Result<FixtureResult> {
    let t0 = Instant::now();
    let c = &o.w.cfg;
    let d = c.hidden;
    // (1) main_x on the prompt rows
    let mut main_x = Cmp::default();
    let mine: Vec<(usize, Vec<f32>)> = fx.prefill_mh.iter().map(|(pos, mh)| (*pos, o.main_x(mh))).collect();
    for ((pos, a), (rp, r)) in mine.iter().zip(&fx.prefill_main_x) {
        ensure!(pos == rp, "main_x rows differ: {pos} vs {rp}");
        main_x.add(a, r);
    }
    log(format!("{}: P = {}, main_x on {} prompt rows: {}", fx.name(), fx.p, mine.len(), main_x.line()));
    // (2) the tap: mean over the hc copies of each target layer's input stream
    let mut taps = Vec::new();
    for (ti, &l) in c.dspark_targets.iter().enumerate() {
        let Some(xin) = fx.x_in(l)? else { log(format!("  tap L{l}: no x_in dump, not checked")); continue };
        let mut cm = Cmp::default();
        for (pos, x) in &xin {
            let Some((_, mh)) = fx.prefill_mh.iter().find(|(p, _)| p == pos) else { continue };
            let tap: Vec<f32> = (0..d).map(|e| {
                let s: f64 = (0..c.hc).map(|j| x[j * d + e] as f64).sum();
                num::bf16r((s / c.hc as f64) as f32)
            }).collect();
            cm.add(&tap, &mh[ti * d..(ti + 1) * d]);
        }
        if cm.n > 0 { log(format!("  tap L{l} (mean of x_in over {} copies) vs dspark_main_hidden: {}", c.hc, cm.line())); }
        taps.push((l, cm));
    }
    // (3) the decode steps
    let mut steps = Vec::new();
    let Some(dmh) = &fx.decode_mh else {
        log(format!("  {}: no G.decode_main_hidden in outputs.safetensors (REFb fixtures before e6701f81): drafts NOT checked", fx.name()));
        return Ok(FixtureResult { name: fx.name(), p: fx.p, main_x, taps, steps, drafts_checked: false, seconds: t0.elapsed().as_secs_f64() });
    };
    let mut rings: Vec<Ring> = o.new_rings();
    for (pos, mx) in &mine { o.seed(&mut rings, mx, *pos); }
    let n = if max_steps > 0 { max_steps.min(fx.drafts.len()) } else { fx.drafts.len() };
    for i in 0..n {
        let p = fx.p + i;
        let token = fx.decode_tokens[i + 1];
        ensure!(fx.drafts[i][0] == token, "step {i}: REFb's draft row starts with {} but decode_tokens[{}] = {token}", fx.drafts[i][0], i + 1);
        let mx = o.main_x(&dmh[i]);
        let ts = Instant::now();
        let body = o.draft_body(p, token, &mx, &mut rings)?;
        let free = o.chain(&body, None, Some(&fx.drafts[i]));
        let forced = o.chain(&body, Some(&fx.drafts[i]), Some(&fx.drafts[i]));
        if let Some(t) = trace.as_deref_mut() { t.push(p, &mx, &body, &free); }
        let sc = StepCmp { step: i, pos: p, ref_ids: fx.drafts[i].clone(), free, forced, ref_conf: fx.confidence[i].clone(), ref_top8: fx.top8[i].clone() };
        let routers: Vec<String> = body.stages.iter().map(|t| t.router_ids.iter().map(|r| r.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("/")).collect::<Vec<_>>().join(",")).collect();
        log(format!("  step {i:2} pos {p}: drafts {} (ref {:?}, mine {:?}{}), flips (row, ref rank, gap) {:?} -> {}, conf max|diff| {:.2e} sigmoid {:.2e} (ref {:?} mine {:?}), top8 equal {}/{} (same set {}), margins {:?}, routers [{}] ({:.1} s)",
            if sc.drafts_equal() { "IDENTICAL" } else { "DIFFER" }, &sc.ref_ids[1..], &sc.free.ids[1..],
            if sc.drafts_equal() { String::new() } else { format!(", forced-chain {}", if sc.forced_equal() { "identical" } else { "differs" }) },
            sc.flips().iter().map(|(j, r, g)| (*j, *r, (g * 1000.0).round() / 1000.0)).collect::<Vec<_>>(), if sc.within_floor() { "within floor" } else { "OUTSIDE FLOOR" },
            sc.conf_diff(), sc.sig_diff(), sc.ref_conf, sc.forced.confidence, sc.top8_equal(), sc.ref_top8.len(), sc.top8_same_set(),
            sc.forced.margin.iter().map(|m| (m * 1000.0).round() / 1000.0).collect::<Vec<_>>(), routers.join(" | "), ts.elapsed().as_secs_f64()));
        steps.push(sc);
    }
    Ok(FixtureResult { name: fx.name(), p: fx.p, main_x, taps, steps, drafts_checked: true, seconds: t0.elapsed().as_secs_f64() })
}

// ------------------------------------------------------------------------------------------------ CLI

/// `--probe-dsv41-oracle --dsv41-oracle-part dspark --model-dir <native> --dsv41-dump-ref <fixture>[,<fixture>...]`
pub fn cli(args: &[String]) -> Result<()> {
    let md = crate::arg_value(args, "--model-dir").context("--model-dir <native checkpoint (or a native slice with mtp.*, embed, head)> is required")?;
    let fx = crate::opts::lookup("dsv41-dump-ref").and_then(crate::opts::get).context("--dsv41-dump-ref <REFb fixture dir>[,<dir>...] is required")?;
    let t0 = Instant::now();
    let w = dspark_weights::DsparkW::load(Path::new(md))?;
    println!("DSpark oracle: {} stages, block {}, noise {}, targets {:?}, {} routed experts top-{}, Markov rank {}; resident weights loaded in {:.1} s",
             w.stages.len(), w.cfg.dspark_block, w.cfg.dspark_noise_token, w.cfg.dspark_targets, w.cfg.dspark_routed, w.cfg.dspark_top_k,
             w.cfg.dspark_markov_rank, t0.elapsed().as_secs_f64());
    let trace_dir = crate::opts::lookup("dsv41-oracle-trace").and_then(crate::opts::get).map(PathBuf::from);
    let mut o = Dspark::new(w);
    o.trace = trace_dir.is_some();
    let mut results = Vec::new();
    for dir in fx.split(',').filter(|s| !s.is_empty()) {
        let f = Fixture::open(Path::new(dir)).with_context(|| format!("fixture {dir}"))?;
        let mut acc = TraceAcc::new(o.w.stages.len());
        let r = run_fixture_traced(&o, &f, 0, &mut |l| println!("{l}"), if o.trace { Some(&mut acc) } else { None })?;
        if let (Some(td), true) = (&trace_dir, acc.steps > 0) {
            let p = acc.write(&td.join(f.name()), &o, &f)?;
            println!("  trace of {} steps written: {}", acc.steps, p.display());
        }
        results.push(r);
    }
    println!("\n| fixture | P | steps | drafts identical (free chain) | forced-chain rows identical | flips: REFb's token rank <= {RANK_TOL} within {GAP_TOL} | max flip gap | conf max abs diff (logit) | conf max diff after sigmoid (tol {SIG_TOL}) | steps within floor | main_x equal | s |");
    println!("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|");
    for r in &results {
        let rows: usize = r.steps.iter().map(|s| s.ref_top8.len()).sum();
        let flips: Vec<(usize, usize, f32)> = r.steps.iter().flat_map(|s| s.flips()).collect();
        println!("| {} | {} | {} | {} | {} / {} | {} | {:.3} | {:.3} | {:.4} | {} / {} | {:.4}% | {:.0} |", r.name, r.p, r.steps.len(), r.drafts_identical(),
                 rows - flips.len(), rows, if flips.iter().all(|f| f.1 <= RANK_TOL && f.2 <= GAP_TOL) { "yes" } else { "NO" }, r.gap_max(), r.conf_max(), r.sig_max(),
                 r.steps.iter().filter(|s| s.within_floor()).count(), r.steps.len(), 100.0 * r.main_x.equal as f64 / r.main_x.n.max(1) as f64, r.seconds);
    }
    let checked = results.iter().filter(|r| r.drafts_checked).count();
    if checked == 0 { bail!("no fixture had the decode-step main hidden (G.decode_main_hidden): the drafts were NOT checked (partial run)"); }
    let bad: Vec<&str> = results.iter().filter(|r| r.drafts_checked && !r.ok()).map(|r| r.name.as_str()).collect();
    if !bad.is_empty() { bail!("outside the reference's own floor (a flip where REFb's token ranks below {RANK_TOL} or more than {GAP_TOL} logits under the argmax, or sigmoid(confidence) off by more than {SIG_TOL}, or main_x / taps not equal) on {bad:?}"); }
    let ident = results.iter().filter(|r| r.drafts_checked).all(|r| r.identical());
    println!("drafts bit-identical to REFb's on every step: {} (not a gate: the reference's own floor makes it unreachable for any other kernel backend)", if ident { "yes" } else { "no" });
    if checked < results.len() { println!("note: {} fixture(s) without decode main hidden were checked on main_x / taps only", results.len() - checked); }
    println!("DSV41_ORACLE_DSPARK_OK");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::st::{self, StOut, StSource};

    #[test]
    fn dsv41_od_fixture_reader_reads_refb_layout() {
        let dir = std::env::temp_dir().join(format!("dsv41_od_fx_{}", std::process::id()));
        let pf = dir.join("prefill_exact");
        std::fs::create_dir_all(&pf).unwrap();
        let bf = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect() };
        let i64b = |v: &[i64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let (mh, rows) = (bf(&[1.0, 2.0, 3.0, 4.0]), i64b(&[7, 8]));
        let g = vec![
            StOut { name: "G.dspark_main_hidden".into(), dtype: "BF16".into(), shape: vec![2, 2], src: StSource::Bytes(&mh) },
            StOut { name: "G.dspark_main_hidden.rows".into(), dtype: "I64".into(), shape: vec![2], src: StSource::Bytes(&rows) },
            StOut { name: "G.dspark_main_x".into(), dtype: "BF16".into(), shape: vec![2, 2], src: StSource::Bytes(&mh) },
            StOut { name: "G.dspark_main_x.rows".into(), dtype: "I64".into(), shape: vec![2], src: StSource::Bytes(&rows) },
        ];
        st::write_file(&pf.join("G.safetensors"), &g, &Default::default(), &mut |_| {}).unwrap();
        let (prompt, toks, drafts, conf, t8) = (i64b(&[1; 9]), i64b(&[5, 6, 7]), i64b(&[6, 1, 2, 3, 4, 5, 7, 1, 2, 3, 4, 5]),
                                                f32b(&[0.5; 10]), i64b(&[3; 2 * 5 * 8]));
        let o = vec![
            StOut { name: "G.prompt_ids".into(), dtype: "I64".into(), shape: vec![9], src: StSource::Bytes(&prompt) },
            StOut { name: "G.decode_tokens".into(), dtype: "I64".into(), shape: vec![3], src: StSource::Bytes(&toks) },
            StOut { name: "G.dspark_drafts".into(), dtype: "I64".into(), shape: vec![2, 6], src: StSource::Bytes(&drafts) },
            StOut { name: "G.dspark_confidence".into(), dtype: "F32".into(), shape: vec![2, 5], src: StSource::Bytes(&conf) },
            StOut { name: "G.dspark_draft_top8".into(), dtype: "I64".into(), shape: vec![2, 5, 8], src: StSource::Bytes(&t8) },
        ];
        st::write_file(&dir.join("outputs.safetensors"), &o, &Default::default(), &mut |_| {}).unwrap();
        let fx = Fixture::open(&dir).unwrap();
        assert_eq!(fx.p, 9);
        assert_eq!(fx.prefill_mh, vec![(7, vec![1.0, 2.0]), (8, vec![3.0, 4.0])]);
        assert_eq!(fx.decode_tokens, vec![5, 6, 7]);
        assert_eq!(fx.drafts[1], vec![7, 1, 2, 3, 4, 5]);
        assert_eq!(fx.top8[1][4], vec![3; 8]);
        assert!(fx.decode_mh.is_none(), "an old fixture has no decode main hidden");
        assert!(fx.x_in(37).unwrap().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dsv41_od_floor_gate_accepts_near_ties_only() {
        let out = |ids: Vec<u32>, conf: Vec<f32>, probe: Vec<(usize, f32)>| DraftOut { ids, confidence: conf, top8: vec![vec![0; 8]; 5], margin: vec![1.0; 5], probe };
        let refids = vec![9, 1, 2, 3, 4, 5];
        let mk = |forced: DraftOut, rc: Vec<f32>| StepCmp { step: 0, pos: 10, ref_ids: refids.clone(), free: forced.clone(), forced, ref_conf: rc, ref_top8: vec![vec![0; 8]; 5] };
        let exact = mk(out(refids.clone(), vec![0.5; 5], vec![(0, 0.0); 5]), vec![0.5; 5]);
        assert!(exact.drafts_equal() && exact.flips().is_empty() && exact.within_floor());
        // row 2 flipped, REFb's token is the runner-up 0.3 below: within the floor
        let near = mk(out(vec![9, 1, 2, 7, 4, 5], vec![0.5; 5], vec![(0, 0.0), (0, 0.0), (1, 0.3), (0, 0.0), (0, 0.0)]), vec![0.5; 5]);
        assert_eq!(near.flips(), vec![(2, 1, 0.3)]);
        assert!(!near.drafts_equal() && near.within_floor());
        // a flip where REFb's token is far down the list, or far below the argmax, is outside
        let far = mk(out(vec![9, 1, 2, 7, 4, 5], vec![0.5; 5], vec![(0, 0.0), (0, 0.0), (5, 0.3), (0, 0.0), (0, 0.0)]), vec![0.5; 5]);
        assert!(!far.within_floor());
        let wide = mk(out(vec![9, 1, 2, 7, 4, 5], vec![0.5; 5], vec![(0, 0.0), (0, 0.0), (1, GAP_TOL + 0.5), (0, 0.0), (0, 0.0)]), vec![0.5; 5]);
        assert!(!wide.within_floor());
        // confidence is compared after the sigmoid: a saturated logit difference is harmless, a mid-range one is not
        let sat = mk(out(refids.clone(), vec![15.0; 5], vec![(0, 0.0); 5]), vec![16.5; 5]);
        assert!(sat.conf_diff() > 1.0 && sat.within_floor());
        let mid = mk(out(refids.clone(), vec![0.0; 5], vec![(0, 0.0); 5]), vec![1.5; 5]);
        assert!(!mid.within_floor());
    }

    /// The full check against REFb's fixtures. Needs the native checkpoint on this box and the
    /// fixtures; skips (passes with a note) when either is absent. Run explicitly:
    /// `cargo test --release --lib dsv41_od_drafts_match_refb -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dsv41_od_drafts_match_refb() {
        let md = Path::new("/models/DeepSeek-V4.1-Flash-1017");
        let fxs = ["/mnt/models/dsv41/ref/short_chat", "/mnt/models/dsv41/ref/agent_4k", "/mnt/models/dsv41/ref/doc_32k"];
        if !md.join("model.safetensors.index.json").exists() { eprintln!("SKIP: no native checkpoint at {}", md.display()); return; }
        let have: Vec<&str> = fxs.iter().copied().filter(|f| Path::new(f).join("outputs.safetensors").exists()).collect();
        if have.is_empty() { eprintln!("SKIP: no REFb fixture"); return; }
        let o = Dspark::new(dspark_weights::DsparkW::load(md).unwrap());
        let mut checked = 0;
        for f in have {
            let fx = Fixture::open(Path::new(f)).unwrap();
            let r = run_fixture(&o, &fx, 0, &mut |l| eprintln!("{l}")).unwrap();
            assert!(r.main_x.equal as f64 >= 0.999 * r.main_x.n as f64, "{f}: main_x {}", r.main_x.line());
            if r.drafts_checked {
                checked += 1;
                assert!(r.ok(), "{f}: drafts identical {}/{}, max flip gap {}, sigmoid(confidence) max diff {}", r.drafts_identical(), r.steps.len(), r.gap_max(), r.sig_max());
            }
        }
        eprintln!("fixtures with drafts checked: {checked}");
    }
}
