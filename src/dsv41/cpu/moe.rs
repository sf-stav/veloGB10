//! DSV4.1 CPU oracle — leg OF, part 1: the MoE FFN (384 routed experts top-6 + 1 shared, SwiGLU limit 10).
//! Semantics from `model.py` `Gate` 794-829, `Expert` 832-853, `MoE` 856-906 (REF map §3.6 and §4 items 12-15):
//!
//! * gate: fp32 logits, `sqrt(softplus(.))`, selection on `score + bias`, weights from the raw scores,
//!   `w /= sum + 1e-20`, `w *= 1.5`;
//! * expert: `gate = bf16(w1 x)`, `up = bf16(w3 x)`, `up` clamped to +-10, `gate` clamped above at 10,
//!   `h = silu(gate) * up`, `h *= routed weight` (routed experts only, in fp32, before the bf16 cast), `out = bf16(w2 bf16(h))`
//!   (every linear quantises its input with `act_quant`);
//! * `y` (fp32) accumulates each routed expert's bf16 output in ASCENDING expert id, then the shared expert's
//!   bf16 output is added, then one cast to bf16.
//!
//! The inner-width split of T0 §3.1 is `expert_forward_split` / `LayerMoe::forward_split`: each expert's inner width
//! (2304) is cut into `parts` slices, every slice computes its slice of `h` and a partial `w2`, and the partials
//! are summed in fp32 (the per-expert bf16 rounding is therefore NOT reproduced; measured by the probe).

#[path = "engram_mod.rs"]
pub mod engram_mod;
#[path = "moe_load.rs"]
pub mod load;
#[path = "oracle_cli.rs"]
pub mod oracle_cli;
#[path = "moe_quant.rs"]
pub mod quant;

use anyhow::{ensure, Result};
use load::Store;
use quant::*;
use std::collections::HashMap;

pub const DIM: usize = 5120;
pub const INTER: usize = 2304;
pub const SWIGLU_LIMIT: f32 = 10.0;
pub const TOPK: usize = 6;
pub const ROUTE_SCALE: f32 = 1.5;

/// Gate parameters of one layer.
pub struct GateW { pub weight: Vec<f32>, pub bias: Vec<f32>, pub n_experts: usize }

impl GateW {
    pub fn load(st: &Store, layer: usize) -> Result<GateW> {
        let weight = st.f32s(&format!("layers.{layer}.ffn.gate.weight"))?;
        let bias = st.f32s(&format!("layers.{layer}.ffn.gate.bias"))?;
        ensure!(weight.len() == bias.len() * DIM, "gate weight {} vs bias {}", weight.len(), bias.len());
        Ok(GateW { n_experts: bias.len(), weight, bias })
    }
}

/// torch `F.softplus` (beta 1, threshold 20) in f32.
#[inline]
pub fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { x.exp().ln_1p() } }

/// Per row: the six chosen expert ids (descending biased score, ties to the lower id), their routed weights, and the
/// biased-score margin between the 6th and 7th candidate (near zero = a tie the reference resolves by torch's order).
pub struct Routing { pub idx: Vec<[usize; TOPK]>, pub w: Vec<[f32; TOPK]>, pub margin: Vec<f32> }

pub fn route(g: &GateW, x: &[f32], n: usize) -> Routing {
    let ne = g.n_experts;
    let (mut idx, mut wv, mut margin) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for r in 0..n {
        let xr = &x[r * DIM..(r + 1) * DIM];
        let scores: Vec<f32> = (0..ne).map(|e| {
            let w = &g.weight[e * DIM..(e + 1) * DIM];
            let mut s = 0f64;
            for i in 0..DIM { s += xr[i] as f64 * w[i] as f64; }
            softplus(s as f32).sqrt()
        }).collect();
        let biased: Vec<f32> = (0..ne).map(|e| scores[e] + g.bias[e]).collect();
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_by(|&a, &b| biased[b].partial_cmp(&biased[a]).unwrap().then(a.cmp(&b)));
        let (mut i6, mut w6, mut sum) = ([0usize; TOPK], [0f32; TOPK], 0f32);
        for j in 0..TOPK { i6[j] = order[j]; w6[j] = scores[order[j]]; sum += w6[j]; }
        for j in 0..TOPK { w6[j] = w6[j] / (sum + 1e-20) * ROUTE_SCALE; }
        margin.push(biased[order[TOPK - 1]] - biased[order[TOPK]]);
        idx.push(i6);
        wv.push(w6);
    }
    Routing { idx, w: wv, margin }
}

/// One expert's three linears.
pub struct ExpertW { pub w1: Box<dyn QRows>, pub w2: Box<dyn QRows>, pub w3: Box<dyn QRows> }

pub struct LayerMoe<'a> { pub st: &'a Store, pub layer: usize, pub gate: GateW, experts: HashMap<usize, ExpertW>, shared: Option<ExpertW> }

impl<'a> LayerMoe<'a> {
    pub fn new(st: &'a Store, layer: usize) -> Result<Self> {
        Ok(LayerMoe { st, layer, gate: GateW::load(st, layer)?, experts: HashMap::new(), shared: None })
    }
    fn load_expert(&mut self, e: usize) -> Result<()> {
        if !self.experts.contains_key(&e) {
            let b = format!("layers.{}.ffn.experts.{e}", self.layer);
            let w = ExpertW { w1: Box::new(self.st.fp4(&format!("{b}.w1"))?), w2: Box::new(self.st.fp4(&format!("{b}.w2"))?), w3: Box::new(self.st.fp4(&format!("{b}.w3"))?) };
            self.experts.insert(e, w);
        }
        Ok(())
    }
    fn load_shared(&mut self) -> Result<()> {
        if self.shared.is_none() {
            let b = format!("layers.{}.ffn.shared_experts", self.layer);
            self.shared = Some(ExpertW { w1: Box::new(self.st.fp8(&format!("{b}.w1"))?), w2: Box::new(self.st.fp8(&format!("{b}.w2"))?), w3: Box::new(self.st.fp8(&format!("{b}.w3"))?) });
        }
        Ok(())
    }

    /// The whole FFN for `n` rows with the given routing. `split` = `None` is the reference; `Some(parts)` is the
    /// inner-width split (partials in fp32, no per-expert rounding). Returns the bf16-rounded output `[n, DIM]`.
    pub fn forward(&mut self, x: &[f32], n: usize, rt: &Routing, split: Option<usize>) -> Result<Vec<f32>> {
        let mut y = vec![0f32; n * DIM];
        // rows grouped by expert, experts walked in ascending id (the reference's order of accumulation)
        let mut by_e: std::collections::BTreeMap<usize, Vec<(usize, f32)>> = Default::default();
        for r in 0..n { for j in 0..TOPK { by_e.entry(rt.idx[r][j]).or_default().push((r, rt.w[r][j])); } }
        for (e, rows) in by_e {
            self.load_expert(e)?;
            let xs: Vec<f32> = rows.iter().flat_map(|&(r, _)| x[r * DIM..(r + 1) * DIM].iter().copied()).collect();
            let ws: Vec<f32> = rows.iter().map(|&(_, w)| w).collect();
            let w = &self.experts[&e];
            let out = match split { None => expert_forward(w, &xs, rows.len(), Some(&ws)), Some(p) => expert_forward_split(w, &xs, rows.len(), Some(&ws), p) };
            for (i, &(r, _)) in rows.iter().enumerate() { for c in 0..DIM { y[r * DIM + c] += out[i * DIM + c]; } }
        }
        self.load_shared()?;
        let w = self.shared.as_ref().unwrap();
        let sh = match split { None => expert_forward(w, x, n, None), Some(p) => expert_forward_split(w, x, n, None, p) };
        for i in 0..n * DIM { y[i] = bf16r(y[i] + sh[i]); }
        Ok(y)
    }
}

/// `Expert.forward` for `m` rows; `weights` = the routed weight per row (None for the shared expert). Output bf16-exact f32.
pub fn expert_forward(w: &ExpertW, x: &[f32], m: usize, weights: Option<&[f32]>) -> Vec<f32> {
    let a = act_quant(x, m, DIM);
    let gate = qgemm_f32(&a, &*w.w1, None);
    let up = qgemm_f32(&a, &*w.w3, None);
    let h = swiglu(&gate, &up, m, weights);
    qlinear_bf16(&h, m, &*w.w2)
}

/// `bf16( weight * silu(min(bf16(gate), 10)) * clamp(bf16(up), -10, 10) )` row-wise (the bf16 casts of the two linears included).
pub fn swiglu(gate: &[f32], up: &[f32], m: usize, weights: Option<&[f32]>) -> Vec<f32> {
    let mut h = vec![0f32; m * INTER];
    for r in 0..m {
        for i in 0..INTER {
            let g = bf16r(gate[r * INTER + i]).min(SWIGLU_LIMIT);
            let u = bf16r(up[r * INTER + i]).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
            let mut v = (g / (1.0 + (-g).exp())) * u;
            if let Some(w) = weights { v *= w[r]; }
            h[r * INTER + i] = bf16r(v);
        }
    }
    h
}

/// T0 §3.1: the expert's inner width cut into `parts` equal slices. Slice `p` owns rows `[p*part, (p+1)*part)` of
/// `w1` / `w3` and the matching K range of `w2`; `act_quant` of `h` is per group of 32 along the inner width, so a
/// slice that is a multiple of 32 quantises exactly as the whole. The returned partial sums are fp32 (slice order),
/// NOT rounded to bf16: the only differences from `expert_forward` are the fp32 summation order and the missing
/// per-expert bf16 rounding.
pub fn expert_forward_split(w: &ExpertW, x: &[f32], m: usize, weights: Option<&[f32]>, parts: usize) -> Vec<f32> {
    let part = INTER / parts;
    assert_eq!(part % GROUP, 0);
    let a = act_quant(x, m, DIM);
    let gate = qgemm_f32(&a, &*w.w1, None);
    let up = qgemm_f32(&a, &*w.w3, None);
    let h = swiglu(&gate, &up, m, weights);
    let ha = act_quant(&h, m, INTER);
    let mut acc = vec![0f32; m * DIM];
    for p in 0..parts {
        let kbs: Vec<usize> = (p * part / GROUP..(p + 1) * part / GROUP).collect();
        let pr = qgemm_f32(&ha, &*w.w2, Some(&kbs));
        for i in 0..m * DIM { acc[i] += pr[i]; }
    }
    acc
}
