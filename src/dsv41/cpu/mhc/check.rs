//! DSV4.1 OM oracle: validation against REFb's reference dumps
//! (`--probe-dsv41-oracle --dsv41-oracle-part mhc --model-dir <pack> --dsv41-dump-ref <dump> [--layers 0-39]`).
//!
//! For every dumped row of every layer, three comparisons:
//!
//! * **step**: each op from the dump's own inputs (mixer coefficients from the dumped stream, `hc_pre`
//!   with the dumped pre-mix, `hc_post` with the dumped coefficients and sublayer output, the router
//!   on the dumped `ffn_in`). Isolates each op's semantics.
//! * **own**: the dumped streams with THIS oracle's coefficients (`x_mid`, `ffn_in`, the block output).
//! * **chain** (INFORMATIONAL, never part of the verdict): this oracle's stream carried from layer 0
//!   through every layer, with the attention and MoE outputs teacher-forced from the dump (legs OA /
//!   OF) and the Engram output taken from the dump at the Engram layers (leg OF); the pre-mix is this
//!   oracle's own `ffn_pre`. It shows that the mixer-only path stays at bf16-rounding level over 40
//!   layers. It is no gate: REFb (report §6.2) measured that a summation-order change alone flips
//!   ~10 % of the expert choices by layer 2 once the sublayers are not teacher-forced, so every
//!   verdict below judges a sublayer on its DUMPED input.
//!
//! Negative controls (the success signal must be visible): the delayed pre-mix is proven by the dump
//! itself, because the WRONG wirings (the FFN collapsing with its own `ffn_pre`; the attention with
//! the block's own `attn_pre`) must miss `ffn_in` / `attn_in` on most elements while the right ones hit.
//!
//! Plus the wiring checks of the dump itself (layer 0 uses the one-hot pre-mix; every later layer's
//! `pre_mix_in` is bit-identical to the previous layer's `ffn_pre`), the head's `hc_pre` (`h_final`),
//! the DSpark tap (`stream_mean` of layers 37-39's inputs), the TP K split of the mixes, and (packs
//! with a plain head) the final norm through the head to the dumped fp32 logits.
//!
//! Bounds (the verdict; `BOUNDS`): stated before the measurement from the fp32 noise of the ops the
//! oracle cannot order like the GPU (the 20,480-long mix dot products, the 5,120-long norm and gate
//! reductions, `rsqrtf`), see report §3.

use super::num::{rel_l2, Cmp};
use super::params::PackParams;
use crate::dsv41::dump::{RefTensor, RefbDump};
use super::router::{route, GateParams, Route};
use super::{hc_coeffs, hc_mix_partial, hc_mixes, hc_mixes_from_partials, hc_post, hc_pre, hc_split_sinkhorn, identity_pre_mix, k_ranges, norm, stream_mean, BlockParams, Coeffs, HcShape};
use anyhow::{ensure, Context, Result};
use std::fmt::Write as _;
use std::path::Path;

/// Rows whose k-th and (k+1)-th router selection keys differ by less than this (relative to the
/// k-th key, floor 1) are near-ties: an fp32-order difference can flip them (REF map §4 item 7).
pub const NEAR_TIE_REL: f32 = 1e-5;

/// The verdict bounds (name, meaning). Checked in `Report::verdict`.
pub const BOUNDS: &[(&str, &str)] = &[
    ("wiring", "the dump's layer-0 pre-mix is one-hot and every later pre_mix_in equals the previous layer's ffn_pre bit for bit"),
    ("hc_post exact", "step x_mid and step block output (vs the next layer's input) bitwise equal on every element"),
    ("hc_pre exact", "step h_final (hc_pre of the last block output with its ffn_pre) and the DSpark tap bitwise equal"),
    ("norms", "step attn_in / ffn_in: at most 1e-4 of the elements differ, by at most 1 bf16 ulp"),
    ("coefficients", "step mixer coefficients (pre / post / comb): rel-L2 <= 1e-6 and max |diff| <= 1e-5 per layer"),
    ("own streams", "x_mid / ffn_in / block out from the dumped streams with THIS oracle's coefficients: at most 1e-3 of the elements differ, rel-L2 <= 4e-4 (a 1e-3 fraction of one-ulp bf16 flips gives ~1.2e-4)"),
    ("router", "on the dumped ffn_in, per layer: top-k sets identical on every row that is not a near-tie (margin < 1e-5 rel); weights on rows with identical ids rel-L2 <= 1e-6"),
    ("delayed pre-mix controls", "the wrong wirings (FFN with ffn_pre; attention with this block's attn_pre) miss more than half of the elements on layers >= 1"),
    ("tp split", "mixes from 2 / 4 / 8 K-slice partials summed in rank order: max relative difference to the unsplit mixes <= 1e-5"),
    ("head", "packs with a plain head: final norm + head on the dumped h_final, logits rel-L2 <= 1e-5 per row and the same argmax"),
];

/// What to check.
#[derive(Clone, Debug)]
pub struct CheckOpts {
    /// layers to check (None = every dumped trunk layer); the chain runs only over a full 0..L prefix
    pub layers: Option<Vec<usize>>,
    /// rows for the head check (0 = skip)
    pub head_rows: usize,
    /// worker threads
    pub threads: usize,
}

impl Default for CheckOpts {
    fn default() -> Self { CheckOpts { layers: None, head_rows: 4, threads: std::thread::available_parallelism().map_or(4, |n| n.get()).min(8) } }
}

/// Router statistics of one layer.
#[derive(Clone, Debug, Default)]
pub struct RouterStats {
    pub rows: u64,
    /// ids identical in order
    pub ids_equal: u64,
    /// ids identical as sets
    pub set_equal: u64,
    /// rows whose set differs (any)
    pub flips: u64,
    /// ... of which at a near-tie (allowed)
    pub flips_near: u64,
    /// near-tie rows (whatever the outcome)
    pub near: u64,
    /// exact ties at the selection boundary (margin 0 in this oracle's keys)
    pub ties: u64,
    /// smallest relative margin seen
    pub min_rel_margin: f64,
    /// weights on rows with identical ids
    pub w: Cmp,
}

impl RouterStats {
    fn new() -> Self { RouterStats { min_rel_margin: f64::INFINITY, ..Default::default() } }
    fn add(&mut self, ours: &Route, ref_ids: &[i64], ref_w: &[f32]) {
        self.rows += 1;
        let rel = ours.margin as f64 / (ours.kth_key.abs() as f64).max(1.0);
        self.min_rel_margin = self.min_rel_margin.min(rel);
        let near = (rel as f32) < NEAR_TIE_REL;
        if near {
            self.near += 1;
        }
        if ours.margin == 0.0 {
            self.ties += 1;
        }
        let ids: Vec<i64> = ours.ids.iter().map(|&v| v as i64).collect();
        let mut a = ids.clone();
        let mut b = ref_ids.to_vec();
        a.sort();
        b.sort();
        if ids == ref_ids {
            self.ids_equal += 1;
            self.w.f32(&ours.weights, ref_w);
        }
        if a == b {
            self.set_equal += 1;
        } else {
            self.flips += 1;
            if near {
                self.flips_near += 1;
            }
        }
    }
    fn merge(&mut self, o: &RouterStats) {
        self.rows += o.rows;
        self.ids_equal += o.ids_equal;
        self.set_equal += o.set_equal;
        self.flips += o.flips;
        self.flips_near += o.flips_near;
        self.near += o.near;
        self.ties += o.ties;
        self.min_rel_margin = self.min_rel_margin.min(o.min_rel_margin);
        self.w.merge(&o.w);
    }
    fn cell(&self) -> String {
        format!(
            "{}/{} sets equal ({} same order); flips {} ({} at near-ties); near-ties {}; exact ties {}; min margin {:.1e}; w {}",
            self.set_equal, self.rows, self.ids_equal, self.flips, self.flips_near, self.near, self.ties, self.min_rel_margin, self.w.cell()
        )
    }
}

/// Every statistic of one layer.
#[derive(Clone, Debug, Default)]
pub struct LayerStats {
    pub layer: usize,
    pub rows: u64,
    pub wiring: Cmp,
    pub attn_coef: Cmp,
    pub ffn_coef: Cmp,
    pub attn_in: Cmp,
    pub x_mid: Cmp,
    pub ffn_in: Cmp,
    pub x_out: Cmp,
    pub own_x_mid: Cmp,
    pub own_ffn_in: Cmp,
    pub own_x_out: Cmp,
    pub chain_x_in: Cmp,
    pub chain_coef: Cmp,
    pub chain_attn_in: Cmp,
    pub chain_x_mid: Cmp,
    pub chain_ffn_in: Cmp,
    pub router: RouterStats,
    pub chain_router: RouterStats,
    /// negative controls: ffn_in collapsed with ffn_pre, attn_in collapsed with this block's attn_pre
    pub ctl_ffn_wrong: Cmp,
    pub ctl_attn_wrong: Cmp,
}

impl LayerStats {
    fn new(layer: usize) -> Self { LayerStats { layer, router: RouterStats::new(), chain_router: RouterStats::new(), ..Default::default() } }
    fn merge(&mut self, o: &LayerStats) {
        self.rows += o.rows;
        for (a, b) in [
            (&mut self.wiring, &o.wiring),
            (&mut self.attn_coef, &o.attn_coef),
            (&mut self.ffn_coef, &o.ffn_coef),
            (&mut self.attn_in, &o.attn_in),
            (&mut self.x_mid, &o.x_mid),
            (&mut self.ffn_in, &o.ffn_in),
            (&mut self.x_out, &o.x_out),
            (&mut self.own_x_mid, &o.own_x_mid),
            (&mut self.own_ffn_in, &o.own_ffn_in),
            (&mut self.own_x_out, &o.own_x_out),
            (&mut self.chain_x_in, &o.chain_x_in),
            (&mut self.chain_coef, &o.chain_coef),
            (&mut self.chain_attn_in, &o.chain_attn_in),
            (&mut self.chain_x_mid, &o.chain_x_mid),
            (&mut self.chain_ffn_in, &o.chain_ffn_in),
            (&mut self.ctl_ffn_wrong, &o.ctl_ffn_wrong),
            (&mut self.ctl_attn_wrong, &o.ctl_attn_wrong),
        ] {
            a.merge(b);
        }
        self.router.merge(&o.router);
        self.chain_router.merge(&o.chain_router);
    }
}

/// The result of a check run.
pub struct Report {
    pub lines: Vec<String>,
    pub failures: Vec<String>,
    pub layers: Vec<LayerStats>,
}

fn coef_vec(c: &Coeffs) -> Vec<f32> { c.pre.iter().chain(&c.post).chain(&c.comb).copied().collect() }

fn dumped_coef(pre: &RefTensor, post: &RefTensor, comb: &RefTensor, i: usize) -> Vec<f32> {
    pre.row_f(i).iter().chain(post.row_f(i)).chain(comb.row_f(i)).copied().collect()
}

/// The dumped tensors of one layer that the checks read.
struct LayerDump {
    x_in: RefTensor,
    pre_mix_in: RefTensor,
    attn_pre: RefTensor,
    attn_post: RefTensor,
    attn_comb: RefTensor,
    attn_in: RefTensor,
    attn_out: RefTensor,
    x_mid: RefTensor,
    ffn_pre: RefTensor,
    ffn_post: RefTensor,
    ffn_comb: RefTensor,
    ffn_in: RefTensor,
    ffn_out: RefTensor,
    router_idx: RefTensor,
    router_w: RefTensor,
    x_pre_engram: Option<RefTensor>,
    /// the stream the block output must equal: the next layer's x_pre_engram (Engram layer) or x_in
    next_in: Option<RefTensor>,
}

fn load_layer(d: &RefbDump, l: usize, last: usize) -> Result<LayerDump> {
    let g = |p: &str| d.layer(l, p);
    let next_in = if l < last {
        match d.layer_opt(l + 1, "x_pre_engram")? {
            Some(t) => Some(t),
            None => d.layer_opt(l + 1, "x_in")?,
        }
    } else {
        None
    };
    Ok(LayerDump {
        x_in: g("x_in")?,
        pre_mix_in: g("pre_mix_in")?,
        attn_pre: g("attn_pre")?,
        attn_post: g("attn_post")?,
        attn_comb: g("attn_comb")?,
        attn_in: g("attn_in")?,
        attn_out: g("attn_out")?,
        x_mid: g("x_mid")?,
        ffn_pre: g("ffn_pre")?,
        ffn_post: g("ffn_post")?,
        ffn_comb: g("ffn_comb")?,
        ffn_in: g("ffn_in")?,
        ffn_out: g("ffn_out")?,
        router_idx: g("router_idx")?,
        router_w: g("router_w")?,
        x_pre_engram: d.layer_opt(l, "x_pre_engram")?,
        next_in,
    })
}

fn idx(t: &RefTensor, pos: i64) -> Result<usize> { t.index_of(pos).with_context(|| format!("{}: no row at position {pos}", t.name)) }

/// The chain state of one row (this oracle's own stream and pre-mix).
#[derive(Clone)]
struct ChainRow {
    x: Vec<f32>,
    pm: Vec<f32>,
}

/// Check one row of one layer; updates the chain state.
#[allow(clippy::too_many_arguments)]
fn check_row(s: &HcShape, b: &BlockParams, g: &GateParams, ld: &LayerDump, prev_ffn_pre: Option<&RefTensor>, pos: i64, chain: Option<&mut ChainRow>, st: &mut LayerStats) -> Result<()> {
    let i = idx(&ld.x_in, pos)?;
    let x_in = ld.x_in.row_f(i);
    st.rows += 1;
    // ---- wiring of the dump: the delayed pre-mix ----
    let pm_in = ld.pre_mix_in.row_f(idx(&ld.pre_mix_in, pos)?);
    match prev_ffn_pre {
        Some(t) => st.wiring.f32(pm_in, t.row_f(idx(t, pos)?)),
        None if st.layer == 0 => st.wiring.f32(pm_in, &identity_pre_mix(s.hc)),
        None => {}
    }
    // ---- step ----
    let ia = idx(&ld.attn_pre, pos)?;
    let ac = hc_coeffs(s, &b.attn_mix, x_in);
    st.attn_coef.f32(&coef_vec(&ac), &dumped_coef(&ld.attn_pre, &ld.attn_post, &ld.attn_comb, ia));
    let attn_in = norm::rms_norm(&hc_pre(s, x_in, pm_in), &b.attn_norm, s.norm_eps);
    st.attn_in.bf16(&attn_in, ld.attn_in.row_f(idx(&ld.attn_in, pos)?));
    let ao = ld.attn_out.row_f(idx(&ld.attn_out, pos)?);
    let x_mid_d = ld.x_mid.row_f(idx(&ld.x_mid, pos)?);
    let x_mid = hc_post(s, ao, x_in, ld.attn_post.row_f(ia), ld.attn_comb.row_f(ia));
    st.x_mid.bf16(&x_mid, x_mid_d);
    let iff = idx(&ld.ffn_pre, pos)?;
    let fc = hc_coeffs(s, &b.ffn_mix, x_mid_d);
    st.ffn_coef.f32(&coef_vec(&fc), &dumped_coef(&ld.ffn_pre, &ld.ffn_post, &ld.ffn_comb, iff));
    let ffn_in_d = ld.ffn_in.row_f(idx(&ld.ffn_in, pos)?);
    let ffn_in = norm::rms_norm(&hc_pre(s, x_mid_d, ld.attn_pre.row_f(ia)), &b.ffn_norm, s.norm_eps);
    st.ffn_in.bf16(&ffn_in, ffn_in_d);
    if st.layer >= 1 {
        // negative controls of the delayed pre-mix (must MISS)
        st.ctl_ffn_wrong.bf16(&norm::rms_norm(&hc_pre(s, x_mid_d, ld.ffn_pre.row_f(iff)), &b.ffn_norm, s.norm_eps), ffn_in_d);
        st.ctl_attn_wrong.bf16(&norm::rms_norm(&hc_pre(s, x_in, ld.attn_pre.row_f(ia)), &b.attn_norm, s.norm_eps), ld.attn_in.row_f(idx(&ld.attn_in, pos)?));
    }
    let ir = idx(&ld.router_idx, pos)?;
    let ref_ids = ld.router_idx.row_i(ir);
    let ref_w = ld.router_w.row_f(idx(&ld.router_w, pos)?);
    st.router.add(&route(g, ffn_in_d, false), ref_ids, ref_w);
    let fo = ld.ffn_out.row_f(idx(&ld.ffn_out, pos)?);
    let x_out = hc_post(s, fo, x_mid_d, ld.ffn_post.row_f(iff), ld.ffn_comb.row_f(iff));
    let next = match &ld.next_in { Some(t) => Some(t.row_f(idx(t, pos)?)), None => None };
    if let Some(n) = next {
        st.x_out.bf16(&x_out, n);
    }
    // ---- own coefficients on the dumped streams ----
    st.own_x_mid.bf16(&hc_post(s, ao, x_in, &ac.post, &ac.comb), x_mid_d);
    st.own_ffn_in.bf16(&norm::rms_norm(&hc_pre(s, x_mid_d, &ac.pre), &b.ffn_norm, s.norm_eps), ffn_in_d);
    if let Some(n) = next {
        st.own_x_out.bf16(&hc_post(s, fo, x_mid_d, &fc.post, &fc.comb), n);
    }
    // ---- chain ----
    if let Some(c) = chain {
        match &ld.x_pre_engram {
            Some(t) => {
                // Engram layer: compare the carried stream with the Engram input, then take the
                // dumped Engram output (leg OF owns the Engram module)
                st.chain_x_in.bf16(&c.x, t.row_f(idx(t, pos)?));
                c.x = x_in.to_vec();
            }
            None => st.chain_x_in.bf16(&c.x, x_in),
        }
        let cac = hc_coeffs(s, &b.attn_mix, &c.x);
        st.chain_coef.f32(&coef_vec(&cac), &dumped_coef(&ld.attn_pre, &ld.attn_post, &ld.attn_comb, ia));
        let c_attn_in = norm::rms_norm(&hc_pre(s, &c.x, &c.pm), &b.attn_norm, s.norm_eps);
        st.chain_attn_in.bf16(&c_attn_in, ld.attn_in.row_f(idx(&ld.attn_in, pos)?));
        let c_mid = hc_post(s, ao, &c.x, &cac.post, &cac.comb);
        st.chain_x_mid.bf16(&c_mid, x_mid_d);
        let cfc = hc_coeffs(s, &b.ffn_mix, &c_mid);
        st.chain_coef.f32(&coef_vec(&cfc), &dumped_coef(&ld.ffn_pre, &ld.ffn_post, &ld.ffn_comb, iff));
        let c_ffn_in = norm::rms_norm(&hc_pre(s, &c_mid, &cac.pre), &b.ffn_norm, s.norm_eps);
        st.chain_ffn_in.bf16(&c_ffn_in, ffn_in_d);
        st.chain_router.add(&route(g, &c_ffn_in, false), ref_ids, ref_w);
        c.x = hc_post(s, fo, &c_mid, &cfc.post, &cfc.comb);
        c.pm = cfc.pre;
    }
    Ok(())
}

/// Run `f` over `items` on up to `threads` scoped threads, each with its own accumulator.
fn par_rows<T: Send, A: Send>(items: &mut [T], threads: usize, mk: impl Fn() -> A + Sync, f: impl Fn(&mut T, &mut A) -> Result<()> + Sync) -> Result<Vec<A>> {
    let n = items.len().max(1);
    let per = (n + threads.max(1) - 1) / threads.max(1);
    std::thread::scope(|sc| {
        let hs: Vec<_> = items
            .chunks_mut(per.max(1))
            .map(|chunk| {
                let (mk, f) = (&mk, &f);
                sc.spawn(move || -> Result<A> {
                    let mut acc = mk();
                    for it in chunk.iter_mut() {
                        f(it, &mut acc)?;
                    }
                    Ok(acc)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().map_err(|_| anyhow::anyhow!("worker panicked"))?).collect()
    })
}

/// Run every check over a dump. `pack` supplies the parameters (native or any EXL3 pack).
pub fn run(pack: &Path, dump: &Path, o: &CheckOpts) -> Result<Report> {
    let t0 = std::time::Instant::now();
    let pp = PackParams::open(pack)?;
    let s = pp.shape;
    let d = RefbDump::open(dump)?;
    let dumped = d.layers();
    ensure!(!dumped.is_empty(), "{}: no dumped layers", d.dir.display());
    let trunk = pp.census.cfg.layers;
    let layers: Vec<usize> = match &o.layers {
        Some(v) => v.clone(),
        None => dumped.iter().copied().filter(|&l| l < trunk).collect(),
    };
    for &l in &layers {
        ensure!(dumped.contains(&l), "layer {l} is not in the dump (dumped: {dumped:?})");
    }
    let last_dumped = *dumped.iter().filter(|&&l| l < trunk).max().unwrap();
    let full_chain = layers.first() == Some(&0) && layers.windows(2).all(|w| w[1] == w[0] + 1);
    let positions: Vec<i64> = d.layer(layers[0], "x_in")?.positions();
    let mut chain: Vec<(i64, Option<ChainRow>)> = positions.iter().map(|&p| (p, None)).collect();
    if full_chain {
        let x0 = d.layer(0, "x_in")?;
        for (p, c) in chain.iter_mut() {
            *c = Some(ChainRow { x: x0.row_f(idx(&x0, *p)?).to_vec(), pm: identity_pre_mix(s.hc) });
        }
    }
    let mut lines = Vec::new();
    let mut failures = Vec::new();
    lines.push(format!(
        "dsv41 OM oracle (mHC + norms + router): pack {} ({}), dump {} (mode {}, P = {}), {} rows x {} layers, chain {}",
        pp.census.dir.display(), pp.census.cfg.quant.method, d.dir.display(), d.mode(), d.p, positions.len(), layers.len(),
        if full_chain { "on" } else { "off (needs layers 0..n)" }
    ));
    let mut stats: Vec<LayerStats> = Vec::new();
    let mut prev_ffn_pre: Option<RefTensor> = None;
    // the stream at each row's embedding: the hc expansion means the four copies of x_in(0) are equal
    let mut expand_bad = 0u64;
    if layers[0] == 0 {
        let x0 = d.layer(0, "x_in")?;
        for r in 0..x0.n_rows() {
            let row = x0.row_f(r);
            for j in 1..s.hc {
                expand_bad += row[j * s.dim..(j + 1) * s.dim].iter().zip(&row[..s.dim]).filter(|(a, b)| a.to_bits() != b.to_bits()).count() as u64;
            }
        }
    }
    let mut x_last: Vec<(i64, Vec<f32>, Vec<f32>)> = Vec::new(); // (pos, step block output, dumped ffn_pre) of the last trunk layer
    for &l in &layers {
        let b = pp.block(l)?;
        let g = pp.gate(l)?;
        let ld = load_layer(&d, l, last_dumped)?;
        let pf = if l > 0 && layers.contains(&(l - 1)) { prev_ffn_pre.as_ref() } else { None };
        let parts = par_rows(&mut chain, o.threads, || LayerStats::new(l), |(pos, c), st| check_row(&s, &b, &g, &ld, pf, *pos, c.as_mut(), st))?;
        let mut st = LayerStats::new(l);
        for p in &parts {
            st.merge(p);
        }
        if l + 1 == trunk {
            for (pos, _) in &chain {
                let i = idx(&ld.x_mid, *pos)?;
                let iff = idx(&ld.ffn_pre, *pos)?;
                let xo = hc_post(&s, ld.ffn_out.row_f(idx(&ld.ffn_out, *pos)?), ld.x_mid.row_f(i), ld.ffn_post.row_f(iff), ld.ffn_comb.row_f(iff));
                x_last.push((*pos, xo, ld.ffn_pre.row_f(iff).to_vec()));
            }
        }
        prev_ffn_pre = Some(ld.ffn_pre);
        stats.push(st);
    }
    // ---- head side: h_final (hc_pre), the chain's h_final, the DSpark tap ----
    let mut h_final = Cmp::default();
    let mut chain_h_final = Cmp::default();
    if let Some(hf) = d.global_opt("h_final")? {
        for (pos, xo, fpre) in &x_last {
            if let Some(i) = hf.index_of(*pos) {
                h_final.bf16(&hc_pre(&s, xo, fpre), hf.row_f(i));
            }
        }
        for (pos, c) in &chain {
            if let (Some(c), Some(i)) = (c, hf.index_of(*pos)) {
                if layers.last() == Some(&(trunk - 1)) {
                    chain_h_final.bf16(&hc_pre(&s, &c.x, &c.pm), hf.row_f(i));
                }
            }
        }
    }
    let mut tap = Cmp::default();
    let targets = pp.census.cfg.dspark_targets.clone();
    if let Some(mh) = d.global_opt("dspark_main_hidden")? {
        if targets.iter().all(|&t| d.has(&RefbDump::key(t, "x_in"))) {
            let xs: Vec<RefTensor> = targets.iter().map(|&t| d.layer(t, "x_in")).collect::<Result<_>>()?;
            for pos in mh.positions() {
                let Some(i) = mh.index_of(pos) else { continue };
                let mut ours = Vec::with_capacity(s.dim * targets.len());
                let mut ok = true;
                for x in &xs {
                    match x.index_of(pos) {
                        Some(r) => ours.extend(stream_mean(&s, x.row_f(r))),
                        None => ok = false,
                    }
                }
                if ok {
                    tap.bf16(&ours, mh.row_f(i));
                }
            }
        }
    }
    // ---- TP K split of the mixes ----
    let mut split_worst: Vec<(usize, f64, f64)> = Vec::new();
    {
        let l = layers[0];
        let b = pp.block(l)?;
        let x = d.layer(l, "x_in")?;
        for n in [2usize, 4, 8] {
            let (mut worst_mix, mut worst_coef) = (0f64, 0f64);
            for r in 0..x.n_rows() {
                let xr = x.row_f(r);
                let full = hc_mixes(&s, &b.attn_mix, xr);
                let parts: Vec<_> = k_ranges(&s, n).into_iter().map(|(k0, k1)| hc_mix_partial(&s, &b.attn_mix, xr, k0, k1)).collect();
                let split = hc_mixes_from_partials(&s, &parts);
                for (a, c) in split.iter().zip(&full) {
                    worst_mix = worst_mix.max(((a - c).abs() / c.abs().max(1e-3)) as f64);
                }
                let cf = coef_vec(&hc_split_sinkhorn(&s, &full, &b.attn_mix.scale, &b.attn_mix.base));
                let cs = coef_vec(&hc_split_sinkhorn(&s, &split, &b.attn_mix.scale, &b.attn_mix.base));
                worst_coef = worst_coef.max(rel_l2(&cs, &cf));
            }
            split_worst.push((n, worst_mix, worst_coef));
        }
    }
    // ---- final norm + head (plain heads only) ----
    let head = head_check(&pp, &d, o)?;
    // ---- report ----
    let mut tot = LayerStats::new(usize::MAX);
    for st in &stats {
        tot.merge(st);
    }
    lines.push(String::new());
    lines.push("| layer | rows | attn coeffs (step) | attn_in (step) | x_mid (step) | ffn coeffs (step) | ffn_in (step) | block out (step) |".into());
    lines.push("|---:|---:|---|---|---|---|---|---|".into());
    for st in &stats {
        lines.push(format!("| {} | {} | {} | {} | {} | {} | {} | {} |", st.layer, st.rows, coef_cell(&st.attn_coef), st.attn_in.cell(), st.x_mid.cell(), coef_cell(&st.ffn_coef), st.ffn_in.cell(), st.x_out.cell()));
    }
    lines.push(String::new());
    lines.push("| layer | own x_mid | own ffn_in | own block out | chain x_in | chain coeffs | chain attn_in | chain x_mid | chain ffn_in |".into());
    lines.push("|---:|---|---|---|---|---|---|---|---|".into());
    for st in &stats {
        lines.push(format!("| {} | {} | {} | {} | {} | {} | {} | {} | {} |", st.layer, st.own_x_mid.cell(), st.own_ffn_in.cell(), st.own_x_out.cell(), st.chain_x_in.cell(), coef_cell(&st.chain_coef), st.chain_attn_in.cell(), st.chain_x_mid.cell(), st.chain_ffn_in.cell()));
    }
    lines.push(String::new());
    lines.push(format!("| layer | router (step, on the dumped ffn_in), gate {} | router (chain) |", pp.gate_dtype(layers[0]).unwrap_or_default()));
    lines.push("|---:|---|---|".into());
    for st in &stats {
        lines.push(format!("| {} | {} | {} |", st.layer, st.router.cell(), st.chain_router.cell()));
    }
    lines.push(String::new());
    lines.push(format!("totals over {} layer-rows:", tot.rows));
    for (name, c) in [
        ("wiring (pre_mix_in vs previous ffn_pre / one-hot)", &tot.wiring),
        ("step attn coeffs", &tot.attn_coef),
        ("step ffn coeffs", &tot.ffn_coef),
        ("step attn_in", &tot.attn_in),
        ("step x_mid", &tot.x_mid),
        ("step ffn_in", &tot.ffn_in),
        ("step block out", &tot.x_out),
        ("own x_mid", &tot.own_x_mid),
        ("own ffn_in", &tot.own_ffn_in),
        ("own block out", &tot.own_x_out),
        ("chain x_in", &tot.chain_x_in),
        ("chain coeffs", &tot.chain_coef),
        ("chain attn_in", &tot.chain_attn_in),
        ("chain x_mid", &tot.chain_x_mid),
        ("chain ffn_in", &tot.chain_ffn_in),
        ("step h_final (hc_pre of the last block)", &h_final),
        ("chain h_final", &chain_h_final),
        ("DSpark tap (stream mean of layers 37-39 inputs)", &tap),
        ("CONTROL ffn_in with ffn_pre (wrong wiring, must miss)", &tot.ctl_ffn_wrong),
        ("CONTROL attn_in with this block's attn_pre (wrong wiring, must miss)", &tot.ctl_attn_wrong),
    ] {
        let info = if name.starts_with("chain") { " [informational]" } else { "" };
        lines.push(format!("  {name}{info}: {}", if name.contains("coeffs") { coef_cell(c) } else { c.cell() }));
    }
    lines.push(format!("  embedding expansion: the {} copies of x_in(0) differ in {expand_bad} elements", s.hc));
    lines.push(format!("  router (step): {}", tot.router.cell()));
    lines.push(format!("  router (chain) [informational]: {}", tot.chain_router.cell()));
    for (n, m, c) in &split_worst {
        lines.push(format!("  TP K split into {n}: max relative mix difference {m:.2e}, coefficient rel-L2 {c:.2e} (layer {}, every row)", layers[0]));
    }
    lines.extend(head.lines.iter().cloned());
    // ---- verdict ----
    let mut fail = |c: bool, what: String| {
        if !c {
            failures.push(what);
        }
    };
    let streams_ok = |c: &Cmp| c.mismatch_frac() <= 1e-3 && c.rel_l2() <= 4e-4;
    fail(tot.wiring.mismatch == 0 && tot.wiring.n > 0, format!("wiring: {}", tot.wiring.cell()));
    fail(tot.x_mid.mismatch == 0 && tot.x_out.mismatch == 0, format!("hc_post exact: x_mid {}, block out {}", tot.x_mid.cell(), tot.x_out.cell()));
    fail(h_final.mismatch == 0 && tap.mismatch == 0, format!("hc_pre exact: h_final {}, DSpark tap {}", h_final.cell(), tap.cell()));
    for st in &stats {
        for (n, c) in [("attn_in", &st.attn_in), ("ffn_in", &st.ffn_in)] {
            fail(c.mismatch_frac() <= 1e-4 && c.max_ulps <= 1, format!("norms: layer {} {n} {}", st.layer, c.cell()));
        }
        for (n, c) in [("attn", &st.attn_coef), ("ffn", &st.ffn_coef)] {
            fail(c.rel_l2() <= 1e-6 && c.max_abs <= 1e-5, format!("coefficients: layer {} {n} {}", st.layer, coef_cell(c)));
        }
    }
    for (n, c) in [
        ("own x_mid", &tot.own_x_mid),
        ("own ffn_in", &tot.own_ffn_in),
        ("own block out", &tot.own_x_out),
    ] {
        fail(streams_ok(c), format!("own streams: {n} {}", c.cell()));
    }
    for st in &stats {
        let r = &st.router;
        fail(r.flips == r.flips_near, format!("router: layer {} has {} flip(s) away from a near-tie: {}", st.layer, r.flips - r.flips_near, r.cell()));
        fail(r.w.rel_l2() <= 1e-6, format!("router: layer {} weights {}", st.layer, r.w.cell()));
    }
    let ctl_n = tot.ctl_ffn_wrong.n + tot.ctl_attn_wrong.n;
    fail(ctl_n == 0 || (tot.ctl_ffn_wrong.mismatch_frac() > 0.5 && tot.ctl_attn_wrong.mismatch_frac() > 0.5),
         format!("delayed pre-mix controls: the wrong wirings do not miss (ffn {}, attn {})", tot.ctl_ffn_wrong.cell(), tot.ctl_attn_wrong.cell()));
    fail(tot.router.rows > 0, "router: no rows checked".into());
    for (n, m, _) in &split_worst {
        fail(*m <= 1e-5, format!("tp split {n}: {m:.2e}"));
    }
    for f in &head.failures {
        fail(false, f.clone());
    }
    fail(expand_bad == 0, format!("embedding expansion: {expand_bad} elements differ between copies"));
    lines.push(String::new());
    lines.push(format!("bounds: {} checked, {} failed; {:.1} s", BOUNDS.len(), failures.len(), t0.elapsed().as_secs_f64()));
    for f in &failures {
        lines.push(format!("  FAILED {f}"));
    }
    Ok(Report { lines, failures, layers: stats })
}

fn coef_cell(c: &Cmp) -> String {
    if c.n == 0 {
        return "-".into();
    }
    format!("rel {:.1e}, max {:.1e} ({}/{} bitwise)", c.rel_l2(), c.max_abs, c.n - c.mismatch, c.n)
}

struct HeadResult {
    lines: Vec<String>,
    failures: Vec<String>,
}

/// Final norm + head on the dumped `h_final` vs the dumped fp32 logits, for the last `o.head_rows`
/// dumped rows (packs whose head is plain; an EXL3 head belongs to the kernel legs).
fn head_check(pp: &PackParams, d: &RefbDump, o: &CheckOpts) -> Result<HeadResult> {
    let mut r = HeadResult { lines: Vec::new(), failures: Vec::new() };
    if o.head_rows == 0 {
        return Ok(r);
    }
    let Some(he) = pp.plain_head() else {
        r.lines.push("  head: not plain in this pack (skipped; the final norm is then checked only through h_final's consumers)".into());
        return Ok(r);
    };
    let (Some(hf), Some(lg)) = (d.global_opt("h_final")?, d.global_opt("logits")?) else {
        r.lines.push("  head: the dump has no h_final / logits (skipped)".into());
        return Ok(r);
    };
    let fnorm = pp.final_norm()?;
    let s = pp.shape;
    let mut pos: Vec<i64> = lg.positions().into_iter().filter(|&p| hf.index_of(p).is_some()).collect();
    pos.sort();
    let pos: Vec<i64> = pos.iter().rev().take(o.head_rows).rev().copied().collect();
    if pos.is_empty() {
        r.lines.push("  head: no row has both h_final and logits (skipped)".into());
        return Ok(r);
    }
    let xs: Vec<Vec<f32>> = pos.iter().map(|&p| norm::rms_norm(hf.row_f(hf.index_of(p).unwrap()), &fnorm, s.norm_eps)).collect();
    let vocab = he.shape[0] as usize;
    let threads = o.threads.max(1);
    let per = (vocab + threads - 1) / threads;
    let chunks: Vec<Result<Vec<Vec<f32>>>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let xs = &xs;
                sc.spawn(move || -> Result<Vec<Vec<f32>>> {
                    let (v0, v1) = (t * per, ((t + 1) * per).min(vocab));
                    let mut out = vec![Vec::with_capacity(v1.saturating_sub(v0)); xs.len()];
                    let mut a = v0;
                    while a < v1 {
                        let bnd = (a + 2048).min(v1);
                        let w = pp.read_rows(he, a, bnd)?;
                        for row in w.chunks(s.dim) {
                            for (k, x) in xs.iter().enumerate() {
                                out[k].push(super::num::dot_f64(row, x) as f32);
                            }
                        }
                        a = bnd;
                    }
                    Ok(out)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("head worker panicked")))).collect()
    });
    let mut logits = vec![Vec::with_capacity(vocab); xs.len()];
    for c in chunks {
        for (k, part) in c?.into_iter().enumerate() {
            logits[k].extend(part);
        }
    }
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0;
    for (k, &p) in pos.iter().enumerate() {
        let refr = lg.row_f(lg.index_of(p).unwrap());
        let rl = rel_l2(&logits[k], refr);
        let (a, b) = (argmax(&logits[k]), argmax(refr));
        let maxd = logits[k].iter().zip(refr).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        r.lines.push(format!("  head row {p}: logits rel-L2 {rl:.2e}, max |diff| {maxd:.2e}, argmax {a} vs {b}"));
        if !(rl <= 1e-5 && a == b) {
            r.failures.push(format!("head row {p}: rel-L2 {rl:.2e}, argmax {a} vs {b}"));
        }
    }
    Ok(r)
}

/// Format the report as one block of text (what the CLI prints).
pub fn render(r: &Report) -> String {
    let mut s = String::new();
    for l in &r.lines {
        let _ = writeln!(s, "{l}");
    }
    s
}
