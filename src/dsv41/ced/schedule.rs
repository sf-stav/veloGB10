//! CE: which (layer, row) pairs a prefill computes. A pure function of the config, the prompt length and the
//! mode; chunk boundaries never enter it, so a chunked prefill computes exactly the rows a single-shot one
//! does (REFb §3.5 / §4, X-DI §1.4: the rectangle is anchored at the prompt end).
//!
//! Per layer L the prefill computes:
//! * **query rows** `[q0, p)`: q path, (indexer), sparse attention, wo, mHC, MoE, hc_post (the layer's output);
//! * **window rows** `[w0, p)`: window K/V (wkv, kv_norm, RoPE, FP8) — the keys the query rows read and the
//!   ring the decode reads; `w0 <= q0`;
//! * **KV-source rows**: a KV source's compressor + index keys on every row it has an input for.
//!
//! The layer needs its attention-norm input (`attn_in`, hence its input stream) on `[a0, p)` with
//! `a0 = 0` for a layer whose KV-source pass covers every row, else `w0`. The schedule is consistent when
//! every layer's `a0` is at or above the previous layer's `q0` (it has the input it reads).
//!
//! Modes (`Mode`):
//! * `Exact`: every layer on every row (the oracle; also the score mode, which needs every row's logits).
//! * `Ced` (the default, REFb's `Schedule.ced_min`): encoder layers (before the first ratio-1 layer) on every
//!   row; the first decoder layer's compressor + index keys on every row; decoder layer L only on its
//!   dependency cone: `q0 = p - 1` at the last trunk layer (the first token's logits), and below it each layer
//!   computes the rows the next layer's window reads (`q0_L = w0_{L+1} = q0_{L+1} - (window - 1)`). EXACT: the
//!   decode state and first-token logits are bit-identical to `Exact` with batch-invariant kernels.
//! * `Replay { keep }`: DeepSeek's "decoder SWA bounded replay": decoder layers run the last `keep` rows with
//!   their window keys truncated to that segment (`win_lo`). Exact for `keep >= 2,541` (= 127 x 20 + 1),
//!   approximate below (REFb: keep 140 gives first-token KL up to 4e-3). Opt-in only; `keep < 134` is raised.
//!
//! A **resume** (prefix state at `k`) clips every range at `k`: rows below it are in the state. A state is
//! `Full` when every layer's ring holds its last window rows at `k` (exact prefill, or the prompt end of a CED
//! prefill, or any decode position): any resume `k <= p - 1` is then exact. A state taken mid-prompt in a CED
//! prefill is `Encoder` (its decoder rings were never computed there): a resume from it is exact only when every
//! decoder layer's window rows lie at or after `k`, i.e. `k <= p - cone`, with `cone = p - min(w0)` over the
//! decoder layers (2,541 for this model in CED).

use crate::dsv41::config::Dsv41Config;
use anyhow::{bail, ensure, Result};

/// DeepSeek's minimum replay: window 128 + 6 backtracking rows (X-DI §1.3).
pub const MIN_KEEP: usize = 134;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Exact,
    Ced,
    Replay { keep: usize },
}

impl Mode {
    /// From the options: `--dsv41-prefill exact|ced` and `--dsv41-ced-keep N` (0 = the exact minimal cone).
    /// A keep below 134 is raised to 134 (returned as the second value so the caller prints one line and counts it).
    pub fn from_opts(prefill: &str, keep: usize) -> Result<(Mode, Option<usize>)> {
        Ok(match prefill {
            "exact" => (Mode::Exact, None),
            "ced" if keep == 0 => (Mode::Ced, None),
            "ced" if keep < MIN_KEEP => (Mode::Replay { keep: MIN_KEEP }, Some(keep)),
            "ced" => (Mode::Replay { keep }, None),
            m => bail!("--dsv41-prefill {m}: ced or exact"),
        })
    }
    pub fn name(&self) -> String {
        match self { Mode::Exact => "exact".into(), Mode::Ced => "ced".into(), Mode::Replay { keep } => format!("replay{keep}") }
    }
}

/// Times a requested `--dsv41-ced-keep` below 134 was raised to 134 (`/metrics`: velogb10_dsv41_ced_keep_raised_total).
pub static KEEP_RAISED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The served prefill mode from `--dsv41-prefill` and `--dsv41-ced-keep`, with ONE boot line when the keep is raised
/// (AGENTS 1b-bis 8: applied and shown, never refused) and one when the opt-in approximate replay is on. The forward
/// that serves calls this once at boot (leg CE; the integration leg wires it into the dsv41 serve boot).
pub fn mode_from_flags() -> Result<Mode> {
    use crate::opts;
    let p = opts::lookup("dsv41-prefill").and_then(opts::get).unwrap_or_else(|| "ced".into());
    let k: usize = opts::lookup("dsv41-ced-keep").and_then(opts::get).map(|s| s.parse().unwrap_or(0)).unwrap_or(0);
    let (m, raised) = Mode::from_opts(&p, k)?;
    if let Some(asked) = raised {
        KEEP_RAISED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::boot_report::notice(&format!("--dsv41-ced-keep {asked} is below the decoder's minimum replay (window 128 + 6 backtracking rows): replaying {MIN_KEEP} rows"));
    }
    if let Mode::Replay { keep } = m {
        let exactness = if keep >= 2541 { "exact (keep >= 2,541)" } else { "APPROXIMATE (keep < 2,541: first-token KL up to ~4e-3 nats measured at keep 140)" };
        crate::boot_report::notice(&format!("dsv41 prefill: decoder bounded replay of the last {keep} rows (opt-in), {exactness}"));
    }
    Ok(m)
}

/// What a prefix state at `k` holds (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateKind {
    /// every layer's ring is valid at `k`: any resume is exact
    Full,
    /// encoder layers' rings + every source's caches; decoder rings missing
    Encoder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerRows {
    pub layer: usize,
    /// query rows `[q0, p)`
    pub q0: usize,
    /// window rows `[w0, p)`, `w0 <= q0`
    pub w0: usize,
    /// window keys below this position are masked (0 = none; a truncated replay sets it)
    pub win_lo: usize,
    /// this layer is a KV source and its compressor + index keys run on every row
    pub kv_all: bool,
    /// the layer is a decoder layer (at or after the first ratio-1 layer)
    pub decoder: bool,
}

impl LayerRows {
    /// First row whose attention-norm input the layer needs.
    pub fn a0(&self) -> usize { if self.kv_all { 0 } else { self.w0 } }
}

#[derive(Clone, Debug)]
pub struct Schedule {
    pub p: usize,
    pub mode: Mode,
    /// rows below this are in the state the prefill resumes from (0 = cold)
    pub from: usize,
    /// the kind of that state (Full when cold)
    pub from_kind: StateKind,
    /// one entry per trunk layer (0..cfg.layers)
    pub layers: Vec<LayerRows>,
    pub first_decoder: usize,
    pub window: usize,
    /// the smallest decoder window start of the plain schedule of `p` (before any `full_at` widening or resume clip):
    /// an encoder-only state resumes this prompt exactly only at or below it (`encoder_resume_cap`)
    enc_cap: usize,
    /// leg CE2: the lowest mid-prompt position whose state must be FULL (`new_with`), None = the plain schedule
    pub full_at: Option<usize>,
}

impl Schedule {
    /// A cold prefill of `p` rows.
    pub fn new(cfg: &Dsv41Config, p: usize, mode: Mode) -> Result<Schedule> { Schedule::new_with(cfg, p, mode, None) }

    /// A cold prefill of `p` rows whose mid-prompt states at `full_at` and above are FULL (leg CE2; `Mode::Ced` only).
    ///
    /// Under the plain CED schedule every mid-prompt state is encoder-only (the decoder layers run only on the cone of
    /// the prompt end), so the prefix pool's grid states near the end of a prompt (`prefix::snapshot_plan`: the last
    /// 128-token blocks at or below `p - 134`) can resume only prompts at least the whole decoder cone (2,541 rows)
    /// longer: an identical repeat or an edited last message gets nothing from them. With `full_at = k` each decoder
    /// layer also computes the cone of a prompt ending at `k` (the union of the two cones: `w0_L = min(w0_L(p),
    /// w0_L(k))`, i.e. `p - k` more rows per decoder layer), so its ring holds its last window rows at every position
    /// from `k` on, and a state taken there is `Full`. The extra rows are exact rows of the same layers (row-local
    /// pieces, CE's bit-identity of any cone), so the decode state and first-token logits at `p` do not change.
    /// `Exact` is full everywhere already; `Replay` is left as it is (its mid-prompt states stay encoder-only and are
    /// labelled so).
    pub fn new_with(cfg: &Dsv41Config, p: usize, mode: Mode, full_at: Option<usize>) -> Result<Schedule> {
        ensure!(p > 0, "empty prompt");
        let n = cfg.layers;
        let d = cfg.first_decoder_layer().unwrap_or(n);
        let w = cfg.window;
        ensure!(w >= 1, "window 0");
        let mut layers = Vec::with_capacity(n);
        for l in 0..n {
            let kv_all = l < d && cfg.ratio(l) > 0 && cfg.kv_sources.contains(&l) || (l == d && cfg.kv_sources.contains(&l));
            layers.push(LayerRows { layer: l, q0: 0, w0: 0, win_lo: 0, kv_all, decoder: l >= d });
        }
        match mode {
            Mode::Exact => {}
            Mode::Ced => {
                // top-down: the last trunk layer needs only the last row; each layer below computes the rows
                // the window of the layer above reads
                let mut need = p - 1; // the rows [need, p) the layer above reads as its input
                for l in (d..n).rev() {
                    let q0 = need;
                    let w0 = q0.saturating_sub(w - 1);
                    layers[l].q0 = q0;
                    layers[l].w0 = w0;
                    need = w0;
                }
                // DSpark taps (inputs of the target layers at the last `window` rows) and the decode rings
                // (each decoder layer's last window - 1 rows) are inside the cone; checked, not assumed
                for &t in &cfg.dspark_targets {
                    if t > d && t < n { ensure!(layers[t - 1].q0 <= p.saturating_sub(w), "DSpark tap of layer {t} outside the cone"); }
                }
            }
            Mode::Replay { keep } => {
                ensure!(keep >= MIN_KEEP, "replay keep {keep} < {MIN_KEEP}");
                let s = p.saturating_sub(keep);
                for l in d..n { layers[l].q0 = s; layers[l].w0 = s; layers[l].win_lo = s; }
            }
        }
        let enc_cap = layers.iter().filter(|r| r.decoder).map(|r| r.w0).min().unwrap_or(p);
        let full_at = match (mode, full_at) {
            (Mode::Ced, Some(k)) if k > 0 && k < p => {
                // the cone of a prompt ending at k, top-down as above; each decoder layer takes the union with p's cone
                let mut need = k - 1;
                for l in (d..n).rev() {
                    let q0 = need;
                    let w0 = q0.saturating_sub(w - 1);
                    layers[l].q0 = layers[l].q0.min(q0);
                    layers[l].w0 = layers[l].w0.min(w0);
                    need = w0;
                }
                Some(k)
            }
            _ => None,
        };
        let s = Schedule { p, mode, from: 0, from_kind: StateKind::Full, layers, first_decoder: d, window: w, enc_cap, full_at };
        s.check()?;
        Ok(s)
    }

    /// The same prefill resumed from a state at `k` of the given kind.
    pub fn resumed(cfg: &Dsv41Config, p: usize, mode: Mode, k: usize, kind: StateKind) -> Result<Schedule> {
        Schedule::resumed_with(cfg, p, mode, k, kind, None)
    }

    /// `resumed` with `new_with`'s `full_at` (the refusal of an encoder-only state depends only on `p`'s own cone).
    pub fn resumed_with(cfg: &Dsv41Config, p: usize, mode: Mode, k: usize, kind: StateKind, full_at: Option<usize>) -> Result<Schedule> {
        let mut s = Schedule::new_with(cfg, p, mode, full_at)?;
        ensure!(k < p, "resume at {k} leaves no row of a {p}-row prompt to compute");
        if kind == StateKind::Encoder {
            ensure!(k <= s.encoder_resume_cap(), "an encoder-only state at {k} cannot resume a {p}-row prompt exactly: the decoder cone starts at {}",
                    s.encoder_resume_cap());
        }
        for r in &mut s.layers {
            r.q0 = r.q0.max(k);
            r.w0 = r.w0.max(k);
        }
        s.from = k;
        s.from_kind = kind;
        s.check()?;
        Ok(s)
    }

    /// Smallest window start over the decoder layers: an encoder-only state must be at or below it.
    pub fn encoder_resume_cap(&self) -> usize { self.enc_cap }

    /// Rows a resumed prefill must recompute at least, from a state of this kind (the prefix pool's cap).
    pub fn min_suffix(&self, kind: StateKind) -> usize {
        match kind {
            StateKind::Full => MIN_KEEP.min(self.p),
            StateKind::Encoder => (self.p - self.encoder_resume_cap()).max(MIN_KEEP),
        }
    }

    /// The kind of the state a lane holds after this prefill has processed rows `[0, k)` (chunked prefill,
    /// snapshot at a chunk boundary `k`): Full iff every layer computed its last `window - 1` window rows below `k`.
    pub fn state_kind_at(&self, k: usize) -> StateKind {
        let back = self.window - 1;
        // rows below `from` come from the restored state: valid for every layer only when that state was Full
        let inherited = |r: &LayerRows| self.from > 0 && self.from_kind == StateKind::Full && r.w0 <= self.from;
        let full = self.layers.iter().all(|r| r.w0 <= k.saturating_sub(back) || inherited(r));
        if full && k <= self.p { StateKind::Full } else { StateKind::Encoder }
    }

    fn check(&self) -> Result<()> {
        for (i, r) in self.layers.iter().enumerate() {
            ensure!(r.w0 <= r.q0 && r.q0 < self.p.max(1) && r.layer == i, "layer {i}: rows {r:?} of p {}", self.p);
            if i > 0 {
                let below = &self.layers[i - 1];
                ensure!(r.a0().max(self.from) >= below.q0, "layer {i} reads rows from {} but layer {} computes from {}", r.a0(), i - 1, below.q0);
            }
        }
        Ok(())
    }

    /// Layer-rows of the full layers (query rows), as REFb counts them.
    pub fn layer_rows(&self) -> usize { self.layers.iter().map(|r| self.p - r.q0).sum() }

    /// The chunk view: rows of `[c0, c1)` the layer computes (query, window, attention-norm input, KV source).
    pub fn chunk(&self, l: usize, c0: usize, c1: usize) -> ChunkRows {
        let r = &self.layers[l];
        let clip = |a: usize| (a.max(c0).max(self.from), c1);
        ChunkRows { q: clip(r.q0), w: clip(r.w0), a: clip(r.a0()), kv: if r.kv_all { Some(clip(0)) } else { None }, win_lo: r.win_lo }
    }
}

/// The `Snapshot::min_suffix` label of a prefix state of `kind` taken by a prefill in `mode` (leg CE2): the rows a prompt
/// must have past the state's position for a resume to be exact, whatever that prompt's length. 0 for a `Full` state (the
/// pool's own floor of 134 applies); for an `Encoder` state the decoder cone of the mode (CED: 127 x 20 + 1 = 2,541; a
/// bounded replay: its keep; `Exact` makes no encoder-only state, so its label hands nothing out). A prompt `P'` then
/// passes the pool's check `pos <= P' - label` exactly when `Schedule::resumed(P', pos, Encoder)` accepts it: the cone
/// is the same for every `P'` past it, and below it both sides refuse (the pool saturates to 0, the cap is 0).
pub fn snapshot_label(cfg: &Dsv41Config, mode: Mode, kind: StateKind) -> usize {
    match kind {
        StateKind::Full => 0,
        StateKind::Encoder => {
            // any prompt longer than every cone gives the constant (the cone is anchored at the prompt end)
            let big = 1usize << 22;
            Schedule::new(cfg, big, mode).map(|s| s.min_suffix(StateKind::Encoder)).unwrap_or(big)
        }
    }
}

/// Can a prefill of a `total`-row prompt in `mode` resume from a state at `k` of `kind`? The exact condition the lane
/// driver applies (`Schedule::resumed`), for the serve shell to ask BEFORE it commits the lane to the state (leg CE2).
pub fn resume_check(cfg: &Dsv41Config, mode: Mode, total: usize, k: usize, kind: StateKind) -> Result<()> {
    Schedule::resumed(cfg, total, mode, k, kind).map(|_| ())
}

/// One layer's rows inside one chunk: half-open `(start, end)`, empty when start >= end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkRows { pub q: (usize, usize), pub w: (usize, usize), pub a: (usize, usize), pub kv: Option<(usize, usize)>, pub win_lo: usize }
pub fn n_of(r: (usize, usize)) -> usize { r.1.saturating_sub(r.0) }

/// Split `[from, p)` into chunks of `chunk` rows (the last one shorter).
pub fn chunks(from: usize, p: usize, chunk: usize) -> Vec<(usize, usize)> {
    let c = chunk.max(1);
    let mut v = Vec::new();
    let mut s = from;
    while s < p { v.push((s, (s + c).min(p))); s += c; }
    v
}

/// Leg P4: `chunks`, except that a last chunk of `1..=merge` rows (with a chunk before it) rides that chunk (up to `chunk + merge` rows):
/// a 4,185-row prompt is two chunks, not three, and no 89-row chunk streams every layer's weights. `merge == 0` = `chunks`.
pub fn chunks_elastic(from: usize, p: usize, chunk: usize, merge: usize) -> Vec<(usize, usize)> {
    let mut v = chunks(from, p, chunk);
    if merge > 0 && v.len() >= 2 {
        let (l0, l1) = v[v.len() - 1];
        if l1 - l0 <= merge { v.pop(); let n = v.len(); v[n - 1].1 = l1; }
    }
    v
}

// --------------------------------------------------------------------------------------------- the cost model

/// Multiply-accumulates of a prefill under a schedule: REFb's cost model (`ref_forward.py prefill_macs`, from
/// the shapes in model.py), ported term for term so the two tables agree. Full rows: q / kv / o projections,
/// attention over window + top-k compressed, indexer (scores grow with the position), MoE (top-k + shared),
/// mHC, Engram. Rows that only provide window K/V (or, at KV sources, the compressor + index keys) are charged
/// those projections only.
pub fn prefill_macs(cfg: &Dsv41Config, s: &Schedule) -> f64 {
    let (d, h, hd, qr) = (cfg.hidden as f64, cfg.heads as f64, cfg.head_dim as f64, cfg.q_lora_rank as f64);
    let expert = 3.0 * d * cfg.moe_inter as f64;
    let dense = d * qr + qr * h * hd + d * hd + h * hd * cfg.o_lora_rank as f64 + cfg.o_groups as f64 * cfg.o_lora_rank as f64 * d;
    let moe = d * cfg.n_routed as f64 + (cfg.top_k as f64 + 1.0) * expert;
    let hc = 2.0 * (2.0 + cfg.hc as f64) * (cfg.hc * cfg.hc) as f64 * d;
    let p = s.p;
    let (ih, ihd) = (cfg.index_heads as f64, cfg.index_head_dim as f64);
    let mut total = 0.0;
    for r in &s.layers {
        let l = r.layer;
        let ratio = cfg.ratio(l) as usize;
        let q0 = r.q0;
        let n = (p - q0) as f64;
        let mut keys = 0.0f64;
        let mut idx = 0.0f64;
        for row in q0..p {
            let mut k = (row + 1).min(cfg.window) as f64;
            if ratio > 0 { k += cfg.index_topk.min((row + 1) / ratio) as f64; idx += ((row + 1) / ratio) as f64; }
            keys += k;
        }
        let mut c = n * (dense + moe + hc) + keys * h * hd * 2.0;
        if cfg.index_sources.contains(&l) { c += n * (qr * ih * ihd + d * ih) + ih * ihd * idx; }
        if cfg.engram_layers.contains(&l) {
            c += n * ((cfg.engram_max_ngram - 1) as f64 * cfg.engram_heads as f64 * cfg.engram_head_dim as f64 * d * (cfg.hc as f64 + 1.0));
        }
        if cfg.kv_sources.contains(&l) {
            c += p as f64 * d * hd * if ratio > 1 { 2.0 } else { 1.0 } + (p / ratio.max(1)) as f64 * hd * ihd;
        }
        // rows that only provide window K/V (REFb's rule: at most the last `window` rows below q0 that a query reads)
        // (REFb charges a KV source the window projection of every row below q0)
        let kv_only = if cfg.kv_sources.contains(&l) { q0 as i64 } else { q0 as i64 - (q0 as i64 - (cfg.window as i64 - 1)).min(p as i64 - cfg.window as i64).max(0) };
        c += kv_only.max(0) as f64 * d * hd;
        total += c;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn cfg() -> Option<Dsv41Config> {
        let d = Path::new("/models/DeepSeek-V4.1-Flash-1017");
        if !d.join("config.json").exists() { return None; }
        Dsv41Config::load(d).ok()
    }

    #[test]
    fn dsv41_ce_ced_cone_is_refbs_minimal_set() {
        let Some(c) = cfg() else { return };
        for p in [30usize, 4197, 33467, 1_048_576] {
            let s = Schedule::new(&c, p, Mode::Ced).unwrap();
            assert_eq!(s.first_decoder, 20);
            for l in 0..20 { assert_eq!((s.layers[l].q0, s.layers[l].w0), (0, 0), "encoder layer {l} on every row"); }
            assert!(s.layers[20].kv_all && !s.layers[21].kv_all && s.layers[2].kv_all && !s.layers[0].kv_all);
            // REFb's ced_min: q0_39 = P-1, q0_38 = P-128, q0_L = P - 128 - 127 (38 - L)
            assert_eq!(s.layers[39].q0, p - 1);
            for l in 20..39 { assert_eq!(s.layers[l].q0, p.saturating_sub(128 + 127 * (38 - l)), "layer {l} p {p}"); }
            if p > 2541 {
                assert_eq!(s.layers[20].q0, p - 2414);
                assert_eq!(s.encoder_resume_cap(), p - 2541);
                // decoder layer-rows are the constant 24,150 of REFb §3.2
                let dec: usize = (20..40).map(|l| p - s.layers[l].q0).sum();
                assert_eq!(dec, 24_150);
            }
        }
        // REFb §3.3: layer-rows / exact = 0.644 at 4,197 and 0.518 at 33,467
        for (p, want) in [(4197usize, 0.644), (33467, 0.518)] {
            let r = Schedule::new(&c, p, Mode::Ced).unwrap().layer_rows() as f64 / Schedule::new(&c, p, Mode::Exact).unwrap().layer_rows() as f64;
            assert!((r - want).abs() < 6e-4, "{p}: {r}");
        }
    }

    #[test]
    fn dsv41_ce_cost_model_matches_refbs_macs_fractions() {
        let Some(c) = cfg() else { return };
        // REFb §3.3 "MACs / exact (cost model)": CED 0.649 (4,197) / 0.519 (33,467); replay 2560 0.810 / 0.540;
        // replay 140 0.523 / 0.503
        for (p, mode, want) in [(4197usize, Mode::Ced, 0.649), (33467, Mode::Ced, 0.519), (4197, Mode::Replay { keep: 2560 }, 0.810),
                                (33467, Mode::Replay { keep: 2560 }, 0.540), (4197, Mode::Replay { keep: 140 }, 0.523), (33467, Mode::Replay { keep: 140 }, 0.503)] {
            let e = prefill_macs(&c, &Schedule::new(&c, p, Mode::Exact).unwrap());
            let v = prefill_macs(&c, &Schedule::new(&c, p, mode).unwrap());
            assert!((v / e - want).abs() < 1.5e-3, "{p} {mode:?}: {} vs {want}", v / e);
        }
    }

    #[test]
    fn dsv41_ce_chunks_cover_each_row_once_and_the_view_never_depends_on_them() {
        let Some(c) = cfg() else { return };
        let p = 4197;
        let s = Schedule::new(&c, p, Mode::Ced).unwrap();
        for ch in [1usize, 1001, 2048, 4095, 4197, 9999] {
            for l in 0..c.layers {
                let mut q = 0;
                let mut w = 0;
                for (c0, c1) in chunks(0, p, ch) { let v = s.chunk(l, c0, c1); q += n_of(v.q); w += n_of(v.w); }
                assert_eq!(q, p - s.layers[l].q0);
                assert_eq!(w, p - s.layers[l].w0);
            }
        }
    }

    #[test]
    fn dsv41_ce_resume_kinds_and_caps() {
        let Some(c) = cfg() else { return };
        let p = 4197;
        let s = Schedule::new(&c, p, Mode::Ced).unwrap();
        // mid-prompt states of a CED prefill are encoder-only; the prompt end is a full state
        assert_eq!(s.state_kind_at(1536), StateKind::Encoder);
        assert_eq!(s.state_kind_at(p), StateKind::Full);
        let e = Schedule::new(&c, p, Mode::Exact).unwrap();
        assert_eq!(e.state_kind_at(1536), StateKind::Full);
        // an encoder state resumes only below the cone; a full state anywhere
        assert!(Schedule::resumed(&c, p, Mode::Ced, 1536, StateKind::Encoder).is_ok());
        assert!(Schedule::resumed(&c, p, Mode::Ced, 1664, StateKind::Encoder).is_err());
        let r = Schedule::resumed(&c, p, Mode::Ced, 3968, StateKind::Full).unwrap();
        assert_eq!(r.layers[39].q0, p - 1);
        assert_eq!(r.layers[20].q0, 3968);
        assert_eq!(r.layers[0].q0, 3968);
        assert_eq!(s.min_suffix(StateKind::Full), 134);
        assert_eq!(s.min_suffix(StateKind::Encoder), 2541);
        // keep below 134 is raised, and reported
        assert_eq!(Mode::from_opts("ced", 100).unwrap(), (Mode::Replay { keep: 134 }, Some(100)));
        assert_eq!(Mode::from_opts("ced", 0).unwrap(), (Mode::Ced, None));
        assert_eq!(Mode::from_opts("exact", 140).unwrap(), (Mode::Exact, None));
        let rp = Schedule::new(&c, p, Mode::Replay { keep: 140 }).unwrap();
        assert_eq!((rp.layers[25].q0, rp.layers[25].w0, rp.layers[25].win_lo), (p - 140, p - 140, p - 140));
        assert_eq!(rp.layers[5].win_lo, 0);
    }

    /// Leg CE2, FIXB's case (`87183a55`, 5,286-token agent prompt): the plain schedule's mid-prompt state at 5,120 is encoder-only
    /// and refused; the old label (134) let the pool offer it; the new label matches the refusal for every prompt length.
    #[test]
    fn dsv41_ce2_label_admits_exactly_what_the_lane_driver_resumes() {
        use crate::dsv41::serve::forward::Snapshot;
        use crate::dsv41::serve::prefix::{PrefixPool, MIN_KEEP as POOL_KEEP};
        let Some(c) = cfg() else { return };
        let p = 5286;
        let s = Schedule::new(&c, p, Mode::Ced).unwrap();
        assert_eq!(s.state_kind_at(5120), StateKind::Encoder);
        let e = resume_check(&c, Mode::Ced, p, 5120, StateKind::Encoder).unwrap_err().to_string();
        assert!(e.contains("an encoder-only state at 5120 cannot resume a 5286-row prompt exactly: the decoder cone starts at 2745"), "{e}");
        let toks: Vec<u32> = (0..20_000u32).collect();
        let pool_with = |label: usize| {
            let mut pool = PrefixPool::new(1 << 40, POOL_KEEP);
            assert!(pool.insert(&toks[..5120], Snapshot { pos: 5120, bytes: 1, data: Box::new(()), min_suffix: label, tail: false }));
            pool
        };
        // the old label: the pool hands the state to the prompt the lane driver refuses (the death)
        assert_eq!(pool_with(MIN_KEEP).lookup(&toks[..p]), Some((1, 5120)));
        // the new labels
        assert_eq!(snapshot_label(&c, Mode::Ced, StateKind::Encoder), 2541);
        assert_eq!(snapshot_label(&c, Mode::Ced, StateKind::Full), 0);
        assert_eq!(snapshot_label(&c, Mode::Replay { keep: 3000 }, StateKind::Encoder), 3000);
        assert!(snapshot_label(&c, Mode::Exact, StateKind::Encoder) > 1 << 21);
        // pool admission == the lane driver's verdict, for every prompt length around the cone and every block position
        for mode in [Mode::Ced, Mode::Replay { keep: 3000 }, Mode::Replay { keep: 140 }] {
            let label = snapshot_label(&c, mode, StateKind::Encoder);
            for pl in (130..9000).step_by(37) {
                for k in (128..pl.min(8000)).step_by(128) {
                    let mut pool = PrefixPool::new(1 << 40, POOL_KEEP);
                    pool.insert(&toks[..k], Snapshot { pos: k, bytes: 1, data: Box::new(()), min_suffix: label, tail: false });
                    let offered = pool.lookup(&toks[..pl]).is_some();
                    let ok = resume_check(&c, mode, pl, k, StateKind::Encoder).is_ok();
                    // the pool's floor (134) may hold back a state the driver could take; it never offers one the driver refuses
                    assert!(!offered || ok, "{mode:?}: pool offers an encoder state at {k} to a {pl}-token prompt the driver refuses");
                    if mode == Mode::Ced { assert_eq!(offered, ok, "{mode:?}: k {k} prompt {pl}"); }
                }
            }
        }
    }

    /// Leg CE2: with `full_at` every mid-prompt grid state of the shell's plan is FULL, for any prompt length; the prompt's own cone,
    /// the encoder layers, the resume cap and the refusal of an encoder-only state do not change; the cost is `p - full_at` rows
    /// per decoder layer at most.
    #[test]
    fn dsv41_ce2_full_at_makes_every_grid_state_full_at_a_bounded_cost() {
        use crate::dsv41::serve::prefix::{snapshot_plan, MIN_KEEP as POOL_KEEP};
        let Some(c) = cfg() else { return };
        for p in [200usize, 777, 1500, 2600, 2745, 4197, 5286, 9000, 33_467, 131_072] {
            let plan = snapshot_plan(p, 0, 4, POOL_KEEP);
            let k = crate::dsv41::tp::serve::full_at_for(0, p, &plan);
            let plain = Schedule::new(&c, p, Mode::Ced).unwrap();
            let s = Schedule::new_with(&c, p, Mode::Ced, k).unwrap();
            assert_eq!(s.encoder_resume_cap(), plain.encoder_resume_cap(), "p {p}");
            for &q in &plan { assert_eq!(s.state_kind_at(q), StateKind::Full, "p {p}: plan {plan:?} state at {q}"); }
            let mut extra = 0;
            for (a, b) in s.layers.iter().zip(&plain.layers) {
                assert!(a.q0 <= b.q0 && a.w0 <= b.w0, "p {p} layer {}: the prompt's own cone is kept", a.layer);
                if !a.decoder { assert_eq!((a.q0, a.w0), (b.q0, b.w0), "p {p}: encoder layer {} unchanged", a.layer); }
                extra += b.q0 - a.q0;
            }
            if let Some(k) = k {
                assert!(extra <= (p - k) * 20, "p {p}: {extra} extra layer-rows");
                assert!(p - k <= 4 * 128 + 134, "p {p}: full_at {k} within the grid");
                if p > 2541 + (p - k) { assert_eq!(extra, (p - k) * 20, "p {p}: exactly p - full_at rows per decoder layer"); }
                // a repeat resumes from the top grid state (full), as the pool offers it; a resumed prefill takes the same full_at
                let top = *plan.iter().filter(|&&q| q < p).max().unwrap();
                assert!(resume_check(&c, Mode::Ced, p, top, StateKind::Full).is_ok());
                assert!(Schedule::resumed_with(&c, p, Mode::Ced, plan[0].min(p - 1), StateKind::Full, Some(k)).is_ok());
            }
            // the plain schedule's mid-prompt states are encoder-only (the defect `full_at` removes)
            for &q in plan.iter().filter(|&&q| q + 1 < p) { assert_eq!(plain.state_kind_at(q), StateKind::Encoder, "p {p} at {q}"); }
            // the other modes ignore it
            assert_eq!(Schedule::new_with(&c, p, Mode::Exact, k).unwrap().layers, Schedule::new(&c, p, Mode::Exact).unwrap().layers);
            if p > 300 {
                let r = Mode::Replay { keep: 300 };
                assert_eq!(Schedule::new_with(&c, p, r, k).unwrap().layers, Schedule::new(&c, p, r).unwrap().layers);
            }
        }
        // FIXB's prompt: 550 more rows per decoder layer (11,000 layer-rows, +8.5 % of the CED prefill)
        let (p, k) = (5286, 4736);
        let a = Schedule::new_with(&c, p, Mode::Ced, Some(k)).unwrap().layer_rows();
        let b = Schedule::new(&c, p, Mode::Ced).unwrap().layer_rows();
        assert_eq!(a - b, 11_000);
    }
}

#[cfg(test)]
mod p4_tail_merge_tests {
    use super::{chunks, chunks_elastic};
    #[test]
    fn dsv41_p4_elastic_last_chunk() {
        assert_eq!(chunks_elastic(0, 4185, 2048, 0), chunks(0, 4185, 2048));
        assert_eq!(chunks_elastic(0, 4185, 2048, 512), vec![(0, 2048), (2048, 4185)]);
        assert_eq!(chunks_elastic(2048, 4185, 2048, 512), vec![(2048, 4185)], "a single chunk stays");
        assert_eq!(chunks_elastic(0, 4608, 2048, 512), vec![(0, 2048), (2048, 4608)], "a remainder of exactly 512 rides along: 2,560 rows");
        assert_eq!(chunks_elastic(0, 4609, 2048, 512), vec![(0, 2048), (2048, 4096), (4096, 4609)], "513 rows: a chunk of its own");
        assert_eq!(chunks_elastic(0, 16180, 2048, 512).len(), 8);
        assert_eq!(chunks_elastic(0, 8280, 2048, 512).last(), Some(&(6144, 8280)));
        for p in 1..9000 { let v = chunks_elastic(0, p, 2048, 512); assert!(v.iter().all(|&(a, b)| b - a <= 2560 && b > a)); assert_eq!(v.last().unwrap().1, p); assert!(v.windows(2).all(|w| w[0].1 == w[1].0)); }
    }
}
