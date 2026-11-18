//! Sampler parameter types for the device sampler (`SampParams`, `RqStage`, `PenParams`, `DraftTail`),
//! `stop_token_stats` and the logits row accessors. Greedy MTP stays bitwise lossless (AGENTS 2.7).
//! See `mod.rs` for the file map.

use super::*;

/// API-parity G2: one row's sampler params for `xq_sample_rows` (temperature -> top-k ->
/// top-p -> min_p -> multinomial). `counter` must differ per sampled row (the uniform is
/// hash(seed, counter)); callers advance it by the rows they consume.
#[derive(Clone, Copy, Debug)]
pub struct SampParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32, // 0 = off
    pub min_p: f32, // 0 = off
    pub seed: u64,
    pub counter: u32,
}

impl SampParams {
    /// None when the params are greedy (temperature ~0 or top_k == 1): the argmax id stands.
    pub fn sampled(temperature: f32, top_p: f32, top_k: usize, min_p: f32, seed: u64, counter: u32) -> Option<Self> {
        if temperature < 1e-6 || top_k == 1 {
            return None;
        }
        let top_p = if top_p <= 0.0 || top_p > 1.0 { 1.0 } else { top_p };
        let top_k = if top_k >= u32::MAX as usize { 0 } else { top_k as u32 };
        Some(Self { temperature, top_p, top_k, min_p: min_p.max(0.0), seed, counter })
    }
    pub(super) fn pack(&self) -> [u32; 8] {
        self.pack_flag(1)
    }
    /// WP24: the same row with row flag `flag` (1 = sampled, 2 = sampled real-q verify row).
    pub(crate) fn pack_flag(&self, flag: u32) -> [u32; 8] {
        [flag, self.temperature.to_bits(), self.top_p.to_bits(), self.top_k,
         self.seed as u32, (self.seed >> 32) as u32, self.counter, self.min_p.to_bits()]
    }
}

/// S-B9-REL-SMALLFIXES item 5(a): where a finishing stop token stood in the distribution it was
/// drawn from. `row` = the step's logits row (f16 bits, AFTER the penalties: xq_pen_rows edits
/// sc.logits in place before the argmax and the sampler), `tok` = the token, `samp` = the row's
/// sampler (None = a greedy row). Returns (p, rank):
///   rank = 1 + the number of tokens with a strictly greater logit (1 = the top token);
///   p    = a sampled row: the token's probability in the law xq_sample_rows draws from (logits / T;
///          candidates = top-k, every logit >= the k-th largest, and the min_p floor
///          l >= l_max + T ln(min_p); then the smallest descending prefix whose mass reaches top_p
///          of the candidates' mass, boundary ties included; renormalized). 0 = truncated away.
///          A greedy row: the softmax probability at temperature 1 (an argmax has no law to take
///          a probability under; rank 1 says the token was the choice).
/// A host recomputation in f64 of what the kernel does in f32: p agrees to rounding, and a top-p
/// boundary decided by the last bits of the mass sum can fall on the other side.
/// None = `tok` out of range, or its logit / the row maximum is not finite.
///
/// S-B9-REL-SURFACE2 item 2 (cost): the logits are f16, so a row holds at most 65,536 distinct
/// values. One histogram pass replaces the sort of the whole vocabulary (the descending order is
/// the order of the bit patterns) and one exp per DISTINCT value replaces one per token. Every
/// floating-point sum still adds the same terms in the same order as the first implementation
/// (kept verbatim in the tests as `stop_token_stats_reference`), so (p, rank) is bit-identical.
/// Measured on a 248,320-logit row: 3.6 ms worst case before (top-k off, top-p < 1), 0.5 ms now.
pub fn stop_token_stats(row: &[u16], tok: usize, samp: Option<&SampParams>) -> Option<(f64, u32)> {
    let lg = |b: u16| half::f16::from_bits(b).to_f32() as f64;
    let xt = lg(*row.get(tok)?);
    let mut cnt = vec![0u32; 1 << 16];
    for &b in row { cnt[b as usize] += 1; }
    // The distinct values present, descending: +inf .. +0.0 are the patterns 0x7C00 down to 0,
    // -0.0 .. -inf the patterns 0x8000 up to 0xFC00 (NaN patterns lie outside both ranges: a NaN
    // logit is never a candidate and never outranks anything). Values that compare equal share a
    // group (+0.0 and -0.0), as they share a tie run in a sorted row.
    let mut groups: Vec<(f64, u32)> = Vec::new();
    for b in (0x0000u32..=0x7C00).rev().chain(0x8000..=0xFC00) {
        let n = cnt[b as usize];
        if n == 0 { continue; }
        let x = lg(b as u16);
        match groups.last_mut() {
            Some(g) if g.0 == x => g.1 += n,
            _ => groups.push((x, n)),
        }
    }
    let lmax = groups.first().map_or(f64::NEG_INFINITY, |g| g.0);
    if !xt.is_finite() || !lmax.is_finite() { return None; }
    let rank = 1 + groups.iter().take_while(|g| g.0 > xt).map(|g| g.1).sum::<u32>();
    let Some(s) = samp else {
        // exp(x - lmax) once per pattern present; the sum runs over the row in row order
        let mut tab = vec![0f64; 1 << 16];
        for (b, &n) in cnt.iter().enumerate() {
            if n > 0 { tab[b] = (lg(b as u16) - lmax).exp(); }
        }
        let z: f64 = row.iter().map(|&b| tab[b as usize]).filter(|w| w.is_finite()).sum();
        return Some(((xt - lmax).exp() / z, rank));
    };
    let t = (s.temperature as f64).max(1e-6);
    let w = |x: f64| ((x - lmax) / t).exp();
    // top-k: tau = the k-th largest logit (every tie of it survives)
    let tau = if s.top_k > 0 && (s.top_k as usize) < row.len() {
        let valid: usize = groups.iter().map(|g| g.1 as usize).sum();
        let k = (s.top_k as usize - 1).min(valid - 1);
        let mut seen = 0usize;
        groups.iter().find(|g| { seen += g.1 as usize; seen > k }).map_or(f64::NEG_INFINITY, |g| g.0)
    } else {
        f64::NEG_INFINITY
    };
    let mfloor = if s.min_p > 0.0 { lmax + t * (s.min_p as f64).ln() } else { f64::NEG_INFINITY };
    let is_cand = |x: f64| x >= tau && x >= mfloor;
    // the candidates' weights, once per pattern. Candidacy is its own table: a weight can be NaN
    // (an infinite temperature and a -inf logit), and the sum must carry it as the first
    // implementation's did.
    let mut wtab = vec![0f64; 1 << 16];
    let mut ctab = vec![false; 1 << 16];
    for (b, &n) in cnt.iter().enumerate() {
        if n > 0 {
            let x = lg(b as u16);
            if is_cand(x) { ctab[b] = true; wtab[b] = w(x); }
        }
    }
    let mass: f64 = row.iter().filter(|&&b| ctab[b as usize]).map(|&b| wtab[b as usize]).sum();
    // top-p: theta = the logit at which the descending mass first reaches top_p * mass
    let mut theta = f64::NEG_INFINITY;
    let cut = s.top_p < 1.0 && mass > 0.0;
    if cut {
        let (target, mut cum) = (s.top_p as f64 * mass, 0.0f64);
        for &(x, n) in groups.iter().filter(|g| is_cand(g.0)) {
            let wx = w(x);
            for _ in 0..n { cum += wx; } // one add per token, as a pass over the sorted row makes
            theta = x;
            if cum >= target { break; }
        }
    }
    if !(is_cand(xt) && xt >= theta) { return Some((0.0, rank)); }
    // the kept mass: descending over the nucleus after a cut, else every candidate in row order
    // (which is `mass`, term for term)
    let z: f64 = if cut {
        groups.iter().filter(|g| is_cand(g.0) && g.0 >= theta)
            .flat_map(|&(x, n)| std::iter::repeat(w(x)).take(n as usize)).sum()
    } else {
        mass
    };
    Some((w(xt) / z, rank))
}

impl FwdModel {
    /// S-B9-REL-SMALLFIXES item 5(a): ONE logits row of the last step, read back to the host
    /// (vocab f16 values, about 0.5 MB). A host sync: call it on a request's FINISHING step only.
    /// Rows `row0 .. row0 + n` of the last step's logits in ONE device-to-host copy (a speculative round's
    /// emitted rows, for the log probabilities of its tokens).
    pub fn logits_rows(&self, sc: &Scratch, row0: usize, n: usize) -> Result<Vec<u16>> {
        let v = self.cfg.vocab_size;
        anyhow::ensure!((row0 + n) * v <= sc.logits.len(), "logits rows {row0}..{} outside the scratch ({} rows)",
                        row0 + n, sc.logits.len() / v);
        let mut out = vec![0u16; n * v];
        self.dtoh_into(&sc.logits.slice(row0 * v..(row0 + n) * v), &mut out)?;
        Ok(out)
    }

    pub fn logits_row(&self, sc: &Scratch, row: usize) -> Result<Vec<u16>> {
        let v = self.cfg.vocab_size;
        anyhow::ensure!((row + 1) * v <= sc.logits.len(), "logits row {row} outside the scratch ({} rows)", sc.logits.len() / v);
        let mut out = vec![0u16; v];
        self.dtoh_into(&sc.logits.slice(row * v..(row + 1) * v), &mut out)?;
        Ok(out)
    }
}

/// WP24: one MTP round's real-q draft sampler (see `FwdModel::set_spec_sampling`). `tau` = the
/// draft temperature tau_d, `top_p` / `top_k` = the REQUEST's (q' must be the law the pass samples
/// from, truncation included), `seed` = the request's (the draft domain separates the streams),
/// `dctr` = the lane's draft RNG counter (pass i draws dctr + i; a separate stream from the verify
/// rows' counters). mode 1 = --wp24-dump list only, 2 = real-q.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RqStage {
    pub mode: u8,
    pub tau: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub seed: u64,
    pub dctr: u32,
}

impl RqStage {
    /// The real-q draft sampler of a lane under `--spec-sampling ratio` (mode 2) or the dump rung
    /// (mode 1), for its NEXT round — None for a greedy lane (SampParams::sampled's rule: greedy
    /// stays on the bitwise match path) and when neither is on. `draft_temp` = --draft-temperature
    /// (None = the request's temperature). ONE rule for the server and the WP24 gate.
    pub fn for_lane(ratio: bool, draft_temp: Option<f32>, temperature: f32, top_p: f32, top_k: usize,
                    min_p: f32, seed: u64, dctr: u32) -> Option<RqStage> {
        let sp = SampParams::sampled(temperature, top_p, top_k, min_p, seed, 0)?;
        let mode = if ratio { 2 } else if crate::wp24::dump_on() { 1 } else { return None };
        Some(RqStage {
            mode,
            tau: draft_temp.unwrap_or(sp.temperature),
            top_p: sp.top_p,
            top_k: sp.top_k,
            seed,
            dctr,
        })
    }
    pub(super) fn pack(&self) -> [u32; 8] {
        [1, self.tau.to_bits(), self.top_p.to_bits(), self.top_k,
         self.seed as u32, (self.seed >> 32) as u32, 0, 0]
    }
}

/// WP15: one row's penalty params for `xq_pen_rows` / `xq_pen_draft` — the reference implementation's ComboSampler
/// semantics (SS_RepP + SS_PresFreqP over `sustain` full-strength + `decay` fading positions of
/// the full sequence, prompt included; the reference implementation server's default is 1024 + 1024).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PenParams {
    pub rep: f32,
    pub pres: f32,
    pub freq: f32,
    pub sustain: u32,
    pub decay: u32,
}

/// Token ring per slot for the penalty window (positions & (PEN_RING - 1)).
pub const PEN_RING_LOG2: u32 = 13;
pub const PEN_RING: usize = 1 << PEN_RING_LOG2;
/// Largest --penalty-range: the window (sustain + decay - 1) plus a verify's width must fit the
/// ring with room to spare, so an in-flight commit never overwrites a position a row still reads.
pub const PEN_RANGE_MAX: usize = (PEN_RING - 64) / 2;
pub(super) const PEN_SPAN: usize = 4096; // XQ_PEN_SPAN

impl PenParams {
    /// None when the params are a no-op (the reference implementation's SS_RepP / SS_PresFreqP `alt()` NoOp rule:
    /// rep_p == 1 and pres == freq == 0, or an empty range) — the row stays bit-identical.
    pub fn new(rep: f32, pres: f32, freq: f32, sustain: usize, decay: usize) -> Option<Self> {
        let s = sustain.min(PEN_RANGE_MAX) as u32;
        let d = decay.min(PEN_RANGE_MAX) as u32;
        let rep_on = rep != 1.0 && rep.is_finite() && rep > 0.0;
        let pf_on = (pres != 0.0 && pres.is_finite()) || (freq != 0.0 && freq.is_finite());
        if (!rep_on && !pf_on) || s + d == 0 {
            return None;
        }
        Some(Self {
            rep: if rep_on { rep } else { 1.0 },
            pres: if pres.is_finite() { pres } else { 0.0 },
            freq: if freq.is_finite() { freq } else { 0.0 },
            sustain: s,
            decay: d,
        })
    }
    fn flag(&self) -> u32 {
        (self.rep != 1.0) as u32 | (((self.pres != 0.0 || self.freq != 0.0) as u32) << 1)
    }
    /// DHEADP (w3): true when xq_pen_tok can only LOWER a logit (rep >= 1, pres >= 0, freq >= 0 —
    /// see the monotonicity note on xq_pen_tok). rep < 1 or a negative pres/freq (accepted with a
    /// warning, as the reference engine does) can raise one: those requests keep the old draft head.
    pub fn monotone(&self) -> bool {
        self.rep >= 1.0 && self.pres >= 0.0 && self.freq >= 0.0
    }
    pub(super) fn pack(&self) -> [u32; 8] {
        [self.flag(), self.rep.to_bits(), self.pres.to_bits(), self.freq.to_bits(),
         self.sustain, self.decay, 0, 0]
    }
}

/// DHEADP (w3): a draft pass's lm_head tail (FwdModel::draft_tail).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DraftTail {
    /// DHEAD: screen + exact rescore (unpenalized drafts)
    Dhead,
    /// DHEADP: screen + bound-expanded, penalized rescore (monotone penalized drafts)
    DheadP,
    /// the full slice head (+ xq_pen_draft when penalized)
    Old,
    /// A5-L4: an unpenalized real-q pass on DHEAD (screen + rescore + xq_rq_draft_dh)
    DheadRq,
}

/// --exl3-pen-draft=0: the draft head proposes from UNpenalized logits (diagnostics A/B of
/// the draft mirror; draft-only either way — the verify decides every emitted token).
pub(super) fn pen_draft_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-pen-draft")).as_deref() != Ok("0"))
}
