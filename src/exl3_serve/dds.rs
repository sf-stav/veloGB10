//! The depth policy of the draft chain: `DraftCal`, the depth bins and cost guard (WP23), `CostFit`, and the world-aware DDS prior.
//! Draft-only: the verify decides every emitted token (greedy stays bitwise lossless at any width up to 16 by batch invariance, AGENTS.md section 2).
//! Neighbours: `step.rs` / `mtp_round.rs` consume these; `boot_prior_word` is the prior as a u32 word.

use super::*;

impl DraftCal {
    const BIN_W: f64 = 1.0;

    const DECAY: f64 = 0.995;

    const MIN_COUNT: f64 = 8.0;

    pub(super) const BURN_IN: f64 = 64.0;

    pub(crate) fn new() -> Self {
        let k = crate::exl3_forward::mtp_depth_default();
        let dbins = if wp23_on(crate::opt!("exl3-dds-depth-bins"), k) {
            Some(vec![Default::default(); WP23_DEPTH_BUCKETS])
        } else {
            None
        };
        Self { bins: Default::default(), total: 0.0, dbins }
    }

    /// `i` = the draft's pass index in its round (0 = the first draft); only the WP23 depth bins
    /// read it — the pooled map is updated exactly as before.
    pub(crate) fn add_label(&mut self, i: usize, score: f32, accepted: bool) {
        let idx = (score as f64 / Self::BIN_W).floor() as i64;
        let b = self.bins.entry(idx).or_insert((0.0, 0.0));
        b.0 += 1.0;
        if accepted { b.1 += 1.0; }
        self.total += 1.0;
        if let Some(d) = self.dbins.as_mut() {
            let b = d[wp23_depth_bucket(i)].entry(idx).or_insert((0.0, 0.0));
            b.0 += 1.0;
            if accepted { b.1 += 1.0; }
        }
    }

    /// Once per verification round, so the map tracks drift (prose vs code) over ~hundreds of rounds.
    pub(crate) fn decay_step(&mut self) {
        for b in self.bins.values_mut() { b.0 *= Self::DECAY; b.1 *= Self::DECAY; }
        self.total *= Self::DECAY;
        if let Some(d) = self.dbins.as_mut() {
            for m in d.iter_mut() {
                for b in m.values_mut() { b.0 *= Self::DECAY; b.1 *= Self::DECAY; }
            }
        }
    }

    /// WP23: acceptance estimate for pass `i`'s draft. Depth bins on: the nearest populated bin
    /// at or below the score in i's depth bucket; a bucket with none there (sparse at deep levels
    /// on prose) falls back to the pooled estimate. Depth bins off: exactly `estimate`.
    pub(crate) fn estimate_at(&self, i: usize, score: f32) -> f64 {
        let Some(d) = self.dbins.as_ref() else { return self.estimate(score) };
        if self.total < Self::BURN_IN { return 1.0; }
        let idx = (score as f64 / Self::BIN_W).floor() as i64;
        let near = d[wp23_depth_bucket(i)].range(..=idx).rev()
            .find(|(_, v)| v.0 >= Self::MIN_COUNT).map(|(_, v)| *v);
        match near { Some((t, a)) => a / t, None => self.estimate(score) }
    }

    /// Estimated conditional acceptance for a draft with this score: the nearest populated bin at
    /// or below it (else the lowest populated); optimistic 1.0 while still learning.
    pub(super) fn estimate(&self, score: f32) -> f64 {
        if self.total < Self::BURN_IN { return 1.0; }
        let idx = (score as f64 / Self::BIN_W).floor() as i64;
        let mut below: Option<(f64, f64)> = None;
        let mut lowest: Option<(f64, f64)> = None;
        for (&k, &v) in self.bins.iter() {
            if v.0 < Self::MIN_COUNT { continue; }
            if lowest.is_none() { lowest = Some(v); }
            if k <= idx { below = Some(v); } else { break; }
        }
        match below.or(lowest) { Some((t, a)) => a / t, None => 1.0 }
    }
}

// ---- WP23 (PLAN/SURPASS_PLAN_2026-09-26.md): draft depth 6/7 — DDS depth bins + cost guard ----
// Draft-only: the verify decides every emitted token (greedy bitwise at any width <= 16 by batch
// invariance); these only choose how many drafts a round proposes.

/// The pre-WP23 depth ceiling: at a served depth <= this, both WP23 DDS parts default OFF, so
/// --exl3-mtp-k=5 reproduces the K5 build's DDS decisions (the brief's confound check).
const WP23_LEGACY_MAX_K: usize = 5;

/// Depth buckets of the WP23 bins: pass {0} | {1,2} | {3,4} | {5..} — levels 6/7 share a bucket so
/// it fills on code at ~1.5 labels/round (a per-level bucket at 1-logit bins stays under MIN_COUNT).
pub(super) const WP23_DEPTH_BUCKETS: usize = 4;

pub(super) fn wp23_depth_bucket(i: usize) -> usize {
    match i { 0 => 0, 1 | 2 => 1, 3 | 4 => 2, _ => 3 }
}

/// WP23 part switch: `name`=0|off / 1|on forces it (diagnostic A/B, never a serving default);
/// unset = on iff the served depth k exceeds the pre-WP23 ceiling.
pub(super) fn wp23_on(name: crate::opts::OptId, k: usize) -> bool {
    match crate::opts::var(name).as_deref() {
        Ok("0") | Ok("off") => false,
        Ok("1") | Ok("on") => true,
        _ => k > WP23_LEGACY_MAX_K,
    }
}

/// WP23 cost guard switch (--exl3-dds-guard=0|1; default on iff k > 5). k = the scheduler's
/// served depth (one per process), so the env is read once.
pub(crate) fn wp23_guard_on(k: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| wp23_on(crate::opt!("exl3-dds-guard"), k))
}

impl Default for CostFit {
    fn default() -> Self {
        CostFit { prior: Self::PRIOR_SLOPE, w: 0.0, sx: 0.0, sy: 0.0, sxx: 0.0, sxy: 0.0, n: 0 }
    }
}

/// TP-I #7 (T2 lever): the WP23 cost guard's prior ĉ for a TP=2 lane. The 6.0 ms/draft seed was fitted
/// to TP=1 costs; a TP=2 draft costs a verify row at the TP=2 width slope (1.92 ms/row, TP-I re-ledger:
/// r0 union m=4 22.91 -> m=8 30.60 ms/verify) plus one sharded draft pass (0.76 ms) = ~2.7 ms. With
/// code that always drafts to the cap the fit never identifies the slope, so the prior IS the guard.
///   unset / on : 2.7 at TP (world 2) — only lanes of a TP scheduler; TP=1 keeps 6.0 by construction
///   0 / off    : 6.0 (the TP=1 constant, the pre-TP-I behaviour)
///   <ms>       : that prior (diagnostic)
/// --tp-dds-prior rides TpConfig's CLI option-registry snapshot; the value joins the TP boot agree hash. The guard's
/// timing inputs stay the HEAD's (tp_round), so both ranks' targets are identical.
pub(crate) const TP2_DDS_PRIOR: f64 = 2.7;

/// World-4 default: an ESTIMATE, not a measurement — TP-4A (24.7-17.59)/5 = 1.42 ms/row + 0.73 draft = 2.15; see TP-4F_PREP_REPORT.md.
pub(crate) const TP4_DDS_PRIOR: f64 = 2.1;

/// CLI-1 (--print-config): the TP=2 prior this process resolves.
pub fn dds_prior_tp() -> f64 { dds_prior(true) }

/// CLI-1 (--print-config): the TP=4 prior this process resolves.
pub fn dds_prior_tp4() -> f64 { dds_prior_w(4) }

pub(crate) fn dds_prior(tp: bool) -> f64 { dds_prior_w(if tp { 2 } else { 1 }) }

/// The prior for a scheduler of `world` ranks (1 = no TP): 6.0 / 2.7 / 2.1 at world 1 / 2 / 4 (any other world
/// keeps the TP=2 value); `--tp-dds-prior` (off = 6.0, a number = that prior) applies at every TP world.
pub(crate) fn dds_prior_w(world: usize) -> f64 {
    if world < 2 { return CostFit::PRIOR_SLOPE; }
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    resolve_dds_prior(RAW.get_or_init(|| crate::opts::var(crate::opt!("tp-dds-prior")).ok()).as_deref(), world)
}

/// Pure resolution of the prior from the raw `--tp-dds-prior` value (None = unset) at a TP `world` >= 2. The
/// world-2 arms are the pre-TP-4F code verbatim: on/unset/garbage = 2.7, off = 6.0, a positive number = itself.
pub(crate) fn resolve_dds_prior(raw: Option<&str>, world: usize) -> f64 {
    let dflt = if world == 4 { TP4_DDS_PRIOR } else { TP2_DDS_PRIOR };
    match raw.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if v == "0" || v == "off" || v == "false" => CostFit::PRIOR_SLOPE,
        Some(v) if v == "1" || v == "on" || v.is_empty() => dflt,
        Some(v) => v.parse::<f64>().ok().filter(|x| *x > 0.0).unwrap_or(dflt),
        None => dflt,
    }
}

/// The prior's word in the TP boot agree hash (`tp_boot_agree`).
pub(crate) fn boot_prior_word(world: usize) -> u32 { (dds_prior_w(world) as f32).to_bits() }

impl CostFit {
    pub(crate) fn with_prior(prior: f64) -> Self { CostFit { prior, ..Default::default() } }

    const DECAY: f64 = 0.97;       // ~33-round memory: the context (and a row's cost) drifts slowly

    pub(super) const PRIOR_SLOPE: f64 = 6.0;  // ms per draft (the brief's seed)

    const PRIOR_W: f64 = 4.0;      // prior weight in (drafts - mean)^2 units

    pub(crate) fn add(&mut self, x: f64, y: f64) {
        let d = Self::DECAY;
        self.w = d * self.w + 1.0;
        self.sx = d * self.sx + x;
        self.sy = d * self.sy + y;
        self.sxx = d * self.sxx + x * x;
        self.sxy = d * self.sxy + x * y;
        self.n += 1;
    }

    /// ĉ (ms per draft), >= 0.
    pub(crate) fn slope(&self) -> f64 {
        let (cxx, cxy) = if self.w > 0.0 {
            ((self.sxx - self.sx * self.sx / self.w).max(0.0), self.sxy - self.sx * self.sy / self.w)
        } else {
            (0.0, 0.0)
        };
        ((cxy + Self::PRIOR_W * self.prior) / (cxx + Self::PRIOR_W)).max(0.0)
    }
}
