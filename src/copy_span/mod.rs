//! WIDEVERIFY1 R1 + R4 (revised): copy-span drafting, the HOST-SIDE proposer.
//!
//! When the output is copying text that already exists in the context (prompt + tokens committed
//! so far), propose the next tokens of that earlier text as the draft of a speculative round. Pure
//! host integer logic: no GPU call, no model tensor, no logits. A wrong proposal is rejected by the
//! target verify exactly like a wrong neural draft, so this module can cost speed, never bytes.
//!
//! Recovered from the parked `archive/lookup-drafter:src/lookup.rs` (Stage A, in-context index
//! only; the corpus-table mode and the bench-only stats are NOT carried over), then extended with
//! what the WIDEVERIFY1 Phase 0 replay showed is required (`PLAN/WIDEVERIFY1_GATE.md`):
//!
//! * **Index (parked rule).** For every n-gram length `L` in `[min_n, max_n]` (`max_n =
//!   max(8, min_n)`) a hash map `key(ngram) -> most recent start position`. An n-gram is indexed
//!   only once its continuation token exists, so a query tail can never match itself. Built once
//!   over the prompt, extended incrementally per committed token. A search tries the LONGEST
//!   suffix first; the most recent earlier occurrence wins; the matched tokens are re-compared on
//!   a hash hit, so a collision can only SUPPRESS a proposal, never fabricate one.
//! * **Copy cursor.** After a round whose proposal was (partly) accepted AND whose model token
//!   equals the next source token, the proposer stays on that source: the next round proposes
//!   `ctx[cur..cur+W]` with no search. Reference semantics: `sim(..., cursor_mode=True)` in
//!   `wvtools/phase0/analyse.py` of the WIDEVERIFY1 fork. This fixes the repetitive-records case
//!   (JSONL: "most recent occurrence" lands on the wrong record).
//! * **Establishment gate (R4, revised).** A SEARCH may start a copy only on a match of at least
//!   `min_n` tokens (serving default 16: min_n 4 lost on agent-style output in the replay). While a
//!   cursor is live no minimum applies. Why a round got no proposal is exposed
//!   (`MissReason::NoMatch` / `BelowMin`, and how the previous copy ended: `CopyEnd`).
//!
//! * **Yield gate (WIDEVERIFY1 refinement).** A copy round always pays for a full-width verify;
//!   on "agent echo" output under sampling the model departs from the quoted source often, so a
//!   copy yields fewer tokens than the neural round it replaced. Per lane, the proposer keeps two
//!   fixed-point moving averages of agreed round outcomes (copy rounds, and the neural rounds the
//!   serve loop reports) and holds back a SEARCH that would start a new copy when copies do not
//!   pay; a live cursor is never gated. See [`YieldGate`].
//!
//! * **Start ramp (WIDEVERIFY1 refinement 1).** A copy STARTS narrow and widens while it is
//!   accepted: the first round of a copy (a SEARCH proposal) carries at most `start_w` tokens,
//!   each following cursor round of the same copy doubles that cap (`start_w`, `2 start_w`, ...)
//!   up to the caller's `max_w`; a dropped cursor ends the copy, so the next copy starts at
//!   `start_w` again. On the MoE target every verify row is paid for (about 3.8 ms on one box),
//!   so a copy the model rejects at its first tokens should cost a 4-row verify, not an 11-row
//!   one. The ramp state is a function of agreed round outcomes only (the proposal it was made
//!   for and the round's accepted count), so it is identical on every TP rank.
//!   `start_w >= max_w` (or `usize::MAX`, the `CopySpanCfg::new` default) = full width from the
//!   first round: exactly the proposals before the ramp existed.
//!
//! One instance per serving lane. Allocation: `propose` allocates the proposal's token `Vec`
//! (at most `max_w` u32) and nothing else (the per-length hash scratch is a field, sized once).
//! `commit` allocates nothing per token beyond the amortised growth of the context vector and the
//! hash maps.
//!
//! Index keys: a per-token multiplicative hash taken over the n-gram REVERSED (last token first),
//! so the keys of every length ending at one position come out of one pass of `max_n` steps; a
//! 64-bit finaliser spreads the bits (the maps use the key as their hash). Determinism: the maps
//! are only probed by key, never iterated, and inserts overwrite, so one token sequence gives one
//! index and one proposal sequence.
//!
//! File map (this was the single `copy_span.rs`; every item keeps its path): `mod.rs` = the host-side copy-span proposer (n-gram index, copy cursor, replay probe; no GPU); `tests.rs` = the inline unit tests (module `tests`, same name).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::Write;

/// The serving default of `--copy-span-min-n` (the WIDEVERIFY1 gate measurement).
pub const DEFAULT_MIN_N: usize = 16;
/// The serving default of `--copy-span-max-w` on the EXL3 chain verify (`MTP_MAX_K`).
pub const DEFAULT_MAX_W: usize = 10;
/// The serving default of `--copy-span-start-w` (refinement 1: a copy starts at 4 tokens, then 8,
/// then the width cap).
pub const DEFAULT_START_W: usize = 4;
/// Shortest match counted as "below the minimum" (diagnostics only; never proposes).
pub const DEFAULT_DIAG_N: usize = 4;
/// Fixed-point scale of the yield gate's averages (value x 256).
pub const YG_ONE: i64 = 256;
/// The yield gate lets every `YG_PROBE_EVERY`-th start it would deny through as a probe.
pub const YG_PROBE_EVERY: u64 = 16;

/// The key is already a hash: pass it through.
#[derive(Default, Clone, Copy)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ b as u64).wrapping_mul(HK);
        }
    }
    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

type KeyMap = HashMap<u64, u32, BuildHasherDefault<KeyHasher>>;

const HK: u64 = 0x9E37_79B9_7F4A_7C15;
const HSEED: u64 = 0x243F_6A88_85A3_08D3;

#[inline]
fn hstep(h: u64, t: u32) -> u64 {
    (h.rotate_left(23) ^ t as u64).wrapping_mul(HK)
}

/// murmur3 fmix64: the map takes the key as its hash, so every bit must be mixed.
#[inline]
fn hfin(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

/// Proposer knobs. `min_n` is the establishment gate of a search; `max_n` the longest suffix a
/// search tries; `diag_n` (< `min_n`, else unused) the shortest match reported as `BelowMin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopySpanCfg {
    pub min_n: usize,
    pub max_n: usize,
    pub diag_n: usize,
    /// The yield gate's `W` (the copy width cap); 0 = the gate is off (every start allowed).
    pub yield_w: usize,
    /// The start ramp: the most tokens the FIRST round of a copy proposes; every following cursor
    /// round of that copy doubles it (capped by the caller's `max_w` each round). `usize::MAX` (the
    /// `new` default) or any value >= `max_w` = full width from the first round (no ramp).
    pub start_w: usize,
    /// COPY27B: the yield gate's cost-ratio rule (x256; 0 = the EXL3 rule with `yield_w`). Nonzero: a NEW copy
    /// may start only while `ec * 256 >= en * yield_ratio` (copy rounds must emit `yield_ratio / 256` times the
    /// tokens of a neural round: the trunk's verify costs nearly the same at 3 and at 16 rows, so the ratio of
    /// the two round costs is the bar). Only read when the gate is armed (`yield_w > 0`).
    pub yield_ratio: i64,
}

impl CopySpanCfg {
    /// The reference configuration: `max_n = max(8, min_n)`, `diag_n = min(4, min_n)`.
    pub fn new(min_n: usize) -> Self {
        assert!(min_n >= 1, "copy-span: min_n must be >= 1");
        CopySpanCfg { min_n, max_n: min_n.max(8), diag_n: DEFAULT_DIAG_N.min(min_n), yield_w: 0, start_w: usize::MAX,
                      yield_ratio: 0 }
    }
    /// COPY27B: the yield gate's cost-ratio rule (`r` x256; see `yield_ratio`). Arm the gate itself with
    /// `with_yield_gate`.
    pub fn with_yield_ratio(mut self, r: i64) -> Self {
        assert!(r >= 0, "copy-span: yield ratio must be >= 0");
        self.yield_ratio = r;
        self
    }
    /// Arm the start ramp: a copy's first round proposes at most `w` tokens (`w >= 1`; a value
    /// >= the caller's `max_w` = no ramp).
    pub fn with_start_w(mut self, w: usize) -> Self {
        assert!(w >= 1, "copy-span: start_w must be >= 1");
        self.start_w = w;
        self
    }
    /// Arm the yield gate with copy width cap `w` (0 = off).
    pub fn with_yield_gate(mut self, w: usize) -> Self {
        self.yield_w = w;
        self
    }
    fn has_diag(&self) -> bool {
        self.diag_n >= 1 && self.diag_n < self.min_n
    }
}

impl Default for CopySpanCfg {
    fn default() -> Self {
        Self::new(DEFAULT_MIN_N)
    }
}

/// Where a proposal came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The live copy cursor (no search ran).
    Cursor,
    /// A fresh index search; `match_len` = the matched suffix length (>= `min_n`).
    Search { match_len: usize },
    /// COPYREC1: the cross-request record (`--copy-span-record`) offered the copy: the first round of a copy whose
    /// source is the record's continuation of a `match_len`-token match (later rounds of it are `Cursor`).
    Record { match_len: usize },
}

/// COPYREC1: `CopySpan::cursor` reports a cursor into the record copy's source with this bit set, so a position in
/// the record copy never equals a position in the context (`--copy-span-prefill`'s schedule keys on it).
pub const EXT_TAG: usize = 1 << 48;

/// A draft proposal. `tokens` is never empty and never runs past the end of the context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    pub tokens: Vec<u32>,
    /// Position in the context of `tokens[0]` (the copy source).
    pub src: usize,
    pub source: Source,
    /// The `max_w` this proposal was made for (a "full" round accepts all `max_w` tokens).
    pub max_w: usize,
    /// Context length when proposed; `commit` ignores a proposal made for another length.
    pub ctx_len: usize,
    /// The start-ramp cap this proposal was made under (`tokens.len() <= min(max_w, ramp)`);
    /// `usize::MAX` = no ramp. A cursor that survives this round's commit doubles it.
    pub ramp: usize,
    /// COPYREC1: `src` indexes the record copy's source (the lane's `ext`), not the context.
    pub ext: bool,
}

/// Why `propose` returned no proposal (set by the search; `None` when a proposal was made).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissReason {
    /// No earlier occurrence of even a `diag_n`-token suffix (or of a `min_n` one when
    /// `diag_n >= min_n`).
    NoMatch,
    /// A match of at least `diag_n` but fewer than `min_n` tokens exists: the gate held it back.
    BelowMin,
    /// A match of at least `min_n` tokens exists, but the yield gate held the new copy back
    /// (copies have not been paying on this lane; see [`YieldGate`]).
    YieldGate,
}

/// How the previous copy ended (set by `commit` when a cursor was not carried forward).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyEnd {
    /// The model's token differed from the source (includes rounds with zero accepted).
    Mismatch,
    /// The accepted span reached the end of the context: there is no next source token.
    EndOfContext,
    /// The round emitted no token after the accepted ones (the request ended).
    ShortEmit,
    /// A cursor was live but the round committed without using a proposal (or with a proposal
    /// made for another context length).
    Unused,
}

/// Per-lane counters (monotonic until `reset`).
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopySpanCounters {
    /// index searches run (a cursor proposal runs none)
    pub searches: u64,
    pub proposals_search: u64,
    pub proposals_cursor: u64,
    pub proposed_tokens: u64,
    /// searches that found nothing / found only a match below `min_n`
    pub miss_no_match: u64,
    pub miss_below_min: u64,
    /// rounds committed with a proposal, and the proposal tokens they accepted
    pub rounds_with_proposal: u64,
    pub accepted_tokens: u64,
    /// tokens committed by rounds with a proposal (accepted + the model token)
    pub emitted_with_proposal: u64,
    pub rounds_zero_accepted: u64,
    /// rounds that accepted all `max_w` proposed tokens
    pub rounds_full: u64,
    /// rounds committed without a proposal
    pub rounds_without_proposal: u64,
    pub committed_tokens: u64,
    pub end_mismatch: u64,
    pub end_of_context: u64,
    pub end_short_emit: u64,
    pub end_unused: u64,
    /// `accepted` larger than the matching prefix of `emitted` (clamped; a caller bug)
    pub inconsistent_accept: u64,
    /// copy starts the yield gate denied (a `min_n` match existed); probes are not counted here
    pub gate_denied: u64,
    /// copy starts the yield gate would have denied but let through as a probe (1 in 16)
    pub gate_probes: u64,
    /// proposals the start ramp held below the round's width cap (`ramp < max_w`): a copy's first
    /// round (search) / a following round still ramping up (cursor)
    pub narrow_search: u64,
    pub narrow_cursor: u64,
}

/// What the yield gate decided for one copy start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateDecision {
    /// the rule allows the start (or the gate has too few samples, or is off)
    Allow,
    /// the rule denies it, but this is the 16th denied start in a row: let it through
    Probe,
    Deny,
}

/// The per-lane yield gate of copy starts. Deterministic on every TP rank: it only ever sees token
/// COUNTS of rounds every rank agreed on (no clock, no floating point); integer fixed point (x256).
///
/// * `ec` = moving average of tokens emitted per copy round (every copy round, weight 1/4);
/// * `en` = moving average of tokens emitted per NEURAL speculative round (weight 1/8; plain and
///   control steps are not counted); the first sample of each initialises it.
///
/// Rule (only when a search would start a NEW copy; cursor continuations are never gated): with
/// fewer than 2 copy samples or no neural sample, allow; otherwise allow iff
/// `ec * (4 + en) >= en * (3 + W)` (W = the copy width cap, everything in the x256 scale).
///
/// Why: a round costs roughly a constant plus a per-drafted-token term, the constant worth about
/// 4 drafted tokens. A neural round drafts about as many tokens as it emits, so it costs ~`4 + en`
/// for `en` tokens; a copy round is a full-width verify minus its draft passes (about one token's
/// worth), ~`4 + W - 1 = 3 + W` for `ec` tokens. A copy pays when `ec / (3 + W) >= en / (4 + en)`.
/// With W = 10: at en = 7.5 the bar is ec >= 8.5; at en = 3 it is ec >= 5.6.
///
/// A denied start leaves the lane "suspended"; every 16th denied start in a row is let through as
/// a probe (its copy rounds update `ec`, so the gate recovers when the text turns copy-heavy). An
/// allowed start ends the suspension. `CopySpan::reset` resets it with the lane.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct YieldGate {
    /// tokens per copy round, x256
    pub ec: i64,
    /// tokens per neural speculative round, x256
    pub en: i64,
    /// copy-round samples / neural-round samples
    pub nc: u64,
    pub nn: u64,
    /// denied starts in a row since the last allowed one (0 = not suspended)
    pub denied_run: u64,
}

impl YieldGate {
    /// One copy round emitted `emitted` tokens.
    pub fn note_copy(&mut self, emitted: usize) {
        let x = emitted as i64 * YG_ONE;
        self.ec = if self.nc == 0 { x } else { self.ec + ((x - self.ec) >> 2) };
        self.nc += 1;
    }
    /// One neural speculative round emitted `emitted` tokens.
    pub fn note_neural(&mut self, emitted: usize) {
        let x = emitted as i64 * YG_ONE;
        self.en = if self.nn == 0 { x } else { self.en + ((x - self.en) >> 3) };
        self.nn += 1;
    }
    /// The rule alone (no suspension bookkeeping): does a copy start pay at width cap `w`?
    pub fn rule_allows(&self, w: usize) -> bool {
        if self.nc < 2 || self.nn == 0 {
            return true;
        }
        self.ec * (4 * YG_ONE + self.en) >= self.en * ((3 + w as i64) * YG_ONE)
    }
    /// COPY27B: the cost-ratio rule alone: with fewer than 2 copy samples or no neural sample, allow; otherwise
    /// allow iff `ec * 256 >= en * ratio` (ratio x256).
    pub fn ratio_allows(&self, ratio: i64) -> bool {
        if self.nc < 2 || self.nn == 0 {
            return true;
        }
        self.ec * YG_ONE >= self.en * ratio
    }
    /// COPY27B: `decide` under the cost-ratio rule (same suspension and probe bookkeeping).
    pub fn decide_ratio(&mut self, ratio: i64) -> GateDecision {
        if self.ratio_allows(ratio) {
            self.denied_run = 0;
            return GateDecision::Allow;
        }
        self.denied_run += 1;
        if self.denied_run % YG_PROBE_EVERY == 0 { GateDecision::Probe } else { GateDecision::Deny }
    }
    /// Decide one copy start (updates the suspension run).
    pub fn decide(&mut self, w: usize) -> GateDecision {
        if self.rule_allows(w) {
            self.denied_run = 0;
            return GateDecision::Allow;
        }
        self.denied_run += 1;
        if self.denied_run % YG_PROBE_EVERY == 0 { GateDecision::Probe } else { GateDecision::Deny }
    }
    /// The last decided start was denied (or probed) and no start has been allowed since.
    pub fn suspended(&self) -> bool {
        self.denied_run > 0
    }
}

/// The copy-span proposer of one serving lane.
pub struct CopySpan {
    cfg: CopySpanCfg,
    /// Prompt + committed tokens.
    ctx: Vec<u32>,
    /// `index[L - min_n]`, L in `[min_n, max_n]`: key -> most recent start with a continuation.
    index: Vec<KeyMap>,
    /// Length `diag_n` (only when `diag_n < min_n`): the `BelowMin` classification.
    diag: KeyMap,
    /// `hs[L-1]` = key of the context suffix of length L (search scratch, sized `max_n`).
    hs: Vec<u64>,
    cursor: Option<usize>,
    /// The start-ramp cap of the live cursor's next proposal (meaningful only while `cursor` is
    /// Some; set by `commit` when a copy continues).
    ramp_next: usize,
    /// COPYPF1 gate session: tokens the live copy has produced in agreement with its source (copied tokens accepted +
    /// the model's own continuing token, summed over the copy's rounds); 0 without a live cursor.
    run: usize,
    /// ECHOPRICE (--copy-span-prefill's priced rule): this request's running mean of the lengths of its ESTABLISHED
    /// copies that have ended (`run` at the round the cursor was dropped, that round's accepted tokens included;
    /// halved weight per new copy). 0 = none ended yet. Replicated under TP (every rank commits the same rounds).
    len_hist: f64,
    last_miss: Option<MissReason>,
    last_end: Option<CopyEnd>,
    counters: CopySpanCounters,
    gate: YieldGate,
    /// COPYREC1: the source of the current record copy (the record's continuation, at most
    /// `copy_record::SNAP_MAX` ids, taken when the copy started: the record may change under it), whether the live
    /// cursor indexes it, the copy's band and probe flag, and this request's reads of the record.
    ext: Vec<u32>,
    cursor_ext: bool,
    ext_band: usize,
    ext_probing: bool,
    rec: crate::copy_record::Reader,
}

impl CopySpan {
    /// Build over the prompt (one bulk index pass).
    pub fn new(prompt: &[u32], cfg: CopySpanCfg) -> Self {
        assert!(cfg.min_n >= 1 && cfg.max_n >= cfg.min_n, "copy-span: need 1 <= min_n <= max_n");
        let n_lens = cfg.max_n - cfg.min_n + 1;
        let mut s = CopySpan {
            cfg,
            ctx: Vec::new(),
            index: (0..n_lens).map(|_| KeyMap::default()).collect(),
            diag: KeyMap::default(),
            hs: vec![0; cfg.max_n],
            cursor: None,
            ramp_next: usize::MAX,
            run: 0,
            len_hist: 0.0,
            last_miss: None,
            last_end: None,
            counters: CopySpanCounters::default(),
            gate: YieldGate::default(),
            ext: Vec::new(),
            cursor_ext: false,
            ext_band: 0,
            ext_probing: false,
            rec: crate::copy_record::Reader::default(),
        };
        s.load_prompt(prompt);
        s
    }

    /// Lane recycling: equal to `CopySpan::new(prompt, cfg)` (counters included), keeping the
    /// allocations. Harvest `counters()` before calling.
    pub fn reset(&mut self, prompt: &[u32]) {
        self.ctx.clear();
        for m in &mut self.index {
            m.clear();
        }
        self.diag.clear();
        self.cursor = None;
        self.ramp_next = usize::MAX;
        self.run = 0;
        self.len_hist = 0.0;
        self.last_miss = None;
        self.last_end = None;
        self.counters = CopySpanCounters::default();
        self.gate = YieldGate::default();
        self.ext.clear();
        self.cursor_ext = false;
        self.ext_band = 0;
        self.ext_probing = false;
        self.rec = crate::copy_record::Reader::default();
        self.load_prompt(prompt);
    }

    fn load_prompt(&mut self, prompt: &[u32]) {
        self.ctx.reserve(prompt.len());
        for m in &mut self.index {
            m.reserve(prompt.len());
        }
        if self.cfg.has_diag() {
            self.diag.reserve(prompt.len());
        }
        self.push_tokens(prompt);
    }

    pub fn cfg(&self) -> CopySpanCfg {
        self.cfg
    }
    pub fn ctx(&self) -> &[u32] {
        &self.ctx
    }
    pub fn ctx_len(&self) -> usize {
        self.ctx.len()
    }
    /// The live copy cursor (next source position), if any. A cursor into a record copy's source carries `EXT_TAG`.
    pub fn cursor(&self) -> Option<usize> {
        self.cursor.map(|c| if self.cursor_ext { c | EXT_TAG } else { c })
    }
    /// COPYREC1: the live cursor follows a record copy.
    pub fn cursor_in_record(&self) -> bool {
        self.cursor.is_some() && self.cursor_ext
    }
    /// The source the live cursor indexes (the context, or the record copy's source).
    fn src_buf(&self) -> &[u32] {
        if self.cursor_ext { &self.ext } else { &self.ctx }
    }
    /// Why the last `propose` returned nothing (`None` after a proposal).
    pub fn last_miss(&self) -> Option<MissReason> {
        self.last_miss
    }
    /// How the last copy ended (`None` while the cursor is live or before any copy).
    pub fn last_end(&self) -> Option<CopyEnd> {
        self.last_end
    }
    pub fn counters(&self) -> &CopySpanCounters {
        &self.counters
    }
    /// The yield gate's state (kept up to date whether or not the gate is armed).
    pub fn gate(&self) -> &YieldGate {
        &self.gate
    }
    /// The serve loop reports one NEURAL speculative round of this lane (`emitted` = the tokens the
    /// round emitted, agreed across TP ranks). Plain steps and control rounds are not reported; copy
    /// rounds are counted by `commit` with their proposal. One call site per round in the serve loop.
    pub fn note_neural_round(&mut self, emitted: usize) {
        self.gate.note_neural(emitted);
    }
    /// Forget the cursor (the next `propose` searches). Not counted as a copy end.
    pub fn drop_cursor(&mut self) {
        self.cursor = None;
        self.cursor_ext = false;
    }

    fn push_tokens(&mut self, toks: &[u32]) {
        self.ctx.reserve(toks.len());
        for &t in toks {
            self.ctx.push(t);
            let p = self.ctx.len() - 1;
            if p >= 1 {
                self.index_completions(p);
            }
        }
    }

    /// Index every n-gram completed by the token at `p` (n-grams ending at `p-1`, continuation
    /// `ctx[p]`): one reversed-hash pass of `min(p, max_n)` steps.
    fn index_completions(&mut self, p: usize) {
        let CopySpan { cfg, ctx, index, diag, .. } = self;
        let diag_n = if cfg.has_diag() { cfg.diag_n } else { 0 };
        let top = cfg.max_n.min(p);
        let mut h = HSEED;
        for l in 1..=top {
            h = hstep(h, ctx[p - l]);
            if l >= cfg.min_n {
                index[l - cfg.min_n].insert(hfin(h), (p - l) as u32); // most recent wins
            } else if l == diag_n {
                diag.insert(hfin(h), (p - l) as u32);
            }
        }
    }

    /// Verified earlier occurrence of the suffix of length `l` (key `key`) in `map`.
    #[inline]
    fn find_in(&self, map: &KeyMap, l: usize, key: u64) -> Option<usize> {
        let n = self.ctx.len();
        let pos = *map.get(&key)? as usize;
        if pos + l >= n {
            return None; // defensive: never a self/overlapping occurrence
        }
        if self.ctx[pos..pos + l] != self.ctx[n - l..n] {
            return None; // hash collision: suppress, never fabricate
        }
        Some(pos)
    }

    /// Propose up to `max_w` tokens: the live cursor's source, else a gated index search. The start
    /// ramp narrows it further: a search proposal carries at most `cfg.start_w` tokens, a cursor
    /// proposal at most the ramp cap its copy reached (doubling per continued round).
    /// `None` = no proposal (`last_miss` says why; `max_w == 0` proposes nothing and sets none).
    pub fn propose(&mut self, max_w: usize) -> Option<Proposal> {
        self.last_miss = None;
        if max_w == 0 {
            return None;
        }
        let n = self.ctx.len();
        if let Some(cur) = self.cursor {
            let (src, ext) = (self.src_buf(), self.cursor_ext);
            if cur < src.len() {
                let ramp = self.ramp_next;
                let tokens = src[cur..(cur + max_w.min(ramp)).min(src.len())].to_vec();
                self.counters.proposals_cursor += 1;
                self.counters.proposed_tokens += tokens.len() as u64;
                self.counters.narrow_cursor += (ramp < max_w) as u64;
                return Some(Proposal { tokens, src: cur, source: Source::Cursor, max_w, ctx_len: n, ramp, ext });
            }
            self.drop_cursor(); // cannot happen through commit (cur < source length when it was kept)
        }
        self.counters.searches += 1;
        let cfg = self.cfg;
        let lo = if cfg.has_diag() { cfg.diag_n } else { cfg.min_n };
        if n < lo + 1 {
            self.miss(MissReason::NoMatch);
            return None;
        }
        // keys of every suffix length up to the longest that leaves room for an earlier start
        let top = cfg.max_n.min(n - 1);
        let mut h = HSEED;
        for l in 1..=top {
            h = hstep(h, self.ctx[n - l]);
            self.hs[l - 1] = hfin(h);
        }
        if top >= cfg.min_n {
            for l in (cfg.min_n..=top).rev() {
                if let Some(pos) = self.find_in(&self.index[l - cfg.min_n], l, self.hs[l - 1]) {
                    // the yield gate: only a NEW copy (a search hit) is gated, never the cursor
                    if cfg.yield_w > 0 {
                        let d = if cfg.yield_ratio > 0 { self.gate.decide_ratio(cfg.yield_ratio) } else { self.gate.decide(cfg.yield_w) };
                        match d {
                            GateDecision::Allow => {}
                            GateDecision::Probe => self.counters.gate_probes += 1,
                            GateDecision::Deny => {
                                self.miss(MissReason::YieldGate);
                                return None;
                            }
                        }
                    }
                    let src = pos + l; // < n by construction: the continuation is never empty
                    let ramp = cfg.start_w; // a NEW copy starts at the ramp's first width
                    let tokens = self.ctx[src..(src + max_w.min(ramp)).min(n)].to_vec();
                    self.counters.proposals_search += 1;
                    self.counters.proposed_tokens += tokens.len() as u64;
                    self.counters.narrow_search += (ramp < max_w) as u64;
                    return Some(Proposal { tokens, src, source: Source::Search { match_len: l }, max_w, ctx_len: n, ramp, ext: false });
                }
            }
        }
        if cfg.has_diag() && top >= cfg.diag_n && self.find_in(&self.diag, cfg.diag_n, self.hs[cfg.diag_n - 1]).is_some() {
            self.miss(MissReason::BelowMin);
        } else {
            self.miss(MissReason::NoMatch);
        }
        None
    }

    /// COPYLANES (`--copy-span-lanes`): what `propose(max_w)` WOULD return, without changing anything (no counter, no
    /// yield-gate run, no cursor move): Some((tokens it would carry, true = the live cursor / false = a search hit)).
    /// The record (`propose_with`) is not asked. The yield gate is applied as `decide` would apply it on its next call
    /// (its rule, or the probe that the next denied start would be), so a `propose` right after agrees with it.
    pub fn peek(&self, max_w: usize) -> Option<(usize, bool)> {
        if max_w == 0 {
            return None;
        }
        let n = self.ctx.len();
        if let Some(cur) = self.cursor {
            let src = self.src_buf();
            if cur < src.len() {
                return Some(((cur + max_w.min(self.ramp_next)).min(src.len()) - cur, true));
            }
            return None; // propose would drop it and search; a lane in this state is not offered (rare, never by commit)
        }
        let cfg = self.cfg;
        let lo = if cfg.has_diag() { cfg.diag_n } else { cfg.min_n };
        if n < lo + 1 {
            return None;
        }
        let top = cfg.max_n.min(n - 1);
        if top < cfg.min_n {
            return None;
        }
        let mut hs = vec![0u64; top];
        let mut h = HSEED;
        for l in 1..=top {
            h = hstep(h, self.ctx[n - l]);
            hs[l - 1] = hfin(h);
        }
        for l in (cfg.min_n..=top).rev() {
            if let Some(pos) = self.find_in(&self.index[l - cfg.min_n], l, hs[l - 1]) {
                // ULTRA2 merge: the same rule `propose` applies (COPY27B's cost-ratio rule when `yield_ratio` is set;
                // 0 on every EXL3 arm, so the EXL3 peek is unchanged)
                let allows = if cfg.yield_ratio > 0 { self.gate.ratio_allows(cfg.yield_ratio) } else { self.gate.rule_allows(cfg.yield_w) };
                if cfg.yield_w > 0 && !allows && (self.gate.denied_run + 1) % YG_PROBE_EVERY != 0 {
                    return None;
                }
                let src = pos + l;
                return Some(((src + max_w.min(cfg.start_w)).min(n) - src, false));
            }
        }
        None
    }

    /// COPYLANES (`--copy-span-lanes`): commit tokens emitted by rounds that used no proposal (a packed or shared step
    /// of a busy server), keeping a live copy alive when those tokens are exactly the source's next tokens: the cursor
    /// then moves past them (the copy continues at its current ramp) and the copy's run grows by them. Anything else is
    /// `commit(None, 0, emitted)` exactly (a live cursor ends `Unused`). Deterministic (replicated on every TP rank).
    pub fn commit_follow(&mut self, emitted: &[u32]) {
        if emitted.is_empty() {
            return;
        }
        if let Some(cur) = self.cursor {
            let src = self.src_buf();
            let end = cur + emitted.len();
            if end < src.len() && src[cur..end] == *emitted {
                self.cursor = Some(end);
                self.run += emitted.len();
                self.counters.rounds_without_proposal += 1;
                self.counters.committed_tokens += emitted.len() as u64;
                self.last_end = None;
                self.push_tokens(emitted);
                return;
            }
        }
        self.commit(None, 0, emitted);
    }

    /// COPYPF1 (`--copy-span-prefill`): a CURSOR-ONLY proposal of up to `max_w` tokens that ignores
    /// the start ramp (a stretch runs only on an established copy; the ramp exists to make a copy's
    /// FIRST rounds cheap). None without a live cursor (no search runs). Counted as a cursor
    /// proposal; `commit` treats it like any proposal: a fully accepted stretch whose model token
    /// continues the source keeps the cursor, anything else drops it. Its `ramp` is unbounded.
    pub fn propose_span(&mut self, max_w: usize) -> Option<Proposal> {
        self.last_miss = None;
        let n = self.ctx.len();
        let cur = self.cursor?;
        let (src, ext) = (self.src_buf(), self.cursor_ext);
        if max_w == 0 || cur >= src.len() {
            return None;
        }
        let tokens = src[cur..(cur + max_w).min(src.len())].to_vec();
        self.counters.proposals_cursor += 1;
        self.counters.proposed_tokens += tokens.len() as u64;
        Some(Proposal { tokens, src: cur, source: Source::Cursor, max_w, ctx_len: n, ramp: usize::MAX, ext })
    }

    /// COPYREC1 (`--copy-span-record`): `propose`, and when the context has no proposal (no match, or only one below
    /// `min_n`; a copy the yield gate held back is NOT replaced), ask the server's record for the longest match of the
    /// context's tail (at least `min_n` tokens). A record copy then runs exactly like a context copy: its first round
    /// proposes at most `start_w` tokens of the record's continuation (taken whole, up to `SNAP_MAX` ids, when the copy
    /// starts), later rounds follow the cursor through it (`propose` / `propose_span`), and `commit` keeps or drops the
    /// cursor by the same rule. The outcomes of earlier record rounds are applied to the record's landing rates first.
    /// `rec = None` is exactly `propose`. Deterministic: the record and every input are identical on all TP ranks.
    pub fn propose_with(&mut self, max_w: usize, rec: Option<&mut crate::copy_record::CopyRecord>) -> Option<Proposal> {
        let Some(rec) = rec else { return self.propose(max_w) };
        self.flush_record(rec);
        self.rec.rounds += 1;
        let p = self.propose(max_w);
        if p.is_some() || max_w == 0 || !matches!(self.last_miss, Some(MissReason::NoMatch) | Some(MissReason::BelowMin)) {
            return p;
        }
        // the bar: tokens a round of this request's own drafts emitted lately (2 before the first one)
        let head_q16 = if self.gate.nn == 0 { 2 << 16 } else { self.gate.en << 8 };
        let w = max_w.min(self.cfg.start_w);
        let off = rec.offer(&self.ctx, w, head_q16, self.cfg.min_n, &mut self.rec)?;
        let n = self.ctx.len();
        let tokens = off.snap[..w.min(off.snap.len())].to_vec();
        rec.count_proposal(off.band, tokens.len());
        self.ext = off.snap;
        self.ext_band = off.band;
        self.ext_probing = off.probing;
        self.last_miss = None;
        self.counters.proposed_tokens += tokens.len() as u64;
        self.counters.narrow_search += (self.cfg.start_w < max_w) as u64;
        Some(Proposal { tokens, src: 0, source: Source::Record { match_len: off.match_len }, max_w, ctx_len: n,
                        ramp: self.cfg.start_w, ext: true })
    }

    /// COPYREC1: apply this request's record-copy outcomes not yet seen by the record (its next ask does it; the lane
    /// release calls it for the last ones).
    pub fn flush_record(&mut self, rec: &mut crate::copy_record::CopyRecord) {
        if !self.rec.pending.is_empty() {
            rec.observe(&self.rec.pending);
            self.rec.pending.clear();
        }
    }

    /// COPYPF1 gate session: tokens the live copy has produced in agreement with its source so far (0 without a live
    /// cursor). `--copy-span-prefill-min-run` reads it.
    pub fn cursor_run(&self) -> usize {
        if self.cursor.is_some() { self.run } else { 0 }
    }

    /// ECHOPRICE: the running mean length of this request's established copies that have ended (0.0 = none yet).
    pub fn copy_len_hist(&self) -> f64 { self.len_hist }

    /// COPYPF1: source tokens left after the live cursor (0 without one).
    pub fn cursor_left(&self) -> usize {
        self.cursor.map_or(0, |c| self.src_buf().len().saturating_sub(c))
    }

    fn miss(&mut self, r: MissReason) {
        self.last_miss = Some(r);
        match r {
            MissReason::NoMatch => self.counters.miss_no_match += 1,
            MissReason::BelowMin => self.counters.miss_below_min += 1,
            MissReason::YieldGate => self.counters.gate_denied += 1,
        }
    }

    fn end(&mut self, e: CopyEnd) {
        self.last_end = Some(e);
        match e {
            CopyEnd::Mismatch => self.counters.end_mismatch += 1,
            CopyEnd::EndOfContext => self.counters.end_of_context += 1,
            CopyEnd::ShortEmit => self.counters.end_short_emit += 1,
            CopyEnd::Unused => self.counters.end_unused += 1,
        }
    }

    /// Commit one round. `proposal_used` = the proposal the round verified (None: the round used
    /// another draft source or none), `accepted` = its leading tokens the model accepted,
    /// `emitted` = every token the round committed (the accepted ones, then the model's own).
    ///
    /// Cursor rule (exactly `sim(cursor_mode=True)`): with `m` accepted and the model token
    /// `g = emitted[m]`, the cursor stays live at `src + m + 1` only if `m > 0` and
    /// `ctx[src + m] == g` (checked against the context BEFORE this round's tokens are appended);
    /// otherwise it is dropped. A round without a proposal drops it too. The emitted tokens are
    /// always appended and indexed. Start ramp: a surviving cursor's next proposal may carry twice
    /// the ramp cap of the proposal just committed (`p.ramp * 2`, saturating).
    pub fn commit(&mut self, proposal_used: Option<&Proposal>, accepted: usize, emitted: &[u32]) {
        let len_before = self.ctx.len();
        let mut next = None;
        let mut next_ext = false;
        let mut run = 0usize;
        // ECHOPRICE: the live copy's length when this round ends it (its run + this round's accepted tokens)
        let mut ended_len: Option<usize> = None;
        let src_len = |p: &Proposal, s: &Self| if p.ext { s.ext.len() } else { len_before };
        match proposal_used {
            Some(p) if p.ctx_len == len_before && p.src + p.tokens.len() <= src_len(p, self) => {
                // the accepted prefix must really equal what was emitted (clamp a caller slip)
                let claim = accepted.min(p.tokens.len()).min(emitted.len());
                let m = p.tokens[..claim].iter().zip(emitted).take_while(|(a, b)| a == b).count();
                if m != accepted {
                    self.counters.inconsistent_accept += 1;
                }
                let c = &mut self.counters;
                c.rounds_with_proposal += 1;
                c.accepted_tokens += m as u64;
                c.emitted_with_proposal += emitted.len() as u64;
                c.rounds_zero_accepted += (m == 0) as u64;
                c.rounds_full += (m == p.tokens.len() && m == p.max_w) as u64;
                self.gate.note_copy(emitted.len());
                if p.ext {
                    // COPYREC1: a record copy round; its outcome reaches the server-wide landing rates at the lane's
                    // next ask (or its release), in round order
                    let first = matches!(p.source, Source::Record { .. });
                    self.rec.pending.push(crate::copy_record::Obs { band: self.ext_band, proposed: p.tokens.len(), accepted: m,
                                                                    probing: first && self.ext_probing });
                    if m == 0 { self.rec.quiet_until = self.rec.rounds + 1 + crate::copy_record::QUIET_ROUNDS; }
                }
                let nxt = p.src + m;
                if p.source == Source::Cursor && self.cursor.is_some() { ended_len = Some(self.run + m); }
                let buf: &[u32] = if p.ext { &self.ext } else { &self.ctx };
                if emitted.len() <= m {
                    self.end(CopyEnd::ShortEmit);
                } else if m == 0 {
                    self.end(CopyEnd::Mismatch);
                } else if nxt >= buf.len() {
                    self.end(CopyEnd::EndOfContext);
                } else if buf[nxt] != emitted[m] {
                    self.end(CopyEnd::Mismatch);
                } else {
                    next = Some(nxt + 1);
                    self.ramp_next = p.ramp.saturating_mul(2);
                    // a cursor proposal continues the live copy's run; a search proposal starts a new one
                    run = if p.source == Source::Cursor { self.run } else { 0 } + m + 1;
                    next_ext = p.ext;
                }
            }
            Some(_) => {
                // a proposal made for another context state: treat as not used
                self.counters.rounds_without_proposal += 1;
                if self.cursor.is_some() {
                    self.end(CopyEnd::Unused);
                }
            }
            None => {
                self.counters.rounds_without_proposal += 1;
                if self.cursor.is_some() {
                    self.end(CopyEnd::Unused);
                }
            }
        }
        if next.is_some() {
            self.last_end = None;
        } else if let Some(l) = ended_len {
            // an established copy ended in this round: fold its length into the request's mean
            self.len_hist = if self.len_hist > 0.0 { 0.5 * (self.len_hist + l as f64) } else { l as f64 };
        }
        self.cursor = next;
        self.cursor_ext = next_ext;
        self.run = run;
        self.counters.committed_tokens += emitted.len() as u64;
        self.push_tokens(emitted);
    }
}

// ---------------------------------------------------------------------------------------------
// Replay (the probe and the tests): walk a realised output round by round, exactly like `sim()`.
// ---------------------------------------------------------------------------------------------

/// Per-record replay totals, the `sim()` fields: `hits` = rounds with a proposal, `tok_hit` =
/// tokens those rounds emitted, `m0` = of them with zero accepted, `full` = all `max_w` accepted.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayStats {
    pub n: u64,
    pub hits: u64,
    pub tok_hit: u64,
    pub m0: u64,
    pub full: u64,
}

impl ReplayStats {
    pub fn add(&mut self, o: &ReplayStats) {
        self.n += o.n;
        self.hits += o.hits;
        self.tok_hit += o.tok_hit;
        self.m0 += o.m0;
        self.full += o.full;
    }
    /// Tokens per round with a proposal.
    pub fn per_hit(&self) -> f64 {
        if self.hits == 0 { 0.0 } else { self.tok_hit as f64 / self.hits as f64 }
    }
}

/// Replay `gen` after `prompt`: a round with a proposal emits `min(m + 1, remaining)` tokens (m =
/// leading proposed tokens equal to `gen`), a round without one emits 1 token. `follow_cursor =
/// false` drops the cursor after every round (the parked always-search rule; tests only).
pub fn replay(prompt: &[u32], gen: &[u32], cfg: CopySpanCfg, max_w: usize, follow_cursor: bool) -> (ReplayStats, CopySpanCounters) {
    let mut cs = CopySpan::new(prompt, cfg);
    let st = replay_on(&mut cs, gen, max_w, follow_cursor);
    (st, *cs.counters())
}

fn replay_on(cs: &mut CopySpan, gen: &[u32], max_w: usize, follow_cursor: bool) -> ReplayStats {
    let big_n = gen.len();
    let mut st = ReplayStats { n: big_n as u64, ..Default::default() };
    let mut pos = 0usize;
    while pos < big_n {
        if !follow_cursor {
            cs.drop_cursor();
        }
        match cs.propose(max_w) {
            Some(p) => {
                let m = p.tokens.iter().zip(&gen[pos..]).take_while(|(a, b)| a == b).count();
                let e = (m + 1).min(big_n - pos);
                st.hits += 1;
                st.tok_hit += e as u64;
                st.m0 += (m == 0) as u64;
                st.full += (m == p.tokens.len() && m == max_w) as u64;
                cs.commit(Some(&p), m, &gen[pos..pos + e]);
                pos += e;
            }
            None => {
                cs.commit(None, 0, &gen[pos..pos + 1]);
                pos += 1;
            }
        }
    }
    st
}

fn json_ids(v: &serde_json::Value, what: &str, name: &str) -> anyhow::Result<Vec<u32>> {
    let a = v.as_array().ok_or_else(|| anyhow::anyhow!("record {name}: {what} is not an array"))?;
    a.iter()
        .map(|x| {
            x.as_u64()
                .filter(|&t| t <= u32::MAX as u64)
                .map(|t| t as u32)
                .ok_or_else(|| anyhow::anyhow!("record {name}: {what} holds a non-token value {x}"))
        })
        .collect()
}

/// `--probe-copy-span-replay --ids-file <json> [--copy-span-min-n N] [--copy-span-max-w W]
/// [--copy-span-start-w S]`:
/// replays each `{"name", "prompt_ids", "gen_ids"}` record and prints one `COPY_SPAN_REPLAY` line
/// per record and a `COPY_SPAN_REPLAY_TOTAL` line (the format `wvtools/phase0/dump_ids.py
/// --print-sim` prints for the Python reference, so the two can be diffed). No model, no GPU.
/// `start_w` = the start ramp (`--copy-span-start-w`; >= `max_w` = full width, the pre-ramp replay).
pub fn probe_replay(ids_file: &str, min_n: usize, max_w: usize, start_w: usize, out: &mut dyn Write) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(ids_file).map_err(|e| anyhow::anyhow!("--ids-file {ids_file}: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("--ids-file {ids_file}: {e}"))?;
    let recs = v.as_array().ok_or_else(|| anyhow::anyhow!("--ids-file {ids_file}: expected a JSON list of records"))?;
    let cfg = CopySpanCfg::new(min_n).with_start_w(start_w);
    writeln!(out, "COPY_SPAN_REPLAY_CFG records={} min_n={} max_n={} diag_n={} max_w={} cursor=on start_w={}{}",
             recs.len(), cfg.min_n, cfg.max_n, cfg.diag_n, max_w, start_w,
             if start_w >= max_w { " (full width: no ramp)" } else { "" })?;
    let mut tot = ReplayStats::default();
    let mut ctot = CopySpanCounters::default();
    let (mut t_build, mut t_replay, mut rounds) = (0f64, 0f64, 0u64);
    for r in recs {
        let name = r.get("name").and_then(|x| x.as_str()).unwrap_or("?").to_string();
        let prompt = json_ids(r.get("prompt_ids").unwrap_or(&serde_json::Value::Null), "prompt_ids", &name)?;
        let gen = json_ids(r.get("gen_ids").unwrap_or(&serde_json::Value::Null), "gen_ids", &name)?;
        let t0 = std::time::Instant::now();
        let mut cs = CopySpan::new(&prompt, cfg);
        let t1 = std::time::Instant::now();
        let st = replay_on(&mut cs, &gen, max_w, true);
        let t2 = std::time::Instant::now();
        t_build += (t1 - t0).as_secs_f64();
        t_replay += (t2 - t1).as_secs_f64();
        let c = *cs.counters();
        rounds += c.rounds_with_proposal + c.rounds_without_proposal;
        writeln!(out, "COPY_SPAN_REPLAY name={name} N={} hits={} tok_hit={} m0={} full={}", st.n, st.hits, st.tok_hit, st.m0, st.full)?;
        tot.add(&st);
        add_counters(&mut ctot, &c);
    }
    writeln!(out, "COPY_SPAN_REPLAY_TOTAL N={} hits={} tok_hit={} m0={} full={}", tot.n, tot.hits, tot.tok_hit, tot.m0, tot.full)?;
    let c = ctot;
    writeln!(out, "COPY_SPAN_COUNTERS searches={} proposals_search={} proposals_cursor={} miss_no_match={} miss_below_min={} \
                   end_mismatch={} end_of_context={} end_short_emit={} end_unused={} rounds_zero_accepted={} rounds_full={} inconsistent_accept={} \
                   proposed_tokens={} accepted_tokens={} narrow_search={} narrow_cursor={}",
             c.searches, c.proposals_search, c.proposals_cursor, c.miss_no_match, c.miss_below_min,
             c.end_mismatch, c.end_of_context, c.end_short_emit, c.end_unused, c.rounds_zero_accepted, c.rounds_full, c.inconsistent_accept,
             c.proposed_tokens, c.accepted_tokens, c.narrow_search, c.narrow_cursor)?;
    writeln!(out, "COPY_SPAN_REPLAY_TIME index_build_ms={:.2} replay_ms={:.2} rounds={} us_per_round={:.3} (host, this machine; propose + commit + index)",
             t_build * 1e3, t_replay * 1e3, rounds, if rounds > 0 { t_replay * 1e6 / rounds as f64 } else { 0.0 })?;
    Ok(())
}

fn add_counters(a: &mut CopySpanCounters, b: &CopySpanCounters) {
    a.searches += b.searches;
    a.proposals_search += b.proposals_search;
    a.proposals_cursor += b.proposals_cursor;
    a.proposed_tokens += b.proposed_tokens;
    a.miss_no_match += b.miss_no_match;
    a.miss_below_min += b.miss_below_min;
    a.rounds_with_proposal += b.rounds_with_proposal;
    a.accepted_tokens += b.accepted_tokens;
    a.emitted_with_proposal += b.emitted_with_proposal;
    a.rounds_zero_accepted += b.rounds_zero_accepted;
    a.rounds_full += b.rounds_full;
    a.rounds_without_proposal += b.rounds_without_proposal;
    a.committed_tokens += b.committed_tokens;
    a.end_mismatch += b.end_mismatch;
    a.end_of_context += b.end_of_context;
    a.end_short_emit += b.end_short_emit;
    a.end_unused += b.end_unused;
    a.inconsistent_accept += b.inconsistent_accept;
    a.gate_denied += b.gate_denied;
    a.gate_probes += b.gate_probes;
    a.narrow_search += b.narrow_search;
    a.narrow_cursor += b.narrow_cursor;
}

/// The `--probe-copy-span-replay` mode (main dispatches here before any model or GPU work).
/// Returns the process exit code: 0 = replayed, 1 = bad input, 2 = bad options.
pub fn probe_replay_cli() -> i32 {
    use crate::opts;
    let mut err = std::io::stderr();
    let Some(ids) = opts::get(crate::opt!("ids-file")) else {
        let _ = writeln!(err, "FATAL: --probe-copy-span-replay needs --ids-file <json> (a list of {{name, prompt_ids, gen_ids}})");
        return 2;
    };
    let min_n = match opts::ranged(crate::opt!("copy-span-min-n")) {
        Ok(v) => v as usize,
        Err(e) => { let _ = writeln!(err, "FATAL: {e}"); return 2; }
    };
    let max_w = match opts::ranged(crate::opt!("copy-span-max-w")) {
        Ok(v) => v as usize,
        Err(e) => { let _ = writeln!(err, "FATAL: {e}"); return 2; }
    };
    let start_w = match opts::ranged(crate::opt!("copy-span-start-w")) {
        Ok(v) => v as usize,
        Err(e) => { let _ = writeln!(err, "FATAL: {e}"); return 2; }
    };
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    match probe_replay(&ids, min_n, max_w, start_w, &mut lock) {
        Ok(()) => 0,
        Err(e) => { let _ = writeln!(err, "COPY_SPAN_REPLAY_FAIL {e:#}"); 1 }
    }
}

#[cfg(test)]
mod tests;
