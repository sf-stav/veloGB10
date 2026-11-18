//! Shadow comparison: the served reading is re-parsed under the other rules and the differences counted, never served.
//! Bounded by size, block count and wall-clock; a panic in it is caught and changes nothing served. Map in `mod.rs`.

use super::*;

/// A panic inside the shadow comparison is caught, but the default panic hook would still print its
/// message first, and a slice or index panic quotes a piece of the text it was reading (independent
/// review, nit). One process-wide hook, installed once: while this thread is shadowing it prints
/// nothing; any other panic goes to the hook that was there before.
pub(super) fn install_shadow_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !shadow_parse() { prev(info); }
        }));
    });
}

/// Largest text the shadow comparison reads, and the most blocks (a degenerate output: a model that
/// repeats an empty block until the token budget; a prompt-injected document).
pub const SHADOW_MAX_BYTES: usize = 512 * 1024;
pub const SHADOW_MAX_BLOCKS: usize = 64;
/// Wall-clock the comparison may spend before it skips the rules it has not read yet.
const SHADOW_BUDGET: std::time::Duration = std::time::Duration::from_millis(100);

pub(super) fn shadow_limit(text: &str) -> Option<&'static str> {
    if text.len() > SHADOW_MAX_BYTES { return Some("size"); }
    if text.matches(CALL_OPEN).take(SHADOW_MAX_BLOCKS + 1).count() > SHADOW_MAX_BLOCKS { return Some("blocks"); }
    None
}

/// The reading a served parse is compared with (the label `other` of the shadow counters).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Other { Off, On, Cost, BlockEnd }

impl Other {
    pub const ALL: [Other; 4] = [Other::Off, Other::On, Other::Cost, Other::BlockEnd];
    pub fn idx(self) -> usize { self as usize }
    pub fn as_str(self) -> &'static str {
        match self { Other::Off => "off", Other::On => "on", Other::Cost => "cost", Other::BlockEnd => "block_end" }
    }
}

/// How two readings of one block differ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DiffKind {
    /// they return a different NUMBER of calls
    CallCount,
    /// the same number of calls, an argument (or a name) differs
    ArgBoundary,
    /// the same calls, the prose around them differs
    Content,
}

impl DiffKind {
    pub const ALL: [DiffKind; 3] = [DiffKind::CallCount, DiffKind::ArgBoundary, DiffKind::Content];
    pub fn idx(self) -> usize { self as usize }
    pub fn as_str(self) -> &'static str {
        match self { DiffKind::CallCount => "call_count", DiffKind::ArgBoundary => "arg_boundary", DiffKind::Content => "content" }
    }
}

/// Which of the two readings returns FEWER calls (a fact; which one is right is not known at serve time).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fewer { Served, Other, Same }

impl Fewer {
    pub const ALL: [Fewer; 3] = [Fewer::Served, Fewer::Other, Fewer::Same];
    pub fn idx(self) -> usize { self as usize }
    pub fn as_str(self) -> &'static str {
        match self { Fewer::Served => "served", Fewer::Other => "other", Fewer::Same => "same" }
    }
}

/// One disagreement between the served rules and another reading of the same text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowDiff {
    pub other: Other,
    pub kind: DiffKind,
    pub fewer: Fewer,
    /// first 8 bytes (hex) of the sha256 of the block body, so that an operator can correlate two
    /// lines without the log carrying any response text
    pub hash: String,
    pub len: usize,
    pub calls_served: usize,
    pub calls_other: usize,
}

/// What the shadow comparison of one response found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Shadow {
    /// the comparison was cut short or not made: `size`, `blocks` (see `shadow_limit`) or `time`
    pub skipped: Option<&'static str>,
    /// blocks the served rules parsed
    pub blocks: usize,
    /// per `Other`: blocks compared against it (`BlockEnd`: responses compared)
    pub compared: [usize; 4],
    /// blocks the cost model could not read within its work budget while it was the OTHER rule:
    /// they are not in `compared[Cost]` (a block read by the v0.7.3 rule says nothing about cost)
    pub cost_declined: usize,
    pub diffs: Vec<ShadowDiff>,
}

pub(super) struct ShadowGuard(bool);

impl ShadowGuard {
    pub(super) fn enter() -> ShadowGuard { ShadowGuard(SHADOW_PARSE.with(|c| c.replace(true))) }
}

impl Drop for ShadowGuard {
    fn drop(&mut self) { SHADOW_PARSE.with(|c| c.set(self.0)); }
}

fn block_hash(body: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(body.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Read `text` under the served rules and under every other rule the parser has, and report where
/// they disagree (S-B9-TOOLPARSE3 T1). Per block: the two value rules that are not served, against
/// the served one, block by block (the block segmentation is the same). Per response: the block
/// rule that is not served, against the served one (its blocks can differ, so the calls of the
/// whole response are compared). No id is minted and nothing is printed while it runs; cost is one
/// extra linear parse per rule (the cost model: at most its work budget), only for a text that holds
/// a `<tool_call>` block. This is the instrument form: ALL the other rules, the cost model included.
/// The server's entry (`parse_served`) reads the cost model only under `--tool-shadow-cost on`.
pub fn shadow_compare(text: &str, tools: Option<&[Value]>, served: &Rules, finish: Option<&str>) -> Shadow {
    let mut sh = Shadow::default();
    if !text.contains(CALL_OPEN) || is_hy3(text) { return sh; }
    if let Some(r) = shadow_limit(text) { sh.skipped = Some(r); return sh; }
    let _guard = ShadowGuard::enter();
    let mut base = Recs::new(true);
    let served_out = parse_core(text, tools, served, finish, Some(&mut base));
    drop(_guard);
    shadow_from(text, tools, served, finish, &served_out, &base.v, true)
}

/// The comparison proper, given the served parse (its output and its per-block records) so that the
/// served rules are not read twice. `with_cost`: also read the cost model when it is not served.
pub(super) fn shadow_from(text: &str, tools: Option<&[Value]>, served: &Rules, finish: Option<&str>,
               served_out: &ParsedOutput, base: &[BlockRec], with_cost: bool) -> Shadow {
    let mut sh = Shadow::default();
    sh.blocks = base.len();
    if base.is_empty() { return sh; }
    let _guard = ShadowGuard::enter();
    #[cfg(test)]
    if SHADOW_FAULT.with(|c| c.get()) {
        // a fault drill: the panic message quotes the text, as a slice or index panic of a rule would
        panic!("injected shadow fault while reading {:?}", &text[..text.len().min(40)]);
    }
    let t0 = std::time::Instant::now();
    let fewer = |s: usize, o: usize| if s < o { Fewer::Served } else if o < s { Fewer::Other } else { Fewer::Same };
    for (other, pairing) in [(Other::Off, QuotePairing::Off), (Other::On, QuotePairing::On), (Other::Cost, QuotePairing::Cost)] {
        if pairing == served.pairing { continue; }
        if pairing == QuotePairing::Cost && !with_cost { continue; }
        if t0.elapsed() > SHADOW_BUDGET { sh.skipped = Some("time"); return sh; }
        let mut recs = Recs::new(true);
        let _ = parse_core(text, tools, &Rules { pairing, ..*served }, finish, Some(&mut recs));
        for (b, o) in base.iter().zip(recs.v.iter()) {
            if o.declined { sh.cost_declined += 1; continue; }
            sh.compared[other.idx()] += 1;
            if b.calls == o.calls { continue; }
            let kind = if b.calls.len() != o.calls.len() { DiffKind::CallCount } else { DiffKind::ArgBoundary };
            sh.diffs.push(ShadowDiff {
                other, kind, fewer: fewer(b.calls.len(), o.calls.len()),
                hash: block_hash(&text[b.span.0..b.span.1]), len: b.span.1 - b.span.0,
                calls_served: b.calls.len(), calls_other: o.calls.len(),
            });
        }
    }
    if t0.elapsed() > SHADOW_BUDGET { sh.skipped = Some("time"); return sh; }
    let other_block = if served.block == BlockEnd::First { BlockEnd::AfterFunction } else { BlockEnd::First };
    let alt = parse_core(text, tools, &Rules { block: other_block, ..*served }, finish, None);
    sh.compared[Other::BlockEnd.idx()] += 1;
    let sig = |o: &ParsedOutput| o.tool_calls.iter().map(|c| (c.function.name.clone(), c.function.arguments.clone())).collect::<Vec<_>>();
    let (a, b) = (sig(served_out), sig(&alt));
    let kind = if a.len() != b.len() { Some(DiffKind::CallCount) }
               else if a != b { Some(DiffKind::ArgBoundary) }
               else if served_out.content != alt.content { Some(DiffKind::Content) }
               else { None };
    if let Some(kind) = kind {
        let region = text.find(CALL_OPEN).map_or(text, |i| &text[i..]);
        sh.diffs.push(ShadowDiff {
            other: Other::BlockEnd, kind, fewer: fewer(a.len(), b.len()),
            hash: block_hash(region), len: region.len(), calls_served: a.len(), calls_other: b.len(),
        });
    }
    sh
}
