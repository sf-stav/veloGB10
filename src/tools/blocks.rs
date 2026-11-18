//! Block ends: where a `<tool_call>` body closes under each `BlockEnd` rule, with the `BlockIndex` used by the
//! after-function rule. Part of the tool-call parser; the map of the directory is in `mod.rs`.

use super::*;

#[cfg(test)]
pub(super) fn scan_ops(n: usize) { SCAN_OPS.with(|c| c.set(c.get() + n)); }
#[cfg(not(test))]
#[inline(always)]
pub(super) fn scan_ops(_n: usize) {}

#[cfg(test)]
thread_local! {
    /// (cost-model work units, blocks declined) of the last `parse_core` on this thread: what the
    /// bound tests read.
    pub(super) static COST_USED: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    /// How many times `parse_core` has run on this thread: the shadow tests count parses, not seconds.
    pub(super) static PARSE_CORE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Where the block whose body starts at byte `start` of `text` ends: (end of the body, start of what
/// follows), both relative to `start`, or `None` when the block never closed (the call is dropped: a
/// half call is worse than none).
pub(super) fn block_close(text: &str, start: usize, rule: BlockEnd, finish: Option<&str>, idx: Option<&BlockIndex>) -> Option<(usize, usize)> {
    match (rule, idx) {
        (BlockEnd::AfterFunction, Some(ix)) => after_function_close(ix, text, start, finish),
        _ => {
            let rest = &text[start..];
            let first = rest.find(CALL_CLOSE)?;
            if rest.trim_start().starts_with('{') {
                if let Some(end) = json_body_close(rest, first) { return Some((end, end + CALL_CLOSE.len())); }
            }
            Some((first, first + CALL_CLOSE.len()))
        }
    }
}

/// Does the text of a JSON-bodied block (one that starts with `{`) end the way a finished JSON call
/// ends: with `}`, white space aside? (The ONE stray `</parameter>` the model imitates from the JSON
/// history form is stripped by `parse_xml_calls_w` only inside a `<function=...>` body, never after a
/// bare JSON object, which then does not parse at all: no allowance for it here.)
fn json_body_ends(before: &str) -> bool {
    before.trim_end().ends_with('}')
}

/// Test instrument (the differential names the texts item 7 moves): does any JSON-bodied block of
/// `text`, read in the segmentation of the first-close rule up to that block, end somewhere else
/// than at its first `</tool_call>`?
#[cfg(test)]
pub(super) fn json_body_moves(text: &str) -> bool {
    let mut base = 0usize;
    while let Some(o) = text[base..].find(CALL_OPEN) {
        let start = base + o + CALL_OPEN.len();
        let rest = &text[start..];
        let Some(first) = rest.find(CALL_CLOSE) else { return false };
        if rest.trim_start().starts_with('{') {
            if let Some(end) = json_body_close(rest, first) { if end != first { return true; } }
        }
        base = start + first + CALL_CLOSE.len();
    }
    false
}

/// Where a JSON-bodied block ends (relative to `rest`, the text after its `<tool_call>`): the first
/// `</tool_call>` the body can end at. A `</tool_call>` inside a properly escaped JSON string (a call
/// that writes a file about the format) is text; the first close that follows `}` is the end. A close
/// that pays for a `<tool_call>` opened inside the span (the next real block's own opener) is never
/// taken, so two real blocks are never merged. `None` = no close qualifies: the caller keeps the
/// first close, as v0.7.3 did. (Independent review J02b: the cut body failed to parse as JSON and fell
/// through to the XML scanner, which returned the QUOTED call, `run_shell rm -rf ~`, and lost the
/// real one.) Linear: one pass over the closes and the opens.
fn json_body_close(rest: &str, first: usize) -> Option<usize> {
    let (mut at, mut owed, mut scanned) = (first, 0usize, 0usize);
    loop {
        // `<tool_call>` openers in rest[scanned..at]: each is owed one close, and a close that pays
        // such a debt is text (it is the close of a block the span itself opened)
        while let Some(rel) = rest[scanned..at].find(CALL_OPEN) { owed += 1; scanned += rel + CALL_OPEN.len(); }
        scanned = scanned.max(at);
        if owed > 0 { owed -= 1; }
        else if json_body_ends(&rest[..at]) { return Some(at); }
        let from = at + CALL_CLOSE.len();
        at = from + rest[from..].find(CALL_CLOSE)?;
    }
}

/// Positions (absolute byte offsets) of the block closes and the block opens of a whole text: one pass.
pub(super) struct BlockIndex {
    closes: Vec<usize>,
    opens: Vec<usize>,
    /// candidates all blocks of the text may still examine together (`BLOCK_WORK` per close, plus a
    /// floor): a block that finds its close early spends little; a hostile text spends the budget and
    /// the rest of its blocks end at their first close, as in v0.7.3
    budget: std::cell::Cell<usize>,
}

impl BlockIndex {
    pub(super) fn new(text: &str) -> BlockIndex {
        scan_ops(text.len());
        let (mut closes, mut opens) = (Vec::new(), Vec::new());
        for (i, _) in text.match_indices('<') {
            let r = &text[i..];
            if r.starts_with(CALL_CLOSE) { closes.push(i); }
            else if r.starts_with(CALL_OPEN) { opens.push(i); }
        }
        let budget = std::cell::Cell::new(BLOCK_WORK * closes.len() + 1024);
        BlockIndex { closes, opens, budget }
    }
}

/// Candidates examined per close of the text, over all blocks (see `BlockIndex::budget`).
const BLOCK_WORK: usize = 8;

/// Does the text of a block end the way a finished call ends: `</function>` or `</parameter>` (a JSON
/// body, one that starts with `{`: `}`), white space aside?
pub(super) fn ends_like_a_call(body: &str) -> bool {
    let json = body.trim_start().starts_with('{');
    let b = body.trim_end();
    if json { b.ends_with('}') } else { b.ends_with("</function>") || b.ends_with("</parameter>") }
}

/// The popular engines' block end (`--tool-block-end after-function`, the default; part B item 12, the
/// owner's ruling of 2026-10-10).
///
/// The format has no escaping, so a value can hold the text `</tool_call>`. vLLM's Qwen3 parser is a
/// state machine in which, inside the arguments, the only terminal is `</function>`: a `</tool_call>`
/// there is ordinary text, and `</tool_call>` is looked for only AFTER the function closed. This is
/// that rule on our parser (which keeps its own value rule, the lookahead since v0.7.3, and every
/// repair): a `</tool_call>` ends the block only when
///
///  1. the text before it, white space aside, ends with `</function>` or `</parameter>` (a JSON body,
///     one that starts with `{`: with `}`): the call has ended there (`</parameter>` is the repair for
///     a missing `</function>`); and
///  2. it does not close a `<tool_call>` the block itself opened (every `<tool_call>` inside the block
///     is owed one `</tool_call>`, and a close that pays such a debt is text): a QUOTED complete block
///     inside a value, which is what cut 5 of the 348 real outputs of SURFACE2B.
///
/// When no close qualifies the FIRST one ends the block, exactly as in v0.7.3: every candidate that
/// fails is a close the old rule would have taken, so this rule only ever moves a block's end LATER,
/// and only past a close that is plainly inside a value (it follows prose, or it pays a quoted
/// opener). There is no search across real blocks (the first version of the structural rule had a
/// second pass, and a rule that looked at what FOLLOWS a close; the independent review found both
/// merging two real calls): the next real block's own `<tool_call>` pays for its close, so a block
/// never extends into its successor.
///
/// A block with no close at all is dropped, as in v0.7.3, whatever the finish reason (see
/// `velogb10_toolcall_unclosed_total`).
///
/// Linear: the opens and the closes are collected in one pass and a candidate costs O(1); a shared
/// budget bounds the candidates a hostile text can make the rule examine.
fn after_function_close(ix: &BlockIndex, text: &str, start: usize, _finish: Option<&str>) -> Option<(usize, usize)> {
    let c0 = ix.closes.partition_point(|&p| p < start);
    // no close at all: dropped, as in v0.7.3
    let &first = ix.closes.get(c0)?;
    let spend = || { let b = ix.budget.get(); if b == 0 { false } else { ix.budget.set(b - 1); scan_ops(1); true } };
    let end_at = |j: usize| Some((j - start, j + CALL_CLOSE.len() - start));
    let (mut owed, mut oi) = (0usize, ix.opens.partition_point(|&p| p < start));
    for &j in &ix.closes[c0..] {
        if !spend() { return end_at(first); }
        while oi < ix.opens.len() && ix.opens[oi] < j { owed += 1; oi += 1; }
        if owed > 0 { owed -= 1; continue; }
        if ends_like_a_call(&text[start..j]) { return end_at(j); }
    }
    // pass 2, without the debts: an UNBALANCED mention of the open tag in a value (prose that says "a
    // <tool_call> block ...") must not make the real close owe a quoted block it never saw. Only a
    // close that follows the end of a call qualifies, as before.
    for &j in &ix.closes[c0..] {
        if !spend() { return end_at(first); }
        if ends_like_a_call(&text[start..j]) { return end_at(j); }
    }
    end_at(first)
}

/// Test instrument (the differential names the texts item 12 moves): is there a block of `text`, read
/// in the segmentation of the first-close rule up to that block, that ends somewhere else than at its
/// first `</tool_call>` under the default rule?
#[cfg(test)]
pub(super) fn after_function_moves(text: &str) -> bool {
    if is_hy3(text) { return false; }
    let ix = BlockIndex::new(text);
    let mut base = 0usize;
    while let Some(o) = text[base..].find(CALL_OPEN) {
        let start = base + o + CALL_OPEN.len();
        let Some(first) = text[start..].find(CALL_CLOSE) else { return false };
        match after_function_close(&ix, text, start, None) {
            Some((end, _)) if end != first => return true,
            _ => {}
        }
        base = start + first + CALL_CLOSE.len();
    }
    false
}
