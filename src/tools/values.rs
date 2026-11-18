//! Where a `<parameter=...>` value ends: the v0.7.3 rule, its cached form, and the opt-in quote-pairing scan
//! (`--tool-quote-pairing`). Default stays byte-identical to v0.7.3. Map of the directory in `mod.rs`.

use super::*;

/// Skip the whitespace that may sit between structural tags. Deliberately the SAME set
/// `find_function_tag_mode` accepts before a bare `function=` repair (newline, CR, space, tab) —
/// not arbitrary Unicode whitespace, which is content.
pub(super) fn skip_tag_ws(s: &str) -> &str {
    let mut i = 0;
    for b in s.bytes() {
        if matches!(b, b'\n' | b'\r' | b' ' | b'\t') { i += 1 } else { break }
    }
    &s[i..]
}

/// The structural-terminator lookahead (v0.7.3): does the text AFTER a candidate `</parameter>`
/// prove the candidate is the real close of the value?
///
/// The template's close is always followed (tag whitespace aside) by one of:
///  * `<parameter=` — the next parameter of the same call;
///  * `</function>` — the call close, itself followed by the end of the block body or a further
///    `<function=` / bare `function=` tag (another call in the same `<tool_call>` block);
///  * the next call's tag directly — `<function=` or a bare `function=` repair (K3: malformed
///    output can OMIT `</function>` between two calls; the same tag detection
///    `find_function_tag_mode` applies decides what counts as a tag);
///  * nothing — the end of the block body (today's tolerance for a missing `</function>`).
///
/// Anything else (e.g. `\n</parameter>…` — the reporter's file-about-tools payload embeds the
/// sequence `</parameter>\n</function>` MID-value) means the candidate is quoted content, not
/// the close.
pub(super) fn structural_continuation(after: &str) -> bool {
    let rest = skip_tag_ws(after);
    if rest.is_empty() || rest.starts_with("<parameter=") {
        return true;
    }
    // K3 (REL v0.7.3 review): the next call's tag DIRECTLY after the close (no
    // `</function>` between the calls) also proves the candidate is the real close.
    // Before this the close was treated as quoted content and the next call's
    // `<parameter=…>` was swallowed into THIS call's value — the pre-v0.7.3 first-close
    // parser returned both calls. `skip_tag_ws` already consumed exactly the boundary
    // characters `find_function_tag_mode` accepts before a bare tag, so a tag at
    // position 0 here is a tag there too.
    if rest.starts_with("<function=") || rest.starts_with("function=") {
        return true;
    }
    let Some(tail) = rest.strip_prefix("</function>") else { return false };
    let tail = skip_tag_ws(tail);
    tail.is_empty() || tail.starts_with("<function=") || tail.starts_with("function=")
}

/// End of a `<parameter=…>` value under the v0.7.3 rule: the byte offset (relative to `vstart`,
/// which runs to the end of the block body) of the STRUCTURAL `</parameter>`, or `None` when the
/// value is never closed (truncated output).
///
/// Pre-v0.7.3 this was `vstart.find("</parameter>")` — the FIRST close tag — so a value quoting
/// the tool format itself was silently truncated at the embedded tag (an agent writing a file
/// ABOUT tool calls lost the tail of the file). Each candidate close is validated by
/// `structural_continuation`; rejected candidates keep the scan going.
///
/// Fallback: when NO candidate satisfies the rule (malformed or truncated output) the FIRST
/// candidate is used — exactly the old behaviour, so garbage in still parses as it did.
///
/// Cost on the common path (exactly one `</parameter>` in the value): one `find` plus one O(1)
/// lookahead check; the fallback scan never runs. The candidate scan is SEQUENTIAL (each `find`
/// resumes where the previous candidate ended), so even a 1 MB value full of embedded close tags
/// costs one linear pass, never quadratic restarts.
///
/// S-B9-REL-SURFACE2B R1: this is the REFERENCE copy of the v0.7.3 rule, its body unchanged. Do not
/// edit it. The DEFAULT parse (`--tool-quote-pairing off`) does not call it: it runs
/// `param_end_v073_cached` below, which returns the same offset (T8a: it remembers, across the values
/// of a block, where a scan found no structural close). What keeps the two equal: the unit test
/// `exhaustive_small_blocks` compares the shipped parser under `off` with the verbatim copy in the
/// tests (`v073_param_end`) on every block of the exhaustive sets; `default_identity.rs` compares the
/// whole default parse with the frozen v0.7.3 parser; and `examples/toolparse_digest.rs` hashes what
/// the public `parse` returns for the same blocks (its digest on master is
/// accf2aa0df21389a4d8f397d28089caf393e51bb8c27e0d6cc084b46a8c31b82). The cost model's far choice
/// (`cost::Block::profile`) and the pairing rule's fallbacks use this function and the cached one.
pub(super) fn param_end_v073(vstart: &str) -> Option<usize> {
    const CLOSE: &str = "</parameter>";
    let first = vstart.find(CLOSE)?;
    let mut cand = first;
    loop {
        let after = &vstart[cand + CLOSE.len()..];
        if structural_continuation(after) {
            return Some(cand);
        }
        let Some(rel) = after.find(CLOSE) else { break };
        cand += CLOSE.len() + rel;
    }
    // No candidate satisfies the structural rule: keep today's behaviour (the first close tag).
    Some(first)
}

/// `param_end_v073` with the exhaustion cache of S-B9-TOOLPARSE3 T8a: the result is the function's,
/// byte for byte. `abs` is the offset of `vstart` in the block body; `dead_from` is the offset from
/// which an earlier scan of this block found NO structural close (it read to the end of the block):
/// `structural_continuation` looks only at the text after a candidate, so a later value whose first
/// close lies at or after that offset has no structural candidate either and gets its first close
/// without a scan. Before this, a block whose closes were all followed by junk rescanned the whole
/// tail once per value (quadratic: 1.6 MB took 26.7 s).
pub(super) fn param_end_v073_cached(vstart: &str, abs: usize, dead_from: &mut usize) -> Option<usize> {
    const CLOSE: &str = "</parameter>";
    let first = vstart.find(CLOSE)?;
    scan_ops(first + 1);
    if abs + first >= *dead_from { return Some(first); }
    let mut cand = first;
    loop {
        let after = &vstart[cand + CLOSE.len()..];
        scan_ops(1);
        if structural_continuation(after) { return Some(cand); }
        let Some(rel) = after.find(CLOSE) else { scan_ops(after.len()); break };
        scan_ops(rel + 1);
        cand += CLOSE.len() + rel;
    }
    *dead_from = abs + first;
    Some(first)
}

/// First occurrence of a fixed needle at or after a position, with a cache (T8a, and independent review
/// M1): the first occurrence at or after a position does not depend on where the search started, so one
/// scan answers every later position up to the occurrence it found, and a scan that found none answers
/// every later position. Answers are identical to `body[from..].find(needle)`.
pub(super) struct NeedleCache { needle: &'static str, at: usize, found: Option<usize>, valid: bool }

impl NeedleCache {
    pub(super) fn new(needle: &'static str) -> NeedleCache { NeedleCache { needle, at: 0, found: None, valid: false } }

    /// Absolute offset of the first `needle` at or after `from`.
    pub(super) fn next_abs(&mut self, body: &str, from: usize) -> Option<usize> {
        if !(self.valid && from >= self.at && self.found.map_or(true, |p| p >= from)) {
            self.found = body[from..].find(self.needle).map(|i| from + i);
            scan_ops(self.found.map_or(body.len() - from, |p| p - from) + 1);
            self.at = from;
            self.valid = true;
        }
        self.found
    }
}

/// The search for the next function tag from a position: the well-formed mode (a block that holds a
/// `<function=`) is cached; the bare `function=` repair depends on the slice start and is not.
///
/// KNOWN LIMIT (S-B9-TOOLPARSE3 part B, independent review; not changed in v0.7.4): a block of bare
/// `function=x` lines with no `<function=` and no `>` anywhere is quadratic (500 KB: about 6 s, the same
/// on v0.7.3). A cache for it was written and dropped: its first version sliced at a non-character
/// boundary, which the in-tree differential test (`default_identity.rs`, generator `gen_bare`) caught;
/// new caching code on the default path is not worth that risk in a release. The shape is not a thing
/// a model writes.
pub(super) struct TagCache { wf_only: bool, cache: NeedleCache }

impl TagCache {
    pub(super) fn new(wf_only: bool) -> TagCache { TagCache { wf_only, cache: NeedleCache::new("<function=") } }

    /// `find_function_tag_mode(&body[from..], wf_only)`, result relative to `from`.
    pub(super) fn next(&mut self, body: &str, from: usize) -> Option<(usize, usize)> {
        if !self.wf_only { return find_function_tag_mode(&body[from..], false); }
        self.cache.next_abs(body, from).map(|p| (p - from, "<function=".len()))
    }
}

/// `--tool-quote-pairing`: how the close of a `<parameter=…>` value is found (XML tool calls).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuotePairing {
    /// The v0.7.3 rule alone (`param_end_v073`). The default: byte-identical to v0.7.3.
    Off,
    /// Pair the tags of quoted calls first (`pair_scan`); the v0.7.3 rule decides whatever the
    /// pairing does not.
    On,
    /// The cost model (`cost::plan`): the cheapest reading of the whole block decides every value
    /// end (S-B9-TOOLPARSE3 T4); the v0.7.3 rule decides what the model has no opinion on.
    Cost,
}

/// The server-wide setting (the registry row's default when the option is not set).
pub fn quote_pairing() -> QuotePairing {
    match crate::opts::choice(crate::opt!("tool-quote-pairing")).as_str() {
        "on" => QuotePairing::On,
        "cost" => QuotePairing::Cost,
        _ => QuotePairing::Off,
    }
}

/// What `pair_scan` found for one value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Scan {
    /// The first candidate the v0.7.3 lookahead accepts, reached without paying any debt: this IS
    /// the v0.7.3 answer (the two scans tested the same candidates in the same order).
    Plain(usize),
    /// A close accepted after `paid` earlier closes were skipped as the closes of quoted
    /// parameters. `tail_clean`: no `<parameter=` and no `<function=` lies between the last
    /// skipped close and this one.
    Paired { cand: usize, tail_clean: bool },
    /// No close was accepted before the end of the block (`scanned` bytes read).
    Undecided { scanned: usize },
}

/// S-B9-REL-SURFACE2 item 10 (nesting), the pairing scan. The v0.7.3 lookahead alone accepts the
/// close of a parameter that was opened INSIDE the value whenever more structure follows it: a
/// value quoting one call with two parameters is cut at the first quoted close (what follows is
/// `<parameter=`), and a value quoting TWO calls comes back as two calls, the outer one cut short
/// and a second, PHANTOM call built from the quoted text, which an agent would execute. So this
/// scan pairs the tags of QUOTED CALLS: once the value has shown a `<function=` tag, every
/// `<parameter=` it passes is owed one `</parameter>`, and a close that pays such a debt is
/// content. Only a close with nothing owed is tested with the lookahead. Quoted calls are balanced
/// text, so the value's own close is the first one left over.
///
/// Why only after a `<function=`: a value may mention a parameter tag ALONE (`old_string` =
/// `<parameter=city>` in an edit of a document about this format). Counting that open would make
/// the value's real close pay for it and send the scan on into the following parameters and calls
/// (two edits in one block, one replacing the open tag and one the close tag, came back as one
/// call). For the same reason an open tag with a BLANK value is not counted: a value that ends in
/// the quoted opening of a call (`<function=x>` + `<parameter=p>`, then the real close) is
/// byte-identical to a quoted parameter with no value.
///
/// The scan reports; `param_end_paired` decides (see there for the guard and the fallbacks).
pub(super) fn pair_scan(vstart: &str, first: usize) -> Scan {
    const OPEN: &str = "<parameter=";
    const CLOSE: &str = "</parameter>";
    const CALL: &str = "<function=";
    // `owed` = `<parameter=` tags opened inside the value, after a `<function=` tag, and not yet
    // closed. `paid_end` = the end of the last close that paid such a debt.
    let (mut owed, mut scanned, mut cand, mut in_quoted_call) = (0usize, 0usize, first, false);
    let mut paid_end: Option<usize> = None;
    loop {
        let window = &vstart[scanned..cand];
        let from = if in_quoted_call { Some(0) } else { window.find(CALL) };
        // this close directly follows an open tag with a blank value
        let mut blank_pair = false;
        if let Some(f) = from {
            in_quoted_call = true;
            let w = &window[f..];
            let mut opened = w.matches(OPEN).count();
            // An open tag with NOTHING between it and this close is not counted: a value that ends
            // in a quoted `<parameter=k>` looks exactly like that (the close is then the value's own).
            if let Some(last) = w.rfind(OPEN) {
                if let Some(gt) = w[last..].find('>') {
                    if skip_tag_ws(&w[last + gt + 1..]).is_empty() { opened -= 1; blank_pair = true; }
                }
            }
            owed += opened;
        }
        scanned = cand + CLOSE.len();
        let after = &vstart[scanned..];
        if owed > 0 {
            // The close of a quoted parameter: content. It pays one debt, unless it is the close
            // of the blank pair just seen: that open was not counted, so its close pays for no
            // OTHER open either (review 2026-10-10: it used to, and a later value holding
            // `<parameter=p></parameter>` then ended an earlier value that mentions the tags).
            if !blank_pair {
                owed -= 1;
                paid_end = Some(scanned);
            }
        } else if structural_continuation(after) {
            return match paid_end {
                None => Scan::Plain(cand),
                Some(p) => {
                    let tail = &vstart[p..cand];
                    Scan::Paired { cand, tail_clean: !tail.contains(OPEN) && !tail.contains(CALL) }
                }
            };
        }
        let Some(rel) = after.find(CLOSE) else { break };
        cand = scanned + rel;
    }
    Scan::Undecided { scanned: vstart.len() }
}

/// Per-block state of `--tool-quote-pairing on`: the bytes pairing scans may still read without
/// deciding anything (S-B9-REL-SURFACE2B R1 [m1]).
///
/// A pairing scan whose close is not used (not trusted, refused by the catalogue, or none found)
/// has read past the close the v0.7.3 rule then picks, and the next value of the block starts
/// before the place the scan reached. A block of many values that
/// each open more quoted parameters than they close made every value scan to the end of the
/// block: quadratic (479 KB took 3.2 s, on the request path). The budget makes the total linear:
/// scans whose close IS used advance the parse by what they read, and the others share
/// `PAIR_BUDGET` x the block length. When it is spent the remaining values of the block use the v0.7.3 rule alone.
pub(super) struct PairState {
    budget: usize,
    /// the offset (in the block body) from which an earlier scan found no structural close: shared
    /// by the fallbacks to the v0.7.3 rule (independent review I2)
    dead_from: usize,
}

/// Undecided pairing scans may read this many times the block length (plus a floor for small
/// blocks) before pairing is switched off for the rest of the block. A block of N values can
/// waste at most N block lengths, so no block of 8 or fewer values is affected.
pub(super) const PAIR_BUDGET: usize = 8;

impl PairState {
    pub(super) fn new(body_len: usize) -> Self { PairState { budget: PAIR_BUDGET * body_len + 4096, dead_from: usize::MAX } }
}

/// The call a value belongs to: what the catalogue tie-break of `param_end_paired` looks at.
pub(super) struct ValueCx<'a> {
    /// the request's `tools` array (`None` = not known: no tie-break)
    pub(super) tools: Option<&'a [Value]>,
    /// the tool the call names
    pub(super) name: &'a str,
    /// the parameter whose value is being read
    pub(super) key: &'a str,
    /// the parameters of this call read so far
    pub(super) given: &'a serde_json::Map<String, Value>,
    /// the offset of the value in the block body (the exhaustion cache of `param_end_v073_cached`)
    pub(super) abs: usize,
}

/// Does the structure right after a `</parameter>` fit the request's catalogue as REAL structure?
/// A next parameter must be declared for the call's tool and not given yet; a next call (directly,
/// or after `</function>`) must name an offered tool; the end of the block fits. `None` = the
/// catalogue says nothing: no catalogue, text that is not structure, or a parameter of a tool that
/// is not offered, declares no properties, accepts additional ones (`additionalProperties` true
/// or a schema; absent counts as closed), or composes its schema (`anyOf` / `oneOf` / `allOf` /
/// `$ref` / `patternProperties`: only the top-level `properties` are read).
pub(super) fn continuation_fits(after: &str, cx: &ValueCx) -> Option<bool> {
    let tools = cx.tools?;
    let tool = |name: &str| tools.iter().find(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name));
    let offered = |r: &str| r.find('>').map(|gt| tool(r[..gt].trim()).is_some());
    fn call_tag(r: &str) -> Option<&str> { r.strip_prefix("<function=").or_else(|| r.strip_prefix("function=")) }
    let rest = skip_tag_ws(after);
    if rest.is_empty() { return Some(true); }
    if let Some(r) = rest.strip_prefix("<parameter=") {
        let k = r[..r.find('>')?].trim();
        let params = tool(cx.name)?.pointer("/function/parameters")?;
        let declared = params.get("properties").and_then(Value::as_object).filter(|p| !p.is_empty())?;
        if matches!(params.get("additionalProperties"), Some(Value::Bool(true)) | Some(Value::Object(_))) { return None; }
        // a schema that composes others may declare parameters this list does not show
        if ["anyOf", "oneOf", "allOf", "$ref", "patternProperties"].iter().any(|k| params.get(*k).is_some()) { return None; }
        return Some(k != cx.key && !cx.given.contains_key(k) && declared.contains_key(k));
    }
    if let Some(r) = call_tag(rest) { return offered(r); }
    let tail = skip_tag_ws(rest.strip_prefix("</function>")?);
    if tail.is_empty() { return Some(true); }
    offered(call_tag(tail)?)
}

/// End of a value under `--tool-quote-pairing on`.
///
/// 1. `pair_scan`. If it accepts a close without having skipped any, that is the v0.7.3 answer.
/// 2. If it skipped closes as the closes of quoted parameters, the result is TRUSTED only when
///    no `<parameter=` and no `<function=` lies between the last skipped close and the accepted
///    one (the guard of the independent review, 2026-10-09). A quoted call is balanced text, so
///    after its last parameter closes nothing opens again before the value's own close. Without
///    the guard a value that merely MENTIONS the format with more open tags than close tags (one
///    prose sentence is enough) ran on into the following real parameters, and a later real
///    parameter with a blank value ended it there: `edit(path, old_string=<that sentence>,
///    new_string="")` lost `new_string`, and two real calls came back as one.
/// 3. A trusted result that differs from the v0.7.3 close is checked against the request's
///    catalogue when there is one (`continuation_fits`): if what follows the v0.7.3 close fits
///    the catalogue and what follows the paired close does NOT (a parameter this tool does not
///    declare or was already given, a call to a tool that is not offered), the paired close lies
///    inside quoted text and the v0.7.3 close is kept. The catalogue only ever breaks this tie:
///    when both fit (a quoted call to an OFFERED tool with its declared parameters) it cannot
///    tell a quotation from real structure, and the pairing stands.
/// 4. Otherwise (not trusted, or no close accepted) the v0.7.3 rule decides, and after it the
///    first close tag.
///
/// The counts of each step against v0.7.3 are in `exhaustive_small_blocks`.
///
/// NOT solved, and not solvable inside this format (pinned by the tests named here):
///  * a value that ENDS with a quoted close followed by a quoted call is byte-for-byte two real
///    calls in one block (`a_value_that_ends_in_a_forged_close_and_call`);
///  * a quoted call whose LAST parameter is blank is not paired (the blank-value exemption of
///    `pair_scan`), so a second quoted call after it is still returned as a call
///    (`a_quoted_call_with_a_blank_last_argument_still_yields_a_phantom`);
///  * a value that opens more quoted parameters than it closes, in a block where a later value
///    holds a `</parameter>` that no open tag of a quoted call accounts for (a lone close tag),
///    runs on to that close: the class this setting reads wrongly and v0.7.3 rightly
///    (`exhaustive_small_blocks`, the "regressed" column). A MALFORMED block (text between a
///    real close and the next tag) can differ too: there both rules run on, to different closes.
pub(super) fn param_end_paired(vstart: &str, st: &mut PairState, cx: &ValueCx) -> Option<usize> {
    const CLOSE: &str = "</parameter>";
    let first = vstart.find(CLOSE)?;
    if st.budget > 0 {
        let scan = pair_scan(vstart, first);
        #[cfg(test)]
        PAIR_READ.with(|c| c.set(c.get() + match scan { Scan::Plain(c) | Scan::Paired { cand: c, .. } => c, Scan::Undecided { scanned } => scanned }));
        match scan {
            Scan::Plain(cand) => return Some(cand),
            Scan::Paired { cand, tail_clean: true } => {
                let old = param_end_v073_cached(vstart, cx.abs, &mut st.dead_from)?;
                let fits = |c: usize| continuation_fits(&vstart[c + CLOSE.len()..], cx);
                let quoted_tail = old != cand && fits(old) == Some(true) && fits(cand) == Some(false);
                if !quoted_tail { return Some(cand); }
                // the scan read to `cand` and the parse only advances to `old`: that is a scan
                // that did not decide, and it is charged like the others (review 2026-10-10:
                // uncharged, a block of such values was quadratic again)
                st.budget = st.budget.saturating_sub(cand);
                return Some(old);
            }
            Scan::Paired { cand, tail_clean: false } => st.budget = st.budget.saturating_sub(cand),
            Scan::Undecided { scanned } => st.budget = st.budget.saturating_sub(scanned),
        }
    }
    param_end_v073_cached(vstart, cx.abs, &mut st.dead_from)
}

#[cfg(test)]
thread_local! {
    /// Bytes the pairing scans of this thread have read (the linearity test reads it).
    pub(super) static PAIR_READ: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
