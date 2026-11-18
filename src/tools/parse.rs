//! Entry points of the parser: `parse`, `parse_served`, `parse_rules`, `parse_core` (the block walk), `block_spans`, `finalize`.
//! Calls into `blocks`, `hy3`, `calls` and `shadow` (map in `mod.rs`); ids come from the process-wide counter (`mint_id`).

use super::*;

/// Parse a completed generation. `tools` is the request's `tools` array, used only to type-coerce
/// arguments; parsing still works (as strings) when it is absent. Format dispatch is by content:
/// hy_v3's `:opensource` markup or qwen's `<function=…>` markup.
pub fn parse(text: &str, tools: Option<&[Value]>) -> ParsedOutput {
    parse_rules(text, tools, &Rules::served(), None)
}

/// The server's entry: `parse` with the engine's finish reason (`--tool-block-end structural`
/// reads it) and the shadow counting of S-B9-TOOLPARSE3 T1: the same text is also read under the
/// rules that are NOT served and the disagreements are counted (`/metrics`, one content-free log
/// line each). The served result is the first parse and nothing of the shadow reads reaches it.
///
/// The shadow is BOUNDED (independent review B1; the first version's bound was only checked between
/// rules and the cost model had none):
///  * the rules the default shadow reads are the cheap ones, `on` and the block rule: each is one more
///    linear parse of the text, as dear as the served parse itself (the report's table: +0.01 to
///    +0.05 ms for ordinary responses, +6 ms in the worst shape, 64 blocks of 8 KB with 100 tools
///    offered); the cost model is read only when the operator asks for it (`--tool-shadow-cost on`);
///  * a text over `SHADOW_MAX_BYTES` or with more than `SHADOW_MAX_BLOCKS` blocks is not compared
///    (counted as skipped);
///  * the cost model, wherever it runs, spends at most `cost::RESPONSE_UNITS` work units per response
///    (a counter, not a clock: `cost` module header) and declines a block over its budget;
///  * the comparison stops between rules when it has used `SHADOW_BUDGET` of wall clock.
/// A panic in an unserved rule is caught and counted (its message is not printed: it can quote text).
pub fn parse_served(text: &str, tools: Option<&[Value]>, finish: &str) -> ParsedOutput {
    let rules = Rules::served();
    if !text.contains(CALL_OPEN) || is_hy3(text) {
        return parse_rules(text, tools, &rules, Some(finish));
    }
    let limit = shadow_limit(text);
    let mut recs = Recs::new(limit.is_none());
    let t_served = std::time::Instant::now();
    let out = parse_core(text, tools, &rules, Some(finish), Some(&mut recs));
    crate::metrics::toolcall_time(0, t_served.elapsed().as_micros() as u64);
    // every response that holds a block is counted, whether or not the comparison is made
    crate::metrics::toolcall_blocks(recs.blocks);
    if let Some(outcome) = recs.unclosed_outcome { crate::metrics::toolcall_unclosed_outcome(outcome); }
    if let Some(reason) = limit {
        crate::metrics::toolcall_shadow_skipped(reason);
        return out;
    }
    // The comparison runs the rules that are NOT served, over every response of every default server: a
    // defect in one of them must never reach a client. A panic is caught, counted and logged, and the
    // served result above is returned unchanged.
    let with_cost = crate::opts::choice(crate::opt!("tool-shadow-cost")) == "on";
    install_shadow_panic_hook();
    let t_shadow = std::time::Instant::now();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        shadow_from(text, tools, &rules, Some(finish), &out, &recs.v, with_cost)
    }));
    crate::metrics::toolcall_time(1, t_shadow.elapsed().as_micros() as u64);
    match r {
        Ok(sh) => {
            if let Some(r) = sh.skipped { crate::metrics::toolcall_shadow_skipped(r); }
            if sh.blocks > 0 { crate::metrics::toolcall_shadow(&sh, &rules); }
        }
        Err(_) => crate::metrics::toolcall_shadow_failed(),
    }
    out
}

/// The `<tool_call>` blocks of `text` under a block rule: for each, the byte range of its body, and
/// whether it closed. Diagnostic API (the statistics of S-B9-TOOLPARSE3 T2 read real output through
/// it); the parser itself walks the text in `parse_core`.
pub fn block_spans(text: &str, block: BlockEnd, finish: Option<&str>) -> Vec<(usize, usize, bool)> {
    let mut out = Vec::new();
    if is_hy3(text) { return out; }
    let idx = if block == BlockEnd::AfterFunction { Some(BlockIndex::new(text)) } else { None };
    let mut base = 0usize;
    while let Some(open) = text[base..].find(CALL_OPEN) {
        let a = base + open + CALL_OPEN.len();
        match block_close(text, a, block, finish, idx.as_ref()) {
            Some((end, next)) => { out.push((a, a + end, true)); base = a + next; }
            None => { out.push((a, text.len(), false)); break; }
        }
    }
    out
}

/// `parse` with the value-terminator rule given explicitly (`--tool-quote-pairing`; the tests and
/// the offline replay of `--dump-tools` collections drive the settings through this).
pub fn parse_with(text: &str, tools: Option<&[Value]>, pairing: QuotePairing) -> ParsedOutput {
    parse_rules(text, tools, &Rules { pairing, ..Rules::V073 }, None)
}

thread_local! {
    /// True while the shadow comparison parses: ids are not minted from the process-wide counter
    /// (the served ids must not depend on shadowing) and no alarm line is printed.
    pub(super) static SHADOW_PARSE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn shadow_parse() -> bool { SHADOW_PARSE.with(|c| c.get()) }

/// A call id: from the process-wide counter, or a fixed placeholder while shadowing.
pub(super) fn mint_id() -> String {
    if shadow_parse() { "shadow".to_string() } else { format!("call_{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed)) }
}

/// What one block of the served text came to (for the shadow counting).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BlockRec {
    /// byte range of the block body in the text
    pub(super) span: (usize, usize),
    /// (name, arguments) of every call it holds
    pub(super) calls: Vec<(String, String)>,
    /// the cost model could not read this block within its work budget (the v0.7.3 rule did)
    pub(super) declined: bool,
}

/// What a parse records about its blocks: how many closed blocks it read, and, when `full`, each
/// block's record (the shadow comparison needs the calls; counting alone does not, so a response
/// the shadow will not read is not cloned).
pub(super) struct Recs {
    full: bool,
    blocks: usize,
    pub(super) v: Vec<BlockRec>,
    /// the text ended inside a block that never closed: whether what it held ends the way a
    /// finished call ends (`</function>`, `</parameter>`, or `}` for a JSON body)
    unclosed: Option<bool>,
    /// F8: how that block was disposed of ("returned" / "dropped_cut" / "dropped_length")
    unclosed_outcome: Option<&'static str>,
}

impl Recs {
    pub(super) fn new(full: bool) -> Recs { Recs { full, blocks: 0, v: Vec::new(), unclosed: None, unclosed_outcome: None } }
}

/// `parse` under explicit rules. `finish` is the engine's finish reason when the caller knows it
/// (`--tool-block-end structural` tells a block cut by the token budget from one that merely lacks
/// its close).
pub fn parse_rules(text: &str, tools: Option<&[Value]>, rules: &Rules, finish: Option<&str>) -> ParsedOutput {
    parse_core(text, tools, rules, finish, None)
}

pub(super) fn parse_core(text: &str, tools: Option<&[Value]>, rules: &Rules, finish: Option<&str>,
              mut recs: Option<&mut Recs>) -> ParsedOutput {
    #[cfg(test)]
    PARSE_CORE_CALLS.with(|c| c.set(c.get() + 1));
    if is_hy3(text) {
        return parse_hy3(text, tools);
    }
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut rest = text;
    // the tag positions of the whole text, once (the structural block rule judges every block by
    // binary searches over them; independent review I1: it used to tokenise the remainder per block)
    let idx = if rules.block == BlockEnd::AfterFunction { Some(BlockIndex::new(text)) } else { None };
    // the cost model's context for this response: the catalogue as a lookup (built when a block needs
    // it) and the work budget the response's blocks share
    let ctx = cost::Ctx::new(tools);

    while let Some(open) = rest.find(CALL_OPEN) {
        content.push_str(&rest[..open]);
        let after = &rest[open + CALL_OPEN.len()..];
        let after_abs = text.len() - after.len();
        // A truncated call (hit max_tokens mid-emit) has no close tag. Drop it entirely: a half-parsed
        // call is worse than none, because the harness would invoke a real function with missing
        // arguments. `rest` MUST be cleared before breaking -- leaving it would append the partial XML
        // to content (and duplicate the prose before it), which is the exact leak this guards against.
        let Some((body_end, next)) = block_close(text, after_abs, rules.block, finish, idx.as_ref()) else {
            // F8 (--tool-unclosed complete): the model never closed the wrapper but the response
            // ended by itself (finish stop). vLLM's qwen3 parser returns every element whose own
            // `</function>` IS there (its state machine ends a call at FUNC_END; the wrapper is only
            // preamble), so read the body to text end with the SAME block parser and keep what the
            // existing value rules accept: a trailing element cut inside an argument still yields
            // nothing (its value never closed). A JSON-led body is dropped (vLLM's XML parser makes
            // no call of it). Every other finish (length, cancel, engine error: the text was CUT,
            // not ended) and `drop` keep the v0.7.3 behaviour: drop the block whole.
            let declined_before = ctx.declined();
            let completed: Vec<ToolCall> = if rules.unclosed == Unclosed::Complete
                && finish == Some("stop") {
                parse_unclosed_body(after, tools, tool_calls.len(), rules, &ctx)
            } else { Vec::new() };
            if ctx.declined() > declined_before && !shadow_parse() { crate::metrics::toolcall_cost_declined(); }
            if let Some(r) = recs.as_deref_mut() {
                r.unclosed = Some(ends_like_a_call(after));
                r.unclosed_outcome = Some(if !completed.is_empty() { "returned" }
                                          else if finish == Some("stop") { "dropped_cut" }
                                          else { "dropped_length" });
            }
            for tc in completed { tool_calls.push(tc); }
            rest = "";
            break;
        };
        let body = &after[..body_end];
        let first_new = tool_calls.len();
        let declined_before = ctx.declined();
        for tc in parse_block(body, tools, tool_calls.len(), rules, &ctx) {
            tool_calls.push(tc);
        }
        let declined = ctx.declined() > declined_before;
        if declined && !shadow_parse() { crate::metrics::toolcall_cost_declined(); }
        if let Some(r) = recs.as_deref_mut() {
            r.blocks += 1;
            if r.full {
                let base = after.as_ptr() as usize - text.as_ptr() as usize;
                r.v.push(BlockRec {
                    span: (base, base + body_end),
                    calls: tool_calls[first_new..].iter().map(|c| (c.function.name.clone(), c.function.arguments.clone())).collect(),
                    declined,
                });
            }
        }
        rest = &after[next..];
    }
    content.push_str(rest);

    #[cfg(test)]
    COST_USED.with(|c| c.set((ctx.used(), ctx.declined())));
    ParsedOutput { content: content.trim().to_string(), tool_calls }
}

/// The ONE final (content, tool_calls, finish_reason) decision for a completed generation,
/// shared by BOTH response modes — streaming SSE and non-streaming JSON.
///
/// This exists because the modes used to decide independently and could diverge (the
/// 2026-08-27 user report): with a call the parser could not read, streaming dropped the
/// block's text entirely while non-streaming returned it verbatim. Both modes now call this
/// on the same post-think-split answer text, so they can only ever agree.
///
/// Rules (the exact old non-streaming behavior, now canonical for both):
///  - at least one call parsed  -> content becomes the prose outside the calls (`None` when
///    empty) and finish becomes `"tool_calls"` — the flag every harness branches on;
///  - no call parsed            -> content is the model's literal answer text UNCHANGED
///    (recoverable by the operator) and `finish` passes through.
pub fn finalize(raw: &str, tools: Option<&[Value]>, finish: &str) -> (Option<String>, Vec<ToolCall>, String) {
    let parsed = parse(raw, tools);
    finalize_parsed(raw, parsed, finish)
}

/// Decision half of [`finalize`] for callers that already ran `parse` (e.g. to log the raw
/// output). Takes the parse result by value so call ids are minted exactly once.
pub fn finalize_parsed(
    raw: &str,
    parsed: ParsedOutput,
    finish: &str,
) -> (Option<String>, Vec<ToolCall>, String) {
    if parsed.tool_calls.is_empty() {
        (Some(raw.to_string()), Vec::new(), finish.to_string())
    } else {
        let c = if parsed.content.is_empty() { None } else { Some(parsed.content) };
        (c, parsed.tool_calls, "tool_calls".to_string())
    }
}
