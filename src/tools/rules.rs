//! The opt-in rule enums of the tool-call parser (`--tool-block-end`, value trim, `--tool-unclosed`) and `Rules`.
//! Defaults keep the v0.7.3 reading; the served rules come from the startup flags. Map of the directory in `mod.rs`.

use super::*;

/// `--tool-block-end`: where a `<tool_call>` block ends (S-B9-TOOLPARSE3 T3; part B item 12).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockEnd {
    /// The next `</tool_call>` in the text (v0.7.3 and earlier), except that a JSON-bodied block ends
    /// at the first close that follows `}` (item 7). `--tool-block-end first`.
    First,
    /// What the popular engines do (vLLM's state machine: inside the arguments the only terminal is
    /// `</function>`, and `</tool_call>` is looked for only AFTER it): a `</tool_call>` ends the block
    /// only where the call has ended (`after_function_close`). The default.
    AfterFunction,
}

/// `--tool-value-trim`: which newlines of an argument value are the format's own (S-B9-TOOLPARSE3
/// part B item 8; the owner's ruling of 2026-10-10: do what the popular engines do).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ValueTrim {
    /// ONE leading and ONE trailing newline (`\n` only: the template's `<parameter=k>\n` and
    /// `\n</parameter>`), nothing else: vLLM's `_trim_wrapping_newlines`, byte for byte. A file the
    /// model ends with a blank line arrives with its final newline. The default.
    One,
    /// Every leading and every trailing newline (v0.7.3 and earlier).
    All,
}

/// The server-wide setting of `--tool-value-trim`.
pub fn value_trim() -> ValueTrim {
    if crate::opts::choice(crate::opt!("tool-value-trim")) == "all" { ValueTrim::All } else { ValueTrim::One }
}

/// Strip exactly one leading and one trailing `\n`, in that order (vLLM's rule: a value that is just
/// `\n` becomes empty, `\r\n` loses only the `\n`).
pub fn trim_wrapping_newlines(v: &str) -> &str {
    let v = v.strip_prefix('\n').unwrap_or(v);
    v.strip_suffix('\n').unwrap_or(v)
}

/// The server-wide setting of `--tool-block-end`.
/// `--tool-unclosed`: what to do with a `<tool_call>` block whose `</tool_call>` never came (F8).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unclosed { Complete, Drop }

/// `drop` (the default since F11, owner 2026-10-12): v0.7.3, the whole block is dropped; `complete`: when the
/// model ended by itself (finish stop), function elements that ARE closed inside the unclosed wrapper come
/// back as calls (vLLM's qwen3 rule).
impl Unclosed {
    /// the flag's value as text (the rules_info gauge and owner_check print it)
    pub fn name(self) -> &'static str { if self == Unclosed::Complete { "complete" } else { "drop" } }
}

/// Test instrument (F8): would the served rules return calls from a wrapper the model never closed
/// that the v0.7.3 rules drop? The narrow class the default identity excuses, finish stop only.
#[cfg(test)]
pub(crate) fn unclosed_block_moves(text: &str) -> bool {
    let served = Rules::served();
    if served.unclosed == Unclosed::Drop || !text.contains(CALL_OPEN) || is_hy3(text) { return false; }
    let dropped = Rules { unclosed: Unclosed::Drop, ..served };
    parse_rules(text, None, &served, Some("stop")).tool_calls.len()
        != parse_rules(text, None, &dropped, Some("stop")).tool_calls.len()
}

pub fn unclosed_rule() -> Unclosed {
    // test-only, per-thread: the tests of the F8 completion read the served rules with `complete` set without
    // touching the process-global option store that parallel tests share (the default is `drop`, F11)
    #[cfg(test)]
    if let Some(u) = TEST_UNCLOSED.with(|c| c.get()) { return u; }
    if crate::opts::choice(crate::opt!("tool-unclosed")) == "complete" { Unclosed::Complete } else { Unclosed::Drop }
}

#[cfg(test)]
thread_local! { static TEST_UNCLOSED: std::cell::Cell<Option<Unclosed>> = const { std::cell::Cell::new(None) }; }

/// Test instrument (F11): the guard restores the previous per-thread override when it drops.
#[cfg(test)]
pub(crate) struct UnclosedGuard(Option<Unclosed>);
#[cfg(test)]
impl Drop for UnclosedGuard { fn drop(&mut self) { TEST_UNCLOSED.with(|c| c.set(self.0)); } }
/// Test instrument (F11): serve `u` on this thread until the guard drops (the registry default is `drop`).
#[cfg(test)]
pub(crate) fn test_unclosed(u: Unclosed) -> UnclosedGuard {
    UnclosedGuard(TEST_UNCLOSED.with(|c| c.replace(Some(u))))
}

pub fn block_end() -> BlockEnd {
    if crate::opts::choice(crate::opt!("tool-block-end")) == "first" { BlockEnd::First } else { BlockEnd::AfterFunction }
}

/// Every rule the XML tool-call parser reads, in one value: how a value ends, how a block ends, and
/// the weights of the cost model. `Rules::served()` is what the server runs (its flags); the
/// tests, the shadow counting and the offline replays pass their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    pub pairing: QuotePairing,
    pub block: BlockEnd,
    pub weights: cost::Weights,
    pub trim: ValueTrim,
    pub unclosed: Unclosed,
}

impl Rules {
    /// The value and block rules of v0.7.3 and its newline rule (`--tool-value-trim all`): what the
    /// escape-hatch options select. (The defect fixes of part B, items 5 to 7, apply to it too.)
    pub const V073: Rules = Rules { pairing: QuotePairing::Off, block: BlockEnd::First, weights: cost::Weights::SHIPPED, trim: ValueTrim::All, unclosed: Unclosed::Drop };

    /// The rules the server was started with.
    pub fn served() -> Rules {
        Rules { pairing: quote_pairing(), block: block_end(), weights: cost::Weights::SHIPPED, trim: value_trim(), unclosed: unclosed_rule() }
    }
}
