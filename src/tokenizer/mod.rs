//! The tokenizer front of the engine: `QwenTokenizer` (tokenizer.json + the model's own chat template through
//! minijinja), the OpenAI chat message types and the byte-level streaming decoder. The Qwen 3.5/3.6 line shares
//! one vocabulary; hy_v3 and the other families load through the same type.
//!
//! Map of this directory (items keep their `crate::tokenizer::` paths):
//! - `qwen.rs`: `impl QwenTokenizer` (load, encode/decode, stop tokens, chat-template rendering) and `load_tokenizer`
//! - `messages.rs`: `ToolCall`, `FunctionCall`, `ImageInput`, `ChatMessage` (request deserialization and template JSON)
//! - `jinja.rs`: `load_chat_env` and the Python-compatible template helpers (`register_pycompat`)
//! - `stream.rs`: `StreamByteDecoder` (whole UTF-8 sequences across byte-level tokens) and its tables
//! - `tests.rs`: the unit tests
//! `mod.rs` keeps the `QwenTokenizer` struct, `TokCache`, `ThinkingMode` and the thinking-kwarg helpers.

use tokenizers::Tokenizer;
use anyhow::Result;
use std::path::Path;

mod jinja;
mod messages;
mod qwen;
mod stream;

use self::jinja::*;
pub use self::messages::*;
pub use self::stream::*;

pub struct QwenTokenizer {
    tokenizer: Tokenizer,
    /// Directory the tokenizer was loaded from — where generation_config.json / tokenizer_config.json
    /// live. Stop tokens are read from the model's own files, never hardcoded.
    model_dir: Option<std::path::PathBuf>,
    /// Rendered from the model's own `chat_template.jinja` (or the `chat_template`
    /// field of `tokenizer_config.json`). `None` only when no template file is found
    /// next to the tokenizer, in which case the legacy hand-rolled template is used.
    chat_env: Option<minijinja::Environment<'static>>,
    /// Server-wide `tool_call_format` template kwarg (Froggeric-class templates: 'xml' default,
    /// 'json' = the model's native Qwen tool-call syntax our parser handles). None = the
    /// template's own default applies.
    pub tool_call_format: Option<String>,
    /// Froggeric-class agentic guard: pass `auto_disable_thinking_with_tools=true` to the
    /// template (thinking auto-disables when the request carries tools — the template's own
    /// fix for tool-call stalls under thinking; default off = the template's default).
    pub auto_disable_think_tools: bool,
    /// Template provenance captured at load: (origin, sha256 hex, byte length). Phase-2 A3
    /// observability — the boot line must show WHICH template the process renders with: the
    /// base model dir ships the STOCK Qwen template, and a leg booted against it silently renders
    /// a different prompt (RENDER_AUDIT.md 9.1).
    pub template_meta: Option<(String, String, u64)>,
    /// True when the loaded template has a `tc.arguments is string` branch (Froggeric-class).
    /// Only then is the RAW OpenAI `arguments` string handed to the template — that is what the
    /// reference renders verbatim inside `<function=…>` (RENDER_AUDIT.md 2.2). Templates without
    /// the branch still need the parsed-object form (they iterate `arguments | items`).
    pub raw_tool_args_history: bool,
    /// WP01: vocab-derived tables built ONCE at load (think markers, eos, vocab size, the stream
    /// decoder's id->bytes/special tables). `None` only under `--wp01-off=1`, which restores the
    /// legacy per-call rebuilds (a 248K-entry `get_vocab(true)` HashMap per call + a 12.8 MB
    /// tokenizer.json re-parse per stream) for A/B and bisect.
    cache: Option<TokCache>,
}

/// WP01 load-time cache. Every field is the value the legacy per-call expression computes; the
/// tokenizer is immutable after `from_file`, so computing it once is exact.
struct TokCache {
    /// `get_vocab(true).contains_key("</think:opensource>")` (hy_v3 markers).
    think_hy: bool,
    /// `get_vocab(true).get("<|endoftext|>")`, 151643 fallback.
    eos_id: u32,
    /// `get_vocab_size(true)` (= `get_vocab(true).len()` in tokenizers 0.19).
    vocab_size: usize,
    stream: std::sync::Arc<StreamTables>,
}

fn wp01_env(o: crate::opts::OptId) -> bool {
    crate::opts::var(o).map_or(false, |v| !v.is_empty() && v != "0")
}

/// The server-wide `--thinking` policy (W1, Phase 13): may the ENGINE override the model
/// template's own thinking default, and in which direction?
///
///   Auto (default) — pass NOTHING to the template. The model's own `chat_template.jinja`
///                    decides. This is what makes a user-edited template (thinking defaulted
///                    OFF) actually take effect: the engine no longer hardcodes
///                    `enable_thinking=true` into every render, which was overriding the
///                    template's own default branch.
///   On             — render the template's thinking branch (`enable_thinking=true`).
///   Off            — render its no-think branch (`enable_thinking=false`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingMode { Auto, On, Off }

impl ThinkingMode {
    /// Parse the CLI spelling. Unknown values are the caller's error (never silently defaulted).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Some(Self::Auto),
            "on" | "true" | "yes" | "1" => Some(Self::On),
            "off" | "false" | "no" | "0" | "no_think" => Some(Self::Off),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self { Self::Auto => "auto", Self::On => "on", Self::Off => "off" }
    }
    /// The value to hand the template, or `None` to pass nothing at all (Auto).
    pub fn as_option(self) -> Option<bool> {
        match self { Self::Auto => None, Self::On => Some(true), Self::Off => Some(false) }
    }
}

/// Resolve `enable_thinking` for ONE render. Precedence, most specific wins:
///   1. the request's `chat_template_kwargs.enable_thinking` (a bool) — the client said it;
///   2. the request's `reasoning_effort` in its off-class (`off` / `no_think`) — OpenAI's
///      no-think conventions, which the Qwen template cannot accept as an effort value;
///   3. the server's `--thinking` policy (On/Off; Auto contributes nothing);
///   4. `None` — pass nothing and let the model's OWN template decide.
///
/// This is the whole W1 fix in one place: a customer sending
/// `chat_template_kwargs: {"enable_thinking": false}` must reach the render, and a customer
/// who edits the template instead must get THEIR default when nobody asks otherwise.
pub fn resolve_enable_thinking(kwarg: Option<bool>, effort: Option<&str>,
                               server: ThinkingMode) -> Option<bool> {
    kwarg
        .or_else(|| match effort { Some("off") | Some("no_think") => Some(false), _ => None })
        .or(server.as_option())
}

/// Extract `enable_thinking` from a request's `chat_template_kwargs`, with the two failure
/// modes that used to be silent:
///   - the field must be a JSON object (not a string/array/null);
///   - `enable_thinking`, when present, must be a boolean.
/// Both raise, so the handler turns them into a loud 400 instead of dropping the client's
/// request on the floor (the accept-and-ignore class of F6).
pub fn enable_thinking_kwarg(kwargs: Option<&serde_json::Value>) -> Result<Option<bool>> {
    let Some(v) = kwargs else { return Ok(None) };
    let Some(map) = v.as_object() else {
        anyhow::bail!("'chat_template_kwargs' must be a JSON object (got {})",
                      match v { serde_json::Value::Null => "null".to_string(), other => other.to_string() });
    };
    match map.get("enable_thinking") {
        None => Ok(None),
        Some(serde_json::Value::Bool(b)) => Ok(Some(*b)),
        Some(other) => anyhow::bail!("'chat_template_kwargs.enable_thinking' must be a boolean (got {other})"),
    }
}

#[cfg(test)]
mod tests;
