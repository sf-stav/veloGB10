//! DeepSeek-V4.1 chat layer: the prompt encoder (a byte-exact port of the release's
//! `encoding/encoding.py`) and the DSML tool-call parser (whole and streaming).
//!
//! Map of the directory:
//! - `pyjson.rs`: Python `json` compatibility (number repr, duplicate keys) over `dsv4_chat::Json`
//! - `jx.rs`: accessors and mutators of the ordered `Json`
//! - `error.rs`: `ChatError` (kinds mirror what `encoding.py` raises)
//! - `effort.rs`: `reasoning_effort` (integer 1..=100 or low/high/max)
//! - `tools.rs`: tool rendering, namespaces, DSML encoding of assistant calls
//! - `preprocess.rs`: tool message merging, result sorting, image placeholders
//! - `render.rs`: `render_message` / `encode_messages`
//! - `special.rs`: special token ids and the stop-token union
//! - `catalogue.rs`: the offered tools and the checks a call must pass
//! - `request.rs`: thinking toggles, effort aliases, tool choice
//! - `stream.rs`: the completion parser (whole text and byte-split streaming)
//! - `tests.rs`: corpus, release fixtures, parser and fuzz tests

pub mod catalogue;
pub mod effort;
pub mod error;
pub mod jx;
pub mod preprocess;
pub mod pyjson;
pub mod render;
pub mod request;
pub mod special;
pub mod stream;
pub mod tools;

pub use error::{ChatError, ChatErrorKind};
pub use preprocess::{parse_tagged_text, IMAGE_PLACEHOLDER};
pub use render::{
    attach_tools, encode_messages, render_message, EncodeOptions, Encoded, ThinkingMode, ASSISTANT_SP_TOKEN, BOS_TOKEN,
    EOS_TOKEN, SYSTEM_SP_TOKEN, USER_SP_TOKEN,
};

#[cfg(test)]
mod tests;
