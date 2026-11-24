//! The Inkling-2 chat surface: the request side (messages to template inputs, reasoning effort, image splice points, the tool
//! catalogue) and the reply side (the model's block stream to reasoning, content and tool calls). Pure functions of their
//! inputs: no sockets, no GPU; `serve/` wires them to HTTP.
//!
//! Map:
//! - `special.rs`:   the structural token ids of the vocabulary (message headers, content kinds, `<|end_message|>`, EOS 200006).
//! - `effort.rs`:    `reasoning_effort` (a name or a number) to the value the template receives.
//! - `catalogue.rs`: the tools a request offered and the checks a parsed call must pass (never invent a call).
//! - `parse.rs`:     the reply parser (token events in, `Event`s out; one engine for the whole text and for streaming).
//! - `request.rs`:   one OpenAI chat body to template inputs (messages, tools, kwargs) and the rendered prompt's image splice points.

pub mod catalogue;
pub mod effort;
pub mod parse;
pub mod request;
pub mod special;

#[cfg(test)]
mod tests;
