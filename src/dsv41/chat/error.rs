//! Error type of the chat layer. The kinds mirror what `encoding.py` raises, so a server can map them
//! to one HTTP 400 and the corpus can assert the same class of refusal.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatErrorKind {
    /// Python `AssertionError` (invalid effort, tool namespace conflicts, task names, ...).
    Assertion,
    /// Python `ValueError` (image placeholder text, malformed image blocks, ...).
    Value,
    /// Python `NotImplementedError` (unknown role, a `tool` role reaching the renderer).
    NotImplemented,
    /// Input the Python code would fail on with a `KeyError`/`TypeError`/`AttributeError`.
    Input,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatError {
    pub kind: ChatErrorKind,
    pub message: String,
}

impl ChatError {
    pub fn assertion(m: impl Into<String>) -> Self {
        Self { kind: ChatErrorKind::Assertion, message: m.into() }
    }
    pub fn value(m: impl Into<String>) -> Self {
        Self { kind: ChatErrorKind::Value, message: m.into() }
    }
    pub fn not_implemented(m: impl Into<String>) -> Self {
        Self { kind: ChatErrorKind::NotImplemented, message: m.into() }
    }
    pub fn input(m: impl Into<String>) -> Self {
        Self { kind: ChatErrorKind::Input, message: m.into() }
    }
}

impl fmt::Display for ChatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ChatError {}
