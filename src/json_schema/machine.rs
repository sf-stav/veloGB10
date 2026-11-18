//! The byte machine proper: object/array phases, the continuation `Frame` stack, `Machine`, `canon_match` and `Roll`.
//! A state is an interned stack snapshot (see `mask.rs`). Map of the directory in `mod.rs`.

use super::*;

// ===========================================================================================
// Byte machine: frames + states
// ===========================================================================================

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) enum ObjPhase {
    /// Whitespace, then `}` or the opening quote of a key.
    KeyOrEnd,
    /// Directly after a `,`: a key must follow (`}` would be a trailing comma).
    KeyReq,
    /// Inside a key string. Parsed *inside* the object frame so phase transitions never need to
    /// reach across stack frames.
    Key(StrState),
    /// Key closed; expect whitespace then `:`. `child` is the schema for the member value.
    Colon { child: u32 },
    /// The member value has been parsed; expect whitespace, `,` or `}`.
    AfterValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum ArrPhase {
    /// Whitespace, then `]` or the start of an item.
    ItemOrEnd,
    /// Directly after a `,`: an item must follow (`]` would be a trailing comma).
    ItemReq,
    /// An item has been parsed; expect whitespace, `,` or `]`.
    AfterItem,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) enum Frame {
    /// Top-level frame: after the value only whitespace (then end) may follow.
    DocEnd,
    /// The byte stream left the schema / the JSON grammar. Absorbs everything after it.
    Dead,
    /// A value matching schema `si` must start here.
    Value { si: u32 },
    Obj { si: u32, phase: ObjPhase, seen: Vec<u32>, extra: Vec<String> },
    Arr { si: u32, n: u64, phase: ArrPhase },
    Str { si: u32, st: StrState },
    /// Remaining bytes of `true` / `false` / `null`.
    Lit { rest: &'static [u8] },
    Num { si: u32, buf: Vec<u8>, st: NumState },
}

/// An interned machine state: a stack of continuation frames.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub(super) struct Machine {
    pub(super) stack: Vec<Frame>,
    /// Consecutive STRUCTURAL whitespace bytes just consumed (whitespace between tokens of the
    /// document, never string content). Capped at [`MAX_WS_RUN`]: without a cap, whitespace is
    /// legal in every structural position, so a model that does not want to start the document
    /// can emit blank space until max_tokens and the reply never parses.
    pub(super) ws: u8,
}

/// Longest run of structural whitespace the machine accepts (enough for pretty-printing a deeply
/// nested document: a newline plus indentation).
pub(super) const MAX_WS_RUN: u8 = 24;

impl Machine {
    pub(super) fn is_dead(&self) -> bool {
        self.stack.iter().any(|f| matches!(f, Frame::Dead))
    }
    /// Exactly the completed-document shape: only `DocEnd` left (whitespace is absorbed in place).
    pub(super) fn at_doc_end(&self) -> bool {
        self.stack.len() == 1 && matches!(self.stack[0], Frame::DocEnd)
    }
    /// True while the next byte is string CONTENT (a value string or an object key).
    pub(super) fn in_string(&self) -> bool {
        matches!(self.stack.last(), Some(Frame::Str { .. }) | Some(Frame::Obj { phase: ObjPhase::Key(_), .. }))
    }
}

/// The canonical JSON encoding (serde_json's: `"` `\\` and the short control escapes, other
/// control characters as lowercase `\u00XX`, everything else as raw UTF-8) of string `v`,
/// without the quotes, compared against `raw` (the bytes emitted so far inside the string).
/// Returns (`raw` is a prefix of the encoding, `raw` is the whole encoding).
///
/// Enum values and declared property names are matched in this ONE canonical spelling. Matching
/// the decoded value instead admits every escaped spelling (`"\u0074"` for `t`), and a model
/// under the mask then wanders into long escape sequences (or into escapes and partial UTF-8
/// sequences that no candidate completes) instead of writing the value.
pub(super) fn canon_match(v: &str, raw: &[u8]) -> (bool, bool) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut i = 0usize;
    let mut buf = [0u8; 6];
    for c in v.chars() {
        let enc: &[u8] = match c {
            '"' => b"\\\"",
            '\\' => b"\\\\",
            '\u{08}' => b"\\b",
            '\u{0C}' => b"\\f",
            '\n' => b"\\n",
            '\r' => b"\\r",
            '\t' => b"\\t",
            c if (c as u32) < 0x20 => {
                buf = [b'\\', b'u', b'0', b'0', HEX[(c as usize) >> 4], HEX[(c as usize) & 15]];
                &buf[..]
            }
            c => c.encode_utf8(&mut buf).as_bytes(),
        };
        for &e in enc {
            if i == raw.len() {
                return (true, false);
            }
            if raw[i] != e {
                return (false, false);
            }
            i += 1;
        }
    }
    // the encoding is exhausted: `raw` matches only if it is exhausted too
    (i == raw.len(), i == raw.len())
}

/// Undo log for a single byte, used by the trie walk to roll a shared machine back instead of
/// cloning it per node. Frames are only ever mutated *after* being popped, so logging the frames
/// that were popped (first occurrence per index) plus the deepest index touched is a complete
/// record of the change.
pub(super) struct Roll {
    orig_len: usize,
    frames: Vec<(usize, Frame)>,
}

impl Roll {
    pub(super) fn new(orig_len: usize) -> Roll {
        Roll { orig_len, frames: Vec::new() }
    }
    pub(super) fn note_pop(&mut self, idx: usize, f: &Frame) {
        if idx >= self.orig_len {
            // Pushed earlier during this same byte — truncation removes it.
            return;
        }
        if self.frames.iter().any(|(i, _)| *i == idx) {
            return;
        }
        self.frames.push((idx, f.clone()));
    }
    pub(super) fn restore(&self, m: &mut Machine) {
        if self.frames.is_empty() {
            return;
        }
        let base = self.frames.iter().map(|(i, _)| *i).min().unwrap_or(0);
        m.stack.truncate(base);
        let mut sorted: Vec<&(usize, Frame)> = self.frames.iter().collect();
        sorted.sort_by_key(|(i, _)| *i);
        for (_, f) in sorted {
            m.stack.push(f.clone());
        }
    }
}
