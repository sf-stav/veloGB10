//! Byte sub-machines for the inside of a JSON string (escapes, UTF-8, `StrState`) and of a number (`NumState`).
//! Used by `Machine` (see `machine.rs`); per the module contract no panics on user input: every refusal is an `Err`/`None`/zeroed bitset.

// ===========================================================================================
// Byte machine: string / number / literal sub-states
// ===========================================================================================

/// Escape-sequence position inside a JSON string.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Esc {
    None,
    /// Just consumed `\`.
    Bslash,
    /// Inside `\uXXXX`, `n` hex digits collected in `acc`.
    Hex { n: u8, acc: u32 },
    /// High surrogate seen, waiting for `\`.
    WantLowBslash { hi: u32 },
    /// High surrogate seen, waiting for `u`.
    WantLowU { hi: u32 },
    /// Inside the low half `\uYYYY` of a surrogate pair.
    LowHex { n: u8, acc: u32, hi: u32 },
}

/// Incremental JSON-string state. `dec` holds the *decoded* bytes and is only kept when a prefix
/// constraint (enum values, property names) needs it — that keeps the buffers bounded by the
/// longest candidate instead of by the emitted document.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) struct StrState {
    esc: Esc,
    pub(super) dec: Vec<u8>,
    /// The raw bytes consumed inside the string (kept with `dec`): enum values and declared keys
    /// are matched in their canonical spelling (see [`canon_match`]).
    pub(super) raw: Vec<u8>,
    pub(super) cp: u32,
    /// Trailing bytes of a not-yet-complete UTF-8 sequence (a token can split a code point).
    pend: Vec<u8>,
    /// Skip this string's minLength / maxLength checks. Only set on the machine used to build a
    /// length-free base mask (see [`SchemaMask::allowed_len_bounded`]); never on a live state.
    pub(super) free_len: bool,
}

impl StrState {
    pub(super) fn new() -> StrState {
        StrState { esc: Esc::None, dec: Vec::new(), raw: Vec::new(), cp: 0, pend: Vec::new(), free_len: false }
    }
    pub(super) fn is_closed_interior(&self) -> bool {
        self.esc == Esc::None && self.pend.is_empty()
    }
}

pub(super) enum Step {
    Open,
    Closed,
    Bad,
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn push_cp(st: &mut StrState, cp: u32, keep: bool) -> bool {
    let Some(c) = char::from_u32(cp) else { return false };
    let mut buf = [0u8; 4];
    let enc = c.encode_utf8(&mut buf);
    st.cp += 1;
    if keep {
        st.dec.extend_from_slice(enc.as_bytes());
    }
    true
}

/// Append one raw (non-escape) content byte, validating UTF-8 across token boundaries.
fn push_raw(st: &mut StrState, b: u8, keep: bool) -> bool {
    if b < 0x80 {
        st.cp += 1;
        if keep {
            st.dec.push(b);
        }
        return true;
    }
    st.pend.push(b);
    // The borrow of `pend` ends inside `map`, so the arms may mutate `st`.
    let decoded = std::str::from_utf8(&st.pend).map(|s| (s.chars().count() as u32, s.as_bytes().to_vec()));
    match decoded {
        Ok((n, bytes)) => {
            st.pend.clear();
            st.cp += n;
            if keep {
                st.dec.extend_from_slice(&bytes);
            }
            true
        }
        Err(e) => e.error_len().is_none() && st.pend.len() < 4,
    }
}

/// Advance one byte inside a string. `keep` mirrors the caller's prefix-constraint need.
pub(super) fn str_advance(st: &mut StrState, b: u8, keep: bool) -> Step {
    if keep && !(st.esc == Esc::None && b == b'"') {
        st.raw.push(b);
    }
    match st.esc {
        // An ASCII byte (including `"` and `\`) while a multi-byte UTF-8 sequence is still
        // incomplete would make the string invalid UTF-8.
        Esc::None if !st.pend.is_empty() && b < 0x80 => Step::Bad,
        Esc::None => match b {
            b'"' => Step::Closed,
            b'\\' => {
                st.esc = Esc::Bslash;
                Step::Open
            }
            // Raw control characters must be escaped in JSON.
            0x00..=0x1F => Step::Bad,
            _ => {
                if push_raw(st, b, keep) {
                    Step::Open
                } else {
                    Step::Bad
                }
            }
        },
        Esc::Bslash => {
            let cp = match b {
                b'"' => 0x22,
                b'\\' => 0x5C,
                b'/' => 0x2F,
                b'b' => 0x08,
                b'f' => 0x0C,
                b'n' => 0x0A,
                b'r' => 0x0D,
                b't' => 0x09,
                b'u' => {
                    st.esc = Esc::Hex { n: 0, acc: 0 };
                    return Step::Open;
                }
                _ => return Step::Bad,
            };
            st.esc = Esc::None;
            if push_cp(st, cp, keep) {
                Step::Open
            } else {
                Step::Bad
            }
        }
        Esc::Hex { n, acc } => {
            let Some(d) = hex_val(b) else { return Step::Bad };
            let acc = acc * 16 + d as u32;
            if n + 1 < 4 {
                // `\uDC`..`\uDF` can only become a lone low surrogate: refuse it now, while
                // another byte can still follow, instead of at the fourth digit
                if n + 1 == 2 && (0xDC..=0xDF).contains(&acc) {
                    return Step::Bad;
                }
                st.esc = Esc::Hex { n: n + 1, acc };
                return Step::Open;
            }
            st.esc = Esc::None;
            if (0xD800..=0xDBFF).contains(&acc) {
                st.esc = Esc::WantLowBslash { hi: acc };
                return Step::Open;
            }
            if (0xDC00..=0xDFFF).contains(&acc) {
                // Lone low surrogate: serde_json rejects it too, so we must not emit it.
                return Step::Bad;
            }
            if push_cp(st, acc, keep) {
                Step::Open
            } else {
                Step::Bad
            }
        }
        Esc::WantLowBslash { hi } => {
            if b != b'\\' {
                return Step::Bad;
            }
            st.esc = Esc::WantLowU { hi };
            Step::Open
        }
        Esc::WantLowU { hi } => {
            if b != b'u' {
                return Step::Bad;
            }
            st.esc = Esc::LowHex { n: 0, acc: 0, hi };
            Step::Open
        }
        Esc::LowHex { n, acc, hi } => {
            let Some(d) = hex_val(b) else { return Step::Bad };
            let acc = acc * 16 + d as u32;
            if n + 1 < 4 {
                // the second half of a pair must be `\uDC00`..`\uDFFF`: refuse a prefix that
                // cannot become one
                if (n + 1 == 1 && acc != 0xD) || (n + 1 == 2 && !(0xDC..=0xDF).contains(&acc)) {
                    return Step::Bad;
                }
                st.esc = Esc::LowHex { n: n + 1, acc, hi };
                return Step::Open;
            }
            st.esc = Esc::None;
            if !(0xDC00..=0xDFFF).contains(&acc) {
                return Step::Bad;
            }
            let cp = 0x10000 + ((hi - 0xD800) << 10) + (acc - 0xDC00);
            if push_cp(st, cp, keep) {
                Step::Open
            } else {
                Step::Bad
            }
        }
    }
}

/// JSON number grammar position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum NumState {
    /// Saw `-`, a digit must follow.
    NegStart,
    Zero,
    Int,
    FracDot,
    Frac,
    ExpChar,
    ExpSign,
    Exp,
}

impl NumState {
    pub(super) fn terminal(self) -> bool {
        matches!(self, NumState::Zero | NumState::Int | NumState::Frac | NumState::Exp)
    }
}

pub(super) fn num_start(b: u8) -> Option<(NumState, Vec<u8>)> {
    match b {
        b'-' => Some((NumState::NegStart, vec![b'-'])),
        b'0' => Some((NumState::Zero, vec![b'0'])),
        b'1'..=b'9' => Some((NumState::Int, vec![b])),
        _ => None,
    }
}

pub(super) fn num_advance(st: NumState, b: u8) -> Option<NumState> {
    use NumState::*;
    match (st, b) {
        (NegStart, b'0') => Some(Zero),
        (NegStart, b'1'..=b'9') => Some(Int),
        (Zero, b'.') => Some(FracDot),
        (Zero, b'e' | b'E') => Some(ExpChar),
        (Int, b'0'..=b'9') => Some(Int),
        (Int, b'.') => Some(FracDot),
        (Int, b'e' | b'E') => Some(ExpChar),
        (FracDot, b'0'..=b'9') => Some(Frac),
        (Frac, b'0'..=b'9') => Some(Frac),
        (Frac, b'e' | b'E') => Some(ExpChar),
        (ExpChar, b'0'..=b'9') => Some(Exp),
        (ExpChar, b'+' | b'-') => Some(ExpSign),
        (ExpSign, b'0'..=b'9') => Some(Exp),
        (Exp, b'0'..=b'9') => Some(Exp),
        _ => None,
    }
}

pub(super) fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}
