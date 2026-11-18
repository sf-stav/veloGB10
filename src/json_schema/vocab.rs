//! The vocabulary trie `allowed()` walks: token id -> raw bytes, string-run summaries, pruning on the first refused byte.
//! Immutable after build; shared through `VocabTable` (`mask.rs`). Map of the directory in `mod.rs`.

use super::*;

// ===========================================================================================
// Vocabulary trie
// ===========================================================================================

#[derive(Default, Debug)]
pub(super) struct TrieNode {
    pub(super) children: Vec<(u8, usize)>,
    pub(super) token: Option<u32>,
}

#[derive(Default, Debug)]
pub(super) struct Vocab {
    pub(super) nodes: Vec<TrieNode>,
    pub(super) by_id: HashMap<u32, Vec<u8>>,
    /// Longest piece in bytes: an upper bound on the characters one token adds to a string.
    pub(super) max_piece: usize,
    /// Per token id, fed from the interior of a value string (no pending escape or UTF-8
    /// sequence): the characters it adds before the string closes or the piece ends, with
    /// [`STR_RUN_CLOSES`] set when it closes the string. [`STR_RUN_BAD`] when the bytes are not
    /// valid string content (such a token is never in a base mask, so it is never consulted).
    pub(super) str_run: Vec<u16>,
}

pub(super) const STR_RUN_CLOSES: u16 = 0x8000;
pub(super) const STR_RUN_BAD: u16 = 0xFFFF;

/// 1 while a started escape or UTF-8 sequence has not produced its character yet.
pub(super) fn pending_char(st: &StrState) -> u64 {
    u64::from(!st.is_closed_interior())
}

/// [`Vocab::str_run`] entry for one piece.
pub(super) fn str_run_of(bytes: &[u8]) -> u16 {
    let mut st = StrState::new();
    for &b in bytes {
        match str_advance(&mut st, b, false) {
            Step::Open => {}
            Step::Closed => return (st.cp.min(0x7FFE) as u16) | STR_RUN_CLOSES,
            Step::Bad => return STR_RUN_BAD,
        }
    }
    ((st.cp as u64 + pending_char(&st)).min(0x7FFE)) as u16
}

impl Vocab {
    /// Build the byte trie. Empty pieces are skipped: a token with no bytes can never advance the
    /// machine, so it is never allowed (see `step`).
    pub(super) fn build(pieces: Vec<(u32, Vec<u8>)>) -> (Vocab, usize) {
        let mut v = Vocab::default();
        v.nodes.push(TrieNode::default());
        let mut vocab_size = 0usize;
        for (id, bytes) in pieces {
            vocab_size = vocab_size.max(id as usize + 1);
            if bytes.is_empty() {
                continue;
            }
            v.max_piece = v.max_piece.max(bytes.len());
            v.by_id.insert(id, bytes.clone());
            let mut cur = 0usize;
            for &b in &bytes {
                let nxt = match v.nodes[cur].children.iter().find(|(c, _)| *c == b) {
                    Some(&(_, n)) => n,
                    None => {
                        v.nodes.push(TrieNode::default());
                        let n = v.nodes.len() - 1;
                        v.nodes[cur].children.push((b, n));
                        n
                    }
                };
                cur = nxt;
            }
            v.nodes[cur].token = Some(id); // duplicate piece: last id wins
        }
        v.str_run = vec![STR_RUN_BAD; vocab_size];
        for (&id, bytes) in &v.by_id {
            v.str_run[id as usize] = str_run_of(bytes);
        }
        (v, vocab_size)
    }
}

pub(super) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A poisoned lock means a previous panic; the bitsets/states are still structurally valid, so
    // keep serving instead of panicking a second time.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Default, Debug)]
pub(super) struct Interner {
    pub(super) states: Vec<Arc<Machine>>,
    pub(super) index: HashMap<Machine, u32>,
}
