//! `SchemaMask` and `VocabTable`: the public constrained-decoding API (states, `allowed`, `step`, completion, defer-until,
//! summary). Another `impl SchemaMask` block with the internals lives in `walk.rs` (map in `mod.rs`).

use super::*;

// ===========================================================================================
// Public API
// ===========================================================================================

/// A compiled schema: byte-level machine + token-level allowed sets, lazily built.
#[derive(Debug)]
pub struct SchemaMask {
    pub(super) schemas: Vec<Schema>,
    pub(super) root: u32,
    /// Index 0 is always the universal permissive schema (`Ty::Any`), used for additional
    /// properties and for `{"type":"array"}` without `items`.
    pub(super) start: u32,
    pub(super) interner: Mutex<Interner>,
    /// `state -> allowed bitset`. Computed on demand; states are interned so identical machine
    /// snapshots share one entry.
    pub(super) cache: Mutex<HashMap<u32, Arc<Vec<u32>>>>,
    /// `mask key -> allowed bitset` (see [`SchemaMask::mask_key`]): states that differ only in a
    /// string's character count, where the count cannot change the allowed set, share one entry.
    /// Without it every token inside a free-text string is a new state and a full trie walk.
    pub(super) mask_cache: Mutex<HashMap<Machine, Arc<Vec<u32>>>>,
    /// Shared, immutable byte trie of the model vocabulary (see [`VocabTable`]): built once per
    /// process and handed to every request's mask, instead of rebuilding a 248K-entry trie per
    /// request.
    pub(super) vocab: Arc<Vocab>,
    pub(super) vocab_size: usize,
    pub(super) words: usize,
    /// When `Some(tok)`, the mask is NOT applied until the lane has emitted `tok` (the model's
    /// think-close token): a request whose prompt opens a reasoning block reasons freely, and
    /// only the answer after the block is constrained. `None` = constrained from the first token.
    pub(super) defer_until: Option<u32>,
}

/// A model vocabulary prepared for constrained decoding: token id -> raw output bytes plus the
/// byte trie `allowed()` walks. Immutable after `build`; share it with `Arc` across requests and
/// attach it with [`SchemaMask::set_vocab_table`].
#[derive(Debug, Clone)]
pub struct VocabTable {
    vocab: Arc<Vocab>,
    size: usize,
}

impl VocabTable {
    /// `pieces` = (token id, the exact bytes that token adds to the output text). Special /
    /// control tokens must be left out (or given empty bytes): they are never allowed.
    pub fn build(pieces: Vec<(u32, Vec<u8>)>) -> Self {
        let (v, size) = Vocab::build(pieces);
        VocabTable { vocab: Arc::new(v), size }
    }

    /// Number of token ids covered (max id + 1).
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl SchemaMask {
    /// The start state id.
    pub fn start_state(&self) -> u32 {
        self.start
    }

    /// Allowed-token bitset for `state`: `vocab_words(vocab_size)` little-endian `u32` words,
    /// bit `t` set = token `t` is allowed. Cached; safe to call from several threads.
    ///
    /// Returns an all-zero bitset if `set_vocab` was never called (`fail closed`: nothing is
    /// emitted rather than unconstrained text).
    pub fn allowed(&self, state: u32) -> Arc<Vec<u32>> {
        if let Some(hit) = lock(&self.cache).get(&state).cloned() {
            return hit;
        }
        let arc = match self.machine(state) {
            Some(m) => self.allowed_for(&m),
            None => Arc::new(vec![0u32; self.words]),
        };
        lock(&self.cache).insert(state, arc.clone());
        arc
    }

    /// Allowed set of a machine snapshot, shared through `mask_cache` by [`SchemaMask::mask_key`].
    fn allowed_for(&self, m: &Machine) -> Arc<Vec<u32>> {
        let key = self.mask_key(m);
        if let Some(hit) = lock(&self.mask_cache).get(&key).cloned() {
            return hit;
        }
        let a = Arc::new(match self.allowed_len_bounded(m) {
            Some(bits) => bits,
            None => self.compute_allowed_machine(m),
        });
        lock(&self.mask_cache).insert(key, a.clone());
        a
    }

    /// Inside a value string whose minLength / maxLength can refuse the next token, derive the
    /// allowed set from the length-free set of the same position instead of walking the trie.
    ///
    /// The string's character count only gates acceptance: it never changes what the bytes
    /// parse to. It only grows while a token is fed, and a token is refused by it exactly when
    /// the count after its last in-string byte exceeds maxLength, or the string closes below
    /// minLength. From an interior state (no pending escape or UTF-8 sequence) the characters a
    /// token adds depend only on its bytes ([`Vocab::str_run`]), so
    ///
    /// `allowed = base ∩ { t : added(t) <= room  and  (closes(t) ⇒ added(t) >= to_min) }`
    ///
    /// where `base` ignores this string's bounds. `base` is one cached mask per position, so the
    /// last characters before maxLength (and the first before minLength) cost a bitset pass each
    /// instead of a full trie walk. `None` when the shortcut does not apply.
    fn allowed_len_bounded(&self, m: &Machine) -> Option<Vec<u32>> {
        let Some(Frame::Str { si, st }) = m.stack.last() else { return None };
        let s = &self.schemas[*si as usize];
        if !s.enum_str.is_empty() || st.free_len || !st.is_closed_interior() || self.vocab.str_run.is_empty() {
            return None;
        }
        if s.min_len.is_none() && s.max_len.is_none() {
            return None;
        }
        let cp = st.cp as u64;
        let to_min = s.min_len.map_or(0, |lo| lo.saturating_sub(cp));
        let room = s.max_len.map_or(u64::MAX, |hi| hi.saturating_sub(cp));
        let mut free = m.clone();
        if let Some(Frame::Str { st, .. }) = free.stack.last_mut() {
            st.free_len = true;
            st.cp = 0;
        }
        let base = self.allowed_for(&free);
        let run = &self.vocab.str_run;
        let mut bits = vec![0u32; base.len()];
        for (w, &word) in base.iter().enumerate() {
            let mut rest = word;
            while rest != 0 {
                let b = rest.trailing_zeros();
                rest &= rest - 1;
                let t = (w << 5) | b as usize;
                let r = run.get(t).copied().unwrap_or(STR_RUN_BAD);
                if r == STR_RUN_BAD {
                    continue;
                }
                let added = (r & !STR_RUN_CLOSES) as u64;
                if added <= room && (r & STR_RUN_CLOSES == 0 || added >= to_min) {
                    bits[w] |= 1u32 << b;
                }
            }
        }
        Some(bits)
    }

    /// The machine with every value string's character count replaced by what the next token
    /// can observe of it: the distance to `minLength` and the room left below `maxLength`, each
    /// capped at one more than the longest piece (a token adds at most that many characters).
    /// Two states with the same key allow exactly the same tokens. Strings whose content is
    /// kept (enum values) are left exact.
    fn mask_key(&self, m: &Machine) -> Machine {
        let cap = self.vocab.max_piece as u64 + 1;
        let mut k = m.clone();
        for f in k.stack.iter_mut() {
            if let Frame::Str { si, st } = f {
                let s = &self.schemas[*si as usize];
                if !s.enum_str.is_empty() || st.free_len {
                    continue;
                }
                let cp = st.cp as u64;
                let to_min = s.min_len.map_or(0, |lo| lo.saturating_sub(cp).min(cap));
                let room = s.max_len.map_or(cap, |hi| hi.saturating_sub(cp).min(cap));
                st.cp = ((to_min << 16) | room) as u32;
            }
        }
        k
    }

    /// Advance the machine by ONE EMITTED TOKEN. `None` when the token is unknown, empty, or its
    /// bytes are refused in `state`.
    pub fn step(&self, state: u32, tok: u32) -> Option<u32> {
        let bytes = self.vocab.by_id.get(&tok)?;
        if bytes.is_empty() {
            return None;
        }
        let mut m = (*self.machine(state)?).clone();
        if m.is_dead() {
            return None;
        }
        for &b in bytes {
            if !self.feed(&mut m, b, None) {
                return None;
            }
        }
        Some(self.intern(m))
    }

    /// True when the JSON value is finished and only trailing whitespace (or end) may follow.
    /// A pending number is finalized here, so `42` counts as finished even though the frame is
    /// still open until the next byte.
    pub fn is_complete(&self, state: u32) -> bool {
        let Some(stored) = self.machine(state) else { return false };
        if stored.is_dead() {
            return false;
        }
        let mut m = (*stored).clone();
        if !self.finalize_pending_number(&mut m) {
            return false;
        }
        m.at_doc_end()
    }

    /// True when a state is DEAD (used to reject after a broken draft chain).
    pub fn is_dead(&self, state: u32) -> bool {
        self.machine(state).map(|m| m.is_dead()).unwrap_or(true)
    }

    /// Set the token-id → piece-bytes table (built from the model tokenizer). MUST be called once
    /// before `allowed()`; also fixes `vocab_size`/`words`. Cached bitsets are dropped because they
    /// were computed against the previous vocabulary.
    pub fn set_vocab(&mut self, pieces: Vec<(u32, Vec<u8>)>) {
        self.set_vocab_table(&VocabTable::build(pieces));
    }

    /// `set_vocab` with a prebuilt, shared table (no per-request trie build).
    pub fn set_vocab_table(&mut self, table: &VocabTable) {
        self.vocab = table.vocab.clone();
        self.vocab_size = table.size;
        self.words = vocab_words(table.size);
        lock(&self.cache).clear();
        lock(&self.mask_cache).clear();
    }

    /// Defer the constraint until the lane emits `tok` (see the `defer_until` field).
    pub fn set_defer_until(&mut self, tok: Option<u32>) {
        self.defer_until = tok;
    }

    /// The token that arms the constraint, if the constraint is deferred.
    pub fn defer_until(&self) -> Option<u32> {
        self.defer_until
    }

    /// A bitset with every token disallowed EXCEPT the model's end-of-generation tokens: when the
    /// document is complete, generation must stop rather than run on. End tokens are deliberately
    /// never produced by `allowed()`.
    pub fn end_mask(&self, stop_ids: &[u32]) -> Arc<Vec<u32>> {
        let words = if self.words > 0 {
            self.words
        } else {
            // Vocab not attached yet: size the bitset to the stop ids so the caller still gets a
            // usable mask instead of an empty one.
            vocab_words(stop_ids.iter().map(|&s| s as usize + 1).max().unwrap_or(0))
        };
        let mut bits = vec![0u32; words];
        for &s in stop_ids {
            let t = s as usize;
            if t < words * 32 {
                bits[t >> 5] |= 1u32 << (t & 31);
            }
        }
        Arc::new(bits)
    }

    /// Human-readable summary for logs.
    pub fn summary(&self) -> String {
        let root = &self.schemas[self.root as usize];
        let max_depth = self.schemas.iter().map(|s| s.depth).max().unwrap_or(0);
        let enum_values: usize = self.schemas.iter().map(|s| s.enum_str.len() + s.enum_int.len()).sum();
        let states = lock(&self.interner).states.len();
        let cached = lock(&self.cache).len();
        let vocab = if self.vocab_size == 0 {
            "unset".to_string()
        } else {
            format!("{} tokens/{} words", self.vocab_size, self.words)
        };
        format!(
            "json_schema: root={} properties={} required={} additionalProperties={} items={} enum_values={} schemas={} max_depth={} states={} cached_masks={} vocab={}",
            root.ty.name(),
            root.props.len(),
            root.required.len(),
            root.additional,
            root.items.is_some(),
            enum_values,
            self.schemas.len(),
            max_depth,
            states,
            cached,
            vocab,
        )
    }
}
