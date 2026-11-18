//! Streaming byte decoder: reassembles multi-byte UTF-8 characters split across byte-level tokens, so a per-token
//! decode never yields U+FFFD for an emoji. Also the stream tables and the legacy decoder. Map in `mod.rs`.

use super::*;

/// GPT-2-style `bytes_to_unicode` INVERSE table (mapped char -> raw byte), mirroring the
/// `tokenizers` crate's CHAR_BYTES. Qwen byte-level BPE vocabs store raw bytes as single
/// mapped chars (e.g. 'Ā' -> 0x00, 'ð' -> 0xF0). The crate's ByteLevel decoder maps them
/// back but decodes the byte stream with `String::from_utf8_lossy` — so a multi-byte UTF-8
/// char split across tokens (every emoji: 2-4 byte-mapped tokens) becomes one U+FFFD "�"
/// per fragment in any per-token decode. `StreamByteDecoder` uses this table to reassemble
/// the raw bytes and decode only complete sequences.
pub(super) fn bytes_to_unicode_inverse() -> std::collections::HashMap<char, u8> {
    let mut inv = std::collections::HashMap::with_capacity(256);
    let mut bs: Vec<u32> = Vec::with_capacity(256);
    let mut cs: Vec<u32> = Vec::with_capacity(256);
    bs.extend(33..=126); bs.extend(161..=172); bs.extend(174..=255);
    cs.extend(33..=126); cs.extend(161..=172); cs.extend(174..=255);
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) { bs.push(b); cs.push(256 + n); n += 1; }
    }
    for (b, c) in bs.iter().zip(cs.iter()) {
        if let Some(c) = char::from_u32(*c) { inv.insert(c, *b as u8); }
    }
    inv
}

/// Raw bytes a vocab piece decodes to under byte-level BPE: byte-mapped chars -> their byte
/// (any non-mapped char in the piece -> the piece's UTF-8 bytes, the crate's fallback), plus
/// the `<0xXX>` byte-fallback form for tokenizers that ship it.
pub(super) fn piece_bytes(piece: &str, inv: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
        if let Ok(b) = u8::from_str_radix(&piece[3..5], 16) { return vec![b]; }
    }
    let mut bytes = Vec::with_capacity(piece.len());
    for c in piece.chars() {
        match inv.get(&c) {
            Some(b) => bytes.push(*b),
            None => return piece.as_bytes().to_vec(),
        }
    }
    bytes
}

/// Special-token ids from tokenizer.json's `added_tokens` (honoring the `special` flag, the
/// crate's skip_special_tokens semantics). Empty on any parse failure (never fires in
/// practice — the crate parsed the same file).
pub(super) fn special_ids(tokenizer_path: &str) -> std::collections::HashSet<u32> {
    let mut set = std::collections::HashSet::new();
    if let Ok(raw) = std::fs::read(tokenizer_path) {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&raw) {
            if let Some(arr) = v["added_tokens"].as_array() {
                for tok in arr {
                    if tok["special"].as_bool().unwrap_or(false) {
                        if let Some(id) = tok["id"].as_u64() { set.insert(id as u32); }
                    }
                }
            }
        }
    }
    set
}

/// The tokenizer.json the stream decoder reads its special set from: `<model_dir>/tokenizer.json`
/// (the legacy `stream_decoder()` expression, unchanged).
pub(super) fn stream_json_path(model_dir: Option<&Path>) -> String {
    model_dir.map(|d| d.join("tokenizer.json").to_string_lossy().into_owned()).unwrap_or_default()
}

impl TokCache {
    /// Everything the request path used to rebuild per call, computed once from ONE
    /// `get_vocab(true)` build + ONE tokenizer.json parse (the same `special_ids` function).
    pub(super) fn build(tokenizer: &Tokenizer, json_path: &str) -> Self {
        let t0 = std::time::Instant::now();
        let vocab = tokenizer.get_vocab(true);
        let think_hy = vocab.contains_key("</think:opensource>");
        let eos_id = vocab.get("<|endoftext|>").copied().unwrap_or(151643) as u32;
        // The crate's own call (not vocab.len()), so the cached value is the legacy value by
        // construction whatever the crate version computes it as.
        let vocab_size = tokenizer.get_vocab_size(true);
        let stream = StreamTables::build(tokenizer, &vocab, &special_ids(json_path));
        if crate::boot_report::debug() {
            crate::deprintln!("[tokenizer] WP01 tables cached at load: vocab={} ids={} specials={} dup_ids={} \
                   think_hy={} eos={} ({:.1} ms)",
                  vocab.len(), stream.kind.len(), stream.n_special, stream.dup_ids, think_hy, eos_id,
                  t0.elapsed().as_secs_f64() * 1e3);
        }
        Self { think_hy, eos_id, vocab_size, stream: std::sync::Arc::new(stream) }
    }
}

/// WP01: the stream decoder's per-id tables, built once at load and shared (`Arc`) by every
/// request's `StreamByteDecoder`. Id `i` is:
///   - `K_SPECIAL`: in the tokenizer.json `added_tokens` special set -> push returns "" (legacy
///     checked `specials` first, before the vocab lookup — same order here);
///   - `K_PIECE`:  its raw bytes are `bytes[off[i]..off[i+1]]` = `piece_bytes(piece, inv)`, the
///     exact function the legacy decoder applied per push;
///   - `K_NONE`:   not in the vocab -> push returns "" without touching `pending` (legacy
///     `vocab.get(&id) == None`). Ids past the table behave the same.
/// The id->piece map is the inversion of `get_vocab(true)` (string->id), as the legacy decoder
/// built it. If two strings share an id (`dup_ids > 0`) the legacy HashMap inversion kept
/// whichever it iterated last — per-process random; here the tie resolves deterministically to
/// the crate's `id_to_token` (what `decode()` uses). The served Qwen vocab has dup_ids = 0.
pub(super) struct StreamTables {
    bytes: Vec<u8>,
    off: Vec<u32>,
    pub(super) kind: Vec<u8>,
    n_special: usize,
    pub(super) dup_ids: usize,
}

const K_NONE: u8 = 0;
pub(super) const K_PIECE: u8 = 1;
pub(super) const K_SPECIAL: u8 = 2;

impl StreamTables {
    pub(super) fn build(tokenizer: &Tokenizer, vocab: &std::collections::HashMap<String, u32>,
             specials: &std::collections::HashSet<u32>) -> Self {
        let inv = bytes_to_unicode_inverse();
        let n = vocab.values().max().map_or(0, |&m| m as usize + 1);
        let mut by_id: Vec<Option<&str>> = vec![None; n];
        let mut dup_ids = 0usize;
        for (s, &id) in vocab {
            let slot = &mut by_id[id as usize];
            if slot.is_some() {
                dup_ids += 1;
                if tokenizer.id_to_token(id).as_deref() != Some(s.as_str()) { continue; }
            }
            *slot = Some(s.as_str());
        }
        let mut bytes: Vec<u8> = Vec::with_capacity(n * 8);
        let mut off: Vec<u32> = Vec::with_capacity(n + 1);
        let mut kind: Vec<u8> = Vec::with_capacity(n);
        let mut n_special = 0usize;
        off.push(0);
        for (id, piece) in by_id.iter().enumerate() {
            if specials.contains(&(id as u32)) {
                kind.push(K_SPECIAL);
                n_special += 1;
            } else if let Some(p) = piece {
                kind.push(K_PIECE);
                bytes.extend(piece_bytes(p, &inv));
            } else {
                kind.push(K_NONE);
            }
            off.push(bytes.len() as u32);
        }
        Self { bytes, off, kind, n_special, dup_ids }
    }

    /// The raw bytes a push of `id` appends, or `None` when the push is a no-op returning ""
    /// (special, or not in the vocab).
    #[inline]
    pub(super) fn piece(&self, id: u32) -> Option<&[u8]> {
        let i = id as usize;
        if self.kind.get(i).copied() != Some(K_PIECE) { return None; }
        Some(&self.bytes[self.off[i] as usize..self.off[i + 1] as usize])
    }
}

/// Incremental byte-level stream decoder: reassembles multi-byte UTF-8 chars split across
/// tokens by accumulating RAW bytes and emitting only complete UTF-8 sequences, holding back
/// the ≤3-byte tail. Replaces the lossy per-token `decode(&[t], true)` in the SSE streaming
/// path (server.rs) — without it, "That's wonderful to hear! 😀" arrives as "�" per byte.
/// Applies to ALL qwen byte-level BPE models via the shared server path.
///
/// WP01: backed by the load-time `StreamTables` (a request allocates only `pending`); the
/// legacy per-request-rebuild decoder stays reachable (`--wp01-off=1`) and, under
/// `--wp01-xcheck=1`, runs in lock-step as a shadow with every push/finish compared.
pub struct StreamByteDecoder {
    imp: StreamImpl,
    shadow: Option<Box<LegacyStreamDecoder>>,
    pushes: u64,
    xmis: u64,
}

enum StreamImpl {
    Cached { tables: std::sync::Arc<StreamTables>, pending: Vec<u8> },
    Legacy(LegacyStreamDecoder),
}

impl StreamByteDecoder {
    /// The legacy constructor: rebuilds the id->piece map and re-parses tokenizer.json.
    pub fn new(tokenizer: &Tokenizer, tokenizer_path: &str) -> Self {
        Self { imp: StreamImpl::Legacy(LegacyStreamDecoder::new(tokenizer, tokenizer_path)),
               shadow: None, pushes: 0, xmis: 0 }
    }

    pub(super) fn from_tables(tables: std::sync::Arc<StreamTables>, shadow: Option<Box<LegacyStreamDecoder>>) -> Self {
        Self { imp: StreamImpl::Cached { tables, pending: Vec::with_capacity(16) },
               shadow, pushes: 0, xmis: 0 }
    }

    /// Feed one generated token id; returns the decodable text (an incomplete trailing
    /// UTF-8 char is held back for the next token). Special tokens are skipped, matching
    /// `decode(ids, true)`.
    pub fn push(&mut self, id: u32) -> String {
        let out = match &mut self.imp {
            StreamImpl::Legacy(d) => d.push(id),
            StreamImpl::Cached { tables, pending } => {
                let Some(piece) = tables.piece(id) else { return self.shadow_check(id, String::new()) };
                pending.extend_from_slice(piece);
                // Emit the longest complete UTF-8 prefix; keep the trailing ≤3 bytes that could
                // be a truncated char. (Verbatim the legacy loop.)
                let mut out = String::new();
                for cut in 0..=3.min(pending.len()) {
                    let end = pending.len() - cut;
                    if let Ok(s) = std::str::from_utf8(&pending[..end]) {
                        out = s.to_string();
                        pending.drain(..end);
                        break;
                    }
                }
                out
            }
        };
        self.shadow_check(id, out)
    }

    #[inline]
    fn shadow_check(&mut self, id: u32, out: String) -> String {
        if let Some(sh) = self.shadow.as_mut() {
            self.pushes += 1;
            let old = sh.push(id);
            if old != out {
                self.xmis += 1;
                if self.xmis <= 8 {
                    eprintln!("[wp01-xcheck] MISMATCH push#{} id={id}: new={out:?} old={old:?}", self.pushes);
                }
            }
        }
        out
    }

    /// Flush the held-back tail at stream end; invalid bytes become U+FFFD (the crate's
    /// lossy semantics for a genuinely truncated sequence).
    pub fn finish(&mut self) -> String {
        let out = match &mut self.imp {
            StreamImpl::Legacy(d) => d.finish(),
            StreamImpl::Cached { pending, .. } => {
                let out = String::from_utf8_lossy(pending).to_string();
                pending.clear();
                out
            }
        };
        if let Some(sh) = self.shadow.as_mut() {
            let old = sh.finish();
            if old != out { self.xmis += 1; }
            eprintln!("[wp01-xcheck] stream decoder: pushes={} mismatches={}{}", self.pushes, self.xmis,
                      if old != out { format!(" (finish: new={out:?} old={old:?})") } else { String::new() });
        }
        out
    }
}

/// The pre-WP01 decoder, kept verbatim: per-request `get_vocab(true)` inversion + tokenizer.json
/// re-parse. Active under `--wp01-off=1`; the shadow reference under `--wp01-xcheck=1`.
pub(super) struct LegacyStreamDecoder {
    pub(super) inv: std::collections::HashMap<char, u8>,
    pub(super) vocab: std::collections::HashMap<u32, String>,
    pub(super) specials: std::collections::HashSet<u32>,
    pending: Vec<u8>,
}

impl LegacyStreamDecoder {
    pub(super) fn new(tokenizer: &Tokenizer, tokenizer_path: &str) -> Self {
        let vocab = tokenizer.get_vocab(true).into_iter()
            .map(|(s, id)| (id as u32, s)).collect();
        Self {
            inv: bytes_to_unicode_inverse(),
            vocab,
            specials: special_ids(tokenizer_path),
            pending: Vec::with_capacity(16),
        }
    }

    /// Feed one generated token id; returns the decodable text (an incomplete trailing
    /// UTF-8 char is held back for the next token). Special tokens are skipped, matching
    /// `decode(ids, true)`.
    pub fn push(&mut self, id: u32) -> String {
        if self.specials.contains(&id) { return String::new(); }
        let piece = match self.vocab.get(&id) { Some(p) => p.clone(), None => return String::new() };
        self.pending.extend(piece_bytes(&piece, &self.inv));
        // Emit the longest complete UTF-8 prefix; keep the trailing ≤3 bytes that could be
        // a truncated char.
        for cut in 0..=3.min(self.pending.len()) {
            let end = self.pending.len() - cut;
            if let Ok(s) = std::str::from_utf8(&self.pending[..end]) {
                let out = s.to_string();
                self.pending.drain(..end);
                return out;
            }
        }
        String::new()
    }

    /// Flush the held-back tail at stream end; invalid bytes become U+FFFD (the crate's
    /// lossy semantics for a genuinely truncated sequence).
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).to_string();
        self.pending.clear();
        out
    }
}
