//! The Engram compressed token map: port of `build_compressed_token_map` (`inference/engram.py`).
//!
//! Every token id maps onto a smaller id space where tokens that normalise alike collapse together
//! (" The", "the" and "THE" share one id). N-grams are hashed over these ids, and every hash multiplier
//! is derived from the size of this space (99,092 for V4.1), so a single difference here silently
//! rehashes the whole table. The normaliser is built from the same `tokenizers` crate components the
//! Python binding wraps:
//!
//! `NFKC, NFD, StripAccents, Lowercase, Replace(/[ \t\r\n]+/, " "), Replace(/^ $/, U+E000), Strip,
//! Replace("\u{e000}", " ")`
//!
//! applied to the raw single-token decode (`decode([id], skip_special_tokens=False)`, no clean-up).
//! A decode that contains U+FFFD (a partial UTF-8 byte token) is keyed by its raw token string; an
//! empty normalisation falls back to the decoded text. New ids are handed out in order of first
//! appearance over ids `0..len(tokenizer)`. Proven against the Python reference by the sha256 of the
//! whole map in `tests/fixtures/dsv41_engram/layout.json`.

use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{Lowercase, Replace, Sequence, Strip, StripAccents, NFD, NFKC};
use tokenizers::{NormalizedString, Normalizer, NormalizerWrapper, Tokenizer};

const SENTINEL: &str = "\u{e000}";

/// raw token id -> compressed id
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedTokenMap {
    map: Vec<u32>,
    vocab: u32,
}

/// The reference normaliser (see the module docs).
pub fn engram_normalizer() -> Result<Sequence> {
    let parts: Vec<NormalizerWrapper> = vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        Replace::new(ReplacePattern::Regex(r"[ \t\r\n]+".into()), " ").map_err(|e| anyhow!("{e}"))?.into(),
        Replace::new(ReplacePattern::Regex("^ $".into()), SENTINEL).map_err(|e| anyhow!("{e}"))?.into(),
        Strip::new(true, true).into(),
        Replace::new(SENTINEL, " ").map_err(|e| anyhow!("{e}"))?.into(),
    ];
    Ok(Sequence::new(parts))
}

/// Normalise one decoded token the way the reference keys it.
pub fn normalize_key(norm: &Sequence, text: &str) -> Result<String> {
    let mut n = NormalizedString::from(text);
    norm.normalize(&mut n).map_err(|e| anyhow!("engram normaliser: {e}"))?;
    Ok(n.get().to_string())
}

impl CompressedTokenMap {
    /// Build from a loaded tokenizer (`tokenizer.json` of the native checkpoint or of any pack).
    pub fn from_tokenizer(tok: &Tokenizer) -> Result<Self> {
        let n = tok.get_vocab_size(true);
        let norm = engram_normalizer()?;
        let mut key_to_new: HashMap<String, u32> = HashMap::with_capacity(n);
        let mut map = Vec::with_capacity(n);
        for id in 0..n as u32 {
            let text = tok.decode(&[id], false).map_err(|e| anyhow!("decode({id}): {e}"))?;
            let key = if text.contains('\u{fffd}') {
                tok.id_to_token(id).unwrap_or_default()
            } else {
                let k = normalize_key(&norm, &text)?;
                if k.is_empty() {
                    text
                } else {
                    k
                }
            };
            let next = key_to_new.len() as u32;
            map.push(*key_to_new.entry(key).or_insert(next));
        }
        let vocab = key_to_new.len() as u32;
        Ok(CompressedTokenMap { map, vocab })
    }

    /// Build from `<dir>/tokenizer.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("tokenizer.json");
        let tok = Tokenizer::from_file(&p).map_err(|e| anyhow!("loading {}: {e}", p.display()))?;
        Self::from_tokenizer(&tok).with_context(|| format!("engram token map from {}", p.display()))
    }

    /// From an explicit table whose compressed space is `0..=max(map)`.
    pub fn from_vec(map: Vec<u32>) -> Result<Self> {
        let vocab = map.iter().copied().max().map_or(0, |m| m + 1);
        Self::from_parts(map, vocab)
    }

    /// From an explicit table and the size of its compressed space (a cached map, or tests).
    pub fn from_parts(map: Vec<u32>, vocab: u32) -> Result<Self> {
        if map.is_empty() {
            bail!("empty token map");
        }
        if let Some(bad) = map.iter().find(|&&v| v >= vocab) {
            bail!("token map entry {bad} outside the compressed space 0..{vocab}");
        }
        Ok(CompressedTokenMap { map, vocab })
    }

    /// Size of the compressed id space (`engram_compressed_vocab_size`).
    pub fn vocab(&self) -> u32 {
        self.vocab
    }
    /// Number of raw token ids covered (`len(tokenizer)`).
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    #[inline]
    pub fn get(&self, raw: u32) -> Option<u32> {
        self.map.get(raw as usize).copied()
    }
    pub fn as_slice(&self) -> &[u32] {
        &self.map
    }
    /// sha256 of the map as little-endian u32 (the fingerprint the Python checker prints).
    pub fn sha256_hex(&self) -> String {
        let mut h = Sha256::new();
        for v in &self.map {
            h.update(v.to_le_bytes());
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normaliser_examples() {
        let n = engram_normalizer().unwrap();
        let k = |s: &str| normalize_key(&n, s).unwrap();
        assert_eq!(k(" The"), "the");
        assert_eq!(k("THE"), "the");
        assert_eq!(k("Café"), "cafe");
        assert_eq!(k("cafe\u{301}"), "cafe");
        assert_eq!(k("ﬁ"), "fi"); // NFKC ligature
        assert_eq!(k("ＡＢＣ"), "abc"); // NFKC full width
        assert_eq!(k(" "), " "); // the sentinel keeps a lone space
        assert_eq!(k("\t"), " ");
        assert_eq!(k("\n\n"), " ");
        assert_eq!(k("  x \t\n y  "), "x y");
        assert_eq!(k("\u{301}"), ""); // only an accent: empty, the caller falls back to the text
    }
}
