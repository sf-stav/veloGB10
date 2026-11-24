//! Engram n-gram hashing: port of `NgramHashState` (`inference/engram.py`), incremental.
//!
//! For position p the reference takes the compressed ids at p, p-1, .., p-(N-1) (N = `max_ngram_size`),
//! replaces every slot at or beyond the first BLOCKED one with the compressed pad id (a slot is blocked
//! when it lies before the sequence start or holds a DEAD position, i.e. an image span; blocking is
//! sticky over the look-back), multiplies slot k by `multipliers[layer][k]`, and XOR-folds:
//!
//! ```text
//! rolling = t0*m0;  for i in 1..N { rolling ^= ti*mi;  for h { id[(i-1)*H + h] = rolling % prime[i-1][h] + offset[(i-1)*H + h] } }
//! ```
//!
//! The only cross-position state is the last N-1 entries (compressed id or BLOCK), so a lane's
//! [`NgramHashState`] is a few words: it is what a prefix-cache entry saves, what a chunked prefill
//! carries between chunks, and what speculation forks ([`EngramHasher::peek`]) and commits
//! ([`EngramHasher::advance`]). Every variant is checked bit-exactly against one full reference pass.
//!
//! Output layout (`out`): `[token][engram layer index][column]`, column `c = g * n_heads + h`, i.e. the
//! reference's `[B, L, n_engram_layers, n_hash_cols]` for one lane. Ids are global rows of the layer's
//! table (`< num_embeddings[l] <= u32::MAX`). DEAD positions get the all-pad ids (as the reference
//! computes them); their gate is zero in the model, so readers may skip those rows.

use super::layout::{EngramConfig, EngramLayout, MAX_NGRAM};
use super::tokmap::CompressedTokenMap;
use anyhow::{bail, Result};
use std::path::Path;

/// History entry for a blocked slot (before the sequence start, or a DEAD position).
const BLOCK: u32 = u32::MAX;

/// Serialized size of [`NgramHashState`] (see [`NgramHashState::to_bytes`]).
pub const STATE_BYTES: usize = 8 + 4 + 4 * (MAX_NGRAM - 1);

/// Per-lane n-gram history: the compressed ids of the last `max_ngram_size - 1` positions
/// (`hist[0]` = the previous position), BLOCK for blocked slots; `pos` = positions consumed so far
/// (the next position's absolute index). `Copy`, 40 bytes serialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NgramHashState {
    hist: [u32; MAX_NGRAM - 1],
    pos: u64,
}

impl Default for NgramHashState {
    fn default() -> Self {
        Self::new()
    }
}

impl NgramHashState {
    /// A fresh sequence (position 0, nothing to look back at).
    pub fn new() -> Self {
        NgramHashState { hist: [BLOCK; MAX_NGRAM - 1], pos: 0 }
    }
    /// Absolute position of the next token.
    pub fn pos(&self) -> u64 {
        self.pos
    }
    /// For a prefix-cache entry: `pos` (u64 LE), a format tag (u32 LE), then the history (u32 LE each).
    pub fn to_bytes(&self) -> [u8; STATE_BYTES] {
        let mut b = [0u8; STATE_BYTES];
        b[..8].copy_from_slice(&self.pos.to_le_bytes());
        b[8..12].copy_from_slice(&STATE_TAG.to_le_bytes());
        for (i, h) in self.hist.iter().enumerate() {
            b[12 + 4 * i..16 + 4 * i].copy_from_slice(&h.to_le_bytes());
        }
        b
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() != STATE_BYTES || u32::from_le_bytes(b[8..12].try_into().unwrap()) != STATE_TAG {
            bail!("engram n-gram state: bad serialized state ({} bytes)", b.len());
        }
        let mut s = NgramHashState { hist: [BLOCK; MAX_NGRAM - 1], pos: u64::from_le_bytes(b[..8].try_into().unwrap()) };
        for (i, h) in s.hist.iter_mut().enumerate() {
            *h = u32::from_le_bytes(b[12 + 4 * i..16 + 4 * i].try_into().unwrap());
        }
        Ok(s)
    }
}

const STATE_TAG: u32 = 0x4e47_5331; // "1SGN": version 1

/// Immutable hashing context, shared by every lane (wrap in an `Arc`).
#[derive(Clone, Debug)]
pub struct EngramHasher {
    map: CompressedTokenMap,
    layout: EngramLayout,
    /// compressed pad id
    pad: u64,
    n: usize,
    n_heads: usize,
    n_layers: usize,
    n_cols: usize,
    /// `[layer][k]`
    mult: Vec<u64>,
    /// `[layer][col]`
    prime: Vec<u64>,
    /// `[layer][col]`
    offset: Vec<u64>,
}

impl EngramHasher {
    /// Check the token map against the config (the reference asserts the compressed vocab size, since
    /// every multiplier derives from it) and flatten the tables.
    pub fn new(map: CompressedTokenMap, layout: EngramLayout) -> Result<Self> {
        let cfg = &layout.cfg;
        if map.vocab() != cfg.compressed_vocab_size {
            bail!(
                "engram: the tokenizer compresses to {} ids but the config says engram_compressed_vocab_size = {}; \
                 hashing would rehash the whole table (wrong tokenizer.json for this model?)",
                map.vocab(),
                cfg.compressed_vocab_size
            );
        }
        let pad = match map.get(cfg.pad_id) {
            Some(p) => p as u64,
            None => bail!("engram pad id {} outside the tokenizer ({} ids)", cfg.pad_id, map.len()),
        };
        let mult = layout.multipliers.iter().flatten().copied().collect();
        let prime = layout.primes.iter().flat_map(|l| l.iter().flatten().copied()).collect();
        let offset = layout.offsets.iter().flatten().copied().collect();
        Ok(EngramHasher {
            pad,
            n: cfg.max_ngram_size,
            n_heads: cfg.n_heads,
            n_layers: layout.n_layers(),
            n_cols: layout.n_cols(),
            mult,
            prime,
            offset,
            map,
            layout,
        })
    }

    /// From a model dir (`config.json` + `tokenizer.json`). `Ok(None)` if the model has no Engram.
    pub fn from_model_dir(dir: &Path) -> Result<Option<Self>> {
        let Some(cfg) = EngramConfig::from_model_dir(dir)? else { return Ok(None) };
        let layout = EngramLayout::new(cfg)?;
        let map = CompressedTokenMap::from_model_dir(dir)?;
        Ok(Some(Self::new(map, layout)?))
    }

    pub fn layout(&self) -> &EngramLayout {
        &self.layout
    }
    pub fn token_map(&self) -> &CompressedTokenMap {
        &self.map
    }
    /// Hash ids written per token (`n_engram_layers * n_cols`, 48 for V4.1).
    pub fn ids_per_token(&self) -> usize {
        self.n_layers * self.n_cols
    }
    /// The compressed pad id.
    pub fn pad_compressed(&self) -> u32 {
        self.pad as u32
    }

    /// Compressed id of a raw token id.
    #[inline]
    pub fn compress(&self, raw: u32) -> Result<u32> {
        match self.map.get(raw) {
            Some(c) => Ok(c),
            None => bail!("engram: token id {raw} outside the tokenizer ({} ids)", self.map.len()),
        }
    }

    /// History entry of one input position.
    #[inline]
    fn entry(&self, raw: u32, live: bool) -> Result<u32> {
        if live {
            self.compress(raw)
        } else {
            Ok(BLOCK)
        }
    }

    /// Hash one position given its entry and the history before it; writes `ids_per_token` ids.
    #[inline]
    fn hash_one(&self, cur: u32, hist: &[u32; MAX_NGRAM - 1], out: &mut [u32]) {
        let n = self.n;
        // look-back slots with sticky blocking -> compressed ids (pad when blocked)
        let mut t = [0u64; MAX_NGRAM];
        let mut blocked = false;
        for (k, slot) in t.iter_mut().enumerate().take(n) {
            let e = if k == 0 { cur } else { hist[k - 1] };
            blocked |= e == BLOCK;
            *slot = if blocked { self.pad } else { e as u64 };
        }
        let h = self.n_heads;
        for l in 0..self.n_layers {
            let m = &self.mult[l * n..(l + 1) * n];
            let base = l * self.n_cols;
            let mut rolling = t[0] * m[0];
            for i in 1..n {
                rolling ^= t[i] * m[i];
                let c0 = (i - 1) * h;
                for hh in 0..h {
                    let c = base + c0 + hh;
                    out[c] = (rolling % self.prime[c] + self.offset[c]) as u32;
                }
            }
        }
    }

    /// Shift the look-back by one (only the first N-1 slots are used; the rest stay BLOCK, so equal
    /// histories are equal states, byte for byte).
    #[inline]
    fn push(&self, hist: &mut [u32; MAX_NGRAM - 1], e: u32) {
        hist.copy_within(0..self.n - 2, 1);
        hist[0] = e;
    }

    fn check_args(&self, tokens: &[u32], live: Option<&[bool]>, out: Option<&[u32]>) -> Result<()> {
        if let Some(l) = live {
            if l.len() != tokens.len() {
                bail!("engram: live mask has {} entries for {} tokens", l.len(), tokens.len());
            }
        }
        if let Some(o) = out {
            if o.len() != tokens.len() * self.ids_per_token() {
                bail!("engram: out has {} slots, need {} x {}", o.len(), tokens.len(), self.ids_per_token());
            }
        }
        Ok(())
    }

    /// Hash a chunk (prefill chunk, decode step, or verify block) that starts at `state.pos()`, write
    /// `tokens.len() * ids_per_token()` ids into `out`, and advance `state` past it.
    /// `live[i] == false` marks a DEAD position (image span); `None` = all live.
    pub fn hash_chunk(&self, state: &mut NgramHashState, tokens: &[u32], live: Option<&[bool]>, out: &mut [u32]) -> Result<()> {
        self.check_args(tokens, live, Some(out))?;
        let per = self.ids_per_token();
        let mut hist = state.hist;
        for (i, &raw) in tokens.iter().enumerate() {
            let e = self.entry(raw, live.map_or(true, |l| l[i]))?;
            self.hash_one(e, &hist, &mut out[i * per..(i + 1) * per]);
            self.push(&mut hist, e);
        }
        state.hist = hist;
        state.pos += tokens.len() as u64;
        Ok(())
    }

    /// Hash without committing (drafts, a verify block, a row prefetch): returns the state the lane
    /// would have after `tokens`. Commit the accepted prefix later with [`Self::advance`].
    pub fn peek(&self, state: &NgramHashState, tokens: &[u32], live: Option<&[bool]>, out: &mut [u32]) -> Result<NgramHashState> {
        let mut s = *state;
        self.hash_chunk(&mut s, tokens, live, out)?;
        Ok(s)
    }

    /// Advance `state` past `tokens` without producing ids (commit accepted tokens, or replay history).
    pub fn advance(&self, state: &mut NgramHashState, tokens: &[u32], live: Option<&[bool]>) -> Result<()> {
        self.check_args(tokens, live, None)?;
        // every live id is checked as in hash_chunk; only the last N-1 entries reach the history
        let start = tokens.len() - (self.n - 1).min(tokens.len());
        let mut hist = state.hist;
        for (i, &raw) in tokens.iter().enumerate() {
            let e = self.entry(raw, live.map_or(true, |l| l[i]))?;
            if i >= start {
                self.push(&mut hist, e);
            }
        }
        state.hist = hist;
        state.pos += tokens.len() as u64;
        Ok(())
    }

    /// The state after a lane's history, rebuilt from its last `max_ngram_size - 1` tokens: `tail` holds
    /// the final tokens of a sequence of length `pos` (the whole sequence when `pos` is shorter than the
    /// look-back). Use it to resume from a prefix cache that stores token ids instead of the state.
    pub fn state_at(&self, pos: u64, tail: &[u32], live_tail: Option<&[bool]>) -> Result<NgramHashState> {
        self.check_args(tail, live_tail, None)?;
        let need = (self.n as u64 - 1).min(pos) as usize;
        if tail.len() < need || tail.len() as u64 > pos {
            bail!("engram state_at: {} tail tokens for position {pos} (need at least {need})", tail.len());
        }
        let mut s = NgramHashState::new();
        let skip = tail.len() - need;
        self.advance(&mut s, &tail[skip..], live_tail.map(|l| &l[skip..]))?;
        s.pos = pos;
        Ok(s)
    }

    /// Split one output id: (g, head, row within the column's bucket range) for Engram layer `layer_idx`
    /// and column `col`. The row is what a head shard file is indexed by.
    #[inline]
    pub fn split_id(&self, layer_idx: usize, col: usize, id: u32) -> (usize, usize, u64) {
        let c = layer_idx * self.n_cols + col;
        (col / self.n_heads, col % self.n_heads, id as u64 - self.offset[c])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The synthetic model's token map: 60 raw ids onto 50 compressed ids (so raw pad 2 -> compressed 17).
    fn tiny_map() -> Vec<u32> {
        (0..60u32).map(|r| (r * 7 + 3) % 50).collect()
    }

    /// A tiny synthetic model: 2 layers, 4-grams, 2 heads, a non-identity token map.
    fn tiny() -> EngramHasher {
        // primes above 999, drawn in order: layer 0 takes the first 6, layer 1 the next 6
        let p: Vec<u64> = (1000..).filter(|&n| super::super::layout::is_prime(n)).take(12).collect();
        let cfg = EngramConfig {
            layer_ids: vec![0, 3],
            max_ngram_size: 4,
            vocab_size: 1000,
            n_heads: 2,
            head_dim: 4,
            num_embeddings: vec![p[..6].iter().sum(), p[6..].iter().sum()],
            pad_id: 2,
            compressed_vocab_size: 50,
        };
        let lay = EngramLayout::new(cfg).unwrap();
        EngramHasher::new(CompressedTokenMap::from_vec(tiny_map()).unwrap(), lay).unwrap()
    }

    /// Direct transcription of the reference forward (one full pass), independent of the hasher's
    /// internals: its own token map and pad, the layout's nested tables, the reference loop order.
    fn reference(hs: &EngramHasher, tokens: &[u32], live: &[bool]) -> Vec<u32> {
        let lay = hs.layout();
        let map = tiny_map();
        let pad = map[lay.cfg.pad_id as usize] as i64;
        let (n, nh) = (lay.cfg.max_ngram_size, lay.cfg.n_heads);
        let per = lay.ids_per_token();
        let cache: Vec<i64> = tokens.iter().zip(live).map(|(&t, &l)| if l { map[t as usize] as i64 } else { -1 }).collect();
        let mut out = vec![0u32; tokens.len() * per];
        for p in 0..tokens.len() {
            let mut blocked = false;
            let mut t = vec![0i64; n];
            for s in 0..n {
                let src = cache[p.saturating_sub(s)];
                blocked = blocked || p < s || src == -1;
                t[s] = if blocked { pad } else { src };
            }
            for l in 0..lay.n_layers() {
                let prods: Vec<i64> = (0..n).map(|k| t[k] * lay.multipliers[l][k] as i64).collect();
                let mut rolling = prods[0];
                for i in 1..n {
                    rolling ^= prods[i];
                    for h in 0..nh {
                        let c = (i - 1) * nh + h;
                        let id = rolling.rem_euclid(lay.primes[l][i - 1][h] as i64) + lay.offsets[l][c] as i64;
                        out[p * per + l * lay.n_cols() + c] = id as u32;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn synthetic_variants_match_reference() {
        let hs = tiny();
        let per = hs.ids_per_token();
        let mut x = 12345u64;
        let mut rnd = |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        for trial in 0..40 {
            let len = 1 + rnd(60) as usize;
            let tokens: Vec<u32> = (0..len).map(|_| rnd(60) as u32).collect();
            let live: Vec<bool> = (0..len).map(|_| trial % 3 == 0 || rnd(5) != 0).collect();
            let want = reference(&hs, &tokens, &live);
            // full pass
            let mut s = NgramHashState::new();
            let mut got = vec![0u32; len * per];
            hs.hash_chunk(&mut s, &tokens, Some(&live), &mut got).unwrap();
            assert_eq!(got, want, "full pass, trial {trial}");
            assert_eq!(s.pos(), len as u64);
            // the same history gives the same state, whichever way it was reached
            let lo = len.saturating_sub(3);
            assert_eq!(hs.state_at(len as u64, &tokens[lo..], Some(&live[lo..])).unwrap(), s);
            let mut s2 = NgramHashState::new();
            hs.advance(&mut s2, &tokens, Some(&live)).unwrap();
            assert_eq!(s2, s);
            assert_eq!(NgramHashState::from_bytes(&s.to_bytes()).unwrap(), s);
            // random chunks, with a serialize round-trip and a rebuilt state between chunks
            let mut s = NgramHashState::new();
            let mut got = vec![0u32; len * per];
            let mut p = 0;
            while p < len {
                let step = (1 + rnd(7) as usize).min(len - p);
                s = match rnd(3) {
                    0 => NgramHashState::from_bytes(&s.to_bytes()).unwrap(),
                    1 => {
                        let lo = p.saturating_sub(3 + rnd(3) as usize);
                        hs.state_at(p as u64, &tokens[lo..p], Some(&live[lo..p])).unwrap()
                    }
                    _ => s,
                };
                // speculate on garbage, then discard
                let junk: Vec<u32> = (0..step).map(|_| rnd(60) as u32).collect();
                let mut scratch = vec![0u32; step * per];
                let _ = hs.peek(&s, &junk, None, &mut scratch).unwrap();
                hs.hash_chunk(&mut s, &tokens[p..p + step], Some(&live[p..p + step]), &mut got[p * per..(p + step) * per]).unwrap();
                p += step;
            }
            assert_eq!(got, want, "chunked, trial {trial}");
        }
    }

    #[test]
    fn rejects_bad_input() {
        let hs = tiny();
        let mut s = NgramHashState::new();
        let mut out = vec![0u32; hs.ids_per_token()];
        assert!(hs.hash_chunk(&mut s, &[60], None, &mut out).is_err()); // outside the token map
        assert_eq!(hs.pad_compressed(), 17);
        assert!(hs.hash_chunk(&mut s, &[1], Some(&[true, true]), &mut out).is_err());
        assert!(hs.hash_chunk(&mut s, &[1, 2], None, &mut out).is_err());
        assert!(hs.state_at(10, &[1, 2], None).is_err()); // not enough look-back
        assert!(NgramHashState::from_bytes(&[0u8; 5]).is_err());
        let mut cfg = hs.layout().cfg.clone();
        cfg.compressed_vocab_size = 49;
        let lay = EngramLayout::new(cfg).unwrap();
        assert!(EngramHasher::new(CompressedTokenMap::from_vec(tiny_map()).unwrap(), lay).is_err());
    }
}
