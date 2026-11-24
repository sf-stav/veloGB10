//! Engram bucket layout: config, primes, column offsets, hash multipliers, and the head shard map.
//!
//! Port of `EngramLayout.from_args`, `find_next_prime` and `compute_hash_multipliers` from the
//! checkpoint's `inference/engram.py`, with no Python at run time. The multipliers come from numpy's
//! `default_rng(10007 * layer_id).integers(0, bound, max_ngram_size, int64)`; [`np_rng`] is a port of
//! numpy's `SeedSequence` + `PCG64` + Lemire bounded draw, proven against numpy in the golden tests
//! (`tests/fixtures/dsv41_engram/layout.json`, written by `tools/dsv41_ref/engram_check.py`).
//!
//! Table geometry (per Engram layer): `n_cols = (max_ngram_size - 1) * n_heads` hash columns,
//! column `c = g * n_heads + h` (g = n-gram index 0.. for the 2-gram, 3-gram, ...; h = head). Column c
//! owns rows `[offsets[l][c], offsets[l][c] + primes[l][g][h])` of the layer's table; the columns tile
//! the table exactly (`sum(primes[l]) == num_embeddings[l]`).
//!
//! Shard unit = head h: its `max_ngram_size - 1` columns in every Engram layer ([`EngramLayout::rows_for_head`]).

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashSet;
use std::path::Path;

/// Upper bound on `max_ngram_size` this implementation accepts (V4.1 uses 4).
pub const MAX_NGRAM: usize = 8;

/// The `engram_*` fields of a V4.1 `config.json` (HF `text_config` or the flat `inference/config.json`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramConfig {
    pub layer_ids: Vec<u32>,
    pub max_ngram_size: usize,
    /// `engram_vocab_size`: every column's prime search starts above `vocab_size - 1`.
    pub vocab_size: u64,
    pub n_heads: usize,
    pub head_dim: usize,
    pub num_embeddings: Vec<u64>,
    /// raw token id that fills n-gram slots with no history (`engram_pad_id` / `engram_pad_token_id`)
    pub pad_id: u32,
    pub compressed_vocab_size: u32,
}

impl EngramConfig {
    /// Parse from a parsed `config.json`. Looks in `text_config` (HF layout) or at the top level
    /// (`inference/config.json`). Returns `Ok(None)` when the model has no Engram layers.
    pub fn from_config_json(v: &serde_json::Value) -> Result<Option<Self>> {
        let c = if v.get("engram_layer_ids").is_some() {
            v
        } else if let Some(t) = v.get("text_config").filter(|t| t.get("engram_layer_ids").is_some()) {
            t
        } else {
            return Ok(None);
        };
        let u64s = |k: &str| -> Result<Vec<u64>> {
            c.get(k)
                .and_then(|x| x.as_array())
                .ok_or_else(|| anyhow!("config: {k} missing or not a list"))?
                .iter()
                .map(|x| x.as_u64().ok_or_else(|| anyhow!("config: {k} has a non-integer entry")))
                .collect()
        };
        let u = |k: &str| -> Result<u64> { c.get(k).and_then(|x| x.as_u64()).ok_or_else(|| anyhow!("config: {k} missing")) };
        let layer_ids: Vec<u32> = u64s("engram_layer_ids")?.into_iter().map(|x| x as u32).collect();
        if layer_ids.is_empty() {
            return Ok(None);
        }
        let pad_id = c
            .get("engram_pad_id")
            .or_else(|| c.get("engram_pad_token_id"))
            .and_then(|x| x.as_u64())
            .unwrap_or(2) as u32; // ModelArgs default, "matches training"
        let cfg = EngramConfig {
            layer_ids,
            max_ngram_size: u("engram_max_ngram_size")? as usize,
            vocab_size: u("engram_vocab_size")?,
            n_heads: u("engram_n_heads")? as usize,
            head_dim: u("engram_head_dim")? as usize,
            num_embeddings: u64s("engram_num_embeddings")?,
            pad_id,
            compressed_vocab_size: u("engram_compressed_vocab_size")? as u32,
        };
        cfg.validate()?;
        Ok(Some(cfg))
    }

    /// `config.json` of a model dir (native checkpoint or any EXL3 pack of the model).
    pub fn from_model_dir(dir: &Path) -> Result<Option<Self>> {
        let p = dir.join("config.json");
        let txt = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        let v: serde_json::Value = serde_json::from_str(&txt).with_context(|| format!("parsing {}", p.display()))?;
        Self::from_config_json(&v)
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_ngram_size < 2 || self.max_ngram_size > MAX_NGRAM {
            bail!("engram_max_ngram_size {} outside 2..={MAX_NGRAM}", self.max_ngram_size);
        }
        if self.n_heads == 0 || self.vocab_size < 2 || self.compressed_vocab_size == 0 {
            bail!("engram config: n_heads / vocab_size / compressed vocab must be positive");
        }
        if self.num_embeddings.len() != self.layer_ids.len() {
            bail!("engram_num_embeddings has {} entries for {} layers", self.num_embeddings.len(), self.layer_ids.len());
        }
        Ok(())
    }

    pub fn n_cols(&self) -> usize {
        (self.max_ngram_size - 1) * self.n_heads
    }
}

/// One column of one head, inside that head's shard file (same order and meaning as the packer's
/// manifest sections: layer-major, then n-gram index g; `first_record` counts records before it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadSection {
    pub layer_idx: usize,
    pub layer_id: u32,
    pub g: usize,
    pub head: usize,
    /// column index in the layer's hash output (`g * n_heads + head`)
    pub col: usize,
    /// first row of the column in the layer's full table (= `offsets[layer_idx][col]`)
    pub source_row_start: u64,
    /// rows in the column (= its prime)
    pub rows: u64,
    /// record index of the column's row 0 inside the head file
    pub first_record: u64,
}

/// Rows a head owns: its sections, in file order, and the record total.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadRows {
    pub head: usize,
    pub sections: Vec<HeadSection>,
    pub total_records: u64,
}

impl HeadRows {
    /// Record index inside the head file of a hash id produced for (`layer_idx`, `g`).
    /// `None` if the id does not belong to this head's column.
    #[inline]
    pub fn record_of(&self, layer_idx: usize, g: usize, hash_id: u64, n_ngram_cols: usize) -> Option<u64> {
        let s = self.sections.get(layer_idx * n_ngram_cols + g)?;
        let row = hash_id.checked_sub(s.source_row_start)?;
        (row < s.rows).then_some(s.first_record + row)
    }
}

/// The resolved layout: everything the hash needs besides the token map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramLayout {
    pub cfg: EngramConfig,
    /// `primes[l][g][h]`: bucket modulus of column (g, h) of Engram layer index l
    pub primes: Vec<Vec<Vec<u64>>>,
    /// `offsets[l][c]`: first table row of column c in layer index l
    pub offsets: Vec<Vec<u64>>,
    /// `multipliers[l][k]`: odd multiplier of look-back k (0 = current token) in layer index l
    pub multipliers: Vec<Vec<u64>>,
}

impl EngramLayout {
    /// Build the layout exactly as `EngramLayout.from_args` + `NgramHashState.__init__` do.
    pub fn new(cfg: EngramConfig) -> Result<Self> {
        cfg.validate()?;
        let n_g = cfg.max_ngram_size - 1;
        let primes = column_primes(cfg.layer_ids.len(), n_g, cfg.n_heads, cfg.vocab_size);
        let offsets = column_offsets(&primes);
        for (l, layer) in primes.iter().enumerate() {
            let acc: u64 = layer.iter().flatten().sum();
            if acc != cfg.num_embeddings[l] {
                bail!(
                    "engram layer {}: columns cover {acc} rows but engram_num_embeddings says {}",
                    cfg.layer_ids[l],
                    cfg.num_embeddings[l]
                );
            }
            if acc > u32::MAX as u64 + 1 {
                bail!("engram layer {}: {acc} rows do not fit u32 hash ids", cfg.layer_ids[l]);
            }
        }
        let multipliers = compute_hash_multipliers(&cfg.layer_ids, cfg.max_ngram_size, cfg.compressed_vocab_size as u64)?;
        Ok(EngramLayout { cfg, primes, offsets, multipliers })
    }

    pub fn n_layers(&self) -> usize {
        self.cfg.layer_ids.len()
    }
    pub fn n_heads(&self) -> usize {
        self.cfg.n_heads
    }
    /// n-gram columns per head (3 for V4.1: the 2-, 3- and 4-gram)
    pub fn n_ngram_cols(&self) -> usize {
        self.cfg.max_ngram_size - 1
    }
    /// hash columns per Engram layer (24 for V4.1)
    pub fn n_cols(&self) -> usize {
        self.cfg.n_cols()
    }
    /// hash ids per token over all Engram layers (48 for V4.1)
    pub fn ids_per_token(&self) -> usize {
        self.n_layers() * self.n_cols()
    }
    #[inline]
    pub fn col(&self, g: usize, head: usize) -> usize {
        g * self.cfg.n_heads + head
    }
    /// (g, head) of a column index.
    #[inline]
    pub fn col_parts(&self, col: usize) -> (usize, usize) {
        (col / self.cfg.n_heads, col % self.cfg.n_heads)
    }
    /// Index of a model layer id among the Engram layers (`layer_hash_index` in the reference).
    pub fn layer_index(&self, layer_id: u32) -> Option<usize> {
        self.cfg.layer_ids.iter().position(|&x| x == layer_id)
    }

    /// The shard unit: the rows of head `head` in every Engram layer, in packer-file order.
    pub fn rows_for_head(&self, head: usize) -> HeadRows {
        assert!(head < self.cfg.n_heads, "engram head {head} out of range");
        head_rows(&self.cfg.layer_ids, &self.primes, &self.offsets, self.cfg.n_heads, head)
    }

    /// Heads owned by `rank` of `world` ranks: TP8 rank r -> {r}; TP4 -> {2r, 2r+1}; TP2 -> {4r..4r+3};
    /// TP3 -> {0..2, 2..5, 5..8}; TP6 -> {0..1, 1..2, 2..4, 4..5, 5..6, 6..8}; TP1 -> all.
    pub fn heads_for_rank(&self, world: usize, rank: usize) -> Result<std::ops::Range<usize>> {
        let h = self.cfg.n_heads;
        if world == 0 || world > h || rank >= world {
            bail!("engram: {h} heads cannot be split over world {world} (rank {rank})");
        }
        Ok((rank * h / world)..((rank + 1) * h / world))
    }
}

// ------------------------------------------------------------------------------------------ primes

/// THE prime search of the bucket layout (`EngramLayout.from_args`): `primes[l][g][h]`, per layer and n-gram
/// index the next `n_heads` primes above `vocab_size - 1`, never reusing a prime across the whole model.
/// The packer's `manifest::Geometry` and this layout both call it (leg EIO: one layout, never two).
pub fn column_primes(n_layers: usize, n_g: usize, n_heads: usize, vocab_size: u64) -> Vec<Vec<Vec<u64>>> {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut primes = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let mut per_ngram = Vec::with_capacity(n_g);
        for _ in 0..n_g {
            let mut current = vocab_size - 1;
            let mut sizes = Vec::with_capacity(n_heads);
            for _ in 0..n_heads {
                current = find_next_prime(current, &seen);
                seen.insert(current);
                sizes.push(current);
            }
            per_ngram.push(sizes);
        }
        primes.push(per_ngram);
    }
    primes
}

/// `offsets[l][c]`: exclusive cumulative sums of the primes in column order `c = g * n_heads + h`.
pub fn column_offsets(primes: &[Vec<Vec<u64>>]) -> Vec<Vec<u64>> {
    primes
        .iter()
        .map(|layer| {
            let mut acc = 0u64;
            layer.iter().flatten().map(|p| { let o = acc; acc += p; o }).collect()
        })
        .collect()
}

/// THE head-file order: head `head`'s columns, layer-major then g, with each column's first record.
/// `EngramLayout::rows_for_head` and the packer's `manifest::sections_for` both call it.
pub fn head_rows(layer_ids: &[u32], primes: &[Vec<Vec<u64>>], offsets: &[Vec<u64>], n_heads: usize, head: usize) -> HeadRows {
    let n_g = primes.first().map_or(0, |l| l.len());
    let mut sections = Vec::with_capacity(layer_ids.len() * n_g);
    let mut next = 0u64;
    for (l, &layer_id) in layer_ids.iter().enumerate() {
        for g in 0..n_g {
            let col = g * n_heads + head;
            let rows = primes[l][g][head];
            sections.push(HeadSection { layer_idx: l, layer_id, g, head, col, source_row_start: offsets[l][col], rows, first_record: next });
            next += rows;
        }
    }
    HeadRows { head, sections, total_records: next }
}

/// Deterministic Miller-Rabin, exact for every u64 (bases = the first 12 primes).
pub fn is_prime(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    for p in BASES {
        if n % p == 0 {
            return n == p;
        }
    }
    let (mut d, mut s) = (n - 1, 0u32);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    let mulmod = |a: u64, b: u64| ((a as u128 * b as u128) % n as u128) as u64;
    let powmod = |mut b: u64, mut e: u64| {
        let mut r = 1u64;
        while e > 0 {
            if e & 1 == 1 {
                r = mulmod(r, b);
            }
            b = mulmod(b, b);
            e >>= 1;
        }
        r
    };
    'bases: for a in BASES {
        let mut x = powmod(a, d);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x);
            if x == n - 1 {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

/// `find_next_prime`: the smallest prime above `start` that has not been handed out yet.
pub fn find_next_prime(start: u64, seen: &HashSet<u64>) -> u64 {
    let mut candidate = start + 1;
    while !is_prime(candidate) || seen.contains(&candidate) {
        candidate += 1;
    }
    candidate
}

/// `compute_hash_multipliers`: per layer, `default_rng(10007 * layer_id).integers(0, bound, n, int64) * 2 + 1`
/// with `bound = max(1, (i64::MAX // compressed_vocab) // 2)`, so `token * multiplier` never overflows i64.
pub fn compute_hash_multipliers(layer_ids: &[u32], max_ngram_size: usize, compressed_vocab: u64) -> Result<Vec<Vec<u64>>> {
    if compressed_vocab == 0 {
        bail!("compressed vocab size 0");
    }
    let bound = std::cmp::max(1, (i64::MAX as u64 / compressed_vocab) / 2);
    let mut rows = Vec::with_capacity(layer_ids.len());
    for &layer_id in layer_ids {
        let seed = np_rng::SeedSequence::from_u128(10007u128 * layer_id as u128);
        let mut g = np_rng::Pcg64::from_seed_sequence(&seed);
        let vals = g.integers_i64(0, bound as i64, max_ngram_size)?;
        rows.push(vals.into_iter().map(|v| v as u64 * 2 + 1).collect::<Vec<u64>>());
    }
    // the reference relies on this to keep int64 products exact
    let max_mult = rows.iter().flatten().copied().max().unwrap_or(1);
    if (max_mult as u128) * (compressed_vocab as u128 - 1) > i64::MAX as u128 {
        bail!("engram multipliers would overflow int64 (max {max_mult}, vocab {compressed_vocab})");
    }
    Ok(rows)
}

/// Port of the parts of numpy's random module the multipliers use: `SeedSequence` (pool size 4),
/// `PCG64` (XSL-RR 128/64, not DXSM) and `Generator.integers` for int64 with a range above 2^32
/// (Lemire's bounded draw on 64-bit outputs). Ranges at or below 2^32 take numpy's buffered 32-bit
/// path, which is not ported: they are refused instead of being drawn differently.
pub mod np_rng {
    use anyhow::{bail, Result};

    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_MULT_L: u32 = 0xca01_f9dd;
    const MIX_MULT_R: u32 = 0x4973_f715;
    const XSHIFT: u32 = 16;
    const POOL: usize = 4;

    #[inline]
    fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
        let mut v = value ^ *hash_const;
        *hash_const = hash_const.wrapping_mul(MULT_A);
        v = v.wrapping_mul(*hash_const);
        v ^ (v >> XSHIFT)
    }

    #[inline]
    fn mix(x: u32, y: u32) -> u32 {
        let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^ (r >> XSHIFT)
    }

    /// `numpy.random.SeedSequence(entropy)` with no spawn key.
    #[derive(Clone, Debug)]
    pub struct SeedSequence {
        pub pool: [u32; POOL],
    }

    impl SeedSequence {
        /// Entropy as numpy coerces a non-negative Python int: 32-bit little-endian words ([0] for 0).
        pub fn from_u128(entropy: u128) -> Self {
            let mut words = Vec::new();
            let mut n = entropy;
            if n == 0 {
                words.push(0u32);
            }
            while n > 0 {
                words.push((n & 0xffff_ffff) as u32);
                n >>= 32;
            }
            Self::from_words(&words)
        }

        pub fn from_words(entropy: &[u32]) -> Self {
            let mut pool = [0u32; POOL];
            let mut hc = INIT_A;
            for (i, slot) in pool.iter_mut().enumerate() {
                *slot = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hc);
            }
            for i_src in 0..POOL {
                for i_dst in 0..POOL {
                    if i_src != i_dst {
                        let h = hashmix(pool[i_src], &mut hc);
                        pool[i_dst] = mix(pool[i_dst], h);
                    }
                }
            }
            for &e in entropy.iter().skip(POOL) {
                for i_dst in 0..POOL {
                    let h = hashmix(e, &mut hc);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
            SeedSequence { pool }
        }

        /// `generate_state(n_words, np.uint64)`
        pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
            let mut hc = INIT_B;
            let mut w32 = Vec::with_capacity(n_words * 2);
            for i in 0..n_words * 2 {
                let mut v = self.pool[i % POOL] ^ hc;
                hc = hc.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hc);
                w32.push(v ^ (v >> XSHIFT));
            }
            w32.chunks(2).map(|c| c[0] as u64 | (c[1] as u64) << 32).collect()
        }
    }

    const PCG_MULT: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341u128;

    /// numpy's `PCG64` bit generator.
    #[derive(Clone, Debug)]
    pub struct Pcg64 {
        state: u128,
        inc: u128,
    }

    impl Pcg64 {
        pub fn from_seed_sequence(ss: &SeedSequence) -> Self {
            let v = ss.generate_state_u64(4);
            let initstate = (v[0] as u128) << 64 | v[1] as u128;
            let initseq = (v[2] as u128) << 64 | v[3] as u128;
            let mut g = Pcg64 { state: 0, inc: (initseq << 1) | 1 };
            g.step();
            g.state = g.state.wrapping_add(initstate);
            g.step();
            g
        }

        #[inline]
        fn step(&mut self) {
            self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
        }

        pub fn next_u64(&mut self) -> u64 {
            self.step();
            let s = self.state;
            let rot = (s >> 122) as u32;
            (((s >> 64) as u64) ^ (s as u64)).rotate_right(rot)
        }

        fn bounded_lemire_u64(&mut self, rng: u64) -> u64 {
            let rng_excl = rng + 1;
            let mut m = self.next_u64() as u128 * rng_excl as u128;
            let mut leftover = m as u64;
            if leftover < rng_excl {
                let threshold = (u64::MAX - rng) % rng_excl;
                while leftover < threshold {
                    m = self.next_u64() as u128 * rng_excl as u128;
                    leftover = m as u64;
                }
            }
            (m >> 64) as u64
        }

        /// `Generator.integers(low, high, size, dtype=np.int64)` (endpoint=False).
        pub fn integers_i64(&mut self, low: i64, high: i64, size: usize) -> Result<Vec<i64>> {
            if high <= low {
                bail!("integers: low >= high");
            }
            let rng = (high - 1).wrapping_sub(low) as u64;
            let off = low as u64;
            let mut out = Vec::with_capacity(size);
            for _ in 0..size {
                let v = if rng == 0 {
                    0
                } else if rng <= 0xffff_ffff {
                    bail!("integers: range {} takes numpy's 32-bit buffered path (not ported)", rng as u128 + 1);
                } else if rng == u64::MAX {
                    self.next_u64()
                } else {
                    self.bounded_lemire_u64(rng)
                };
                out.push(off.wrapping_add(v) as i64);
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v41() -> EngramConfig {
        EngramConfig {
            layer_ids: vec![1, 14],
            max_ngram_size: 4,
            vocab_size: 16_000_000,
            n_heads: 8,
            head_dim: 256,
            num_embeddings: vec![384_006_168, 384_016_682],
            pad_id: 2,
            compressed_vocab_size: 99_092,
        }
    }

    #[test]
    fn primes_basic() {
        let small: Vec<u64> = (0..60).filter(|&n| is_prime(n)).collect();
        assert_eq!(small, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]);
        assert!(is_prime(16_000_057) && !is_prime(16_000_058));
        assert!(is_prime(18_446_744_073_709_551_557)); // largest u64 prime
        assert!(!is_prime(3_215_031_751)); // strong pseudoprime to bases 2, 3, 5, 7
        // brute force cross-check on a window above the V4.1 start
        let brute = |n: u64| n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0);
        for n in 16_000_000..16_002_000u64 {
            assert_eq!(is_prime(n), brute(n), "{n}");
        }
    }

    #[test]
    fn layout_tiles_and_shards() {
        let lay = EngramLayout::new(v41()).unwrap();
        assert_eq!((lay.n_layers(), lay.n_cols(), lay.ids_per_token()), (2, 24, 48));
        assert_eq!(lay.primes[0][0][0], 16_000_057);
        // every head file: 6 sections, contiguous records; all heads together cover each table once
        let mut per_layer = [0u64; 2];
        for h in 0..8 {
            let hr = lay.rows_for_head(h);
            assert_eq!(hr.sections.len(), 6);
            let mut next = 0;
            for s in &hr.sections {
                assert_eq!(s.first_record, next);
                assert_eq!(s.col, s.g * 8 + h);
                assert_eq!(s.source_row_start, lay.offsets[s.layer_idx][s.col]);
                next += s.rows;
                per_layer[s.layer_idx] += s.rows;
            }
            assert_eq!(hr.total_records, next);
            // record_of: first and last row of a section, and a row of another head's column
            let s = hr.sections[4];
            assert_eq!(hr.record_of(1, 1, s.source_row_start, 3), Some(s.first_record));
            assert_eq!(hr.record_of(1, 1, s.source_row_start + s.rows - 1, 3), Some(s.first_record + s.rows - 1));
            assert_eq!(hr.record_of(1, 1, s.source_row_start + s.rows, 3), None);
        }
        assert_eq!(per_layer.to_vec(), lay.cfg.num_embeddings);
        assert_eq!(lay.heads_for_rank(8, 5).unwrap(), 5..6);
        assert_eq!(lay.heads_for_rank(6, 2).unwrap(), 2..4);
        assert_eq!(lay.heads_for_rank(4, 1).unwrap(), 2..4);
        assert_eq!(lay.heads_for_rank(3, 0).unwrap(), 0..2);
        assert_eq!(lay.heads_for_rank(3, 1).unwrap(), 2..5);
        assert_eq!(lay.heads_for_rank(3, 2).unwrap(), 5..8);
        assert_eq!(lay.heads_for_rank(2, 1).unwrap(), 4..8);
        assert_eq!(lay.heads_for_rank(1, 0).unwrap(), 0..8);
        assert!(lay.heads_for_rank(9, 0).is_err());
    }

    #[test]
    fn layout_refuses_wrong_table_size() {
        let mut c = v41();
        c.num_embeddings[1] += 1;
        assert!(EngramLayout::new(c).is_err());
    }

    #[test]
    fn config_from_both_json_shapes() {
        let flat = serde_json::json!({
            "engram_layer_ids": [1, 14], "engram_vocab_size": 16000000,
            "engram_num_embeddings": [384006168, 384016682], "engram_max_ngram_size": 4,
            "engram_pad_id": 2, "engram_compressed_vocab_size": 99092, "engram_n_heads": 8, "engram_head_dim": 256
        });
        let hf = serde_json::json!({"model_type": "deepseek_v41", "text_config": {
            "engram_layer_ids": [1, 14], "engram_vocab_size": 16000000,
            "engram_num_embeddings": [384006168, 384016682], "engram_max_ngram_size": 4,
            "engram_pad_token_id": 2, "engram_compressed_vocab_size": 99092, "engram_n_heads": 8, "engram_head_dim": 256
        }});
        assert_eq!(EngramConfig::from_config_json(&flat).unwrap().unwrap(), v41());
        assert_eq!(EngramConfig::from_config_json(&hf).unwrap().unwrap(), v41());
        assert!(EngramConfig::from_config_json(&serde_json::json!({"vocab_size": 5})).unwrap().is_none());
    }
}
