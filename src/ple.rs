//! qwen4_exp (Qwen3.8-Flash-Next) PLE — the hashed n-gram embedding table.
//!
//! The PLE layer looks up, for every token, `ngram_heads` rows of a 320M-row table, indexed by
//! hashes of the token's 2-gram and 3-gram contexts (the current token plus the 1–2 preceding ones,
//! with an EOS-resets-the-context rule). This module owns everything that is NOT a GPU kernel:
//!
//!   * the hash (`PleHash`): splitmix64-seeded per-position multipliers, the per-head prime vocab
//!     sizes and offsets, and the EOS context rule — a line-by-line transcription of
//!     `Qwen4ExpTextNGramEmbedding` (modeling_qwen4_exp.py). The device kernel `ple_hash_b`
//!     computes the same ids from the same tables; this host version is the reference for it AND
//!     the producer of row ids when the table is offloaded to SSD;
//!   * the on-disk table produced by `--quantize` (`ple_ngram_nvfp4.bin` + `.json`, see
//!     `quant::quantize_ple_rows`): 96-B NVFP4 row records, one reciprocal global scale per source
//!     shard;
//!   * the SSD reader (`PleSsd`): `pread` of the requested records, fanned out over threads, into
//!     a host staging buffer the GPU dequantizes. Random 96-B reads: 16 per decode token, N*16 per
//!     prefill chunk — the NVMe page cache keeps the hot rows.

use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::Path;

const MASK64: u128 = (1u128 << 64) - 1;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10007;

fn splitmix64(v: u64) -> u64 {
    let mut v = v.wrapping_add(SPLITMIX_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1);
    v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2);
    v ^ (v >> 31)
}

/// `_build_layer_multipliers`: one odd multiplier per n-gram position, bounded so token*mult never
/// exceeds i64::MAX (the reference multiplies in int64 and never wraps — see the assert below).
pub fn layer_multipliers(unigram_vocab: u64, ngram_size: usize, ple_layer_index: u64, seed: u64) -> Vec<i64> {
    let max_long: u64 = i64::MAX as u64;
    let multiplier_max = max_long / unigram_vocab.max(1);
    let half_bound = (multiplier_max / 2).max(1);
    let base_seed = seed.wrapping_add(PRIME_1.wrapping_mul(ple_layer_index));
    (0..ngram_size).map(|i| {
        let v = ((base_seed as u128 + (SPLITMIX_GAMMA as u128) * (i as u128 + 1)) & MASK64) as u64;
        (2 * (splitmix64(v) % half_bound) + 1) as i64
    }).collect()
}

fn is_prime(v: u64) -> bool {
    if v < 2 { return false; }
    if v % 2 == 0 { return v == 2; }
    let mut d = 3u64;
    while d * d <= v { if v % d == 0 { return false; } d += 2; }
    true
}

/// `_find_nth_prime_after(start, count)`: the count-th prime strictly greater than `start`.
pub fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut p = start;
    for _ in 0..count { p += 1; while !is_prime(p) { p += 1; } }
    p
}

/// The n-gram hash tables for one PLE layer.
#[derive(Clone, Debug)]
pub struct PleHash {
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub multipliers: Vec<i64>,     // [ngram_size]
    pub head_vocab: Vec<i64>,      // [ngram_heads]
    pub head_offset: Vec<i64>,     // [ngram_heads]
    pub total_rows: usize,         // padded to the config divisor
    pub eos: i32,
}

impl PleHash {
    pub fn new(cfg: &crate::qwen::Config) -> Self {
        let ngram_size = cfg.ple_ngram_size;
        let heads = cfg.ple_heads();
        let mut head_vocab = Vec::with_capacity(heads);
        let mut head_offset = Vec::with_capacity(heads);
        let mut total = 0u64;
        // Primes are consecutive: p_{k+1} = next prime after p_k, so walk once instead of k times.
        let mut p = cfg.ple_vocab_base as u64 - 1;
        for _ in 0..heads {
            p = nth_prime_after(p, 1);
            head_vocab.push(p as i64);
            head_offset.push(total as i64);
            total += p;
        }
        let div = cfg.ple_vocab_divisor as u64;
        let total_rows = (total.div_ceil(div) * div) as usize;
        Self {
            ngram_size, heads_per_ngram: cfg.ple_heads_per_ngram,
            multipliers: layer_multipliers(cfg.vocab_size as u64, ngram_size, 0, cfg.ple_seed),
            head_vocab, head_offset, total_rows, eos: cfg.eos_token_id as i32,
        }
    }

    pub fn ngram_heads(&self) -> usize { self.head_vocab.len() }

    /// The i64 table the device kernel reads: [multipliers | head_vocab | head_offset].
    pub fn device_table(&self) -> Vec<i64> {
        let mut t = self.multipliers.clone();
        t.extend_from_slice(&self.head_vocab);
        t.extend_from_slice(&self.head_offset);
        t
    }

    /// Row ids for one token. `ctx` is the token's context OLDEST FIRST, `ngram_size` long, ending
    /// with the token itself (`ctx[ngram_size-1]`); missing history is EOS. The EOS rule of
    /// `_shift_right_ignore_eos`: the token `p` positions back is replaced by EOS if any token in
    /// positions [i-p, i-1] is EOS (an EOS resets the n-gram context of everything after it).
    pub fn row_ids(&self, ctx: &[i32], out: &mut [i64]) {
        let n = self.ngram_size;
        debug_assert_eq!(ctx.len(), n);
        debug_assert_eq!(out.len(), self.ngram_heads());
        // shifted[p] = token p back (EOS-normalized)
        let mut shifted = [0i64; 8];
        shifted[0] = ctx[n - 1] as i64;
        let mut blocked = false;
        for p in 1..n {
            let t = ctx[n - 1 - p];
            if t == self.eos { blocked = true; }
            shifted[p] = if blocked { self.eos as i64 } else { t as i64 };
        }
        for ng in 2..=n {
            let start = (ng - 2) * self.heads_per_ngram;
            let mut mixed: i64 = shifted[0].wrapping_mul(self.multipliers[0]);
            for p in 1..ng { mixed ^= shifted[p].wrapping_mul(self.multipliers[p]); }
            for j in 0..self.heads_per_ngram {
                let h = start + j;
                out[h] = mixed.rem_euclid(self.head_vocab[h]) + self.head_offset[h];
            }
        }
    }
}

/// `ple_ngram_nvfp4.json` — the quantizer's table sidecar.
#[derive(Clone, Debug)]
pub struct PleTableMeta {
    pub file: std::path::PathBuf,
    pub dim: usize,
    pub record_bytes: usize,
    pub num_shards: usize,
    pub rows_per_shard: usize,
    pub total_rows: usize,
    pub shard_global_scales: Vec<f32>,
}

impl PleTableMeta {
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let p = model_dir.join("ple_ngram_nvfp4.json");
        let raw = std::fs::read_to_string(&p).map_err(|e| anyhow::anyhow!("{}: {e} (quantize with `pletable:nvfp4`)", p.display()))?;
        let j: serde_json::Value = serde_json::from_str(&raw)?;
        anyhow::ensure!(j["format"].as_str() == Some("ple-rows-nvfp4-v1"), "unknown PLE table format {:?}", j["format"]);
        anyhow::ensure!(j["complete"].as_bool() == Some(true), "PLE table is marked INCOMPLETE (smoke-run artifact?)");
        let scales: Vec<f32> = j["shard_global_scales"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let m = Self {
            file: model_dir.join(j["file"].as_str().unwrap_or("ple_ngram_nvfp4.bin")),
            dim: j["dim"].as_u64().unwrap() as usize,
            record_bytes: j["record_bytes"].as_u64().unwrap() as usize,
            num_shards: j["num_shards"].as_u64().unwrap() as usize,
            rows_per_shard: j["rows_per_shard"].as_u64().unwrap() as usize,
            total_rows: j["total_rows"].as_u64().unwrap() as usize,
            shard_global_scales: scales,
        };
        anyhow::ensure!(m.shard_global_scales.len() == m.num_shards, "PLE sidecar: scale count != shard count");
        anyhow::ensure!(m.dim == crate::quant::PLE_DIM && m.record_bytes == crate::quant::PLE_REC_BYTES,
                        "PLE sidecar geometry {}x{} unsupported", m.dim, m.record_bytes);
        let len = std::fs::metadata(&m.file)?.len();
        anyhow::ensure!(len == (m.total_rows * m.record_bytes) as u64,
                        "{}: {len} bytes, expected {}", m.file.display(), m.total_rows * m.record_bytes);
        Ok(m)
    }
}

/// SSD-resident PLE table: records are read on demand with `pread`. The file stays on disk; the
/// OS page cache is the only caching layer (the hot n-grams of a conversation are a few MB).
pub struct PleSsd {
    file: std::fs::File,
    pub meta: PleTableMeta,
    /// Reads in flight per gather: one thread per `rows_per_thread` records.
    threads: usize,
    /// Optional application-level cache of the last records (row -> record) for repeated tokens.
    cache: parking_lot::Mutex<HashMap<i64, [u8; 96]>>,
    cache_cap: usize,
    pub stats: parking_lot::Mutex<PleSsdStats>,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct PleSsdStats { pub gathers: u64, pub rows: u64, pub cache_hits: u64, pub read_ns: u64 }

impl PleSsd {
    pub fn open(meta: PleTableMeta) -> anyhow::Result<Self> {
        let file = std::fs::File::open(&meta.file)?;
        let threads = std::env::var("GB10_PLE_SSD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(32usize).max(1);
        let cache_cap = std::env::var("GB10_PLE_SSD_CACHE_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize << 18);
        Ok(Self { file, meta, threads, cache: parking_lot::Mutex::new(HashMap::new()), cache_cap,
                  stats: parking_lot::Mutex::new(PleSsdStats::default()) })
    }

    /// Read the records for `rows` into `out` (`rows.len() * 96` bytes), in order.
    pub fn gather(&self, rows: &[i64], out: &mut [u8]) -> anyhow::Result<()> {
        const RB: usize = crate::quant::PLE_REC_BYTES;
        assert_eq!(out.len(), rows.len() * RB);
        let t0 = std::time::Instant::now();
        let mut hits = 0u64;
        // 1. cache pass: fill what we have, collect the misses.
        let mut misses: Vec<(usize, i64)> = Vec::new();
        {
            let c = self.cache.lock();
            for (i, &r) in rows.iter().enumerate() {
                if let Some(rec) = c.get(&r) { out[i * RB..(i + 1) * RB].copy_from_slice(rec); hits += 1; }
                else { misses.push((i, r)); }
            }
        }
        // 2. pread the misses, fanned out over threads (each thread owns disjoint output rows).
        if !misses.is_empty() {
            let total = self.meta.total_rows as i64;
            for &(_, r) in &misses { anyhow::ensure!(r >= 0 && r < total, "PLE row {r} out of range [0,{total})"); }
            let nthr = self.threads.min(misses.len());
            let per = misses.len().div_ceil(nthr);
            let file = &self.file;
            let out_ptr = out.as_mut_ptr() as usize;
            let res: Vec<std::io::Result<()>> = std::thread::scope(|sc| {
                let hs: Vec<_> = misses.chunks(per).map(|chunk| {
                    sc.spawn(move || -> std::io::Result<()> {
                        for &(i, r) in chunk {
                            // SAFETY: rows are unique per index i → disjoint 96-B slices of `out`.
                            let dst = unsafe { std::slice::from_raw_parts_mut((out_ptr + i * RB) as *mut u8, RB) };
                            file.read_exact_at(dst, r as u64 * RB as u64)?;
                        }
                        Ok(())
                    })
                }).collect();
                hs.into_iter().map(|h| h.join().expect("ple ssd thread")).collect()
            });
            for r in res { r?; }
            let mut c = self.cache.lock();
            if c.len() + misses.len() > self.cache_cap { c.clear(); }
            for &(i, r) in &misses {
                let mut rec = [0u8; 96];
                rec.copy_from_slice(&out[i * RB..(i + 1) * RB]);
                c.insert(r, rec);
            }
        }
        let mut s = self.stats.lock();
        s.gathers += 1; s.rows += rows.len() as u64; s.cache_hits += hits;
        s.read_ns += t0.elapsed().as_nanos() as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn primes_and_multipliers_match_reference() {
        // Reference values computed with the HF modeling code (sympy.nextprime chain from 19_999_999).
        assert_eq!(nth_prime_after(19_999_999, 1), 20_000_003);
        assert!(is_prime(20_000_003));
        // Exactly the `layer_multipliers` / `ngram_heads_vocab_sizes` buffers stored in the
        // Qwen3.8-Flash-Next checkpoints.
        let m = layer_multipliers(248_320, 3, 0, 1234);
        assert_eq!(m, vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]);
        let mut p = 19_999_999u64; let mut primes = Vec::new();
        for _ in 0..16 { p = nth_prime_after(p, 1); primes.push(p); }
        assert_eq!(primes, vec![20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]);
        assert_eq!(primes.iter().sum::<u64>(), 320_001_446);
    }
}
