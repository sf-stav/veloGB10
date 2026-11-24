//! Golden tests: the Rust Engram hashing against DeepSeek's `inference/engram.py`, bit-exact.
//!
//! Fixture: `tests/fixtures/dsv41_engram/{layout.json, corpus.json, corpus_hash.u32}`, written by
//! `tools/dsv41_ref/engram_check.py` (the real `engram.py` + the model's tokenizer, one full pass per
//! sequence). The token-map tests also need the model's `tokenizer.json`: they look for
//! `models/deepseek-ai/DeepSeek-V4.1-Flash-1017` and the EXL3 pack dirs in every ancestor of the crate
//! root. Missing fixture or model: the test prints SKIP and returns (the public repo ships no tests/
//! fixtures); every test prints what it compared, so a skip can never pass for a check.

use super::hash::{EngramHasher, NgramHashState};
use super::layout::{np_rng, EngramConfig, EngramLayout};
use super::tokmap::CompressedTokenMap;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const MODEL_DIRS: [&str; 2] = [
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw",
];

struct Fixture {
    layout: Value,
    corpus: Value,
    hashes: Vec<u32>,
}

struct Seq {
    name: String,
    tokens: Vec<u32>,
    live: Vec<bool>,
    /// compressed id per position, -1 = DEAD (the reference's cache)
    comp: Vec<i64>,
    hash_off: usize,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv41_engram")
}

fn fixture() -> Option<&'static Fixture> {
    static FX: OnceLock<Option<Fixture>> = OnceLock::new();
    FX.get_or_init(|| {
        let d = fixture_dir();
        let layout = std::fs::read_to_string(d.join("layout.json")).ok()?;
        let corpus = std::fs::read_to_string(d.join("corpus.json")).ok()?;
        let raw = std::fs::read(d.join("corpus_hash.u32")).ok()?;
        let corpus: Value = serde_json::from_str(&corpus).expect("corpus.json");
        let sha = hex(&Sha256::digest(&raw));
        assert_eq!(sha, corpus["hash_file_sha256"].as_str().unwrap(), "corpus_hash.u32 corrupt");
        let hashes = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Some(Fixture { layout: serde_json::from_str(&layout).expect("layout.json"), corpus, hashes })
    })
    .as_ref()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn model_dirs() -> Vec<PathBuf> {
    let mut found = vec![];
    for name in MODEL_DIRS {
        let mut d = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
        while let Some(p) = d {
            let c = p.join("models").join(name);
            if c.join("tokenizer.json").is_file() && c.join("config.json").is_file() {
                found.push(c);
                break;
            }
            d = p.parent();
        }
    }
    found
}

fn seqs(fx: &Fixture) -> Vec<Seq> {
    fx.corpus["sequences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let n = s["n"].as_u64().unwrap() as usize;
            let tokens: Vec<u32> = s["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let comp: Vec<i64> = s["comp"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            let mut live = vec![true; n];
            for span in s["dead"].as_array().unwrap() {
                let (lo, hi) = (span[0].as_u64().unwrap() as usize, span[1].as_u64().unwrap() as usize);
                live[lo..hi].iter_mut().for_each(|x| *x = false);
            }
            assert_eq!((tokens.len(), comp.len()), (n, n));
            Seq { name: s["name"].as_str().unwrap().to_string(), tokens, live, comp, hash_off: s["hash_offset_u32"].as_u64().unwrap() as usize }
        })
        .collect()
}

fn fixture_config(fx: &Fixture) -> EngramConfig {
    EngramConfig::from_config_json(&fx.layout["config"]).unwrap().expect("fixture config has Engram fields")
}

/// Hasher on the real tokenizer (first model dir found), built once.
fn real_hasher() -> Option<&'static (EngramHasher, PathBuf)> {
    static H: OnceLock<Option<(EngramHasher, PathBuf)>> = OnceLock::new();
    H.get_or_init(|| {
        let d = model_dirs().into_iter().next()?;
        let h = EngramHasher::from_model_dir(&d).expect("hasher from model dir").expect("model has Engram");
        Some((h, d))
    })
    .as_ref()
}

/// Hasher whose token map is taken from the fixture's (raw, compressed) pairs: tests the hash math
/// independently of the tokenizer.
fn fixture_hasher(fx: &Fixture) -> EngramHasher {
    let n = fx.layout["tokenizer_len"].as_u64().unwrap() as usize;
    let mut map = vec![0u32; n];
    for s in seqs(fx) {
        for i in 0..s.tokens.len() {
            if s.live[i] {
                map[s.tokens[i] as usize] = s.comp[i] as u32;
            }
        }
    }
    let cfg = fixture_config(fx);
    map[cfg.pad_id as usize] = fx.layout["pad_compressed"].as_u64().unwrap() as u32;
    let tm = CompressedTokenMap::from_parts(map, fx.layout["compressed_vocab_size"].as_u64().unwrap() as u32).unwrap();
    EngramHasher::new(tm, EngramLayout::new(cfg).unwrap()).unwrap()
}

fn want<'a>(fx: &'a Fixture, s: &Seq, per: usize) -> &'a [u32] {
    &fx.hashes[s.hash_off..s.hash_off + s.tokens.len() * per]
}

/// First mismatch, for a readable failure.
fn first_diff(got: &[u32], want: &[u32], per: usize, n_cols: usize) -> Option<String> {
    let i = got.iter().zip(want).position(|(a, b)| a != b)?;
    let (t, r) = (i / per, i % per);
    Some(format!("token {t}, engram layer {}, column {}: got {} want {}", r / n_cols, r % n_cols, got[i], want[i]))
}

macro_rules! skip_unless {
    ($e:expr, $what:expr) => {
        match $e {
            Some(v) => v,
            None => {
                eprintln!("SKIP {}: {}", module_path!(), $what);
                return;
            }
        }
    };
}

#[test]
fn golden_layout_matches_engram_py() {
    let fx = skip_unless!(fixture(), "no tests/fixtures/dsv41_engram");
    let cfg = fixture_config(fx);
    let lay = EngramLayout::new(cfg.clone()).unwrap();
    let as_u64s = |v: &Value| -> Vec<u64> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect() };
    let primes: Vec<Vec<Vec<u64>>> = fx.layout["primes"].as_array().unwrap().iter().map(|l| l.as_array().unwrap().iter().map(as_u64s).collect()).collect();
    let offsets: Vec<Vec<u64>> = fx.layout["offsets"].as_array().unwrap().iter().map(as_u64s).collect();
    let mults: Vec<Vec<u64>> = fx.layout["multipliers"].as_array().unwrap().iter().map(as_u64s).collect();
    assert_eq!(lay.primes, primes, "primes");
    assert_eq!(lay.offsets, offsets, "offsets");
    assert_eq!(lay.multipliers, mults, "multipliers");
    assert_eq!(as_u64s(&fx.layout["row_sum"]), cfg.num_embeddings);
    assert_eq!(fx.layout["compressed_vocab_size"].as_u64().unwrap(), 99_092);
    // the numpy RNG port, on seeds and ranges beyond the ones the model uses
    let checks = fx.layout["rng_checks"].as_array().unwrap();
    for c in checks {
        let seed: u128 = c["seed"].as_str().unwrap().parse().unwrap();
        let high = c["high"].as_u64().unwrap() as i64;
        let want: Vec<i64> = c["values"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let mut g = np_rng::Pcg64::from_seed_sequence(&np_rng::SeedSequence::from_u128(seed));
        assert_eq!(g.integers_i64(0, high, want.len()).unwrap(), want, "rng seed {seed} high {high}");
    }
    // every model dir found parses to the same Engram config (pack-generic)
    for d in model_dirs() {
        assert_eq!(EngramConfig::from_model_dir(&d).unwrap().unwrap(), cfg, "{}", d.display());
    }
    eprintln!(
        "engram layout: {} layers x {} columns, primes/offsets/multipliers equal; {} numpy-RNG draws equal",
        lay.n_layers(),
        lay.n_cols(),
        checks.len()
    );
}

#[test]
fn golden_tokmap_matches_engram_py() {
    let fx = skip_unless!(fixture(), "no tests/fixtures/dsv41_engram");
    let dirs = model_dirs();
    if dirs.is_empty() {
        eprintln!("SKIP {}: no DeepSeek-V4.1 tokenizer.json under any ancestor models/ dir", module_path!());
        return;
    }
    let want_sha = fx.layout["tokmap_sha256_u32le"].as_str().unwrap();
    let all = seqs(fx);
    for d in &dirs {
        let t0 = std::time::Instant::now();
        let m = CompressedTokenMap::from_model_dir(d).unwrap();
        let dt = t0.elapsed();
        assert_eq!(m.len() as u64, fx.layout["tokenizer_len"].as_u64().unwrap(), "{}", d.display());
        assert_eq!(m.vocab(), 99_092, "{}", d.display());
        assert_eq!(m.sha256_hex(), want_sha, "token map differs from engram.py ({})", d.display());
        for (k, v) in fx.layout["tokmap_sample"].as_object().unwrap() {
            assert_eq!(m.get(k.parse().unwrap()), Some(v.as_u64().unwrap() as u32), "id {k}");
        }
        let mut n = 0;
        for s in &all {
            for i in 0..s.tokens.len() {
                if s.live[i] {
                    assert_eq!(m.get(s.tokens[i]).unwrap() as i64, s.comp[i], "{} pos {i}", s.name);
                    n += 1;
                }
            }
        }
        eprintln!("engram token map {}: {} ids -> {} (sha256 equal), {n} corpus tokens equal, built in {dt:?}", d.display(), m.len(), m.vocab());
    }
}

fn check_full_pass(fx: &Fixture, h: &EngramHasher, label: &str) {
    let per = h.ids_per_token();
    let n_cols = h.layout().n_cols();
    assert_eq!((per, fx.corpus["n_layers"].as_u64().unwrap() as usize * fx.corpus["n_cols"].as_u64().unwrap() as usize), (48, 48));
    let mut tokens = 0;
    for s in seqs(fx) {
        let mut st = NgramHashState::new();
        let mut out = vec![0u32; s.tokens.len() * per];
        h.hash_chunk(&mut st, &s.tokens, Some(&s.live), &mut out).unwrap();
        if let Some(d) = first_diff(&out, want(fx, &s, per), per, n_cols) {
            panic!("{label}: {} differs from engram.py at {d}", s.name);
        }
        tokens += s.tokens.len();
    }
    assert_eq!(tokens as u64, fx.corpus["total_tokens"].as_u64().unwrap());
    assert!(tokens >= 10_000, "corpus too small: {tokens}");
    eprintln!("engram full pass ({label}): {tokens} tokens x {per} ids bit-exact against engram.py");
}

#[test]
fn golden_full_pass_from_compressed_ids() {
    let fx = skip_unless!(fixture(), "no tests/fixtures/dsv41_engram");
    check_full_pass(fx, &fixture_hasher(fx), "fixture token map");
}

#[test]
fn golden_full_pass_from_tokenizer() {
    let fx = skip_unless!(fixture(), "no tests/fixtures/dsv41_engram");
    let (h, d) = skip_unless!(real_hasher(), "no DeepSeek-V4.1 tokenizer.json found");
    check_full_pass(fx, h, &format!("tokenizer {}", d.display()));
}

/// Chunked prefill at odd sizes, then one-token decode steps with a speculative 6-token verify block
/// peeked before each step (only the accepted prefix is committed), with resumes from a serialized
/// state and from token history along the way. Every id must equal the single full pass.
#[test]
fn golden_chunked_decode_resume_equal_full_pass() {
    let fx = skip_unless!(fixture(), "no tests/fixtures/dsv41_engram");
    let owned;
    let (h, label) = match real_hasher() {
        Some((h, d)) => (h, format!("tokenizer {}", d.display())),
        None => {
            owned = fixture_hasher(fx);
            (&owned, "fixture token map".to_string())
        }
    };
    let per = h.ids_per_token();
    let n_cols = h.layout().n_cols();
    const CHUNKS: [usize; 11] = [1, 3, 5, 7, 11, 13, 17, 31, 61, 127, 509];
    let (mut tokens, mut resumes, mut spec_rounds) = (0usize, 0usize, 0usize);
    for s in seqs(fx) {
        let n = s.tokens.len();
        let want = want(fx, &s, per);
        let decode_from = n.saturating_sub(40); // the last 40 positions are decode steps
        let resume_at: Vec<usize> = vec![n / 3, n / 2, (2 * n) / 3, decode_from + 7];
        let mut got = vec![0u32; n * per];
        let mut st = NgramHashState::new();
        let (mut p, mut k) = (0usize, 0usize);
        while p < n {
            if resume_at.contains(&p) && p > 0 {
                // lane restarted from a prefix-cache entry: alternate the two resume forms
                st = if resumes % 2 == 0 {
                    NgramHashState::from_bytes(&st.to_bytes()).unwrap()
                } else {
                    let lo = p.saturating_sub(3 + resumes % 5);
                    h.state_at(p as u64, &s.tokens[lo..p], Some(&s.live[lo..p])).unwrap()
                };
                resumes += 1;
            }
            let step = if p >= decode_from { 1 } else { CHUNKS[k % CHUNKS.len()].min(decode_from - p) };
            if p >= decode_from {
                // speculative verify block: the true next tokens, the tail replaced by a wrong draft
                let blk = (n - p).min(6);
                let accept = k % (blk + 1);
                let mut draft: Vec<u32> = s.tokens[p..p + blk].to_vec();
                for (j, t) in draft.iter_mut().enumerate().skip(accept) {
                    *t = (*t + 7 + j as u32) % 1000;
                }
                let dlive: Vec<bool> = s.live[p..p + blk].to_vec();
                let mut spec = vec![0u32; blk * per];
                let after = h.peek(&st, &draft, Some(&dlive), &mut spec).unwrap();
                assert_eq!(after.pos(), (p + blk) as u64);
                // the accepted prefix of the block must already carry the final ids
                assert_eq!(&spec[..accept * per], &want[p * per..(p + accept) * per], "{} spec at {p}", s.name);
                spec_rounds += 1;
            }
            h.hash_chunk(&mut st, &s.tokens[p..p + step], Some(&s.live[p..p + step]), &mut got[p * per..(p + step) * per]).unwrap();
            p += step;
            k += 1;
        }
        assert_eq!(st.pos(), n as u64);
        if let Some(d) = first_diff(&got, want, per, n_cols) {
            panic!("{label}: chunked/decode/resume {} differs from engram.py at {d}", s.name);
        }
        // commit-only replay (advance) reaches the same state as hashing everything
        let mut st2 = NgramHashState::new();
        h.advance(&mut st2, &s.tokens, Some(&s.live)).unwrap();
        assert_eq!(st2, st, "{} advance", s.name);
        tokens += n;
    }
    eprintln!(
        "engram chunked prefill + decode + resume ({label}): {tokens} tokens bit-exact against engram.py \
         ({resumes} resumes, {spec_rounds} speculative blocks)"
    );
}

/// The shard map agrees with the packer contract: column c = g * 8 + h; a head's rows are its 3
/// columns in both layers; every corpus id lands inside the head file of its column's head.
#[test]
fn golden_ids_land_in_their_head_rows() {
    let fx = skip_unless!(fixture(), "no tests/fixtures/dsv41_engram");
    let h = fixture_hasher(fx);
    let lay = h.layout();
    let heads: Vec<_> = (0..lay.n_heads()).map(|x| lay.rows_for_head(x)).collect();
    let per = h.ids_per_token();
    let mut n = 0u64;
    for s in seqs(fx) {
        for (t, ids) in want(fx, &s, per).chunks_exact(per).enumerate() {
            for l in 0..lay.n_layers() {
                for c in 0..lay.n_cols() {
                    let id = ids[l * lay.n_cols() + c];
                    let (g, head, row) = h.split_id(l, c, id);
                    assert_eq!(lay.col(g, head), c);
                    assert!(row < lay.primes[l][g][head], "{} token {t}", s.name);
                    let rec = heads[head].record_of(l, g, id as u64, lay.n_ngram_cols()).expect("id in its head's rows");
                    assert!(rec < heads[head].total_records);
                    for (o, other) in heads.iter().enumerate() {
                        if o != head {
                            assert!(other.record_of(l, g, id as u64, lay.n_ngram_cols()).is_none());
                        }
                    }
                    n += 1;
                }
            }
        }
    }
    eprintln!("engram shard map: {n} corpus ids each in exactly one head's rows");
}

/// Host cost of hashing (no assertion on speed; the number goes in the report): the corpus repeated
/// to ~1M positions as one prefill, and as 1-token decode steps.
#[test]
fn golden_hash_throughput() {
    let fx = skip_unless!(fixture(), "no tests/fixtures/dsv41_engram");
    let h = fixture_hasher(fx);
    let per = h.ids_per_token();
    let base: Vec<u32> = seqs(fx).into_iter().flat_map(|s| s.tokens).collect();
    let toks: Vec<u32> = base.iter().copied().cycle().take(1 << 20).collect();
    let mut out = vec![0u32; toks.len() * per];
    let mut st = NgramHashState::new();
    let t0 = std::time::Instant::now();
    h.hash_chunk(&mut st, &toks, None, &mut out).unwrap();
    let prefill = t0.elapsed();
    let mut st = NgramHashState::new();
    let t0 = std::time::Instant::now();
    for (i, t) in toks[..65536].iter().enumerate() {
        h.hash_chunk(&mut st, std::slice::from_ref(t), None, &mut out[i * per..(i + 1) * per]).unwrap();
    }
    let decode = t0.elapsed();
    let x: u64 = out.iter().map(|&v| v as u64).sum();
    eprintln!(
        "engram hash throughput (1 thread): prefill {} positions in {prefill:?} ({:.1} ns/pos); decode step {:.1} ns/pos [checksum {x}]",
        toks.len(),
        prefill.as_nanos() as f64 / toks.len() as f64,
        decode.as_nanos() as f64 / 65536.0
    );
}
